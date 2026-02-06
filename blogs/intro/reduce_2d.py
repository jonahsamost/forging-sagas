import torch
import inspect
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr
from cute_viz import render_layout_svg, display_layout


class ReduceSumTiledCopy:
    def __init__(self, shape, dtype, dim=-1):
        self.shape = shape
        self.dtype = dtype
        self.num_warps = 4
        self.warp_size = 32
        self.threads_per_block = self.num_warps * self.warp_size
        self.NEG_INF = Float32(float('-inf'))
        self.dim = dim
        self.reduce_size = self.shape[self.dim]
        self.blocks = self.shape[0] if dim != 0 else self.shape[-1]

        self.order_shape = (self.num_warps, self.warp_size) if self.dim == -1 else (self.warp_size, self.num_warps)
        self.order = (1, 0) if self.dim == -1 else (0, 1)

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):
        blocks_over_reduce_dim = cute.ceil_div(self.reduce_size, self.warp_size)
        
        # construct a "super tile" that spans the entire portion of the reduction
        # dimension handled by this thread block
        tiler_mn = (
            (self.num_warps, blocks_over_reduce_dim * self.warp_size)
            if self.dim == -1 else
            (blocks_over_reduce_dim * self.warp_size, self.num_warps)
        )

        # copy atom answers how should a single thread load its data
        # need to ensure our copy atom and val layout are compatible
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=self.dtype.width)
        val_layout = cute.make_layout((1, 1))
        
        # define how threads are organized within a tile
        thr_layout = cute.make_ordered_layout(self.order_shape, self.order)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        blocks = cute.ceil_div(self.blocks, self.num_warps)
        self.kernel(
            gInput, gOutput, tiler_mn, tiled_copy
        ).launch(
            grid=(blocks, 1, 1),
            block=(self.threads_per_block, 1, 1),
            stream=stream
        )
    
    @cute.kernel
    def kernel(self, gInput: cute.Tensor, gOutput: cute.Tensor, tiler_mn: cute.Shape, tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // self.warp_size
        lane_idx = tidx % self.warp_size

        # gX is now a "super" tile over all subtiles that our thread group will work on
        gX = cute.local_tile(gInput, tiler_mn, (bidx, 0) if self.dim == -1 else (0, bidx))
        # where will this thread load its data from relative to our thread layout
        tidxSlice = tiled_copy.get_slice(tidx)  
        # get me all the addresses that our thread is responsible for across our "super" tile
        tidxIndices = tidxSlice.partition_S(gX)
        # create registers, load our data, reduce, and write
        tidxRegs = cute.make_rmem_tensor_like(tidxIndices)
        cute.autovec_copy(tidxIndices, tidxRegs)
        tidxValues = tidxRegs.load()
        tidLocalSum = tidxValues.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        rowSum = cute.arch.warp_reduction_sum(tidLocalSum)
        if lane_idx == 0:
            gOutput[bidx * self.num_warps + warp_idx] = rowSum


class ReduceSumCompositional:
    def __init__(self, shape, dtype, dim=-1):
        self.shape = shape
        self.dtype = dtype
        self.num_warps = 4
        self.warp_size = 32
        self.threads_per_block = self.num_warps * self.warp_size
        self.NEG_INF = Float32(float('-inf'))
        self.dim = dim
        self.reduce_size = self.shape[self.dim]
        self.blocks = self.shape[0] if dim != 0 else self.shape[-1]

        self.order_shape = (self.num_warps, self.warp_size) if self.dim == -1 else (self.warp_size, self.num_warps)
        self.order = (1, 0) if self.dim == -1 else (0, 1)

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):
        val_layout = cute.make_layout((1, 1))  # each thread loads a single value
        # threads either row or column major order depending on dimension were reducing over
        thr_layout = cute.make_ordered_layout(self.order_shape, self.order)
        tiled_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)

        # decompose our tensor into tiles defined by tiled_mn
        # gX now has the shape: ((TileM, TileN), (RestM, RestN))
        #                       ((num_warps, warp_size), (M // num_warps, N // warp_size))
        gX = cute.zipped_divide(gInput, tiled_mn)

        blocks = cute.ceil_div(self.blocks, self.num_warps)
        self.kernel(
            gX, gOutput, layout_tv
        ).launch(
            grid=(blocks, 1, 1),
            block=(self.threads_per_block, 1, 1),
            stream=stream
        )
    
    @cute.kernel
    def kernel(self, gX: cute.Tensor, gOutput: cute.Tensor, layout_tv: cute.Layout):
        tidx, _, _ = cute.arch.thread_idx()  # tidx, tidy, tidz
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // self.warp_size
        lane_idx = tidx % self.warp_size

        acc = Float32(0.0)
        ntiles = cute.ceil_div(self.reduce_size, self.warp_size)
        for tile_idx in range(ntiles):
            # rows of blocks when summing across, else columns of blocks
            blk_coord = (bidx, tile_idx) if self.dim == -1 else (tile_idx, bidx)
            # gX's shape is: ((TileM, TileN), (RestM, RestN))
            # we want to load a complete subtile (i.e. all TileM x TileN values)
            # the block coordinate is a function of the dimension and if the subtiles go across or down
            subtile = gX[((None, None), blk_coord)]
            # map thread layout to data
            thr_block_frag = cute.composition(subtile, layout_tv)
            # grab this threads fragment and load
            thr_data = thr_block_frag[(tidx, None)]
            thr_data.load()
            acc += thr_data[0]

        acc = cute.arch.warp_reduction_sum(acc)
        if lane_idx == 0:
            gOutput[bidx * self.num_warps + warp_idx] = acc


def benchmark(dim=0, is_compositional=True):
    print(f"Testing dim: {dim}")
    import time

    M, N = 4096, 4096
    dtype = torch.float32
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dtype]
    
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    out_shape = M if dim == -1 else N
    output = torch.empty((out_shape,), device=x.device, dtype=x.dtype)

    #### 

    input_shape = [cute.sym_int() for _ in range(len(x.shape))]
    input_shape[dim] = x.shape[dim]
    input_cute = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        tuple(input_shape),
        stride_order=tuple(range(len(x.shape)))[::-1]
    )

    output_shape = [cute.sym_int() for _ in range(len(x.shape))]
    output_shape.pop(dim)
    output_cute = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        tuple(output_shape),
        stride_order=tuple(range(len(x.shape) - 1))[::-1]
    ) 
    cls = ReduceSumCompositional if is_compositional else ReduceSumTiledCopy
    softmax = cls(x.shape, dtype_map[dtype], dim=dim)
    fn = cute.compile(
        softmax, input_cute, output_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi", 
    )
    fn(x, output)
    
    print("Correctness check:")
    expected = x.sum(dim=dim)
    is_close = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print(f"  dim={dim}: {'✓ PASS' if is_close else '✗ FAIL'}")
    if not is_close:
        max_diff = (output - expected).abs().max().item()
        print(f"         max diff: {max_diff}")
    
    print("\nBenchmarks:")
    
    # Warmup
    for _ in range(10):
        fn(x, output)
    torch.cuda.synchronize()

    # Benchmark CuTe kernel
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        fn(x, output)
    end.record()

    torch.cuda.synchronize()
    cute_ms = start.elapsed_time(end) / 100
    print(f"reduce sum cute dim={dim}: {cute_ms:.3f} ms")

    # Compare to PyTorch
    start.record()
    for _ in range(100):
        _ = x.sum(dim=dim)
    end.record()

    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 100
    print(f"torch reduce sum dim={dim}: {torch_ms:.3f} ms")


benchmark(dim=0, is_compositional=False)
benchmark(dim=-1, is_compositional=False)