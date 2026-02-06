import torch
import inspect
import math
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr
from cute_viz import render_layout_svg, display_layout


class ReduceSum:
    def __init__(self, shape, dtype, dim=-1):
        self.shape = shape
        self.dtype = dtype
        self.bits_read = self.dtype.width
        self.num_warps = 4
        self.warp_size = 32
        self.threads_per_block = self.num_warps * self.warp_size
        self.NEG_INF = Float32(float('-inf'))
        self.dim = dim

        rank = len(self.shape)
        strides = [1]
        mult = 1
        for d in self.shape[1:][::-1]:
            mult *= d
            strides.insert(0, mult)
        self.strides = strides

        self.reduce_size = shape[dim]

        self.before_shape = shape[:dim]
        self.after_shape = shape[dim + 1:]

        self.before_prod = math.prod(self.before_shape) if dim > 0 else 1
        self.after_prod = math.prod(self.after_shape) if dim < rank - 1 else 1

        self.before_stride = strides[dim - 1] if dim > 0 else 0
        self.after_stride = strides[-1] if dim < rank - 1 else 0
        self.reduce_stride = strides[dim]
        
        self.output_shape = (*self.before_shape, *self.after_shape)
        self.blocks = math.prod(self.output_shape)

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):

        reduce_cover_blocks = cute.ceil_div(self.reduce_size, self.warp_size)
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=self.bits_read)

        in_layout = cute.make_layout(
            (self.before_prod, self.reduce_size, self.after_prod),
            stride=(self.before_stride, self.reduce_stride, self.after_stride)
        ) 
        gInputView = cute.make_tensor(gInput.iterator, in_layout)
        out_layout = cute.make_layout(
            (self.before_prod, self.after_prod),
            stride=(self.after_prod, 1)
        )
        gOutputView = cute.make_tensor(gOutput.iterator, out_layout)

        thr_layout = cute.make_ordered_layout(
            (self.num_warps, self.warp_size),
            order=(1, 0)
        )
        val_layout = cute.make_layout((1, 1))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        tiler_nd = (
            1,
            reduce_cover_blocks * self.warp_size,
            1
        )
        blocks = cute.ceil_div(self.blocks, self.num_warps)
        self.kernel(
            gInputView, gOutputView, tiler_nd, tiled_copy
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

        out_idx = bidx * self.num_warps + warp_idx
        before_idx = out_idx // self.after_prod
        after_idx = out_idx % self.after_prod

        gX = cute.local_tile(gInput, tiler_mn, (before_idx, None, after_idx))
        tidxSlice = tiled_copy.get_slice(tidx)  
        tidxIndices = tidxSlice.partition_S(gX)
        tidxRegs = cute.make_rmem_tensor_like(tidxIndices)
        cute.autovec_copy(tidxIndices, tidxRegs)
        tidxValues = tidxRegs.load()
        tidLocalSum = tidxValues.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        rowSum = cute.arch.warp_reduction_sum(tidLocalSum)
        if lane_idx == 0:
            gOutput[before_idx, after_idx] = rowSum


def benchmark(shape, dim=1, do_bencharmk=False):
    print(f"Testing dim: {dim}, {shape}")
    import time

    dtype = torch.float32
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dtype]
    
    x = torch.randn(shape, device='cuda', dtype=dtype)
    output_shape = list(x.shape)
    output_shape.pop(dim)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

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
    softmax = ReduceSum(x.shape, dtype_map[dtype], dim=dim)
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
    
    if do_bencharmk:
        print("\nBenchmarks:")
        
        # Warmup
        for _ in range(10):
            fn(x, output)
        torch.cuda.synchronize()
        
        # Benchmark our softmax
        start = time.perf_counter()
        for _ in range(100):
            fn(x, output)
        torch.cuda.synchronize()
        print(f"  reduce sum cute dim=-1: {(time.perf_counter() - start) * 10:.3f} ms")
        
        # Compare to PyTorch
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.nn.functional.softmax(x, dim=-1)
        torch.cuda.synchronize()
        print(f"  torch reduce sum dim=-1:  {(time.perf_counter() - start) * 10:.3f} ms")


shape = []
for dim in [64, 32, 128, 256]:
    shape.append(dim)
    for i in range(len(shape)):
        benchmark(shape, dim=i)