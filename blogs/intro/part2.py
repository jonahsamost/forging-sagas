import argparse
import time
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cute_viz import render_tv_layout_svg, display_tv_layout


custom_cache = {}


@cute.kernel
def _reduce_sum(input: cute.Tensor, output: cute.Tensor, tv_layout, num_warps: int):
    smem_alloc = cutlass.utils.SmemAllocator()
    shmem = smem_alloc.allocate_tensor(cute.Float32, cute.make_layout((32,)))

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    acc = cute.Float32(0.0)

    _, mode1 = input.shape
    ntiles = mode1[1]

    for tile_idx in range(ntiles):
        blk_coord = ((0, None), (bidx, tile_idx)) # all values in this [bidx, tile_idx] tile 
        blk = input[blk_coord]
        thr_frag = cute.composition(blk, tv_layout)
        thr_val  = thr_frag[tidx, 0]
        acc += thr_val

    acc = cute.arch.warp_reduction_sum(acc)
    if lane == 0:
        shmem[warp] = acc
    cute.arch.sync_threads()

    if warp == 0:
        acc2 = shmem[lane] if lane < num_warps else 0.0
        acc2 = cute.arch.warp_reduction_sum(acc2)
        if lane == 0:
            output[bidx] = acc2

@cute.jit
def reduce_sum(x, output):
    num_warps = 16
    threads_per_block = 512
    M, N = x.shape

    thr_layout = cute.make_layout((threads_per_block,), stride=(1,))
    val_layout = cute.make_layout((1,), stride=(1,))
    tiled_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gX = cute.zipped_divide(x, (1, tiled_mn[0]))
    
    _reduce_sum(gX, output, tv_layout, num_warps).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def _invoke_reduce_sum(gInput, dim=-1):
    # a_ = from_dlpack(gInput)

    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)
    # b_ = from_dlpack(gOutput)

    key = (gInput.dtype, gInput.shape)
    if key not in custom_cache:
        kernel_fn = cute.compile(
            reduce_sum,
            from_dlpack(gInput),
            from_dlpack(gOutput)
        )
        custom_cache[key] = kernel_fn
    custom_cache[key](gInput, gOutput)
    return gOutput


def simple_launch(dim=-1):
    M, N = 512, 1024

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    output = _invoke_reduce_sum(a, dim=-1)

    torch_sum = a.sum(dim=dim)
    assert torch.allclose(torch_sum, output, rtol=1e-4, atol=1e-4)
    print("Success!")



def compare_torch_initial():
    M, N = 4096, 4096

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # warm-up
    for _ in range(10):
        _invoke_reduce_sum(a, dim=-1)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _invoke_reduce_sum(a, dim=-1)
    end_event.record()
    torch.cuda.synchronize()
    print(f"  cute dsl reduce sum: {start_event.elapsed_time(end_event) / 100:.3f} ms")

    # warm-up torch
    for _ in range(10):
        _ = torch.sum(a, dim=-1)
    torch.cuda.synchronize()
    
    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)
    
    start_event2.record()
    for _ in range(100):
        _ = torch.sum(a, dim=-1)
    end_event2.record()
    torch.cuda.synchronize()
    print(f"  torch kernel sum: {start_event2.elapsed_time(end_event2) / 100:.3f} ms")


@cute.jit
def print_layout_trash(x):
    M, N = 32, 64
    thr_layout = cute.make_layout(((M//4, N//32), (4, 32)) , stride=((4*N, 32), (N, 1)))
    val_layout = cute.make_layout((1,),  stride=(1,))
    tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
    render_tv_layout_svg(layout_tv, tiler_mn, "tv_layout.svg")
    gX = cute.zipped_divide(x, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    print(f"gX: {gX}")


def compile_print():
    x = torch.randn(256, 512)
    gx = from_dlpack(x)
    fn = cute.compile(
        print_layout_trash,
        gx
    )
    fn(gx)
compile_print()