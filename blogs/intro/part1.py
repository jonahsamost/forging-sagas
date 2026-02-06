# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUTLASS_CUDA_ARCH'] = '86'

import argparse
import time
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

custom_cache = {}

@cute.kernel
def _reduce_sum_kernel(gA: cute.Tensor, output: cute.Tensor):
    smem_alloc = cutlass.utils.SmemAllocator()
    shmem = smem_alloc.allocate_tensor(cute.Float32, cute.make_layout((32,)))  # warp size

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    M, N = gA.shape
    ntiles = cute.ceil_div(N, bdimx)
    acc = 0.0
    for tile_idx in range(ntiles):
        idx = tile_idx * bdimx + tidx
        if idx < N:
            acc += gA[bidx, idx]

    acc = cute.arch.warp_reduction_sum(acc)
    if lane == 0:
        shmem[warp] = acc
    cute.arch.sync_threads()

    if warp == 0:
        block_warps = bdimx // 32
        acc2 = shmem[lane] if lane < block_warps else 0.0
        acc2 = cute.arch.warp_reduction_sum(acc2)
        if lane == 0:
            output[bidx] = acc2


@cute.jit
def reduce_sum(x: cute.Tensor, output: cute.Tensor):
    num_warps = 4
    threads_per_block = num_warps * 32
    M, N = x.shape

    _reduce_sum_kernel(x, output).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )

@cute.jit
def reduce_sum_dynamic(x, output):
    num_warps = 4
    threads_per_block = num_warps * 32
    M, N = x.shape

    _reduce_sum_kernel(x, output).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )

def _invoke_reduce_sum(gInput, dim=-1):
    a_ = from_dlpack(gInput)

    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)
    b_ = from_dlpack(gOutput)

    kernel_fn = cute.compile(reduce_sum, a_, b_,)
    kernel_fn(a_, b_)
    return gOutput


def _invoke_reduce_sum_with_cache(gInput, dim=-1):
    a_ = from_dlpack(gInput)

    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)
    b_ = from_dlpack(gOutput)

    key = f'reduce_sum_{gInput.dtype}'
    if key not in custom_cache:
        custom_cache[key] = cute.compile(reduce_sum, a_, b_)
    fn = custom_cache[key]
    fn(gInput, gOutput)
    return gOutput


def _invoke_reduce_sum_with_cache_dynamic(gInput, dim=-1):
    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)

    key = f'reduce_sum_{gInput.dtype}'
    if key not in custom_cache:
        custom_cache[key] = cute.compile(reduce_sum_dynamic, gInput, gOutput)
    fn = custom_cache[key]
    fn(gInput, gOutput)
    return gOutput


def simple_launch(dim=-1):
    M, N = 1234, 4321

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    output = _invoke_reduce_sum(a, dim=-1)

    torch_sum = a.sum(dim=dim)
    assert torch.allclose(torch_sum, output, rtol=1e-4, atol=1e-4)
    print("Success!")


def bug():
    M, N = 4096, 4096
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    initial = _invoke_reduce_sum_with_cache(a, dim=-1)
    torch.allclose(initial, a.sum(dim=-1), rtol=1e-4, atol=1e-4)

    M, N = 100, 300
    b = torch.randn(M, N, device="cuda", dtype=torch.float32)
    after = _invoke_reduce_sum_with_cache(b, dim=-1)
    torch.allclose(after, b.sum(dim=-1), rtol=1e-4, atol=1e-4)


def compare_torch_initial_cached():
    M, N = 4096, 4096

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # warm-up
    for _ in range(10):
        _invoke_reduce_sum_with_cache(a, dim=-1)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _invoke_reduce_sum_with_cache(a, dim=-1)
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


def main():
    parser = argparse.ArgumentParser(description="CuTe DSL Part 1 Examples")
    parser.add_argument(
        "command",
        choices=[
            "simple_launch",
            "compare_torch_initial",
            "compare_torch_initial_cached",
        ],
        help="Which example to run",
    )
    args = parser.parse_args()

    if args.command == "simple_launch":
        simple_launch()
    elif args.command == "compare_torch_initial":
        compare_torch_initial()
    elif args.command == "compare_torch_initial_cached":
        compare_torch_initial_cached()


# if __name__ == "__main__":
#     main()