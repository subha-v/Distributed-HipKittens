#!POPCORN leaderboard amd-gemm-rs
#!POPCORN gpu MI300x8

import faulthandler
import math
import os
import sys

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t
from torch import Tensor
from torch.utils.cpp_extension import load_inline
from triton.compiler.compiler import LazyDict

# this will print segfault to stderr
faulthandler.enable(file=sys.stderr, all_threads=True)

cuda_src = r"""
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <cooperative_groups.h>
#include <stdio.h>

#define STRINGIFY(x) #x
#define CUDA_CHECK(call)                                                             \
  do {                                                                               \
    cudaError_t err = call;                                                          \
    TORCH_CHECK(err == cudaSuccess, STRINGIFY(call), ": ", cudaGetErrorString(err)); \
  } while (0)

at::Tensor malloc_with_flags(int64_t size, int64_t flag) {
  void *ptr;
  CUDA_CHECK(hipExtMallocWithFlags(&ptr, size, flag));

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  auto options = at::TensorOptions().dtype(at::kChar).device(at::kCUDA, device);
  return torch::from_blob(ptr, {size}, [](void *ptr){ CUDA_CHECK(hipFree(ptr)); }, options);
}

// input is CUDA, but output is CPU
at::Tensor get_ipc_handle(const at::Tensor& x) {
  // IPC handle as a tensor
  auto options = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
  at::Tensor h = at::empty({sizeof(cudaIpcMemHandle_t)}, options);
  auto h_ptr = reinterpret_cast<cudaIpcMemHandle_t *>(h.data_ptr());
  CUDA_CHECK(cudaIpcGetMemHandle(h_ptr, x.data_ptr()));
  return h;
}

int64_t open_ipc_handle(const at::Tensor& h) {
  void *ptr;
  auto h_ptr = reinterpret_cast<cudaIpcMemHandle_t *>(h.data_ptr());
  CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h_ptr[0], cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}

void close_ipc_handle(int64_t addr) {
  void *ptr = reinterpret_cast<void *>(addr);
  CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
}

TORCH_LIBRARY(p2p_module, m) {
  m.def("malloc_with_flags(int size, int flag) -> Tensor");
  m.impl("malloc_with_flags", &malloc_with_flags);

  m.def("get_ipc_handle(Tensor x) -> Tensor");
  m.impl("get_ipc_handle", &get_ipc_handle);

  m.def("open_ipc_handle(Tensor handle) -> int");
  m.impl("open_ipc_handle", &open_ipc_handle);

  m.def("close_ipc_handle(int addr) -> ()");
  m.impl("close_ipc_handle", &close_ipc_handle);
}
"""

load_inline(
    "p2p_module",
    cpp_sources=[""],
    cuda_sources=[cuda_src],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
)
ops = torch.ops.p2p_module


class P2PState:
    def __init__(self, rank: int, world_size: int, size: int = 1 << 30) -> None:
        torch.cuda.set_device(rank)

        finegrained = 0x1
        uncached = 0x3
        heap = ops.malloc_with_flags(size, finegrained)
        assert heap.device.index == rank

        handle = ops.get_ipc_handle(heap).cuda()
        all_handles = torch.empty(world_size, 64, dtype=torch.int8, device="cuda")
        dist.all_gather_into_tensor(all_handles.view(-1), handle)
        assert (all_handles[rank] == handle).all()

        all_handles = all_handles.cpu()
        heap_bases = [heap.data_ptr() if i == rank else ops.open_ipc_handle(all_handles[i]) for i in range(world_size)]
        heap_bases = torch.tensor(heap_bases, dtype=torch.int64, device="cuda")

        self.rank = rank
        self.world_size = world_size
        self.heap = heap
        self.heap_bases = heap_bases
        self.size = size

        self.ptr = 0

        # largest problem size
        m = 8192
        n = 8192
        self.C = self.malloc_symmetric((m * n,), dtype=torch.bfloat16)
        self.flag = self.malloc_symmetric(((m // 8) * (n // 32),), dtype=torch.int32).zero_()

        # make sure everyone has finished setting up symmetric memory
        torch.cuda.synchronize()
        dist.barrier()

    def close(self):
        # print(f"{self.rank=}: Close IPC handles", file=sys.stderr, flush=True)

        torch.cuda.set_device(self.rank)
        for i, base in enumerate(self.heap_bases.tolist()):
            if i != self.rank:
                ops.close_ipc_handle(base)

    def malloc_symmetric(self, shape: tuple[int, ...], dtype: torch.dtype, alignment: int = 128) -> Tensor:
        start = triton.cdiv(self.ptr, alignment) * alignment
        end = start + math.prod(shape) * dtype.itemsize
        assert end <= self.size
        out = self.heap[start:end].view(dtype).view(shape)
        self.ptr = end
        return out

    @staticmethod
    def malloc_finegrained(shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        size = math.prod(shape) * dtype.itemsize
        finegrained = 0x1
        uncached = 0x3
        return ops.malloc_with_flags(size, finegrained).view(dtype).view(shape)


P2P_STATE: P2PState | None = None

original_init = dist.init_process_group
original_destroy = dist.destroy_process_group


def patched_init(*args, rank, world_size, **kwargs):
    original_init(*args, rank=rank, world_size=world_size, **kwargs)

    global P2P_STATE
    assert P2P_STATE is None
    P2P_STATE = P2PState(rank, world_size)


def patched_destroy():
    global P2P_STATE
    dist.barrier()
    P2P_STATE.close()
    P2P_STATE = None
    original_destroy()


dist.init_process_group = patched_init
dist.destroy_process_group = patched_destroy


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid


@triton.jit
def compute_pid(pid, grid_m, grid_n, GROUP_M: tl.constexpr, REMAP_XCD: tl.constexpr = True):
    if REMAP_XCD:
        # most of the time, this if beneficial
        # 4096, 4096, 512
        pid = remap_xcd(pid, grid_m * grid_n)

    if GROUP_M == 1:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)

    return pid_m, pid_n


@triton.jit
def triton_sleep(cycles: tl.constexpr):
    if cycles > 0:
        return tl.inline_asm_elementwise(
            asm="s_sleep $1",
            constraints="=s,n",
            args=[cycles],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )


@triton.jit
def translate(ptr, src_base, dst_base):
    offset = tl.cast(ptr, tl.int64) - src_base
    return tl.cast(dst_base + offset, ptr.dtype)


@triton.jit
def triton_mm_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K]
    C_ptr,  # [M, N], symmetric
    flag_ptr,  # [grid_m, grid_n], symmetric
    bias_ptr,  # [N]
    M,
    N,
    K: tl.constexpr,
    local_rank: tl.constexpr,
    heap_bases,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 4,
    HAS_BIAS: tl.constexpr = False,
    REMAP_XCD: tl.constexpr = True,
    cache_modifier: tl.constexpr = "",
    WORLD_SIZE: tl.constexpr = 8,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)

    # re-order program ID for better L2 performance
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m_per_rank = grid_m // WORLD_SIZE

    pid_m, pid_n = compute_pid(pid, grid_m, grid_n, GROUP_M, REMAP_XCD)
    pid_m = (local_rank * num_pid_m_per_rank + pid_m) % grid_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    stride_am, stride_ak = K, 1
    stride_bn, stride_bk = K, 1
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    EVEN_K: tl.constexpr = K % BLOCK_K == 0
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B, cache_modifier=cache_modifier)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0, cache_modifier=cache_modifier)
        acc = tl.dot(a, b, acc)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + idx_n, mask=idx_n < N)
        acc += bias.to(tl.float32)

    dst_rank = pid_m // num_pid_m_per_rank
    local_base = tl.load(heap_bases + local_rank)
    dst_base = tl.load(heap_bases + dst_rank)
    dst_C_ptr = translate(C_ptr, local_base, dst_base)
    dst_flag_ptr = translate(flag_ptr, local_base, dst_base)

    dst_pid_m = local_rank * num_pid_m_per_rank + pid_m % num_pid_m_per_rank
    idx_m = dst_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    # inductor generates a suffix
    stride_cm, stride_cn = N, 1
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(dst_C_ptr + xindex, acc, mask=(idx_m < M) & (idx_n < N))

    # signal
    # not sure why need to spin here...
    addr = dst_flag_ptr + (dst_pid_m * grid_n + pid_n)
    # assert tl.load(addr) == 0, "flag is not 0"
    # tl.atomic_cas(addr, 0, 1, sem="release", scope="sys")
    while tl.atomic_cas(addr, 0, 1, sem="release", scope="sys") == 1:
        triton_sleep(5)


@triton.jit
def recv_kernel(
    C_ptr,  # [M, N], symmetric
    flag_ptr,  # [grid_m, grid_n], symmetric
    out_ptr,  # [local_M, N]
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    local_rank: tl.constexpr,
    WORLD_SIZE: tl.constexpr = 8,
):
    local_M = M // WORLD_SIZE

    pid = tl.program_id(0)

    num_pid_m_per_rank = tl.cdiv(local_M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    EVEN_N: tl.constexpr = (N % BLOCK_N) == 0

    for rank_offset in range(WORLD_SIZE):
        src_rank = (local_rank + rank_offset) % WORLD_SIZE
        src_pid_m = src_rank * num_pid_m_per_rank + pid_m

        # spin-lock
        addr = flag_ptr + (src_pid_m * grid_n + pid_n)
        while tl.atomic_cas(addr, 1, 0, sem="acquire", scope="sys") == 0:
            triton_sleep(5)

        # use mask when N = 2880. M is guaranteed to be divisible by BLOCK_M
        offs_m = src_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        if EVEN_N:
            data = tl.load(C_ptr + (offs_m * N + offs_n))
        else:
            data = tl.load(C_ptr + (offs_m * N + offs_n), mask=offs_n < N, other=0.0)
        acc += data.to(tl.float32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    if EVEN_N:
        tl.store(out_ptr + (offs_m * N + offs_n), acc)
    else:
        tl.store(out_ptr + (offs_m * N + offs_n), acc, mask=offs_n < N)


kernel_cache = dict()


def launch_triton(input, weight, bias, out):
    M, K = input.shape
    N, _ = weight.shape

    rank = P2P_STATE.rank
    world_size = P2P_STATE.world_size

    if M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 8, 128, 128
        GROUP_M = 4
    elif M <= 512:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
        GROUP_M = 2
    elif M <= 2048:
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
        GROUP_M = 2
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
        GROUP_M = 4
    HAS_BIAS = bias is not None
    REMAP_XCD = True
    cache_modifier = ".cg" if M <= 64 else ""

    args1 = (
        input,
        weight,
        P2P_STATE.C,
        P2P_STATE.flag,
        bias,
        M,
        N,
        K,
        rank,
        P2P_STATE.heap_bases,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        HAS_BIAS,
        REMAP_XCD,
        cache_modifier,
        world_size,
    )
    args2 = (P2P_STATE.C, P2P_STATE.flag, out, M, N, BLOCK_M, BLOCK_N, rank, world_size)
    num_pids1 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_pids2 = num_pids1 // world_size

    key = (M, N, K, HAS_BIAS, rank)
    if key not in kernel_cache:
        amd_kwargs = dict(
            waves_per_eu=2,
            matrix_instr_nonkdim=16,
            kpack=1,
            num_warps=8,
            num_stages=2,
        )
        kernel1 = triton_mm_kernel[(num_pids1, 1, 1)](*args1, **amd_kwargs)
        kernel2 = recv_kernel[(num_pids2, 1, 1)](*args2)
        kernel_cache[key] = (kernel1, kernel2)
    else:
        kernel1, kernel2 = kernel_cache[key]
        stream = torch.cuda.current_stream().cuda_stream
        for kernel, args, num_pids in [
            (kernel1, args1, num_pids1),
            (kernel2, args2, num_pids2),
        ]:
            launcher = kernel.run
            launcher.launch(
                launcher.launch_cooperative_grid,
                num_pids,
                1,
                1,
                stream,
                kernel.function,
                None,  # online env requires this
                kernel.packed_metadata,
                LazyDict(dict(name=kernel.name, function=kernel.function, stream=stream)),
                triton.knobs.runtime.launch_enter_hook,
                triton.knobs.runtime.launch_exit_hook,
                *args,
            )
    return kernel1, kernel2


def custom_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    # input: [M, local_K]
    # weight: [N, local_K]
    # bias: [N]
    # output: [local_M, N]

    M, local_K = input.shape
    N, _ = weight.shape

    rank = P2P_STATE.rank
    world_size = P2P_STATE.world_size

    out = input.new_empty(M // world_size, N)

    # launch_triton(input, weight, bias, out)
    # dist.reduce_scatter_tensor(out, F.linear(input, weight, bias))
    if M <= 2048:
        launch_triton(input, weight, bias, out)
    else:
        dist.reduce_scatter_tensor(out, F.linear(input, weight, bias))

    if False:
        torch.cuda.synchronize()
        dist.barrier()
        with torch.profiler.profile(with_stack=True) as prof:
            kernel1, kernel2 = launch_triton(input, weight, bias, out)

        K = local_K * world_size
        prof.export_chrome_trace(f"gemm-rs_{M}-{N}-{K}_rank{rank}.json.gz")
        if rank == 0:
            print(f"{kernel1.n_regs=}, {kernel1.n_spills=}")
        raise

    return out
