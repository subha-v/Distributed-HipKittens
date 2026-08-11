#!POPCORN leaderboard  amd-gemm-rs
import sys
import time
from task import input_t, output_t
from typing import Optional
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
import zlib
import base64
import os
os.environ.update(
    {
        "HSA_XNACK": "0",
        "CXX": "clang++",
        "PYTORCH_ROCM_ARCH": "gfx942",
    }
)


CPP_WRAPPER = zlib.decompress(base64.b64decode(b'eNoDAAAAAAE=')).decode("utf-8")
CUDA_SRC = zlib.decompress(base64.b64decode(b'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')).decode("utf-8")

module = load_inline(
    name='perf_gemm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__", "-D__GPUMODE_BENCHMARK__"],
    extra_cflags=["-Ofast", "-ffast-math", "-march=native", "-funroll-loops", "-fomit-frame-pointer"],
)

comm = None
should_udpate_comm = True
round_trip = 0

orignal_init_pg = dist.init_process_group

def hooked_init_pg(*args, **kwargs):
    global should_update_comm
    should_update_comm = True
    ret = orignal_init_pg(*args, **kwargs)
    # print0(f"init pg: {args}, {kwargs}", True)
    return ret

def dist_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)

def dist_print_err(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True, file=sys.stderr)
        


def all_get_comm():
    global comm, should_update_comm, round_trip
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    if should_update_comm:
        should_update_comm = False
        round_trip = 0
        # create a new comm
        world_size = dist.get_world_size()
        dist_print(f"create new comm: {rank} {world_size}")
        del comm
        comm = module.GemmRS(rank, world_size)
        ipc_handle = comm.get_ipc_handle()
        ipc_handles = [None] * world_size
        dist.all_gather_object(ipc_handles, ipc_handle)
        comm.init_dist(ipc_handles)
        dist.barrier()
        torch.cuda.synchronize()
    round_trip += 1
    return comm, round_trip

dist.init_process_group = hooked_init_pg


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output

def ref_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.distributed.barrier()
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    comm, round_trip = all_get_comm()
    if round_trip < 8:
        dist_print_err(f"rank {rank} gemm time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
        dist_print_err(f"shape: {M}x{N}x{local_K}, time: {((t2 - t1)*1e6):.2f}μs, perf: {((M*N*2)/(t2 - t1) * 1e-9):.2f}GB/s")
    return rs_output

def gemm_then_rs_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    output = module.launch_gemm(input, weight, bias, signal_tensor[rank], round_trip, ipc_tensor[rank])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if round_trip < 8:
        dist_print_err(f"shape: {M}x{N}x{local_K}, gemm time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
    torch.distributed.barrier()

    output = module.launch_reduce_scatter(ipc_tensor, signal_tensor, round_trip, rank)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if round_trip < 8:
        dist_print_err(f"shape: {M}x{N}x{local_K}, rs time: {((t2 - t1)*1e6):.2f}μs, perf: {((M*N*2)/(t2 - t1) * 1e-9):.2f}GB/s")
    return output
    rs_output = ref_kernel(data)
    if not torch.allclose(output, rs_output, atol=1e-2, rtol=1e-2):
        dist_print_err("mismatch in gemm_then_rs_kernel")
        dist_print_err("output:", output)
        dist_print_err("rs_output:", rs_output)
        diff = torch.abs(output - rs_output)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return rs_output

def gemm_then_rs_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    output = module.launch_gemm(input, weight, bias, signal_tensor[rank], round_trip, ipc_tensor[rank])
    output = module.launch_reduce_scatter(ipc_tensor, signal_tensor, round_trip, rank)
    return output


def gemm_rs_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, _ = input.shape
    N = weight.shape[0]
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    rs_output = module.launch_fused(input, weight, bias, ipc_tensor, signal_tensor, round_trip, rank)
    # dist_print_err(f"gemm_rs_kernel {input.shape}x{weight.shape} rank {rank} round_trip {round_trip}")
    return rs_output

def gemm_rs_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    rs_output = module.launch_fused(input, weight, bias, ipc_tensor, signal_tensor, round_trip, rank)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.distributed.barrier()
    if round_trip < 8:
        dist_print_err(f"shape {M}x{local_K}x{N} ? {torch.all(bias == 0).item()} fused time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
    rs_output_ref = ref_kernel(data)
    if not torch.allclose(rs_output, rs_output_ref, atol=1e-2, rtol=1e-2):
        dist_print_err("mismatch in gemm_rs_kernel")
        dist_print_err("rs_output:", rs_output)
        dist_print_err("rs_output_ref:", rs_output_ref)
        diff = torch.abs(rs_output - rs_output_ref)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return rs_output_ref
    

def check_implementation(data: input_t) -> output_t:
    rf_res = gemm_rs_kernel(data)
    ref_res = ref_kernel(data)
    x_shape = data[0].shape
    w_shape = data[1].shape
    
    if not torch.allclose(rf_res, ref_res, atol=1e-2, rtol=1e-2):
        dist_print_err(f"{x_shape[0]}x{x_shape[1]}x{w_shape[0]} mismatch")
        dist_print_err("rf_res:", rf_res)
        dist_print_err("ref_res:", ref_res)
        diff = torch.abs(rf_res - ref_res)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return ref_res 

# custom_kernel = check_implementation
custom_kernel = gemm_rs_kernel
# custom_kernel = gemm_then_rs_kernel_debug
# custom_kernel = ref_kernel
# custom_kernel = ref_kernel_debug
# custom_kernel = gemm_rs_kernel_debug