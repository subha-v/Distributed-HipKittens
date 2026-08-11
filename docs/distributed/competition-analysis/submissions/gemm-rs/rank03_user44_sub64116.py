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
    torch.distributed.barrier()

    output = module.launch_reduce_scatter(ipc_tensor, signal_tensor, round_trip, rank)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if round_trip < 8:
        dist_print_err(f"rank {rank} gemm time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
        dist_print_err(f"shape: {M}x{N}x{local_K}, time: {((t2 - t1)*1e6):.2f}μs, perf: {((M*N*2)/(t2 - t1) * 1e-9):.2f}GB/s")
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
        dist_print_err(f"rank {rank} fused time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
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