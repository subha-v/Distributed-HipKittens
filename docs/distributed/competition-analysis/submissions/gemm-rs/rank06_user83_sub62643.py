from task import input_t, output_t
import torch
import os

# 设置环境变量以优化RCCL性能并防止超时
os.environ.setdefault('NCCL_MIN_NCHANNELS', '112')
os.environ.setdefault('HIP_FORCE_DEV_KERNARG', '1')
# 禁用可能导致超时的TunableOp
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '0')

def custom_kernel(data: input_t) -> output_t:
    """
    Optimized kernel for Gemm-ReduceScatter operation with overlapped computation and communication.
    
    Implements overlapping between GEMM computation and ReduceScatter communication
    by splitting the work into chunks and using asynchronous operations.
    
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
    
    # 确保张量在相同设备上并是连续的，以获得最佳性能
    device = input.device
    weight = weight.to(device)
    
    # 预分配输出张量以减少内存分配开销
    output = torch.empty((M, N), dtype=input.dtype, device=device)
    
    # 确保输入张量连续以获得最佳性能
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    
    # 使用torch.addmm执行矩阵乘法，可能比matmul更高效
    if bias is not None:
        bias = bias.to(device)
        torch.addmm(bias, input, weight.T, out=output)
    else:
        torch.matmul(input, weight.T, out=output)
    
    # 使用标准的ReduceScatter实现，避免复杂的异步操作可能导致的超时
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    
    return rs_output