from task import input_t, output_t
import torch
import torch.distributed as dist
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def custom_kernel(data: input_t) -> output_t:
    """
    AG-GEMM with backend optimizations enabled.
    
    Enables:
    - cuDNN benchmark mode (auto-selects best algorithm)
    - TF32 for faster computation (slight precision trade-off)
    
    May save 5-10μs
    """
    input, weight, bias = data
    local_M, K = input.shape
    world_size = dist.get_world_size()
    
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    dist.all_gather_into_tensor(full_input, input)
    
    if local_M >= 2048 or local_M <= 4096:
        output = F.linear(full_input, weight, bias)
    else:
        output = torch.matmul(full_input, weight.T)
    
        if bias is not None:
            output = output + bias
    
    return output 