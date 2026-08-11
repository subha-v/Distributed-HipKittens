from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    local_M, K = input.shape
    full_input = torch.empty((local_M * 8, K), dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(full_input, input)
    output = torch.addmm(bias, full_input, weight.T) if bias is not None else torch.matmul(full_input, weight.T)

    return output
