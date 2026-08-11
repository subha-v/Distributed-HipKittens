import torch, torch.nn.functional as F, torch.distributed as dist
_c = {}
def custom_kernel(d):
    o = F.linear(*d)
    k = ((d[0].shape[0]//8, d[1].shape[0]), o.dtype, o.device)
    if k not in _c: _c[k] = torch.empty(k[0], dtype=o.dtype, device=o.device)
    dist.reduce_scatter_tensor(_c[k], o)
    return _c[k]