import os
for k,v in{
 "OMP_NUM_THREADS":"1",
 "HSA_FORCE_FINE_GRAIN_PCIE":"1",
 "RCCL_P2P_ENABLE":"1",
 "RCCL_ALLOW_IB_RAILS":"1",
 "TORCH_NCCL_ASYNC_ERROR_HANDLING":"1",
 "TORCH_NCCL_BLOCKING_WAIT":"0",
 "RCCL_BUFFER_SIZE":"67108864",
 "RCCL_DISABLE_COLL_TRACE":"1",
 "RCCL_DEBUG":"ERROR"
}.items():os.environ.setdefault(k,v)

import torch, torch.nn.functional as F, torch.distributed as dist
_c={}
def custom_kernel(d):
    o=F.linear(*d)
    k=((d[0].shape[0]//8,d[1].shape[0]),o.dtype,o.device)
    if k not in _c:_c[k]=torch.empty(k[0],dtype=o.dtype,device=o.device)
    dist.reduce_scatter_tensor(_c[k],o)
    return _c[k]