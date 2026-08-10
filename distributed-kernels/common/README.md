# Distributed-kernel runtime boundary

IRIS owns the symmetric fine-grained heap, IPC handle exchange, rank-to-device
mapping, and host/MPI barriers. HipKittens owns device-side views and ordering.
`iris_adapter.hpp` snapshots IRIS's heap bases on the host, validates that the
runtime world matches the PGL world, and shifts every base by the local
allocation's heap offset. Operator kernels should carry the resulting named
allocation bases or a `kittens::pgl`, not `iris::iris_device_view`, through
their device globals.

The local allocation base passed to `snapshot_layout` must be the base of the
IRIS allocation containing `local.raw_ptr`. The adapter first preserves its
heap offset; PGL then preserves the independent view offset inside that
allocation. Both levels are required when neither the allocation nor tensor
view begins at offset zero. This address projection is intentionally separate
from synchronization.

The adapter targets the public `<iris/iris.hpp>` API used by
`distributed-kernels/CMakeLists.txt`. The required device view operations are
`world_size()`, `cur_rank()`, and `get_heap_base(rank)`.

Operator-specific host ABI policy stays beside each candidate:
`gemm_rs/gemm_rs_host_abi.hpp` converts two IRIS allocations into exact device
PODs, while `fused_moe/moe_host_abi.hpp` snapshots the shared heap and extends
the frozen descriptor vector. Neither helper allocates device memory, uploads,
launches, or hides lifetime ownership.
