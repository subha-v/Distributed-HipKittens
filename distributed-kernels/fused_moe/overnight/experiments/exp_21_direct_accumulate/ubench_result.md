# ubench_fabric_rate — results (gbt350-odcdh2-c05-1, gfx950, one link dev1→dev0)

Coarse-grain hipMalloc + P2P; rate is a fabric-path property, not heap grain.

| pattern | GB/s | G op/s |
|---|---:|---:|
| `stores-PEER` (16 B packets, pool's shape) | 54.9 | — |
| **`atomic-PEER` (4 B pk-bf16, coalesced, epilogue shape)** | **52.8** | **13.2** |
| `atomdr-PEER` = `atomic` + per-task `vmcnt(0)` | 52.8 | 13.2 |
| `atomSC-PEER` (64 distinct lines per wave) | 4.1 | 1.03 |
| `atomic-LOCAL` (same-op L2 reference) | ~1,140 | 285 |
| `atomic-PEER` with the TARGET GPU streaming weights | 52.8 | 13.2 |

Flat in writer-CTA count (64 → 256).

## Readings, with the exact consequences for mode 12/13

1. **Coalesced 4-B remote atomics run at the SAME per-link byte rate as 16-B
   stores** (52.8 vs 54.9 GB/s). The xGMI path here is **byte-limited, not
   op-limited** when atomics come in coalesced half-wave bursts — exactly the
   epilogue's pattern (32 consecutive dwords). The 4× op-count of the
   4-B-atomic design costs nothing at the byte level.
2. **The merge is load-bearing:** scatter the same op count across distinct
   lines and throughput collapses 13× (1.03 G op/s). If routing skew ever
   makes the epilogue's half-waves address-disjoint, the fabric op rate
   becomes the wall — a skew-stress caveat for the campaign, not the uniform
   route.
3. **The per-task `vmcnt(0)` drain is free** in steady state (identical 13.2
   G op/s): the task-done hook's drain does not expose fabric ACK latency.
4. **Target-side traffic does not slow inbound atomics** (52.8 GB/s with the
   owner GPU streaming weights full-rate). The mode-12 epilogue's outbound
   stream will not be throttled by owners' MFMA operand streams.
5. **The open question is source-side contention** (does issuing the atomic
   stream from compute CTAs hurt their OWN MFMA issue/L2 — exp_20 §2b's
   in-situ +624 µs shape). A microbenchmark cannot answer that; only the
   mode-12/13 in-kernel phase stamps do.

Per-link requirement check (mode 12): ~312 MB/rank spread over 7 links
≈ 44.6 MB/link; at 52.8 GB/s sustained that is ~845 µs of fabric occupancy
for the whole combine payload — inside M7's 1,586 µs envelope with ~2× slack.
