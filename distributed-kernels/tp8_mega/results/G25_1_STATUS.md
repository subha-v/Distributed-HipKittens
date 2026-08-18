# G25-1 status — honest interim (2026-08-18 ~20:15 UTC)

## Where the gate stands

**Correctness: PASS everywhere.** Every arm and configuration of the
boundary rig verifies exact bf16 sums with zero timeouts across all runs.

**Performance: the fused CDAR schedule does NOT yet beat production's
GEMM -> RCCL at the single-boundary level.** Best measured (16K tokens,
256-row slabs, MFMA body):

| arm | wall (k_inner=2048, bps=2, 256 blocks) |
|---|---:|
| compute floor (two bursts) | 1,439 us |
| GEMM -> RCCL -> GEMM (production schedule) | 2,909 us |
| GEMM -> phased CDAR -> GEMM (unfused control) | 3,653 us |
| fused CDAR (current best schedule attempt) | 3,606 us |

Two separable deficits: (1) CDAR standalone transport is ~1.5x slower than
RCCL's AR (125 vs ~192 GB/s effective); (2) the fused schedule achieves
~zero hiding — fused tracks phased within noise in every variant tried.

## Hypotheses tried and falsified (each is now encoded in the rig)

1. **Per-item drain serialization** (v2: defer drain, arrive one item late,
   stores ride under the next MFMA burst) — no change.
2. **Load->store chain re-copy** (v3: write-once from registers, the M15
   epilogue's actual profile) — no change.
3. **No specialized consuming pool** (v4: duty blocks skip production and
   gather in the producers' shadow; 256 blocks; producer fallthrough) — no
   change.
4. **Item granularity** (v5: fragments-per-slab runtime knob; at bps=16 the
   commits spread across the compute span) — fused got WORSE (+180 us of
   protocol) with still no hiding.

## What is established

- K1 = store-towers (93.6 -> 117 GB/s at 256-row slabs; atomic caps at 67,
  dword issue rate). K3 = 256-row slabs (117 GB/s; 512 gives 93, 1024
  gives 55). K4 = flat {4..32} in pure transport; unresolved in the fused
  regime because no fused variant has yet reached the regime where it
  would matter.
- RCCL reference on this node, this shape: AR ~1.2-1.4 ms for 235 MB
  (grows ~+200 us when co-resident with MFMA — RCCL contends too).
- The MFMA-class body itself is sound (floor scales with blocks and
  k_inner as expected).

## Next step (before ANY further schedule mutation)

Instrument the rig with a device phase ledger (the [MPS TS] discipline):
per block-class timestamps for produce span, first/last commit, first/last
certify, first/last pull, burst2 span. The four falsified hypotheses were
all schedule-side guesses; the ledger will say whether the wall is (a)
consumer-side remote-READ bandwidth in MAG pull, (b) certify latency chain
(gather polls parked behind produce), (c) reduce HBM bandwidth colliding
with burst2 reads, or (d) something else. One instrumented run replaces
the next four guesses.

Secondary avenue once the ledger speaks: the MAG pull can become an
owner-push (store-class, write bandwidth) multicast during the reduce
drain — but only if the ledger blames (a).
