# The MoE Megakernel Atlas

One place that organizes every MoE megakernel we have built across this repo's branches (and
the useful pieces of `amd-master`), the evidence for why each is fast or slow, and the
**data-movement abstractions** that fall out. Built for the anatomy conversation: it is fine
that some kernels are fast and some are slow — the deliverable is the *mechanism* behind each,
and the portable contracts those mechanisms imply.

## How to read this atlas

| doc | what it holds |
|---|---|
| [`01_KERNEL_CATALOG.md`](01_KERNEL_CATALOG.md) | every megakernel variant across every branch: where it lives (`branch:path`), its data-movement strategy, its verdict + evidence |
| [`02_WHY_FAST_WHY_SLOW.md`](02_WHY_FAST_WHY_SLOW.md) | the causal ledger: the anchor-A waterfall (7,712 → 5,822 µs), the anchor-B inversion (the same instincts lose), the failure museum, the measured sign flip |
| [`03_ABSTRACTIONS.md`](03_ABSTRACTIONS.md) | the data-movement contracts (Iris/IrisX-aligned, hardware-agnostic) — each mapped to the kernel variants that instantiate it and the workload property that governs its parameters |
| [`04_EVIDENCE_INDEX.md`](04_EVIDENCE_INDEX.md) | pointer table into the primary evidence: design docs, results ledgers, banked laws |
| [`05_AMD_MASTER_IMPORTS.md`](05_AMD_MASTER_IMPORTS.md) | what lives in `~/repos/amd-master` (abstractions, findings, vLLM integrations) and where |
| [`06_DRAFT8_RESULTS.md`](06_DRAFT8_RESULTS.md) | slide-by-slide ledger of every number presented in draft8 (incl. the +8.40% C=512 e2e headline), its evidence home, and its current standing |

## The primary design documents (already in this branch)

The experts asked for a cost model and an enumeration of the workload properties that control
the schedule. That work exists, at
`distributed-kernels/fused_moe/overnight/aug18-ablations/design/`:

| doc | role |
|---|---|
| `W_VECTOR.md` | the workload state vector — "change X in the workload ⇒ mediating quantity Q crosses threshold θ ⇒ schedule changes by Y" |
| `S_VECTOR.md` | the schedule state vector — the kernel degrees of freedom, so "hundreds of kernels" collapse into parameterized skeletons + a manifest |
| `COST_MODEL.md` | M(W, S; θ): the analytic model, exactly two named+bounded fudge factors, calibration plan, predicted decision boundaries |
| `OVERLAP_ATLAS.md` | every comm/compute overlap opportunity across every regime — the corrective to "you only exploited one overlap region" |
| `EXPERIMENT_LADDER.md`, `METRICS_PROTOCOL.md`, `G_L4_INTEGRATION_MAP.md` | the gated experiment plan, per-workload metric definitions, and the TP8+EP serving seam |

The active campaign plan is
`distributed-kernels/fused_moe/overnight/aug18-ablations/UNDERSTANDING_PLAN.md` —
Track 1 (beat production at low concurrency: G-L0 → G-L4 gates) plus Track 2 (the anatomy run
list R2–R11 that closes the open attribution questions Q1–Q12).

## The thirty-second version

- **Anchor A (DP8/EP8 prefill, token all-to-all):** fusion −804 µs, bounded injection −615 µs,
  producer/consumer pool −662 µs ⇒ **M15 at 0.7544× of production.** Fast because each step
  fixes a named data-movement defect: launch boundaries, unbounded admission, injector
  concurrency.
- **Anchor B (TP8+EP prefill, 235 MB all-reduce):** the same fuse-and-own-the-transport design
  **loses by 24%** — the fused protocol costs +15 µs (free); the deficit is transport β=1.529
  vs RCCL plus co-residency tax. The winning shape is **rent RCCL, fill the non-CU-bound AR
  window with the shared expert** (measured: AR loses 0.00% beside a full-GPU MFMA; 44–53% of
  exposed AR absorbed free).
- **Therefore:** no single kernel wins everywhere; the schedule is a function of the workload
  through nameable mediating quantities — and the portable layer is a small set of
  **data-movement contracts** (`03_ABSTRACTIONS.md`), of which our HIP headers are one backend.
