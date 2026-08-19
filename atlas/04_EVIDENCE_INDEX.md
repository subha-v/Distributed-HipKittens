# Evidence index — where every number lives

Pointer table into the primary sources on this branch (paths relative to repo root unless a
branch is named). Nothing in the atlas is quotable without a row here.

## Design layer (the cost-model / workload→schedule answer)

Base: `distributed-kernels/fused_moe/overnight/aug18-ablations/design/`

| file | one-line takeaway |
|---|---|
| `W_VECTOR.md` | workload state vector + the mediating quantities: change X ⇒ Q crosses θ ⇒ schedule changes by Y |
| `S_VECTOR.md` | schedule state vector: kernel degrees of freedom as parameterized skeletons + manifest, not one-off kernels |
| `COST_MODEL.md` | M(W, S; θ) with exactly two named, bounded fudge factors; calibration plan; predicted decision boundaries |
| `OVERLAP_ATLAS.md` | every overlap opportunity in every regime, sized where the repo has a number, screened against banked laws |
| `EXPERIMENT_LADDER.md` | the gated grid that would validate the cost model |
| `METRICS_PROTOCOL.md` | which metric (TTFT/TPOT/throughput) is primary per workload cell — the "TPS without batch size means nothing" corrective |
| `G_L4_INTEGRATION_MAP.md` | the TP8+EP serving seam: MoE region bounds, AR sites, shared-expert site, replace/keep table |

## Grounding briefs (banked facts the design layer cites)

Base: `distributed-kernels/fused_moe/overnight/aug18-ablations/grounding/`

| file | one-line takeaway |
|---|---|
| `BANKED_LAWS.md` | LAW-n: every reproducible mechanism-level finding with its evidence |
| `KNOB_INVENTORY.md` | S, K1–K5, S0–S8, F1–F10 — every schedule knob we have ever actuated |
| `WORKLOAD_EVIDENCE.md` | W1–W10, H1–H17 — what we know about workload structure |
| `TP8_STATE.md` | the TP8 rig state, M1–M9 phases, A/B arm inventory |

## Results ledgers (measured numbers)

| file | one-line takeaway |
|---|---|
| `…/aug18-ablations/UNDERSTANDING_PLAN.md` §2, §8 | the specimen tables + gate outcomes: ρ-ladder 48/48 PASS, filler GO, β=1.529 direct |
| `…/aug18-ablations/results/RHO_LADDER_G0A.md` | fusion hides nothing at any reachable ρ; four G25-1 nulls are ρ-independent; β stable 1.43–1.60 |
| `…/aug18-ablations/results/FILLER_MICROCOSM.md` | 235 MB AR loses 0.00% beside full-GPU MFMA; 44–53% of exposed AR absorbed free ⇒ G-L2 GO |
| `…/aug18-ablations/results/runs.jsonl` + `clocks_rho_ladder.csv` | raw run manifest + clock/thermal guard data |
| `…/aug18-ablations/review/*.md` | the adversarial reviews (alignment, mechanism, contribution, feasibility) folded into plan r2 |
| `…/aug12/exp_03_m15_slab_combine/result.md` | M15's C-response curve, phase attribution, m15b/m17 adjudications |
| `…/aug14/M18_REPLICATION_RESULTS.md` | skew measurement, replay rig, the replication ladder with CIs |
| `…/aug18/V6_EVIDENCE_AND_DESIGN.md` | training bubble ledger, every falsified wgrad-filler arm, LAW-52 |
| `distributed-kernels/tp8_mega/results/G25_1_STATUS.md` | the four falsified fused-CDAR hypotheses + K1/K3/K4 knob results |
| `distributed-kernels/tp8_mega/LEDGER_NOTES.md` | the phase-ledger instrument, five LAW-63 guards, ρ↔k_inner map |
| `nightshift/REPORT.md` | M1–M12 measurement table with validity classes + seven retractions — the best "what is and isn't proven" doc |

## Prefill campaign designs (anchor lineage)

Base: `distributed-kernels/fused_moe/overnight/aug18-prefill/`

| file | one-line takeaway |
|---|---|
| `SHARED_EXPERT_FILLER_DESIGN.md` | the G-L2 filler design: shared-expert GEMM under the AR window |
| `M25_TP8_MEGA_DESIGN.md` | the TP8 mega arm (anchor B) |
| `TP8_OVERLAP_ANALYSIS.md` | where overlap can live in the TP8 step |
| `M23_RAGGED_SEAL_DESIGN.md` | the coverage fix: mega ran ~2% of serving steps before the ragged seal |
| `M15_EPLB0_COMPOSE.md` | composing M15 with EPLB routing balance |
| `BOTTLENECK_SIGNALS.md` | three-estimator composition correction (mega not losing in serving, r≈0.95) + the 62.4%-padding finding |

## Kernel sources on this branch

Base: `distributed-kernels/fused_moe/` — `k0pf6gm_device_tile*.hip` are the device images
(base, `_m15`, `_mps`, `_t2b`, `_t2v6`, `_t3`), `moe_host_abi.hpp` / `*_abi.hpp` the host↔device
contracts, `n2_phase*` the phase drivers. `distributed-kernels/tp8_mega/` holds the anchor-B
arms (rcclserial / rcclchunk / rcclhybrid_indep / rcclhybrid_merged / filleronly behind
`-DM25_G1=1`). Cross-branch variants are cataloged in `01_KERNEL_CATALOG.md`.

## Existing abstraction layer (the extracted primitives)

| file | one-line takeaway |
|---|---|
| `docs/distributed/OVERLAP_ABSTRACTIONS.md` | the primitive-extraction doc the headers cite (knobs → primitives, arch cards) |
| `docs/distributed/PRIMITIVES.md`, `ARCHITECTURE.md` | the device-side multi-GPU layer: pgl, peer-memory, publication/acquire, replay-lifetime, role partitions |
| `include/{cdna3,cdna4}/ops/group/distributed/*.cuh` | the extracted contracts as code: `credit.cuh` (bounded injection, K4, the −615 µs knob), `slab.cuh` (coarse prefix certification, K3, replaced ~926k row ops with 2×8 certificates), `counter.cuh` (counted fan-in), `lifetime.cuh` (ready ≠ reusable), `order.cuh`, `roles.cuh` (carrier identity), `peer.cuh`, `packet.cuh`, `sync.cuh`, `completion.cuh` — each header carries its measured provenance + gfx942/gfx950 arch cards |
| `docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md` | why pre-M23 serving A/Bs are void; the current attested methodology |

The four-generation abstraction lineage in the companion repo (`k0_tile` → `flow` → `hkp` →
`mk`, plus the portable `amd_tile` core and the iris/irisx substrate) is surveyed in
`05_AMD_MASTER_IMPORTS.md`.

## Serving-level protocol + findings

| file | one-line takeaway |
|---|---|
| `nightshift/BENCHMARK_PROTOCOL.md` | order-balanced pairs, ≥5 pairs, alternation-within-session rule (LAW-60) |
| `nightshift/CORPUS_FINDINGS.md` | the serving corpus audit (31%-dummy finding) |
