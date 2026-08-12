# exp_08 (axis A8) — die-level vs CTA-level service pool: the mechanism confirmed, the trade-off measured

**Verdict: the interference hypothesis is CONFIRMED and the cure is worse than
the disease.** Confining the service pool to whole XCDs cuts M7 interference by
**36%** — direct evidence that the interference is per-XCD L2 contention — but
slows the service pool itself by **54%**, because a pool squeezed onto two dies
must share those two dies' L2 and memory ports. Net **26% worse** overall.

The finding is not "placement doesn't matter". It is sharper than that:

> **The interference and the service throughput are the same resource seen from
> two sides. You cannot isolate the communication engine from the compute
> without also starving it.**

## The change — mode 3, no ABI or harness edit

Workgroups go round-robin to XCDs with chunk size one (`xcd = bid % 8`), so
mode 2's static-tail reservation **spreads** the pool one-eighth per die and
touches all eight 4 MB L2s. Mode 3 is mode 2 with exactly one difference:
residue classes `[0, C/32)` are service, `[C/32, 8)` are compute. One XCD is
`256/8 = 32` CTAs, so `C` must be a whole number of dies; with the existing
`C ≤ 64` cap the legal points are `C = 32` (one die) and `C = 64` (two dies).

`mode` is already an 8-bit config field parsed as an integer, so mode 3 needed
**no descriptor change, no `moe_host_abi.hpp` change and no harness edit** —
`K0_MPS_CFG="C=64,g=1,mode=3,flush_rows=16"` just works.

The compute set is no longer the dense prefix `[0, nct−C)`, so the M7 task loop
needed a dense remap: `compute_id = (bid/8)·(8−D) + (bid%8 − D)`. The vendored
phase-2 body already exposed `N2GM_TASK_START` for exactly this ("so the
includer can map a dense logical compute-id space onto the same task set"), so
it was a macro swap rather than surgery.

**Resource-free**: SGPR 104 / VGPR 256 / AGPR 256 / scratch 128 B / LDS
155,428 B — byte-identical to mode 2. The extra descriptor read for the task
start cost nothing.

**Correctness validated**: all three screened configurations passed
`[MOK GATE]`, `control_fails=True` and `[MPS SOAK] 600/600`. Since a wrong
dense-id remap would skip or duplicate M7 tasks, the green gates are a real
check on the remap, not a formality.

## Result

`C = 64` — same 192 compute CTAs, same 64 service CTAs, **only placement
differs**. Matched mode-0 control (no service traffic): M7 = 2,019.7 µs.

| placement | M7 | interference | combine | service drain | total |
|---|---:|---:|---:|---:|---:|
| mode 2 — spread over 8 XCDs | 3,125.3 | **+1,105.6** | 3,162.3 | 5,914.1 | 10,091.3 |
| mode 3 — 2 whole XCDs | 2,723.4 | **+703.7 (−36%)** | 6,462.0 | 9,127.7 (+54%) | **12,710.6 (+26%)** |

`C = 32` — one whole die. Matched control M7 = 1,784.1 µs.

| placement | M7 | interference | combine | total |
|---|---:|---:|---:|---:|
| mode 2 — spread | 2,479.1 | +695.0 | 8,226.5 | 14,774.8 |
| mode 3 — 1 die | 2,269.3 | **+485.2 (−30%)** | 15,385.3 | **22,083.2 (+49%)** |

Both `C` points tell the same story, and the magnitudes are consistent:
**−30 to −36% interference, +54 to +87% service cost.**

## Reading

1. **The interference mechanism is now positively identified, not just
   inferred.** Moving the service CTAs to different physical dies, changing
   nothing else, removes a third of the M7 inflation. That can only be a
   per-XCD locality effect. Combined with exp_07 (coalescing atomics onto few
   lines is catastrophic) and exp_06 (ordering scope is ~30% of the floor), the
   picture is complete: **the service pool's scattered atomics contend in the
   L2s of whichever dies issue them.**
2. **But the service pool needs those same dies.** Sixty-four service CTAs on
   two XCDs share two dies' worth of L2 capacity, bandwidth and fabric ports,
   and self-contend. The 36% M7 saving is bought with a 54% service slowdown on
   a phase that is already the post-M6 critical path (exp_05), so the trade is
   badly negative.
3. **A8 is closed.** Both of its arms are now measured: the spread arm (mode 2,
   all night) and the concentrated arm (mode 3, here). Concentration loses at
   every `C` tested. The pre-registered framing — "contiguous = pollute eight
   4 MB L2s; strided = sacrifice one die, keep seven clean" — was the right
   question, and the answer is that the sacrificed dies cannot carry the work.
4. **A hypothetical middle exists and is not obviously better.** Three or four
   service dies would trade less M7 relief for more service throughput, but the
   two measured points already bracket it: the interference saving is roughly
   linear in dies-freed while the service penalty grows faster, and mode 2
   (eight-way spread, the limit of the series) is the best point of the family.

## Disposition

**Mode 3 is KEPT in the source** as a validated, gate-green, resource-free
alternative placement, and as the mechanism probe it turned out to be. It is
**not** a candidate — mode 2 at `C=64, g=1` remains the best MPS point at
10,075.8 µs (dec05). Nothing about the ratchet changes.

## Primitives

`roles.cuh` gives `finish_order_partition` → `role_partition` with
`is_service` / `is_compute` and dense `service_id` / `compute_id`. That is
**exactly the right surface** and this experiment is evidence for it: the entire
placement change reduced to swapping one predicate and two dense-id functions,
with the vendored GEMM body untouched. What the library does **not** offer is
any notion of **physical placement** — the caller must know that
`xcd = bid % 8` on this part, hard-code `32` CTAs per die, and derive its own
residue-class arithmetic. A `role_partition` variant parameterised by
*hardware domain* (per-XCD, per-shader-engine) rather than by CTA index would
have made mode 3 a parameter instead of a new mode, and would carry the
`bid % 8` fact in one place instead of in every kernel that ever wants it.
Given that placement is now measured to move M7 by 36%, that is a primitive
worth having.

## Artifacts

`~/overnight-scratch/E08xcd_*.log`, `~/overnight-scratch/screen_E08xcd.csv`,
JIT hash `cd75270a36a5`, commit `9c1f49c2`.
