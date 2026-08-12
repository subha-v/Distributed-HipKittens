# exp_01 — discriminator matrix: the fault is CONFIG-INDEPENDENT

Five full runs (no debug stop), `K0_MOK_ARMS=production,mps_mega`, warmup/timed
= 1/1, one process each, serialized. 2026-08-11 07:08Z–07:17Z.

| run | `K0_MPS_CFG` | service pool | mem-faults | verdict |
|---|---|---:|---:|---|
| `pullfb` | `C=8,g=2,mode=2,flush_rows=16,pull_fallback=1` | 8 CTAs | **8** | FAULT |
| `mode0c8` | `C=8,mode=0` | — | 0 | **VOID** — config rejected by host |
| `mode1c0` | `C=0,mode=1` | — | 0 | **VOID** — config rejected by host |
| `mode0c8v2` | `C=8,g=2,mode=0,flush_rows=16` | 8 CTAs, push OFF | **8** | FAULT |
| `mode1c0v2` | `C=0,g=2,mode=1,flush_rows=16` | **none (C=0)** | **8** | FAULT |

The two VOID rows are mine to own: `K0_MPS_CFG` requires all four of
`C,g,mode,flush_rows`, I passed two, and the host raised
`ValueError: K0_MPS_CFG requires C,g,mode,flush_rows ...` on every rank before
any launch. Their "0 faults" is the absence of a run, not a clean run. Rerun as
`*v2`. Lesson: always pass the full four-key config; a partial config fails
*silently-looking* (no fault, no progress log) and reads like a pass.

## What this overturns

Every arm faults, in all three modes, with the push mechanism on AND off, and
**with `C=0`**. `mode=1` requires `C=0`, so that run reserved **zero** service
CTAs — there was no service pool in existence — and it still took 8 faults.

Therefore:

- **Suspect #3 (`run_service` internals) is DEAD.** With `C=0` no service CTA
  exists, so no service-loop code can be the faulting agent.
- **Suspect #2 (M8 dynamic-claim / slot path) is not *specifically* implicated,**
  and `pull_fallback=1` — which reroutes M8 to remote `part` pulls — did not
  save it either.
- **The service/stream protocol as a whole is exonerated as the *trigger*.** The
  handoff's reading of the `pull_fallback` discriminator ("if it still faults,
  it's in the service/stream protocol") assumed the stream was the only variable.
  The `C=0` run breaks that assumption: the fault survives the removal of the
  entire mechanism.

The fault is in something the M7.6/M8/M9 tail executes **unconditionally**,
independent of `mode`, of `C`, and of `pull_fallback`. That is a much smaller
search space than the handoff's three suspects, and it is consistent with the
debug-stop bisect (stops 1–6 are clean precisely because they never enter the
tail).

## How this composes with the descriptor evidence

`desc_analysis.md` showed `desc[61]` lives in the torch-allocator address family
(256 B-packed, `0x7C98…`) rather than the 2 MiB-aligned symmetric-heap family
(`0x7C5C…–0x7C5F…`). If the tail dereferences that arena — or its peer
translation — on a path every config takes, then every config faults. The two
independent findings agree on one story: **a pointer consumed unconditionally by
the tail is not the kind of pointer the tail assumes it is.**

What still needs explaining, and must not be waved away: a wrong-family pointer
translated as `local - local_heap_base + peer_base[r]` yields a large garbage
address, whereas the reported fault address is exactly `(nil)`. Either the
faulting access is a *different*, genuinely null pointer, or the translation
produces 0 for a reason we have not yet written down. The fix is not credible
until the `(nil)` is derived, not assumed.

## Collateral evidence (from the `pullfb` log)

The reference machinery is healthy — all four K0PF gates pass with zero diffs:
`plan_equivalence pass=True` (all diff counters 0, `perr=0/0`, `padded=33440`,
`T_loc=21816`), `quant_exact pass=True byte_diffs=0`, `gather_exact pass=True
byte_diffs=0`, `combine_bit_exact pass=True bf16_bit_diffs=0`. So plan, quant,
gather and the combine reference are not suspects.

One anomaly to hand to benchmark-review, not blocking: `production` reports
`rel_L2=0.00647 pperr=0` — matching its known-good ~0.0065 — yet `pass=False`.
A known-good arm marked failing means the `pass` predicate is measuring
something other than what its name suggests, and I will not trust that flag as a
gate until it is explained.

## Node lease

Preflight clean before each launch; `pgrep -af 'torchrun|mpirun'` empty and
`rocm-smi --showpids` showing only `gpuagent` (pid 44579) throughout. No foreign
GPU process appeared; no preemption was necessary.
