# exp_05 stage 0 — measured phase attribution, and the interference finding

**The instrument works, and it found something the whole-kernel timings could
not see: the service pool inflates M7 by 57% through pure memory-system
interference, on top of the 25% it costs by taking CTAs away.** That is a
mechanism cost of CTA role specialization which the mode-0 control — the
control we had been trusting all night — is structurally blind to.

## Getting the instrument working

The kernel already wrote a timestamp block and the harness already allocated it
and never read it. Three things had to be fixed:

1. **`realtime_now()` was missing its wait.** `moe_mps_adapter.cuh:127` issued
   `s_memrealtime` with no `s_waitcnt lgkmcnt(0)`. `s_memrealtime` is an SMEM
   instruction whose destination SGPR pair is not valid until that wait, so
   callers read whatever the register happened to hold. The symptom was
   diagnostic: `LAST_READY` and `M7_DONE` returned real absolute clocks while
   `DRAIN`, `REDUCE_DONE`, `M2_DONE` and `M6_DONE` all returned exactly **1** —
   and `LAST_READY` and `DRAIN` sit in the *same function* under the *identical*
   guard, which ruled out enablement, null pointers and layout. The sites that
   worked happened to follow LDS traffic, which carries its own `lgkmcnt` wait.
   Fixed with the wait and the correct `"=s"` (scalar pair) constraint. **This
   was a pre-existing defect** — two of the four dead cells were never touched
   by this work, so every prior `K0P6_MPS_TS_*` reading was unreliable.
2. **A `KSTART` stamp at kernel entry is provably dead.** M0 zeroes the entire
   state block including all 8 timestamp cells
   (`k0pf6gm_device_tile_mps.hip:593-596`). Removed. The upside of that reset:
   the block is **per-epoch, not cumulative**, so after a 600-epoch soak the
   max-stamps all come from the final epoch and are mutually consistent.
3. **Units.** The device wall clock is **100 MHz, so 1 tick = 0.01 µs**
   (`hipDeviceAttributeWallClockRate = 100000 kHz`). The shader clock is 2.2 GHz
   — using it would have been a 22x error.

Added `K0P6_MPS_TS_M2_DONE` and `K0P6_MPS_TS_M6_DONE`, plus an additive,
print-only, exception-guarded host dump (mirrored into
`MPS_OVERNIGHT_HARNESS_NOTE.md`). **Resource-free**: SGPR 104 / VGPR 256 /
AGPR 256 / LDS 155,428 B, unchanged.

**Validation.** The independently measured combine phase for the near-parity
configuration is **1,287.9 µs** against the inherited exp_35 figure of
**1,309 µs** — 1.6% apart. The instrument agrees with the number it was never
told about.

## The measurements (single epoch, 1 tick = 0.01 µs)

| configuration | compute CTAs | service CTAs | plan+M6 | M7 | combine | M2→end | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| C=2, mode 0 | 254 | 0 | 2,976.5 | **1,609.7** | 1,287.9 | 5,874.0 | 7,126.5 |
| C=64, mode 0 | 192 | 0 | 2,969.6 | **2,019.7** | 1,481.6 | 6,470.9 | 7,386.4 |
| C=64, g=1, mode 2 | 192 | 64 | 3,048.2 | **3,179.0** | 2,874.9 | 9,102.1 | 10,313.9 |
| C=32, g=1, mode 2 | 224 | 32 | 3,017.9 | 2,479.1 | 8,226.5 | 13,723.5 | 14,687.2 |
| C=0, mode 1 | 256 | 0 | 2,975.0 | 1,579.8 | 2,782.1 | 7,336.9 | 8,028.0 |

**Sanity check that validates the whole table: `plan+M6` is invariant at
2,970–3,048 µs across every configuration.** The reservation happens at M6.9,
so nothing before it should move, and nothing does. Any systematic error in the
instrument would have shown up here first.

## Finding 1 — the service pool's interference cost, isolated

`C=64, mode 0` and `C=64, g=1, mode 2` run M7 on **exactly the same 192 compute
CTAs**. The only difference is whether 64 service CTAs are concurrently moving
combine payloads. So the two effects separate cleanly:

| effect | M7 | delta |
|---|---:|---:|
| baseline, 254 compute CTAs | 1,609.7 µs | — |
| **capacity**: 254 → 192 CTAs, no service traffic | 2,019.7 µs | **+410 µs (+25.5%)** |
| **interference**: same 192 CTAs, service pool now running | 3,179.0 µs | **+1,159 µs (+57.4%)** |

**The interference costs 2.8x more than the capacity tax it hides behind.**

This does not contradict the A7 strike — it completes it. A7 was struck because
occupancy is one block per CU, so a service CTA is never co-resident with an
MFMA CTA and cannot steal issue slots. That is correct and remains correct.
**But CTAs that never share a SIMD still share the L2, the Infinity Cache and
the fabric**, and this measures that channel at +57% on the phase we most wanted
to protect. The right statement is: *CTA role specialization removes issue-slot
contention and leaves memory-system contention untouched.*

It also explains a puzzle from exp_03. Mode 0 measured the reservation as cheap
(+6.1% at C=64) and we read that as "the split is nearly free". It is nearly
free **only when the service pool is idle**. The mode-0 control cannot see the
dominant cost of the mechanism it is the control for.

## Finding 2 — the dispatch is exposed, and is at least as large as assumed

`dispatch = total − (M2→end)` gives 1,252 / 916 / 691 / 1,212 / 964 µs across
the five well-behaved rows: **roughly 0.7–1.25 ms, or 10–17% of the kernel**.
That brackets and if anything exceeds CLAUDE.md's inferred 757 µs, and it sits
behind four grid barriers with zero overlap (`exp_05/design.md`).

The spread is a known artefact and worth stating plainly: the phase deltas come
from a single epoch while `total` is a p50 over timed iterations, so the
subtraction mixes two distributions. The `C=8, g=2` row even yields a small
negative and is excluded. **The phase deltas themselves are all single-epoch and
internally consistent; only the derived dispatch figure carries this
uncertainty.** Tightening it needs a `KSTART` stamp taken *after* M0's reset
barrier — a five-line change now that the mechanism is understood.

## Finding 3 — where mode 2's time actually goes

At `C=64, g=1`: the service drain runs **5,952 µs past M6**, while M7 + combine
together occupy 6,054 µs. The service pool is not "overlapped work" — it *is*
the post-M6 critical path, matching it almost exactly. The first tile event is
observed 192 µs *before* the last CTA finishes M6 (M6_DONE is a max over CTAs,
so early CTAs are already producing), which confirms the queue starts filling
promptly and the lag is not in readiness — it is in drain rate.

Mode 1 is the clean counterpoint: the bulk push runs after M7, so **M7 is
untouched (1,579.8 µs, the same as the 256-CTA baseline)** and the entire cost
lands in the combine phase (1,287.9 → 2,782.1 µs). Vertical (mode 1) and
horizontal (mode 2) fusion move the same bytes and pay in different phases —
and mode 1's total (8,028) beats every mode 2 point except C=64.

## What this changes

1. **Mode 0 is retired as the control for role specialization.** It measures the
   capacity tax and nothing else. Any future role-split experiment needs a
   matched-CTA-count comparison like the one above.
2. **The 0.88x ceiling from exp_03 was optimistic.** It assumed the only cost of
   reserving `C` CTAs was mode 0's tax. Adding +57% interference on M7 pushes
   the achievable floor higher still. The combine boundary is closed more firmly
   than exp_03 could establish.
3. **The dispatch boundary remains the right target** — the prize is measured at
   0.7–1.25 ms and fully exposed — **but it inherits the interference finding**.
   A dispatch service pool will slow M6 the same way this one slows M7. Stage 1
   (local-tokens-first ordering with no role split, hence no interference) is
   now clearly the better first move than stage 2.

## Primitives

`realtime_now()` is a library-adjacent device helper whose contract was silently
violated for every caller. The lesson is narrow but sharp: **an inline-asm
primitive that returns a value from an asynchronous unit must carry its own
wait**, or every call site inherits an obligation the signature does not
express. `ts_first`/`ts_last` looked correct at every call site and were wrong
at four of six.

## Artifacts

`~/overnight-scratch/ts2_*.log`, `~/overnight-scratch/screen_ts2.csv`,
JIT hashes `a589f3b2476f` / `9131e831968d`, commit `cc0f9001`.
