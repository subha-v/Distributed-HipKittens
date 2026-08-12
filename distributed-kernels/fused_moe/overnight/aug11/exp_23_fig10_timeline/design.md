# exp_23 design — per-CTA phase event ring + timeline plotter

## Event ring (additive descriptor slot, MPS ABI append-only)

| field | value |
|---|---|
| slot | next free MPS descriptor slot (append after 62; bump `K0P6_MPS_D_LEN`) |
| shape | `u64[256][24]` = 48 KB, agent visibility, zeroed at M0 |
| entry | `(phase_id << 56) | (s_memrealtime() & 0x00FFFFFFFFFFFFFF)` |
| writer | tid 0 of each CTA, plain relaxed agent store, one per boundary crossed |
| reader | host, after launch completes (no in-flight reads) |

`s_memrealtime` is the 100 MHz constant-rate wall counter, coherent across
CUs → 10 ns ticks; 56 bits ≈ 22 years, no wrap handling needed.

## Phase-ID enum (order = write order; absence = phase not run by that CTA)

```
 0 M0_START      6 M5_BARRIER      12 SVC_STRIPE_DONE (per event stripe, svc only)
 1 M1_START      7 M6_START        13 SVC_FLUSH       (per flush group, svc only)
 2 M2_START      8 M7_START        14 M8_START
 3 M2_PASSB      9 M7_TASK_DRAIN   15 M9_START
 4 M3_BARRIER   10 M75_START       16 EPOCH_END
 5 M4_PLAN      11 SVC_START (service role entry)
```

Compute CTAs write ≤ 17 events; service CTAs add one per stripe drain + one
per flush group (bounded by ring depth 24 — at C=64/g=1/flush_rows=1 the
stripe count per CTA is small; assert on overflow by dropping with a count in
entry 23, never by wrapping).

Overhead: ≤ 24 uncontended stores per CTA per epoch ≈ noise. The A13
timestamps mechanism already proved the flag-gating pattern; this widens the
buffer, not the mechanism.

## Bytes-per-phase table (host side, analytic — no in-kernel counting)

| phase | xGMI bytes | HBM bytes |
|---|---|---|
| M1 dispatch | `Σ_pe≠cur n_s·ROW_STRIDE` (from pushed_count) | LDS-staged reads of hidden |
| M2 unpack | 0 | live_rows · (ROW_STRIDE + unpack writes) |
| M6/M7 | 0 | A/W streams (known per-tile) |
| service pushes | rows·14,336 to peers | equal read side locally |
| M7.5 flags | world·live_rows·4 | — |
| M8 combine | pull arms only: fanout·14,336/row | slot reads + out writes |

All quantities are already device-published (`pushed_count`, `nvi`,
`row_remaining` popcounts), so the host derives exact per-arm byte totals —
these also feed the integral check.

## Binning

10 µs bins over [min ts, max ts] of rank 0. Per bin:

- `mfma[b]` = |{CTAs whose interval [M6_START..M7_TASK_DRAIN] covers b}| / 256
  (M6/M7 split into two series if per-phase coloring is wanted).
- `hbm[b]`, `xgmi[b]` = Σ over phase intervals covering b of
  (phase bytes ÷ phase duration). Service per-stripe events give the xGMI
  strip intra-phase texture instead of one flat rectangle.

## B0 arm (no kernel work)

torch.profiler (or rocprofv3) trace of one eager MoE layer on the vLLM/B0
stack OR the harness's production arm; kernel-name → resource-class map
(`rccl*`/`mori*` → xGMI, `*gemm*`/`*mfma*` → MFMA, scatter/quant/combine →
HBM) checked in as `b0_kernel_map.json`. Strips are binary occupancy bars —
exactly NanoFlow's non-overlap panel.

## Plot spec (plot_timeline.py)

3×3 grid: rows = MFMA % / HBM GB/s / xGMI GB/s, columns = B0 / pf6gm / mps.
Shared x (µs from epoch start), per-row shared y. Phase boundaries as faint
verticals on megakernel columns. Annotations: B0 comm-share % (width of its
xGMI blocks), MPS M7 elongation vs pf6gm (the interference term).

## Alternatives rejected

1. **rocprof PC-sampling / counter timelines for the megakernel arms** — no
   per-phase attribution inside one 7 ms launch; counters are the
   cross-check, not the source.
2. **In-kernel byte counters** — adds atomics to the paths being pictured;
   analytic bytes are exact and free.
3. **LDS-staged event buffering** — LDS is fully committed (155,428 B);
   direct global stores are rare enough not to matter.

## Primitives

Uses only existing spellings (relaxed agent stores, `s_memrealtime` via the
A13 pattern). Watch item for result.md: whether a `phase_mark(ring, id)`
helper belongs in the library or stays operator-local (it encodes no
ordering, so current lean: operator-local).
