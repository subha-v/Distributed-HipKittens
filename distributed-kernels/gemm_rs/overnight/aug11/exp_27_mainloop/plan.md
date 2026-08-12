# exp_27 — the GEMM mainloop schedule: cycle account, candidate ranking, one recommendation

**Type: research + design. No kernel code written, no GPU job run, no lease
taken.** The only node work is a `hipcc -c --save-temps` compile, which is a
host-side CPU compile, run inside `dhk-gemmrs` with `-o` pointed at
`aug11/exp_27_mainloop/isa/`. `harness/build/` was not touched, no `.so` was
produced, no other experiment's tree was written.

## Why this experiment exists

exp_20's refreshed attribution plus its g4 counter pass sized the target:

| shape | total µs | GEMM pool µs | MFMA busy µs | **non-MFMA mainloop** |
|---|---:|---:|---:|---:|
| 6 — 8192×8192×29568 | 1632.5 | 992.2 (60.8%) | 421.2 | **571.0 µs (35% of the whole op)** |
| 5 — 8192×4096×14336 | 641.9 | 305.2 (47.6%) | 101.7 | **203.5 µs** |

Shapes 5 and 6 carry the entire remaining graded gap to rank-1 (per-shape
graded ratios 1.286× and 1.208× against a 1.098× geomean). Nothing else in the
kernel is this large. The question this experiment answers is not "is it big"
— exp_20 settled that — but **"is it recoverable, by what, and for how
much."**

## Scope and ownership

Writes only under `overnight/aug11/exp_27_mainloop/`. Reads:
`gemm_rs_mi300x.cpp`, `gemm_rs_mi300x_hk_adapter.cuh`,
`gemm_rs_mi300x_constants.cuh`, `include/cdna3/types/shared/st.cuh`
(read-only shared HipKittens), `gemm_rs_device_tile.cpp` (read-only gfx950
donor), `HANDOFF.md`, `aug11/LESSONS.md`, `exp_20_attribution/result.md`,
`experiments/exp_03_mainloop/result.md`.

## Method, and the measured / counted / inferred discipline

Three classes of number appear in `design.md` and every one is labelled:

- **MEASURED** — comes from a GPU run someone else already did. Only two
  sources are used: exp_20's ablation pools (`ablation.json`) and its g4
  counter cells (`counters.json`, `SQ_VALU_MFMA_BUSY_CYCLES` and
  `SQ_INSTS_MFMA`).
- **COUNTED** — read off the ISA this experiment compiled. `isa_census.sh`
  builds `gemm_rs_mi300x.cpp` with `--save-temps` and `kloop_hist.py` finds the
  innermost `v_mfma`-bearing natural loop per instantiation by back-edge
  analysis and histograms it. Back edges are used rather than the assembler's
  `in Loop: Header=` comments because exp_03 §"tooling defects" showed those
  comments under-report k-loop membership after loop rotation.
- **INFERRED** — a cycle model applied on top of the two above. Every rate
  used (MFMA 16 cyc, wave64 VALU 4 cyc, LDS 128 B/clk, L1 64 B/clk) is stated
  with its assumption at the point of use, and the conclusions are checked for
  sensitivity to the two rates that are not measured.

## Pre-registrations, written before the analysis was finished

1. The k-loop of `<256,256,32,*>` issues 64 `v_mfma_f32_16x16x16_bf16`, 24
   `ds_read_b64`, 8 `ds_write_b64`, 4 `global_load_dwordx4` and 1 `s_barrier`
   per trip. — **CONFIRMED exactly** (design.md §2).
2. No single pipe is saturated; the 2× gap between the measured trip and the
   MFMA floor is serialization, not throughput. — **CONFIRMED** (§3).
3. Single-buffered `BK=64` reduces the barrier count per tile. — **REFUTED.**
   It is exactly a wash, on arithmetic (§4). This is the load-bearing negative
   of the experiment and it closes directions 1 and 2 of the dispatch.
4. `<256,256,32,true>` (shape 6) costs more per trip than
   `<256,256,32,false>` (shape 5) because of the K-tail mask. — **CONFIRMED,
   and larger than expected: +63 instructions per trip (+32%) and a 4-block
   loop against a 1-block loop.**

## Deliverables

| file | contents |
|---|---|
| `design.md` | the cycle account, the closure arithmetic, the ranked candidate table, the recommendation, the gate plan and the expected-value statement |
| `kloop_census.json` (node) | raw per-block instruction census, all 7 instantiations, written by `kloop_hist.py` |
| `kloop_census_summary.json` | plot-ready: counted mix + derived pipe occupancy + the per-trip cycle budget for shapes 5 and 6. Schema in `design.md` §8 |
| `isa_census.sh`, `run_census.sh`, `kloop_hist.py` | the (CPU-only, re-runnable) instrument |
| `census.log`, `isa/compile.log` (node) | raw run |

## What would invalidate this experiment

- A different `gemm_rs_mi300x.cpp` than the one censused. The compile recorded
  `sha256[0:16] = 5f9f47258842c64d` for the kernel and `e9b3d1c17d49ee0b` for
  the adapter, at 2026-08-12T11:06:49Z. Another agent landing a mainloop change
  after that timestamp invalidates §2 and everything derived from it; re-run
  `run_census.sh` (≈100 s, CPU only) before acting on the ranking.
- exp_20's shape-6 `full` moving outside its freshness gate, which would move
  the 4063 cycles/trip calibration.
- A measured LDS rate other than 128 B/clk/CU on gfx942, which would move the
  LDS occupancy line of §3 by 2× (the conclusion survives either value — §3.1).
