# exp_27 addendum — is the mainloop memory-path bound? No. The arithmetic, and where exp_21 and exp_20 diverge

**Verdict: exp_21's panel-a plateau does NOT transfer to the production kernel,
and the conflict is resolvable tonight from data that already exists.** exp_20's
counter pass measured the production kernel's L2 miss traffic directly, and on
shape 6 the operand stream below L2 runs at **440 GB/s over the GEMM pool —
9.6% of the 4593 GB/s memory-path plateau.** No memory level in the production
kernel is above ~13% of its delivery rate. The ~571 µs is **exposed latency,
not bandwidth wait**, and §4 of `design.md` (the barrier-count closure) stands.

The user's core point is nonetheless correct and it *strengthens* the closure:
`bytes/FLOP = 1/BM + 1/BN` is independent of `BK`, so single-buffered `BK=64`
buys no traffic either. **It is now dead twice over — no barrier win, no
traffic win.** That was the dispatch's headline candidate and it should be
struck.

---

## 1. Per-shape bound derivation, arithmetic shown

Both shapes run `<256,256,32>`, so both have

```
bytes/FLOP = (BM+BN)·BK·2 / (2·BM·BN·BK) = (BM+BN)/(BM·BN) = 1/256 + 1/256
           = 7.8125e-3 B/FLOP          <- BK cancels, exactly as claimed
```

which matches exp_21's 7.813e-3 to the digit. Now the four rates, per shape.

| | shape 5 (8192×4096×14336) | shape 6 (8192×8192×29568) |
|---|---:|---:|
| `K_local`, `k_iters`, tiles, `NG`, waves | 1792, 56, 512, 272, 2 | 3696, 116, 1024, 256, 4 |
| FLOP per rank `2·M·N·K_local` | 120.3 GFLOP | 496.1 GFLOP |
| GEMM pool (MEASURED, exp_20) | 305.2 µs | 992.2 µs |
| **achieved** | **394.1 TFLOPS** | **500.0 TFLOPS** |
| producer-CU peak @1900 MHz | 1058 (272 CU) | 996 (256 CU) |
| fraction of producer peak | **37.2%** | **50.2%** |
| **① L1-side operand demand** = TFLOPS × 7.8125e-3 | **3.079 TB/s** | **3.906 TB/s** |
| per-CU: ① / CU / 1.9 GHz, vs 64 B/clk vL1D | 5.96 B/clk = **9.3%** | 8.03 B/clk = **12.5%** |
| ① vs ~31 TB/s aggregate L2 read | **9.9%** | **12.6%** |
| **② below-L2 reads**, `ea_read_requests`×64 B (MEASURED) | 1,963,124 → **125.6 MB** | 6,826,695 → **436.9 MB** |
| implied **L2 read hit ratio** = 1 − ②/① ·pool | **86.6%** | **88.8%** |
| ② over the GEMM pool | **411.6 GB/s** | **440.3 GB/s** |
| **② vs the 4593 GB/s measured plateau** | **9.0%** | **9.6%** |
| ② at the *MFMA roofline* (same hit ratio) | 1105 GB/s = 24% | 871 GB/s = **19%** |

Two notes on ②, both of which make the case stronger rather than weaker.
`ea_read_requests` counts *every* L2 read miss, including the reducers' 8-source
pulls (~134 MB on shape 6, ~67 MB on shape 5) — so the operand-only miss stream
is materially smaller than the number tabulated. And the compulsory floor is
A+B read once: 121.1 MB (shape 6), 44.1 MB (shape 5), so measured ② is only
3.6× / 2.8× compulsory. The `WGM = num_pid_m` column-major order is already
delivering ~8.9× per-XCD L2 reuse.

**Regime, per shape: both are latency/schedule-bound.** Neither is within 8× of
the memory path, within 4× of L1, or within 4× of L2. Shape 5 is *further* from
every bandwidth limit than shape 6 while being *further below* MFMA peak
(37.2% vs 50.2%) — the exact opposite of what a bandwidth bound predicts.

## 2. Where exp_21 and exp_20 actually diverge

They do not measure the same quantity.

- **exp_21 panel a** runs on `a_mat`/`b_mat` = 15.2 MB, "16 distinct 256×256
  tiles, **deliberately cache-resident**", with the stated design intent
  "panel a must price MFMA issue, **not HBM**" (`exp_21/plan.md:129`,
  `design.md:87`). Its 4549 GB/s is a **CU-side request rate on a
  cache-resident working set**.
- **exp_21 panel b** runs on a 512 MB working set chosen to exceed the 256 MB
  Infinity Cache. Its 4593 GB/s is a genuine **HBM rate**.

Equating them requires panel a's requests to miss to HBM, which its design
explicitly prevents. **The 99.0% agreement is a coincidence between two
different levels of the hierarchy, not a closure.** Two independent checks that
it must be:

1. 4549 GB/s over 304 CUs at 1.9 GHz is **7.9 B/clk/CU against a 64 B/clk
   vL1D** — 12%. A cache-resident stream cannot be capped at 12% of the return
   rate of the cache serving it.
2. `HANDOFF.md`'s own traffic control: identical `waves·BM·BN` and identical
   `waves·k_iters`, **1.78× the traffic → 6.2% slower**. A bandwidth-bound body
   at 1.78× traffic is ~78% slower. 6.2% is what a ~10%-utilised path with a
   modest queueing term costs. That control has been read as evidence *for* a
   traffic bound; it is evidence *against* one.

What panel a most likely measured is the same thing exp_20 measured: **the
mainloop body tops out near 45–50% of MFMA peak for schedule reasons**, and
582.4 TFLOPS (45% of 1307.4 at 2.1 GHz) versus our 500.0 TFLOPS on 256
producer CUs (50.2% of 996 at 1.9 GHz) are the *same* ceiling seen twice.

### The measurement that settles it — and it is cheap

**Primary, no new code:** one `rocprofv3` pass at the production config on
shapes 5 and 6 with `SQ_WAIT_INST_LDS`, `SQ_WAIT_ANY`, `SQ_BUSY_CYCLES`,
`SQ_INSTS_VALU`, `SQ_LDS_IDX_ACTIVE`, `SQ_LDS_BANK_CONFLICT`. This attributes
the 2015 idle cycles/trip to LDS-wait vs VMEM-wait vs barrier directly. Plumbing
exists in `exp_20_attribution/run_counters.sh`. ~20 min.

**Pre-registration:** LDS-wait + barrier > 60% of `SQ_WAIT_ANY`, VMEM-wait
< 25%. **Falsifier: if VMEM-wait exceeds 50%, I am wrong, exp_21 is right, the
mainloop is a traffic problem, and every row below C should be abandoned.**

**Secondary, on exp_21's own harness:** re-run panel a at the *same* tile with a
working set enlarged past the 256 MB MALL. Bandwidth-bound predicts the plateau
falls; schedule-bound predicts 582 TFLOPS again. Do **not** test this by
changing `BM`/`BN` — that changes MFMA-work-per-barrier too and confounds the
two hypotheses.

## 3. Re-ranked candidates

The §5 table of `design.md` survives; these are the amendments.

| # | mechanism | shape 6 µs | shape 5 µs | change vs design.md §5 |
|---|---|---:|---:|---|
| **A** | counter pass adjudicating §2 | 0 | 0 | **promoted** — it now settles a live two-instrument conflict, not just a residual |
| **B** | hoist `load_commit` above half-1 MFMAs | −37…−73 | −11…−23 | unchanged; **recommended** |
| **C** | prefill swizzled LDS offsets + SGPR bases | −25…−60 | −8…−18 | unchanged |
| **D** | retire `K_TAIL=true` on shape 6 | −40…−90 | 0 | unchanged |
| **E** | phase-offset warp-group ping-pong | −150…−350 | −45…−105 | unchanged; still the only >15% ceiling |
| **I** | **XCD-aware tile order** (new; untested, `MI300X_DESIGN.md` §2) | **−0…−15** | −0…−5 | **priced and found small.** Perfect L2 locality would take ② from 437 MB to the 121 MB compulsory floor — 316 MB, i.e. 319 GB/s of a plateau we use 9.6% of. It cannot pay while the body is schedule-bound. Keep it for **shape 4**, whose pool is 42.4% egress |
| **H** | `v_mfma_f32_32x32x8` | −20…−60 | −6…−18 | **re-examined under the traffic hypothesis and still low.** Warp-level LDS reads are `(WM+WN)·BK·2` and global bytes are `(BM+BN)·BK·2` **regardless of atom shape**; the atom changes operand reuse per *instruction*, not per *warp tile*. Both atoms are 256 MAC/cyc/SIMD, so it is an issue-slot lever only |
| **X** | `S=1, BK=64` / rolling half-`BK` | **0** | **0** | **closed twice**: no barrier win (design.md §4) and no traffic win (`bytes/FLOP` is `BK`-independent) |

**On release granularity.** exp_20 sized it at 65.4 µs on shape 5 and exp_21
priced the protocol at 0.440× of egress bandwidth; with fabric amplification at
1.0007× the whole egress residue is granularity, and it is the best cheap win
on the board. **It is already owned by exp_26**, which has arms built, an M9
re-golding path and a paired campaign running, and `STATUS.md` prices it at
~32 µs on shape 5 ≈ **0.8% geomean**. I am not recommending it because
recommending it would duplicate a live experiment, not because it is wrong —
if exp_26 lands it, that is the night's cheapest µs.

## 4. Recommendation, revised

**Run A, then land B.** Unchanged in substance from `design.md` §6, with A
promoted from prerequisite to adjudicator.

- **Pre-registered:** B moves shape 6's GEMM pool −5.0% ± 3.0% (992.2 → 942 µs),
  shape 6's wall −3.1%, shape 5's −2.3%, **geomean −0.91%** (209.56 → 207.66 µs).
- **Falsifier:** |Δ| < 1.5% on shape 6 over ≥3 paired draws (half reversed) with
  the ISA confirming `vmcnt(0)` moved between the two 32-MFMA runs ⇒ the whole
  B/C/D family is worth < 3%; close it and go to E.
- **Second falsifier, from A:** VMEM-wait > 50% of `SQ_WAIT_ANY` ⇒ this entire
  document's regime call is wrong; abandon the mainloop for traffic work.

## 5. Is the ~571 µs recoverable? Partly, and by less than its size

**It is not bandwidth, so it is not unrecoverable for bandwidth reasons** — but
571 µs is the distance to a perfect-overlap roofline, and we already run at
50.2% of producer-CU peak where tuned MI300X bf16 GEMMs reach 60–75%. A
defensible target is **~250 µs on shape 6 and ~90 µs on shape 5**, which is
**−5.15% at the geomean** (209.56 → 198.77 µs) and takes the graded gap from
1.098× to ~1.046×. The recommended arm alone is **−0.91%**. Shapes 1–4 are
host-bound or inside their noise floors and contribute nothing, so a −15% win
on the two shapes that matter is a −5% score. That dilution is arithmetic, and
it should be stated before anyone spends a night expecting otherwise.
