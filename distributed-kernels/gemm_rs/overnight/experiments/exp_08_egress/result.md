# exp_08 — E2, XGMI egress. WIN: geomean 255.98 → 230.84 µs (−9.8%)

**Verdict up front. There is no write amplification and there never was: the
fabric carries 117.48 MB against 117.44 MB of useful payload, a ratio of
1.0003, with 99.9% of transactions at the full 64 B. Egress is also not bunched
at the release — two thirds of it already leaves L2 by ordinary capacity
eviction spread across the tile loop. Both pre-registered hypotheses are dead.
The actual defect was that a rank was using only 2.02 of its 8 egress links at
any instant, because the inherited `WGM = 4` tile order makes all concurrently
resident CTAs write to the same 2–3 peers. Fixing the tile order moved shape 6
from 2537.44 to 1828.89 µs (−27.9%) and shape 5 from 748.90 to 644.39 µs
(−14.0%) while moving exactly zero bytes.**

Denominator throughout: our harness, current validated best, per-shape
`76.90 / 104.70 / 91.47 / 201.06 / 748.90 / 2537.44`, geomean **255.98 µs**.

---

## 1. Step 1, the measurement

### 1.1 Method

`rocprofv3` 1.1.0 (`/opt/rocm/bin`, container `dhk-gemmrs`), counter collection
over an 8-rank single-process run of the real kernel
(`experiments/exp_08_egress/prof_driver.py`, 2 warmup + 4 measured launches per
rank; the aggregator keeps only the measured window and filters rows to
`gemm_rs_mi300x_kernel`). Counters were taken in groups of four to stay inside
the TCC hardware slots; `prof_matrix.sh` and `prof_confirm.sh` are the drivers,
raw CSVs under `prof/`.

**Two validity checks were run before any number was used, because both failure
modes are silent and both deflate traffic counters toward zero.**

1. *Does the protocol survive instrumentation?* Counter collection inserts
   barriers around dispatches. If it serialized the eight agents, reducers would
   spin on ready flags no peer had published, the bounded waits would time out,
   the sticky error bit would be set, and **every producer would return early
   without emitting**. Every profiled run reports `correct=1 tight=1
   errors=none`, and `max|diff|` is the same `1.221e-04` / `4.883e-04` as an
   unprofiled run. Clean.
2. *Do the counter labels mean what I claim?* The `emit_local` ablation arm
   writes the identical bytes through the identical instructions to the local
   rank's own slot. Its off-die write requests **must** collapse to ~0 if
   `TCC_EA0_WRREQ − TCC_EA0_WRREQ_DRAM` really is fabric-bound traffic. They
   collapse from 1,836,800 to 1,792 (0.1%), and the residual 1,792 is exactly
   the `publish_band_epoch` flag traffic. The label is validated, not assumed.

### 1.2 Counters used, with their verbatim `rocprofv3 --list-avail` definitions

| counter | definition (verbatim) |
|---|---|
| `TCC_EA0_WRREQ` | "Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Atomics may travel over the same interface and are generally classified as write requests. This does not include probe commands." |
| `TCC_EA0_WRREQ_64B` | "Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface." |
| `TCC_EA0_WRREQ_DRAM` | "Number of TCC/EA write requests (either 32-byte of 64-byte) destined for DRAM (MC)." |
| `TCC_EA0_WRREQ_STALL` | "Number of cycles a write request was stalled." |
| `TCC_EA0_WRREQ_{GMI,IO,DRAM}_CREDIT_STALL` | "Number of cycles a EA write request was stalled because the interface was out of {GMI,IO,DRAM} credits." |
| `TCC_EA0_WRREQ_LEVEL` | "The sum of the number of EA write requests in flight. This is primarily meant for measure average EA write latency. Average write latency = TCC_PERF_SEL_EA_WRREQ_LEVEL/TCC_PERF_SEL_EA_WRREQ." |
| `TCC_WRITEBACK` | "Number of lines written back to main memory. This includes writebacks of dirty lines and uncached write/atomic requests." |
| `TCC_ALL_TC_OP_WB_WRITEBACK` | "Number of writebacks due to all TC_OP writeback requests." — i.e. the `buffer_wbl2` of the release |
| `TCC_NORMAL_WRITEBACK` | "Number of writebacks due to requests that are not writeback requests." — i.e. capacity eviction |
| `TCC_{UC,NC,CC}_REQ` | "The number of {uncached, noncoherently cached, coherently cached} requests. This is measured at the tag block." |
| `TCC_EA0_WR_UNCACHED_32B` | "Number of 32-byte write/atomic going over the TC_EA_wrreq interface due to uncached traffic… A 64-byte request will be counted as 2" |
| `TCC_STREAMING_REQ` | "Number of streaming requests. This is measured at the tag block." |
| `TCC_TAG_STALL` | "Number of cycles the normal request pipeline in the tag was stalled for any reason." |

Derived: `bytes_to_EA = 64·WRREQ_64B + 32·(WRREQ − WRREQ_64B)`;
`fabric requests = WRREQ − WRREQ_DRAM`. All figures are **per rank per launch**
(the sum over 8 agents × 4 dispatches, divided by 32). `TCC_*` values are sums
over the 16 TCC instances × 8 XCC.

### 1.3 Q1 — fabric bytes vs useful bytes: NO AMPLIFICATION

Shape 6, `8192×8192×29568`, production arm:

| quantity | measured | analytic | ratio |
|---|---:|---:|---:|
| EA write requests | 2,361,648 | — | — |
| of which 64 B | 2,359,296 (**99.9%**) | — | — |
| of which 32 B | 2,352 (0.1%) | — | — |
| fabric (non-DRAM) requests | 1,836,800 | 1,835,008 payload + 1,792 flags | 1.0000 |
| **fabric bytes** | **117.48 MB** | **117.44 MB** (7/8 · 8192·8192·2) | **1.0003** |
| DRAM write requests | 524,848 | 524,288 (local slot 16.78 + `out` 16.78 MB) | 1.001 |

Shape 5, `8192×4096×14336`: fabric 58.74 MB against an analytic 58.72 MB, again
99.9% at 64 B. Consistency checks that make this hard to misread:
`TCC_WRITEBACK` = 1,182,120 lines × 2 = 2,364,240 ≈ `WRREQ`, so essentially
every EA write is a 128 B line writeback split into two 64 B transactions; and
`TCC_NC_REQ` = 1,836,800 exactly equals the fabric request count while
`TCC_UC_REQ` = 0, so peer payload is **noncoherently cached in the local L2**,
not uncached.

Also: `TCC_EA0_RDREQ_DRAM` = 1,052,034 against `TCC_EA0_RDREQ` = 1,051,990 in
the no-mainloop arm — **100% of reads are DRAM-destined, zero remote reads.**
There is no read-for-ownership over the fabric; full-line writes are being
detected.

**Consequences.** The "scattered 16 B packets waste 50–75% of every fabric
transaction" premise is falsified, and with it the highest-ranked intervention
in `experiments/e2_research.md` (I1, LDS-stage the packets to make stores
wave-contiguous) is worth **exactly zero** — reading `emit_band_packets` shows
why: with `cols = BN = 256` there are 32 packets per row and `tid` strides by 1,
so lanes 0–31 of every wave already write one contiguous 512 B run and lanes
32–63 the next row's, all 512 B-aligned. The addresses were already right. I2
(adding `sc0 sc1` to payload stores) and I3 (`nt`) would both risk *breaking* a
path that is already at 100% transaction efficiency. `TCC_STREAMING_REQ` = 0
confirms nothing is non-temporal today, and it should stay that way.

### 1.4 Q2 — bunched at the release, or spread? MOSTLY SPREAD

Shape 6, lines written back per rank per launch:

| origin | shipped `WGM=4` | share |
|---|---:|---:|
| `TCC_ALL_TC_OP_WB_WRITEBACK` (`buffer_wbl2`, the release) | 380,094 | 32.2% |
| `TCC_NORMAL_WRITEBACK` (capacity eviction, during emit/mainloop) | 796,035 | 67.3% |
| `TCC_WRITEBACK` (total) | 1,182,120 | 100% |

So **two thirds of the egress already streams out during the tile loop** and
only a third waits for the release. That matches the ablation independently: the
release cut is 219.3 µs of the 1149.9 µs egress pool, ≈19%. "Start peer traffic
earlier and spread it" was therefore not the lever either — it is largely
already happening.

### 1.5 Q3 — is the fabric saturated? The credit counters cannot answer, and one of them is unusable

| counter, shape 6 | shipped `WGM=4` |
|---|---:|
| `TCC_EA0_WRREQ_STALL` | 81,159,385 cycles |
| `TCC_EA0_WRREQ_GMI_CREDIT_STALL` | **0.8** |
| `TCC_EA0_WRREQ_IO_CREDIT_STALL` | **0.0** |
| `TCC_EA0_WRREQ_DRAM_CREDIT_STALL` | **0.0** |
| `TCC_TOO_MANY_EA_WRREQS_STALL` | 0.0 |
| `TCC_CYCLE` | 504,835,763 |

**Report this as a tooling finding: on this ASIC the three
`TCC_EA0_WRREQ_*_CREDIT_STALL` sub-counters do not resolve
`TCC_EA0_WRREQ_STALL`.** The total is 81 M cycles (16% of `TCC_CYCLE`, and 264×
the `emit_local` arm's 305 K, so it is unambiguously a property of *remote*
writes) while all three attribution counters read ~0. Anyone who reads
`GMI_CREDIT_STALL ≈ 0` as "xGMI is not backpressured" will draw the wrong
conclusion — I nearly did. The usable saturation signals here are
`TCC_EA0_WRREQ_LEVEL / TCC_EA0_WRREQ` (average EA write latency, per the
counter's own documented use) and `TCC_EA0_WRREQ_STALL`.

---

## 2. What the deficit actually was

The measurement says the bytes and the transactions are perfect and the timing
is 3× off. That leaves *when and where* the bytes go. `destmap.py` answers it
with arithmetic alone — no GPU, no measurement uncertainty — by replaying the
shipped tile map (`gemm_rs_mi300x.cpp:236-244`) and the destination rule
(`row0 = tm·BM + b·EB`, `dest = row0 / (M/8)`), and asserting that every tile is
covered exactly once.

A producer CTA's whole tile lands on **one** peer. A round of the tile loop is
the `num_gemm_ctas = 272` tiles `{pid + i·272}`, all resident at once. So **the
set of `tm` values in a round is the set of xGMI links that rank is using at
that instant.** With `WGM = 4`, `in_group = 4·num_pid_n = 128` tiles on shape 6
— *smaller than the round* — so a round spans about two M-groups:

| shape 6 round | destinations reached | busiest link's share |
|---|---|---:|
| 0 | 3 (`{0:128, 1:128, 2:16}`) | 47.1% |
| 1 | 3 (`{2:112, 3:128, 4:32}`) | 47.1% |
| 2 | 3 (`{4:96, 5:128, 6:48}`) | 47.1% |
| 3 | 2 (`{6:80, 7:128}`) | 61.5% |

Effective concurrent links, `1/worst-share` round-weighted: **2.02 of 8.** Five
of the seven links sit idle while one carries half the round.

**The arithmetic closes with no free parameters, on two shapes:**

| shape | fabric bytes | egress pool (ablation) | achieved | effective links × 47 GB/s | achieved / predicted |
|---|---:|---:|---:|---:|---:|
| 6 | 117.44 MB | 1149.9 µs | 102 GB/s | 2.02 × 47 = 95 GB/s | **1.07** |
| 5 | 58.74 MB | 295.0 µs | 199 GB/s | 4.02 × 47 = 189 GB/s | **1.05** |

We were within 7% of the ceiling *of the links we were using*. That also
explains the 16% `WRREQ_STALL` (those links genuinely were backpressured) and
why the per-shape deficit against the reference tracks link count and nothing
else:

| shape | effective links | vs reference (graded, same-run) |
|---|---:|---|
| 1 | 8.00 | win |
| 2 | 4.13 (egress only 20.8 µs) | win |
| 3 | 8.00 | win |
| 4 | 8.00 | tie |
| 5 | **4.02** | lose 1.25× |
| 6 | **2.02** | lose 1.74× |

The two shapes we lose are exactly the two with degraded egress-link
concurrency. Nothing else in the ledger predicts that split.

---

## 3. Step 2, the intervention

One expression in `gemm_rs_mi300x.cpp`, in the producer role's tile-order
constant. **No store, fence, barrier, wait, cache-policy bit, release site,
publication site or signal index is touched, and the GEMM mainloop — including
the load-bearing `acquire_frags` anchor — is untouched.** The decode expression
itself is byte-identical; only the constant feeding it moved.

```c++
#if HK_GEMM_RS_MI300X_WGM4
        const int WGM = 4;
#else
        const int WGM = (tiles <= g.num_gemm_ctas) ? 4 : num_pid_m;
#endif
```

`WGM = num_pid_m` makes `in_group = tiles`, so `group` and `first` collapse to
0 and `gsize` to `num_pid_m`: the decode becomes column-major, `tm` varies
fastest, and every round covers all 8 destinations (**7.53** effective links on
shapes 5 and 6, 8.00 on shape 2).

**Why the `tiles <= num_gemm_ctas` guard, and why it is derived rather than
tuned.** A first arm applied column-major unconditionally. It gated clean and
won 255.98 → 239.54 µs, but it *regressed shapes 3 (+5.2%) and 4 (+17.5%)* —
precisely the shapes my own pre-registered analysis had called free controls
because they already run at 8.00 of 8 links and their egress therefore cannot
improve. When `tiles ≤ num_gemm_ctas` every tile is resident simultaneously, so
the order cannot change egress concurrency **at all** and only its operand
locality is left; there `WGM = 4` is better. The mechanism is countable: with
CTAs assigned to XCDs as `pid mod 8`, on `4096×4096×4096` column-major gives
each XCD 2 distinct A tiles and 16 distinct B tiles (18 operand tiles) where
`WGM = 4` gives it 4 and 8 (12). The guard separates the six scored shapes
exactly as measured — 1/3/4 keep `WGM=4`, 2/5/6 go column-major — and it also
covers the generic correctness row, where `WGM=4` is *worse* than on any scored
shape (`8192×8192×28672`: `in_group = 512`, a 280-CTA round spans `tm ∈ [0,8)`,
i.e. **one** destination).

Interventions considered and **not** taken, with reasons:
- **LDS staging for coalescing (I1).** Dead: amplification is 1.0003 and the
  stores are already 512 B-contiguous per half-wave.
- **`sc0 sc1` on payload stores (I2).** Dead and dangerous: the payload is
  already cached-NC and leaving as 100% 64 B transactions; forcing system scope
  can only split that.
- **`nt` on payload stores (I3).** Not needed for egress and unproven for the
  release; kept in reserve.
- **Rotating the destination sweep by rank (I4).** Subsumed. Rotation balances
  *ingress* at destinations; it does not raise the number of links a rank
  egresses on concurrently, which was the whole defect. With the tile order
  fixed every rank already uses all 7 links every round, so rotation has nothing
  left to buy. Not built, not spent.

Protocol invariants were re-read in
`experiments/exp_05_release_granularity/protocol_review.md` §1–§2 before the
first GPU run; the argument is in `protocol_note.md`. Summary: the change is a
permutation of the tile→CTA assignment, coverage stays a bijection (asserted
mechanically in `destmap.py` for all six shapes under both orders), the release
still dominates every publication in its own iteration, `(tm,tn) ↦ (dest,lrow,
tn)` stays injective so intra-CTA slot WAW remains impossible, and the §2.4
wait-for graph is untouched because its edges are between epochs, not tiles. The
one real change is **cross-peer arrival order**, and §2.3 settles it: a reducer
waits only on the eight `ready` cells of its own output tile and never reads
another tile's slots, and the release is destination-agnostic. No path depends
on which peer lands first.

---

## 4. Gates

`tools/gate_ladder.sh exp_08_egress`, clocks pinned at 1900 MHz, node verified
clean (0 KFD pids) before the run and again before timing. Logs in `logs/`.

| gate | verdict |
|---|---|
| M0 node clean | PASS, 0 KFD pids |
| M1 build | PASS, all modules built, production exports only `gemm_rs_mi300x` |
| M2 resources / ISA | PASS, 6/6 distinct instantiations. **Unchanged from the previous best**: `246`/`248` VGPR on the two `256/256/32` rows, `AGPR 0`, `SGPR 106`, **scratch 0, spills 0** everywhere. The ternary costs nothing. |
| M3 correctness, 17 shapes, 1e-2 **and** 2e-3 | PASS 17/17 at both. Worst `max|diff|` `1.221e-04`; scored set `4.883e-04` — **identical to the previous best**, as predicted, since arithmetic is untouched |
| M4 negative controls | PASS, all three fail exactly as designed |
| M5 600-epoch skewed soak | PASS, epochs and signals exact at every checkpoint, worst `max|diff|` `4.883e-04` |
| M7 timing | below |

## 5. Numbers

3 rotations × 50 pipelined iterations, order-rotated, every timed run verified.

| # | shape | best before | exp_08 | ratio | tile order |
|---|---|---:|---:|---:|---|
| 1 | 64×7168×18432 | 76.90 | 77.88 | 1.013 | `WGM=4` (unchanged) |
| 2 | 512×4096×12288 | 104.70 | **91.18** | **0.871** | column-major |
| 3 | 2048×2880×2880 | 91.47 | 90.25 | 0.987 | `WGM=4` (unchanged) |
| 4 | 4096×4096×4096 | 201.06 | 200.33 | 0.996 | `WGM=4` (unchanged) |
| 5 | 8192×4096×14336 | 748.90 | **644.39** | **0.860** | column-major |
| 6 | 8192×8192×29568 | 2537.44 | **1828.89** | **0.721** | column-major |
| | **geomean** | **255.98** | **230.84** | **0.902** | |

Device-max geomean 228.06 µs. Stdevs `0.05 / 0.29 / 0.37 / 0.73 / 10.76 /
18.97` µs, i.e. ≤1.7% on the two large shapes.

Honest caveats:
- **Shape 1 is +1.3% (0.98 µs) and that is larger than its 0.05 µs stdev.** Its
  order is unchanged, so the only candidates are the `const`-vs-`constexpr`
  `WGM` and the added compare/select — or drift in the ~66 µs shape-independent
  floor the ledger records for this shape. It is 1 µs against 708 µs won on
  shape 6, so it is recorded rather than chased.
- The unconditional-column-major arm (geomean 239.54 µs) also passed the full
  ladder and is kept in the record as the intermediate; the guarded rule
  supersedes it on every shape.

## 6. Mechanism confirmation on the winning binary

The claim is that the win came from link concurrency and nothing else. That is
falsifiable: the change must move **zero** bytes. Same counters, same driver,
winning binary vs a `-DHK_GEMM_RS_MI300X_WGM4=1` rebuild of the same source
(`prof_confirm.sh`), shape 6, per rank per launch:

| counter | `WGM=4` | column-major | Δ |
|---|---:|---:|---|
| `TCC_EA0_WRREQ` | 2,361,648 | 2,361,648 | **identical** |
| `TCC_EA0_WRREQ_64B` | 2,359,296 | 2,359,296 | **identical** |
| `TCC_EA0_WRREQ_DRAM` | 524,848 | 524,848 | **identical** |
| fabric bytes | 117.48 MB | 117.48 MB | **identical** |
| `TCC_WRITEBACK` | 1,182,120 | 1,182,120 | **identical** |
| **avg EA write latency** (`LEVEL/WRREQ`) | **4,705 cyc** | **2,887 cyc** | **−38.6%** |
| `TCC_EA0_WRREQ_STALL` | 81,159,385 | 54,220,335 | −33.2% |
| `TCC_TAG_STALL` | 64,029,595 | 44,963,870 | −29.8% |
| `TCC_EA0_RDREQ` | 10,051,402 | 7,063,020 | −29.7% |
| `TCC_ALL_TC_OP_WB_WRITEBACK` | 380,094 (32%) | 620,987 (53%) | +63% |
| `TCC_NORMAL_WRITEBACK` | 796,035 (67%) | 556,928 (47%) | −30% |

Exactly the same bytes in exactly the same transaction sizes, with average EA
write latency down 38.6% — the direct queueing signature of spreading a fixed
number of requests across 7 links instead of 2. Two unadvertised side effects,
both reported because they matter for what comes next:

1. **`TCC_EA0_RDREQ` fell 29.7%**, so column-major also improved *operand*
   locality on shape 6. Part of the 708 µs is a read-side win, not egress. The
   shape-6 attribution must be re-measured before the next axis is ranked.
2. **The bunched share of egress rose from 32% to 53%.** With the fabric no
   longer backed up, payload lines now survive in L2 until the release pushes
   them. **E3 (release granularity) just got bigger relative to this binary, not
   smaller** — the opposite of what the old attribution implied.

## 7. Reproduction

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_08_egress
bash $E/destmap.sh                     # CPU only: per-round egress link usage
bash $E/prof_matrix.sh                 # step 1, the measurement
bash $E/gate.sh                        # pin clocks + full gate ladder
bash $E/prof_confirm.sh                # step 3, mechanism confirmation
bash $E/cleanup.sh                     # remove the scratch .so files

# the WGM=4 control arm, if a paired A/B is ever wanted again:
#   hipcc ... -DHK_GEMM_RS_MI300X_WGM4=1 -DTK_MODNAME=gemm_rs_wgm4 \
#     $ON/../gemm_rs_mi300x.cpp -o $ON/harness/build/gemm_rs_wgm4.so
# (full command line is in prof_confirm.sh)
```

## 8. What is now open

- **Re-run `exp_ablation.py` on this winner before ranking anything.** Its
  cached `gemm_rs_abl_*.so` are built from the pre-exp_08 source and must be
  deleted first. Both the egress pool and the mainloop pool moved.
- **7.53 of 8 links is not 8.00.** The residual is the ±4-CTA imbalance from
  `272 mod 32 = 16`; worth ~6% of the egress pool at most.
- **E3 is now the largest identified lever on shape 6** (53% of egress bytes
  wait for `buffer_wbl2`, and the release cut was already 219 µs *before* this
  change concentrated more traffic there).
- The three `TCC_EA0_WRREQ_*_CREDIT_STALL` counters are unusable on this ASIC —
  do not build an argument on them.
