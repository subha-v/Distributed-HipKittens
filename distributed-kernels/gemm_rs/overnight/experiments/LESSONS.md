# GEMM-RS overnight lessons ledger (append-only; supersede, never delete)

## Session 2 — 2026-08-11 overnight (optimization session)

- **WIN, exp_13 / E4 re-opened: `NUM_REDUCER_CTAS` was NOT settled. Two rows
  moved — shape 1 32 → 56 (−15.4%) and shape 6 32 → 48 (−4.9%) — for a paired
  geomean of −3.6%.** Graded, same run against frozen rank-1: ours 353.59 →
  **348.64 µs** against an anchor that reproduced to 0.6% (306.60 vs
  306.23–308.80), so the ratio goes **1.145–1.158× → 1.137×**. Full ladder
  passed: M3 17/17 at both tolerances, M4 3/3, M5 600 epochs, M2 resource tuples
  unchanged (the split is a host plan constant, no device code changes). Full
  detail in `exp_13_cta_split/result.md`.

  **The split does not buy CTAs, it buys ROUNDS, and the function is a plateau
  with cliffs.** Producer critical path is `ceil(gemm_tiles / (304 − NR))` whole
  tile waves, reducer path is `ceil(red_tiles / NR)`; both are flat over wide
  ranges of `NR` and then step. So the optimum is always **the largest `NR` that
  does not add a producer wave**, and the whole axis is predictable from the tile
  counts before running anything. The pre-registered table in
  `exp_13_cta_split/plan.md` called every cliff correctly: `NR=56` costs one
  extra producer wave on shapes 2/4/5/6 and measured +23.3 / +45.3 / +11.5 /
  +9.6%.
  **Corollary that kills the obvious hypothesis:** lowering `NR` to give the GEMM
  more CTAs cannot work here. 1024 shape-6 tiles over 272 *or* 288 producers is
  4 waves either way, so `NR=16` buys **zero** GEMM and doubles reduce rounds —
  measured **+3.6%**, against a naive prediction of −3.5% from linear scaling.
  Never reason about a persistent strided loop as if its cost were proportional
  to CTA count.

  **Both moved rows were created by earlier landed wins, which is the general
  lesson: a landed optimization can silently invalidate a settled constant.**
  exp_04a's retile of shape 1 (`32/256/32 → 32/64/64`) took `red_tiles` 28 → 112
  and so took its reduce side from 1 round to 4 at `NR=32` — it quadrupled the
  reducer critical path and nobody re-swept the split. exp_08 cutting the egress
  pool 1149.9 → 412.4 µs is what made shape 6's fourth reduce round visible.
  **`(BN, NR)` is a joint optimum** (`red_tiles = lrow · ceil(N/BN)`) and neither
  sweep knew about the other; that is the next experiment.
  Also revised: **the "~66 µs shape-1 floor that no stage cut explains" was
  partly reduce rounds.** Shape 1 now runs 63.2 µs *with* the mainloop, below the
  66.0 µs that removing the entire mainloop left at `NR=32`.

- **TRAP, and the most important measurement finding of the session: the
  pipelined harness's bias is PER-ALLOCATION, not positional, and rotating arm
  order does not remove it. On shape 6 it is 4.28%.** Two arms with an identical
  split, identical shared operand tensors and identical code, interleaved under
  *complete* positional rotation (`passes = len(arms)`, so every arm occupies
  every slot exactly once), separated by **4.28% on shape 6 and 2.41% on shape
  4** — while the measured position residual never exceeded **±0.47%** on any
  shape in any run. The only thing that differs between the twins is which
  `hipMalloc` returned their 134 MB payload heap. The previously recorded 3.9%
  null-arm figure was therefore attributed to the wrong cause, and order
  rotation — the defence it motivated — is close to useless against it.
  Two consequences. (1) **Every arm-identity comparison on this node must carry
  a duplicate arm**; several past sub-5% verdicts were taken with an instrument
  whose floor was never measured on that shape. (2) The way to beat it is to
  **pool over independent allocations** and look for *disjoint ranges* rather
  than percentages: shapes 1 and 6 each had three candidate allocations that
  were all faster than all six `NR=32` allocations (`p = 1/C(9,3) = 1.2%`), which
  is why they landed, while shapes 4 and 5 showed 2–3% "wins" in single runs that
  sat entirely inside the same-configuration range and were correctly rejected.

- **TRAP: the recorded per-shape denominator vector is BEST-of-arm, while gate M7
  prints MEANS.** `77.47 / 88.28 / 91.79 / 199.43 / 613.70 / 1716.76` (geomean
  225.62) are bests; exp_05's own table records shape 5 as `613.70 / 645.93` =
  best / median. Comparing an M7 mean geomean to it manufactured a phantom "+7.3%
  regression on shape 5" for a row whose configuration and device code had not
  changed — shape 5 measured 644.71–669.39 across seven independent allocations
  today, i.e. exactly its recorded *median*. **State the statistic with the
  vector**, and prefer the same-run paired comparison: best-vs-best across
  sessions is fragile for any shape with a fat lower tail, which shape 5 has.

- **NEGATIVE, by arithmetic and with zero GPU time: the dual-role /
  work-stealing reducer CTA cannot pay on any scored shape.** Reducer CTAs are
  idle >90% of a shape-6 kernel, which makes stealing GEMM tiles sound
  compelling. But the producer critical path is quantized in whole tile waves,
  and going from the landed producer count to all 304 CTAs removes a wave on
  **none** of the six shapes (1/2/1/1/2/4 waves either way) — the stolen work
  would land in slack that is already free. It only pays where tiles-per-CTA is
  large, i.e. the unscored generic row (118 tiles per CTA). Closed before writing
  code, and before spending a protocol review on an epoch-lifetime argument for a
  CTA that changes role mid-kernel.

- **BLOCKER B IS DEAD: rank-1 RUNS ON THIS NODE, and we are 1.78× behind it.**
  Correct on all six graded shapes at both `1e-2` and `2e-3`, all 8 ranks, and
  measured **same-run interleaved against our kernel** (one pool, same inputs,
  arm order reversed per rep, 240 samples/arm/shape, evaluator `full` protocol).
  Geomean best: ours **543.61** vs rank-1 **305.22 µs** → **1.781×**; geomean
  mean 586.52 vs 322.98 → 1.816×. **rank-1 wins every shape.** Per-shape
  ours/rank-1 on best: `1.033 / 1.382 / 1.266 / 3.090 / 1.304 / 4.387`.
  Frozen hash `7940fcb8…f0dc5` verified; the frozen file was never written.
  Full record in `exp_10_rank1/result.md`.
  *The blocker was never the heap bootstrap.* **`CREATE_SHEMEM_CODE` is dead
  code** — it appears exactly once, at its own definition, and the submission
  contains no `subprocess`/`Popen`/`multiprocessing`/`fork`/`exec` at all. So
  **`heap_bases_*.pkl` is unreachable and its count of 0 was never a failure
  signal**; three sessions gated on a probe that could not fire. Retire
  `tools/check_rank1d.sh`. Repair #5 (the `hipIpcMemHandle_t` alias) patches a
  symbol referenced only inside that dead string and is **moot**. The real
  bootstrap artifact, `ipc_handles_rank*.bin`, had been appearing all along.
  *The actual blocker, and the fix is one env var with zero source edits:*
  **`AMDGCN_USE_BUFFER_OPS=0`.** rank-1 reaches peer heaps by passing a
  100-element bf16 tensor as the kernel base pointer plus a large negative
  element offset (−6.6e8…−4.4e9). Triton 3.6.0's AMD backend gates `buffer_*`
  lowering per argument on `HIPBackend.is_within_2gb(arg)` (`compiler.py:184`);
  a 200-byte tensor qualifies, so the epilogue became `buffer_store_dwordx2`
  whose voffset is **32-bit**, silently narrowing an offset that is correct
  `i64` in the TTIR. Proof: all four fault addresses equal
  `hack + 2*int32(base_addr[i])` + a tile offset inside that shape's 896 KiB
  region, landing in 32 GiB `---p` reserved VA — hence "write to a read-only
  page". Ruled out first with evidence: all 8 heaps map `rw-s` 1 GiB, host-side
  peer writes all succeed, and a stage bisect showed `init_shmem` **and** the C++
  `dist_barrier` peer atomics pass on all 8 ranks. Disclosed bias: the knob is
  global to rank-1's Triton kernels so its A/B loads lose buffer ops too, which
  can only **penalize rank-1** — the bias runs against the winner, so the verdict
  is safe. Our arm is hand-written HIP, so the knob cannot touch it.
  *Two traps for anyone touching the graded protocol.* (1) **`eval.py` gives every
  graded shape a bias.** Its parser does `int(val)` and keeps the string on
  failure, so `has_bias: False` becomes truthy `"False"`. rank-1's author guessed
  it (`#####?????? bench all have bias?` on `__conf`), and its cached path does
  `bias.data_ptr()` unconditionally, so with a genuine `None` it raises on the
  *second* call. Both arms were therefore run with a bias on all six shapes.
  (2) **`eval.py` reports nanoseconds.** (3) Also verified all six shapes are in
  rank-1's `__conf`, so none of this is `origin()`, its silent torch fallback.
  *Where the gap is:* concentrated in the large shapes, not the mainloop — shape
  6 is 4.39× and shape 4 is 3.09×, while shape 1 is nearly a tie at 1.03×. Our
  relative sd is 16–18% on the small shapes against rank-1's 2–6%, and the graded
  score is a mean, so our tail is charged to us. Confirms the standing read that
  the next work belongs in the cold-L2 / per-call egress path.
  *Staging trap that cost previous sessions:* `dhk-eval` and `dhk-gemmrs` are
  **separate containers with separate filesystems**, so staging iris into
  `dhk-eval`'s `/usr/local/lib` never helped. Use `docker exec -u 0 dhk-gemmrs`.
  Also clear `TRITON_CACHE_DIR` when changing the knob or the cached `hsaco`
  keeps `buffer_store` and the fix looks ineffective.

- **WIN, exp_08/E2: the XGMI deficit was never bandwidth efficiency, it was
  egress-link CONCURRENCY, and the fix is the tile order. Geomean 255.98 →
  230.84 µs (−9.8%).** Per-shape `77.88 / 91.18 / 90.25 / 200.33 / 644.39 /
  1828.89`, i.e. `+1.3% / −12.9% / −1.3% / −0.4% / −14.0% / −27.9%`. Full ladder
  passed (17/17 at both 1e-2 and 2e-3, worst `max|diff|` 4.883e-04 on the scored
  set — *identical* to the previous best, as it must be since arithmetic is
  untouched; all three controls; 600-epoch soak; M2 resources unchanged at
  246/248 VGPR, zero scratch, zero spills). Full record in
  `exp_08_egress/result.md`.
  *The mechanism.* A producer CTA's whole tile lands on **one** peer
  (`dest = (tm·BM)/(M/8)`), so the set of `tm` values across the 272
  concurrently-resident CTAs **is** the set of xGMI links the rank is using at
  that instant. The donors' `WGM = 4` makes `in_group = 4·num_pid_n = 128` tiles
  on shape 6 — *smaller than a 272-CTA round* — so a round spans ~2 M-groups and
  only **2–3 of the 8 destinations**, with one link carrying 47–62% of the round.
  `WGM = num_pid_m` collapses the decode to column-major (`group`/`first` → 0,
  `gsize` → `num_pid_m`), putting all 8 destinations in every round.
  *The arithmetic closes with zero free parameters, on two shapes:* shape 6
  moved 117.44 MB in 1149.9 µs = 102 GB/s against `2.02 effective links ×
  47 GB/s = 95`; shape 5, 58.74 MB in 295.0 µs = 199 GB/s against
  `4.02 × 47 = 189`. **We were within 7% of the ceiling of the links we were
  using.** It also predicts the shape-dependence of the whole deficit: the only
  two shapes we lose to the reference are the only two with degraded link
  concurrency (2.02 and 4.02); every shape we win runs at 8.00.

- **exp_08: BOTH pre-registered E2 hypotheses are FALSIFIED by measurement, and
  the top-ranked intervention in `e2_research.md` was worth exactly zero.**
  rocprofv3 TCC/EA counters, shape 6, per rank per launch: fabric bytes
  **117.48 MB against 117.44 MB of useful payload = 1.0003×**, with **99.9% of
  EA write transactions at the full 64 B** and `TCC_NC_REQ` exactly equal to the
  fabric request count (`TCC_UC_REQ` = 0), i.e. peer payload is cached-NC and
  coalesced into whole-line writebacks. Zero remote reads
  (`RDREQ_DRAM` == `RDREQ`), so no read-for-ownership over the fabric either.
  **There is no write amplification, so I1 (LDS-stage the packets for
  wave-contiguity) buys nothing** — and reading `emit_band_packets` shows why:
  at `cols = BN = 256` there are 32 packets per row and `tid` strides by 1, so
  lanes 0–31 of every wave already write one 512 B-aligned contiguous run. The
  addresses were always right. I2 (`sc0 sc1` on payload) and I3 (`nt`) can only
  *break* a path already at 100% transaction efficiency; `TCC_STREAMING_REQ` = 0
  confirms nothing is non-temporal and it should stay that way.
  Egress was also **not bunched at the release**: 67.3% of lines left L2 via
  `TCC_NORMAL_WRITEBACK` (capacity eviction, spread through the tile loop) vs
  32.2% via `TCC_ALL_TC_OP_WB_WRITEBACK` (`buffer_wbl2`), matching the
  independent 219/1150 µs release/egress ablation split. **Generalisable: three
  plausible mechanism stories, two of them ISA-motivated, all wrong — and one
  profiling matrix plus one page of arithmetic on the tile map settled it. The
  free CPU-only `destmap.py` was worth more than any of them.**

- **exp_08, and the sharpest methodology trap of the session: the three
  `TCC_EA0_WRREQ_*_CREDIT_STALL` counters are UNUSABLE on gfx942 and reading
  them at face value inverts the conclusion.** `TCC_EA0_WRREQ_STALL` is
  81,159,385 cycles per rank per launch on shape 6 (16% of `TCC_CYCLE`, and
  **264×** the `emit_local` arm's 305 K, so unambiguously a property of *remote*
  writes) — while `GMI_CREDIT_STALL` = 0.8, `IO_CREDIT_STALL` = 0.0,
  `DRAM_CREDIT_STALL` = 0.0 and `TCC_TOO_MANY_EA_WRREQS_STALL` = 0. Taken
  literally that says "xGMI is never backpressured, the fabric has spare
  capacity at every instant", which is the exact opposite of the truth: the 2–3
  links in use were saturated. The usable signals are
  **`TCC_EA0_WRREQ_LEVEL / TCC_EA0_WRREQ`** (average EA write latency, per the
  counter's own documented purpose) and `TCC_EA0_WRREQ_STALL`. On the winner the
  latency reads **4,705 → 2,887 cycles (−38.6%)** for byte-identical traffic,
  which is the direct queueing signature of spreading a fixed request count over
  7 links instead of 2.

- **exp_08 method notes worth reusing.** (1) **rocprofv3 counter collection does
  NOT break this 8-rank single-process protocol** — every profiled run returns
  `correct=1 tight=1 errors=none` with unchanged `max|diff|`. That had to be
  checked first: if instrumentation serialized the agents, the bounded ready
  waits would time out, the sticky bit would be set, producers would return
  **without emitting**, and every traffic counter would deflate toward zero and
  look like a triumph. (2) **Validate counter labels against a known answer
  before interpreting them.** The `emit_local` ablation arm writes identical
  bytes through identical instructions to the local slot, so its off-die request
  count *must* collapse: 1,836,800 → 1,792, and the residual 1,792 is exactly
  the flag traffic. (3) `emit_local` is a good *label* control but a poor *byte*
  control — it aims all 8 bands at one slot, so it writes 8× fewer distinct
  lines (89 vs 151 MB out of L2) and therefore **overstates** the XGMI pool.

- **exp_08: an unconditional "improvement" regressed the two shapes my own
  pre-registration had named as controls, and the guard that fixes it is
  derivable rather than tuned.** Column-major everywhere gated clean at
  239.54 µs (−6.4%) but cost shape 3 +5.2% and shape 4 **+17.5%** — precisely
  the shapes `destmap.py` says already run at 8.00 of 8 links and whose egress
  therefore *cannot* improve. The separator is not a fitted table entry:
  **when `tiles <= num_gemm_ctas` every tile is resident at once, so the tile
  order cannot affect egress concurrency at all and only its operand locality is
  left** — and there `WGM = 4` wins, countably: with CTAs on XCDs as
  `pid mod 8`, column-major gives each XCD 2 A-tiles + 16 B-tiles (18 operand
  tiles) on 4096×4096×4096 where `WGM = 4` gives 4 + 8 (12). The one-line guard
  `(tiles <= g.num_gemm_ctas) ? 4 : num_pid_m` reproduces the per-shape best on
  all six scored shapes and also fixes the generic correctness row, where
  `WGM = 4` is worse than on any scored shape (`8192×8192×28672`:
  `in_group = 512`, a 280-CTA round spans `tm ∈ [0,8)`, i.e. **one**
  destination). **Lesson: when a candidate wins the geomean but moves a shape
  your pre-registration called a control, the control is the finding — do not
  bank the geomean and move on.**

- **exp_08 aftermath, two side effects that re-rank the next axis.** (1)
  `TCC_EA0_RDREQ` fell **29.7%** on shape 6, so column-major improved *operand*
  locality too and part of the 708 µs is a read-side win — **re-run
  `exp_ablation.py` on this winner before ranking anything else, and delete the
  cached `gemm_rs_abl_*.so` first, they are built from the pre-exp_08 source.**
  (2) The bunched share of egress rose **32% → 53%**: with the fabric no longer
  backed up, payload lines now survive in L2 until `buffer_wbl2` pushes them.
  **E3 (release granularity) therefore got BIGGER relative to this binary, not
  smaller** — the opposite of what the old attribution implied. Also still open:
  7.53 of 8 links is not 8.00, the residual being the ±4-CTA imbalance from
  `272 mod 32 = 16`, worth ~6% of the egress pool at most.

- **WIN, exp_03/E1(b): the mainloop now overlaps its global loads with its
  MFMAs. Geomean 269.96 → 256.09 µs (−5.14%), confirmed at 255.59 µs on an
  independent repeat.** Per-shape `77.31 / 104.93 / 91.68 / 200.49 / 750.55 /
  2520.29`, i.e. **−10.0% / −7.8% / −5.4% / −0.7% / −2.4% / −4.3%**. Full ladder
  passed (17/17 at both 1e-2 and 2e-3, worst `max|diff|` 4.883e-04 — identical
  to the baseline; all three controls; 600-epoch soak). `kittens::load` was cut
  at its `vmcnt(0)` into `load_issue` (global loads, no wait) and `load_commit`
  (`vmcnt(0)`, ds_writes, `lgkmcnt(0)`) in our own adapter, with the issue
  hoisted above the MFMA block and the commit sunk below it. ISA confirms the
  `s_waitcnt vmcnt(0)` now sits **after** all 64 MFMAs, and all 4
  `global_load_dwordx4` are in flight together where the fused helper drained
  after every 2 — so two exposed round trips per k-iteration became one covered
  one. Full record in `exp_03_mainloop/result.md`. Only full `vmcnt(0)` /
  `lgkmcnt(0)` waits; no counted wait was introduced.

- **TRAP, and the most transferable thing in exp_03: a bare `s_waitcnt
  lgkmcnt(0)` does NOT protect a register filled from LDS, and the failure is
  silent, total, and shape-dependent.** The first build put the wait between the
  fragment `ds_read`s and the MFMAs exactly as designed, and failed M3 on **14
  of 17 shapes** with `max|diff|` 1e19–9e30 and NaN, ~99.9% of elements wrong,
  **no error bit set** — healthy protocol, garbage arithmetic. `s_waitcnt` has
  no operands, so it carries no data dependence; the `ds_read` is an
  `asm volatile` whose output the compiler thinks is live immediately; and
  `v_mfma` is a plain intrinsic with no memory effects, so the machine scheduler
  hoisted MFMAs above the wait. It did so in **exactly** the three
  `K_TAIL=false` instantiations (3 of 4 MFMAs per k-iteration in
  `<32,64,64,false>`, 4 in `<64,64,64,false>`, 1 of 64 in
  `<256,256,32,false>`) and nowhere else. Hazard ⇔ failure, exactly. The
  `K_TAIL=true` instantiations were clean **only because `mask_a_k_tail`
  fragments the block** — pure luck, invisible in the source.
  Fix: `frag_anchor` launders each base tile's register pair through
  `asm volatile("" : "+v"(...))` after the wait, which is a real def, so every
  consumer is ordered by data dependence rather than scheduler goodwill. **Zero
  instructions, zero registers** (233 VGPR before and after). A
  `sched_barrier(0)` would likely also work but only constrains a pass instead
  of establishing a dependence — weaker for the same price.
  Generalisable: **in this tree every memory op is inside `asm volatile`, so
  `SIInsertWaitcnts` is blind and every wait is load-bearing source, not a
  compiler service.** Corollary: `__syncthreads()` emits a bare `s_barrier`
  with no `lgkmcnt`, and a spill of a register whose `global_load` has not
  landed would store stale contents with no `vmcnt` to stop it.
  `exp_03_mainloop/lds_race_check.sh` now checks all of this mechanically for
  both counters (tracking asm-issued `ds_read`/`global_load` destinations since
  the last drain, counted waits handled). **Run it after every mainloop change:
  a 30-second static check that replaces a 3-minute M3 failure.** It also
  confirms the previously-recorded latent last-k-iteration `lgkmcnt` race is now
  repaired.

- **exp_03/P3: splitting the k-step into two `BK/2` halves cleared the spills
  and is timing-neutral; it is a prerequisite, not a win.** 270.12 µs vs 269.96
  (+0.06%). But `<256,256,32,*>` went **256 → 233 VGPR with the 12 B scratch and
  both VGPR spills gone**, and `<128,256,32,true>` 170 → 153. Bit-exact:
  `mma_ABt` already chains its k tiles ascending into the same accumulator
  element, so splitting the k range preserves every addition's order — observed
  `max|diff|` unchanged. Occupancy did **not** move (2 waves/SIMD): residency is
  LDS-bound at 1 CTA/CU, so freed registers buy headroom, never occupancy. That
  headroom is the whole point — E1(b) needs +13/+15 VGPRs and lands at 246/248
  against the 256 cap. **E1(b) on unsplit fragments would have needed ~269 and
  spilled the staging buffer, which is not merely slow but WRONG.** Sequencing
  two arms so the cheap one pays for the expensive one was the right call.

- **NOTE, sizing the next axis: the mainloop is no longer 46% of shape 6.**
  Re-run the attribution on the new winner before picking the next mechanism;
  E2 (XGMI egress) and E3 (release granularity) are now proportionally larger.
  Also **E1(c) is worth less than pre-registered**: the scheduler already
  interleaves the second half's `ds_read`s with the first half's MFMAs for free.
  And **shape 4 (4096×4096×4096) gained nothing (−0.7%)** — with only 16
  k-iterations per tile its cost is not in the mainloop, and it is now the worst
  `× SOL` row at 3.06.

- **TOOLING DEFECT: `tools/gate_ladder.sh:53` asserts the M2 metadata table has
  ≥7 rows, but the dispatch has exactly 6 distinct instantiations, so the
  committed ladder cannot pass on the committed baseline.** 7 was right until
  exp_04 retiled row 1 from `<32,256,32,false>` to `<32,64,64,false>`, which
  collapsed it into the generic even-K row. Stale against `c0a6bdd2`, not
  against any experiment. exp_03 gated both arms with
  `experiments/exp_03_mainloop/ladder.sh`, a faithful copy whose M2 asserts all
  six expected tuples **by name** (strictly stronger than a row count: vacuous
  table, missing instantiation and unexpected extra all fail); nothing the
  benchmark measures was changed. **Fix the constant in `tools/`.**

- **TOOLING DEFECT: the assembler's loop-depth comments are not a reliable
  guide to k-loop membership, and `p0_30_kloop.sh` trusts them.** After P3 the
  prefetch block `.LBB3_86` is annotated `in Loop: Header=BB3_84 Depth=1` —
  outside the k-loop, which would be illegal — and the k-loop inventory
  consequently showed **no `global_load` at all**. Reading the branches settles
  it: the latch does `s_cbranch_scc0 .LBB3_86` / `s_branch .LBB3_87`, i.e. the
  compiler turned the in-body `if (k + 1 < k_iters)` guard into a **choice of
  latch target**, entering the prefetch block on every iteration but the last.
  `MachineLoopInfo` simply picked `BB3_87` as the natural-loop header of a now
  multi-entry region. Use `exp_03_mainloop/cfg_dump.sh` or `where_prefetch.sh`
  when a block appears to have vanished from a loop.

- **TRAP: the previous session left three GPU processes running for seven
  hours, and the obvious `ps | grep` could not see them.** At session start
  `rocm-smi` showed GPUs 2/6/7 at 100% utilization drawing 218–233 W (idle is
  ~140 W) with 5.5 GB VRAM each, from three `python3` processes inside our own
  `dhk-eval` container — the remains of the rank-1 attempt that was killed
  mid-flight. Two compounding reasons this was nearly missed:
  1. `ps -eo cmd | grep -E 'eval\.py|mp_smoke'` matched **nothing**: children
     spawned by `multiprocessing` carry only
     `from multiprocessing.spawn import spawn_main` on their command line.
     **Detect stale GPU work from `rocm-smi --showpids`, never from a hopeful
     cmdline pattern.**
  2. They run as **root inside the container** while we are uid 15523 on the
     host, so a host-side `kill -TERM` fails with `EPERM` and, with stderr
     discarded, prints nothing and looks like success. Signals must be sent
     with `docker exec <container> pkill -TERM -f <pat>` — from inside, where
     the PID namespace also differs, so match by pattern and not by host PID.
  `tools/reap_stale.sh` now does both correctly and refuses to touch any KFD
  PID that is not in one of our two containers. `tools/run_baseline.sh` aborts
  if any KFD PID exists. Mitigating detail, established after the fact: all
  three showed `CU OCCUPANCY 0` — they were CPU spin-waiting, not occupying
  CUs — and the clean-node baseline below reproduces the previous session's
  number to 0.25%, so the earlier measurements appear not to have been
  corrupted. The detection and signalling failures are the durable lesson.

- **`tools/nsh.ps1` was returning exit code 127 on every single run.**
  PowerShell appends a CRLF when piping a string to a native command's stdin,
  so every remote script ended with a stray `\r` line and bash reported
  `line N: $'\r': command not found`. The real script had already completed, so
  the failure was invisible in the output but poisoned every exit code — which
  would have made automated pass/fail detection meaningless all night. Fixed by
  base64-encoding the body and decoding on the far side; exit codes are now
  trustworthy.

- **Ratchet baseline re-established on a verified-clean node:** per-shape means
  µs `107.74 / 115.12 / 96.77 / 203.55 / 765.67 / 2865.78`, **geomean
  285.02 µs** (previous session recorded 285.7 µs — reproduces to 0.25%).
  This is the denominator every experiment tonight is measured against until a
  competitor number exists.

- **The evaluator's benchmark loop is READ, and it settles two open questions
  at once.** Source: `eval.py::_run_distributed_benchmark`, on the node at
  `…/exp026-…/cwd/eval.py:311-412`.
  1. **Call counts are symmetric — the epoch-lockstep worry is closed.**
     `should_stop` is computed on rank 0 only and pushed to everyone with
     `dist.broadcast(stop_tensor, 0)`, so every rank calls `custom_kernel` the
     same number of times. Submission state (epochs, signal cells) may
     therefore be carried across `destroy_process_group` safely. This was the
     one assumption the Track A fix could not verify.
  2. **The graded protocol is per-call latency with barriers, not pipelined
     throughput** — and this changes what is worth optimizing:
     ```
     clear_l2_cache(); torch.cuda.synchronize(); dist.barrier()
     t0 = perf_counter_ns()          # rank 0 only
     output = custom_kernel(...)     # host issue is ON the critical path
     torch.cuda.synchronize(); dist.barrier()
     t1 = perf_counter_ns()
     ```
     Our `m7_bench` reports both columns and the gap is large: shape 1 is
     `pipelined = 107.86` but `single_wall = 235.20` against
     `device_max = 106.25`, and the geomean of `single_wall` is ~476 µs versus
     285.02 pipelined.
     **Do not read that 476 µs as a predicted graded score — it is an upper
     bound and a pessimistic one.** Our harness is single-process/8-device, so
     its `single_wall` and its 62 µs `hostIss` include **all eight launches
     issued serially from one process**. The evaluator runs one process per
     rank, each issuing a single launch concurrently, so the per-rank host
     issue is nearer 62/8 ≈ 8 µs. A defensible estimate for shape 1 under the
     graded protocol is `~8 (host) + 106 (device) + barrier`, i.e. ~120-140 µs,
     not 235 — a graded geomean maybe 1.2-1.3× the pipelined one rather than
     1.7×. The honest statement is that **neither column is the graded number**
     and only an evaluator run settles it (Track A).
     What survives regardless: there is **no pipelining across calls** in the
     graded protocol, so per-call fixed cost — host issue, launch latency, the
     kernel's own prologue/epilogue — is worth exactly as much as device time,
     and it falls hardest on the three small shapes that a geometric mean
     weights most. Host issue is already down 90 → 62 µs from the
     `hipFuncSetAttribute` fix. The pipelined geomean stays the ratchet (per
     `CLAUDE.md`), but every result should record `single_wall` alongside it.
  Note also that benchmark mode passes `recheck=False`, so **it never
  re-verifies correctness** — which is how the previously-recorded
  303/463/2099/3043/21509 µs run happily timed garbage.

- **TRAP (measurement, not yet paid): `run_ours_evaluator.sh` defaults
  `HK_DEBUG=1`.** That path puts a `torch.cuda.synchronize()` and unbuffered
  stderr writes *inside every timed call*. Any evaluator number intended for
  `RESULTS.md` must be taken with `HK_DEBUG=0`. Debug output is for localizing
  the hang, never for a number.

- **WIN, exp_03: the mainloop now overlaps global loads with MFMAs. Geomean
  269.96 → 256.09 µs (−5.14%)**, reproduced at 255.59 µs on an independent
  repeat. Per shape `77.31 / 104.93 / 91.68 / 200.49 / 750.55 / 2520.29`, i.e.
  −10.0% / −7.8% / −5.4% / −0.7% / −2.4% / −4.3%. Cumulative for the night:
  **285.02 → 256.09 µs, −10.2%.**
  Two arms, and the order mattered:
  - **P3 (split the k-step into two BK/2 halves)** was **timing-neutral
    (+0.06%)** but cleared the 12 B scratch and **both VGPR spills** and freed
    **23 registers** on the 256-row configs (256 → 233). Bit-exact, as
    pre-registered, because `mma_ABt` already chains k ascending per
    accumulator element. Occupancy did not move — residency is LDS-bound at
    1 CTA/CU, not VGPR-bound. **Keep it: it is what pays for E1(b)**, which
    lands at 246/248 VGPR against the 256 cap with 8-10 to spare. A neutral
    result that buys headroom for the next change is not a failed experiment.
  - **E1(b) (split `G::load` into issue and commit)** is the win. The ISA
    confirms the reordering survived: `s_waitcnt vmcnt(0)` now sits **after**
    all 64 MFMAs, and all four `global_load_dwordx4` are in flight together
    where the fused helper drained after every two. **Two exposed global round
    trips per k-iteration became one covered one.**

- **RANK-1 RUNS, AND WE ARE 1.238× BEHIND IT — down from 1.784× after the
  `has_bias` fix.** Same-run interleaved, 192 pooled samples per arm per shape,
  all 12 arm-shape pairs correct at **both** `1e-2` and `2e-3`, with
  `bias_forced` and `bias_present` true on all 48 per-rank JSONs.

  | # | shape | ours best/med | rank-1 best/med | ratio (best) |
  |---|---|---|---|---|
  | 1 | 64×7168×18432 | 181.19 / 191.64 | 179.17 / 183.59 | **1.011 tie** |
  | 2 | 512×4096×12288 | 186.69 / 200.43 | 135.80 / 140.36 | 1.375 |
  | 3 | 2048×2880×2880 | 194.31 / 218.46 | 157.61 / 162.76 | 1.233 |
  | 4 | 4096×4096×4096 | 312.94 / 324.73 | 265.69 / 282.21 | 1.178 |
  | 5 | 8192×4096×14336 | 762.76 / 795.68 | 576.60 / 601.75 | 1.323 |
  | 6 | 8192×8192×29568 | 1970.98 / 2006.21 | 1460.07 / 1483.59 | 1.350 |
  | | **geomean** | **381.69** | **308.25** | **1.238** |

  Shapes 4 and 6 came in exactly where the fix predicted (312.94 against ~325;
  1970.98 against ~1970-2000, a 3.28× improvement from 6469.97), which is what
  confirms the fix was live in the measured binary. rank-1 reproduced its
  pre-fix numbers to within ±2% on all six shapes, which is what licenses the
  before/after comparison at all.
  Shape 1 was subject to the same bug but did not move: at m=64 it is
  latency-bound at ~181 µs, so tile geometry is not what sets its time.

- **WIN: E4 re-opened and the boundary DID move — but on only two shapes, and
  for a reason opposite to the one that motivated re-opening.** Landed table:
  shape 1 `NR=32 → 56` (**−15.4%**), shape 6 `NR=32 → 48` (**−4.9%**), shapes
  2–5 keep 32. Paired geomean **−3.6%**. Full ladder passed (M3 17/17 both
  tolerances, M4 3/3, M5 600 epochs, M2 tuples unchanged, `.so` md5-identical
  since this is a host-side constant). Graded against frozen rank-1 in one
  pool: ours **353.59 → 348.64 µs** with the anchor reproducing to 0.6%, so the
  ratio goes **1.145-1.158× → 1.137×**. Shape 1's graded best fell 11.52 µs
  against a pipelined device saving of 11.45 µs — **the two instruments agree
  to within a microsecond**, which is the best cross-validation we have.

- **THE MECHANISM, and it corrects the reasoning that motivated the sweep: the
  split buys ROUNDS, not CTAs.** Both roles are strided persistent loops, so
  what matters is `ceil(tiles / count)`, not the count itself. Shape 6's 1024
  tiles over 272 or 288 producers is **4 waves either way**, so the
  "reduce is only 86.5 µs but holds 32 CTAs for 1717 µs, therefore fewer
  reducers" argument was **backwards**: `NR=16` buys zero GEMM waves and
  doubles the reduce rounds, measured **+3.6%** against a naive prediction of
  −3.5%. The wins are at the **last split before the producer wave count
  steps**, where reduce rounds fall for free.
  Both moved rows were *created by earlier landed wins*: exp_04a's retile took
  shape 1's `red_tiles` from 28 to 112 and quadrupled its reduce rounds, and
  exp_08 cutting egress is what made shape 6's fourth round visible. **Landing a
  win can re-open an axis that was correctly closed before it.**

- **NEGATIVE by arithmetic, not run: the dual-role / work-stealing reducer.**
  Going from the landed producer count to all 304 CTAs removes a producer wave
  on **none** of the six scored shapes, so the stolen work would land in slack
  that is already free. It pays only on the unscored generic row. Killed before
  it cost GPU time, by the same round-counting that explains the sweep.

- **TRAP CORRECTED — the harness bias is PER-ALLOCATION, not positional, and
  rotation does not defend against it.** Position was controlled completely
  (`passes = len(arms)` with a one-step rotation, so every arm sat in every slot
  exactly once) and the positional residual **never exceeded ±0.47%** anywhere.
  Yet two arms with **identical splits, identical shared operand tensors and
  identical code** separated by **4.28% on shape 6**. The only difference is
  which `hipMalloc` returned their 134 MB payload heap.
  Null-arm floors, worst across four runs: `1.34 / 0.56 / 0.61 / 2.41 / 2.17 /
  4.28 %` for shapes 1–6. **Shape 6's floor is 4.28%** — larger than most
  effects we have chased.
  So the earlier "positional bias up to 3.9%" was attributed to the wrong
  cause. **The only defence is pooling over several independent allocations of
  each arm and requiring disjoint ranges.** That is how both landed rows were
  cleared: every `NR=48` allocation beat every `NR=32` allocation on shape 6
  across four runs (`p = 1/C(9,3) = 1.2%`), and likewise for shape 1.
  Shapes 4 and 5 were correctly *rejected* by this: shape 4 showed a 1.9% "win"
  at `NR=40` while its own null twin beat it by 2.1%.

- **Two ledger corrections from this experiment.**
  1. **The recorded denominator vector is best-of-arm, while M7 prints means.**
     That mismatch is the entire explanation of a phantom 7.3% regression on
     shape 5. Compare like with like.
  2. The **"~66 µs shape-1 floor that no stage cut explains"** was **partly
     reduce rounds**. Shape 1 now runs **63.2 µs with the mainloop included**,
     so the floor was never as irreducible as it looked — it was the
     producer/consumer split, visible only after the retile changed
     `red_tiles`.

- **WIN (narrow): E3 release grouping ships as `RELEASE_GROUP = 4` with a
  full-group condition. Shape 6 graded 1955.46 → 1877.9 µs (−4.0%), its gap to
  rank-1 1.338× → 1.280×.** Pipelined geomean 229.33 → 225.62 (0.9838).
  Reproducible to 0.05% across two runs, with rank-1's same-run anchor stable
  to 0.54%.
  **The geomean does not move beyond what the instruments resolve** (−0.70%
  graded; rank-1's own geomean wandered ±1% between runs), because five of six
  scored shapes own too few tiles per CTA to group at all. The win is exactly
  the size the mechanism predicts: shape 6's release is 134.9 µs of 1777.9, so
  eliminating three of four releases is ~5.7% predicted against 6.1% measured.
  All four arms passed the full ladder plus M9. Single-shot latency improved
  alongside throughput on shape 6 (1939.41 → 1877.76), so the protocol review's
  pipelined-throughput risk did not materialize.

- **NEGATIVE, and it is the reason the shipped rule has a condition: grouping a
  PARTIAL group is a net loss.** Unconditional `N ∈ {2,4}` regresses
  512×4096×12288 by **3.6-3.9%** across four independent paired runs against a
  0.60% floor. Mechanism: deferring the first of two tiles' publication starves
  that shape's reducers for half the producer phase, which costs more than
  halving two releases can save. Grouping only where a CTA owns a *full* group
  brings shape 2 back to +0.60%, i.e. exactly the null arm. **Batching a
  producer's output is only free when there is enough of it to batch** — the
  consumer's starvation is the hidden cost.

- **TRAP, and it invalidates casual A/B on this harness: our pipelined A/B has a
  POSITIONAL BIAS of up to 3.9% per shape and ~0.8% on the geomean.** Found by
  running a **null arm** — two builds of `RELEASE_GROUP = 1` under different
  module names, which are behaviourally identical by construction. An earlier
  pass "measured" a 2% regression on 4096×4096×4096, **a shape with one tile per
  CTA that cannot group at all**; that was the instrument, not the change.
  Shape 5 is indeterminate for the same reason (floor 3.6%, and two
  behaviourally identical arms differ by 2.2%).
  **Run a null arm before believing any per-shape delta under ~4%.** This is
  cheap, and it is the only thing that distinguishes a small real effect from
  the harness.

- **`CTRL_PUBLISH_EARLY` works and is now a real gate**: it fails on **6 of 6
  shapes**, each firing a NaN plus a bitwise difference plus a `2e-3` failure.
  Its power is concentrated at **epoch 1** — the window is widest on the first
  launch, before the credit fast path throttles anything — so it is a
  deterministic epoch-1 detector rather than a continuous one. That is
  sufficient: one launch of a batched build with the order inverted is caught
  with certainty. Shapes 1, 3 and 4 are now **gated** bit-identical to the
  pre-E3 build (`torch.equal` on all 8 ranks over 30 poisoned epochs with
  changing inputs) rather than argued to be.

- **Disclosed, not absorbed: shape 3 moves +2.13% (floor 0.24%) and it is code
  generation, not the release.** Whether `rgroup` is a folded constant, a known
  bound, or a runtime value gives the 128×256 instantiation three different
  schedules. Worth knowing that a control shape can move for reasons that have
  nothing to do with the mechanism under test.

- **The ungated E3 diff would have silently reverted exp_08.** `e3_wip_ungated.diff`
  hard-coded `constexpr int WGM = 4` inside its decode helper — which would
  have undone the 9.8% egress-concurrency win *and* made the publish loop decode
  different tiles than the emit loop. The shipped helper takes `wgm` as a
  runtime argument. Two independent bugs from one stale constant; this is the
  third time an inherited `WGM`/table constant has bitten.

- **The ~100 µs per-call cost is DECOMPOSED and the axis is CLOSED. Most of it
  was never ours.** Three extra timestamps inside the graded timed region
  (`perf_counter_ns` is `CLOCK_MONOTONIC`, so the eight ranks' stamps are
  directly comparable, which also yields cross-rank issue skew for free):

  | # | graded total | clone | **host** | device | **barrier** | wait-for-slowest |
  |---|---|---|---|---|---|---|
  | 1 | 247.9 | 21.9 | **38.6** | 82.3 | **86.3** | 18.7 |
  | 4 | 335.2 | 15.7 | **25.5** | 196.7 | **75.8** | 21.6 |
  | 6 | 1927.0 | 13.9 | **21.5** | 1823.9 | **73.9** | −6.2 |

  The largest term is the evaluator's own trailing
  `torch.cuda.synchronize() + dist.barrier()`. A `floor` arm — the identical
  barrier-bracketed structure with **nothing inside the timed region** — costs
  **57-79 µs**. Every submission pays it, rank-1 included. It also wanders
  between 54 and 86 µs with host state, **which is why the small shapes' graded
  numbers are ±10% noisy no matter what the kernel does.** The clone is the
  evaluator's too and scales with input bytes like the memcpy it is.
  **Every device-side candidate is falsified.** A probe kernel matched to our
  launch geometry costs the same at **1, 8, 76, 152, 304 and 608 CTAs**
  (70.1-71.1 µs against a 57.5 µs floor), so a launch costs 12-16 µs and that
  cost is **per-launch, not per-CTA** — which kills the idea of shrinking the
  grid. The 64 KB dynamic-LDS request is free. A kernel running `epoch32`
  verbatim on the per-CTA cell plus the sticky error-bit load in all 304 CTAs
  is indistinguishable from an empty one, so the start-up handshake costs
  nothing (and the epoch/credit protocol never had to be opened). Graded device
  time equals our pipelined `dev_max` within ±3% on all six shapes, so there is
  no drain cost either.

- **WIN: prebound launch path. Graded geomean 381.69 → 357.45 µs against
  rank-1's 306.23 — the gap goes 1.238× → 1.167×, and shape 1 flips to a WIN
  at 0.959×.** The one reducible term was our own host path at 21-39 µs, whose
  largest piece is inside pybind: `pyutils`' `gl<>` converter re-derives
  `__class__.__name__`, `is_contiguous()`, `device.type`, `shape` and
  `data_ptr()` off four duck-typed tensors **on every call**, recomputing
  values that cannot change on a cached state.
  Fix is additive: `configure(...)` binds everything invariant for one
  `(rank, shape)` into a `mi300x_globals` and returns an opaque handle;
  `run(handle, a, b, bias, stream)` copies that POD, writes three pointers and
  the stream, and calls the same dispatch — same POD, same dispatch table,
  **still one launch per call**. Host cost fell to **4.8-7.4 µs** against a
  measured 4.86 µs floor for a bare pybind call plus `hipLaunchKernel`.
  Issue skew collapsed with it (29.2 → 12.9 µs on shape 1), and **because the
  protocol runs at the pace of the last rank to launch, the total gain exceeds
  the host-mean saving.** Two paired same-run A/Bs: 424.1 → 383.0 (0.903×) and
  382.3 → 355.9 (0.931×). Pipelined geomean unchanged at 231.16 µs, exactly as
  predicted — **this axis is invisible in our own harness.**
  Validation note: the ladder drives the *old* entry point, so the new one was
  checked separately and more strictly — the two launch paths are
  **bit-identical** (`torch.equal` True, `max|diff| = 0.000e+00`) on all six
  shapes on all eight ranks.

- **Axis closed, with the residual quantified.** Of the ~103 µs of non-device
  cost per call, **~92 µs is harness machinery every submission pays** and the
  ~11 µs that is ours sits within ~6 µs of the floor for issuing any HIP kernel
  from Python at all. The next microsecond would require removing Python from
  the call entirely, for a ceiling of ~5 µs. Not worth it.
  **The corollary matters more than the win:** because that ~92 µs is a
  constant added to *both* arms, the kernel-to-kernel ratio is **worse** than
  the 1.167× headline. Netted out, shapes 5 and 6 are **1.36× and 1.38× on
  device work alone.**

- **TRAP: `harness/submission.py` is a node-only COPY of `hk_submission.py`,
  and every `mp_*` harness imports `submission`.** Editing `hk_submission.py`
  without recopying measures the old code and looks exactly like a clean null
  result — the most dangerous kind of failure, because a null result is a
  plausible outcome. The build step now resyncs and asserts it.

- **SUPERSEDED (decomposed above): the remaining gap is a ~100 µs per-call
  fixed cost, not the mainloop and not the egress.** Subtracting our own pipelined device time from our graded
  best, per shape: **103 / 98 / 103 / 110 / 111 / 152 µs**. Nearly constant, and
  on shape 1 it is **57% of the entire runtime**. That is also why the gap
  became *uniform* (1.18-1.38×) after the shape-table fix instead of staying
  concentrated in two shapes — a constant additive cost divided into six very
  different totals produces exactly that signature.
  Arithmetic for the payoff: halving it would give roughly
  `130 / 137 / 143 / 258 / 707 / 1895`, a geomean near **309 µs against
  rank-1's 308.25** — i.e. **parity from this one axis alone**, without
  touching the GEMM or the fabric.
  Candidate mechanisms, untested: launching and draining a **304-CTA persistent
  grid** that must be fully co-resident before the protocol can make progress;
  the initial epoch RMW plus the first credit wait; and the fact that the
  graded protocol has **no pipelining**, so a cost our own harness amortizes
  over 50 iterations is paid in full on every graded call. Note this is
  structurally a consequence of the persistent-megakernel/CTA-split design, so
  it belongs to the same research question as E4/E7 rather than being separate
  from it.

- **exp_12: THE ~100 µs IS 70% THE EVALUATOR'S OWN BARRIER, and the ~20 µs of
  it that was ours is now gone. Graded geomean 1.238× → 1.167× behind rank-1;
  shape 1 flips to a WIN (0.959×).** This supersedes the candidate list in the
  entry above: three of its four device-side mechanisms are falsified outright.
  `mp_percall.py` reproduces the graded region in 8 processes with three extra
  timestamps inside it, splitting it into clone / host / device / barrier, and
  `nullk.cpp` launches probe kernels at our exact launch geometry.

  | # | graded total | clone | **host** | device | **barrier** | resid |
  |---|---:|---:|---:|---:|---:|---:|
  | 1 | 247.9 | 21.9 | **38.6** | 82.3 | **86.3** | 18.7 |
  | 4 | 335.2 | 15.7 | **25.5** | 196.7 | **75.8** | 21.6 |
  | 6 | 1927.0 | 13.9 | **21.5** | 1823.9 | **73.9** | −6.2 |

  - **The `floor` arm — the identical structure with NOTHING in the timed region
    — costs 57–79 µs**, essentially all of it `dist.barrier()`. That is the
    evaluator's, it is charged to every submission including rank-1's, and it is
    not reducible by us. It also varies 54–86 µs between runs, which is why the
    small shapes' graded numbers are ±10% noisy no matter what the kernel does.
  - **Grid size is irrelevant: 1, 8, 76, 152, 304 and 608 CTAs all cost the same
    launch** (70.1–71.1 µs on shape 4 against a 57.5 µs floor). A launch costs
    12–16 µs and it is per-launch, not per-CTA. Kills "shrink the grid".
  - **The start-up handshake costs zero.** `epoch304` runs `epoch32` verbatim on
    the per-CTA cell plus the sticky error-bit load in all 304 CTAs and is
    indistinguishable from an empty kernel. 304 atomics at kernel start are not
    a serialization point, so the epoch/credit protocol was never opened.
  - **The 64 KB dynamic-LDS request costs zero**, and **drain costs zero**:
    graded device time equals pipelined `dev_max` within ±3% on all six shapes
    (shapes 5 and 6 measure *below* it).
  - **The whole reducible term was our python/pybind host path**, 21–39 µs, of
    which the largest single piece is `pyutils`' `from_object<GL>` re-deriving
    `__class__.__name__`, `is_contiguous()`, `device.type`, `shape` and
    `data_ptr()` off four duck-typed tensors on every call.
  - **Fix: a prebound launch pair** (`gemm_rs_mi300x_configure` /
    `_run`) that binds everything invariant for one (rank, shape) once and passes
    only the three input pointers and the stream per call, plus a
    `custom_kernel` fast path making the same decision with the redundant work
    removed. Additive; the original binding still serves `harness_lib`.
    **Host 21–39 → 4.8–7.4 µs, against a measured 4.86 µs floor for a bare
    pybind call plus `hipLaunchKernel` — i.e. at the floor.** Issue skew across
    ranks collapsed with it (29.2 → 12.9 µs on shape 1), and since the protocol
    runs at the pace of the last rank to launch, the total gain exceeds the
    host-mean saving.
  - Two paired same-run A/Bs with `_FAST` toggled between arms: geomean best
    424.1 → 383.0 (0.903×) and 382.3 → 355.9 (0.931×). Full ladder green,
    M3 17/17 at both tolerances, and `04_equiv.py` shows the two launch paths
    are **bit-identical** (`torch.equal` True, max|diff| 0.000e+00) on all six
    shapes on all eight ranks. Pipelined geomean unchanged at 231.16 µs, exactly
    as predicted — the single-process harness cannot see this change.
  - **This axis is now CLOSED.** ~92 of the remaining ~103 µs per call is
    harness machinery every submission pays; the ~11 µs that is ours is within
    ~6 µs of the floor for issuing any HIP kernel from python. The corollary:
    because that ~92 µs is a constant added to both arms, the kernel-to-kernel
    ratio is *worse* than 1.167× — netting it out, shapes 5 and 6 are 1.36× and
    1.38× on device work alone. **E2 (XGMI egress on the two large shapes) is
    where the rest of the gap lives.**

- **TRAP: `harness/submission.py` is a COPY of `hk_submission.py` that exists
  only on the node.** The evaluator and every `mp_*` harness import
  `submission`, and various scripts create it with `cp`. Editing
  `hk_submission.py` and re-pushing measures the *old* code and looks exactly
  like a clean null result. Resync and `cmp` it as part of the build step.

- **METHODOLOGY: means are unusable on this node, even with same-run
  interleaving.** Shape 3's mean (327.77) sat far above its median (218.46),
  with outliers synchronized across all 8 ranks at identical sample indices.
  Re-running that shape alone made our tail vanish (mean 221, sd 23%) and the
  same tail appeared on **rank-1's** arm instead (sd 67%, worst 775.87 µs). It
  is per-pool environmental jitter that attaches to an arbitrary arm.
  **Report best and median; treat any mean-based ratio as suspect.** (This also
  supersedes the earlier worry that our arm has a tail the reference does not —
  the tail is not ours, it is the pool's.)

- **TRAP: `push.ps1` resets every source mtime to push time**, so
  "is the `.so` newer than the source?" is **not** a usable freshness check
  after a push — the earlier guidance to use it is wrong in that case. Verify a
  fix is live by its *behaviour* instead (here: querying the resolver directly,
  and checking the measured shapes landed where the fix predicted).

- **E1(c) is a FLAT AXIS — closed. Ships at 0.9969×, which is inside the noise
  floor and should not be quoted as a win.** Alternating paired A/B with both
  states rebuilt in the same session: candidate 230.09 µs vs base 230.80 µs,
  worst within-arm spread **1.27%**. Only two shapes moved for real: shape 2
  **−3.1%** (88.46 / 88.77 / 88.77 / 88.49 against 91.21–91.93, no sample
  overlap) and shape 3 **+0.7%**. **Shapes 5 and 6 — the two this experiment
  targeted, and the only two we lose — never left the noise.**
  The arm is kept because it passed the full ladder and its one unambiguous
  per-shape effect is positive, not because 0.31% is meaningful.

- **THE DURABLE FINDING: scheduling *directives* do not move MFMAs in this TU;
  *data-dependence anchors* do.** This generalizes and should be reached for
  first next time.
  - `sched_group_barrier` was a **literal no-op**: identical block structure,
    identical MFMA run-lengths (`8,4,4,4,4,4,4,32`), identical resource tuples.
    Two independent confirmed reasons — the four `global_load_dwordx4` live in
    their **own basic block**, and a scheduling region never spans one; and
    every memory op here is `asm volatile … : "memory"`, so their mutual order
    is fixed no matter what the scheduler is told. **VMEM issue position is set
    by source placement, full stop.**
  - `sched_barrier(0x7F6)` likewise: with comments and labels stripped the two
    ISAs have **the same 17,613 instructions**, and all 1,156 differing lines
    are the register allocator choosing different physical registers.
  - What *did* work: replacing `if (more)` with a clamped index collapsed the
    k-loop from 6 basic blocks to 1, and an `acc_anchor`
    (`asm volatile("" : "+v"(...))` per accumulator tile — the same trick as the
    load-bearing `frag_anchor`) put **all 64 MFMAs above the commit's
    `vmcnt(0)` on the first try**, where `sched_barrier` had left 33. The
    intended schedule, achieved exactly, and worth nothing measurable.
  - Rejected: peeling the last k-iteration achieves the same de-branching but
    duplicates the MFMA block into one live range — **30 spills, 50 scratch
    stores**, failed M2 and was never timed.
  - `s_setprio` alone on the base is **negative** (1.007×). Its shape-2 win only
    appears on the single-basic-block body, where priority decides which of the
    two waves per SIMD wins the issue port — and row 2 is the highest-occupancy
    row, which is incidental fresh evidence for the E1(a) occupancy story.
  - **Arm C (counted `s_waitcnt`) deliberately NOT run.** It was conditional on
    A or B winning decisively; with the whole axis inside a 1.3% noise floor, a
    mechanism whose documented failure mode is a **30× error that still passes a
    2e-2 gate** is not worth the risk. Judgement, not omission.
  - Known cheap follow-up: shape 3's +0.7% has a diagnosed cause — the clamped
    prefetch adds one redundant tile read per tile, i.e. `1/k_iters` of mainloop
    traffic, and shape 3 has `k_iters = 12` (8.3%) against shape 6's 116 (0.9%).
    Fix is to guard only the commit and add one `wait_vmcnt0()` after the loop.

- **A free noise-floor calibration worth reusing:** two *byte-equivalent*
  instruction streams (the `sched_barrier` pair) reported shape-6 means **1.2%
  apart**. That is a hard lower bound on what any single measurement can
  resolve on this node, obtained at zero cost. **Nothing below ~1.3% is a
  result without paired A/B.**

- **TRAP: CRLF, not build caching, caused four apparent "compile failures" —
  and one of them left the BASE `.so` sitting under the candidate's source.**
  That is the worst possible failure mode: a stale binary silently measured
  against new source, i.e. a number attributed to a change that was never
  compiled. Always confirm the `.so` mtime is later than the source mtime
  before believing a measurement (the shipped state was verified this way:
  `.so` 19:03:51 against source 19:02:46).

- **TRAP: `lds_race_check.sh` throws 10 false positives once loop rotation moves
  a drain to the top of the rotated body**, and it also counts a VALU
  destination as a read. **Do not relax it** — the real hazard class (a bare
  `s_waitcnt` with no data dependence silently corrupting 14 of 17 shapes) is
  still live. Read its hits, do not suppress them.

- **Attribution re-measured on the exp_08 winner, and the ranking FLIPPED BACK
  to the GEMM mainloop.** Fresh table (µs), verified fresh by asserting `full`
  against the independently known total (1777.9 against 1829, within 8%):

  | shape | full | **GEMM** | XGMI | reduce | sync | release |
  |---|---|---|---|---|---|---|
  | 64×7168×18432 | 82.0 | 16.9 | 5.7 | 6.0 | 16.4 | 5.1 |
  | 512×4096×12288 | 91.8 | 25.7 | 13.3 | 3.5 | 17.1 | 13.8 |
  | 2048×2880×2880 | 91.3 | 17.8 | 25.2 | 5.5 | 18.1 | 5.3 |
  | 4096×4096×4096 | 203.1 | 40.8 | **85.3** | 19.2 | 31.3 | 18.4 |
  | 8192×4096×14336 | 656.5 | **309.7** | 195.4 | 51.5 | 86.7 | 46.4 |
  | 8192×8192×29568 | 1777.9 | **1142.8 (64%)** | 412.4 | 86.5 | 191.0 | 134.9 |

  exp_08 cut the shape-6 XGMI pool **1149.9 → 412.4 µs**, a 737 µs reduction
  that accounts for essentially the whole 708 µs win. **GEMM is now 64% of
  shape 6 and 47% of shape 5 — the only two shapes we still lose.** The pool
  ranking has now inverted twice in one session (GEMM → egress → GEMM), which
  is why re-attributing after every accepted win is not bookkeeping.

- **TRAP, paid twice: `exp_ablation.py` SKIPS compilation when an arm `.so`
  already exists**, printing "already built". After a kernel change it then
  silently re-reports the **previous** kernel's attribution. Caught only
  because `full` for shape 6 read 2527 µs when the real kernel was 1829 —
  i.e. by cross-checking against a number obtained another way, not by anything
  in the tool. `tools/reattribute.sh` now force-removes the arm binaries and
  **asserts `full` matches an expected total within a tolerance**, refusing to
  print a table it cannot show is fresh. Generalizable: any cache keyed on
  existence rather than on content will eventually hand you a stale answer that
  looks exactly like a fresh one.

- **WE NOW BEAT THE REFERENCE GEMM+RCCL BASELINE ON THIS NODE.** Same-run
  interleaved, both arms in one process pool under the graded protocol, 50
  iterations × 2 reps, `allclose=True` on all six shapes:

  | # | shape | ours best | ref best | verdict |
  |---|---|---|---|---|
  | 1 | 64×7168×18432 | 218.55 | 228.33 | win |
  | 2 | 512×4096×12288 | 233.04 | 280.99 | **win 1.21×** |
  | 3 | 2048×2880×2880 | 236.24 | 301.21 | **win 1.28×** |
  | 4 | 4096×4096×4096 | 325.61 | 329.81 | win |
  | 5 | 8192×4096×14336 | 788.78 | 698.41 | lose 1.13× |
  | 6 | 8192×8192×29568 | 1972.23 | 1589.40 | lose 1.24× |
  | | **geomean best** | **427.39** | **438.15** | **0.975 win** |
  | | **geomean mean** | **462.86** | **497.53** | **0.930 win** |

  From 1.06× behind to **2.5% ahead on best and 7.0% ahead on mean**, entirely
  from the WGM egress-concurrency fix. The evaluator scores on **mean**, which
  is the more favourable column for us.
  **One robustness caveat that matters under a mean-based score:** our arm has
  occasional large outliers the reference does not — shape 4 best 325.6 /
  median 337.6 but max 3392 (sd 80.4%) against the reference's tight sd 3.4%.
  We still win on mean only because the reference has its own outliers
  elsewhere (shape 3 sd 139.6%). **Killing our tail is now worth more than
  shaving our median.**

- **WIN, exp_08: `WGM` is an xGMI egress-link-concurrency knob, not the
  L2-locality knob it looks like. Geomean 255.98 → 230.84 µs (−9.8%), shape 6
  −27.9%, while moving exactly zero bytes differently.**
  A producer CTA's whole tile lands on **one** peer (`dest = (tm·BM)/(M/8)`), so
  the set of `tm` values live across the resident CTAs **is** the set of egress
  links in use at that instant. The donors' `WGM = 4` makes
  `in_group = 4·num_pid_n` **smaller than a 272-CTA round** (128 vs 272 tiles on
  shape 6), so a round reached only 2–3 of 8 destinations with one link
  carrying up to 62%. **Effective links: 2.02 of 8.** The arithmetic closes with
  no free parameters on both affected shapes — shape 6 achieved 102 GB/s
  against 2.02 × 47 = 95, shape 5 199 GB/s against 4.02 × 47 = 189 — so **we
  were within 7% of the ceiling of the links we were using while five sat
  idle**, and it predicts exactly which two shapes lost to the reference.
  Fix: `WGM = (tiles <= num_gemm_ctas) ? 4 : num_pid_m`. The guard is derived,
  not tuned: when every tile is resident the order cannot affect egress, and
  there `WGM = 4` has better operand locality. An unconditional
  `WGM = num_pid_m` won the geomean outright (239.54 µs) but regressed the two
  shapes the pre-registration had named as controls — treated as a finding
  rather than banked.
  **This is the third donor constant to be wrong** (after `NUM_REDUCER_CTAS` and
  `BM/BN/BK`). Together they were worth ~19%.

- **NEGATIVE, and the profiler is what killed it: there is no write
  amplification in the egress path, and there never was.** On shape 6 the
  fabric carries **117.48 MB against 117.44 MB of useful payload — 1.0003× —
  with 99.9% of EA write transactions at the full 64 B.** `TCC_WRITEBACK × 2 =
  TCC_EA0_WRREQ`, so essentially every fabric write is a whole 128 B line
  writeback; `TCC_NC_REQ` equals the fabric request count with `TCC_UC_REQ = 0`,
  so peer payload is cached and coalesced rather than uncached; reads are 100%
  DRAM-destined, so there is no read-for-ownership over the fabric either.
  Egress was also already **mostly spread, not bunched**: 67.3% of payload lines
  left L2 by ordinary capacity eviction during the tile loop and only 32.2%
  waited for the release, which independently matches the 219/1150 µs ablation
  split.
  Therefore the research report's **top-ranked** intervention — LDS-staging
  packets for wave contiguity — was worth **exactly zero**, and
  `emit_band_packets` shows why: at `BN = 256` there are 32 packets per row and
  `tid` strides by 1, so lanes 0–31 of every wave were already writing one
  512 B-aligned contiguous run. **The addresses were always right.** A
  well-argued, well-cited hypothesis that a single profiler run destroyed —
  measure the mechanism before building the fix.

- **TRAP: `TCC_EA0_WRREQ_STALL` is unusable on this ASIC.** It reads 81 M cycles
  (264× the local-write control) while all three `*_CREDIT_STALL` sub-counters
  read ~0 — which taken literally says xGMI is never backpressured, the exact
  opposite of the truth. Do not build an argument on those three counters.

- **Two consequences of the WGM win for what comes next.** Re-profiling the
  winner shows byte-identical traffic (same `WRREQ`, same 64 B share, same
  117.48 MB) with **average EA write latency down 38.6%** — the queueing
  signature of spreading a fixed request count over seven links instead of two.
  But: **operand reads fell 29.7%**, so part of shape 6's win is read-side and
  **the attribution must be re-run before ranking anything else**; and the
  bunched share of egress rose from 32% to **53%**, so **E3 got bigger relative
  to this binary, not smaller.**

- **The bistable-slow-mode hypothesis is FALSIFIED for our arm, and the raw
  per-iteration series settles the whole evaluator puzzle.** Scanning every
  captured sample rather than the summary statistics: our shape-4 series is
  min 331.3, p50 338.7, p95 378.6, **max 456.3**, sd 5.5% — tight, unimodal,
  and it never approaches 1100 µs. The earlier "our tail reaches 1102.6 µs"
  came from a summary statistic that did not survive contact with the raw data.
  The single `BIMODAL` flag in the entire scan landed on the **reference** arm
  (shape 2, a 1590 µs mode in 4% of iterations), not ours.
  **Conclusion: the evaluator's shape-4 and shape-6 numbers are simply not
  reproducible, and chasing them is closed.** Methodology note worth keeping:
  summary statistics manufactured a hypothesis that the raw series destroyed in
  one pass. Plot the series before theorising about a tail.

- **The remaining deficit is entirely shapes 5 and 6, and the mechanism is
  clean: fusion wins when communication is latency-bound, bulk collectives win
  when it is bandwidth-bound.** Same-run medians, µs:

  | shape | ours | reference | verdict |
  |---|---|---|---|
  | 64×7168×18432 | 226.0 | 230.9 | win |
  | 512×4096×12288 | 244.4 | 290.3 | **win** |
  | 2048×2880×2880 | 244.8 | 307.9 | **win** |
  | 4096×4096×4096 | 338.7 | 342.8 | tie |
  | 8192×4096×14336 | 872.4 | 698.4 | **lose 1.25×** |
  | 8192×8192×29568 | 2785.7 | 1596.7 | **lose 1.74×** |
  | geomean | 472.7 | 446.4 | 1.06× |

  Our fused epilogue avoids RCCL's launch and synchronization overhead on the
  small shapes; RCCL's bulk transfers beat our 16-byte packet stream on the
  large ones. **E2 is therefore the only remaining target**, and it has a
  concrete number attached: each rank ships ~117 MB off-node on shape 6 against
  an exposed egress cost of 1149.9 µs ≈ **102 GB/s**, against **315-336 GB/s**
  achievable. At 316 GB/s that traffic would take ~370 µs, which would put
  shape 6 near the reference's 1597 µs and flip the geomean.

- **CORRECTION to the entry below: the real same-run denominator is 1.06×, not
  1.37×, and shape 4 is a tie.** The 1.37× came from two *separately staged*
  evaluator runs, which is not a same-run paired comparison — the very thing
  `CLAUDE.md` says is the only valid denominator. Running both arms through the
  identical protocol in **one interleaved process pool** gives, `best` µs:

  | # | shape | ours | reference | ratio | the 2-run number had said |
  |---|---|---|---|---|---|
  | 1 | 64×7168×18432 | 220.1 | 226.3 | 0.97 win | 0.85 win |
  | 2 | 512×4096×12288 | 235.1 | 277.8 | **0.85 win** | 0.74 win |
  | 3 | 2048×2880×2880 | 236.2 | 297.8 | **0.79 win** | 0.75 win |
  | 4 | 4096×4096×4096 | 330.8 | 337.0 | **0.98 tie** | 2.72 lose |
  | 5 | 8192×4096×14336 | 860.2 | 667.1 | 1.29 lose | 1.23 lose |
  | 6 | 8192×8192×29568 | 2682.6 | 1568.1 | **1.71 lose** | 4.16 lose |
  | | **geomean** | **458.8** | **433.1** | **1.06 lose** | 1.37 lose |

  We win three shapes, tie one, and lose two. **The real, reproducible deficit
  is shapes 5 and 6 — both large-egress — which points at E2, not E1.**

- **ALL FIVE hypotheses for the shape-4/6 evaluator anomaly are FALSIFIED, and
  the graded protocol is exonerated.** A multi-process harness running the
  graded structure verbatim against the same `submission.custom_kernel` the
  evaluator imports reproduces the evaluator within 3-11% on four of six shapes
  and **fails to reproduce exactly the two in dispute** (shape 4: 332 µs vs the
  evaluator's 1116; shape 6: 2687 vs 7156).
  - **H1, `clear_l2_cache()`, dead.** The flush is a 256 MiB write per rank
    (`torch.empty((32,1024,1024), int64).fill_(42)`) — big enough to evict
    MI300X's entire 256 MB Infinity Cache — and it sits at `eval.py:349`
    *before* the barrier, so its effect is genuinely paid inside the call.
    Replicated byte-exactly on all 8 devices before every timed iteration it
    costs **1-2%, uniformly**. Shape 4 needed 323 → 1116; it moved to 330.
    A good hypothesis, cheap to test, and completely wrong — which is why it
    was worth testing first rather than reasoning about.
  - **H2, the timed `_clone_data`, dead**: worth 28 µs on shape 4, 45 µs on
    shape 6.
  - **H4, epoch/credit waits assuming a warm steady state, dead**: correctness
    stayed clean on every shape and rank with barriers *and* a flush between
    every call.
  - **H5, wave-quantization geometry, dead**: the same geometry under the same
    protocol runs at 332 µs.
  - **H3, launch skew, confirmed at ~100 µs but size-independent** and already
    accounted for.
- **New leading hypothesis: a bistable slow mode in our arm.** Our shape-4 tail
  reaches **1102.6 µs with sd 30.5%** against the reference's tight 5.2%, and
  that maximum is **within 1.2% of the evaluator's shape-4 `best` of 1115.8** —
  i.e. the evaluator run looks like our slow mode entered and never left, which
  also explains its 12% and 43% relative stdevs. The tails concentrate in the
  arm where the 256 MiB alloc/free and the clone churn the caching allocator in
  the same iteration, a plausible trigger for knocking the eight ranks out of
  phase. RCCL resynchronizes every call; our spin-and-credit design may not.
  Note the diagnostic asymmetry that supports this: the **reference** arm
  reproduces with a uniform 13-19% offset on every shape, while **our** arm
  matches on four and misses on two — different error shapes, different causes.
  Next decisive test, and it is cheap: re-run shape 4's `full` arm for 500+
  iterations and plot the **raw per-iteration series** rather than summary
  statistics. Every sample is already written to per-rank JSON; only the
  analysis is missing. If it is bimodal with a ~1.1 ms mode, the evaluator
  number is fully explained.

- **SUPERSEDED (see the correction above): under the graded protocol we
  are ~1.37× SLOWER than the naive GEMM+RCCL reference on this node.** Both
  arms ran through the identical `eval.py`, same container, same shapes, back
  to back, clean node, `HK_DEBUG=0`. Full write-up in
  `exp_01_evaluator_integration/graded_protocol_result.md`.
  Using `best` (minimum of 100 runs; relative stdevs run 12–93%, so means are
  unreliable), µs: ours `251.4 / 246.1 / 262.2 / 1115.8 / 913.6 / 7155.5`
  (geomean **700.0**) against reference `295.2 / 334.7 / 347.4 / 410.8 / 740.0
  / 1718.6` (geomean **511.6**).
  **We beat the naive baseline on all three small shapes (0.74–0.85×) and lose
  badly on the large ones (2.72× on shape 4, 4.16× on shape 6).**
  Two things make this credible rather than a broken run:
  - **Test mode fails identically for BOTH arms** — the reference times out at
    `test.0` too, on `rets = [el.get(60)]`. So the 60 s per-case limit is an
    environment property of this staging, **not a defect in our submission**,
    and the two arms were measured under identical conditions.
  - Our own `check: pass` came back on the benchmark arm.
  **The inflation is not a constant per-call overhead.** Comparing our harness
  to the evaluator: shape 1 inflates 3.3×, shape 4 **5.5×**, shape 6 **2.8×**,
  but shape 5 only 1.2×. A fixed additive cost would hurt the *small* shapes
  most; instead the damage scales with payload, which rules that out.
  Leading hypothesis, and the one to test first because it is the only one that
  explains the payload-size scaling: **the evaluator calls `clear_l2_cache()`
  before every timed iteration.** We established from the ISA that our payload
  peer stores carry **no cache-scope bits**, so they land in local L2 and egress
  only when `buffer_wbl2 sc0 sc1` flushes them — this kernel is unusually
  L2-dependent by construction, and a cold L2 every call may cost us far more
  than it costs NCCL. Other candidates: ~100 µs launch skew paid per call with
  no pipelining to hide it (explains the small shapes, not shape 6), the
  ~121 MB `_clone_data` inside the timed region on shape 6, and epoch/credit
  waits that assume a warm steady state.
  **Consequence for planning: the pipelined geomean is not the graded score,
  and the gap is not a constant factor.** Tonight's −10.2% is real device time
  and the kernel is genuinely faster, but further mainloop work has low
  marginal value for the graded statistic. The next experiment should be in the
  egress/L2 path, not the GEMM.

- **BLOCKER A IS ROOT-CAUSED AND FIXED. The evaluator hang was ours, and it was
  one line — a strong reference.** `hk_submission.py` cached the current
  `ProcessGroup` in a module global `_LAST_PG` as a **strong** reference, to
  distinguish a live group from a recycled address. That reference outlives
  `destroy_process_group()` and **pins the old NCCL communicator and its
  TCPStore**, so the *next* `init_process_group` on `eval.py`'s fixed
  `MASTER_PORT` can never complete. The in-code justification was right about
  addresses and wrong about lifetime. Demoting `_LAST_PG` to a **weakref**
  fixes it: a dead referent returns `None`, which still defeats address reuse,
  and torch holds its own reference while a group is current so all ranks
  compute the same `settled` answer at the same call — collective symmetry is
  preserved.
  Verified: 8/8 ranks, two shapes, two process groups, `allclose=True` at
  `max|diff| = 9.766e-04`. A **control** using plain `torch.matmul` +
  `reduce_scatter_tensor` — no globals, no IPC — passed the identical
  init/destroy/init sequence, which is what proves the fault was ours and not
  torch, NCCL or the container.
  What this retires from the ledger:
  - **Suspect #1 (repeated `hipIpcOpenMemHandle` → `hipErrorAlreadyMapped`) is
    dead.** Under `eval.py` all 8 ranks completed the full IPC exchange, both
    regions, all 7 peers, with zero `FAILED peer` and zero `AlreadyMapped`.
  - **The "6 of 8 ranks reach `state ready`" signature is superseded** — all 8
    now reach it. The earlier signature came from the pre-fix cache, and the
    falsifiable prediction written into the fix held exactly: no rank stranded
    at `setup barrier returned`.
  - The kernel itself was never implicated. Under the evaluator the 8 ranks ran
    **101 consecutive `custom_kernel` calls each — 808 total, zero error bits,
    in 1.38 s** — and passed the evaluator's *own* correctness oracle
    (`_run_distributed_benchmark` runs one obligatory `wrap_check_implementation`
    and bails on failure, so 100 subsequent repeats prove it passed).
  **Still to do: no numbers were taken.** The 60-minute box went entirely to
  root-causing. Test mode has not been run in-session and every run used
  `HK_DEBUG=1`, so nothing from that session is quotable under our own rule.
  Two branches remain unexercised on hardware: the fixed path has seen only
  three shapes, and the **mixed-vote** branch (some workers holding the key,
  others not, after a worker↔rank permutation) has still never executed —
  both `mp_smoke` cases voted unanimously `False`. Watch the `cache vote` lines.

- **TRAP: `push.ps1` is not safe to run while another experiment is mid-edit.**
  It `scp`s the kernel sources as well as the tree, so pushing while a
  concurrent arm has a half-written `gemm_rs_mi300x.cpp` ships broken code —
  or, worse, silently reverts a validated kernel while leaving `build/*.so`
  intact, so the next run measures something nobody chose. Track A correctly
  refused to push for this reason after finding the node and Windows copies
  differed. Use `tools/fix_node_crlf.sh` when only line-ending normalization is
  wanted; it touches files already on the node and copies nothing. Check first
  with an LF-normalized hash comparison rather than assuming.

- **Attribution RE-MEASURED on the post-E1(b) winner, and it reorders
  everything.** The old table was taken on a binary with neither `NR=32` nor
  the mainloop overlap, so it described a kernel that no longer exists. Fresh
  numbers (µs, single-cut deltas, so they overlap and need not sum):

  | shape | full | GEMM | XGMI | reduce | sync | release |
  |---|---|---|---|---|---|---|
  | 64×7168×18432 | 78.4 | 12.4 | 2.2 | 1.4 | 9.9 | 1.1 |
  | 512×4096×12288 | 105.3 | 39.9 | 20.8 | 4.5 | 22.5 | 21.4 |
  | 2048×2880×2880 | 91.5 | 17.5 | 25.5 | 10.0 | 17.1 | 9.4 |
  | 4096×4096×4096 | 203.5 | 41.6 | **86.5** | 19.7 | 31.9 | 20.1 |
  | 8192×4096×14336 | 748.7 | 305.5 | 295.0 | 41.6 | 48.4 | 58.9 |
  | 8192×8192×29568 | 2500.8 | **1153.3** | **1149.9** | 78.7 | 89.9 | 219.3 |

  - **XGMI egress rose from 919.7 µs (32%) to 1149.9 µs (46%) and is now tied
    with GEMM as the largest pool on shape 6.** It went *up* in absolute terms
    even though nothing in the egress path changed: speeding up the mainloop
    **unhid** communication that used to sit behind it. Expect this whenever a
    compute pool shrinks — the ablation measures *exposed* cost, not work.
  - `sync` collapsed 246.9 → 89.9 and `reduce` 219.5 → 78.7, both largely from
    `NR=32` giving the reduce side four times the CTAs.
  - On shape 4, XGMI alone is **42.5%** of the total; it is an egress-bound
    shape and the mainloop work there is only 41.6 µs.
  - **Re-run the attribution after every accepted win.** Ranking experiments off
    a stale table is how a night gets spent optimizing the wrong pool.

- **The largest remaining geomean lever is a ~65-74 µs floor on the three small
  shapes, and no stage cut explains it.** Removing the *entire* GEMM mainloop
  leaves shape 1 at **66.0 µs** of its 78.4, shape 2 at 65.3 of 105.3, and
  shape 3 at 74.0 of 91.5 — a nearly shape-independent floor. Summing shape 1's
  attributed stages gives 12.4+2.2+1.4+9.9+1.1 = 27 µs against a full 78.4, so
  **~51 µs is unattributed to any stage**.
  Why this outranks the big pools: the ranking statistic is a **geometric**
  mean, so a proportional win counts equally on every shape and therefore a
  microsecond is worth far more on a small shape. Shape 1 is 77.31 µs and shape
  6 is 2520.29 µs, so **1 µs on shape 1 is worth 33× the geomean of 1 µs on
  shape 6**. Halving the 66 µs floor would be worth ~7% of geomean — more than
  any single pool cut currently on the table.
  Candidate mechanisms, none yet tested: cross-rank protocol round-trip latency
  (a reducer cannot finish until all eight peers publish); the `s_sleep(4)`
  poll quantum in `detail::pause()`
  (`include/cdna3/ops/group/distributed/sync.cuh:81-86`, ~256 clocks ≈ 135 ns
  per miss, which alone does not explain 66 µs); persistent-grid launch and the
  per-CTA epoch RMW; and the per-tile epilogue on a shape where every CTA runs
  exactly one tile so nothing amortizes. **Measure before optimizing** — a
  combined-cut arm (noMain + noRed + noProto together) would establish the true
  floor, which single-cut deltas cannot.

- **TRAP, and the sharpest one of the session: `s_waitcnt lgkmcnt(0)` alone
  does NOT protect a register loaded from LDS.** A bare wait has no operands,
  so it creates no data dependence. The `ds_read` is an `asm volatile` whose
  output the compiler believes is live immediately, and `v_mfma` is a plain
  intrinsic with no memory effects — so the scheduler is free to hoist MFMAs
  *above* the wait, and it did, in **exactly** the three `K_TAIL=false`
  instantiations and nowhere else. Result: **14 of 17 shapes failed with
  `max|diff|` between 1e19 and 9e30, NaNs, and NO error bit set.** The
  `K_TAIL=true` instantiations were clean only by accident, because
  `mask_a_k_tail` happened to fragment the scheduling region.
  Fix: launder each base tile through `asm volatile("" : "+v"(...))` after the
  wait, creating a real def-use edge. **Zero instructions, zero registers.**
  Generalizable rule: **when LDS traffic is hidden behind `asm volatile`, the
  wait and the consumer must be tied by a data dependence, not by program
  order.** Ordering alone is not a contract with the scheduler.
  This is the same root cause as the previously-recorded latent last-k-iteration
  race — `SIInsertWaitcnts` cannot see LDS events it never observes — and
  `experiments/exp_03_mainloop/lds_race_check.sh` now catches the whole class
  mechanically for both counters in ~30 s. Run it after any mainloop edit. It
  also confirms the old latent race is now repaired.

- **TRAP (self-inflicted): a gate assertion that is too strict blocks
  everything, and it is as bad as one that is too loose.** After fixing M2's
  silent false pass I asserted **7** instantiation rows — but exp_04 retiled
  config row 1 onto the pre-existing `<32,64,64,false>` template, so there are
  now **6** distinct instantiations (rows 4 and 5 also share
  `<256,256,32,false>`). The committed ladder could not pass on the committed
  baseline. Fixed with a named `M2_EXPECT` constant and a failure message that
  says to update it when the dispatch table changes. **Count distinct
  instantiations, not config rows.**

- **E2: the XGMI ceiling is 315-336 GB/s per GPU, not 448, and our 127 GB/s is
  ~39% of achievable — but the "16-byte scattered stores" diagnosis is
  probably WRONG.** Two separate findings, and the second one matters more.
  *The ceiling.* MI300X gives each GPU **seven independent point-to-point
  links**, one per peer — no aggregation, no 2-hop peers. Per link 64 GB/s
  theoretical but **45-48 GB/s achievable** (~25% protocol/CRC haircut that is
  not recoverable in software), so ~**315-336 GB/s** aggregate.
  [AMD, Understanding xGMI and RCCL bandwidth on MI300X](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html).
  Crucially AMD's own 47 GB/s figure is **kernel-issued sender-side push
  stores** (TransferBench `USE_DMA_EXEC=0`, GFX executor) — the same pattern we
  use — so our approach is architecturally sound and there is no higher
  SDMA-only tier to chase. Real headroom is ~2.5×, i.e. ~450-520 µs of the
  920 µs pool, ~16-18% end-to-end. Damning comparison: TransferBench reaches
  329 GB/s using **8 CUs per link (56 total)**; we use ~272 producer CTAs to
  reach 127. **We are not short of parallelism**, and `vmcnt` is 6 bits (63
  outstanding per wave), so "more packets in flight" is not the lever either.
  Also settled: `global_store_dwordx4` **is** the widest store on gfx942 —
  there is no `dwordx8`. Width was never available as a knob.
  *The diagnosis correction.* The premise was that scattered 16 B packets waste
  50-75% of every 32/64 B Infinity Fabric transaction. **But the built ISA
  shows the payload stores carry NO cache-scope bits at all** —
  `global_store_dwordx4 v[56:57], v[24:27], off` and
  `flat_store_dwordx4 v[44:45], v[86:89]`, bare, while `buffer_wbl2 sc0 sc1`
  and `buffer_inv sc0 sc1` do carry them. Bare stores are not system-scope, so
  they land in the **local L2** and only reach the fabric when the release
  flushes it. L2 therefore coalesces the 16 B packets into full lines *before*
  egress, and the write-amplification story largely dissolves.
  The likelier cause of 127 GB/s is that **egress is bursty and serialized at
  the release points** rather than streamed under compute — which is consistent
  with the ablation having measured it as one lump.
  **Consequence: E2 and E3 are the same mechanism seen from two sides.** The
  `buffer_wbl2 sc0 sc1` at the release *is* the egress trigger, so release
  grouping changes burst size and frequency, not just fence count — E3 may be
  worth more than the ~3% its tile-count arithmetic suggests. It also means
  adding `sc0 sc1` to payload stores would likely make things **worse** by
  bypassing the L2 coalescing that is currently helping us.
  **This is a hypothesis from static ISA evidence and is NOT yet confirmed.**
  The decisive measurement is a profiler run counting 32 B/64 B write requests
  and fabric bytes against useful bytes. Do that before implementing either
  LDS-staging or any cache-bit change.

- **TRAP: Gate M2 could not fail. It was a silent false pass all session.**
  `tools/m2_report.sh` reads `overnight/build/m1a.log` and
  `overnight/build/isa/*.s`, which only `tools/m1_build.sh` and
  `tools/m2_isa.sh` produce — and `overnight/build/` does not exist in a freshly
  synced tree. The script runs `set -uo pipefail` **without `-e`**, and every
  step is a `grep` or a `python3` heredoc, so with its inputs missing it printed
  six missing-file errors plus a `FileNotFoundError` and **exited 0**. Any
  ladder calling it recorded "M2 PASS" while checking nothing — which would have
  hidden exactly the spill regression E1(b) is most likely to cause.
  `tools/gate_ladder.sh` now runs the two prerequisites first, greps the report
  for `Error|Traceback|No such file`, and asserts the metadata table contains
  all **7** instantiation rows before proceeding. General lesson: a gate that
  has never failed is not evidence of correctness; make it fail on purpose once.

- **E3 is worth ~3%, not 9% — the attribution that motivated it was measured
  under the OLD reducer split.** Three corrections from the protocol review,
  all of which survive independent arithmetic:
  1. **Half the scored shapes cannot benefit at all.** At `NR=32` there are 272
     producer CTAs, and tiles-per-CTA is **1 / 2 / 1 / 1 / 2 / 4** for shapes
     1-6. A release-grouping rule can only amortize where a CTA emits more than
     one tile, so shapes 1, 3 and 4 are already minimal and become free built-in
     control arms. Best-case geomean gain is ~3%.
  2. **The `vmcnt(0)` half of the release is not recoverable.** The next tile's
     `G::load` immediately issues `global_load_dwordx4 → s_waitcnt vmcnt(0)`,
     and gfx9 `vmcnt` retires in order, so deferred peer stores get drained a
     few instructions into the next tile regardless. Only the `buffer_wbl2` and
     the two barriers are recoverable — **E3's value is coupled to E1(b)**.
  3. The 250.3 µs figure was measured with the pre-`exp_02` `NR=8` split — its
     shape-6 total of 2861.7 µs matches `NR=8`'s 2850.2, not `NR=32`'s 2632.1.
     **Re-measure the attribution before sizing E3.**
  This does not cancel the exp_04 finding that E3 gates further tile work; it
  sharpens it into a *synergy*: finer tiling is what raises tiles-per-CTA, and
  higher tiles-per-CTA is exactly what E3 needs in order to amortize. Neither
  is worth much alone on shapes 1/3/4; together they may be.
  Rejected outright: **per-peer batching** (one release already covers all
  destinations, so splitting by peer strictly *increases* the release count) and
  **unbounded per-CTA-per-epoch** (the eleven official shapes take the generic
  row, where 8192×8192×28672 gives **117 tiles per CTA**).

- **LATENT CORRECTNESS HAZARD found in the shipped mainloop by ISA read
  (exp_03/P0). Not a measurement artifact — read the ISA before dismissing
  it.** In `<256,256,32,*>` the k-loop is three blocks: a header issuing 24
  `ds_read_b64` (inline asm, **no waitcnt**), a *guarded* prefetch block
  holding the only `s_waitcnt lgkmcnt(0)`, and a latch of 64 MFMAs. The guard
  `s_cbranch_scc1 .LBB3_86` (L11152) **skips the prefetch block on the last
  k-iteration of every tile**, so on that path the `ds_read`s that fill
  `A_frag`/`B_frag` reach the MFMAs with no intervening `lgkmcnt(0)`, and the
  `s_barrier` at L10966 is bare. Root cause: every LDS op is wrapped in
  `asm volatile`, so `SIInsertWaitcnts` never sees an LDS event and believes
  `lgkmcnt` is already 0 — it emits neither the use-wait nor the
  `__syncthreads()` wait. Identical in `<256,256,32,true>` (guard L16135, bare
  barrier L16373).
  On GCN/CDNA there is no hardware interlock on an LDS load's destination
  register; `s_waitcnt` is mandatory. The gates pass today (17/17 at `2e-3`,
  600-epoch soak, worst `max|diff| = 4.9e-4`), so the reads evidently land
  during the branch and MFMA issue overhead — but **this is timing luck, not a
  guarantee, and any scheduling change can expose it.**
  Two consequences: (1) the E1(b) restructure adds an explicit `lgkmcnt(0)`
  before the MFMA block, so it *repairs* this as a side effect; (2) do not
  "simplify" that wait away later as redundant.

- **The 12 B scratch / 2 spills are NOT in the k-loop (exp_03/P0).** All four
  scratch instructions in each 256/256/32 symbol are once-per-CTA (L10287,
  L10514) or once-per-tile (L10622 in the tile-loop header, L11401 in the
  `error_bit_set` epilogue); **zero** are inside the k-loop, which runs 116×
  per tile on shape 6. The spill therefore costs ~2 instructions per tile and
  is worth approximately nothing. This independently confirms that **E1(a) was
  never worth GPU time** — and it cost only a read of an already-built ISA to
  establish. Cheap static evidence before an expensive dynamic experiment.

- **The mainloop is worse than the pre-registration assumed, in a useful
  direction.** Only **2** `global_load_dwordx4` are in flight before each
  `vmcnt(0)`, so the 32 KB `BK=32` slab is drained in **two serialized halves**
  — two fully exposed global round trips per k-iteration, not one. Also
  established: counted `vmcnt(N>0)` waits already exist in this very TU (126 of
  them, `vmcnt(1..7)`, all in the reducer accumulate path), so counted waits
  are proven available on gfx942 here and are not a portability question.
  One correction to the earlier claim: the 24 `ds_read`s *are* issued before
  the global loads and not drained until after them, so LDS-read latency is
  already hidden. "No global-load/MFMA overlap" is right; "no overlap at all"
  was too strong.

- **WIN, exp_02: uniform NR=32 landed. Geomean 285.02 → 280.74 µs (−1.5%).**
  Shape 6 **2865.78 → 2633.04 µs (−8.1%)**; shapes 1/3/4/5 within noise, shape
  2 −1.5%. Full ladder passed (M3 17/17 at both tolerances, M4 all three
  controls, M5 600-epoch soak). The pre-registered prediction was 2632 µs on
  shape 6 and ~281 µs geomean — the mechanism (the donor's `NR=8` leaves eight
  reducer CTAs starving against 296 producers on the largest output) predicted
  the magnitude to within a microsecond.

- **WIN, exp_04a: the tile table was inherited too. Geomean 280.74 → 269.96 µs
  (−3.8%), from shape 1 alone at −20.3% (107.86 → 85.93 µs).** Shape 1
  (64×7168×18432) was running `32/256/32`, which gives `(64/32)·⌈7168/256⌉ =
  56` tiles against **272 producer CTAs — 21% of the machine, with 216 CTAs
  executing the tile loop zero times** — and 72 k-iterations. Moving it to
  `32/64/64` gives 224 tiles (0.82 waves) and 36 k-iterations, needs **no new
  instantiation** (that template already existed as the generic fallback row),
  and is a two-line change. Full ladder passed.
  Generalizable lesson: **every column of a donor's shape table is a
  hypothesis, not a constant.** Both the `NR` column and the `BM/BN/BK` column
  came from RadeonFlow's submitted values and both were wrong for this kernel;
  together they were worth 5.3%. The existing attribution had missed this
  because it was measured only on the largest shape, where occupancy is a
  non-issue — but the ranking statistic is a *geometric* mean, so the small
  shapes carry equal weight.

- **NEGATIVE (by analysis, not run — and the reasoning is the useful part):
  do NOT retile shape 3, and more generally the per-tile release tax gates the
  whole tile axis.** Shape 3 (2048×2880×2880) is ragged in N (`⌈2880/256⌉ = 12`
  vs `11.25`, so 6% of its GEMM is padding) and uses only 0.71 of the producer
  CTAs, so `BN=64` (45 columns exactly, 720 tiles) looks obviously right. It is
  not: `producer_drain_release` is issued **once per tile**, shape 3 already
  spends **12.5 µs** there at 192 tiles, and 720 tiles is 3.75× that — about
  **+35 µs** against a ragged-N saving of ~1.5 µs, since shape 3's GEMM is only
  23.6 µs of its 96.9. Shape 1 tolerated 4× the tiles only because its release
  cost was 0.4 µs.
  **Consequence: E3 (release granularity) is a prerequisite for E4, not an
  independent 9%.** Amortizing the release unlocks an axis that is otherwise
  closed everywhere except the one shape where it happened not to matter.
  Recomputing the geometry for the other rows shows none has shape 1's defect
  (last-wave fill is 94% on shapes 2, 4, 5 and 6), so **the tile axis is close
  to exhausted** until E3 lands.

- **exp_09_sched / E1(c) — mainloop scheduling: LANDED at 0.997×, but the axis
  is flat and the durable finding is about the compiler, not the kernel.**
  Full detail and ISA evidence in `exp_09_sched/result.md`. Shipped arm
  `b_setprio` = clamped branchless prefetch + accumulator anchor +
  `s_setprio` over the MFMA block. Paired alternating A/B, both states rebuilt
  tonight: candidate `77.77 / 88.77 / 90.81 / 201.56 / 640.97 / 1831.78`,
  geomean **230.09** vs base **230.80** (the 230.84 denominator reproduced to
  0.02%), ratio **0.9969**. Resources unchanged — 256/256/32 still 246/248 of
  256 with zero spills; 128/256/32 improved 165 → 163. M3 17/17 at both `1e-2`
  and `2e-3`, M4 3/3, M5 600 epochs, on every gated arm.

  **Scheduling directives do not move MFMAs in this TU. Data-dependence anchors
  do.** `sched_group_barrier` asking for VMEM spread through the MFMA block
  changed *not one instruction*. `sched_barrier(0x7F6)` (everything except MFMA
  may cross), placed to hold the MFMAs above the commit, produced an ISA with
  the **same 17613 instructions** — all 1156 differing lines were register
  renaming — and the commit's `vmcnt(0)` still sat after 33 of 64 MFMAs. An
  `asm volatile("" : "+v"(...))` tie on each accumulator tile (`acc_anchor`, the
  same trick as the load-bearing `frag_anchor`) got all 64 above it on the first
  try. Two independent reasons the hints cannot work: a scheduling region never
  spans a basic block, and every memory op here is `asm volatile ... : "memory"`
  so their mutual order is fixed. **VMEM issue position is set by source
  placement, not by builtins.**

  Structural results worth reusing: replacing `if (more)` with a clamped index
  `kn = (k+1 < k_iters) ? k+1 : k` collapses the k-loop from **6 basic blocks to
  1** (and 10 → 5 for `<256,256,32,true>`, whose extra splits come from the
  K-tail compare that shape 6 pays on all 116 iterations for a tail only the
  last can take). **Peeling** the last iteration achieves the same branch
  removal but duplicates the MFMA block and costs **30 VGPR spills / 50 scratch
  stores** on the 256×256 rows — rejected at M2, do not retry without solving
  the live range. Straight-lining also *loses* prefetch coverage: the block
  boundary used to force all 64 MFMAs above the commit's `vmcnt(0)`, and in one
  region the scheduler sinks 31 of them below it (21 even below
  `__syncthreads()`, which is legal — those MFMAs read registers already drained
  by `acquire_frags`).

  **NEGATIVE: `s_setprio` alone is worth nothing (232.35 µs, 1.007×).** The
  −3.1% it produces on shape 2 appears *only* on top of the single-region body,
  where the whole iteration is one instruction stream and priority decides which
  of the 2 waves/SIMD wins the issue port. Row 2 is 64/64/64 at 91 VGPRs, the
  highest-occupancy row. **That is a measured occupancy/issue-contention effect,
  which is new evidence for E1(a) (AGPR accumulators): `AGPRs: 0` everywhere and
  the 256-row pair is pinned at 246/248 of 256.**

  Also real: shape 3 **regressed 0.7%**, cause identified — the clamped prefetch
  issues one redundant tile read per tile, i.e. `1/k_iters` of mainloop global
  traffic, and shape 3 has `k_iters = 12` (8.3%) against shape 6's 116 (0.9%).
  Fix for the next session: guard only the *commit* (a block split after all the
  MFMAs is harmless) plus one `wait_vmcnt0()` after the loop.

  Arm C (counted `s_waitcnt`) was **not run**: it was gated on A or B winning
  decisively, and with the whole axis inside the noise floor a mechanism whose
  documented failure mode is a 30× error that still passes a 2e-2 gate is not
  worth it. Per the kill rule, **the directive-based approach to E1(c) is
  closed**; what remains on the mainloop is work-changing, not order-changing:
  E1(a) AGPRs, `v_mfma_f32_32x32x8` vs `16x16x16`, and the untouched `ds_read`
  side (24 `ds_read_b64` and two full `lgkmcnt(0)` per k-iteration).

- **TRAP: the LDS/VMEM race checker is layout-order, and loop rotation makes it
  cry wolf.** `exp_03_mainloop/lds_race_check.sh` went 0 → 10 hazards the moment
  a restructure let LLVM rotate the k-loop, and all ten are false positives, for
  two separate reasons: it counts a VALU **destination** as a read
  (`v_add_u32_e32 v130, s51, v196` reported as "reads v130"), and its linear
  scan cannot see a drain that rotation moved to the *top* of the rotated body,
  so on the exit path it misses the `s_waitcnt vmcnt(0)` that every execution
  path provably passes through. Signature of a false positive: the flagged
  register is redefined at the flagged line, and the loop's `vmcnt(0)` dominates
  the loop exit. **Do not relax the checker** — the silent-corruption class it
  exists for is still real; verify by hand instead
  (`exp_09_sched/11_haz_window.sh`).

- **TRAP: CRLF, not caching, is behind the "transient" build failures — and it
  can leave the binary disagreeing with the source.** Four builds died with
  `syntax error: unexpected end of file` (a `\r` on a heredoc's terminator) or
  `$'\r': command not found`, at a different line each time, all looking exactly
  like compile errors. `push.ps1`'s normalizing `sed` does not reliably win the
  race: `harness/build.sh` was CRLF when a restore build ran and LF when
  inspected minutes later. `nsh.ps1` launders only the script it transports, so
  the exposure is precisely the scripts that one `docker exec`s. Worse, that
  failed restore left the **base** `.so` under the **candidate's** source.
  Normalize *and prove it* with `bash -n` inside `dhk-gemmrs` before any build,
  and rebuild-and-verify after every source swap
  (`exp_09_sched/run_ladder.sh`).

- **Noise floor, measured properly for once: 1.27% on shapes 5/6.** Two arms
  whose ISA differed only in register numbering (identical 17613-instruction
  stream) reported shape-6 means 1.2% apart. Alternating paired reps put the
  worst within-arm spread at 1.27%. **Any single-run claim below ~1.3% on the
  big shapes, or ~0.5% on the geomean, is not evidence** — which retroactively
  means every geomean delta in this experiment except the pooled shape-2 result
  needed the paired protocol to be believable.

- **Six wedged `spawn_main` workers held GPU 1 at 0% CU occupancy** for eight
  minutes without draining, blocking M0. `tools/reap_stale.sh` identified all
  six as ours (in `dhk-gemmrs`) and one SIGTERM sweep cleared them. Waiting is
  right, but wait with a bound.

## Session 1 — 2026-08-11 (bring-up)

- **2026-08-11 (baseline, pre-overnight).** First hardware bring-up of the
  MI300X/gfx942 GEMM-RS port. Before this the port had never been compiled for
  gfx942 and had no runtime binding; every gate was `PENDING_GFX942_VALIDATION`.
  Gates M1, M3, M4, M5, M8 PASS; M2 partial. Correct on 17/17 shapes (six
  graded + eleven official), worst `max|diff| = 4.883e-4` against a `1e-2`
  tolerance; 600-epoch skewed changing-input soak clean with epoch and signal
  cells exact; all three negative controls fail as designed; graph replay
  advances device-derived epochs. Full record in `../RESULTS.md`.

- **2026-08-11 baseline timing (our harness, single process / 8 devices / peer
  access, pinned clocks, order-rotated 3×50).** Per-shape means µs:
  `108.2 / 115.4 / 97.3 / 203.5 / 765.4 / 2872.3`; **geomean 285.7 µs**; graph
  mode geomean 279.5 µs. Run-to-run spread <1%. This is NOT comparable to
  rank-1's published 413.139 µs (different machine, different protocol) — the
  like-for-like evaluator run is the open item.

- **Attribution (macro-gated ablation, largest shape 2861.7 µs):** GEMM
  mainloop **1317.8 µs (46%)**, XGMI egress **919.7 µs (32%)**, per-tile L2
  writeback release **250.3 µs (9%)**, cross-rank sync 246.9 µs, reduce
  219.5 µs. Optimization order follows this, not intuition.

- **NUM_REDUCER_CTAS is settled: a uniform NR=32.** Swept {8,16,24,32,40,48}
  per shape, all correct. NR=32 is optimal or within noise everywhere and beats
  the inherited RadeonFlow table (32/48/48/48/32/8) by **8.3% on shape 6**
  (2850.2 → 2632.1 µs). Carrying over a donor's submitted constants was not
  free. Not yet landed in the shape table.

- **NEGATIVE: allocation granularity does not matter.** Payload heap
  fine-grained (uncached) vs coarse-grained (cached) is within 1% on every
  shape, and correct in all arms including coarse signals. The hypothesis was
  that uncached payload defeated L2 on the reducer's local reads of all eight
  slots; it is wrong. The protocol's `buffer_wbl2` / `buffer_inv` handshake is
  doing its job. Do not re-run.

- **NEGATIVE (partial): a per-launch host API call was real but not the floor.**
  `hipFuncSetAttribute` was being issued on every launch inside `launch_fixed`;
  it is a property of the instantiation and is now set once. This cut host
  issue cost ~90 → ~62 µs per operation but did not move device time. Committed.

- **TRAP: GPU clock ramp invalidated an entire attribution.** Idle sclk here is
  ~120–132 MHz vs ~1900 MHz loaded. A fixed 10-iteration warmup on a ~100 µs
  kernel is a few ms of load — nowhere near steady state. The same shape read
  95.7 µs in a back-to-back sweep and 153.1 µs in a fresh process. Fixed with
  `rocm-smi --setperfdeterminism 1900` plus duration-based warmup; spread is now
  <1%. Always pin before timing.

- **TRAP: a line-ending normalizer corrupted compiled modules.** `sed -i
  's/\r$//'` walked every file under the harness including `build/*.so`,
  stripping `0x0D` bytes out of the binaries. Every process then segfaulted in
  `dlopen`, which looked exactly like a multiprocessing or HIP-init bug and cost
  about an hour of misdirected debugging. `tools/push.ps1` now filters by
  extension and excludes `build/`.

- **TRAP: the graded tolerance barely detects protocol corruption.** With
  inputs in ±0.01, `CTRL_REROUTE_SLOT` corrupting a whole 1-of-8 reduction
  contribution produced `max|diff| = 7.5e-3` — inside `allclose(1e-2, 1e-2)`.
  A healthy run is `4.9e-4`. Every check now also asserts a tight `2e-3`.

- **TRAP: `CTRL_REROUTE_SLOT` is invisible under unchanged inputs.** The
  rerouted band leaves the true destination slot holding its previous epoch's
  bytes, which are the correct answer if the inputs did not change. The control
  reported a false pass until the run was restructured to use two epochs with
  different seeds compared bitwise against an unrerouted run.

- **TRAP: the evaluator's worker↔rank map is not stable.** `eval.py` drives
  ranks with `multiprocessing.Pool(8)` and reassigns workers between test cases.
  A submission-side cache keyed only by shape handed rank 1 an allocation
  belonging to `cuda:6`; the evaluator caught it as
  `Output device mismatch: cuda:6 != cuda:1`. Any per-process state must be
  keyed by rank **and** by process-group identity. The timings emitted by that
  broken run (303/463/2099/3043/21509 µs) are meaningless — benchmark mode does
  not re-check correctness.

- **The SOL table is not a target.** The published speed-of-light numbers are
  the bf16 MFMA roofline with zero budget for the reduce-scatter (shape 6:
  `2·8192·8192·3696 / 1.307e15 = 380 µs` vs a tabulated 379.43 µs). rank-1's own
  score is ~10.4× that table. Size everything against a competitor measured on
  this node.

- **OPEN: rank-1 needs version archaeology, not an install.** Four disclosed
  compatibility repairs so far (iris staged at its hardcoded python3.10 path;
  no-op `sudo`; a Triton 3.6.0 `wrap_handle_tensor_descriptor` stub verified
  never called; `packed_metadata` 6→3 fields, behaviour-preserving because the
  dropped `clusterDim*` are never read). A fifth — aliasing
  `iris.hip.hipIpcMemHandle_t` to `gpuIpcMemHandle_t`, which **all 8** iris
  checkouts on this node renamed — is written but untested. Details and the
  suspect ranking in `../HANDOFF.md`.
