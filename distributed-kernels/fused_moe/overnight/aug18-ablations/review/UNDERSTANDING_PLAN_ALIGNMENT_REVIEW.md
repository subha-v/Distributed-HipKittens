# Mentor-alignment review of UNDERSTANDING_PLAN.md (draft r1)

Opus adversarial reviewer, 2026-08-18. Reviewed the pre-Track-1 draft. Verdict: the
reorientation is real (W-grid, prediction machinery, Phase-4 ladder genuinely gone; §3 correctly
built as deltas between existing arms), but not committable: it explained the fast arm well and
the slow arms poorly, measured anchor B at an operating point that guarantees the answer, had no
instrument-perturbation control, and §5 was still a benchmark report. All items folded into r2.

## BLOCKING

- **BL-1**: anchor B scheduled at ρ=0.65 (deployment 2.6–3.2) with the one-argv ρ ladder dropped
  from both the kept and dropped columns — an h_ledger/I_co at that point describes the rig, not
  the kernel. Fix: fold ρ ∈ {0.65,1,2,3} into the ledger session; no anchor-B mechanism claim at
  a single ρ.
- **BL-2**: no instrument-perturbation control in an instrumentation-first campaign. LAW-32
  (+726.9 µs from making code merely reachable) and the S8 receipt (STAGED=1 ⇒ 96 scratch loads +
  97 vmcnt(0), throttle annihilated) say a "free ledger" does not exist at 256 VGPR/AGPR +
  155,428 B LDS. Fix: per instrumented arm — .text sha256; flag-OFF proven .text-identical;
  flag-ON proven to differ; instrumented-vs-uninstrumented wall parity within the noise band,
  else the arm's spans are void.
- **BL-3**: the ledger does not exist for anchor A; the DP-body port is a second, larger,
  unpriced build; [MPS TS] is not a substitute as-is (running maxima; rank-0-only print vs a
  max-across-ranks estimand — +38% on production's combine). Fix: split R2a/R2b(TP8)/R2c(DP
  port), price the port, require cross-rank reduction in the ledger contract.
- **BL-4**: anchor B's PRODUCTION baseline never itemized — the 2.8 ms/layer exposed-AR prize
  carries an explicit "never profiler-confirmed on the native TP8 arm" honesty flag. Fix: one
  profiled native-TP8 step (~0.3–0.5 h) or strike every prize-pool statement.
- **BL-5**: the mentor's "low concurrency" half is never explained on our own slow arm: the
  fused family inverts below T≈1,600 (0.7557×→1.0940×) and the two-point fit already gives
  fixed cost a=1,068.2 µs vs production's 181.3 µs (unfitted midpoint holds to ~1–2%). Itemizing
  that 1,068 µs on the EXISTING tgen-body arms is explanation, not a sweep. Fix: add Q11.

## MAJOR (abridged)

- **MAJ-1**: R9's size ladder is the dropped B2 sweep in disguise; Q9 needs a matched-size
  decomposition at 235 MB. Cut the ladder.
- **MAJ-2**: K4's uniform C grid was built for a falsifier that no longer exists; explain the
  knee by measuring the two rates, trim the grid, share {30,32} with hang forensics.
- **MAJ-3**: EXPLAINED/PARTIAL/OPEN has no closure criterion. Fix: per Q — an attribution bar
  (itemized spans account for ≥X% of the banked delta, residual below the noise-scaled
  resolution) and one line of "what we would see if this mechanism were wrong."
- **MAJ-4**: no kill criteria/degradation ladder — three lines needed (R2 inconclusive; R7
  non-reproducing → hang-rate table + static wait-graph hazard; R2c slips → waterfall from HIP
  events + rocprof only).
- **MAJ-5**: missing specimens we own at zero node cost: S8 staged wide-push (built, never
  validated, ISA receipt in hand — the register-pressure card); mode-14 reachability (+726.9 µs,
  source untouched — the most teachable non-obvious result); T3 named-as-excluded.
- **MAJ-6**: Q8's phenomenon lives on the DP/mps chassis; B.res is a boundary-rig cell — scope
  or move the second-kernel arm to the DP chassis.
- **MAJ-7**: the dedicated-carrier row is internally inconsistent (0.888× is FASTER than
  homogeneous 0.8958×); the verdict is carrier-conditional (won in staged-push, lost
  monotonically in producer-carried) — that unattributed fast/slow pair IS the mentor's framing.
- **MAJ-8**: one σ cannot size three instruments. [Operator cut the dedicated arm; r2 uses
  banked per-instrument bands + free repeats.]
- **MAJ-9**: repricings — R4 ≈3.0–3.6 h (was 1.5–2.0); R7 ≈1.0–1.5; R8 ≈1.2 at the 3-min
  endpoint; R9 −0.5; missing +20% retry allowance and the 0.6–0.7 node-availability derate;
  no K/n column anywhere; quote both endpoints of the uncited [1,3] min boundary unit.
- **MAJ-10**: §5 is a benchmark report — restructure MoK-shaped: (1) problem + anchors with
  full tuples; (2) THE MEGAKERNEL DESCRIBED (phases M0–M9, roles, occupancy contract,
  certificate protocol, fusion boundary, one dataflow figure) — the biggest single gap vs MoK;
  (3) design decisions with the evidence that forced each; (4) what did not work; (5) the
  anatomy/waterfalls; (6) abstractions; (7) lessons + scope. Plus "How we differ from MoK, and
  why" (their dedicated comm SMs work; ours cost +831 µs of role overhead on an all-CU,
  grid-barriered, occupancy-1 CDNA grid — the paper's most interesting page) and a
  reproducibility/manifest note.
- **MAJ-11**: free mechanism cards left on the floor: layout as schedule decision (52.8 vs 4.1
  GB/s); smallest early-EXECUTABLE unit (t50 = 91.7%); push-vs-pull by progress ownership;
  coalescing protocol atomics ≥10× SLOWER (atomics resolve at the line — maximize L, the
  opposite of the load rule); XCD-confined pool (net 7–23% worse); pacing (monotone, closed);
  and the M23 coverage defect (mega ran ~2% of in-bucket steps; every pre-M23 serving A/B
  measured a doubly-handicapped candidate) as an integration lesson.

Plus 16 text-level fixes (workload tuples on every table; "235 MB boundary" relabeled
"low concurrency, largest message"; X2 constraint pinned: any T≠4096 number must come from the
tgen body and be labelled as such; tgen-body provenance on every T-sweep ratio per the archived
t2048/t1024 retraction; GEMMs = 88.1% of the interior stated up front so the comm story is
honestly framed as a fight over ~12%).
