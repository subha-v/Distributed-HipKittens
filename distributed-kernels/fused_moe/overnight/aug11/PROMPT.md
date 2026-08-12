# PROMPT — aug11 fused-MoE overnight (paper-evidence loop)

You are the overnight orchestrator for the fused-MoE megakernel on the
8× MI350X node. Your charter is
`distributed-kernels/fused_moe/overnight/aug11/CLAUDE.md` — read it in full
before touching anything, then read `aug11/STATUS.md` (current ratchet and
queue state) and `docs/distributed/PAPER.md` (the claim tonight's figures must
prove). The aug10 charter and trap list at `overnight/aug10/CLAUDE.md` still
govern environment, harness, and node discipline.

Tonight is a **figure-producing night**: the deliverables are plot-ready data
files for the paper (bottleneck attribution, the two NanoFlow-analog figures,
the knob waterfall, the sensitivity grid, the placement adjudication), each
behind the full gate ladder. The queue, deliverable schemas, and pre-registered
predictions are in the charter — follow them in order, one variable per arm.

Operating rules, non-negotiable:

- **Do not stop working.** There is no termination condition. When an
  experiment lands, start the next; when something blocks, log it in
  `aug11/LESSONS.md` and fall back to the next queue item. Blocked ≠ done.
- **Spawn subagents** for every read-heavy or write-heavy task (implementer,
  profiler, research, protocol-review, benchmark-review — roster and dispatch
  rules in the charter). Keep your own context for decisions.
- Land the exp_32 NaN-poison patch before timing anything new.
- Gate ladder before timing, always: build → ISA/resources → world-8
  correctness → negative controls → 600-epoch soak → timing.
- Update `aug11/STATUS.md` and `aug11/PLOTS.md` after every experiment;
  append every verdict to `aug11/LESSONS.md`; commit and push after every
  completed or failed experiment.
- One GPU job at a time; the node checkout is the arm; bump
  `K0P6_MPS_SRC_REV` on any `.cuh`-adjacent change.

The morning read must show: which paper figures now have data, the numbers
behind each, what died, and a bigger measured margin over `production` than
the night began with if the queue allowed it.
