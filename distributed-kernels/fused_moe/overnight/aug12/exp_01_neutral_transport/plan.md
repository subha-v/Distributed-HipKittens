# exp_01 — neutral transport plane

Status: implementation in progress  
Feeds: methodology Stages 0–1; engine-crossover figure

## Question

For an identical two-rank producer → transport → dependent-consumer DAG, where
do CU push, CU pull, host-enqueued copy, MORI P2P, and MORI SDMA cross over as
record size changes? This experiment identifies transport crossovers only; it
does not select an overlap carrier or MoE kernel.

## Stage-0 anchor

- ranks: 2, one rank per GPU
- record size: 64 KiB
- total payload: 64 MiB per rank
- fanout: 1, symmetric bidirectional rounds
- routing: deterministic one-peer work list
- consumer: exact integer checksum/reduction over received records
- lifetimes: one epoch and ping-pong epochs
- compute repeats: frozen once against the CU-push `Tc/Tm ≈ 1` anchor

Required complete methods:

1. bulk synchronous produce → transport → consume
2. CU push
3. CU pull
4. reserved service-CTA staged push
5. homogeneous producer-carried staged push
6. direct producer publication
7. host-enqueued copy path
8. MORI nonblocking put forced to P2P
9. MORI nonblocking put forced to SDMA
10. separate persistent producer/consumer kernels

Stage 0 exits only after every implemented arm passes semantic parity,
timer agreement, mandatory negative controls, and a 600-epoch soak. Missing
backends remain explicit `blocked` records; they are never silently dropped.

## Stage-1 grid

- ranks: 2, both directions
- record bytes:
  `256, 1024, 4096, 14336, 65536, 262144, 1048576, 4194304,
  16777216, 67108864`
- fixed-volume family: 64 MiB/rank
- fixed-count family: 4096 records/rank while the two-buffer footprint plus
  workspace is at most 50% HBM; larger records reduce count geometrically
- methods: CU push, CU pull, traced host copy, same-API MORI forced-P2P, and
  same-API MORI forced-SDMA
- rotations: method and direction order rotate within paired campaigns

## Measurements

Primary:

- synchronized host/coordinator `Tjoint` from epoch release until both ranks
  complete
- p50 and p95 per-iteration global makespan
- paired campaign deltas and confidence intervals

Explanatory:

- amortized time per record
- aggregate/per-link GB/s
- enqueue/descriptor time
- completion latency after the final payload operation
- source/destination CU activity
- executor trace and memory stratum
- payload/control bytes and publication counts
- rank-max minus rank-median completion tail

## Correctness and negative controls

Every timed point uses epoch-dependent payload values, poisons inbox/output,
checks an exact digest plus sampled elements, checks zero surviving poison,
checks generation monotonicity, performs an all-rank error reduction, and
completes a 600-epoch soak.

Required failures:

1. omit one publication
2. publish before payload completion
3. redirect one destination
4. omit one credit return for reusable-buffer arms
5. reuse one ping-pong slot one epoch early
6. place SDMA payload and signal on unordered queues

## Decision rules

- Select one size below, nearest, and above every measured method crossover for
  Stage 2.
- A point without executor verification can enter the complete-path curve but
  not the hardware-engine curve.
- A bandwidth win without lower global makespan is a transport result, not an
  overlap win.
- Differences below 2% require a power calculation and at least seven paired
  campaigns.

## Reuse and isolation

Fork exp_22's MFMA/XGMI bodies, timestamp conventions, JSONL resume behavior,
checksum style, and ISA/resource scripts. Do not mutate exp_22. Keep the
one-process HIP driver as a fast diagnostic only; final Stage-0/1 results use
the common rank-per-GPU launcher.

## Deliverables

- `design.md`
- `schema.json`
- `transport_crossover.json`
- `result.md`
- `raw/` build, executor-trace, gate, soak, and campaign logs
