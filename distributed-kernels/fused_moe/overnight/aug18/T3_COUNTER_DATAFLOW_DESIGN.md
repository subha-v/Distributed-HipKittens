# T3 — the counter-dataflow training megakernel (MoK synthesis, 2026-08-18)

Source study: /Users/subha/repos/mixture-of-kittens (Cursor's Blackwell MoE
training megakernel, `csrc/mok_megakernel.cuh`, ~3.3k lines, one fwd + one
bwd megakernel per MoE layer).  This doc records what transfers to gfx950
and the port plan.  Companion evidence: V6_EVIDENCE_AND_DESIGN.md (tonight's
falsifications and the banked 1,587.8 ms / 20,739 tok/s stack).

## 1. What MoK does (cited in the study transcript)

- ONE backward megakernel per layer: reverse-combine (dY dispatch with
  router-weight scaling in-flight), dgrad GEMMs, swiglu-bwd, **wgrad as
  first-class tasks**, in-kernel activation REPLAY (re-dispatches tokens
  over the wire for macrobatches beyond the ring buffer), router-grad
  writeback to origin ranks.  No side streams, no separate wgrad kernel,
  no NCCL in the hot path, no CUDA graphs, no CPU-GPU sync.
- Grid = one 2-CTA cluster PER TASK (thousands); Blackwell CLC
  work-stealing turns it into a persistent worker pool.  NOT an all-SM
  residency kernel.  Surplus clusters early-exit off a device-resident
  token count.
- ZERO grid-wide barriers.  Every dependency = flat HBM counter
  (ld.relaxed spin + nanosleep, red.release.add), tile/row-block granular.
  Producers get lower task indices than consumers; dequeue order is
  monotonic, so launch order == dependency order.
- Wgrad streams BEHIND dgrad: each dW task owns one 256x256 tile of one
  expert, iterates K over that expert's rows, and `wait_for_wgrad_operands`
  spins per producing row-block exactly when the K loop crosses it — the
  drain bubbles of dgrad are wgrad's execution slots.  Cross-macrobatch dW
  accumulation = TMA store_add serialized by a buffer-done counter
  (determinism as a scheduling property).
- ZERO cross-rank waits inside the kernel: routing all-gathered once
  (multicast), the global schedule recomputed identically on every rank,
  dispatch PULLS from peer buffers sealed by a pre-kernel device barrier,
  combine PUSHES to buffers read after the post-kernel barrier.  Drift is
  realized ONCE per layer per direction.  Dedicated comm SMs (24-40,
  precision-dependent) inside the same kernel.
- Host seam: free functions, ~6-8 launches/layer, weights prequantized
  outside (for FSDP), dW returned to the framework optimizer, one stream.
- Benchmark discipline worth adopting: metric = median over iters of
  MAX-across-ranks latency (drift-inclusive critical path).

## 2. Why this explains tonight's falsifications

Our all-256-CTA grid barrier creates the residency contract that made
every separate wgrad kernel standoff or serialize, quantizes execution
into lockstep phases whose bubbles are small (~150 CTA-ms vs wgrad's
~3,360), and realizes cross-rank drift at three points per launch
(M0/M2/M8) instead of one per layer.  MoK never built any of those
constructs, so our pathologies are structurally absent there.

## 3. The T3 port (gfx950)

Keep: GEMM cores (phase-1b/2b bodies, wgrad tile body), LL128 wire +
in-flight fp8 quant, saved-plan reuse, symmetric-heap protocol
primitives, the t2v6 cursor/ticket machinery (it IS the CLC analogue:
atomic ticket on a 256-CTA worker grid gives identical dequeue
semantics — AMD's 1-block-per-CU at our LDS makes 256 the natural pool).

Replace, in order of leverage:
1. **M2 grid barrier → per-(source,chunk) arrival gating.**  chunk_ready
   epoch words already exist per (source, chunk); the backward's saved
   plan makes block→(source,chunk) needs fully static.  Compute tasks
   gate on their chunks' arrival (a2_done idiom), so a fast rank starts
   phase-1b on early sources while the straggler's rows are in flight.
2. **Slab rendezvous + M8 16-word certificate → block/minibatch-granular
   counters** consistent with the paper's own "signal at the consumer's
   frontier" law (MoK gates per minibatch, not per row — our serving
   Finding 2 stands).
3. **Wgrad = ticket tasks with per-K-slab operand waits**
   (wait_for_wgrad_operands analogue over a2_done/dZq-block counters),
   replacing both the M8.5 phase and the windows; dW accumulation via the
   existing accumulate path serialized by the ticket order.
4. **One cross-rank seal per layer per direction**: pre-staged inputs
   (dY already is), outputs read post-kernel; retire/epoch machinery
   collapses toward MoK's two-barrier shape.
5. Later: shared-expert tasks in the same queue (MoK runs the shared
   expert in-kernel), replay-style recompute if activation memory
   becomes the binding constraint.

Non-transferable: CLC (atomic ticket replaces it), TMA bulk peer loads
(comm workgroups do global loads over xGMI — our epilogue-carried RMW
already covers combine), multimem multicast barrier (per-peer release
stores + acquire spins — our retired[] pattern).

## 4. Projection

On top of the banked 1,587.8: in-mega drift realization (~150 ms) →
mostly deleted; wgrad visible (~200 ms) → mostly absorbed (MoK proves the
mechanism at production scale); boundary export shrinks with it.
Landing zone ~1,280-1,330 = production parity from the skeleton swap
alone, then shared-expert-in-kernel and attention-wgrad tasks carry into
the 1,020-1,100 target band.  Risk ledger: counter-dataflow reintroduces
fine-grained polling the serving campaign measured as toxic in its
carrier-pool form — the difference is these counters gate CONSUMERS on
certified-enough frontiers (block/minibatch), not carriers on rows; keep
the exp_29/exp_30 lesson as the granularity floor.
