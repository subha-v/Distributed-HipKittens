# exp_16 — single-contributor fast path: a NULL, and it decomposes the remaining bottleneck

**Verdict: NULL, reverted. But it is the most informative null of the campaign,
because it bounds how much of the remaining M7 interference the atomics can
possibly account for — and the answer is "less than a fifth".**

## The change

When `row_rem[r] == 1`, exactly one block contributes to row `r`, and the event
being processed is from that block. The arrival `fetch_add_acq_rel` is then
provably pointless:

- the counter could only ever reach 1, so the RMW cannot report anything the
  lane does not already know;
- the acquire edge it would carry is already held — the producing CTA published
  the event with a release and the wave did `thread_acquire<agent>` after
  reading it, and that producer is the row's **only** contributor;
- uniqueness is structural rather than earned: no other event covers `(r, nc)`.

So the arrival atomic was skipped entirely under
`live && g == 1u && target == 1u`. Correct by construction, resource-neutral,
gate-green.

## Result

Screened at g=1, mode 2, timestamps on, against exp_14 at the identical configs:

| C | exp_14 M7 | exp_16 M7 | delta | exp_14 total | exp_16 total |
|---:|---:|---:|---:|---:|---:|
| 48 | — | 2,704.3 | — | — | 6,941.0 |
| **64** | 2,835.3 | 2,845.3 | **+10.0** | 6,949.2 | 6,936.2 |
| 96 | 3,290.0 | 3,291.1 | **+1.1** | 7,157.4 | 7,054.9 |

Every delta is inside the screening band. **No effect.**

## Why it is null, and what that tells us

The most likely explanation is simply that `row_rem[r] == 1` is **rare** on this
workload. My estimate of "~1.53 average, so most rows are singletons" was
derived from `padded = 33,440` pairs over ~21,816 distinct rows, and an average
of 1.53 does not imply a mode of 1 — with `topk = 8` spreading each token across
experts, the distribution is evidently concentrated above 1.

The useful consequence is a **rate**, calibrated from exp_14. That experiment
removed ~698,112 atomics per rank per epoch and bought **−306 µs** of M7, i.e.
roughly **0.44 µs per thousand atomics removed**. Applying that rate to the
535,040 arrival atomics that remain:

> Removing **every** remaining arrival atomic would be worth about **−235 µs**,
> against the **+1,249 µs** of M7 interference still outstanding. **The atomics
> can account for at most ~19% of what is left.**

## What that means for the next move — the remaining interference is PAYLOAD

This does not contradict exp_05, and it is worth being precise about why.
exp_05 showed interference **rises with `g`** while payload bytes are constant,
which proves atomic traffic **contributes**. It never showed payload contributes
nothing. The exp_14/exp_16 rate now bounds the atomic share, and the residue —
roughly 1,000 µs — has to be the service pool's own memory traffic contending
with M7's.

That reopens something I retracted too early. The retraction of M10 (SDMA) was
argued as "SDMA offloads the payload, and the payload is not the problem". The
first half of that is right and **the second half is now wrong**: the payload is
most of what is left.

Two candidate directions, and they are not equivalent:

1. **Move the bytes off the CU memory path** (M10, mori CCO device-side
   `ccoSdma`). Recovers the contention wholesale if SDMA is genuinely a separate
   path on this part; historical AMD guidance says SDMA loses to vector stores
   at 4–64 KB, but that guidance is about *throughput in isolation*, not about
   interference with a concurrent GEMM, which is the cost that decides it here.
2. **Halve the bytes.** The service pool currently does `load(part)` **plus**
   `store(peer)` — it touches every byte twice. A compute CTA writing its
   epilogue result directly into the owner's slot touches each byte once, and
   the value is already in registers, so the 312 MB read of `part` disappears
   from the M7 window entirely. That is A11/M11, it has a measured precedent
   (AMD Research SC24, 12% on fused GEMM+All-to-All), and CLAUDE.md requires
   **protocol-review signoff before build** — correctly, because it makes slice
   completion a remote-atomic ordering problem.

Direction 2 is the larger prize and the larger risk. Direction 1 is bounded by
whether `ccoSdma` is usable from inside a persistent megakernel.

## Disposition

**Reverted.** The change is correct and costs nothing, but it buys nothing on
this workload's row distribution and it adds a branch to the hottest loop in the
service path. Reverting also returns the tree to the exact state that campaign
`dec07` validated (6,866.1 µs, 0.888× production), so the ratchet stays on a
configuration with a decision-quality number behind it.

**Re-testable** on any workload whose routing produces more singleton rows —
lower `topk`, or heavier skew. It is five lines and the correctness argument is
workload-independent.

## Primitives

Nothing added. The argument that made this change safe — "a group of size one
needs no completion protocol" — is the third time this campaign that a
degenerate case of the group-completion protocol had to be re-derived by hand
(exp_14 for the probe, exp_14 for the claim, exp_16 for the arrival). A
`counter.cuh` that owned group completion would specialise all three
automatically from the group size. Seventh convergent finding.
