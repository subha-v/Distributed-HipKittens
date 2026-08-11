# exp_20 — Tier 1: characterize the 815 µs of M7 interference

## The target

At `C=64`, M7 costs **2,835 µs** under CTA specialization against **2,020 µs**
for a matched-CTA-count control (mode 0: reserve the same 64 CTAs, have them do
nothing). The 434 µs of that gap attributable to running M7 on 192 instead of
254 CTAs is capacity; the remaining **815 µs is interference** — the same 192
compute CTAs, damaged by what the pool is doing next to them.

Ruled out already: MFMA issue-slot contention (one block per CU, never
co-resident), cache capacity/pollution (exp_17, `nt` provably in the ISA, null),
atomics (≤235 µs, exp_14/16), peak-bandwidth saturation (~296 GB/s of 8 TB/s
HBM; ~111 GB/s of ~537 GB/s egress). 36% is per-XCD L2 (exp_08). The mechanism
behind the rest is unknown.

## The three questions

- **E1 — rate or volume?** Pace the pusher at constant total bytes. If M7
  recovers as the pool is slowed, the interference is a queueing/contention
  effect and a scheduler can find an optimum. If it does not recover at any
  pacing, only moving fewer bytes helps and every pacing/scheduling idea in
  Tier 3 is dead.
- **E2 — read side, peer write, or fabric?** Four diagnostic pools at matched
  `C`, differing only in which half of the copy they perform and whether the
  destination is a peer or this rank.
- **E3 — occupied but silent.** A pool that spins on LDS alone: CTAs consumed,
  instructions issued, no memory request. Separates CU occupancy from traffic.

## Pre-registered expectations (from ABLATION_QUEUE.md Tier 1)

| # | expectation |
|---|---|
| E1 | **rate-driven**, because the pool bursts hard and the fabric and L2 request queues are the suspected resource |
| E2 | **the local read dominates**, because it competes with M7's weight streaming in the same HBM/LLC path, whereas the peer write leaves the die |
| E3 | **indistinguishable from the idle pool** — nothing about merely occupying a CU matters, the whole effect is memory traffic. A surprise here would overturn the A7 strike |

## Method

Three new `mode` values. `mode` is 8 bits and the host parses it as a plain
int, so this needs no descriptor change, no host-ABI change and no harness edit
— exactly how mode 3 was added. Diagnostic magnitudes are carried in existing
config fields, reinterpreted inside the diagnostic mode only.

| mode | what it is | field reinterpretation |
|---|---|---|
| 4 | mode 2 + a tunable `s_sleep` delay in the push path | `flush_rows` → `1 + pacing units`; the real flag-batch depth is pinned at 16 so pacing is the only variable |
| 5 | mode 0 + a synthetic traffic generator on the reserved pool | `g` → variant (1 read+write, 2 read-only, 4 write-only); `pull_fallback` → write local slots instead of a peer's; `flush_rows` → active window in units of 100 µs |
| 6 | mode 0 + an LDS-only spin on the reserved pool | `flush_rows` → active window in units of 100 µs |

**All three are correctness-preserving and run the full gate ladder.** Modes 5
and 6 keep mode 0's parity M7.5 publication and mode 0's parity M8 remote-`part`
pull, which is what makes `K0P6_D_MPS_SLOTS` dead memory that a traffic
generator may write into freely. The generator only *reads* `part`.

**Matched-CTA discipline.** Every diagnostic is compared against **mode 0 at
the same `C`**, measured **in the same build**, never against the homogeneous
kernel. The mode-2 ratchet point is also re-measured in the diagnostic build so
the +16 B/lane of scratch the diagnostic code costs cannot be mistaken for a
result.

**Window sizing.** The mode 5/6 window is set to 1,600 µs, strictly below the
mode-0 M7 of 2,020 µs, so the diagnostic pool never delays the M7.5 rendezvous
and no downstream bounded wait is put at risk. Every variant therefore occupies
the *same* wall-clock window regardless of its throughput, which is the right
control for "what does running this traffic class flat out next to M7 cost".

## Ladders

1. **E20pace** — mode 2 reference, mode 0 reference, then mode 4 at pacing
   0/1/2/4/8/16/32/63 units. One unit is `s_sleep 4` = 256 core clocks ≈ 0.12 µs
   against a measured ≈2.1 µs of service time per push. Pacing 0 must reproduce
   the mode-2 reference — that is the built-in self-check.
2. **E20attr** — mode 5 at (read+peer-write, read-only, peer-write-only,
   read+local-write, local-write-only) and mode 6, all at `C=64`,
   window 1,600 µs.

Screening (1 process, 1 warmup, 1 timed) is used for shape; it tracked
campaigns to 0.1–2.6% on three separate checks. Any conclusion resting on a
sub-5% delta gets re-run.
