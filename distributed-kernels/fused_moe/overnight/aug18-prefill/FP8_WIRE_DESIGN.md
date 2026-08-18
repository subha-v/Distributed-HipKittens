# fp8-on-wire combine + source-side pre-reduce (M15 prefill megakernel)

**Status:** design only. No kernel code changed. Every ISA/resource number in §4
and §7 was measured this session on the gfx950 node (ROCm 7.2.4, CPU-only
compiles, no GPU workload launched); every timing number is either a cited
banked measurement or an explicitly-labelled model.

**Target:** `distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`
(entry `k0pf6gm_m15_mega`, 256×256, world 8, TOPK 8) and its phase-2 body
`n2_phase2_gm_mps.cpp`.

**Headline of the design:** the fp8 lever is **not reachable from the mode-12
remote-RMW epilogue** — gfx950 has no fp8 remote read-modify-write, and
quantizing inside the epilogue is forbidden by the occupancy law (§4). It is
reachable only on top of the already-coded-but-never-measured **staged arm**
(`K0P6_M15_STAGED=1`), where the epilogue folds partials into a *local* bf16
stage and carrier CTAs push folded rows as posted stores. On that base,
fp8-on-wire is a **pure push-pass and M8 change with literally zero delta to
the M7 epilogue instruction stream** (proved in §4: 282 `pk_add_bf16`, 96
`vmcnt(4)`, 180 `v_mfma` identical across both builds). Source-side pre-reduce
is not an additional feature; it *is* the staged arm, and it is the thing that
makes an fp8 slab well-defined at all.

**Headline blocker found while measuring:** `K0P6_M15_STAGED=1` today compiles
to **96 `scratch_load_dword` + 97 `s_waitcnt vmcnt(0)` sites within 24
instructions of the remote atomic** (default build: 0 and 1). That is exp_24's
LDS-base-eviction pathology and exp_38's throttle annihilation, both live, in
the arm this whole program depends on. It must be cleared before any A/B is
run; five bisect steps did not clear it (§4.4). This is P0.

---

## 0. The measured baseline this is priced against

| quantity | value | source |
|---|---:|---|
| M15 ratchet (C=24) | 5,848.5 µs = 0.7589× production | OVERLAP_ABSTRACTIONS.md §evidence |
| M15 (C=16) | 6,292.4 µs = 0.8165× | same |
| M7 GEMM-2 total | 2,701.84 ± 23.74 µs (46.2 % of interior) | exp_33 |
| — GEMM proper (proxy) | ~1,804–1,885 µs | exp_33 |
| — **epilogue surcharge (proxy)** | **~817–898 µs** | exp_33 |
| combine M8/M9 | 324.21 ± 21.12 µs (σ floor: distrust deltas < 60 µs) | exp_33 |
| production combine stage | 1,356.02 µs rank-max (16.5 % of its stage sum, 17.6 % of its p50) | exp_33 §4 |
| depth-4 vmcnt throttle | −615 µs; depth 16/32 is a cliff | exp_24 |
| unbounded remote RMW | +211 µs vs no payload at all | exp_25 |
| MORI EP8 @4096 tok, MI350X | combine bf16 **366 GB/s**, combine `fp8_direct_cast` **642 GB/s effective**, dispatch fp8 342 GB/s | M21_OVERLAP_DIRECTION.md §2 |
| per-link / egress ceiling | 54.9 GB/s per link, **349–355 GB/s egress**, knee 8 CTAs/link | OVERLAP_ABSTRACTIONS.md §5 |
| op-class issue ladder | posted store **0.127 µs/op** vs remote RMW **0.571 µs/op** | m15b header, OVERLAP_ABSTRACTIONS.md §2 |

The parent brief quotes "combine ≈ 20.5 % of production's region"; the banked
exp_33 stamp is 16.5 % of the production stage sum / 17.6 % of its p50. The
difference is instrument (rank-max per stage vs p50) and does not change any
conclusion below — I use the µs, not the percentage.

### Routing arithmetic (TOPK=8 distinct experts of 256, E=32/rank, world=8)

`P(rank not hit) = C(224,8)/C(256,8) = 0.338225`, so

| quantity | value |
|---|---:|
| E[distinct destination ranks / token] | **5.2942** (brief says 5.31; within model spread) |
| same-rank expert multiplicity | 8 / 5.2942 = **1.5111** |
| E[**remote** distinct ranks / token] | 5.2942 − 0.661775 = **4.6324** |
| E[**remote** (token, expert) partials / token] | 8 × 7/8 = **7.000** |
| fold factor available at the source | 7.000 / 4.6324 = **1.511×** |

The m15b header claims "~1.44× fewer remote bytes"; the exact figure under the
uniform MoK router is 1.511×. Under the real grouped/run-correlated router the
fold factor is *larger* (same-prompt tokens route alike ⇒ higher same-rank
multiplicity), which is the one place this design gets better under production
routing rather than worse.

---

## 1. Wire format

### 1.1 What the wire carries today

`n2_phase2_gm_mps.cpp:236` — one `accumulate_peer_bf162` per (row, dword) =
`global_atomic_pk_add_bf16`, 4 B covering 2 bf16 elements, one RMW stream per
output row, throttled at `vmcnt(4)` (`:240` → `throttle_epilogue_rmw`).

Per (token, expert) partial: **7168 elements × 2 B = 14,336 B**, emitted as
3,584 four-byte remote RMWs. There is no scale, no header, no padding.

### 1.2 Proposed format — fp8 e4m3 + per-128 fp32 scales, inline tail

Mirror the M1 dispatch quantizer exactly. `hkp_quant.hpp`
(`fp8_e4m3_group_quantizer<128>::quantize_chunk16`) is the one implementation
and its numerics are frozen:

- group = **exactly 128 elements** = 8 consecutive lanes × 16 elements;
- amax by `__shfl_xor` over masks 1/2/4 (never leaves the 8-lane segment);
- `scale = max(amax, 1e-6f) / 448.0f` — **the epsilon is not optional**
  (empty/zero rows give amax 0 ⇒ 448/0 = NaN);
- `quantize: clamp(x/scale, ±448)`, `__HIP_SATFINITE`, `__HIP_E4M3`.

The M15 mega already open-codes this identical math for dispatch at
`k0pf6gm_device_tile_m15.hip:1058-1079`. The combine push reuses
`hkp::fp8_e4m3_group_quantizer<128>` verbatim, which is why "mirror M1" is
literal here and not aspirational.

Row of 7168 elements ⇒ **56 groups**, which is exactly the `K0P6_NG = 56` group
count the kernel already uses for fc2 scales — no new constant class.

**Scales live inline, in the tail of the slab row, inside the existing
symmetric `slots` buffer.** Layout:

```
fp8 slab row (K0P6_M15_FP8_ROW = 7,392 B, 16-B aligned = 462 × 16)
  [    0 ..  7168)   7,168 B   payload, e4m3, element i at byte i
  [ 7168 ..  7392)     224 B   56 × fp32 group scales, scale[g] at 7168 + 4g
```

`slots` is addressed as `unsigned short*` today at stride 7168 elements; under
the fp8 arm it is addressed as `unsigned char*` at stride 7,392 B.

### 1.3 Bytes per row, before and after

| format | payload | scales | row bytes | vs bf16 | remote bytes/rank/epoch* |
|---|---:|---:|---:|---:|---:|
| bf16 RMW (today) | 14,336 | — | 14,336 | 1.000× | **411.04 MB** (28,672 partials) |
| bf16 staged (m15b) | 14,336 | — | 14,336 | 1.000× | **272.03 MB** (18,975 rows) |
| **fp8 e4m3 + 56 fp32 scales** | 7,168 | 224 | **7,392** | **1.939×** | **140.26 MB** |
| fp8 e4m3 + 56 bf16 scales | 7,168 | 112 | 7,280 | 1.969× | 138.13 MB |
| fp8 `direct_cast` (MORI parity) | 7,168 | 0 | 7,168 | 2.000× | 136.01 MB |

\* T = 4096 tokens/rank; RMW row = 7.000 remote partials/token, staged row =
4.6324 remote rows/token.

Compounding: **411.04 → 272.03 (fold, 1.511×) → 140.26 (format, 1.940×) =
2.931× fewer remote bytes than today.**

At the measured 349–355 GB/s egress ceiling (352 midpoint) that is
**1,168 µs → 773 µs → 398 µs** of pure wire time per rank per epoch. Note the
first number: the RMW combine's 1,168 µs of wire time is today *hidden inside*
M7's 2,702 µs; the staged numbers are, in the currently-coded M7.7 shape,
**not hidden** (§3.4). That asymmetry is the whole risk of this program and is
priced explicitly in §7.

### 1.4 Alternatives considered and rejected

| alternative | verdict |
|---|---|
| **fp8 e5m2** | 2 mantissa bits vs 3. Its extra exponent range is worthless once a per-128 scale exists. REJECTED. |
| **fp6 / fp4 (OCP MX)** | gfx950 has the conversion ops, but there is no MORI precedent for combine, the group metadata grows, and the accuracy budget (§5) has no room for a 2-bit mantissa summed over 5.3 slabs. REJECTED for v1; note as a follow-on if fp8 lands green. |
| **bf16 payload, 2:1 structured sparsity or delta coding** | no fabric support; decode cost lands in M8, the phase we are trying to make cheaper. REJECTED. |
| **scales in a separate symmetric array `[world][MAXTOK][56]` fp32 (7.3 MB)** | needs a new descriptor slot, a second posted store per row, and an *ordering* between payload and scale visibility that the single slab word does not provide. REJECTED — inline tail is strictly better. |
| **scales in a row *header* (bytes 0..224, payload after)** | breaks the 1024-B / 512-B chunk alignment of the M8 loop (`off = (c<<10)+(lane<<4)`), and misaligns the CSPLIT column boundary. REJECTED. |
| **bf16 scales (112 B, 1.969×)** | 0.6 % more bytes saved, but diverges from the frozen M1 fp32 scale idiom and from `scatter_group_scales`. Keep fp32; revisit only if the byte budget is ever the binding constraint. |
| **`direct_cast`, no scales at all** | MORI's shipped production mode and the source of the 642 GB/s number. 2.000× and zero scale plumbing, but flushes elements below the e4m3 subnormal floor (2⁻⁹ ≈ 1.95e-3) to zero with no per-group rescaling. Kept as `K0P6_M15_FP8WIRE=2`, an *ablation arm*, not the default (§5.3). |

---

## 2. Landing strategy: posted stores + owner-side dequant-reduce

`accumulate_peer_bf162` (`packet.cuh:202-211`) is documented as the *only*
accumulating transport, and its contract note is explicit: "4 bytes is the
hardware's only packed-fp16 atomic width on gfx950; there is no 16-byte atomic
form." There is likewise **no fp8 atomic form of any width**. So fp8 on the
wire forces posted stores, which forces a staging slab, which forces the owner
to do the reduce. That is what M8 already is.

### 2.1 Why this is not the falsified mode 16 (+75.8 µs)

Mode 16 deferred the combine **across the launch boundary** — K5 (consumer
placement) moved from "after-phase full grid" to "next launch", and it added
parity state to make that legal. OVERLAP_ABSTRACTIONS.md §2.1 records the
verdict: "the wait it escaped was mostly not on the critical path, and the
parity state it added was."

This design does not touch K5 at all:

| knob | today (mode 12) | mode 16 (falsified) | this design |
|---|---|---|---|
| K1 carrier | producer epilogue, remote RMW | producer epilogue, remote RMW | **producer epilogue → local stage; carrier CTAs → posted stores** |
| K2 producer order | nc-major | nc-major | nc-major (unchanged) |
| K3 certification | 2×8 slab epoch words | + parity generation | 2×8 slab epoch words (unchanged) |
| K4 flow control | vmcnt depth 4 | depth 4 | depth 4 on the (now local) RMW; §2.3 for the push |
| K5 consumer placement | M8, same launch | **next launch** | **M8, same launch** (unchanged) |
| *format* (new axis) | bf16 | bf16 | **fp8 e4m3 + per-128 scale** |

M8 already exists as a full reduce phase reading `fanout` slabs per token and
summing in fp32 (`k0pf6gm_device_tile_m15.hip:652-746`). Nothing is deferred,
no generation state is added, no epoch is put in flight. The only new state on
the whole path is 224 bytes of scale per slab row, inside a buffer that already
exists.

### 2.2 Quantified M8 work change

M8's `k0p6_m15_m8_batch` per element today: unpack bf16 (`k0p6_bf16_to_f32`, a
shift) + `acc += `. Under fp8: `__builtin_amdgcn_cvt_pk_f32_fp8` (2 elements per
instruction) + an FMA with the group scale, and one scalar scale load per lane
per chunk (4 distinct dwords per 64-lane warp — L1-resident).

| M8 term | bf16 today | fp8 | Δ |
|---|---:|---:|---|
| slab bytes read | 310.9 MB | 160.3 MB | **−48.4 %** |
| consume-and-zero bytes written | 310.9 MB | 160.3 MB | **−48.4 %** |
| unpack instructions | 8 shifts / 16 B | 4 `cvt_pk_f32_fp8` / 16 B | −4 VALU |
| accumulate | 8 `v_add_f32` | 8 `v_fma_f32` | 0 (same slot count) |
| scale loads | 0 | 1 dword / lane / chunk | +14 loads / row / lane |
| accumulator registers | `acc[NT][8]` | **`acc[NT][8]` — unchanged**, see §6.4 | 0 |

Rows read per rank per epoch = 4096 × 5.2942 = **21,685**. exp_29 measured the
combine consumer at 1.57 TB/s (20 % of HBM, latency-bound). Halving 621.8 MB of
read+zero traffic to 320.6 MB is worth ~192 µs *of byte time*, but M8's measured
phase total is only 324.2 µs (because M15 already consumes the front half inside
slab 1's shadow), so the achievable phase delta is bounded by the phase itself.
**Honest claim: M8 −100 to −160 µs, and exp_33's 60 µs resolution floor means
this must be judged on the coupled M7+combine block (σ = 10.16 µs), never on the
combine stamp alone.**

M8 gets *less* work, not more. This is the structural reason the mode-16 analogy
does not apply: mode 16 moved a phase and paid for the move; here the phase
stays put and its input shrinks by half.

### 2.3 Certification: what proves a staging slab complete

Nothing new is needed — the staged arm's existing protocol certifies the
*pushes*, which is exactly the right event.

Chain, as coded (`k0pf6gm_device_tile_m15.hip:1694-1730` and `:1743-1866`):

1. **Per-task drain** (`N2GM_TASK_DONE_DRAIN_HOOK`, deferred: task *t*−1's
   `s_waitcnt vmcnt(0)` is paid at task *t*'s head, `:373-388`) — under the
   staged arm these are *local* stage RMWs, so the drain is an L2 event, not a
   fabric round trip.
2. **Slab rendezvous** (`:1696-1702`): `release_cta_payload_system()` + grid
   barrier, once per slab. After R1 for slab 1, every local stage row is
   complete grid-wide.
3. **M7.7 push pass** (`:1750-1819`): ticketed front halves + grid-strided back
   halves, `store_peer_packets` into peer slots.
4. **`producer_drain_release<system>()` + R2 grid barrier** (`:1820-1826`).
5. **One leader publishes both `(rank, slab)` epoch words to all 8 peers**
   (`:1848-1865`), `release_signal_batch_system()` first.
6. **M8** polls 2×8 words (`:1878-1887`) then `acquire_payload_system()` inside
   the batch body (`:689`).

The fp8 arm changes **only step 3's payload format**. The certification is
byte-identical and the `#if !K0P6_M15_STAGED` split at `:1706` already routes
the RMW arm's earlier publication away.

Two fp8-specific ordering notes:

- **Scales are inside the same posted store range as the payload they scale**
  (§1.2), and a group's scale is written by the same lane group that quantizes
  it, in the same `store_peer_packets`-class loop, before the same
  `producer_drain_release`. There is therefore no payload-vs-scale ordering
  question at all. This is the entire reason for the inline-tail choice.
- **The CSPLIT front/back boundary is a work split, not a correctness
  boundary, in the staged arm.** In the RMW arm CSPLIT=7 (column 3,584 = nc 8)
  must coincide with the slab column cut because the pool consumes front halves
  mid-slab-1. The staged arm has no mid-slab consumer (`#if !K0P6_M15_STAGED`
  at `:1566` compiles the pool sweep out) and M8 runs static stripes after R2
  (`:1932-1944`), so the split is free. §6.4 keeps it at column 3,584 anyway,
  because doing so lets the fp8 M8 chunk bookkeeping stay textually identical.

### 2.4 The depth-4 throttle law applied to posted stores

The law (K4) is about *injection bounding under co-resident compute*: unbounded
remote RMW measured +211 µs worse than no payload at all; depth 4 measured
−615 µs; depth 16/32 is a cliff. Three consequences here, in order of
confidence:

1. **The epilogue keeps depth 4 unchanged.** Under the staged arm the RMW
   target is local, but the instruction stream is bit-identical (M15-DELTA (B)
   in `n2_phase2_gm_mps.cpp:380-388`, confirmed in §4.1: 96 `vmcnt(4)` sites in
   both builds). Do not "optimise away" the throttle for local RMWs — that is
   an independent variable and belongs in its own experiment.
2. **`store_peer_packets` is already self-throttled and must stay that way.**
   `packet.cuh:48-58` is a *dependent load→store chain* under `#pragma unroll
   1`, and the header states it plainly: "each lane keeps exactly ONE memory
   operation in flight." The new quantize-push loop inserts a `__shfl_xor`
   amax reduction between the load and the store. **Keep `#pragma unroll 1` on
   the row loop**, or the compiler will software-pipeline the loads and silently
   raise the in-flight count — the exp_38 failure mode with a different
   trigger. Assert it from the ISA (§6.6 gate 4).
3. **Do not add an explicit `vmcnt(N)` bound to the push in v1.** In the coded
   M7.7 shape the push runs after R1 with no GEMM in flight, so there is no
   co-resident compute for congestion to damage. If and only if the push is
   later moved into slab 1's shadow (§3.5) does a bound become mandatory, and
   then it must use the exp_24 four-instantiation + SGPR-mask-plan idiom
   (`n2_phase2_gm_mps.cpp:115-178`), never a runtime depth: `s_waitcnt vmcnt(N)`
   encodes N in simm16 and has no register form.

Two MORI disciplines that transfer and are cheap:

- **Link interleaving.** M7.7's ticket loop walks `r = q*8 .. q*8+8`
  (`:1781`), and `owner = r / MAXTOK` (`:1782`) — so all 8 rows of a quantum
  share one owner and hit **one xGMI link**. That is precisely the pathology
  MORI documents (consecutive same-destination traffic drives 2–3 of 7 links;
  their LDS round-robin fix measured 822 → 497.7 µs, 65 %). Fix: rotate the
  quantum→row map so a quantum's 8 rows hit 8 different owners. Plan-side, zero
  protocol, one expression — the m17 lesson. Ship behind its own macro
  `K0P6_M15_PUSH_RR` so it stays a separable variable.
- **`s_sleep` backoff** already exists as `pace_delay` /
  `poll_backoff_delay` (`moe_mps_adapter.cuh:891-923`); the ticket loop's
  `fetch_add_relaxed` spin should use it. Separate one-line item, separate
  macro.

### 2.5 The no-stage alternative, and why it is rejected

One could keep the RMW-free epilogue *without* a stage by widening the slab
index to `(computing rank, receive row, expert-slot)` so every partial gets its
own fp8 slab and the fold happens only in M8.

Depth needed: `k_p | k_p ≥ 1` where `k_p ~ Binomial(8, 1/8)` ⇒
`P(1)=0.598, P(2)=0.299, P(≥3)=0.103`. Depth 2 covers 89.7 %; the 10.3 % tail
needs an escape hatch, i.e. **two formats on the wire plus a per-slab depth
counter plus an fp8-vs-bf16 discriminator, all decided inside the epilogue**.
It also still requires quantize-in-epilogue, which §4.3 rejects on its own.
Memory: depth 2 in fp8 = 484 MB, i.e. no better than today's bf16 slots.
**REJECTED**: worse on protocol, worse on registers, no better on memory.

---

## 3. Source-side pre-reduce

### 3.1 Can same-token partials from different local experts meet in a CTA? No.

Read the phase-2 task structure (`n2_phase2_gm_mps.cpp:479-500`):

```
const int tile = task / kNChunksP2;      // 16 n-chunks per tile
const int b0   = n2gm_tile_b0(tile);     // first 32-block of the tile
const int e    = sorted_eid[b0];         // ONE expert for the whole task
```

A tile is `kGM = 3` **consecutive same-expert** 32-blocks (exp_59 G-stacking,
`tile_desc` packs `(b0<<4)|gcount`), and `load_w2` (`:598-607`) uses that single
`ew` for every fragment of the task. So:

- one CTA touches **exactly one expert per task**;
- two partials of the same token from two different local experts live in two
  different sorted blocks, in **different tiles by construction** (tiles are
  expert-aligned), hence in different tasks, hence generally on different CTAs
  and certainly at different times;
- the epilogue's LDS scratch `xp_lds[kWaves][kBlockM][32]` is per-wave transpose
  scratch that is reused every sub-block (`:675-679`, serialized by
  `wave_barrier`) — it holds one tile's rows, never a cross-task accumulator.

**Conclusion: an LDS pre-reduce is structurally impossible without cross-task
buffering.** Pre-reduce requires a memory-resident local accumulator with the
lifetime of the whole M7 phase. That accumulator is the stage.

### 3.2 The stage, and why the pre-reduce is then free

`K0P6_M15_STAGED` (`k0pf6gm_device_tile_m15.hip:65-70`, `:239-244`,
`:440-454`) already implements exactly this: descriptor slot 63 holds a
host-zeroed **local** `[world*MAXTOK][7168]` bf16 stage, and
`N2GM_M7TAB_FILL` points the epilogue's peer table at it:

```
m7tab[tid] = m15_stage + tid * MAXTOK * kHidden * 2
```

so `tab[xr>>sh] + (xr & mask)*kHidden*2` — the donor's exact address
expression — lands on `stage[receive_row]`. **Only the pointer values change;
the epilogue's instruction stream is bit-identical** (verified in §4.1).

Because the stage is indexed by *receive row* `r = (source, pos)`, every local
expert that the token hits accumulates into the same `stage[r]`. The 1.511
same-rank partials fold **for free**, by the same `global_atomic_pk_add_bf16`
that already runs, with the fabric taken out of the loop.

The stage must stay **bf16**: it is an accumulator, and fp8 has no RMW. The
quantization happens exactly once, on the *read* side of the stage, in the push
pass. This is also why the design adds nothing to the epilogue's registers.

### 3.3 Does this collide with the carrying-pool falsification?

The falsification is: "CTA pools that CARRY payload (+340 µs); only pools
CONSUMING a certified immutable prefix in the producer's shadow win", grounded
in exp_22's measurement that payload rides ~free on producer stores
(concurrent/isolated = 0.9951), so a carrier pool buys nothing and pays the
capacity tax.

**As coded, M7.7 is not a pool.** It runs after both slabs, with **all 256
CTAs** participating (`:1772-1819`) — no CTAs are reserved, no capacity tax is
paid, and the C knob is untouched. The falsification literally does not apply.

The *premise* of the falsification also fails here: exp_22's "payload rides free
on producer stores" assumed the producer's store is already remote. Under the
staged arm the producer's store is local, so the remote bytes have no producer
to ride on and someone must carry them. That is a premise failure, not
counter-evidence — flag it as such and do not claim it as support.

**What the falsification does forbid** is §3.5's overlap variant. Keep them
separate arms.

### 3.4 The honest cost of the stage

The m15b header claims "~1.44× fewer remote bytes" and stops there. The full
ledger, per rank per epoch, is:

| term | RMW arm | staged arm | Δ |
|---|---:|---:|---|
| remote egress | 411.04 MB | 272.03 MB (bf16) / 140.26 MB (fp8) | −139 / −271 MB |
| epilogue RMW target | 411.0 MB remote + 58.8 MB local | **469.8 MB local** | fabric → L2/HBM |
| stage read (push pass) | 0 | **+310.9 MB local read** | new |
| stage consume-and-zero | 0 | **+310.9 MB local write** | new |
| self-rank push (owner == cur) | 0 | +20.0 MB local | new |
| stage HBM footprint | 0 | **+469.8 MB** resident | new |

**+642 MB of new local traffic** — 80 µs at HBM peak, ~409 µs at the 1.57 TB/s
rate exp_29 actually measured for this access class. Plus 470 MB of HBM
residency, which must be checked against the M20 slot-pool's ~1 GB budget if
those arms are ever combined (they are mutually exclusive today: both claim
descriptor slot 63, `:102-104`).

And the structural cost: **in the coded M7.7 shape the push is a serial tail**,
not overlapped with any GEMM. 272.03 MB at 352 GB/s = 773 µs of serial wire
time (bf16), 398 µs (fp8). The RMW arm's 1,168 µs of wire time is hidden inside
M7's 2,702 µs; the staged arm's is not. **This, not accuracy and not registers,
is the thing that decides whether the program wins.** It is why §7 predicts A1
(staged bf16) may well *lose* and A2 (staged fp8) wins mainly by halving a tail
that A1 created.

### 3.5 The overlap variant (separate arm, do not bundle)

Slab 0's stage front halves are certified complete at the slab-0 rendezvous, so
they can be pushed during slab 1 on the same dynamic wave tickets the RMW arm's
pool uses for combine sweeps (`:1567-1690`). That would put ~half the push under
the slab-1 GEMM shadow and cut the serial tail roughly in half again.

It is a **carrier pool**, and the corpus says carrier pools lose. Ship it only
as a follow-on arm, after A1/A2 have separated the fold and format effects, and
with the depth bound of §2.4(3) mandatory. Pre-register it as "expected to lose
by the exp_22 law; measured to test whether the premise failure of §3.3 changes
the sign."

---

## 4. Register / LDS budget — measured, not estimated

All numbers from this session on the node: ROCm 7.2.4,
`hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 -DKITTENS_CDNA4
-DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1
-DK0P6GM_G=3 -DN2GM_G=3 -Rpass-analysis=kernel-resource-usage`, entry
`k0pf6gm_m15_mega`, disassembly via `clang-offload-bundler --unbundle` +
`llvm-objdump -d --mcpu=gfx950`.

### 4.1 Resource tuple and ISA counters

| build | SGPR | VGPR | AGPR | scratch B/lane | SGPR spill | VGPR spill | LDS B | occ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M15 default** (mode 12, RMW) | 106 | 256 | 256 | 72 | 137 | 17 | 155,496 | 1 |
| **`K0P6_M15_STAGED=1`** | 106 | 256 | 256 | 80 | 117 | 21 | 155,504 | 1 |

| ISA counter | default | staged | verdict |
|---|---:|---:|---|
| `flat_atomic_pk_add_bf16` | **282** | **282** | identical (matches the exp_24 gate value) |
| `s_waitcnt vmcnt(4)` | **96** | **96** | identical — the throttle survives |
| `s_waitcnt vmcnt(8)` | 96 | 96 | identical |
| `v_mfma*` | **180** | **180** | identical (matches the exp_24 gate value) |
| scratch ops inside either MFMA span | **0** | **0** | clean |
| `global/flat_store_dwordx4` | 90 | 74 | staged replaces 16 stores with the push path |

**This is the load-bearing result of §4:** the staged arm's M7 epilogue emits
the same 282 atomics under the same 96 depth-4 throttle sites and the same 180
MFMAs as the shipped ratchet. M15-DELTA (B)'s claim ("only the pointer VALUES
change") is verified from the ISA, and therefore **fp8-on-wire, which touches
only the push pass and M8, adds exactly zero to the M7 epilogue's register
pressure.**

Note on the brief's figures: "438 VGPR / 182 AGPR / 0 spill / LDS 120,200
(exp_62)" describes the amd-master donor `k0pf6gm_mega`, a different
translation unit. The aug11 gate tuple for *this* kernel family is
`SGPR 104 / VGPR 256 / AGPR 256 / scratch 128 / LDS 155,428 / MFMA 180 /
pk_add_bf16 282`; my default build reproduces MFMA and pk_add_bf16 exactly and
differs on SGPR/scratch/LDS by compiler-version drift. Use the same-session
paired build as the control, never a quoted historical tuple.

### 4.2 What the push-side quantizer costs

Nothing in the epilogue. In M7.7, per lane per 16-element chunk:

| resource | cost |
|---|---|
| VGPR | `float f[16]` + amax + scale ≈ 18 live, in a phase whose only competition is a pointer walk. M7.7 is not register-constrained (it sits after both MFMA spans). |
| LDS | **0** — the amax is `__shfl_xor` over 8 lanes, register-only |
| new `__shared__` | none (M7.7 already declares `__shared__ int s_q2`) |
| ALU | 16 `v_max3_f32`-class + 3 shfl + 1 rcp/mul + 16 `cvt_f32→fp8` per 16 elements |
| memory | one 32-B load (bf16) → one 16-B store (fp8) + 1 scalar scale store per 8 lanes |

Pre-registered gate: the FP8WIRE build's tuple must match the STAGED build's on
VGPR/AGPR/occupancy and its `pk_add_bf16`/`vmcnt(4)`/`v_mfma` counts must be
282/96/180 exactly.

### 4.3 Why quantize-in-epilogue is rejected (the counted version)

The brief asks what fp8 quantize *in the epilogue* would add. Counted from
`epilogue_write` (`n2_phase2_gm_mps.cpp:180-265`):

- **Geometry does not fit.** Phase 2's half-wave holds `dcol = lane & 31` = one
  dword = 2 adjacent columns of **one** row, iterating over 16 different rows
  (`row = 2*i + rowh`). A 128-element quant group along the column axis would
  span 64 lanes across both `rowh` halves (different rows) — impossible. The
  largest column group available inside a half-wave is **64 columns**, and the
  wave's column span is `kWaveCols2 = 112` (`col_base = 448*nc + 112*wv`), which
  is not 64-aligned for odd `wv`. So the group size would have to be a ragged
  64/48 pair at wave-dependent offsets — a wire format the receiver cannot
  address with a uniform expression.
- **Cross-lane cost.** A 32-lane amax tree is 5 `shfl_xor` + 5 `v_max` per row,
  ×16 rows ×2 `epilogue_write` calls ×3 sub-blocks = **960 extra cross-lane ops
  per task**, inserted between the `ds_read` and the remote op, i.e. exactly the
  region OVERLAP_ABSTRACTIONS.md §2.3 calls "a liability".
- **Register cost.** +2 live VGPRs (amax accumulator, scale) across the
  16-iteration unrolled body, at a measured 256 VGPR / 256 AGPR ceiling with 17
  VGPR spills already present. exp_38's precedent: a silent regalloc cliff there
  turned depth-4 into depth-1 for **+727 µs with all gates green**. exp_24's
  precedent: one extra live integer evicted the peer table's LDS base and put a
  `scratch_load_dword` five instructions ahead of every remote atomic, 96 sites.
- **Fabric cost.** The output op becomes a **2-byte posted store** (one bf16
  dword quantizes to 2 fp8 bytes) — sub-dword remote stores are the worst op
  class on this fabric and destroy the 128-B-per-half-wave contiguity the donor
  comment at `:222-224` says is what exp_21 measures.

**REJECTED on four independent grounds.** The design does not do it, and no
variant of this program should.

### 4.4 ⚠️ P0 BLOCKER: the staged arm is already in the exp_24 spill state

Measured, same session, same compiler, paired builds:

| gate | default | **staged** |
|---|---:|---:|
| total `scratch_*` ops | 21 | **122** |
| `scratch_*` within 24 instructions of a remote atomic | **0** | **96** |
| `s_waitcnt vmcnt(0)` within 24 instructions of a remote atomic | 1 | **97** |
| scratch ops inside either MFMA span | 0 | 0 |

The 101 new ops are all `scratch_load_dword`, and the pattern at every site is:

```
scratch_load_dword v7, off, off        <- the m7tab LDS base, reloaded
v_accvgpr_read_b32 v6, a236
v_lshrrev_b32_e32  v6, v6, v223        <- xr >> maxtok_sh
s_waitcnt vmcnt(0)                     <- FULL DRAIN, forced by the scratch load
v_lshl_add_u32     v6, v6, 3, v7
ds_read_b64        v[18:19], v6        <- m7tab[owner]
...
flat_atomic_pk_add_bf16 v[18:19], v6
```

This is verbatim the pathology `n2_phase2_gm_mps.cpp:345-349` documents ("the
allocator evicted the peer table's LDS base and reloaded it from scratch once
per remote atomic (96 static sites)"), **and** it is worse than exp_38's cliff:
the `s_waitcnt vmcnt(0)` is a consequence of the scratch load (scratch counts in
`vmcnt`), so the negotiated depth-4 throttle becomes **depth 0** — a full VMEM
drain per remote accumulate. One root cause, two symptoms.

Bisect performed this session (each an independent paired build + disassembly);
none cleared it:

1. `__builtin_amdgcn_readfirstlane` on both descriptor reads in
   `N2GM_M7TAB_FILL` → 96/97 unchanged.
2. Replace the fill's 64-bit multiply with a host-baked 8-entry pointer table
   loaded from the stage header → 96/97 unchanged.
3. Compile the entire M7.7 block out (`K0P6_M15_STAGED=1` minus the push pass)
   → 96/97 unchanged (LDS returns to 155,496).
4. Restore the `#if !K0P6_M15_STAGED` pool sweep and slab publication under the
   staged build → 96/97 unchanged.
5. (control) default build → 0/1.

So it is a whole-function allocation effect of taking the
`#ifndef N2GM_M7TAB_FILL` branch — not of any single statement inside it. The
remaining hypothesis to test is that the **default** fill's `m7_slot_off`
computation and `m7sym` loads are what *keep* the allocator from evicting the
LDS base (exp_24's own note: folding `slot_off` "frees the AGPR pair that held
it"), i.e. the staged fill should be rewritten as the default fill's token
stream with only the per-owner base substituted, rather than as a wholesale
replacement — the "one textual copy" discipline the corpus keeps re-proving.

**Consequence for this program: any A/B of the staged arm today measures this
bug, not the mechanism.** Clearing it to `scratch_near_atomic == 0` and
`vmcnt0_near_atomic <= 1` is item 0 of §6 and a hard gate on every arm.

---

## 5. Accuracy

### 5.1 Where error enters today

1. M7 accumulates each (token, expert) partial in **fp32** across all 16 K128
   groups, then multiplies by `sorted_w` and packs to **bf16** (8 mantissa bits,
   `:198-202`).
2. `global_atomic_pk_add_bf16` accumulates the ~1.511 same-rank partials **in
   bf16** at the destination.
3. M8 sums the 5.294 slabs in **fp32** (`acc[t][e] += ...`, `:720`) and packs
   the result to bf16 RNE (`:742`).

Measured: MoK global L1 relative error **0.0155** against a blocking gate of
**0.1** (`BENCHMARKING.md:48`). The strict k0 `rel_L2 ≤ 0.01` is recorded as a
**nonblocking diagnostic**, never as a gate (`BENCHMARKING.md:49`).

### 5.2 Where fp8 adds error, and how much

The staged arm changes nothing above: the fold is still a bf16 RMW, just local.
fp8 inserts **one** new rounding, on the *folded* value, in the push pass:

- e4m3 has 3 explicit mantissa bits ⇒ relative quantization step 2⁻⁴ = 6.25 %
  at the bottom of a binade; round-to-nearest RMS relative error
  ≈ 2⁻⁴ / (2√3) = **1.80 %** per element.
- Per-128 scaling with `scale = amax/448` gives an intra-group dynamic range of
  448 × 2⁹ ≈ 2.3 × 10⁵ (down to the e4m3 subnormal floor), so **no element of a
  group is scale-limited**. The 1e-6 epsilon covers the all-zero group.
- M8 then sums `n = 5.294` slabs in fp32. The quantization errors are
  independent across slabs, so for comparable slab magnitudes the relative error
  of the sum is `1.80 % / √n = ` **0.78 %**.

In quadrature with the existing 1.55 %:

```
sqrt(0.0155^2 + 0.0078^2) = 0.01735   -> 5.8x margin under the 0.1 gate
```

Pessimistic sensitivity (assume the fp8 term is 3× my estimate, 2.34 %, e.g.
because slab magnitudes are unequal so the √n averaging does not apply):

```
sqrt(0.0155^2 + 0.0234^2) = 0.0281    -> 3.6x margin
```

**Verdict: the MoK gate survives with 3.6–5.8× margin.** Pre-register 0.0174 as
the expected value and treat anything above 0.035 as a bug signal, not a
tolerance question. The strict k0 `rel_L2` diagnostic **will** move and is
nonblocking by protocol — record it, never tune to it, never relax it.

### 5.3 Precedent and the `direct_cast` arm

MORI-EP ships `fp8_direct_cast` on the wire in its production MI350X EP8
combine path — it is where the 642 GB/s effective number comes from
(M21_OVERLAP_DIRECTION.md §2). That is the precedent for fp8 combine being
production-acceptable in a DeepSeek-class MoE at this exact shape.

Our scaled variant is *strictly more accurate* than the shipped MORI mode:
`direct_cast` has no per-group rescaling, so any element below the e4m3
subnormal floor (≈1.95e-3) flushes to zero and anything above 448 saturates.
Gate-weighted MLP outputs sit comfortably inside that window on the MoK
synthetic, but the flush-to-zero tail is routing- and layer-dependent and has no
error bound. Ship scaled by default (`K0P6_M15_FP8WIRE=1`); keep `direct_cast`
as `=2` purely as a **MORI-parity ablation** that answers "how much of the 2.000×
vs 1.939× is worth having" and provides a like-for-like number against their
published envelope.

### 5.4 Poison and negative control

`K0_MOK_POISON_OUT` (exp_32's NaN poison) must be on. Its failure mode — "row
not written" — is exactly the failure mode a staged push can produce (a row
whose owner never received its slab), and it is the only instrument that catches
a silently-skipped `pos >= s_ns[owner]` row. Negative control must fire.

---

## 6. Implementation plan

Macro `K0P6_M15_FP8WIRE`, **default 0**. Values: 0 = off, 1 = e4m3 + per-128
fp32 scales, 2 = `direct_cast` (no scales). Requires `K0P6_M15_STAGED=1`.
Line anchors are current-HEAD; the file's own convention is ±30 drift, grep the
construct.

**`n2_phase2_gm_mps.cpp` is not touched. `moe_mps_adapter.cuh` is not touched.
No new descriptor slot. No host allocation change.**

### 6.0 Item 0 (P0, blocking) — clear the staged-arm spill

Fix `N2GM_M7TAB_FILL` (`k0pf6gm_device_tile_m15.hip:440-454`) until the paired
disassembly gate reads `scratch_near_atomic == 0` and
`vmcnt0_near_atomic <= 1`. Leading candidate (§4.4): rewrite the staged fill as
the *default* fill's token stream with only the per-owner base substituted,
introducing a narrow `N2GM_M7TAB_BASE` seam instead of replacing the whole
block. **2–4 h**, entirely CPU-side (build + `llvm-objdump` + the counting
script in §6.6). Nothing else in this plan may be measured until this is green.

### 6.1 Macro plumbing and constants — **0.5 h**

- `:63` — bump `K0P6_M15_SRC_REV 2` → `3` (JIT cache guard; confirm fresh
  `.hsaco` mtime with `stat -L`).
- `:65-70` — new block next to `K0P6_M15_STAGED`:
  ```
  #ifndef K0P6_M15_FP8WIRE
  #define K0P6_M15_FP8WIRE 0
  #endif
  #if K0P6_M15_FP8WIRE && !K0P6_M15_STAGED
  #error "K0P6_M15_FP8WIRE requires K0P6_M15_STAGED (fp8 has no remote RMW)"
  #endif
  ```
- `:172-179` — constants next to `K0P6_H`:
  ```
  #define K0P6_M15_FP8_NG  56      // 7168 / 128 groups per row (== K0P6_NG)
  #define K0P6_M15_FP8_SCB 224     // 56 fp32 scales
  #define K0P6_M15_FP8_ROW 7392    // 7168 payload + 224 scales, 462 x 16 B
  #define K0P6_M15_FP8_SPLIT 3584  // push/M8 front-back cut == column 3584
  ```
  Under `=2`, `FP8_SCB 0` / `FP8_ROW 7168`.
- `:799-819` entry guard — add
  `(size_t)world * MAXTOK * K0P6_M15_FP8_ROW > mps_slots_bytes(MAXTOK)` →
  `K0P6_MPS_ERR_CONFIG`. Capacity today: needed 242,221,056 B vs allocated
  469,762,048 B (`moe_host_abi.hpp:204-208`), 48 % headroom, **no host change**.

### 6.2 Push-side quantizer helper — **1.5 h**

New `__device__ __forceinline__` next to the M8 batch helper (`:647`), local to
this TU. Promote to `include/cdna4/ops/group/distributed/packet.cuh` only after
it measures — the corpus rule is that primitives ship with their verifier.

```
// One row: bf16 stage [byte_lo, byte_hi) of the bf16 row -> fp8 slab row.
// Group = 128 elements = 8 consecutive lanes x 16, hkp_quant.hpp verbatim.
__device__ __forceinline__ void k0p6_m15_push_fp8_span(
    unsigned char* __restrict__ dst_row,        // peer slab row base
    const unsigned short* __restrict__ src_row, // local bf16 stage row
    int chunk_lo, int chunk_hi,                 // 16-element chunks
    unsigned int tid, unsigned int threads);
```

Body, per iteration (`#pragma unroll 1` — §2.4(2)):
`uint4 q = hkp::fp8_e4m3_group_quantizer<128>::quantize_chunk16(src_row + 16*c,
lane, scale); *(uint4*)(dst_row + 16*c) = q; if ((lane & 7) == 0)
*(float*)(dst_row + 7168 + 4*(c >> 3)) = scale;`

Lane→chunk mapping check: with `c = chunk_lo + tid + i*threads` and
`threads = 256`, lanes 0..7 of a wave get 8 consecutive `c`, i.e. 128
contiguous elements, and all 8 share `c >> 3` — the `__shfl_xor` masks 1/2/4
tree stays inside the group and inside the wave. This is the same lane
discipline as M1's dispatch quantize (`:1050-1083`).

Under `K0P6_M15_FP8WIRE == 2`: skip the amax tree and the scale store, use
`scale = 1.0f`.

### 6.3 M7.7 push pass — **1.5 h**

`k0pf6gm_device_tile_m15.hip:1743-1866`, three edits, all inside
`#if K0P6_M15_FP8WIRE`:

- `:1765-1766` — `slots` becomes `unsigned char*`; row base
  `slots_b + ((size_t)cur*MAXTOK + pos) * K0P6_M15_FP8_ROW`.
- `:1793-1795` (front, ticketed) — replace `store_peer_packets(dst, src,
  CSPLIT*1024, tid, blockDim.x)` with
  `k0p6_m15_push_fp8_span(dst, (const unsigned short*)src, 0, 224, tid,
  blockDim.x)` (chunks 0..223 = elements 0..3583 = groups 0..27, whose scales
  land at bytes 7168..7280 — 16-B aligned).
- `:1815-1818` (back, grid-strided) — chunks 224..447 = groups 28..55.
- `:1834-1845` (consume-and-zero of the **local stage**) — unchanged; the stage
  stays bf16 at 14,336 B/row.

The slab certification (`producer_drain_release` + R2 + the leader publication,
`:1820-1865`) is untouched.

### 6.4 M8 fp8 batch sibling — **3 h**

`k0pf6gm_device_tile_m15.hip:652-746` gains
`k0p6_m15_m8_batch_fp8<NT, CLO, CHI>`, chosen at `:1937-1942` under the macro.
Deltas, all of which preserve the existing register shape:

- `pbase[t]` (`:681-684`) — byte arithmetic at stride `K0P6_M15_FP8_ROW`.
- **Chunk grain: 14 chunks of 512 B, `off = (c << 9) + (lane << 3)`** — i.e.
  8 fp8 bytes per lane instead of 16 bf16 bytes. This keeps `CHI - CLO`, the
  `CSPLIT = 7` boundary (= fp8 byte 3,584 = column 3,584) and the
  **`acc[NT][8]` accumulator array byte-identical**. A 16-B/lane form would
  cover 16 columns per lane and force `acc[NT][16]`, doubling M8's
  accumulators — rejected. A warp still issues 512 B contiguous per chunk.
- Group scale: `g = (c << 2) + (lane >> 4)`, `sc = *(const float*)(pb + 7168 +
  4*g)` — 4 distinct dwords per warp, L1-resident.
- Unpack: `__builtin_amdgcn_cvt_pk_f32_fp8` (2 elements/instruction), then
  `acc[t][e] = fmaf(sc, v, acc[t][e])`.
- Consume-and-zero (`:724-728`): zero 8 B (`uint2`) instead of 16 B. **Payload
  only** — a zero payload is zero under any scale, so a stale scale is inert.
  Document that inline, mode-12 slot-lifetime style.
- Output store (`:736-743`) unchanged: bf16 `uint4` at `orow + 2*off`.

### 6.5 Optional co-shipped one-liners (own macros, own arms) — **0.5 h**

- `K0P6_M15_PUSH_RR` — rotate the M7.7 quantum→row map so a quantum's 8 rows
  hit 8 owners (§2.4). Plan-side only.
- `K0P6_M15_PUSH_BACKOFF` — `pace_delay` in the M7.7 ticket spin
  (`moe_mps_adapter.cuh:891`).
- `K0P6_M15_PUSH_NT` — `store_peer_packets_streaming`'s `nt` hint
  (`packet.cuh:87`). Only relevant if the push ever overlaps GEMM (§3.5);
  exp_16 bounded payload cache contention at ~1,000 µs of the concurrent phase.

### 6.6 Build-gate checklist (every gate is blocking)

| # | gate | how |
|---|---|---|
| 0 | **staged spill cleared** | `scratch_*` within 24 instrs of `pk_add_bf16` == 0; `vmcnt(0)` within 24 == ≤1 (§4.4 script) |
| 1 | **default build byte-identical** | `K0P6_M15_FP8WIRE` and `K0P6_M15_STAGED` both 0 ⇒ `sha256` of the unbundled `.co` equals the pre-change build's |
| 2 | resource tuple | `-Rpass-analysis=kernel-resource-usage`: FP8WIRE build matches the STAGED build on VGPR/AGPR/occupancy; report SGPR, VGPR, AGPR, scratch B/lane, SGPR+VGPR spill, LDS, occupancy for **all three** builds |
| 3 | epilogue ISA identity | `pk_add_bf16` == 282, `vmcnt(4)` == 96, `v_mfma` == 180 in every arm |
| 4 | throttle survives, push self-throttles | zero scratch ops inside either MFMA span; the push row loop must show one dependent load→store chain per lane (no software-pipelined multi-load), i.e. `#pragma unroll 1` reached the ISA |
| 5 | store widths | payload lowers to `global_store_dwordx4` (16 B); **zero sub-dword payload stores**; scales as `global_store_dword` |
| 6 | slab capacity | entry-guard assertion (§6.1) fires on an undersized `slots` |
| 7 | MoK correctness | blocking, L1 rel ≤ 0.1; pre-registered expectation 0.0174; strict k0 `rel_L2` recorded nonblocking |
| 8 | negative control | must fail as required; NaN poison on |
| 9 | 600-epoch soak | `pperr == 0` throughout; nonzero is terminal, never clear-and-retry |

The §4.4 counting script (three lines of python over `llvm-objdump` output)
should be checked in next to the arm as the *verifier that ships with the
mechanism* — OVERLAP_ABSTRACTIONS.md §4.1's rule: a bound that can silently die
is not a primitive.

### 6.7 Sizing

| item | hours |
|---|---:|
| 0. clear the staged spill (P0, CPU-only) | 2–4 |
| 1. macro plumbing + constants + guard | 0.5 |
| 2. push-side quantizer helper | 1.5 |
| 3. M7.7 push rewrite | 1.5 |
| 4. M8 fp8 batch sibling | 3.0 |
| 5. optional one-liners | 0.5 |
| 6. build/ISA gates + verifier script | 1.0 |
| **engineering total** | **10–12 h** |
| correctness + negative control + soak (node, GPU) | 2 |
| MoK campaigns, 4 arms × 5 rotations | 4 wall |

---

## 7. MoK A/B protocol and the honest expected delta

### 7.1 Arms

One arm *name* (`mps_mega`) per `BENCHMARKING.md:171-188` — the sweep must never
be encoded as arm names, because `run_campaign.sh:83-89` rotates over the arm
list. But `K0P6_M15_STAGED` / `K0P6_M15_FP8WIRE` are **compile-time**, so each is
a separate build and therefore a separate campaign invocation, always with
`K0_MOK_ARMS=production,pf6gm_mega,mps_mega`.

| campaign | build | `K0_MPS_CFG` | what it isolates |
|---|---|---|---|
| **A0** | default (RMW) | `C=24,g=33,mode=12,flush_rows=16` | the banked ratchet control, in-session |
| **A1** | `STAGED=1` (post item 0) | same | **the fold alone** (K1 swap, bf16 format) |
| **A2** | `STAGED=1 FP8WIRE=1` | same | **the format alone**, on top of A1 |
| **A3** | `STAGED=1 FP8WIRE=2` | same | MORI-parity `direct_cast` ablation |

5 rotations each, same session, same seeds, `K0_PF6GM_G=3` forwarded (the
`BENCHMARKING.md:87-103` trap — check `pf6gm_mega/production_p50 ≈ 0.898`
before trusting anything).

Decision numbers: **A1 − A0** (fold), **A2 − A1** (format), **A2 − A0** (total),
each with its 5-rotation spread. A ratio without its rotation spread is not a
result.

Attribution runs (`timestamps=1`, **not scored**) at A0 and A2 for the M7 and
combine stamps. Judge M7 on its own stamp (σ = 23.74 µs, ~1 % resolution);
**never judge the combine stamp alone** — σ = 21.12 µs and exp_33's explicit
warning is that any claim of moving combine by < 60 µs measured alone is
untrustworthy. Judge the coupled M7+combine block (σ = 10.16 µs, 0.34 %).

### 7.2 The ledger, and the bound

Per rank per epoch, model at 352 GB/s egress and exp_29's 1.57 TB/s local rate:

| term | A0 (RMW) | A1 (staged bf16) | A2 (staged fp8) |
|---|---:|---:|---:|
| remote egress | 411.04 MB | 272.03 MB | **140.26 MB** |
| wire time | 1,168 µs, **hidden in M7** | 773 µs, **serial tail** | **398 µs, serial tail** |
| new local traffic | — | +642 MB (80–409 µs) | +642 MB (80–409 µs) |
| M8 slab read + zero | 621.8 MB | 621.8 MB | **320.6 MB** |
| epilogue transport | remote RMW @ 0.571 µs/op | local RMW | local RMW |

Model:

- **A1 − A0** = −(epilogue RMW stall recovered; ≤ 817–898 µs, unknown fraction)
  + 773 µs serial push + 80…409 µs local traffic ⇒ **+0 to +400 µs. Likely a
  loss.** The staged arm converts hidden wire time into exposed wire time.
- **A2 − A1** = −375 µs (push tail halves) − 100…160 µs (M8) ⇒
  **−475 to −535 µs. High confidence in the sign** — the byte reduction is
  arithmetic, and neither term is on a contended resource.
- **A2 − A0** ⇒ **−535 to −75 µs**, midpoint ≈ **−300 µs**. On the C=16
  baseline of 6,292.4 µs that is −4.8 %, i.e. 0.8165× → ~0.777× production. On
  the C=24 baseline of 5,848.5 µs, ~0.721×.
- **Hard upper bound.** The design can only attack the M7 epilogue surcharge
  (817–898 µs) plus M8's byte-bound component (≤ ~160 µs). **Any measured win
  above ~1,050 µs is an instrumentation error, not a result.** Say so before
  running.

Sanity cross-check against MORI: their combine bf16 → `fp8_direct_cast` is
366 → 642 GB/s effective, 1.754×; my format factor is 1.940× on bytes (their
number folds in their own kernel's overheads). Applied to production's measured
1,356 µs combine stage, MORI's lever would be worth ~580 µs *in a design where
combine is a separate exposed launch*. Ours is already overlapped, which is
exactly why our honest ceiling is the 817–898 µs surcharge and not the full
combine cost. Do not quote the 1.75× as our expected win.

### 7.3 Pre-registered falsifiers

1. **If A1 − A0 > +400 µs**, the staged carrier is falsified at this shape. Do
   not proceed to §3.5's overlap variant hoping to rescue it; report that the
   fold's fabric-op-class win (0.127 vs 0.571 µs/op) does not pay for exposing
   the wire time, and that the exp_22 carrier law extends to all-CTA carriers.
2. **If A2 − A1 > −200 µs**, the byte model is wrong and the push is not
   bandwidth-bound. Re-measure the push in isolation (a `debug_stop` after R2)
   before spending anything further.
3. **If MoK L1 rel err > 0.035**, treat it as a bug (most likely a group/scale
   index mismatch between the push and M8), not a tolerance question.
4. **If gate 0 or gate 3 regresses in any arm**, the number is void — exp_38's
   lesson is that all the *functional* gates can be green while the throttle is
   silently dead.

### 7.4 What a negative result buys

fp8 on the wire is reachable **only** through a staged transport (§2.5 closes
the only alternative). If A1 falsifies the staged carrier, then the honest,
publishable finding is: *MORI's 1.75× byte lever does not transfer to an
epilogue-carried remote-RMW megakernel, because the format win is only
collectible by a transport that un-hides wire time the fused design had already
hidden.* That is a genuine result about the K1×format interaction in the
five-knob model, and it is worth writing down either way.

---

## 8. Summary of decisions

1. **Format:** fp8 e4m3, per-128-element groups, `hkp_quant.hpp` verbatim, 56
   fp32 scales in an **inline tail** of the slab row. Row 14,336 B → **7,392 B**
   (1.939×). `direct_cast` (7,168 B, 2.000×) as an ablation only.
2. **Landing:** posted `store_peer_packets`-class stores into the existing
   symmetric `slots` buffer, reinterpreted at 7,392-B stride; owner-side
   dequant-reduce in M8, whose input bytes **halve**. Certification is the
   staged arm's existing R2 + `(rank, slab)` epoch words, unchanged.
3. **Pre-reduce:** impossible in LDS (one expert per CTA task, proved from the
   code); it is exactly the existing local bf16 stage, and it folds 1.511
   same-rank partials for free. Total remote-byte reduction **2.931×**.
4. **Registers:** **zero** epilogue cost — measured: 282 `pk_add_bf16` / 96
   `vmcnt(4)` / 180 `v_mfma` identical between the RMW and staged builds.
   Quantize-in-epilogue rejected on geometry, cross-lane cost, register
   pressure, and sub-dword fabric ops.
5. **Accuracy:** +0.78 % in quadrature ⇒ 0.0174 vs a 0.1 gate, 5.8× margin
   (3.6× under a 3× pessimistic assumption). MORI ships fp8 combine in
   production; our scaled variant is strictly more accurate than theirs.
6. **Blocker:** `K0P6_M15_STAGED=1` currently emits 96 scratch reloads + 97
   `vmcnt(0)` next to the remote atomic — the exp_24 pathology with the exp_38
   consequence. Nothing here can be measured until that reads 0.
7. **Expected:** A2 − A0 ≈ **−300 µs** (range −535…−75), hard-bounded at
   −1,050 µs. A1 is the risky arm and must be measured separately.
