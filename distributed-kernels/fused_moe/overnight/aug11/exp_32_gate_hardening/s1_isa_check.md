# S-1 ISA check — is M8's slot load already predicated?

**Verdict: UNCONDITIONAL (15 of 16), BUT THE TRAFFIC COLLAPSES ONTO ONE
14,336 B RANK-LOCAL REGION.** The "154 MiB" is real as *issued-instruction and
L1/L2-request volume* and is very nearly **zero** in DRAM and fabric terms.
Revised prize **≈ 4 µs (band 0–15 µs) = 0.06 % end-to-end**, an order of
magnitude below screen resolution — and the patch as specified carries a
**regression risk larger than the prize**, because the unconditional issue *is*
the load pipeline. **Do not build it.** The surviving descendant is a different
one-liner (§6a) worth 2–3× as much at strictly lower risk.

No GPU work was run. No file was modified. This is a read of an existing
disassembly.

---

## 1. Provenance of the evidence

| artifact | identity |
|---|---|
| disassembly read | `~/overnight-scratch/e26/out2/D.isa`, sha256 `d6f7588773b70706ba96fa8fd4babb9802c14cad72b311bc507776c1c9d3d4b8`, 29,500 lines |
| from | `D.hsaco`, sha256 `36520658c2ec860a63cd7d1f7c038a814af79a8b3061915a9cf8efdb120da15e`, 175,288 B |
| built by | `overnight/aug11/tools/e26b_build.sh` — CPU-only `hipcc --genco --offload-arch=gfx950 -O3 -DK0P6GM_G=3 -DN2GM_G=3 -mllvm -amdgpu-mfma-vgpr-form=1`, donor phase-1 include |
| kernel source | `k0pf6gm_device_tile_mps.hip` sha256 `0c92a1c9713fcf31adfb759336e381f41fb1b8b27022f62c05d10f8ebc1832b3`, 1,863 lines |
| **identity check** | that sha256 is **byte-identical** to the live node checkout at HEAD `0b82cd19` — so this disassembly *is* the ratchet's M8 |
| symbols | exactly one: `k0pf6gm_mps_mega`. MFMA census 180 (96 + 84), matching the recorded parity tuple |

M8 lives entirely in the `.hip`, so the phase-1 include choice (`D` = donor)
cannot affect it.

## 2. Locating the mode-12 M8 body

`k0p6_mps_m8_batch` is instantiated six times (`KRN:1760/1766/1787/1803/1821/1836`).
All six appear inlined. They separate cleanly on two counts:

| ISA lines (loads) | `flat_load_dwordx4` | `ds_bpermute_b32` | `flat_store_dwordx4` per c-iteration | instantiation |
|---|---|---|---|---|
| 21181–21247 | 16 | 32 | 4 | `Zero=false` (out stores only) |
| 22365–22431 | 16 | 32 | 4 | `Zero=false` |
| 23481–23547 | 16 | 32 | 4 | `Zero=false` |
| **24511–24577** | **16** | **40** | **20** | **`<NT, base_slot&, base_slot&, false, true>` — mode 12** |
| 25884–26020 | 32 | 64 | mixed | `Detect=true, Zero=true` (the dual-write detector: two bases, two loads per slot) |
| 28340–28406 | 16 | 32 | 4 | `Zero=false` |

`20 = 16 guarded zero stores + 4 out stores` is unique to `Zero=true, Detect=false`,
which is exactly the shipped mode-12 path (`KRN:1808-1825`,
`mode_is_direct_accum`). **The mode-12 M8 body is ISA lines 24494–25110.**

Cross-check on the addressing, which pins the source correspondence exactly:
`ds_bpermute_b32 vdst, vaddr, vsrc offset:N` selects lane `(vaddr + N)/4`. The
observed offsets are `0,4,8,12` on `(v64,v65)`; `32,36,40,44` on `(v66,v67)`;
`64,68,72,76` on `(v68,v69)`; `96,100,104,108` on `(v70,v71)` — i.e.
`offset = 4·(8t + jj)` for `t = 0..3`, `jj = 0..3`, which is
`__shfl(pbase[t], 8*t + jb + jj)` at `KRN:546` with `jb` folded into the `vaddr`
VGPR `v126`. There is **no** `offset:16/20/24/28` anywhere in the disassembly,
confirming the `jb` loop stayed a loop (`#pragma unroll 1`) with one body in the
binary rather than being unrolled into a second copy.

## 3. The answer: 15 of the 16 loads are issued unconditionally

Verbatim, ISA lines 24503–24578 (comments/encodings stripped for width; the
`v_lshl_add_u64` lines that form each 64-bit address are elided where they only
add noise):

```
24503   ds_bpermute_b32 v2, v126, v64 offset:4      <-- t=0 jj=1
24504   ds_bpermute_b32 v3, v126, v65 offset:4
24505   ds_bpermute_b32 v4, v126, v64 offset:8
...
24509   s_waitcnt lgkmcnt(4)
24511   flat_load_dwordx4 v[58:61], v[2:3]          <-- LOAD  1, no exec mask
24517   flat_load_dwordx4 v[54:57], v[4:5]          <-- LOAD  2
24518   flat_load_dwordx4 v[50:53], v[6:7]          <-- LOAD  3
24526   flat_load_dwordx4 v[46:49], v[2:3]          <-- LOAD  4
24529   flat_load_dwordx4 v[42:45], v[2:3]          <-- LOAD  5
24536   flat_load_dwordx4 v[38:41], v[2:3]          <-- LOAD  6
24537   flat_load_dwordx4 v[34:37], v[4:5]          <-- LOAD  7
24544   flat_load_dwordx4 v[30:33], v[2:3]          <-- LOAD  8
24546   flat_load_dwordx4 v[26:29], v[2:3]          <-- LOAD  9
24552   flat_load_dwordx4 v[22:25], v[4:5]          <-- LOAD 10
24553   flat_load_dwordx4 v[18:21], v[6:7]          <-- LOAD 11
24561   flat_load_dwordx4 v[14:17], v[2:3]          <-- LOAD 12
24564   flat_load_dwordx4 v[10:13], v[2:3]          <-- LOAD 13
24567   flat_load_dwordx4 v[6:9],  v[2:3]           <-- LOAD 14
24569   flat_load_dwordx4 v[2:5],  v[4:5]           <-- LOAD 15
24570   ds_bpermute_b32 v112, v126, v64             <-- t=0 jj=0 (offset:0)
24571   ds_bpermute_b32 v113, v126, v65
24572   v_cmp_lt_i32_e32 vcc, s20, v119             <-- (jb+0) < fanout[0]
24573   s_and_saveexec_b64 s[44:45], vcc
24574   s_cbranch_execz 35
24575   s_waitcnt lgkmcnt(0)
24576   v_lshl_add_u64 v[112:113], v[112:113], 0, v[62:63]
24577   flat_load_dwordx4 v[156:159], v[112:113]    <-- LOAD 16, PREDICATED
24578   s_waitcnt vmcnt(0) lgkmcnt(0)               <-- one full drain, covers 1..15
24579   v_and_b32_e32   v113, 0xffff0000, v156      <-- the bf16 -> f32 accumulate
24580   v_lshlrev_b32_e32 v112, 16, v156
24581   v_pk_add_f32    v[108:109], v[108:109], v[112:113]
...
24591   ds_bpermute_b32 v112, v126, v64             <-- the guarded zero store
24592   ds_bpermute_b32 v113, v126, v65             (re-computes the same base!)
24595   flat_store_dwordx4 v[112:113], v[152:155]
24596   s_or_b64 exec, exec, s[44:45]
24597   s_or_b32 s21, s20, 1
24598   v_cmp_lt_i32_e32 vcc, s21, v119             <-- slot 2's guard...
24599   s_and_saveexec_b64 s[22:23], vcc
24600   s_cbranch_execz 31
24603   v_and_b32_e32 v113, 0xffff0000, v58         <-- ...consumes LOAD 1's registers
```

Three things follow, and they are the whole answer.

**(a) There is no `v_cmp`, no `s_and_saveexec_b64`, no `s_cbranch` and no
`v_cndmask` anywhere between 24503 and 24569.** Loads 1–15 are issued
unconditionally, exactly as the source is written at `KRN:546-548`. So the S-1
premise is correct: **LLVM did not sink or predicate the load.**

**(b) The one load it did sink bought nothing.** The predicated slot is
`(t=0, jj=0)`, whose guard is `jb + 0 < fanout[0]`. For the `jb = 0` iteration
that is `0 < fanout[0]`, **always true for a live token** (`pull_ptr` gives every
live token at least one contribution). Only in the `jb = 4` iteration does it ever
skip. So essentially all of the dead loads are in the unconditional set.

**(c) The guard is EXEC-mask-based, not scalar.** `v_cmp_lt_i32_e32 vcc, s20,
v119`: the slot index `jb+jj` is an **SGPR** (`s20/s21/s25/s29` = `jb`, `jb|1`,
`jb|2`, `jb|3`) but `fanout[t]` is a **VGPR** (`v119/v120/v121/v123` for
`t = 0..3`). **LLVM does not know `fanout[t]` is wave-uniform** — see §6a, which
is where the real prize turns out to be.

## 4. The traffic: confirmed volume, refuted cost

The dead-slot address is `base_slot(cur, 0)` (`KRN:496-518`), and
`base_slot(p,row) = slots + ((p·MAXTOK + row − cur·MAXTOK)·H)`, so at `p = cur`,
`row = 0` it is **`slots + 0` exactly**. Over `c ∈ [0,14)` and `lane ∈ [0,64)`,
`off = (c<<10) + (lane<<4)` spans bytes `0 … 14,335`. Therefore **every dead load
on a rank, from every wave of every CTA in every batch, reads the same
14,336 B = 224 × 64 B cache lines.**

Volume (independently confirms `mode12_protocol_map.md` census d5/d6):

| quantity | value | derivation |
|---|---|---|
| loads per batch per wave | 448 | 14 c × 2 jb × (4 t × 4 jj) |
| bytes per load per wave | 1,024 B | 64 lanes × 16 B, contiguous |
| batches per rank per epoch | 1,024 | `⌈T/NT⌉ = 4096/4`, `NT = 4` at `KRN:1755` |
| **total slot-load volume** | **469,762,048 B = 448 MiB** | census d5 |
| live volume | `F × 14,336 = 21,504 × 14,336 = 308,281,344 B = 294 MiB` | census d6 |
| **dead volume** | **161,480,704 B = 154.0 MiB = 34.4 %** | `448 − 294`; mean fanout `21,504/4,096 = 5.25` of 8 |
| deletable (dead **and** unconditional) | ≈ 15/16 × 154 MiB ≈ **144 MiB** | slot `jb+0` is already predicated |

So the "**~154 MiB / 34 % of the combine's read instructions**" figure is
**correct as issued volume**. What it is *not*:

- **Zero incremental HBM bytes.** 14,336 B is 1/293 of one XCD's 4 MB L2 and
  1/18,725 of the 256 MB Infinity Cache. After the first 224-line touch it is
  resident everywhere; 14 KiB is also inside a CU's vector L1.
- **Zero xGMI bytes.** `slots` is rank-local (`K0P6_D_MPS_SLOTS`), and the ISA
  confirms it: these are plain `flat_load_dwordx4` with no peer translation, in
  the branch where `symmetric` is `nullptr` (`KRN:1718-1724`).

What remains is instruction issue, address VALU, `ds_bpermute` LDS-crossbar slots
and ~2.5 M L1/L2 tag lookups per rank per epoch — all of which **hit**.

## 5. Revised prize

Executed instructions per `(c, jb)` iteration, from the census of ISA
24494–25000 (507 static instructions: 16 `flat_load_dwordx4`, 16
`flat_store_dwordx4`, 64 `ds_bpermute_b32`, 32 `v_lshl_add_u64`, 25
`v_cmp_lt_i32_e32`, 25 `s_and_saveexec_b64`, 25 `s_or_b64 exec`, 64
`v_pk_add_f32`, 77 `v_lshlrev_b32`, 18 `s_waitcnt vmcnt(0)`, 23 `s_waitcnt
lgkmcnt(0)`):

| block | executed | note |
|---|---:|---|
| unconditional load prologue | ~66 | 15 loads + 30 bpermute + 15 `v_lshl_add_u64` + 6 lgkm waits |
| 16 guard headers | ~64 | `v_cmp` + `s_and_saveexec` + `s_cbranch` + `s_or_b64`, always executed |
| taken guarded bodies | ~273 | `0.656 × 16 = 10.5` bodies × ~26 instructions |
| **total** | **~403** | `s_cbranch_execz` skips the untaken bodies |

Deletable if the guard moved above the load: `0.344 × 16 = 5.5` dead slots, of
which `15/16` are unconditional ⇒ **5.16 slots**, each carrying 1 `flat_load` +
2 `ds_bpermute` + 1 `v_lshl_add_u64` = **20.6 instructions ⇒ 5.1 % of M8's
executed instruction stream.**

Converting to time — shared-unit occupancy, which is the binding term (one block
per CU, 4 waves on 4 SIMDs sharing one TA/TD/L1 and one LDS):

```
per deleted slot : 1 flat_load (4 TA cycles, wave64 at 16 lanes/clk)
                 + 2 ds_bpermute (~3 LDS cycles each)
                 + 1 v_lshl_add_u64 (4 cycles)          ≈ 14 cycles
per CU per epoch : 5.16 slots × 28 (c,jb) iterations × 4 waves = 578 slots
                 : 578 × 14 = 8,092 cycles ÷ 2.2 GHz     ≈ 3.7 us
```

**Revised prize ≈ 4 µs, band 0–15 µs = 0.06 % end-to-end** (0.9 % of the ~430 µs
M8+M9 pool). For scale, the same accounting puts M8's whole *issue-bound floor* at
~5 µs per epoch against a **measured ~430 µs** — M8 is ~85× away from being
issue-bound. It is dominated by the ≥21,504 system-scope `row_ready` polls
(census d2) and memory latency, not by instruction count. Deleting 5 % of a stream
that is not the limiter buys ~nothing.

**Against a 0.52 % 1-σ on `ratio_vs_prod` and a 2 % single-screen resolution, this
is unmeasurable.** Even `timestamps=1` `ts_combine_us`, which resolves the combine
pool directly, would need to call a 0.9 % move.

### And the patch would probably make it slower

This is the part worth carrying forward. Look again at 24569 → 24578: fifteen
loads issued back-to-back, then **one** `s_waitcnt vmcnt(0)` drains all of them,
then 10.5 guarded bodies consume pre-loaded registers. That batching exists
*because* the load is unconditional — under a guard LLVM cannot prove the address
dereferenceable and **cannot hoist**. Every slot would then take the shape LLVM
already emits for the one it sank (24575–24581): `bpermute → v_lshl_add_u64 →
flat_load → s_waitcnt vmcnt(0) → accumulate`.

```
today   : 15 loads in flight, 1 drain per (c,jb) iteration
patched : ~10.5 SERIALIZED issue-then-drain round trips per (c,jb) iteration
        : 10.5 × ~350 cycles (L2 hit) × 28 iterations ≈ 103k cycles ≈ 47 us/wave
```

Only partly covered by the 4 waves per CU — and an order of magnitude the wrong
way against a ~4 µs prize. **S-1 as specified is closed. It is not a
"disappointing win", it is a likely regression, and the ISA says why.**

## 6. What survives, in priority order

**(a) `fanout[t]` is provably wave-uniform and LLVM does not know it — worth
2–3× S-1 at lower risk.** ISA proof: `v_cmp_lt_i32_e32 vcc, s20, v119` (scalar
slot index vs **vector** fanout) ⇒ 16 × (`v_cmp` + `s_and_saveexec_b64` +
`s_or_b64 exec`) per `(c,jb)` iteration, plus 9 duplicated guard headers on the
branch-split paths (25 of each in the body for 16 logical guards). Source proof of
uniformity: `tok0 = batch·NT` with `batch = *batch_slot` broadcast through LDS
behind `__syncwarp()` (`KRN:1813-1819`), and `live[t] = tok < T`, so `lo2[t]` and
`fanout[t]` at `KRN:484-489` are wave-uniform **by construction**. LLVM cannot see
through the LDS broadcast. A `__builtin_amdgcn_readfirstlane` on `lo2[t]` and
`fanout[t]` turns each guard into `s_cmp_lt_i32` + `s_cbranch_scc0`, deleting
**~32–48 of ~403 executed instructions (8–12 %)** and **leaving the load schedule
untouched**, because the loads stay unconditional. Still only ~0.1 % end-to-end,
so it must ride along with another M8 edit rather than be its own campaign — but
it is the correct one-liner, and it also makes the guarded zero store's duplicated
`ds_bpermute` (24591–24592 recomputing 24570–24571) available for CSE.

**(b) The dummy base aliases a live, concurrently-written slot row — free to
fix.** `slots + 0` is slot index 0, i.e. producer rank 0's contribution for local
token 0. Under `Zero=true` that row is being zeroed by whichever wave owns token 0
(`KRN:579-587`) while every other wave on the rank is reading it as a dummy. Bytes
are discarded so there is no correctness issue (`KRN:461-462` says exactly this),
but it is gratuitous read/write sharing on one 14 KiB region across 256 CUs and 8
XCDs. Pointing the dummy at a never-written 14,336 B pad is zero instructions and
zero risk. **Unmeasured** — recorded, not claimed.

**(c) Every M8 payload load is `flat_`, not `global_` — the biggest lead here.**
The kernel contains **355 `flat_load_dwordx4`, 351 `flat_load_dwordx2`, and 975
`v_lshl_add_u64`, against 1 `global_load_dwordx4`.** The base arrives as an
`unsigned long long` cast to `const uint4*` (`KRN:546-548`), so LLVM cannot prove
the global address space and falls back to FLAT. FLAT loses the SGPR-base +
immediate-offset form — hence a 64-bit VGPR address and a `v_lshl_add_u64` per
load, and the `ds_bpermute` pair that feeds it. Recovering `global_load_dwordx4`
with an explicit `__global__` address-space cast would delete one VALU per load
across the whole kernel and let the wave-uniform base live in SGPRs. Confidence:
the FLAT-vs-GLOBAL fact is **DOCUMENTED** by this disassembly; the additional
claim that gfx9-family FLAT counts against both `vmcnt` **and** `lgkmcnt` (which
would serialize these loads against the 64 `ds_bpermute` per iteration, and the
body does carry 23 `s_waitcnt lgkmcnt(0)`) is **UNVERIFIED** here and must be
checked against the CDNA4 ISA reference before anyone quotes a number from it.

## 7. The one-line patch — not issued

The deliverable asked for the exact one-line patch "only if the prize survives".
It does not, so none is given. For the record, the patch would have been to move
the `KRN:561` guard above the `KRN:547` load:

```
// NOT RECOMMENDED -- see §5. This forbids the hoist at ISA 24503-24569 and
// converts one batched 15-load issue into ~10.5 serialized round trips.
```

The patch that *should* be written when someone next owns
`k0pf6gm_device_tile_mps.hip` is §6a, and it belongs to the same edit as whatever
else touches M8 — never as a standalone campaign, because it cannot be resolved by
a screen.

## 8. Corrections to record in `LESSONS.md`

1. **exp_29's S-1 arithmetic is confirmed; its valuation is not.** 154 MiB and
   34 % of M8's read instructions are both right, and independently reproduce
   `mode12_protocol_map.md` census d5/d6. But the dead loads all land on one
   14,336 B rank-local region, so the bandwidth is ~0 and the prize is ~4 µs, not
   a combine-scale win. "**Worth more than the entire pipelining mechanism, with
   none of its risk**" is **retracted on both halves**: it is worth ~4 µs, and it
   carries a serialization risk larger than its prize.
2. **`mode12_protocol_map.md` census d5 is correct and should be strengthened.**
   Its note that out-of-fanout lanes read `base_slot(cur,0) = slots + 0` is exactly
   right; add that the 154 MiB dead component therefore addresses **one 14,336 B
   region = 224 cache lines**, so it is issued-instruction volume, not bandwidth.
3. **New, ISA-backed:** `fanout[t]` and `lo2[t]` are wave-uniform by construction
   but reach LLVM as VGPRs, so all 16 M8 slot guards are EXEC-mask guards rather
   than scalar branches (§6a). And every M8 payload load is FLAT rather than
   GLOBAL (§6c), which costs one `v_lshl_add_u64` per load kernel-wide.
