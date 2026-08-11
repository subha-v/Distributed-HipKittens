# exp_03 — ISA read of the `store_peer_packets` push loop (`k0pf6gm_mps_mega`, gfx950)

`/tmp/k0pf6gm_device_tile_mps.gfx950.hsaco` is a clang offload bundle; unbundled
`--type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950` → `/tmp/mps.co` →
`llvm-objdump -d --mcpu=gfx950`. 19,404 instructions, one kernel at `0x1900`.
CPU-only work in `subha_k1`; no GPU touched.

## Answer

**Exactly ONE load in flight per lane.** The inlined copy loop emits one
`flat_load_dwordx4` (16 B/lane), then an unconditional `s_waitcnt vmcnt(0)
lgkmcnt(0)`, then one `flat_store_dwordx4` — 10 instructions, `#pragma unroll 1`
honoured, no unrolling, no software pipelining of the load across the back edge.
The only cross-iteration overlap is that iteration *k*'s store is still
outstanding when *k+1*'s load issues (a 2-deep VMEM window), and the `vmcnt(0)`
then drains both, so never more than one *load* per lane is live. The copy is
therefore **latency-bound at one memory round trip per 16 bytes per lane**, and a
64-lane wave holds at most 1,024 B of payload in flight. Little's law makes the
consequence concrete: at 4 waves per service CTA, in-flight bytes are `4096·C`,
so the cost model's ~148 GB/s needs a per-iteration round trip under **221 ns at
C=8** (886 ns at C=32, 1.77 µs at C=64). 221 ns must cover a local global load
*plus* the vmcnt retire of a cross-die peer `flat_store` — independent ISA
support for the A1 correction, and evidence that the MLP=1 loop shape, not the
fence count, is the first-order limiter. Second finding, load-bearing for M1:
**the payload store carries no `sc0`, no `sc1`, no `nt`**, so it does not take
Coherent Cache Bypass and the release's `buffer_wbl2` is *not* redundant.

## The loop, disassembled

Two inlined instances, identical shape. **Loop A = `push_slice_group`**
(`moe_mps_adapter.cuh:233`), head `0x14450`, tail `0x14488`, 10 instructions.
Identified by the preheader `s_movk_i32 s20, 0x1c00` (7168) → `v_mad_u64_u32 …
v42, s20` → `v_lshl_add_u64 …, 1, v[40:41]` (`*2` bf16 + `env.part`), and the
`flat_atomic_add … sc0` / `ds_read_b32` on `wave_scratch` right after. `s[42:43]`
is set to `0x400` at `0x13E28`, never redefined before the loop → stride 1,024 B
= 64 lanes × 16 B.

```
; LOOP BODY (back-edge target 0x14450)
000000014450: flat_load_dwordx4 v[58:61], v[44:45]            ; <-- the ONLY load
000000014458: v_lshl_add_u64 v[46:47], v[46:47], 0, 64        ; packet += threads(64)
000000014460: v_cmp_ge_u64_e32 vcc, v[46:47], v[34:35]        ; packet >= count (runtime)
000000014464: v_lshl_add_u64 v[44:45], v[44:45], 0, s[42:43]  ; src += 1024
00000001446C: s_or_b64 s[50:51], vcc, s[50:51]
000000014470: s_waitcnt vmcnt(0) lgkmcnt(0)                   ; <-- full drain
000000014474: flat_store_dwordx4 v[42:43], v[58:61]           ; no sc0/sc1/nt
00000001447C: v_lshl_add_u64 v[42:43], v[42:43], 0, s[42:43]  ; dst += 1024
000000014484: s_andn2_b64 exec, exec, s[50:51]
000000014488: s_cbranch_execnz 65521    // -> 0x14450
```

**Loop B = mode-1 whole-row push** (`k0pf6gm_device_tile_mps.hip:1363`), head
`0x157CC`, tail `0x15828`, 14 instructions, same 1-load/`vmcnt(0)`/1-store body
(`flat_load_dwordx4 v[30:33], v[30:31]` at `0x157D4`; `s_waitcnt vmcnt(0)
lgkmcnt(0)` at `0x15818`; `flat_store_dwordx4 v[34:35], v[30:33]` at `0x1581C`).
Its bound is the immediate `s_mov_b64 s[0:1], 0x37f` (895) because
`row_bytes = 7168*2` is compile-time (896 packets); stride is `blockDim.x` via
`v_readlane_b32 s0, v255, 8`.

Both use `flat_`, not `global_`: `peer_ptr` builds `dst` with a `v_cndmask` chain
over the 8 heap bases (`v_cmp_eq_u32 vcc, 5/6/7, v58` above loop A), so address
space 1 is not provable. Hence the combined `vmcnt(0) lgkmcnt(0)`, coupling the
payload drain to `wave_scratch`'s `ds_read`/`ds_write` (`INFERRED` from uniform
pairing on every FLAT here; confirm in ISA §4.4).

## waitcnt accounting

| item | value |
|---|---|
| `*_load_dwordx4` before the first `s_waitcnt` | **1** |
| `vmcnt(N)` argument on that wait | **`vmcnt(0)`** (plus `lgkmcnt(0)`) |
| stores after the wait / further waits in body | 1 / none |
| outstanding **loads** per lane | **1** |
| bytes in flight, per lane / per 64-lane wave | 16 B / **1,024 B** |
| unrolled? | **no** — `#pragma unroll 1` honoured |
| software-pipelined across back edge? | **no** — no peeled prologue load |
| body instruction count | **10** (A), **14** (B) |
| trip count | **dynamic** for A (`v_cmp_ge_u64` vs runtime `v[34:35]`, `count = 56·g`); static 896 for B |

Whole-kernel wait census: **595 `vmcnt(0)` vs 2 `vmcnt(1)`** — essentially no
partial waits anywhere, so M5 cannot arrive from scheduling; it needs source with
>1 load live. Per-lane trips, `count = bytes>>4 = 56·g`, `threads = 64`: `g=1` →
56 packets, lanes 0–55 do **1** iteration and 56–63 do 0; `g=2` → 112 (2 / 1);
`g=4` → 224 (4 / 3); `g=16` → 896, all lanes 14. **At `g=1` the body runs once**
— 14 B per lane; there is no loop to pipeline.

## Cache bits

**Payload store (item 6):** `flat_store_dwordx4 v[42:43], v[58:61]` encodes
`DC7C0000` — bit 16 (`sc0`), bit 17 (`nt`), bit 25 (`sc1`) all **clear**.
Encoding cross-checks in the same object: `flat_load_dwordx4 … nt` = `DC5E0000`
(+0x20000 = bit 17); `flat_store_dwordx2 … sc0 sc1` = `DE750000` vs plain
`DC740000` (+0x02010000 = bits 16 and 25). The payload write is fully cached, no
bypass, no hit-evict. 126 of 251 stores kernel-wide *do* carry `sc0 sc1` — the
flags do, the payload does not.

**`flush_pending` release (item 5):**

```
000000014524: s_waitcnt vmcnt(0)             ; inline-asm wave drain (adapter:247)
000000014528: s_and_b64 exec, exec, s[26:27] ; lane == 0
000000014530: buffer_wbl2 sc0 sc1            ; thread_release<system>
000000014538: s_waitcnt vmcnt(0)             ; writeback ACK
000000014570: flat_store_dword v[42:43], v31 sc1       ; publish_epoch<agent>
0000000146D8: flat_store_dword v[42:43], v48 sc0 sc1   ; publish_epoch<system>, peer owner
0000000146E8: flat_store_dword v[42:43], v48 sc1       ; row_rem self-clean, agent
```

`buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)`, and **no `buffer_inv`** (correct —
pure release). Fingerprint drift: `sync.cuh`'s comment records the donor lowering
as `buffer_wbl2 sc1`; ROCm 7.2.4 emits `sc0 sc1`. Kernel-wide 40 `buffer_wbl2` /
38 `buffer_inv`, 8 of the `wbl2` being the `sc0 sc1` flavour.

**M1's premise is currently false.** It assumes the payload store already takes
Coherent Cache Bypass so the writeback is provably redundant; the store carries no
`sc` bits, so the payload *is* left dirty in the local 4 MB L2 and the writeback
is load-bearing. M1 becomes two steps — add `sc0 sc1` to the payload store and
prove the bits landed, *then* delete the writeback — with the
bits-on/writeback-still-present arm as its control.

## What would raise MLP

Two additive `packet.cuh` overloads; `store_peer_packets` is untouched so every
existing caller stays bit-identical. Do **not** re-express it as
`store_peer_packets_batched<1>` — that would perturb its codegen.

**(a) Compile-time batch factor plus remainder loop.** `bytes` is runtime, so the
staging array needs a template `Batch`; compile-time indices keep it in VGPRs
(dynamic indices demote it to scratch, which this file's own comment warns about
and the 60 B scratch budget forbids):

```cpp
template<unsigned int Batch>
KITTENS_DISTRIBUTED_DEVICE_INLINE void store_peer_packets_batched(
        void* __restrict__ destination, const void* __restrict__ source,
        std::size_t bytes, unsigned int tid, unsigned int threads) {
    static_assert(Batch > 0u && Batch <= 8u);
    auto* const out = reinterpret_cast<packet16*>(destination);
    const auto* const in = reinterpret_cast<const packet16*>(source);
    const std::size_t count = bytes >> 4;
    const std::size_t span  = (std::size_t)threads * Batch;
    const std::size_t end   = (count >= span) ? (count - span + 1u) : 0u;  // no underflow
    std::size_t p = tid;
    for (; p < end; p += span) {
        packet16 staged[Batch];
#pragma unroll
        for (unsigned int k = 0; k < Batch; ++k) staged[k] = in[p + (std::size_t)k * threads];
#pragma unroll
        for (unsigned int k = 0; k < Batch; ++k) out[p + (std::size_t)k * threads] = staged[k];
    }
#pragma unroll 1
    for (; p < count; p += threads) out[p] = in[p];   // the existing MLP=1 tail
}
```

MLP becomes `Batch` for `4·Batch` VGPRs, so `Batch=4` is the only safe first
point against the 256 ArchVGPR / 256 AGPR parity tuple, and the ISA diff must
confirm scratch did not grow.

**(b) The overload this kernel actually needs — fan-out across streams.** (a)
only engages when `count ≥ threads·Batch = 256` packets, i.e. `g ≥ 5`: it does
nothing for `g ∈ {1,2,4}` and helps only at `g=16`. The binding constraint is
that 896·g bytes over 64 lanes is too little work per lane, not the loop shape.
So add a second overload batching across *independent* (dst, src) streams — the
several claimed groups a service wave already holds — issuing all `Ways` loads
before any store: `template<unsigned int Ways> store_peer_packets_fanout(void*
const (&dst)[Ways], const void* const (&src)[Ways], std::size_t bytes, unsigned
tid, unsigned threads)`. `push_slice_group` then pushes `Ways` groups per call,
reaching MLP = `Ways` at every `g` including `g=1` — the change that makes the
221 ns / C=8 budget reachable without inflating `C`. Both are weaker than **M3**
(≥256 KiB bands), and this read supports M3 directly: the 10-instruction MLP=1
loop is the mechanism by which a 14 KiB push unit sits ~70× below COMET's ~1 MiB
knee.
