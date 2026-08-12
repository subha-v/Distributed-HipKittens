# exp_23 patch spec — per-CTA phase stamp array (apply-ready)

**Status:** specification only. No file outside
`overnight/aug11/exp_23_fig10_timeline/**` was touched by this agent.
`k0pf6gm_device_tile_mps.hip` and `moe_mps_adapter.cuh` are owned tonight by the
mode-14 implementer and carry uncommitted changes; every line number below is
read from the **working tree** at local HEAD `b5215081` + those uncommitted
changes, on 2026-08-12. **They will drift when mode 14 lands.** Every edit
therefore names a unique anchor string; §0 gives the one command that re-derives
all line numbers in ten seconds. Do not apply by line number alone.

---

## 0. Re-deriving the anchors after the tree moves

```bash
cd $DHK/distributed-kernels/fused_moe
grep -n 'K0P6_MPS_TS_COUNT 8\|#define K0P6_MPS_SRC_REV\|ts_last(\|K0P6_MPS_ST_WORDS + 2 \* K0P6_MPS_TS_COUNT\|__syncthreads();$\|mode_is_stream(scfg)\|run_service(env\|release_cta_payload_system\|const bool payload_ok\|hk_moe::mps::ts_last(mps_ts + K0P6_MPS_TS_REDUCE_DONE' \
  k0pf6gm_device_tile_mps.hip moe_mps_adapter.cuh
```

Every anchor in this document appears in that output. If an anchor is missing,
the implementer changed the site and this spec needs a five-minute re-read — do
not guess.

---

## 1. Design in one paragraph, and why it is shaped this way

A **fixed-slot per-CTA stamp array**, not a ring: cell `(bid, phase_id)` holds
the raw 64-bit `s_memrealtime` value of the last time CTA `bid` crossed boundary
`phase_id`. It lives in the **spare tail of the existing `K0P6_D_MPS_STATE`
buffer** (descriptor slot 60), immediately after the 8 u32 scalars and the 8 u64
coarse timestamps. It is written only by `tid == 0`, only inside code that
already runs under `cfg.timestamps`, and it is read by the host after the
launch. Consequences, each of which is the reason a more obvious design was
rejected:

| property | why it matters |
|---|---|
| **No new descriptor slot** | `design.md` proposed slot 63 + bumping `K0P6_MPS_D_LEN`. That costs edits to `moe_host_abi.hpp` (`append_mps_descriptor`, `validate_mps_descriptor`), `benchmarks/mok_synthetic_prefill/mps_host_bridge.cpp` (a new `append_and_validate` argument), a **pybind module rebuild**, and the `assert pf6_state["desc_mps"].numel() == 63` at `e004pf_k0pf_ab.py:~2816`. Growing slot 60's buffer instead is **one host line** and the descriptor stays 63 words. |
| **No append ticket** | `design.md`'s ring needed `atomicAdd` for a write cursor plus an overflow drop counter. A fixed slot makes overflow **structurally impossible** and adds **zero atomics** — which is what protects the `flat_atomic_pk_add_bf16` census of 282. |
| **No packing** | `design.md` stored `(phase_id << 56) | ticks`. The slot index already carries the phase. Storing the raw clock removes a shift and an or per site, and makes the reconciliation check in §6.2 **exact** instead of approximate. |
| **No M0 zeroing** | The host tensor is `torch.zeros`, and a stale cell from an earlier epoch lands far outside the final-epoch window, so the parser *detects* staleness instead of relying on a reset. This removes an edit to M0's zero loop entirely — one fewer register-risk site. |
| **No LDS** | LDS must read exactly `155,496 B`. The array is global. |
| **Nothing crosses a phase** | Every site re-derives its own base pointer from a local `k0p6_dread`, the donor discipline the existing stamps already follow. No new value is live at the phase-2 pointer peak, which is where this kernel's documented spill cliff lives (`k0pf6gm_device_tile_mps.hip:1693-1700`: a second copy of one release sequence cost +16 B/lane of scratch and pushed SGPR spills 186 → 221). |

Timestamp source is the existing `hk_moe::mps::realtime_now()`
(`moe_mps_adapter.cuh:496-513`), i.e. `s_memrealtime` + `s_waitcnt lgkmcnt(0)`,
**1 tick = 0.01 µs**. Do not add a second clock read anywhere a stamp already
exists — see `ts_mark` in §3.

Diagnostics flag: **`cfg.timestamps`, packed bit 33 of `K0P6_D_MPS_CFG`**
(`moe_mps_adapter.cuh:66`, `109`), reached from the harness as the optional
`timestamps=1` key of `K0_MPS_CFG`. This is deliberate and load-bearing: a new
environment variable would **not work**, because `run_campaign.sh` forwards a
fixed `-e` list and `K0_PF6GM_DECOMP` is already stranded outside it
(`PLOTS.md` caveat 4). `timestamps` is already inside `K0_MPS_CFG`'s grammar
(`_parse_mps_config`, `e004pf_k0pf_ab.py:389`), so no harness edit is needed to
turn the instrument on.

---

## 2. Layout

```
K0P6_D_MPS_STATE (slot 60) buffer, byte offsets:
  [  0 ..  32)  8 x u32   scalar lanes      (K0P6_MPS_ST_WORDS = 8)
  [ 32 ..  96)  8 x u64   coarse timestamps (K0P6_MPS_TS_COUNT = 8)
  [ 96 .. 32864)  256 x 16 x u64  per-CTA phase stamps   <-- NEW
total 32,864 B  (was 96 B)
```

Cell address:

```
u64* ring = (u64*)(mps_state + K0P6_MPS_ST_WORDS) + K0P6_MPS_TS_COUNT;
ring[(size_t)bid * K0P6_MPS_E23_SLOTS + phase_id]
```

`bid < 256` is guaranteed by the existing shape guard `nct != 256`
(`k0pf6gm_device_tile_mps.hip:738`), so the index is bounded without a new
check. 16 slots × 8 B = 128 B/CTA; one CTA's stamps occupy exactly two 64-B
cache lines and never share a line with another CTA.

### Phase ids

| id | name | interval it opens/closes | who writes it | tier |
|---:|---|---|---|---|
| 0 | `KSTART` | opens `dispatch` | all CTAs | B |
| 1 | `M2_DONE` | closes `dispatch`, opens `plan` | all | **A** |
| 2 | `M5_DONE` | closes `plan`, opens `M6` | all | **A** |
| 3 | `M6_DONE` | closes `M6`, opens `M7` | all | **A** |
| 4 | `M7_DONE` | closes `M7` | compute CTAs only | **A** |
| 5 | `SVC_ENTER` | opens `service` | stream modes | C |
| 6 | `SVC_EXIT` | closes `service` | stream modes | C |
| 7 | `M75_ENTER` | opens `m75` | modes 0/5/6/14 | C |
| 8 | `M75_BAR_DONE` | splits `m75` into fence vs barrier-wait | modes 0/5/6/14 | C |
| 9 | `M75_EXIT` | closes `m75` | modes 0/5/6/14 | C |
| 10 | `M8_ENTER` | opens `combine` | all | B |
| 11 | `REDUCE_DONE` | closes `combine` | all | **A** |
| 12 | `M9_DONE` | closes `tail` | all | B |
| 13 | — | reserved | — | — |
| 14 | — | reserved | — | — |
| 15 | `META` | `(role << 32) | count` — role 0 compute / 1 service; count = batches or stripes handled | all | C |

**Tiers are the build ladder, not decoration.** Tier A is five marks, all of
them inside `if (cfg.timestamps)` blocks that already exist and already read the
clock — their entire marginal cost is one `global_store_dwordx2` and its address
arithmetic per site. Tier B adds three marks in existing non-MFMA scopes that
have no stamp today (each costs one descriptor read + one config decode + one
clock read). Tier C adds the service/M7.5 marks and the meta word, and it is the
tier that touches the region with the documented allocation cliff.

**Gate Tier A first, then A+B, then A+B+C, and ship the largest tier whose
resource tuple is exact.** Tier A alone is sufficient for the figure: per-CTA
`plan` / `M6` / `M7` / `combine` intervals *are* the NanoFlow-v2 Fig-10 panel.
B widens the x-axis to the whole kernel; C adds the service strip's texture.

---

## 3. Edit 1 — `moe_mps_adapter.cuh` (additive; no existing line changes)

**Anchor** (working tree line 58):

```cpp
#define K0P6_MPS_TS_COUNT 8
```

**Insert immediately after it:**

```cpp
// ---- exp_23: per-CTA phase stamp array (diagnostic, behind cfg.timestamps) ---
// Lives in the SPARE TAIL of K0P6_D_MPS_STATE, after the 8 u32 scalars and the
// 8 u64 coarse cells. FIXED SLOT per (CTA, boundary): overflow is structurally
// impossible and no write cursor -- hence no atomic -- is needed. A boundary a
// CTA crosses more than once per epoch is last-write-wins by design; slot
// K0P6_MPS_E23_META carries the repeat count so the host can say so. Cells are
// NOT reset per epoch: after the soak every live cell is from the FINAL epoch,
// and a cell left over from an earlier epoch falls outside the final-epoch
// window, so the host DETECTS staleness rather than trusting a reset.
#define K0P6_MPS_E23_SLOTS 16
#define K0P6_MPS_E23_KSTART 0
#define K0P6_MPS_E23_M2_DONE 1
#define K0P6_MPS_E23_M5_DONE 2
#define K0P6_MPS_E23_M6_DONE 3
#define K0P6_MPS_E23_M7_DONE 4
#define K0P6_MPS_E23_SVC_ENTER 5
#define K0P6_MPS_E23_SVC_EXIT 6
#define K0P6_MPS_E23_M75_ENTER 7
#define K0P6_MPS_E23_M75_BAR 8
#define K0P6_MPS_E23_M75_EXIT 9
#define K0P6_MPS_E23_M8_ENTER 10
#define K0P6_MPS_E23_REDUCE_DONE 11
#define K0P6_MPS_E23_M9_DONE 12
#define K0P6_MPS_E23_META 15
```

**Anchor** (working tree line 526, the closing brace of `ts_last`):

```cpp
__device__ __forceinline__ void ts_last(std::uint64_t* cell, bool enable) {
    if (!enable || cell == nullptr) return;
    atomicMax(reinterpret_cast<unsigned long long*>(cell), realtime_now());
}
```

**Insert immediately after it:**

```cpp
// exp_23. `ts_base` is (u64*)(mps_state + K0P6_MPS_ST_WORDS) -- the same pointer
// the coarse stamps already compute at every site, so the stamp array costs NO
// new descriptor read and NO new live value: the base is that pointer plus a
// COMPILE-TIME constant.
__device__ __forceinline__ std::uint64_t* e23_ring(std::uint64_t* ts_base) {
    return ts_base + K0P6_MPS_TS_COUNT;
}

// Plain (non-atomic, non-volatile) store. The only reader is the HOST, after
// kernel completion and `torch.cuda.synchronize()`; end-of-kernel implicit
// release plus the stream sync are sufficient, so no scope qualifier and no
// fence is required or wanted here.
__device__ __forceinline__ void e23_mark(std::uint64_t* ts_base, int bid,
                                         std::uint32_t slot, bool enable) {
    if (!enable || ts_base == nullptr) return;
    e23_ring(ts_base)[(std::size_t)bid * K0P6_MPS_E23_SLOTS + slot] =
        realtime_now();
}

// Coarse cell + per-CTA cell from ONE clock read. This is what makes the
// reconciliation check exact: max over CTAs of the per-CTA cell is then
// bit-identical to the coarse atomicMax cell, by construction. Use this at
// every site that already called ts_last; never call ts_last and e23_mark as a
// pair, which would read the clock twice and break the identity.
__device__ __forceinline__ void ts_mark(std::uint64_t* cell,
                                        std::uint64_t* ts_base, int bid,
                                        std::uint32_t slot, bool enable) {
    if (!enable) return;
    const std::uint64_t t = realtime_now();
    if (cell != nullptr) {
        atomicMax(reinterpret_cast<unsigned long long*>(cell), t);
    }
    if (ts_base != nullptr) {
        e23_ring(ts_base)[(std::size_t)bid * K0P6_MPS_E23_SLOTS + slot] = t;
    }
}

__device__ __forceinline__ void e23_meta(std::uint64_t* ts_base, int bid,
                                         std::uint32_t role,
                                         std::uint32_t count, bool enable) {
    if (!enable || ts_base == nullptr) return;
    e23_ring(ts_base)[(std::size_t)bid * K0P6_MPS_E23_SLOTS +
                      K0P6_MPS_E23_META] =
        ((std::uint64_t)role << 32) | (std::uint64_t)count;
}
```

No existing line in `moe_mps_adapter.cuh` changes. `ts_last` stays, because the
service side and the readiness stamps still use it.

---

## 4. Edit 2 — `k0pf6gm_device_tile_mps.hip`

### 4.0 (mandatory) bump the source revision

**Anchor** line 107: `#define K0P6_MPS_SRC_REV 27` → `28` (or whatever the
mode-14 implementer left it at, **+1**). `.cuh`-only edits do not invalidate the
mori JIT cache; this `.hip` is hashed. After the first run confirm
`hsaco_before != hsaco_after` with `readlink -f` + `stat -L`.

### 4.1 Tier A — five sites, each an in-place `ts_last` → `ts_mark`

Each of these is a **one-line replacement** inside an `if (tid == 0)` block that
already decodes the config and already computes the u64 base.

**A1 — M2_DONE.** Anchor at line 1213:

```cpp
      hk_moe::mps::ts_last(
          (std::uint64_t*)(mps_state2 + K0P6_MPS_ST_WORDS) +
              K0P6_MPS_TS_M2_DONE,
          cfg2.timestamps);
```
becomes
```cpp
      std::uint64_t* ts2 = (std::uint64_t*)(mps_state2 + K0P6_MPS_ST_WORDS);
      hk_moe::mps::ts_mark(ts2 + K0P6_MPS_TS_M2_DONE, ts2, bid,
                           K0P6_MPS_E23_M2_DONE, cfg2.timestamps);
```
*Placement:* after the Pass-A→B grid barrier and the `pperr` acquire, i.e. the
existing dispatch-complete point. Unchanged.

**A2 — M5_DONE.** Anchor at line 1447 (`ts_last(... K0P6_MPS_TS_M5_DONE ...)`
with `cfg5`). Same transformation, `ts5`, `K0P6_MPS_E23_M5_DONE`. This site sits
**inside the `{ ... }` scope that ends at line 1451**, so `bid` is in scope.

**A3 — M6_DONE.** Anchor at line 1487 (`cfg6`, `mps_state6`). Same, `ts6`,
`K0P6_MPS_E23_M6_DONE`.

**A4 — M7_DONE.** Anchor at line 1535:
```cpp
        hk_moe::mps::ts_last(mps_ts + K0P6_MPS_TS_M7_DONE, m7cfg.timestamps);
```
becomes
```cpp
        hk_moe::mps::ts_mark(mps_ts + K0P6_MPS_TS_M7_DONE, mps_ts, bid,
                             K0P6_MPS_E23_M7_DONE, m7cfg.timestamps);
```
`mps_ts` is already the u64 base at this site. *Note:* this block is inside
`if (!is_service_cta(...))`, so service CTAs leave slot 4 at zero — which is
exactly how the parser learns each CTA's role without a meta word, and is why
Tier A works with Tier C dropped.

**A5 — REDUCE_DONE.** Anchor at line 2105:
```cpp
      hk_moe::mps::ts_last(mps_ts + K0P6_MPS_TS_REDUCE_DONE,
                           m8cfg.timestamps);
```
becomes
```cpp
      hk_moe::mps::ts_mark(mps_ts + K0P6_MPS_TS_REDUCE_DONE, mps_ts, bid,
                           K0P6_MPS_E23_REDUCE_DONE, m8cfg.timestamps);
```

### 4.2 Tier B — three new marks

Tier B sites have no existing stamp, so each needs its own config decode. Gate
the decode with `__builtin_amdgcn_readfirstlane` for the reason line 1731 of the
same file documents: the mode arrives through a *volatile* descriptor read, LLVM
cannot prove it uniform, and without the readfirstlane the predicate lands in a
VGPR and extends a **vector** live range.

**B1 — `KSTART`.** Anchor: the `__syncthreads();` at **line 889**, the last
statement of the M0 block, immediately before the closing `}` and
`if (debug_stop == 1) return;`.

```cpp
    __syncthreads();
    // exp_23: per-CTA epoch origin. The comment at the top of this kernel
    // (line ~707) correctly says a KSTART stamp in the COARSE block is dead,
    // because M0 zeroes that block. The per-CTA array is NOT zeroed by M0 and
    // this mark is AFTER the zeroing, so it survives. It is the x-axis origin
    // of the timeline and nothing else reads it.
    if (tid == 0) {
      std::uint32_t* mps_state0 =
          (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
      const hk_moe::mps::config cfg0 = hk_moe::mps::decode_config(
          (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
      hk_moe::mps::e23_mark(
          (std::uint64_t*)(mps_state0 + K0P6_MPS_ST_WORDS), bid,
          K0P6_MPS_E23_KSTART,
          __builtin_amdgcn_readfirstlane((int)cfg0.timestamps) != 0);
    }
```

Placement rationale: after M0's grid barrier and after the MPS state zeroing, so
(a) it is not wiped, and (b) all 256 CTAs are already rendezvoused, so the spread
of `KSTART` across CTAs is launch skew *after* the retire wait — which is the
honest origin for a per-layer timeline.

**B2 — `M8_ENTER`.** Anchor at line 1971, immediately after `payload_ok` is
computed in the M8 scope:

```cpp
    const int T = payload_ok ? (int)k0p6_dread(desc, K0P6_D_T) : 0;
```
Insert **before** that line:
```cpp
    if (tid == 0) {
      hk_moe::mps::e23_mark(mps_ts, bid, K0P6_MPS_E23_M8_ENTER,
                            m8cfg.timestamps);
    }
```
`mps_ts` and `m8cfg` are both already in scope (lines 1946 and 1933). This mark
is therefore **as cheap as a Tier A mark** despite being new — no extra
descriptor read, no extra decode. If Tier B has to be cut, cut B1 and B3 and
keep B2.

**B3 — `M9_DONE`.** Anchor: the end of the M9 block, line 2149-2150
(`      }\n    }\n  }`). Insert as the last statement inside the
`if (tid == 0)` at line 2113, **after** the `if ((old % gridDim.x) == ...)`
block closes:

```cpp
      {
        std::uint32_t* mps_state9 =
            (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE);
        const hk_moe::mps::config cfg9 = hk_moe::mps::decode_config(
            (std::uint64_t)k0p6_dread(desc, K0P6_D_MPS_CFG));
        hk_moe::mps::e23_mark(
            (std::uint64_t*)(mps_state9 + K0P6_MPS_ST_WORDS), bid,
            K0P6_MPS_E23_M9_DONE,
            __builtin_amdgcn_readfirstlane((int)cfg9.timestamps) != 0);
      }
```

**M9 hazard, read this before applying B3.** The elected last-arriver inside
that `if` resets `combine_done[0] = 0u` and pokes `retired`, which releases
peers into the next epoch. Placing the mark **after** that publication means a
peer can already be in epoch N+1 while this store lands — harmless, because the
cell is per-CTA and last-write-wins, but it means `M9_DONE` for CTA 0 on the
final soak epoch can be a *next-epoch* value on a rank that gets re-launched.
The parser's window check (§6.1) catches it. If it ever fires, move B3 to before
`release_signal_batch_agent()` and accept that it then excludes the retire
pokes.

### 4.3 Tier C — service and M7.5

**C1/C2 — `SVC_ENTER` / `SVC_EXIT`.** In the M7.6 block. Anchor at line 1565
(`if (service_cta_here) {`) and at line 1614 (`run_service(env, ...)`):

```cpp
    if (service_cta_here) {
      if (tid == 0) {
        std::uint64_t* tsv = (std::uint64_t*)(
            (std::uint32_t*)k0p6_dread(desc, K0P6_D_MPS_STATE) +
            K0P6_MPS_ST_WORDS);
        hk_moe::mps::e23_mark(tsv, bid, K0P6_MPS_E23_SVC_ENTER,
                              scfg.timestamps);
      }
      ...
        hk_moe::mps::run_service(env, *mps_scratch, wid, lane);
        if (tid == 0) {
          hk_moe::mps::e23_mark(env.ts, bid, K0P6_MPS_E23_SVC_EXIT,
                                scfg.timestamps);
        }
```

Note `env.ts` (line 1587) is already the u64 base, so `SVC_EXIT` is free;
`SVC_ENTER` needs its own read because it precedes `env`'s construction. If
Tier C is trimmed, keep `SVC_EXIT` and drop `SVC_ENTER` — the service interval's
left edge can be taken as this CTA's `M6_DONE` (service CTAs skip M7 entirely,
so they enter the drain immediately).

**Explicitly NOT done:** `design.md`'s per-stripe (`SVC_STRIPE_DONE`) and
per-flush (`SVC_FLUSH`) events. Those live inside `run_service`'s push/flush hot
loop — the exact code the figure is *about*. Instrumenting it is the highest-risk
edit in the whole plan and it is the one place where a store could land between
a payload store and its flush fence. The stripe **count** in slot 15 gives the
host the same texture information (stripes ÷ interval = mean stripe rate)
without touching the loop.

**C3/C4/C5 — M7.5.** Anchors: line 1740 (`hk_moe::release_cta_payload_system();`),
line 1751 (`const bool payload_ok = ...`), line 1841 (`__syncthreads();` at the
end of the `coarse75` block). One `tid == 0` mark before the release, one after
the barrier+acquire, one at the end. `cfg75` is already decoded at line 1724.
**This is the block the source itself flags as an allocation cliff
(lines 1693-1700).** If any tier fails the parity gate, this is the first thing
to delete.

**C6 — `META`.** In M8, after the dynamic-ticket `while` loops, `tid == 0`
writes `e23_meta(mps_ts, bid, is_service ? 1 : 0, batches_seen, ...)`. Requires a
per-CTA counter in the ticket loop, i.e. one new live integer inside M8. **Lowest
value, highest incremental risk of Tier C — drop it first.** The parser infers
role from `M7_DONE == 0` without it.

---

## 5. Edit 3 — host side (two lines, no ABI change, no rebuild)

File: `~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py`.
**Not in this repo and not owned by the kernel implementer.**

**H1 — grow the buffer.** Anchor at line 1520:

```python
        # 8 u32 words followed by 8 u64 timestamps = 96 bytes.
        _pf6_mps_state = torch.zeros((12,), dtype=torch.int64, device="cuda")
```
becomes
```python
        # 8 u32 words, 8 u64 timestamps, then exp_23's 256x16 u64 per-CTA phase
        # stamps = 96 + 32,768 = 32,864 bytes. The descriptor is UNCHANGED (63
        # words); only slot 60's buffer grew, so no ABI, bridge or assert moves.
        _pf6_mps_state = torch.zeros((12 + 256 * 16,), dtype=torch.int64,
                                     device="cuda")
```
Optionally update `R["pf6mps"]["buffer_bytes"]["state"]` at line ~1560 from `96`
to `32864`.

**H2 — dump it.** Anchor: the `[MPS SPIN]` `try:` at line ~5112, inside the
existing `if rank == 0:`. Append after it:

```python
            # --- exp_23 (Distributed-HipKittens overnight): ADDITIVE diagnostic.
            # Per-CTA phase stamps from the FINAL soak epoch. Print-only; one
            # line per CTA so the parser needs no binary channel. Guarded so a
            # shape change can never fail a run.
            try:
                _e23 = _mps_ts_t.detach().cpu().tolist()
                if len(_e23) >= 12 + 256 * 16:
                    print(f"[MPS E23] slots=16 ctas=256 tick_ns=10 "
                          f"cfg={os.environ.get('K0_MPS_CFG','')}", flush=True)
                    for _b in range(256):
                        _row = [int(x) & 0xFFFFFFFFFFFFFFFF
                                for x in _e23[12 + _b * 16: 12 + (_b + 1) * 16]]
                        if any(_row):
                            print("[MPS E23 CTA] " + str(_b) + " " +
                                  " ".join(str(v) for v in _row), flush=True)
            except Exception as _e3:
                print(f"[MPS E23] unavailable: {type(_e3).__name__}: {_e3}",
                      flush=True)
```

`_mps_ts_t` is already resolved at line ~5085 and already `int64`. 256 extra log
lines per arm per process; at 5 processes × 3 arms that is 3,840 lines, which is
noise next to the existing logs. The parser in §6 consumes exactly this format.

**Why print rather than write a file:** the campaign's container writes to `/out`
and only a fixed set of artifacts is copied back; stdout is already captured per
run (`run<N>.log`) and is already how `[MPS TS]` reaches us. Zero new plumbing.

---

## 6. Validation commands

### Parity gate (CPU only, no GPU)

`tools/e23_gate_build.sh` in this folder. It is `e27_build.sh`'s structure with
three TUs:

* `R` — the node checkout as-is (the arm being published),
* `A0` — the exp_23 source with **every tier compiled in**, `timestamps` off at
  runtime (the runtime flag does not change codegen, so `A0` *is* the shipped
  binary),
* `A1..A3` — one TU per tier, to localize a failure.

Gates, all read from `-Rpass-analysis=kernel-resource-usage` on
`k0pf6gm_mps_mega` plus the disassembly:

| gate | requirement | why |
|---|---|---|
| G0 | compiles clean | — |
| **G1** | `SGPR 106 / ArchVGPR 256 / AGPR 256 / ScratchSize 128 B / LDS 155,496 B` **exactly** | the tuple in `STATUS.md:145` for the mode-14 build; if the ring costs any of these the traced timeline is not the arm's timeline |
| **G2** | `Occupancy [waves/SIMD]: 1` printed and equal to 1 | **asserted, never assumed.** A sibling experiment saw LDS shrink → by-reference pointer arrays migrate LDS→scratch → occupancy 1→2. Nothing here shrinks LDS, but the gate must say so out loud, and G1's LDS-exact term is what makes it impossible. |
| **G3** | `v_mfma` census `== 180`; `flat_atomic_pk_add_bf16 == 282` | the K-loops and the epilogue's atomic count are untouched |
| **G4** | `scratch_load`/`scratch_store` count inside both MFMA spans `== 0` | the standing rule |
| G5 | `SGPR spills ≤ 186`, `VGPR spills ≤ 14` | the exp_26/exp_27 reference; a rise here is the early warning that G1 is about to fail on the next edit |
| G6 | `.text(A0) != .text(R)` | proves the instrument actually compiled in — an accidental `#if`-out would pass G1..G5 trivially |
| G7 | `s_memrealtime` count rises by exactly the number of Tier-B/C sites (Tier A must add **zero**) | proves `ts_mark` shared the clock read instead of adding one |

G7 is the check that catches the most likely silent mistake: writing
`ts_last(...); e23_mark(...)` instead of `ts_mark(...)`, which doubles the clock
reads and breaks the exactness of §6.2's reconciliation.

### Data collection (GPU)

The stamps come from the **final soak epoch**, so a *screen* is sufficient —
no 5-process campaign is needed for the figure.

```bash
setsid timeout 1800 env \
  K0_MOK_ARMS=production,pf6gm_mega,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-e23-r12 \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh e23r12 1
```

then, per arm,

```bash
python3 tools/parse_events.py \
    --log  $HOME/k0-mok-e23-r12/run1/run1.log \
    --arm  mps_mega --cfg "C=16,g=353,mode=12,flush_rows=16,timestamps=1" \
    --summary $HOME/k0-mok-e23-r12/summary.json \
    --out  rank0_events.json
python3 tools/bin_timeline.py --events rank0_events.json \
    --bin-us 10 --out timeline_bins.csv --check-integrals
```

**`plan.md`'s validation command is wrong on two counts** and is superseded by
the above: `K0_MPS_TRACE=1` writes `progress_rank<N>.log` (three host-side file
appends, no device data) and has nothing to do with the stamps; and `timestamps`
is a **`K0_MPS_CFG` key**, not an environment variable — a config missing any of
the four required keys raises `ValueError` on all eight ranks and *looks like a
pass* (empty run dir, no `[MOK GATE]` line).

### Arms

| panel | how | note |
|---|---|---|
| ratchet | `C=16,g=353,mode=12,flush_rows=16,timestamps=1` | the published arm |
| bulk / non-overlapped | `C=0,g=1,mode=0,flush_rows=1,timestamps=1` | see §7 — this is **not** `pf6gm_mega` |
| `production` | torch profiler, separate run | see §8 |

---

## 7. `pf6gm_mega` cannot carry the ring — the substitution, stated honestly

`plan.md` and `design.md` both assume the in-kernel instrument reaches
`pf6gm_mega`. **It cannot.** `pf6gm_mega` is a different kernel built from
`k0pf6gm_mega.hip` with a **55-word** descriptor
(`assert pf6_state["desc_gm"].numel() == 55`, `e004pf_k0pf_ab.py:~2807`) and no
`K0P6_D_MPS_STATE` slot at all. Instrumenting it means a second kernel edit plus
a second host allocation plus a second descriptor, in a file this spec does not
own — several hours, not twenty minutes.

**Substitute `mps_mega` at `C=0, g=1, mode=0, flush_rows=1`.** It is legal
(`config_is_valid`: mode 0 is not a stream mode and not a diag pool, so the
`C == 0` rejections at `moe_mps_adapter.cuh:434` and `:436` do not apply, and the
`mode == 1` C-must-be-zero rule at `:443` is irrelevant). At `C = 0`,
`is_service_cta` is false for every CTA, so all 256 run M7 at the donor's
full-grid stride; M7.5 runs the parity bulk publication and M8 pulls the
producer's remote `part` row. That is a **bulk, non-overlapped, homogeneous
transport arm inside the same kernel, same descriptor and same resource tuple**
as the ratchet — which is scientifically the right control for a timeline (one
binary, one tuple, one clock) even though it is not the literal `pf6gm_mega`
build.

What this costs in honesty, and what `result.md` must say: mode 0 carries the MPS
descriptor and the parity publish/pull path, so its end-to-end µs is **not**
`pf6gm_mega`'s. Report both numbers side by side from the same screen
(`pf6gm_us` and the mode-0 `mps_us`) so a reader can price the substitution, and
label the panel "bulk, no overlap (mode 0)" — never "homogeneous baseline".

---

## 8. The `production` arm — what is comparable and what is not

**Capture.** There is no profiler hook in the harness and adding one needs both a
harness edit and an `-e` forwarding entry, so the trace comes from a **standalone
8-rank replay** of one production MoE layer at the campaign's shapes (`T=4096`,
`H=7168`, `TOPK=8`, `world=8`, `K0_PF6GM_G=3`, same `K0_SYNTH_ROUTE` /
`K0_SYNTH_SEED`), wrapped in

```python
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False) as prof:
    for _ in range(3):
        production_layer()          # warm; take the LAST iteration
        torch.cuda.synchronize()
prof.export_chrome_trace("prod_rank0.json")
```

Keep only `ph == "X"`, `cat` in `{kernel, gpu_memcpy}` events from the final
iteration. Map kernel name → resource class with `b0_kernel_map.json`
(`rccl*`/`mori*`/`*ll128*` → xGMI, `*gemm*`/`*_mfma*`/`aiter*` → MFMA,
`*scatter*`/`*quant*`/`*combine*`/`*index*` → HBM); **any unmapped kernel name is
an error, not a default** — the parser must refuse rather than silently drop time.

**This run is NOT gate-covered.** A direct `torchrun` defaults
`K0_MOK_ABSOLUTE_TOLERANCE` to 1.0 and `K0_PF6GM_G` to 2 (harness recipe §2.3),
so `K0_PF6GM_G=3` must be set explicitly and the run's *numerics* must not be
quoted for anything. Its only output is kernel interval widths.

**Time base.** Both `torch.profiler` device timestamps on ROCm and
`s_memrealtime` derive from the HSA system clock, so they share a *rate*
(100 MHz, 10 ns ticks) — but they are different processes and different launches,
so **there is no common origin and no absolute alignment is possible.** Do not
fabricate one. Normalize each arm to its own layer start = 0 and give the three
panels a shared *duration* axis. If a single shared axis is wanted, the only
defensible one is normalized time (fraction of that arm's own layer).

**Genuinely not comparable, and each of these must be visible in the figure:**

1. **Units differ.** `production`'s strips are *binary whole-GPU kernel
   occupancy*; the megakernel's MFMA strip is *fraction of 256 CTAs in an MFMA
   phase*. Draw them differently (solid bars vs filled curve) and label them
   differently, or the figure claims a measurement it does not have.
2. **The megakernel's MFMA strip is a phase-occupancy proxy, not MFMA
   utilization.** A CTA stalled on `vmcnt` inside M7 counts as "in MFMA".
   `result.md` §5's rocprof check bounds how wrong that is; the caption must
   carry it. Note also that `amd-smi`'s `gfx_activity` **cannot** bound it — it
   saturates near 100 % for the whole soak (`result.md` §5).
3. **`production` has host launch gaps and stream dependencies inside the
   layer.** Those gaps are real and part of its cost. Do not smooth them.
4. **Different GEMM.** `production` is AITER; its MFMA mix, tiling and
   efficiency are not the megakernel's, so "MFMA busy" is not the same quantity
   even where both are measured.
5. **`production`'s xGMI traffic is MoRI/RCCL protocol traffic** the analytic
   byte table does not model. Its xGMI strip must come from counters or be left
   out — never from the megakernel's byte table.
6. **Different iterations.** The megakernel stamps are the final *soak* epoch;
   the production trace is a warm steady-state iteration. Same regime, not the
   same iteration.

The one number that **is** quantitatively comparable across all three panels is
the **communication share of layer time**, because it is a ratio of intervals
within each arm's own layer. That is the Fig-1 quantity and it is what the
overlay should be annotated with.

---

## 9. Go / no-go

**Go, at Tier A. Conditional go at A+B. No-go at A+B+C until gated.**

Honest probability that the instrument perturbs the measurement (i.e. the parity
gate fails), by tier:

| tier | P(fail) | dominant mechanism |
|---|---|---|
| A (5 marks, all in existing stamp blocks) | **~15 %** | whole-function allocation is not local; 5 extra stores can shift a spill decision even though nothing new is live |
| A+B (+3 marks, 2 new decodes) | **~30 %** | B1 and B3 each add a descriptor read + config decode in a new scope |
| A+B+C (+6, incl. M7.5) | **~50 %** | the M7.5 block is documented *in the source* (lines 1693-1700) as sitting on an allocation cliff: a second copy of one release sequence there cost **+16 B/lane scratch, VGPR spills 15→21, SGPR spills 186→221** |

**The risk that worries me most is not any of those** — a parity failure is loud
and I fall back a tier. The one that is *quiet* is **G7**: if the implementer
applies Tier A as `ts_last(...)` followed by `e23_mark(...)` instead of the fused
`ts_mark(...)`, everything compiles, the tuple may still pass, the figure looks
fine — and the per-CTA cells no longer reconcile exactly with the coarse stamps,
because the two calls read the clock at different instants. The reconciliation
check would then be silently downgraded from a theorem to a ±few-tick
approximation, and the one instrument that could have proven the ring indexes the
right cell in the right epoch stops proving it. Hence G7 is a **hard gate**, not
a diagnostic.

**Drop order if the gate fails**, most expendable first:
C6 (`META`) → C3/C4/C5 (M7.5) → C1 (`SVC_ENTER`, keep `SVC_EXIT`) → B3 (`M9_DONE`)
→ B1 (`KSTART`) → B2 (`M8_ENTER`) → A5 (`REDUCE_DONE`). The floor that still
produces the figure is **A1-A4**: per-CTA `plan` / `M6` / `M7` intervals, with
the x-axis origin taken from `min(M2_DONE)` instead of `KSTART`. That loses the
dispatch strip and the combine strip's right edge, and it is still the
NanoFlow-v2 Fig-10 panel.

---

## 10. What this spec found wrong in `plan.md` / `design.md`

1. **Ratchet identity.** `plan.md` names `mps_mega C=64 g=1 mode 2` as the
   ratchet. It is `C=16, g=353, mode=12, flush_rows=16`, and after exp_27 the
   number is **6,482.7 µs / 0.8408×**, not the 6,568 / 0.8522 in `CLAUDE.md`.
2. **`pf6gm_mega` is not instrumentable.** §7. This is the largest structural
   error in the plan.
3. **Descriptor slot 63.** `design.md`'s "append after 62; bump
   `K0P6_MPS_D_LEN`" requires `moe_host_abi.hpp` + `mps_host_bridge.cpp` + a
   pybind rebuild + an assert change. Growing slot 60's buffer is one host line.
4. **Ring mechanics.** The append ticket, the `(phase_id << 56)` packing and the
   "drop with a count in entry 23" overflow policy are all unnecessary; fixed
   slots make overflow impossible and add no atomic.
5. **LDS figure.** `design.md` says LDS is "fully committed (155,428 B)"; the
   measured value is **155,496 B** (exp_27 `build.md`, exp_34). `CLAUDE.md`'s
   standing tuple line (`SGPR 104 / LDS 155,428`) is also stale against
   `STATUS.md:145`'s `SGPR 106 / LDS 155,496`. The gate must use 106 / 155,496.
6. **Per-stripe service events.** `design.md`'s ids 12/13 instrument
   `run_service`'s push/flush hot loop. Highest-risk edit in the plan for the
   least data; replaced by a count.
7. **"22 years, no wrap handling"** is right about the counter and irrelevant to
   the real hazard, which is **epoch** ambiguity, not wrap: cells are not reset,
   so the parser needs the window check in the tooling (§6.1 of `result.md`
   schema), which `design.md` does not mention.
8. **Validation command.** `K0_MPS_TRACE=1` is not the stamp switch;
   `timestamps=1` inside `K0_MPS_CFG` is.
9. **Integral tolerance.** `plan.md`'s flat ±10 % is right for the end-to-end
   reconciliation and far too loose for the coarse reconciliation, which is
   exact by construction, and too loose for the interior closure, which is
   arithmetic (±1 µs).
10. **`KSTART` is not dead.** Line ~707 of the kernel says a KSTART stamp was
    removed as provably dead. True for the M0-zeroed coarse block; **false** for
    an unzeroed per-CTA cell marked after M0's `__syncthreads()`.
