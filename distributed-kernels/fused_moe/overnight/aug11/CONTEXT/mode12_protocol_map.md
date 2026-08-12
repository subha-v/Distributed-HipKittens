# mode 12 (`kModeRemoteAccum`) — structural protocol map

Scope: `mps_mega` (`k0pf6gm_mps_mega`) at `C=16, g=33, mode=12, flush_rows=16`,
world 8, MI350X/gfx950. Read-only map; no performance claims.

Short file names used below:

| tag | path (repo-relative) |
|---|---|
| `KRN` | `distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip` (1,864 lines) |
| `ADP` | `distributed-kernels/fused_moe/moe_mps_adapter.cuh` (966 lines) |
| `P2` | `distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp` (659 lines) |
| `ABI` | `distributed-kernels/fused_moe/moe_host_abi.hpp` (275 lines) |
| `HKA` | `distributed-kernels/fused_moe/moe_hk_adapter.cuh` (207 lines) |
| `SYNC`/`CMPL`/`ROLES`/`PKT`/`PEER`/`CTR`/`LIFE` | `include/cdna4/ops/group/distributed/{sync,completion,roles,packet,peer,counter,lifetime}.cuh` |
| `DRV` | `.node/mps_remote_e004pf_k0pf_ab.py` (node mirror of `~/amd-master/.../prefill_opt/host/e004pf_k0pf_ab.py`) |
| `CMP` | `.node/mps_remote_run_campaign.sh` (node mirror of the campaign `run_campaign.sh`) |

`g=33` decodes as: physical `g = 33 & 0xF = 1` (`ADP:205,218-221`),
detect bit `0x10` **clear** (`ADP:206,213-216` ⇒ `detect_dual == false`),
throttle bit `0x20` **set** (`P2:272`). `pull_fallback = 0` (forced by
`ADP:268`). `timestamps = 0` unless explicitly requested.

---

## 1. Phase map

Every phase re-reads its pointers through `k0p6_dread` (`KRN:204-209`), a
volatile descriptor load; nothing but `desc` lives across phases.

| phase | source span | computes | reads | writes |
|---|---|---|---|---|
| entry guard | `KRN:613-691` | shape + MPS config validation; `T_ext = world*MAXTOK` (`KRN:688`) | `desc[TOPK,E,CUR,WORLD,TLOCMAX,PADMAX,MAXTOK,T,SYMMETRIC,MPS_*]` | `pperr` on failure (`131072`, `K0P6_MPS_ERR_CONFIG`) |
| **M0** | `KRN:692-797` (`debug_stop==1` exit `KRN:798`) | peer-retirement wait, per-epoch zeroing | `retired[0..7]` (system poll `KRN:702-704`), `mega_count` | `a2_done`, `part_done` (`723-726`); `mps_q`, `mps_arr`, `mps_pushed`, `mps_claim`, `mps_state` (`731-749`); `dest_counter[(epoch+1)&1][*]` (`762-772`) |
| **M1** | `KRN:800-1012` (`debug_stop==2` exit `1013`) | quantize + LL128 push of this rank's `T` hidden rows to their destination ranks | `my_ids`, `my_wgt`, `hidden` | peer `a_ll` rows (`936-941`), `dest_counter[epoch&1][cur]` on each dest (`846-854`), `pull_stage` (`859-860`), `pull_cnt` (`862`), `pushed_count` (`934`), `rows_done` + `chunk_ready` on all peers (`984-1008`) |
| **M2** | `KRN:1016-1211` (`debug_stop==3` exit `1212`) | chunk-acquire, unpack `a_ll` → `a_dst`/`sc_stage`/`recv_eid`/`recv_wgt`, per-expert histogram, `row_remaining` | `chunk_ready`, `rows_done`, `a_ll` | `a_dst`, `sc_stage`, `recv_eid`, `recv_wgt`, `row_remaining` (`1168`), `recv_eid = -1` hole sentinels (`1187-1207`), `hcnt` (`1210`) |
| **M3–M5** | `KRN:1214-1315` (`debug_stop==4` exit `1316`) | expert histogram reduce + scan, `sei` cursors, `tile_desc`/`num_tiles`, CSR `pull_ptr`, `part` zero + scale transpose, pad + scatter + `pull_src` fill | `hcnt`, `recv_eid`, `recv_wgt`, `pull_cnt`, `pull_stage`, `sc_stage` | `scratch`, `nvi`, `sei`, `tile_desc`, `num_tiles` (`1239-1273`), `pull_ptr` (`1277`), `part`+`sc_dst` (`1284-1288`), `sti`/`swt` (`1296-1297`), `pull_src` (`1299`) |
| **M6** | `KRN:1318-1330` (`debug_stop==5` exit `1331`); body `.node/mps_n2_phase1_gm.cpp` | GEMM-1 (W13 gate/up, SiLU, fp8 requant). Task = (tile, intermediate chunk `g∈[0,8)`) | `a_dst`, `sc_dst`, `W13`, `S13`, `sti`, `sei`, `nvi`, `tile_desc` | `a2q`, `dq2`; per live sub-block `a2_done[b]` (`KRN:227-234`) |
| **M6.9** | `KRN:1346-1365` (comment), predicate `KRN:1368-1370` | **role reservation point** — static tail, no barrier, no ticket | `desc[K0P6_D_MPS_CFG]` | nothing |
| **M7** | `KRN:1366-1392` (`debug_stop==6` exit `1393`); body `P2:198-567` | GEMM-2 split-N full-K, write-once epilogue. Compute CTAs only (`bid < 240`) | `a2q`, `dq2`, `W2`, `S2`, `sti`, `swt`, `sei`, `nvi`, `tile_desc` | **mode 12: remote `slots` via `global_atomic_pk_add_bf16`** (`P2:148-149`); event queue `mps_q` + `mps_state[TAIL]` (`KRN:324-325`) |
| **M7.6** | `KRN:1395-1467`; body `ADP:638-838` | service drain: event dequeue → `nc_arr` arrivals → `pushed` chunk count → batched `row_ready` flags. **All 256 CTAs enter** (`KRN:1414`) | `mps_q`, `sti`, `row_rem` | `nc_arr`, `pushed`, `row_ready` (owner-only), `row_rem := 0` |
| M7.6c | `KRN:1469-1509` | diag pool bodies, modes 5/6 | — | — (**skipped in mode 12**) |
| M7.5 | `KRN:1511-1589` | parity bulk publication + its grid barrier | — | — (**skipped in mode 12**, gated by `mode_is_parity_publish`, `ADP:229-231`) |
| M7.6b | `KRN:1591-1658` | mode-1 bulk push | — | — (**skipped in mode 12**) |
| **M8** | `KRN:1660-1820`; batch body `KRN:471-610` | combine: poll `row_ready`, reduce 8 slot rows → `out`, **consume-and-zero** the slot rows | `pull_ptr`, `pull_src`, `row_ready`, local `slots` | `out` (`KRN:602`), zeroes `slots` rows (`KRN:579-587`), `mps_state[M8NEXT]` |
| **M9** | `KRN:1822-1862` | counted grid arrival, resets, retirement pokes | `combine_done` | `pushed_count[0..7]`, `qpush_done`, `combine_done`, `mega_count`, `retired[cur]` on self + 7 peers |

### Grid barriers (mode 12)

`hkp::grid_barrier(bar, tid, bar_err)` on the shared `gbar` cell, cumulative
target (no reset):

1. `KRN:716` — M0 retirement rendezvous.
2. `KRN:1101` — M2 Pass A → Pass B.
3. `KRN:1236` — M3.
4. `KRN:1291` — M4.
5. `KRN:1301` — M5.

`KRN:1555` (M7.5) and `KRN:1611` (mode 1) are the only other grid barriers in
the file and **both are mode-gated off in mode 12**. There is therefore **no
grid barrier anywhere between M6 and the end of the kernel** — the M9
`combine_done` counted arrival (`KRN:1828`) is the only grid-wide rendezvous
after M5, and it is a counter, not a barrier.

`hkp::grid_barrier` itself lives in `hkp_sync.hpp`, outside this repo.
**UNRESOLVED:** the per-barrier atomic/fence count is not determinable here.

### Reservation / `role_partition`

- The library's `finish_order_partition` (`ROLES:62-124`) is **not called**.
  `DESIGN_MPS.md:118-125` records it as rolled back (+24 B/lane spill at the
  phase-2 pointer peak).
- The shipped mechanism is a static tail predicate: `is_service_cta`
  (`ADP:292-297`) = `bid >= nct - C`. At `C=16, nct=256` ⇒ service =
  `bid ∈ [240, 256)`, compute = `bid ∈ [0, 240)`.
- Dense compute id: `compute_id_of` (`ADP:299-306`) returns `bid` verbatim for
  every mode except 3. M7 start = `k0p6_mps_task_start` (`KRN:280-285`) =
  `blockIdx.x`; stride = `k0p6_mps_stride` (`KRN:273-276`) = `nct - C = 240`.
  Wired in at `KRN:419-420`.
- `service_id_of` (`ADP:308-316`) is used only by the mode-5/6 diag pool
  (`KRN:1487-1488`); mode 12 never calls it.

---

## 2. M7 task/epilogue structure in mode 12

### Task indexing

- Task space (`P2:283-284`): `num_tasks = num_tiles * 16`.
- `tile = task / 16`, `nc = task % 16` (`P2:331-332`).
- `b0 = tile_desc[tile] >> 4`, `gcount = tile_desc[tile] & 0xF` (`P2:295-300,
  333-335`). `tile_desc` is built in M4 (`KRN:1250-1273`): each local expert
  `e` contributes `ceil(nsb_e / G)` tiles with `G = K0P6GM_G = 3`
  (`KRN:192-197`), `b0 = blk0_e + t*G`, `gcount = min(G, nsb_e - t*G)`.
- A task therefore owns `gcount` consecutive same-expert 32-blocks
  (`kMrows = 32*G = 96` sorted rows max) × one 448-column N-chunk. Output tile
  shape per sub-block: **32 rows × 448 columns**, bf16.
- Task loop: `for (task = blockIdx.x; task < num_tasks; task += 240)`
  (`P2:326-327` with the macros above).

Per-rank counts, exact formulas:

```
blocks           B    = nvi[0] / 32                       (nvi[0] = padded row count, M4 scan)
tiles            num_tiles = sum_e ceil(nsb_e / 3)
tasks            num_tasks = 16 * num_tiles
task-sub-blocks  sum_tasks gcount = 16 * B                (= events_total, KRN:1418)
tasks per CTA    num_tasks / 240
```

Measured/derived campaign values (see §7 for provenance):
`B ≈ 1,045`, `events_total ≈ 16,720` (`ADP:635`), `num_tiles ≈ 350-380`,
`num_tasks ≈ 5,600-6,100`, ≈ 23-25 tasks per compute CTA.

### The mode-12 epilogue path

Peer table construction runs **once per body**, before the task loop
(`P2:234-278`), guarded by `mode_is_direct_accum` (`P2:249-250`):

| symbol | value | line |
|---|---|---|
| `m7tab[8]` (LDS, 64 B) | `tid==cur ? m7sym->local_heap_base : m7sym->heap_bases.select<8>(tid)`, filled by threads 0..7 | `P2:238, 257-262` |
| `m7_sh` | `31 - __clz(MAXTOK)` = `log2(4096)` = **12** | `P2:263` |
| `m7_tok_mask` | `MAXTOK - 1` = **4095** | `P2:264` |
| `m7_slot_off` | `(slots_ptr - local_heap_base) + cur * MAXTOK * kHidden * 2` | `P2:265-268` |
| `m7_dual` | `detect_dual(cfg)` = **false** at `g=33` | `P2:270-271` |
| `m7_throttle` | `((cfg >> 8) & 0x20) != 0` = **true** at `g=33` | `P2:272` |

Guard: `mode_is_direct_accum && (MAXTOK & (MAXTOK-1)) != 0` is a hard reject
(`KRN:678-683`) — the shift/mask above requires a power-of-two `MAXTOK`.

The accumulate (`P2:131-167`), inside `epilogue_write<JMAX>`:

```
if (dcol < 8 * JMAX)                                            // P2:131
  for i in [0,16):                                              // P2:139
    row = 2*i + rowh;   d = xp[row][dcol ^ (2*(row&15))];        // P2:140-141
    if (xtok[i] < T) {                                           // P2:142
      xr = (unsigned)xtok[i];                                    // P2:143
      a  = peer_tab[xr >> 12] + slot_off
           + ((xr & 4095) * 7168 + col_base + 2*dcol) * 2;       // P2:144-147
      accumulate_peer_bf162((void*)a, d);                        // P2:148-149
      if (throttle) asm volatile("s_waitcnt vmcnt(8)");           // P2:150-159
      if (dual) unsafeAtomicAdd(part + xtok[i]*7168 + ...);      // P2:160-165  (OFF)
    }
```

`accumulate_peer_bf162` (`PKT:202-211`) is `unsafeAtomicAdd` on a
`__hip_bfloat162*` ⇒ one `global_atomic_pk_add_bf16`, **4 bytes**, no scope
argument, no return value, no fence; the RMW resolves at the coherence point
of the address (fabric-side for a peer VA — `PKT:177-201`).

Lane geometry (`P2:227-231, 312-313`): `lane ∈ [0,64)`, `rowh = lane>>5`,
`dcol = lane&31`. Column base per wave: `col_base = 448*nc + 112*wv`
(`P2:530`) for the `JMAX=4` call and `+64` (`P2:536-537`) for the `JMAX=3`
call. Each atomic covers columns `col_base + 2*dcol` and `+1`.

#### Atomic arithmetic per unit

Per wave, per live sub-block:

```
epilogue_write<4>:  dcol < 32 ⇒ all 64 lanes active × 16 rows      = 1,024 atomics
epilogue_write<3>:  dcol < 24 ⇒ 2 halves × 24 = 48 lanes × 16 rows =   768 atomics
                                                              wave total = 1,792
```

Per **CTA** (4 waves), per live sub-block: `4 × 1,792 = 7,168` atomics.

Byte check: `7,168 × 4 B = 28,672 B = 32 rows × 448 cols × 2 B` ✓ — the tile is
written exactly once, no split-K amplification.

Per **task**: `gcount × 7,168` atomics = `gcount × 28,672 B`
(at `gcount=3`: 21,504 atomics / 86,016 B).

Per **rank per epoch** (live rows only; padding lanes are EXEC-masked and form
no address, `P2:128-130`):

```
live sorted slots P = sum_r row_remaining[r]  ≈ 32,768      (= W*T*TOPK / W, balanced)
atomics = P × (7168 cols / 2 per atomic)
        = 32,768 × 3,584 = 117,440,512 lane-atomics
bytes   = 117,440,512 × 4 = 469,762,048 B = 448 MiB
cross-check: 32,768 slots × 7,168 cols × 2 B = 469,762,048 B ✓
remote share (uniform routing, owner != cur for 7/8 of slots):
        ≈ 102,760,448 atomics ≈ 392 MiB over xGMI
local  share ≈ 14,680,064 atomics ≈ 56 MiB
```

Note on the "~312 MB" figure in `ADP:176-178`: that is the **distinct slot
footprint** `R_live × 14,336 B = 21,816 × 14,336 = 312,754,176 B`, not the RMW
volume. Multiple local experts contributing to the same receive row alias onto
the *same* slot cell (the address depends only on `(owner, pos, col)`,
**not** on the expert — `P2:144-147`), which is exactly why a bf16 atomic is
required. Issued RMW bytes are 448 MiB; distinct bytes touched ≈ 298 MiB.

Throttle instruction count: `s_waitcnt vmcnt(8)` (`P2:157`) is a scalar
executed once per wave per `i` iteration with any active lane:
`16,720 sub-block-tasks × 4 waves × (16 + 16) = 2,140,160` wave-level
executions per rank per epoch.

### Owner / slot index math (M7 element → destination)

Let `rr = sti[b*32 + i] & 0x00FFFFFF` be the receive-row index of an M7 output
row (`P2:350`, masked; `sti` packs `token | slot<<24`). Then, on the producing
rank `cur`:

```
owner = rr >> 12          = rr / MAXTOK                          (P2:145)
pos   = rr & 4095         = rr % MAXTOK                          (P2:146)
col   = 448*nc + 112*wv [+64] + 2*dcol   (and col+1)             (P2:146, 530, 536-537)
destination = owner_rank's  slots[cur][pos][col]                 (P2:144-147)
```

The `owner` field is structural: M1 assigns `row = cur*MAXTOK + pos`
(`KRN:856`) where `cur` is the **source** rank of the token, so the high bits
of every receive-row index name the rank that owns the output token.

The mirror on the owner side, M8 `base_slot` (`KRN:1719-1725`):

```
base_slot(p, row) = slots + ((p*MAXTOK) + row - cur*MAXTOK) * 7168
                  = slots[p][pos]        since row = cur*MAXTOK + pos on the owner
```

⇒ producer `p` writes `owner.slots[p][pos]`; owner `cur` reads
`slots[p][row - cur*MAXTOK]`. Consistent.

---

## 3. THE ATOMIC CENSUS (M6 → M9 window, mode 12, C=16, g=33, flush_rows=16)

Symbols (per rank per epoch), all data-dependent quantities defined by formula
plus the campaign value:

| symbol | definition | value |
|---|---|---|
| `B` | live 32-blocks = `nvi[0] / 32` | ≈ 1,045 |
| `EV` | events = `(nvi[0] >> 5) * 16 = 16*B` (`KRN:1418`) | ≈ 16,720 |
| `P` | live sorted slots = `Σ_r row_remaining[r]` = (row,local-expert) pairs | ≈ 32,768 |
| `R` | live receive rows | ≈ 21,816 |
| `S` | completing slices = `R × 16` | ≈ 349,056 |
| `F` | fanout entries = `pull_ptr[T]` = `Σ_τ pull_cnt[τ]` | ≈ 21,504 |
| `NT8` | M8 batches = `ceil(T/4) = ceil(4096/4)` | 1,024 |
| `WV` | waves entering the drain = `256 CTAs × 4` | 1,024 |
| `TASKS` | `16 × num_tiles` | ≈ 5,600-6,100 |

### (a) M7 epilogue (compute CTAs, `bid < 240`)

| # | site | op | scope | target | count per rank per epoch | arithmetic |
|---|---|---|---|---|---|---|
| a1 | `P2:148-149` (`PKT:202-211`) | `global_atomic_pk_add_bf16` (RMW, 4 B) | address coherence point (system/fabric for peer VA; none requested by the builtin) | peer/local `slots[cur][pos][col]` | **117,440,512** | `P × 7168/2 = 32,768 × 3,584` |
| a2 | `P2:157` | `s_waitcnt vmcnt(8)` | wave | — | **2,140,160** (wave-level) | `EV × 4 waves × 32 i-iterations` |
| a3 | `P2:124` | `fence(ACQ_REL,"wavefront")` | wavefront | LDS transpose RAW | **133,760** | `EV × 4 waves × 2 epilogue_write calls` |
| a4 | `P2:183` | `fence(ACQ_REL,"wavefront")` | wavefront | LDS transpose WAR | **133,760** | same |
| a5 | `P2:125` | `s_waitcnt lgkmcnt(0)` (`0xc07f`) | wave | — | **133,760** | same |
| a6 | `P2:126` | `wave_barrier()` | wave | — | **133,760** | same |
| a7 | `P2:164` | `unsafeAtomicAdd` on local `part` (dual-write detector) | agent | `part[xtok][col]` | **0** | detect bit `0x10` clear at `g=33` |
| a8 | `P2:472,494,506` | `lds_cta_barrier()` | CTA | LDS double buffer | `TASKS × 17` ≈ **97,700** | 1 pre-loop + 2 per K-step × 8 steps |

`k0p6_mps_task_drain` (`KRN:344-353`, wired at `KRN:410`) emits **no**
`s_waitcnt vmcnt(0)` in mode 12 — the guard `if (m != 12 && m != 13)` at
`KRN:350` suppresses it. It still performs one volatile `k0p6_dread` of
`K0P6_D_MPS_CFG` per task.

### (b) Readiness / event enqueue on compute CTAs

Publication is **deferred by one task** (`KRN:332-383`, `P2:315-330, 555-566`):
`k0p6_mps_task_done_maybe_defer` (`KRN:366-383`) only buffers `(b0, gcount,
nc)` into `k0p6_defer m7_pend` (`KRN:338-342`, `P2:323`); the actual publish
happens in `k0p6_mps_task_flush_defer` (`KRN:355-364`) at the **head of the
next task** (`P2:328-330`) and once after the loop (`P2:564-566`).

| # | site | op | scope | target | count | arithmetic |
|---|---|---|---|---|---|---|
| b1 | `KRN:358` | `s_waitcnt vmcnt(0)` | wave | drains the previous task's remote RMWs | `TASKS` ≈ **5,900** | 1 per task with a pending |
| b2 | `KRN:359` | `__syncthreads()` | CTA | — | `TASKS` ≈ **5,900** | 1 per task |
| b3 | `ADP:386` (via `KRN:324-325`) | `fetch_add_relaxed<agent>` (RMW) | **agent** | `mps_state[K0P6_MPS_ST_TAIL]` — **one 32-bit cell, grid-wide** | `EV` = **16,720** | 1 per (b,nc) event |
| b4 | `ROLES:136` (`publish_tile_release`) | `fence(RELEASE,"agent")` | **agent** | — | `EV` = **16,720** | 1 per event |
| b5 | `ROLES:137` | `store_relaxed<agent>` (4 B) | **agent** | `mps_q[ticket]` | `EV` = **16,720** | 1 per event |
| b6 | `P2:551` | `__syncthreads()` | CTA | LDS reuse | `TASKS` ≈ **5,900** | 1 per task |
| b7 | `KRN:241-243` (`HKA:106-115`) | bounded relaxed loads on `a2_done[b]`, expected 8 | agent | `a2_done` | ≥ `EV` = **≥16,720** loads | 1 per live sub-block per task, ≥1 spin each |
| b8 | `KRN:248` → `HKA:88-90` → `SYNC:178-181` | `cta_barrier()` + `fence(ACQUIRE,"agent")` | **agent** | A2/DQ2 payload | `EV` = **16,720** of each | 1 per live sub-block per task |
| b9 | `KRN:249-250` (`HKA:196-205`) | `load_relaxed<agent>(pperr)` + `__shfl` | agent | `pperr` | `EV × 4` = **66,880** | one lane-0 load per wave |

**Documentation contradiction (resolved by the call site):** `ADP:377-380`
states "exp_21 mode 12 releases at SYSTEM scope". The call site
`KRN:324-325` invokes `enqueue_tile_release` with **no explicit template
argument**, so `EnqueueScope` takes its default `scope::agent`
(`ADP:381`). `KRN:305-316` independently and explicitly states the enqueue
stays AGENT for all modes including 12/13. **The compiled behaviour is agent
scope.** The `ADP:377-380` comment is stale.

For completeness, the M6-side producer of `a2_done` (`KRN:227-234`,
`HKA:92-97`): per live sub-block per M6 task, `arrive_local_count<true>` =
`fence(RELEASE,"agent")` + `fetch_add_relaxed<agent>(a2_done+b, 1)` ⇒
`8 × B ≈ 8,360` of each per rank per epoch (M6 tasks are `(tile, g∈[0,8))`).

### (c) Service pool loop — `run_service` (`ADP:638-838`)

**All 256 CTAs / 1,024 waves execute this** in mode 12: the predicate at
`KRN:1414` is `mode_is_stream(scfg)` (true for mode 12, `ADP:158-161`), not
`is_service_cta`. The 16 reserved CTAs arrive immediately (they skipped M7);
the 240 compute CTAs fall through as they drain (`KRN:1406-1413`).

| # | site | op | scope | target | count | arithmetic |
|---|---|---|---|---|---|---|
| c1 | `ADP:649-650` | `fetch_add_relaxed<agent>` (RMW) | **agent** | `mps_state[K0P6_MPS_ST_EVNEXT]` — one cell | `EV + WV` = **≈17,744** | 1 per claim + 1 terminating claim per wave |
| c2 | `ADP:453` | `load_relaxed<agent>` | agent | `mps_q[k]` | ≥ `EV` = **≥16,720** | ≥1 spin per event, unbounded above |
| c3 | `ADP:455` | `load_relaxed<agent>` | agent | `pperr` — **one word polled by all 1,024 waves** | ≥ `EV` = **≥16,720** | 1 per spin |
| c4 | `ADP:458` | `atomicOr(pperr, ...)` | agent | `pperr` | **0** in a passing run | timeout only |
| c5 | `ADP:670` | `thread_acquire<agent>` = `fence(ACQUIRE,"agent")` | **agent** | — | `EV` = **16,720** | 1 per event, wave-level |
| c6 | `ADP:676-677` | `atomicMax` (`ts_first`/`ts_last`, `ADP:357-366`) | agent | `mps_state` TS block | **0** | `timestamps=0` ⇒ early return |
| c7 | `ADP:690-691` | `load_relaxed<agent>` | agent | `row_rem[r]` | `P × 16` = **524,288** | 1 per live lane per event |
| c8 | `ADP:720-722` | `fetch_add_acq_rel<agent>` (RMW) | **agent** | `nc_arr[r*16 + nc]` | `P × 16` = **524,288** | 1 per live lane per event; **32 distinct cache lines per event** (32 different `r`) |
| c9 | `ADP:750-756` | group-probe `fetch_add_acq_rel<agent>(…, 0)` | agent | `nc_arr` | **0** | `g==1` fast path `ADP:731-747` |
| c10 | `ADP:764-765` | `atomicOr(claim + r, bit)` | agent | `claim` | **0** | `g==1` fast path ⇒ `claim` buffer is **dead in mode 12** |
| c11 | `ADP:780` | `atomicAdd(&s.push_count, 1)` | **LDS** | wave scratch | `S` = **349,056** | 1 per completing slice |
| c12 | `ADP:605-606` (`retire_pushed_row`, called at `ADP:796`) | `fetch_add_relaxed<agent>` (RMW) | **agent** | `pushed[r]`, target 16 | `S` = **349,056** | 1 per completing slice |
| c13 | `ADP:612` | `__syncwarp()` | wave | — | `S` = **349,056** | 1 per completing slice |
| c14 | `ADP:560` | `s_waitcnt vmcnt(0)` | wave | — | ≈ **1,364-2,400** | 1 per non-empty flush batch |
| c15 | `ADP:570` | `release_signal_batch_system()` = `thread_release<system>` = `fence(RELEASE,"")` | **system** | — | ≈ **1,364-2,400** | floor `ceil(R/16) = ceil(21,816/16) = 1,364`; measured ≈2,400 (partial per-wave remainders, `exp_21 result.md:84`) |
| c16 | `ADP:579` | `publish_epoch<agent>` = `store_relaxed<agent>` (4 B) | **agent** | own `row_ready[cur][r]` | ≈ `R/8` = **≈2,727** | rows whose owner == cur |
| c17 | `ADP:581-583` | `publish_epoch<system>(peer_ptr(...))` = `store_relaxed<system>` (4 B) | **system** | owner's `row_ready[cur][r]` | ≈ `7R/8` = **≈19,089** | one store per row, **owner only** (not 8) |
| c18 | `ADP:587` | `publish_relaxed<agent>(row_rem + r, 0)` (4 B) | agent | `row_rem[r]` | `R` = **21,816** | self-clean, one per flagged row |
| c19 | `ADP:562, 591` | `__syncwarp()` | wave | — | 2 × flush count ≈ **2,700-4,800** | |
| c20 | `ADP:834` | terminal `flush_pending` | — | — | `WV` = **1,024** calls | most return at `ADP:556` with `flag_count==0` |
| c21 | `ADP:836` | `ts_last(DRAIN)` | agent | `mps_state` TS | **0** | `timestamps=0` |
| — | `ADP:512-548` | `store_peer_packets` / `store_peer_packets_multi` (payload push) | — | `slots` | **0 — unreachable** | `ADP:798` `continue` short-circuits the mode-12 branch before the push loop |

### (d) M8 / M9 owner side

M8 instantiation for mode 12: `k0p6_mps_m8_batch<NT=4, base_slot&, base_slot&,
Detect=false, Zero=true>` (`KRN:1782-1799`).

| # | site | op | scope | target | count | arithmetic |
|---|---|---|---|---|---|---|
| d1 | `KRN:1788-1790` | `fetch_add_relaxed<agent>` (RMW) | **agent** | `mps_state[K0P6_MPS_ST_M8NEXT]` — one cell | `NT8 + WV` = **2,048** | 1,024 real batches + 1,024 terminating claims |
| d2 | `KRN:511-514` (`HKA:151-160`) | bounded `load_relaxed<system>` on `row_ready[p*T_loc_max + row]` | **system** | `row_ready` | ≥ `F` = **≥21,504** | 1 per fanout entry, ≥1 spin, unbounded above |
| d3 | `KRN:526` → `HKA:84-86` → `SYNC:107-109` | `thread_acquire<system>` = `fence(ACQUIRE,"")` | **system** | slot payload | `NT8` = **1,024** | 1 per batch that passes the ballot |
| d4 | `KRN:527` | `__syncwarp()` | wave | — | **1,024** | |
| d5 | `KRN:546-555` | plain `uint4` loads (16 B) | — | local `slots` | 469,762,048 B = **448 MiB** | `NT8 × (8 shuffle pos × 4 tok × 14 chunks × 64 lanes × 16 B)`; out-of-fanout lanes read the dummy base `base_slot(cur,0) = slots + 0` (`KRN:496-518`) |
| d6 | `KRN:579-587` | `uint4` zero store (16 B) — **consume-and-zero** | — | local `slots` rows | `F × 14,336` = **308,281,344 B ≈ 294 MiB** | fanout-guarded, same lane/16 B geometry as the read |
| d7 | `KRN:595-602` | `uint4` store (16 B) | — | `out[tok][*]` | `T × 14,336` = **58,720,256 B = 56 MiB** | `14 chunks × 64 lanes × 16 B = 14,336 B` per token |
| d8 | `KRN:607` | `atomicOr(pperr, K0P6_MPS_ERR_DUAL)` | agent | `pperr` | **0** | `Detect=false` |
| d9 | `KRN:1816-1819` | `ts_last(REDUCE_DONE)` | agent | `mps_state` TS | **0** | `timestamps=0` |
| d10 | `KRN:1824` | `__syncthreads()` | CTA | — | **256** | 1 per CTA |
| d11 | `KRN:1828-1829` | `__hip_atomic_fetch_add(combine_done, 1, ACQ_REL, AGENT)` | **agent** | `combine_done` — one cell | **256** | 1 per CTA |
| d12 | `KRN:1840-1847` | plain stores: `pushed_count[0..7]=0`, `qpush_done[0]=0`, `combine_done[0]=0`, `mega_count[0]=epoch` | — | — | **11** | last arriver only |
| d13 | `KRN:1848` | `release_signal_batch_agent()` = `fence(RELEASE,"agent")` | **agent** | — | **1** | last arriver only |
| d14 | `KRN:1851-1852` | `publish_value<agent>` (8 B store) | agent | own `retired[cur]` | **1** | |
| d15 | `KRN:1856-1857` | `publish_value<system>(peer_ptr(...))` (8 B store) | **system** | peers' `retired[cur]` | **7** | |

**Observation (not a claim):** `KRN:1848` issues an **agent**-scope release
before the seven **system**-scope `retired` pokes at `KRN:1856-1857`. In mode
12 the payload that retirement gates is M8's consume-and-zero of local `slots`
rows (`KRN:579-587`), which peers read/RMW over xGMI in the next epoch. This
is inherited verbatim from the parity port; flagged here only because mode 12
is the first mode whose slot lifetime depends on it.

### Totals per rank per epoch (mode 12, C=16, g=33, flush_rows=16)

**Global-memory atomic RMWs**

| class | count |
|---|---|
| payload (`global_atomic_pk_add_bf16`, a1) | **117,440,512** |
| `nc_arr` acq_rel agent (c8) | 524,288 |
| `pushed` relaxed agent (c12) | 349,056 |
| `ev_next` relaxed agent (c1) | ≈17,744 |
| event `TAIL` relaxed agent (b3) | 16,720 |
| `a2_done` relaxed agent (M6 producer) | ≈8,360 |
| `m8_next` relaxed agent (d1) | 2,048 |
| `combine_done` acq_rel agent (d11) | 256 |
| **protocol subtotal (non-payload)** | **≈918,472** |
| **grand total** | **≈118,358,984** |

Ratio: protocol RMWs are `918,472 / 117,440,512 ≈ 0.78 %` of all global RMWs
by count; by bytes the payload is 448 MiB against ~3.5 MiB of protocol words.

**Fences**

| fence | scope | count |
|---|---|---|
| `fence(RELEASE,"agent")` — b4 + M6 arrivals + d13 | agent | ≈25,081 |
| `fence(ACQUIRE,"agent")` — b8 + c5 | agent | **33,440** |
| `fence(RELEASE,"")` — c15 | **system** | **≈1,364-2,400** |
| `fence(ACQUIRE,"")` — d3 | **system** | **1,024** |
| `fence(ACQ_REL,"wavefront")` — a3 + a4 | wavefront | 267,520 |
| `s_waitcnt vmcnt(0)` — b1 + c14 | wave | ≈7,300-8,300 |
| `s_waitcnt vmcnt(8)` — a2 | wave | ≤2,140,160 |
| `buffer_wbl2` / `producer_drain_release` | system | **0** in M6→M9 |

`producer_drain_release` (`SYNC:166-174`, the `buffer_wbl2 sc1` lowering)
appears at `KRN:961` (M1), `KRN:1547` (M7.5, mode 0/5/6 only), `KRN:1606` and
`KRN:1640` (mode 1 only). **None of them execute in mode 12.**

**Flag / readiness stores**

| store | scope | width | count |
|---|---|---|---|
| `mps_q[ticket]` (b5) | agent relaxed | 4 B | 16,720 |
| `row_ready` self (c16) | agent relaxed | 4 B | ≈2,727 |
| `row_ready` peer (c17) | **system relaxed** | 4 B | ≈19,089 |
| `row_rem := 0` (c18) | agent relaxed | 4 B | 21,816 |
| `retired` self (d14) | agent relaxed | 8 B | 1 |
| `retired` peer (d15) | **system relaxed** | 8 B | 7 |

Cross-rank (system-scope) traffic in the whole M6→M9 window reduces to:
**≈19,089 four-byte `row_ready` stores + 7 eight-byte `retired` stores +
≈102.8 M four-byte remote atomics**, behind ≈1,364-2,400 system release fences
and 1,024 system acquire fences.

---

## 4. The readiness protocol, precisely

### The event queue

- Storage: `K0P6_D_MPS_Q` = descriptor slot 56 (`ADP:20`), `uint32[(PADMAX/32)
  * 16]` = `8,223 × 16 = 131,568` words = **526,272 B**, local (agent-visible
  only), allocated at `DRV:1492-1494` as `torch.zeros((PADMAX//32, 16), int32)`.
- Entry: **one 32-bit word** (`ADP:318-333`):
  `word = 0x80000000 | (b << 4) | nc`, `b < 2^14` (guard `PADMAX/32 > 16383`
  rejected at `KRN:669`), `nc ∈ [0,16)`. The MARK bit makes 0 mean "empty",
  which is what makes the M0 zeroing (`KRN:739`) a valid initializer.
- **Enqueuer:** the M7 compute CTA's `tid == 0`, one entry per live sub-block
  per task, via `enqueue_tile_release` (`ADP:381-389`): claim a monotonic
  ticket from `mps_state[K0P6_MPS_ST_TAIL]`, then agent-release + relaxed store
  into `q[ticket]`. Single writer per slot by construction (the ticket is
  unique).
- **Dequeuer:** any wave of any CTA in `run_service`, via a **claim ticket**
  from `mps_state[K0P6_MPS_ST_EVNEXT]` (`ADP:649-650`) — not a static stripe.
  Consumption stops at `k >= events_total` (`ADP:654`), where
  `events_total = (nvi[0] >> 5) * 16` (`KRN:1418`). Capacity is checked once
  (`KRN:1422-1423`, `K0P6_MPS_ERR_CONFIG` on overflow); structurally
  `events_total = 16·B ≤ 16·bcap = queue_capacity`.
- Waiting: `wait_event_nonempty` (`ADP:448-464`) spins on the slot with a
  relaxed agent load plus a relaxed agent load of `pperr`, bounded by
  `spin_limit` (`K0_SPIN_LIMIT=2000000`, `CMP:128`).

### `nc_arr`, `pushed`, `row_rem`, `claim`, `row_ready`

| buffer | slot | type/shape | bytes | visibility | reset | update rule |
|---|---|---|---|---|---|---|
| `nc_arr` (`K0P6_D_MPS_NCARR`) | 57 | `uint32[T_ext][16]` = `[32768][16]` | 2,097,152 | agent | M0 `KRN:740` | `fetch_add_acq_rel<agent>(nc_arr[r*16+nc], 1)` per live lane per event (`ADP:720-722`); the lane observing `old+1 == row_rem[r]` is the unique slice finisher |
| `pushed` (`K0P6_D_MPS_PUSHED`) | 58 | `uint32[T_ext]` | 131,072 | agent | M0 `KRN:741-743` | `fetch_add_relaxed<agent>(pushed[r], 1)` per completing slice (`ADP:605-606`); target **16** (`ADP:607`) — in mode 12 it counts *chunks done*, not pushes (`ADP:788-794`) |
| `claim` (`K0P6_D_MPS_CLAIM`) | 59 | `uint32[T_ext]` | 131,072 | agent | M0 `KRN:741-744` | **never written in mode 12** (`g==1` fast path, `ADP:731-747`) |
| `row_rem` (`K0P6_D_ROW_REM`, `row_remaining`) | 26 | `uint32[T_LOC_MAX]` = `[40960]` | 163,840 | agent, local | not zeroed at M0 — **self-cleaned** | written once per live row in M2 as `popcount(live expert ballot)` (`KRN:1163-1169`); read as the arrival target (`ADP:690-691`); zeroed by the unique flag publisher (`ADP:587`) |
| `row_ready` (`K0P6_D_ROW_READY`) | 25 | `uint32[world][T_LOC_MAX]` | 1,310,720 | **symmetric** (peer-written) | never — monotonic epoch values | producer `p` stores `epoch32` into `row_ready[p][r]` on the **owner only** (`ADP:575-584`); owner polls `>=` (`CMPL:57-63`, `KRN:511-514`) |
| `mps_state` | 60 | `uint32[8]` + `uint64[8]` | 96 | agent | M0 `KRN:745-748` | lanes: `[0]` role ticket (unused in mode 12), `[1]` event TAIL, `[2]` M8 batch ticket, `[3]` event-claim ticket (`ADP:34-38`); `[8..]` diagnostic timestamps (`ADP:39-52`) |
| `slots` | 61 | `bf16[world][MAXTOK][7168]` | 469,762,048 (448 MiB) | **symmetric** | zeroed once at setup (`DRV:1509`); thereafter **consume-and-zero** by M8 | accumulated by remote RMW (`P2:148-149`); zeroed after read (`KRN:579-587`) |
| `mps_cfg` | 62 | packed `uint64` | 8 | host-written | — | `ADP:54-91` |

### What the service pool does in mode 12, step by step

`run_service` (`ADP:638-838`), per wave, per claimed event:

1. lane 0 claims a ticket `k` from `mps_state[EVNEXT]` (`ADP:649-650`);
   `k >= events_total` ⇒ exit (`ADP:654`).
2. lane 0 spins on `q[k]` until nonzero (`ADP:656-657`); zero return ⇒ exit.
3. **whole wave** `thread_acquire<agent>` (`ADP:670`).
4. Decode `b = (ev>>4)&0x3FFF`, `nc = ev & 0xF` (`ADP:679-680`).
5. Lanes 0..31 read `r = sti[b*32 + lane] & 0x00FFFFFF` (`ADP:687`);
   `live = r < T_ext` (`ADP:688`); live lanes load `target = row_rem[r]`
   (`ADP:690-691`).
6. Each live lane does `fetch_add_acq_rel<agent>(nc_arr[r*16+nc], 1)`
   (`ADP:720-722`). `old + 1 == target` ⇒ this lane is the unique finisher of
   slice `(r, nc)`. With `g == 1` it becomes `push_lead` immediately — no probe
   loop, no claim RMW (`ADP:731-747`).
7. `__ballot(push_lead)` → the set lanes append `(r, group_base)` into the LDS
   `wave_scratch` via an LDS `atomicAdd` on `push_count` (`ADP:772-785`).
8. **Mode-12 branch (`ADP:787-799`): bookkeeping only.** For each appended row,
   `retire_pushed_row(env, s, r, 1u, lane)` — `fetch_add_relaxed<agent>(pushed
   [r], 1)`; when it reaches 16, append `r` to `flag_rows` (`ADP:603-611`).
   Then `continue` — the payload push loop at `ADP:816-832` is never reached.
   **No payload is read or written by the pool in mode 12.**
9. When `flag_count >= flush_rows (=16)` (`ADP:613`), `flush_pending`
   (`ADP:554-592`): wave `s_waitcnt vmcnt(0)`, `__syncwarp`, lane 0 issues one
   `fence(RELEASE,"")`, then per flagged row one `row_ready` store (agent for
   self, system for peers) plus `row_rem[r] := 0`.
10. On exit, one unconditional `flush_pending` per wave (`ADP:834`).

### What M8 does in mode 12, step by step

`KRN:1782-1799` (claim loop) + `KRN:471-610` (batch body), `NT = 4`,
`Zero = true`, `Detect = false`:

1. lane 0 claims batch `k` from `mps_state[M8NEXT]` (`KRN:1787-1790`);
   `k >= ceil(T/4)` ⇒ exit.
2. For the 4 tokens `tok0 + t`: read `lo2[t] = pull_ptr[tok]` and
   `fanout[t] = pull_ptr[tok+1] - lo2[t]` (`KRN:483-489`).
3. Lane `lane` with `lane>>3 == t` and `j2 = lane&7 < fanout[t]` reads
   `p = pull_src[(lo2+j2)*2+0]`, `row = pull_src[(lo2+j2)*2+1]`
   (`KRN:505-506`), then **system-scope** polls
   `row_ready[p*T_loc_max + row] >= epoch32` (`KRN:511-514`).
   Out-of-fanout lanes take the dummy base `(cur, 0)` (`KRN:496-503`).
4. `__ballot(batch_ready) != ~0` ⇒ the whole batch is skipped uniformly
   (`KRN:523`) — a timeout never mints a usable payload.
5. One `acquire_payload_system()` (`KRN:526`) then `__syncwarp` (`KRN:527`).
6. 14-chunk loop (`KRN:529-604`): for each 1 KiB chunk `c`, each lane loads 8
   shuffled 16 B slot fragments per token, accumulates the `fanout[t]`-guarded
   ones into `float acc[4][8]` (`KRN:539-565`).
7. **Consume-and-zero (`KRN:579-587`):** for each live contribution, the same
   lane stores a zero `uint4` back into the slot at the same 16 B offset it
   just read, *after* the read.
8. Pack to bf16 RNE and store 16 B into `out[tok][off]` (`KRN:592-603`).
9. `ts_last(REDUCE_DONE)` by tid 0 (`KRN:1816-1819`), no-op at `timestamps=0`.

Soundness argument recorded in-source (`KRN:454-462`): rows are single-consumer
(unique `(p,row)` per `pull_src` entry), zeroing happens after the read and
before the M9 retirement, and the retirement gate is what protects the next
epoch's producers.

### The M9 cross-rank handshake

1. Every CTA: `__syncthreads()` then tid 0 does an agent acq_rel
   `fetch_add` on `combine_done` (`KRN:1824-1829`).
2. The arriver with `old % 256 == 255` (`KRN:1830`) is the elected finisher.
   It resets `pushed_count[0..7]`, `qpush_done`, `combine_done`, and advances
   `mega_count[0] = epoch` (`KRN:1840-1847`).
3. One `fence(RELEASE,"agent")` (`KRN:1848`), then `retired[cur] = epoch` to
   self (agent, `KRN:1851-1852`) and to all 7 peers (system,
   `KRN:1856-1857`). `retired` is `uint64[world]`, symmetric.
4. Next launch's M0: threads `tid < world` poll
   `retired[tid] >= epoch - 1` at **system** scope (`KRN:702-704`), then the
   whole grid rendezvouses at `KRN:716` and `acquire_payload_agent()`
   (`KRN:718`) before any reset or M1 store.
5. `dest_counter` is reset in M0 for the **next** parity `(epoch+1)&1`
   (`KRN:762-772`), verified by the base-zero guard at `KRN:781-789`
   (pperr bit `262144`). M9 deliberately does **not** reset it
   (`KRN:1841-1844`).

---

## 5. The owner / token mapping

### Q1 — from an M7 output row on rank `r`, what identifies `tau` and its owner?

**The owner is recoverable; `tau` is not.**

- The M7 output row is identified by `xtok = sti[b*32 + i] & 0x00FFFFFF`
  (`P2:350, 525`) — a **receive-row index** `rr ∈ [0, T_ext)`, not a token id.
- Owner: `rr >> 12` (`P2:145`, shift built at `P2:263`). Position within the
  owner's segment: `rr & 4095` (`P2:146`, mask at `P2:264`).
- Why the owner is in the index: M1 computes
  `row = cur*MAXTOK + pos` (`KRN:856`) where `pos` comes from
  `reserve_row` on the **destination's** `dest_counter[cur]` slot
  (`KRN:844-854`) and `cur` is the **source** rank. Segment-per-source layout
  (exp_56 Option C, `KRN:106-113, 839-842`) is what makes the owner implicit.
- **`tau` is nowhere on rank `r`.** The dispatch payload `a_ll` carries only
  the fp8 row, its 56 scales, `eid[8]` and `wgt[8]` (`KRN:865-921`); no token
  index. `recv_eid`/`recv_wgt` on the receiver are indexed by receive-row and
  contain expert ids and weights only. The `pos → tau` map exists **only on
  the owner**, as the (never-inverted) inverse of `pull_stage`.
- First known: `pos` at `KRN:847-856` (M1, waitless fetch_add). The owner-side
  record `pull_stage[(tau*TOPK + s)*2 + {0,1}] = {dest, row}` is written in the
  same M1 loop at `KRN:859-860`. `pull_ptr` (CSR offsets) is built in M3
  (`KRN:1277`, `hkp::csr_scan_block256`). `pull_src` (the compacted
  `(p, row)` list) is filled at the end of M5 (`KRN:1299`,
  `hkp::pull_src_fill`).

### Q2 — does any buffer hold `(owner rank, original token index)` per (dispatched slot, route index)?

**No.** The closest three, exhaustively:

| buffer | slot | shape | bytes | written | what it actually holds |
|---|---|---|---|---|---|
| `pull_stage` | 12 | `int32[T*TOPK][2]` = `[32768][2]` | 262,144 | `KRN:859-860` (M1), by the **owner** for its own token `tau` | `(dest_rank, global receive-row)` for route slot `s` of local token `tau`. `tau` is the **index**, not a stored field. `(-1,-1)` for non-primary slots. |
| `pull_src` | 19 | `int32[T*WORLD][2]` = `[32768][2]` | 262,144 | `KRN:1299` (M5) | CSR-compacted `(p, row)` pairs; entry range `[pull_ptr[tau], pull_ptr[tau+1])`. Again `tau` is the index. |
| `pull_ptr` | 18 | `int32[T+1]` = `[4097]` | 16,388 | `KRN:1277` (M3) | CSR offsets, `pull_ptr[T] = F` |

Both are **owner-local** (`DRV:1678, 1684-1685` — plain `torch.zeros`, not
`mori_t`, so not symmetric) and both are keyed **by** `tau`. There is no
producer-side table mapping a receive row back to `(owner, tau)`, and none of
the three is readable by a peer. A producer that needed `tau` would have to
either carry it in the dispatch payload or have the owner publish an inverse of
`pull_stage`; neither exists today.

### Q3 — the final output buffer `out`

| property | value | source |
|---|---|---|
| descriptor slot | 32 (`K0P6_D_OUT`) | `KRN:156`; harness entry 32 = `cand_out.data_ptr()` (`DRV:2678`) |
| allocation | `torch.zeros((T, H), dtype=bfloat16)` | `DRV:1799` |
| shape | `[4096, 7168]` | `T = 4096`, `H = K0P6_H = 7168` (`KRN:118`) |
| dtype | bf16 (kernel type `unsigned short`) | `KRN:1705` |
| elements per rank | `4096 × 7168 = 29,360,128` | |
| bytes per rank | `58,720,256` = **56 MiB** | `29,360,128 × 2` |
| visibility | **local**, not symmetric | `DRV:1799` uses plain `torch.zeros` |
| written by | M8 only, `KRN:592-603` | `orow = out + (tok0+t)*7168`; `uint4` store at `off = (c<<10) + (lane<<4)` for `c ∈ [0,14)`, `lane ∈ [0,64)` ⇒ `14 × 64 × 16 = 14,336 B` = one full row |
| written by M9 | **no** | `KRN:1822-1862` touches no output |
| write multiplicity | exactly once per token, by the single wave that claimed its batch | batch tickets `KRN:1787-1790` |

---

## 6. Config plumbing

### `K0_MPS_CFG` → kernel, end to end

1. **Env** `K0_MPS_CFG="C=16,g=33,mode=12,flush_rows=16"`. Optional extra keys
   `pull_fallback`, `timestamps` (default 0). Parsed by `_parse_mps_config`
   (`DRV:389-423`); missing/extra keys raise `ValueError` at module import —
   **which looks like a pass** (no `[MARK]` line, empty run dir).
2. **Encode**: `k0_mps_host_abi.encode_config(C, g, mode, flush_rows,
   pull_fallback, timestamps)` (`DRV:417-422`) → the pybind shim
   `.node/mps_host_bridge.cpp:18-31` → `hk_moe::mps::encode_config`
   (`ADP:70-80`) + `config_is_valid` (`ADP:249-286`); invalid ⇒ throw.
3. **Descriptor**: `k0_mps_host_abi.append_and_validate(donor_55_words, heap
   desc, queue, arrivals, pushed, claim, state, slots, config)`
   (`DRV:2750-2760`) → `mps_host_bridge.cpp:33-51` →
   `hk_moe::host_abi::append_mps_descriptor` (`ABI:240-252`) →
   `patch_mps_slots` (`ABI:220-234`) puts the word in **slot 62**
   (`ABI:153`), then `validate_mps_descriptor` (`ABI:255-272`).
4. **Upload**: `pf6_state["desc_mps"] = torch.tensor(..., int64)`
   (`DRV:2761-2766`), asserted to be exactly 63 words.
5. **Device**: every consuming site re-reads it through
   `k0p6_dread(desc, K0P6_D_MPS_CFG)` (slot 62, `ADP:26`) and decodes locally
   with `hk_moe::mps::decode_config` (`ADP:82-91`). Sites:
   `KRN:274, 282-284, 317-318, 348-349, 369-370, 667-668, 1118-1119,
   1309-1310, 1339-1340, 1368-1369, 1387-1388, 1404-1405, 1483-1484,
   1539-1540, 1596-1597, 1676-1677` and `P2:247-248`.

Packed layout (`ADP:54-91`):

```
[0:8)   C           reserved service CTAs
[8:16)  g           group_slices field (see bit table below)
[16:24) mode
[24:32) flush_rows  1..64
[32]    pull_fallback
[33]    timestamps
```

### Mode constants — every number currently taken

| mode | symbol / name | defined | notes |
|---:|---|---|---|
| 0 | reserve-only control | `ADP:229-231` (`mode_is_parity_publish`) | parity M7.5 publish + parity M8 remote-`part` pull |
| 1 | bulk push after M7 | `KRN:1596-1597`, `ADP:281-283` | must carry `C == 0` |
| 2 | stream (static tail pool) | `ADP:158-161` | the pre-exp_21 ratchet |
| 3 | stream, whole-die pool | `ADP:93-113, 294, 301-305` | `C % 32 == 0` (`ADP:280`) |
| 4 | mode 2 + push pacing | `ADP:114-139, 234, 238` | `flush_rows` reinterpreted as pacing units |
| 5 | mode 0 + synthetic traffic pool | `ADP:224-226`, `KRN:1489-1502` | `g` = variant, `pull_fallback` = local dst, `flush_rows` = window; `g == 16` rejected (`ADP:278`) |
| 6 | mode 0 + LDS-only spin pool | `ADP:224-226`, `KRN:1503-1506` | `flush_rows` = window |
| 7 | mode 2, payload copy deleted | `ADP:141-147`, `KRN:1453` | requires `pull_fallback == 1` (`ADP:273`) |
| 8 | mode 2 + event-poll backoff | `ADP:148-157, 242` | `flush_rows` = backoff units |
| 9 | `kModeFlushAgent` | `ADP:172`, used `ADP:567-568` | flush release weakened to agent; **diagnostic, not correctness-preserving** |
| **10** | — | — | **FREE** — no reference anywhere in the tree |
| **11** | — | — | **FREE** — no reference anywhere in the tree |
| 12 | `kModeRemoteAccum` | `ADP:203` | this document |
| 13 | `kModeDirectRows` | `ADP:204` | per-row target counters, no `pushed` |

Upper bound: `if (c.mode > 13u) return false;` — `ADP:271`.

**Next free mode number: 10** (then 11). Both are unclaimed gaps inside the
existing validator bound, so a new mode taking 10 needs **no change to
`config_is_valid`**. Taking 14 instead would require raising the `mode > 13u`
bound at `ADP:271`. Also check the mode-list predicates when adding: a new
stream-like mode must be added to `mode_is_stream` (`ADP:158-161`) **and** to
the `k0p6_mps_task_done` enqueue predicate (`KRN:319-320`) — the in-source
warning at `KRN:296-303` records that an open-ended `>= 2` there would have
silently enqueued events nobody consumes.

### `g` field bit decode

| bits | meaning | decoded at | value at `g=33` |
|---|---|---|---|
| `0x0F` | physical `g` (`kRemoteAccumGMask`) | `ADP:205, 218-221` (`physical_g`) | **1** |
| `0x10` | dual-write detector (`kRemoteAccumDetectBit`) | `ADP:206, 213-216` (`detect_dual`) | **0** |
| `0x20` | epilogue RMW throttle | **`P2:272` only** — read raw from the packed word, `((m7cfg >> 8) & 0x20)`; validated as a legal bit at `ADP:263-265` | **1** |
| `0xC0` | — | rejected by `ADP:263-265` | 0 |

`33 = 0b100001` ⇒ physical `g=1`, no detect, throttle on. Validation path:
`ADP:262` requires `(g & 0xF) == 1`; `ADP:263-265` requires
`(g & ~(0xF|0x10|0x20)) == 0`. Note the throttle bit is **not** a field of
`struct config` (`ADP:61-68`) and is **not** surfaced by `physical_g` or any
accessor — `P2:272` is the sole decoder. Consumers outside the phase-2 body
cannot see it.

In non-direct-accum modes `g` must be one of `{1,2,4,16}` (`ADP:269-270`) and
is used verbatim as the push group width.

### Environment knobs

**The device kernel reads no environment variables** — every runtime value
arrives through the 63-word descriptor. The knobs are host-side:

| knob | read at | effect |
|---|---|---|
| `K0_MPS_CFG` | `DRV:426` | the packed config word (slot 62) |
| `K0_MPS_SOAK_ITERS` | `DRV:4843, 4917` | must be exactly 600 or `DRV:4918-4921` raises |
| `K0_PF6GM_G` | `DRV:248` | G-stack width; `mps_mega` forces `PF6MPS_G = 3` and raises on mismatch (`DRV:249-254`) |
| `K0_PF6GM_DEBUG_PHASE` / `K0_PF6_DEBUG_PHASE` | `DRV:2724-2727`, `CMP:129` | descriptor slot 49 = `K0P6_D_DEBUG`, consumed at `KRN:690` and the `debug_stop == N` exits (`KRN:798, 1013, 1212, 1316, 1331, 1393`) |
| `K0_SPIN_LIMIT` | `CMP:128` | the `spin_limit` kernarg (default 2,000,000) — bounds every poll |
| `K0_T`, `K0_T_LOC_MAX`, `K0_PADMAX`, `K0_MAXTOK`, `K0_MAXTOK_PROD` | `DRV:117-131`, `CMP:114-118` | shapes; `MAXTOK` also becomes descriptor slot 51 (`DRV:2730`) |
| `K0_ARMS` / `K0_MOK_ARMS` | `DRV:238`, `CMP:15` | arm selection; `mps_mega` gates the whole MPS build |
| `K0_MOK_WARMUP_ITERS`, `K0_MOK_TIMED_ITERS`, `K0_MOK_ABSOLUTE_TOLERANCE`, `K0_MOK_RELATIVE_TOLERANCE`, `K0_MOK_SEED_BASE` | `DRV:141-152, 1830`, `CMP:121-126` | timing/gate parameters. **The campaign forces `ABSOLUTE_TOLERANCE=0.1`** (`CMP:124`) against the source default of `1.0` (`DRV:144`) |
| `K0_INPUT_MODE`, `K0_BENCHMARK_PROTOCOL` | `DRV:133-140`, `CMP:111-112` | `mok_synthetic` / `mok_eager` |
| `MORI_SHMEM_HEAP_SIZE`, `MORI_GPU_ARCHS`, `HSA_XNACK` | `CMP:103-105` | 32 GiB symmetric heap, gfx950, XNACK on |

**UNRESOLVED:** `K0_MPS_DEBUG_STOP`, `K0_MPS_DESC_DUMP`, `K0_MPS_TRACE`,
`K0_MPS_SKIP_LAUNCH` are named in `overnight/aug10/CLAUDE.md` but appear in
**no** source file in this repo, including the node mirrors `DRV` and `CMP`.
Either they live only in the authoritative `~/amd-master` harness (newer than
these mirrors) or the names in CLAUDE.md are stale. The only debug channel
visible here is `K0_PF6GM_DEBUG_PHASE` → descriptor slot 49.

---

## 7. Shapes and constants

### Compile-time (in this repo)

| symbol | value | source |
|---|---|---|
| `K0P6_H` (hidden) | 7168 | `KRN:118` |
| `K0P6_NG` | 56 | `KRN:117` |
| `K0P6_MAXE` | 64 | `KRN:116` |
| `K0P6GM_G` | **3** (hard `#error` on any other value) | `KRN:192-197` |
| `K0P6C_NCHUNK_MAX` | 8 | `KRN:114` |
| `K0P6_MPS_D_LEN` / descriptor words | **63** | `ADP:27`, `ABI:146` |
| `world_size` (host ABI) | 8 | `ABI:23` |
| `donor_descriptor_words` | 55 | `ABI:24` |
| `K0P6_MPS_ST_WORDS` / `TS_COUNT` | 8 / 8 | `ADP:38, 52` |
| `kEventMark` | `0x80000000` | `ADP:321` |
| `kSliceBytes` | `448*2 = 896` | `ADP:479` |
| `kPushBatch` | 4 | `ADP:485` |
| `sizeof(wave_scratch)` | **528 B** (static_assert) | `ADP:476-477` |
| `kCtasPerXcd` / `kXcds` | 32 / 8 | `ADP:107-108` |
| `C` cap | ≤ 128 | `ADP:257` |
| `flush_rows` range | 1..64 | `ADP:284` |
| `kNChunks` | 16 | `P2:67` |
| `kChunkN` | `7168/16 = 448` | `P2:68` |
| `kWaveTiles` | `448/16/4 = 7` | `P2:69` |
| `kWaveCols2` | `16*7 = 112` | `P2:70` |
| `kNGroups2` | `7168/128 = 56` | `P2:72` |
| `kTaskTiles` | `448/16 = 28` | `P2:73` |
| `kMrows` | `32*3 = 96` | `P2:216` |
| grid / block | `256 CTAs × 256 threads`, `__launch_bounds__(256,1)` | `KRN:613`; `nct != 256` rejected at `KRN:656` |

**Derived from `static_assert`s and use** (the defining header
`n2_fused_moe.hpp` is in `~/amd-master/auto-gpu-kernel/k0_fused_moe/solution/
hip/`, **not in this repo** — flagged as inference):
`kHidden = 7168`, `kInter = 2048`, `kBlockM = 32`, `kWaves = 4`,
`kThreads = 256`, `kCTAs = 256`, `kExperts = 32`, `kKGroups2 = 2048/128 = 16`,
`kAChunks = 8`, `kAChunkBytes = 16` — from the `static_assert`s at `P2:75-80`
(`kThreads == kBlockM * kAChunks`, `kAChunks * kAChunkBytes == 128`), the wave
split `kWaveTiles = kChunkN/16/kWaves = 7` (`P2:69`), and the lane decode at
`P2:228-231, 308-309, 312-313`.

**UNRESOLVED (headers outside this repo):** `K0P5_K`, `K0P5_ROW_STRIDE`,
`K0P5_SC_BYTES`, `K0P5_ROW_U64` (`k0pf5_ll128.hpp`) and `K0P6_CHUNK`
(`k0pf6_chunk_release.hpp`). Derivable from use: `K0P5_K = 7168`;
`K0P5_SC_BYTES = 56*4 = 224` (from `KRN:900-901`, `c ∈ [0,7)`,
`(lane>>3) ∈ [0,8)` ⇒ 56 floats); `K0P6_CHUNK = 512` (from
`K0P6C_NCHUNK_MAX = ceil(4096/512) = 8`, `KRN:114`, and the hole range
`[base+fill, base+512)` at `KRN:1182, 1203`). `K0P5_ROW_STRIDE ≥ 7168 + 224 +
64 = 7456`, consistent with the 7,488 B LDS staging slice at `KRN:631`, but
its exact padded value is not determinable here.

### Runtime shapes — the MoK synthetic prefill campaign

| quantity | value | source |
|---|---|---|
| `world` | 8 | `CMP:145` (`--nproc-per-node=8`), `KRN:652` guard |
| `T` (input tokens per rank) | **4,096** | `CMP:114`, `DRV:117` |
| `TOPK` | **8** (hard guard `KRN:652`) | `DRV:103` |
| `E` (local experts) | **32**; `E_GLOBAL = 256` | `DRV:103` |
| hidden `H` = `K` | **7,168** | `DRV:103` |
| intermediate `INTER` | **2,048** | `DRV:103` |
| `NG` | 56 | `DRV:103` |
| `MAXTOK` | **4,096** (= `T`; descriptor slot 51) | `CMP:117`, `DRV:130, 2730` |
| `T_ext` | `8 × 4096 = ` **32,768** | `KRN:688` |
| `T_LOC_MAX` | **40,960** | `CMP:115`, `DRV:118` |
| `MROWS` | `= T_LOC_MAX =` 40,960 | `DRV:129` |
| `PADMAX` | `8·4096·8 + 31·32 = 262,144 + 992 = ` **263,136** | `CMP:116`, `DRV:123-124` |
| `bcap` = `PADMAX/32` | **8,223** (≤ 16,383 guard, `KRN:669`) | |
| `maxb` | `PADMAX/32 = 8,223` | `DRV:1484` |
| N2 rowcap | `8·4096·8 + 256·32 − 8 = 270,328` | `DRV:2627` |
| epochs | **one epoch per launch**; campaign = 500 warmup + 100 timed × 5 rotations; soak = exactly 600 | `CMP:122-123`, `DRV:4918-4921` |
| `spin_limit` | 2,000,000 | `CMP:128` |
| `C` / compute CTAs | 16 / **240** | config; `KRN:1370`, `ADP:296` |

### Buffer sizes (per rank)

| buffer | slot | shape | bytes | symmetric? |
|---|---|---|---|---|
| **`slots`** (`K0P6_D_MPS_SLOTS`) | 61 | `bf16[8][4096][7168]` | **469,762,048 = 448 MiB** | **yes** (`DRV:1506-1509`, `mori_t`) |
| **`part`** (`K0P6_D_PART`) | 21 | `bf16[40960][7168]` | **587,202,560 = 560 MiB** | yes (`DRV:1483`) |
| `mps_q` | 56 | `uint32[8223][16]` | 526,272 | no |
| `nc_arr` | 57 | `uint32[32768][16]` | 2,097,152 | no |
| `pushed` | 58 | `uint32[32768]` | 131,072 | no |
| `claim` | 59 | `uint32[32768]` | 131,072 | no (**dead in mode 12**) |
| `mps_state` | 60 | `uint32[8] + uint64[8]` | 96 | no |
| `row_ready` | 25 | `uint32[8][40960]` | 1,310,720 | yes |
| `row_remaining` | 26 | `uint32[40960]` | 163,840 | no |
| `retired` | 31 | `uint64[8]` | 64 | yes |
| `out` | 32 | `bf16[4096][7168]` | 58,720,256 = 56 MiB | no |
| `a_ll` | 3 | `uint8[40960][K0P5_ROW_STRIDE]` | ≈ 305 MiB at stride 7,456 | yes |
| `a2q` | 27 | `bf16-view[270328][1024]` (= 2,048 B rows fp8) | 553,631,744 | no |
| `dq2` | 28 | `float32[270328][16]` | 17,300,992 | no |
| `sti` / `swt` | 14 / 15 | `int32[263136]` / `float32[263136]` | 1,052,544 each | no |
| `sei` | 16 | `int32[8223]` | 32,892 | no |
| `pull_stage` | 12 | `int32[32768][2]` | 262,144 | no |
| `pull_src` | 19 | `int32[32768][2]` | 262,144 | no |
| `pull_ptr` | 18 | `int32[4097]` | 16,388 | no |
| `a2_done` / `part_done` | 29 / 30 | `int32[8223]` each | 32,892 each | no |
| `chunk_ready` | 50 | `int64[8][8]` | 512 | yes |
| `dest_counter` | 4 | `int32[2][8]` | 64 | yes |

### Structural observation on `part` in mode 12

`part` (560 MiB, slot 21) is **neither read nor written** in mode 12:

- M7 receives it as the `OUT` kernarg (`KRN:1380`) but the `peer_tab != nullptr`
  branch (`P2:132`) never dereferences it, and the `dual` write (`P2:160-165`)
  is off at `g=33`.
- M8 sets `part = nullptr` (`KRN:1694-1696`) because
  `m8_pull == false` and `m7_detect == false`.
- The service pool's `env.part` (`KRN:1433`) is only used by
  `slice_group_src` (`ADP:491-495`), which mode 12 never reaches (`ADP:798`).

Yet M5 still zeroes and scale-transposes all `T_ext` rows of it every epoch
(`KRN:1284-1288`, `hkp::zero_part_scale_transpose<14>`), i.e.
`32,768 × 7,168 × 2 = 469,762,048 B = 448 MiB` of stores per rank per epoch
that nothing in mode 12 consumes. Stated as structure, not as a cost claim.

---

## 8. Open items / contradictions found while mapping

1. **`ADP:377-380` vs `KRN:305-316` + `KRN:324-325`** — the adapter comment
   claims the mode-12 event enqueue releases at **system** scope; the call site
   uses the `scope::agent` default and the kernel comment says agent
   explicitly. Compiled behaviour: **agent**. §3(b) counts it as agent.
2. **`K0_MPS_DEBUG_STOP` / `K0_MPS_DESC_DUMP` / `K0_MPS_TRACE` /
   `K0_MPS_SKIP_LAUNCH`** — named in `overnight/aug10/CLAUDE.md`, absent from
   every file in this repo including the node harness mirrors.
3. **`hkp::grid_barrier` / `hkp::local_grid_epoch` / `hkp::fail_closed`**
   (`hkp_sync.hpp`) and **`k0p6_sort::*`** (`hkp_sort.hpp`) are outside this
   repo — their per-call atomic and fence counts are not in the §3 census.
4. **`n2_fused_moe.hpp` / `n2_device_common.cuh`** (`kHidden`, `kInter`,
   `kBlockM`, `kThreads`, `kWaves`, `kCTAs`, `kExperts`, `kKGroups2`,
   `kAChunks`, `kAChunkBytes`) are outside this repo; §7 lists them as
   inferences from `static_assert`s.
5. **Per-epoch data-dependent counts** (`nvi[0]`, `num_tiles`, `R`, `P`, `F`)
   are not statically determinable. §3 gives every count as a formula; the
   numeric values come from `ADP:635` (`~16,720` events), `ADP:744`
   (`~698,112 of ~1.58M`, i.e. 349,056 completing slices ⇒ `R = 21,816`),
   `ADP:636` (`535,040` arrival RMWs at the padded 32-lane bound),
   `KRN:1074` (`~21,816 live rows`), and the balanced-routing identity
   `P = W·T·TOPK / W = 32,768`. A skewed route changes `R`, `P`, `F` and
   `num_tiles`, and therefore every count in §3 except the fixed 256/1,024
   grid quantities.
6. **`ADP:176-178`'s "~312 MB of remote RMWs"** is the distinct slot footprint
   (`R × 14,336`), not the RMW volume (`P × 14,336 = 448 MiB`). §2 shows both
   derivations.
7. **`KRN:1848`** releases at agent scope before the seven system-scope
   `retired` pokes; mode 12 is the first mode whose slot lifetime depends on
   that edge (§3(d)).
8. **`claim` (slot 59, 128 KiB)** is allocated, zeroed every epoch
   (`KRN:741-744`), and never written in mode 12 (`g == 1` fast path).
