# exp_05 — dispatch/M6 overlap (COMET "layer-0") — READ-ONLY feasibility scope

Scope pass only. No source edited, no build, no GPU job. Paths relative to
`distributed-kernels/fused_moe/` unless prefixed. `.node/mps_n2_phase1_gm.cpp` is a
node-pulled MIRROR of the donor include `n2_phase1_gm.cpp` — see UNKNOWN-1.

## Verdict

The premise splits in two. One half is confirmed; the other is dead as stated.

1. **"M1+M2 is exposed before M6" — CONFIRMED, and worse than assumed.** There are
   **four grid barriers plus one system acquire** between the last dispatch store
   and M6's first MFMA (`k0pf6gm_device_tile_mps.hip:924, 1046, 1101, 1111`, acquire
   at `:958`). M6 contains **no readiness poll of any kind** — phase 1 gets an
   *arrive* hook only, never a wait (`:297` vs phase 2's `:309-310`), and reads its
   input through a bounded, non-polling buffer descriptor
   (`.node/mps_n2_phase1_gm.cpp:146-148, 192`). So there is exactly **zero**
   dispatch/compute overlap today and no per-row readiness path to reuse. Per
   `DESIGN_MPS.md:78-79` the exposed pre-M6 region is `M1 557 + M2 200 + M3M4M5 291`
   = 1,048 µs measured at G=3.
2. **"Reorder M6's input local-tokens-first" — DEAD as a cheap change; it is a
   rewrite.** M6's input array *does not exist* until the destination-counting sort
   finishes, and a **local** token's sorted row depends on the counts of **all
   remote** tokens: `count_reduce` → `scan` over the global histogram
   (`:1050-1052`) sets every expert's padded row-begin, then `scatter` (`:1107`)
   places rows. COMET's layer-0 permutes an array that already exists; here the
   permutation *is* the plan, so "local first" means a two-segment
   (local | remote) per-expert layout with reserved local capacity — a rewrite of
   the M3–M5 contract, not a reorder. Cost class: **rewrite**.
3. **Even a perfect layer-0 cannot hide 757 µs.** M1's 557 µs is this rank's OWN work
   — 7 × (bf16 amax + fp8 quantize) per token plus the fanout stores (`:695-765`) —
   already spread over all 256 CTAs. Work is not wait; reordering M6 can only overlap
   the *wait* fraction, **which is currently unmeasured and is the one number that
   decides this line.** It is already recorded on device and never read back (exp 1).

Recommendation: do not build a layer-0 arm. One harness-only run measures the wait
fraction; that answer selects between "role-split the M1/M2 boundary" and "close it".

## Phase map M0..M6

All ranges in `k0pf6gm_device_tile_mps.hip`.

| Phase | Lines | What it does | Class |
|---|---|---|---|
| entry guards | 479-513 | shape guard (`:487`), MPS config + buffer-presence guard (`:495-510`), `T_ext = world*MAXTOK` (`:511`) | local |
| M0 | 515-621 | polls each peer's `retired[]` to `epoch-1` (`:525-527`) — the only cross-rank edge here; grid barrier (`:539`); zeroes `a2_done`/`part_done` (`:546-549`), all MPS state (`:554-572`), next-parity `dest_counter` (`:585-613`) | **comm (wait)** + local |
| M1 | 623-836 | **dispatch push.** Per local token: route (`:654-656`), remote row reservation by `reserve_row<system>` on the peer's `dest_counter` (`:673-677`), bf16→fp8 quant into LDS (`:695-728`), `store_dispatch_row` into peer `a_ll` (`:759-764`). Then per-CTA system release (`:784`) and the last grid arriver publishes `rows_done` + all 8 `chunk_ready[cur][c]` epoch words to every peer (`:789-834`) | **comm (send)** |
| M1‖M2 seam | 837 | `// no barrier: CTAs flow straight into the streaming phase` | — |
| M2 | 839-1022 | **dispatch receive.** Step A: tid0 polls all 8 peers' `rows_done` (`:871-890`). Pass A: wave-striped `poll_epoch_word_system(chunk_ready[s][c])` (`:904-916`) — the peer-arrival wait, and the only spin instrumented by `SPIN_DBG` (`:913`). Pass A→B grid barrier (`:920-925`), one convergent `acquire_payload_system` (`:958`). Pass B: row-parallel unpack + per-expert histogram + `row_remaining[r]` (`:966-981`); hole sentinel (`:997-1017`); `hcnt` publish (`:1020`) | **comm (wait)** + local |
| M3–M4 | 1046-1101 | grid barrier (`:1046`), then the plan split. `bid==0`: `count_reduce`/`scan`/`cursors_sei` then **builds `tile_desc`** serially on tid0 (`:1060-1083`). `bid==1`: CSR scan → `pull_ptr` (`:1084-1087`). `bid>=2`: zero/transpose `part`+`sc_dst` (`:1088-1099`). Barrier `:1101` | local |
| M5 | 1104-1112 | `pad` (`:1106`), `scatter` → `sti`/`swt` (`:1107`), `pull_src_fill` (`:1109`), barrier `:1111` | local |
| M6 | 1116-1129 | `n2p6gm_phase1_body` — GEMM1 (gate/up) over `(tile, intermediate-chunk)` tasks. No poll, no cross-rank traffic; task-end hook is `k0p6_a2_arrive` (`:297`, `:214-221`), a release-only local counter bump | **local compute** |

The claimed 757 µs = **M1 (623-836) 557 µs + M2 (839-1022) 200 µs** per
`DESIGN_MPS.md:78-79`; M3M4M5 adds another 291 µs of equally-exposed local plan work
that the proposal does not address.

## The dispatch -> M6 dependency

Not a per-row or per-tile readiness flag. It is a chain of **grid-wide
rendezvous**, and the plan M6 iterates is rebuilt inside that chain.

```1111:1117:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
    hkp::grid_barrier(bar, tid, bar_err);   // M5 publishes sti/swt/pull_src/part/sc_dst
    if (hk_moe::error_bit_set_agent(pperr, 2097152)) return;
  }
  if (debug_stop == 4) return;

  // ================= M6: N2 phase1 (G-stacked; a2_done arrivals per live sub-block) =========
  production_fused_moe::n2::n2p6gm_phase1_body(
```

The earliest hard edge is M2's own Pass A→B barrier, which exists precisely so
that no CTA reads payload before *every* peer chunk has been observed:

```920:925:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
    {
      unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
      const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
      const hkp::fail_closed bar_err{pperr, 2097152};
      hkp::grid_barrier(bar, tid, bar_err);
    }
```

Full chain: `M1 stores → chunk_ready publish (:828-831) → Pass-A poll (:910) →
barrier (:924) → acquire (:958) → Pass-B unpack (:970) → barrier M3 (:1046) →
tile_desc/scan/pull_ptr (:1049-1099) → barrier M4 (:1101) → pad/scatter (:1106-1109)
→ barrier M5 (:1111) → M6 (:1117)`. Four barriers. **The premise that "the overlap
may already exist" is refuted: it provably does not.**

## How M6 picks work

**Static loop index over a data-driven table.** The iteration is a plain
`blockIdx.x`-strided walk; the *content* of task `t` comes from `tile_desc[]`:

```156:165:.node/mps_n2_phase1_gm.cpp
  for (int task = blockIdx.x; task < num_tasks; task += kCTAs) {
    const int tile = task / kNChunksP1;
    const int g = task - tile * kNChunksP1;   // intermediate chunk 0..7
    const int b0 = n2gm_tile_b0(tile);        // first 32-block of the tile
    const int e = sorted_eid[b0];
    const int gcount = n2gm_tile_gcount(tile, b0, e);   // live sub-blocks (1..kGM)
```

with `num_tasks = num_tiles * 8` (`:137`), `b0 = tile_desc[t] >> 4` and
`gcount = tile_desc[t] & 0xF` (`:138-143`). `tile_desc`/`num_tiles` are
**device-written in M4**, serially on tid 0 of block 0
(`k0pf6gm_device_tile_mps.hip:1060-1083`); the host only allocates them
(`.node/mps_remote_e004pf_k0pf_ab.py:1477-1482`).

Explicitly: **it is data-driven, and the driving table is built on device, not on
the host.** So permuting M6's traversal order is a ~10-line change to one serial
append loop — genuinely cheap. But it is **the wrong knob**: permuting the order in
which tiles are visited does not change *what data each tile needs*. Every tile
still reads `sti`/`a_dst` rows produced by `scatter` after all remote arrivals were
counted. Reordering buys zero overlap.

## Ownership / sort key availability

The sort key exists already, at zero cost, and survives the sort.

1. **Pre-sort:** the exp_56 segmented receive layout encodes the producer in the
   row index itself — `row = cur*MAXTOK + pos` (`:679`, contract at `:662-665`).
   M2 already recovers it for free: `const int s = r / MAXTOK;` (`:967`). "Is this
   token local?" is `(r / MAXTOK) == cur`. No new buffer, no new pass.
2. **Post-sort:** `sti` packs `token|slot<<24`; consumers mask with `0x00FFFFFF`
   (`.node/mps_n2_phase1_gm.cpp:171`, `n2_phase2_gm_mps.cpp:153, 240`). That token
   *is* the receive-row index, so `(sti[b*32+i] & 0x00FFFFFF) / MAXTOK` gives the
   source rank of any sorted row — also free.
3. **Combine-side ownership** is a separate map, keyed by owner token and never by
   sorted row: `pull_stage[(tau*TOPK+s)*2+{0,1}] = {dest, row}` and `pull_cnt[tau]`
   in M1 (`:682-685`), compacted by the M4 CSR scan (`:1087`) + `pull_src_fill` (`:1109`).

So a "local-first" sort key needs no new plumbing. What is missing is not the key
— it is a layout in which local rows have addresses that do not depend on remote
counts.

## Blast radius of reordering

Two distinct changes with very different radii. **(A) permuting `tile_desc` order**
is safe; **(B) changing the sorted-row assignment** (the actual layer-0) touches
almost everything.

**(A) `tile_desc` permutation — SAFE.** Every downstream consumer is keyed by the
absolute 32-block `b` or the receive-row `r`, never by tile ordinal:
`a2_done[b]`/`part_done[b]` (`:219`, `:225`, `:251`), `A2q`/`DQ2` rows
(`n2_phase2_gm_mps.cpp:261`), the MPS event queue (monotonic tail ticket,
`:280-281`), `row_remaining[r]`/`row_ready[r]` (`:978`, `:1213`), `mps_arr`/
`mps_pushed`/`mps_claim` (`:563-566`). M6 and M7 both derive `b` from the *same*
`tile_desc`, so the a2 arrive/wait edge permutes consistently.

**(B) two-segment (local | remote) layout — the real risk list:**

1. `k0p6_sort::scan(s_cnt, E, T_ext, scratch, nvi, tid)` (`:1052`) — per-expert
   padded row-begin `scratch[K0P6_SC_ERB+e]`. Reserving local capacity per expert
   changes `nvi[0]` (padded total) and therefore the whole downstream sizing.
2. `nvi[0]` is the bounded A2 descriptor record count (`n2_phase2_gm_mps.cpp:208-210`)
   **and** the MPS event capacity check `events_total = (nvi[0]>>5)*16` vs
   `(PADMAX/32)*16` (`:1197-1202`, error `K0P6_MPS_ERR_CONFIG`). Inflating padded
   rows can trip the capacity guard outright.
3. `cursors_sei` + `pad` (`:1053`, `:1106`) against `PADMAX`, error bit 4194304 —
   reserved-but-unused local capacity is pure pad inflation, i.e. wasted MFMA.
4. **The single-expert 32-block invariant**: `e = sorted_eid[b0]` is read once per
   tile (`.node/mps_n2_phase1_gm.cpp:164`, `n2_phase2_gm_mps.cpp:224`). A local/remote
   split boundary that falls inside a 32-block is a correctness break, not a
   slowdown. Segment boundaries must be 32-row aligned — and 32-aligning per
   (expert × source) at E≤64 is where the pad inflation of (3) comes from.
5. `tile_desc`'s `(b0<<4)|gcount` contract (`:1075`) assumes a tile's `gcount`
   sub-blocks are **contiguous and same-expert**. Two segments per expert means
   ragged tails ×2 per expert.
6. `scatter` (`:1107`) is deliberately atomic-cursor nondeterministic, so only the
   *structural* controls validate it — per-expert live counts, `nvi[0]` match vs
   production, total live pairs (`.node/mps_remote_e004pf_k0pf_ab.py:4503-4531`).
7. `hkp::pull_src_fill` (`:1109`) and `row_remaining[r]` (`:978`) are token-keyed and
   survive — **but only if `sti`'s packed token field still means "receive row"**.
   Re-basing it breaks M7's `part` target and M8's `pull_src` gather together.
8. `k0p6_a2_wait` polls `a2_done[b]` for a hard-coded count of 8
   (`poll_local_count<8>`, `:228`). Change how many M6 tasks touch a 32-block and
   this constant is wrong: a silent hang to `spin_limit`, then pperr 16777216.

## Cheapest first experiment

Ranked by information gained / build risk. All three answer the same question:
**how much of the 757 µs is peer-wait (hideable) vs local work (not hideable by
reordering)?**

**1 — Read the spin counter that is already being recorded. ZERO device code.**
`K0P6_D_SPIN_DBG` is `uint32[2] = {max spins to SUCCESS, max spins at FAIL}` and is
already wired into M2's Pass-A chunk poll (`:913`, buffer at
`.node/mps_remote_e004pf_k0pf_ab.py:1476`, in all three pf6 descriptors at `.py:2714,
2731, 2746`). It is even already printed — but the readout is gated to one arm we
do not run: `if name == "pf6c_mega":` (`.py:4509`), readout at `.py:4535-4540`.
- Touch: `e004pf_k0pf_ab.py` (mirror `.node/mps_remote_e004pf_k0pf_ab.py`), the
  `for name, body, buf in ARMS:` eager loop. Widen the `:4509` gate to
  `name in ("pf6c_mega", "pf6gm_mega", "mps_mega")` and zero `pf6_state["spin_dbg"]`
  at the top of each arm's iteration (it is a never-reset session-wide max shared by
  all pf6 arms, so per-arm attribution requires the zero). ~4 lines, logged in the
  harness note per the ownership rules.
- Confirms: large `success_max` ⇒ the chunk words are NOT there when polled, peer
  wait dominates, a dispatch-boundary role split has a real prize.
- Refutes: near-zero `success_max` ⇒ payload has already landed by the time M2
  polls, M2's 200 µs is pure unpack work and M1's 557 µs is pure local work. Then
  **layer-0 as COMET states it is dead on this kernel** and the axis closes with a
  measured negative.
- Register/LDS impact: none (host-side only).

**2 — Two device timestamps into the free TS slots. ~6 device lines, ~6 host lines.**
`moe_mps_adapter.cuh:38-43` defines 5 timestamp slots of `K0P6_MPS_TS_COUNT 8`, so
slots **5, 6, 7 are free** with no ABI or allocation change (host already allocates
`12 × int64` = 8 u32 + 8 u64 at `.py:1505`; M0 already zeroes exactly
`K0P6_MPS_ST_WORDS + 2*K0P6_MPS_TS_COUNT` words at `:568-571`).
- Touch: `k0pf6gm_device_tile_mps.hip` — one `if (tid == 0) ts_last(mps_ts + slot,
  cfg.timestamps)` at M1 end (`:834`), at the Pass A→B barrier exit (`:925`), and at
  the M5 barrier exit (`:1112`), using the existing idiom verbatim (`:1169-1176`,
  `:1515-1518`). Host: add a post-launch readback of `mps_state` into
  `R["pf6mps"]["timestamps"]` — none exists today; follow the identical pattern
  already in `pf6mps_mega_body` (`.py:3328-3341`).
- Gives absolute µs for `(M2_ready − M1_end)` = the hideable wait, and
  `(M6_start − M2_ready)` = the plan cost. This is the decisive number.
- Register/LDS impact: **none material.** Each stamp is a tid0-only `s_memrealtime` +
  `atomicMax` in a scope that already re-reads `desc`; nothing new crosses the M6→M7
  boundary, so the `VGPR 256 / AGPR 256 / scratch 60 B / LDS 155,428 B` tuple and the
  "no MPS value lives across the MFMA body" rule (`:1147-1150`) both hold. ISA-gate it.
- Confirm/refute thresholds (pre-register before running): `>200 µs` ⇒ build the
  role split at the M1/M2 boundary; `<50 µs` ⇒ close the axis in `LESSONS.md`.

**3 — `tile_desc` permutation as a pre-registered NULL control. ~10 device lines** —
reverse or stride the append order in the M4 serial builder (`:1060-1083`).
- Information for *this* question: ≈ 0 by construction (§"How M6 picks work"). Its
  value is that it *proves the knob is free* — the mechanical prerequisite for any
  future two-segment layout — and it doubles as an XCD/L2-locality axis, since
  `xcd = wg_id % 8` makes tile→CTA assignment a placement decision.
- Confirms: bit-identical output with a measurable timing delta ⇒ tile order is a
  real (free) tuning axis. Any pperr or correctness change ⇒ risk item (A) above is
  wrong and must be re-derived before anything else in this line is built.

**Do NOT build yet:** the mechanism experiments 1–2 would unlock is *not* layer-0. It
is a service pool at the **M1/M2** boundary doing route-independent `W13`/`S13`
prefetch (`.node/mps_n2_phase1_gm.cpp:96-97, 182`) — no plan change, no layout
change. Out of scope here; its own experiment folder.

## UNKNOWNs

- **UNKNOWN-1:** `n2_phase1_gm.cpp` (the authoritative M6 body,
  `k0pf6gm_device_tile_mps.hip:298`) is not in this repo, so every M6 claim above is
  cited from the `.node/mps_n2_phase1_gm.cpp` mirror. Settles it: sha256-compare the
  mirror against the donor include on the node before acting.
- **UNKNOWN-2:** spins→µs for experiment 1. The backoff constant lives in
  `poll_epoch_word_system` in `k0pf6_chunk_release.hpp` (`:89`), also not in this
  repo. Settles it: read that header on the node, or treat the spin count as a
  relative signal only (which is sufficient for the confirm/refute test as stated).
- **UNKNOWN-3 (a trap, not really unknown):** `K0_MPS_DEBUG_STOP` looks like a free
  per-phase timer but is **not** valid for >1 launch. Any early return skips M9, so
  `mega_count` never advances (`:1546`) and the epoch pins at 1; the retire-wait then
  passes trivially (`:525-527`) but M0 keeps zeroing the wrong `dest_counter` parity
  (`:585-595` vs M1's `par` at `:635`), so `pos` accumulates and trips pperr 65536
  (`:678`) from launch 2 on. Single-launch coarse probe only — a campaign-length
  debug_stop timing run is contaminated and violates "nonzero pperr is terminal".
