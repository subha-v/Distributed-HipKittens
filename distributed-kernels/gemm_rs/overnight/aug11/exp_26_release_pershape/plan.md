# exp_26 — per-shape release group (`rgroup = min(RELEASE_GROUP, tiles_per_cta)`)

Pre-registered before the first GPU run. One mechanism, one constant's worth of
logic.

## The finding this acts on

`aug11/exp_20_attribution/ablation.json`: `release` collapsed 134.9 → 15.0 µs on
shape 6 (below that shape's 69.87 µs floor) once `RELEASE_GROUP=4` landed, but it
is still **65.4 µs, 10.2% of shape 5's 641.9 µs**, and shape 5 is one of the two
shapes we still lose to the frozen rank-1 submission (graded ratio 1.286×).

Cause: the shipped rule is a step function of `tiles_per_cta` against a single
constant — `rgroup = (tiles_per_cta >= 4) ? 4 : 1`. Tiles per producer CTA across
the six graded rows is **1 / 1 / 1 / 1 / 2 / 4**, so shape 6 groups and shape 5
issues one `release_payload_system()` per tile. Independent corroboration from
the same experiment's counter pass: release-driven L2 writebacks are **13.1% on
shape 6 against 56.4% on shape 5**.

## The change

`gemm_rs_mi300x.cpp`, producer role, the `rgroup` expression only:

```c
const int rcap   = tiles_per_cta < RELEASE_GROUP ? tiles_per_cta : RELEASE_GROUP;
const int rgroup = rcap > 1 ? rcap : 1;
```

behind `HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE` (default 1), nested inside the
existing `FULL_ONLY` branch so the incumbent rule remains compilable as an arm.
Nothing else moves: the group loop, the release site, the publish loop, the
per-tile credit wait, `decode_tile`, the tile order, `BM/BN/BK`,
`num_reducer_ctas`, `find_scored_config` and the 72-byte descriptor ABI are
untouched.

Effective group per graded row: `1/1/1/1/1/4` → `1/1/1/1/**2**/4`. Generic row
(118 tiles/CTA) stays at 4. **Only shape 5 changes.**

### Where the value comes from, and why

Runtime-derived, exactly as today — *not* promoted to a compile-time constant.
Three reasons:

1. **Rows 4 and 5 share one instantiation.** `dispatch_gemm_rs_mi300x` sends both
   to `launch_fixed<256,256,32,false>`; they differ only in `M`. A template
   constant cannot separate them without adding an instantiation, which changes
   `M2_EXPECT` and re-allocates registers for a row whose delta this experiment
   is trying to read.
2. **The incumbent's `rgroup` is already runtime.** Folding it now would be a
   second, confounded change on top of the one being measured — and exp_09
   showed that folded / bounded / runtime give the 128×256 instantiation three
   different schedules (2.13% on shape 3). Keeping it runtime is what makes the
   four one-tile rows real controls.
3. Routing it through the descriptor would touch the ABI for a value both the
   host and the device already derive from `tiles` and `num_gemm_ctas`.

### Fullness, and the risk that is actually left

`rgroup ≤ tiles_per_cta = ceil(tiles / num_gemm_ctas)`, the tile count of the CTA
that owns the most, and `emitted = min(left, rgroup)` still truncates a short
CTA's last group to what it owns. So no CTA ever defers a publication waiting for
a tile that does not exist — which is the defect of the unconditional arm, where
`RELEASE_GROUP=4` on a 2-tile shape strode the loop by four CTAs' worth of tiles
and could only ever half-fill a group.

**But the honest residual is publication delay, and it is not removed.** On shape
5 the first of a CTA's two tiles is now published one mainloop later. That is the
same trade E3 measured as **−3.75%** on the then-2-tile 512×4096×12288 and as
inside its own noise on 8192×4096×14336 itself. Shape 2 has since been retiled to
64/128 (256 tiles, 1 per CTA), so it can no longer group at all and is now a
control rather than a casualty — but the mechanism that cost it 3.75% is exactly
what shape 5 will now do, at 7× the shape size and with 2 reduce rounds. This is
a measurement, not a deduction.

## Pre-registered expectations

| # | shape | t/CTA | rgroup 0→1 | expected |
|---|---|---:|---|---|
| 1 | 64×7168×18432 | 1 | 1 → 1 | **unchanged** (control; host-bound anyway) |
| 2 | 512×4096×12288 | 1 | 1 → 1 | **unchanged** (control) |
| 3 | 2048×2880×2880 | 1 | 1 → 1 | **unchanged** (control) |
| 4 | 4096×4096×4096 | 1 | 1 → 1 | **unchanged** (control) |
| 5 | 8192×4096×14336 | 2 | 1 → **2** | release pool halves: ~33 µs of 641.9 ≈ **−5%** |
| 6 | 8192×8192×29568 | 4 | 4 → 4 | **unchanged** (control) |

If a control moves it is codegen, not the release, and it gets investigated
rather than absorbed. If shape 5 loses, the publication-delay term dominates the
writeback term at this size and the incumbent source state is kept.

Denominators (our harness, current best-of-arm µs):
`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63`.
Null-arm floors, µs-equivalent: `3.48 / 0.93 / 1.61 / 4.90 / 13.93 / 69.87`.
**Shape 5's floor is ~13.9 µs, so the pre-registered −33 µs is ~2.4× floor.**

## Measurement protocol

- Full gate ladder (`tools/gate_ladder.sh exp_26_release_pershape`): M0 clean
  node → M1 build → M2 resources/ISA → M3 17 shapes at **both** 1e-2 and 2e-3 →
  M4 three negative controls → M5 600-epoch soak → M7.
- Then **`harness/m9_stale_slot.py`**, mandatory here: poisoned heap, bitwise
  golden against the frozen pre-E3 build, and `CTRL_PUBLISH_EARLY`, which must
  still FAIL (it currently fails on 6 of 6 shapes, concentrated at epoch 1). If
  it stops failing the gate has gone blind and a pass means nothing.
- Then `ab_pershape.py`, four arms interleaved in one process, in **both**
  allocation orders: `ps0` (incumbent), `ps1` (candidate), `rg2c`
  (`RELEASE_GROUP=2`, incumbent rule — computes rgroup 2 on shape 5 by a
  different expression, so agreement there is behavioural evidence the
  candidate's group really became 2), `ps0b` (**null arm**, a second build of
  `ps0`). Best and median only; a delta counts only if it clears that shape's
  own null arm with disjoint sample ranges in both directions.
- Freshness is proved by behaviour, never by mtime: `isa_diff.sh` requires the
  `ps0`/`ps1` ISA to differ, and `build_arms.sh` removes each `.so` first.
- Clocks pinned; duration-based warmup; one 8-GPU job at a time.

## Protocol conditions re-read before the first run

`experiments/exp_05_release_granularity/protocol_review.md` §2 and C1–C8. The
change touches none of the structures those conditions constrain — it only picks
a different value for an existing bounded, runtime `rgroup`:

- **C1** publication moves with the release — unchanged, one release site
  dominating one publish loop inside the group body.
- **C2** the group body cannot run with zero tiles — unchanged (`t0 < tiles`).
- **C3** publish recomputes via `decode_tile` — unchanged.
- **C4** compile-time hard cap — unchanged (`RELEASE_GROUP ≤ 8` static_assert);
  the new rule can only ever *lower* the group below the cap.
- **C5–C8** the one-tile rows must be bit-identical: enforced twice, by the A/B's
  cross-arm `torch.equal` and by m9's bitwise golden.
- `rgroup == 0` would make `t0 += rgroup * stride` an infinite loop; the `max(1,
  ·)` clamp removes that value from the expression's range.
