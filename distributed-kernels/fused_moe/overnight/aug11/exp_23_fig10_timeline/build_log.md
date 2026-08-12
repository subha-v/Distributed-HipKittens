# exp_23 Tier A — build log and parity gate

**Verdict: parity GREEN.** The per-CTA phase ring, compiled in and runtime-off,
is byte-identical to the published arm on every element of the resource tuple,
on the MFMA and atomic censuses, on the spill counts and on occupancy. The
timeline we trace will therefore be the timeline of the arm we publish.

Zero GPU. Four CPU-only `hipcc --genco` builds in container `subha_k1`, from the
scratch clone `~/e23/DHK` pinned at `5b1450d4` (0 dirty files). The pinned
checkout `~/Distributed-HipKittens` (`291dfa08`, held by the mode-14 campaign)
was never read or written. Gate script: `tools/e23_gate_A.sh`; raw output:
`gate_A_output.txt`.

## What was built

| TU | source | flags | what it is |
|---|---|---|---|
| `R` | scratch clone at `5b1450d4`, `SRC_REV 28` | — | **the arm.** `ts_last` at all five boundaries, no ring. |
| `TA` | Tier A, `SRC_REV 29` | — | **the shipped binary.** Ring compiled in, gated at runtime on `cfg.timestamps`. |
| `TAON` | Tier A | `-DK0P6_MPS_E23_FORCE_ON=1` | `enable` folded to compile-time `true`: coarse stamps *and* ring unconditional. |
| `TCON` | Tier A | `-DK0P6_MPS_E23_FORCE_ON=1 -DK0P6_MPS_E23_RING=0` | attribution control: coarse stamps unconditional, ring compiled **out**. |

`TAON` and `TCON` exist because "ring on" is not a build in the shipped design —
the ring is turned on by `K0_MPS_CFG timestamps=1` at runtime, and a runtime flag
cannot change a resource tuple. Folding `enable` to a constant is the only way to
ask the compiler what it would do if it could not hide the stamps behind a
branch, and `TCON` separates the cost of `timestamps=1` (which the pre-existing
coarse `atomicMax` cells already pay, and which is already known to cost ~15 µs
end-to-end) from the cost of the ring itself.

## The tuples

| | SGPR | VGPR | AGPR | scratch B/lane | LDS B/block | occupancy w/SIMD | SGPR spill | VGPR spill |
|---|---|---|---|---|---|---|---|---|
| **reference** (STATUS.md:145) | 106 | 256 | 256 | 128 | 155,496 | 1 | — | — |
| `R` — the arm | **106** | **256** | **256** | **128** | **155,496** | **1** | 217 | 17 |
| `TA` — **ring off, shipped** | **106** | **256** | **256** | **128** | **155,496** | **1** | **217** | **17** |
| `TAON` — ring + coarse forced on | 106 | 256 | 256 | 144 | 155,496 | 1 | 217 | 21 |
| `TCON` — coarse forced on, ring out | 106 | 256 | 256 | 144 | 155,496 | 1 | 217 | 21 |

**Ring-off vs the arm: equal on all eight columns.** Nothing was rounded or
tolerated; these are exact integers from
`-Rpass-analysis=kernel-resource-usage` on `k0pf6gm_mps_mega`.

**Occupancy is asserted, not assumed.** `Occupancy [waves/SIMD]: 1` is read out
of the remark for each TU and compared to `REF_OCC=1` by the gate; a rise to 2
prints `FAIL(occupancy=... -- ASSERTED)`. It held at 1 on all four TUs, including
both force-on builds, so the LDS/scratch failure mode that voided a sibling
experiment's figure did not occur here.

**Ring-on cost, attributed.** `TAON` and `TCON` are *identical* on scratch
(144 B), spills (217/21) and scratch op count (172). Their opcode histograms
differ by exactly the five ring stores and their address arithmetic. So the
whole +16 B/lane of scratch and +4 VGPR spills belong to making the
**pre-existing coarse stamps** unconditional — i.e. to `timestamps=1` — and
**the ring adds zero registers and zero scratch even when unconditionally
active.** Its cost is 5 × 8 B of plain global stores per CTA per epoch, and
nothing else. (Still: never compare a ring-on number to a ring-off one.)

## Gate results

| gate | requirement | result |
|---|---|---|
| G0 | four builds compile clean | **PASS** — 0 errors, ~7 s each |
| G1 | tuple exactly equals the reference | **PASS** for `TA`. `TAON`/`TCON` differ at scratch 144 as designed and are not arms. |
| G2 | occupancy == 1, asserted | **PASS** on all four |
| G3 | `v_mfma` == 180, `flat_atomic_pk_add_bf16` == 282 | **PASS** on all four |
| G4 | zero scratch ops inside either MFMA span | **PASS** on all four: spans of 96 and 84 MFMA, 0 scratch ops inside each |
| G5 | spills unchanged | **PASS** — `TA` 217/17 == `R` 217/17 |
| G6 | `.text(TA) != .text(R)` | **PASS** — `192,640 B` vs `192,448 B`, different sha256 |
| **G7a** | source census: `ts_mark`==5, `ts_last`==0, `e23_mark`==0 in the kernel | **PASS** |
| **G7b** | `s_memrealtime(TA) == s_memrealtime(R)` | **PASS — 12 == 12** |

### G7, the named risk, in detail

The failure mode being guarded is writing `ts_last(cell, e); e23_mark(base, ...)`
instead of the fused `ts_mark(...)`. It compiles, it passes the tuple, the figure
still looks right — and the two calls read the clock at two different instants,
which silently demotes the coarse/per-CTA reconciliation from an identity to a
few-tick approximation. That reconciliation is the one check that proves the array
indexes the right cell in the right epoch, so losing it quietly is worse than any
loud failure.

Both halves are mechanical, neither is an eyeball:

- **G7a**, source: `ts_mark(` == 5, `e23_mark(` == 0 and `ts_last(` == 0 in
  `k0pf6gm_device_tile_mps.hip`. A bare `e23_mark` at a stamp site — the
  substitution — makes the second count nonzero and fails.
- **G7b**, ISA: `s_memrealtime` census, `R` = 12, `TA` = 12, **delta 0**. Two
  clock reads per site would have made this 17.

Corroborated by the full opcode-histogram delta (`tools/e23_isa_delta.sh`),
which is the tightest statement available: `flat_atomic_umax_x2` is 8 in both
`R` and `TA`, so the coarse cells are still maintained by the same eight
atomics, and `TA − R` is *only*

```
flat_store_dwordx2  +5      the five ring stores
s_cmpk_gt_u32       +5      the bid < 256 bounds check
s_lshl_b32          +5   \
s_mov_b32           +5    |  ring address arithmetic
v_lshl_add_u64      +5    |
v_mov_b64_e32       +5   /
s_cbranch_scc1      +5      the bounds-check branch
s_nop               +3      scheduler padding
```

Eight opcodes, five of each, +192 B of `.text`, and nothing else moved.

## Reproduce

```bash
# on the node, CPU only
bash ~/overnight-scratch/e23/e23_gate_A.sh          # G0-G7, four TUs, ~30 s
bash ~/overnight-scratch/e23/e23_isa_delta.sh       # opcode histogram deltas
python3 ~/overnight-scratch/e23/e23_g4_spans.py \
  R=out/R.isa TA=out/TA.isa TAON=out/TAON.isa TCON=out/TCON.isa
```

Note on G4: the span check groups `v_mfma` indices into runs with gaps < 400
disassembly lines and counts scratch traffic strictly inside each run — the
established definition from `aug11/tools/e34_42_gate1.sh`. An earlier draft of
this gate used "any scratch op after the first `v_mfma`", which fires on the
**reference arm itself**, because the two MFMA spans are separated by ordinary
spill-carrying code. That draft was wrong, not the arm; it is fixed in
`tools/e23_g4_spans.py`.
