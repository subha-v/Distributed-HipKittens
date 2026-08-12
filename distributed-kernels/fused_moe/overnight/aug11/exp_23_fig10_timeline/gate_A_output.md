# exp_23 Tier A parity gate - raw output

Verbatim stdout of tools/e23_gate_A.sh, then tools/e23_isa_delta.sh, then
tools/e23_g4_spans.py, run on the node (CPU only, container subha_k1).
Interpretation is in build_log.md.

```text
Checking authorization now...
===INPUTS===
tierA_k0pf6gm_device_tile_mps.hip          7f9fc1f75556e79c  2176 lines
tier_moe_mps_adapter.cuh                   cc99ede896fa746b  1260 lines
scratch clone HEAD: 5b1450d4  0 dirty files
R .hip     (scratch clone)                 decc4b203fc4b069
R adapter  (scratch clone)                 91cc37f5696c550e

===LAY OUT THE TUs===
--- TA vs R source diff (must be ONLY the SRC_REV bump + 5 stamp sites) ---
  hip     @@ -104,7 +104,7 @@
  hip     @@ -1233,10 +1233,11 @@
  hip     @@ -1467,9 +1468,9 @@
  hip     @@ -1507,9 +1508,9 @@
  hip     @@ -1555,7 +1556,8 @@
  hip     @@ -2125,8 +2127,8 @@
  adapter @@ -57,6 +57,53 @@
  adapter @@ -554,6 +601,65 @@
--- SRC_REV ---
  R    SRC_REV=28
  TA   SRC_REV=29

===G7a: SOURCE census (mechanical, not eyeball)===
TU   ts_mark    e23_mark    e23_meta    ts_last
R    0          0           0           5
TA   5          0           0           0
G7a PASS

===BUILD (subha_k1, CPU-only genco)===
TAON exit=0 secs=7 errors=0
R exit=0 secs=7 errors=0
TCON exit=0 secs=7 errors=0
TA exit=0 secs=7 errors=0

===G0: compile errors===
---- R (0
0) ----
---- TA (0
0) ----
---- TAON (0
0) ----
---- TCON (0
0) ----
(no output above == all three compiled clean)

===G1/G2/G5: resource tuple, occupancy, spills (k0pf6gm_mps_mega)===
---- R ----
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark: Function Name: k0pf6gm_mps_mega [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     TotalSGPRs: 106 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     AGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     ScratchSize [bytes/lane]: 128 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     Occupancy [waves/SIMD]: 1 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     SGPRs Spill: 217 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs Spill: 17 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/R/k0pf6gm_device_tile_mps.hip:719:1: remark:     LDS Size [bytes/block]: 155496 [-Rpass-analysis=kernel-resource-usage]
---- TA ----
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark: Function Name: k0pf6gm_mps_mega [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     TotalSGPRs: 106 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     AGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     ScratchSize [bytes/lane]: 128 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     Occupancy [waves/SIMD]: 1 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     SGPRs Spill: 217 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs Spill: 17 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     LDS Size [bytes/block]: 155496 [-Rpass-analysis=kernel-resource-usage]
---- TAON ----
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark: Function Name: k0pf6gm_mps_mega [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     TotalSGPRs: 106 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     AGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     ScratchSize [bytes/lane]: 144 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     Occupancy [waves/SIMD]: 1 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     SGPRs Spill: 217 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs Spill: 21 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     LDS Size [bytes/block]: 155496 [-Rpass-analysis=kernel-resource-usage]
---- TCON ----
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark: Function Name: k0pf6gm_mps_mega [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     TotalSGPRs: 106 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     AGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     ScratchSize [bytes/lane]: 144 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     Occupancy [waves/SIMD]: 1 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     SGPRs Spill: 217 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     VGPRs Spill: 21 [-Rpass-analysis=kernel-resource-usage]
/home/subvadla/overnight-scratch/e23/tu/TA/k0pf6gm_device_tile_mps.hip:719:1: remark:     LDS Size [bytes/block]: 155496 [-Rpass-analysis=kernel-resource-usage]

--- G1/G2 verdict: exact equality with the reference tuple ---
G1/G2 R    PASS
G1/G2 TA   PASS
G1/G2 TAON  FAIL(scratch=144 want 128)
G1/G2 TCON  FAIL(scratch=144 want 128)

===G3/G4/G7b: ISA census===
TU   v_mfma    pk_add_bf16   scratch_ops  s_memrealtime  s_barrier
R    180       282           168          12             82
TA   180       282           168          12             82
TAON 180       282           172          12             82
TCON 180       282           172          12             82
G7b PASS: s_memrealtime R=12 TA=12 -- tier A added ZERO clock reads,
     so one read feeds both the coarse cell and the ring cell.
G4 scratch ops INSIDE either MFMA span (must be 0 on every TU).
  ---- R ----
  span   7231-7854     96 mfma  scratch ops inside = 0
  span  11148-11727    84 mfma  scratch ops inside = 0
  G4 PASS (2 span(s), 0 total scratch ops inside MFMA)
  ---- TA ----
  span   7243-7866     96 mfma  scratch ops inside = 0
  span  11166-11745    84 mfma  scratch ops inside = 0
  G4 PASS (2 span(s), 0 total scratch ops inside MFMA)
  ---- TAON ----
  span   7239-7868     96 mfma  scratch ops inside = 0
  span  11146-11725    84 mfma  scratch ops inside = 0
  G4 PASS (2 span(s), 0 total scratch ops inside MFMA)
  ---- TCON ----
  span   7225-7854     96 mfma  scratch ops inside = 0
  span  11125-11704    84 mfma  scratch ops inside = 0
  G4 PASS (2 span(s), 0 total scratch ops inside MFMA)

===G6: .text sha256 (TA must DIFFER from R)===
R    668f26081494c97dcb6a51f89e0bceab  192448 B
TA   ac39c503ea037b645f4b84c7a2d47b2d  192640 B
TAON a48a637ee7520271dd752f4c61fefb38  192576 B
TCON fefaf361ac3369c12495ccf02c55239f  192384 B
G6 TA PASS: .text differs from R
G6 TAON PASS: .text differs from R
G6 TCON PASS: .text differs from R
===DONE===

########## ISA DELTA ##########
=== headline counts ===
TU    s_memrealtime  flat_atomic_umax_x2 s_waitcnt   isa_lines   text_B
R     12             8                2121        34702       192448
TA    12             8                2121        34740       192640
TAON  12             8                2115        34712       192576
TCON  12             8                2115        34674       192384

=== opcode histogram delta: TA vs R (only opcodes whose count moved) ===
opcode                                    R       TA    delta
flat_store_dwordx2                      117      122       +5
s_cbranch_scc1                           31       36       +5
s_cmpk_gt_u32                             0        5       +5
s_lshl_b32                               28       33       +5
s_mov_b32                               137      142       +5
s_nop                                  1050     1053       +3
v_lshl_add_u64                          943      948       +5
v_mov_b64_e32                           217      222       +5

=== opcode histogram delta: TAON vs R (only opcodes whose count moved) ===
opcode                                    R     TAON    delta
flat_store_dwordx2                      117      122       +5
s_and_b64                              1111     1106       -5
s_cbranch_execz                        1947     1943       -4
s_cbranch_scc1                           31       36       +5
s_cmpk_gt_u32                             0        5       +5
scratch_load_dwordx2                      7        9       +2
scratch_store_dwordx2                     7        9       +2
s_lshl_b32                               28       33       +5
s_mov_b32                               137      142       +5
s_nop                                  1050     1043       -7
s_waitcnt                              2121     2115       -6
s_xor_b64                               595      594       -1
v_accvgpr_read_b32                      728      732       +4
v_accvgpr_write_b32                     121      129       +8
v_and_b32_e32                           881      876       -5
v_cmp_ne_u32_e32                        100       95       -5
v_fma_f32                                33       27       -6
v_lshl_add_u64                          943      948       +5
v_mov_b32_e32                          1680     1676       -4
v_mov_b64_e32                           217      222       +5
v_mul_f32_e64                            42       40       -2
v_pk_fma_f32                            408      411       +3
v_pk_mul_f32                            165      166       +1

=== opcode histogram delta: TCON vs R (only opcodes whose count moved) ===
opcode                                    R     TCON    delta
s_and_b64                              1111     1106       -5
s_cbranch_execz                        1947     1943       -4
scratch_load_dwordx2                      7        9       +2
scratch_store_dwordx2                     7        9       +2
s_nop                                  1050     1040      -10
s_waitcnt                              2121     2115       -6
s_xor_b64                               595      594       -1
v_accvgpr_read_b32                      728      732       +4
v_accvgpr_write_b32                     121      129       +8
v_and_b32_e32                           881      876       -5
v_cmp_ne_u32_e32                        100       95       -5
v_fma_f32                                33       27       -6
v_mov_b32_e32                          1680     1676       -4
v_mul_f32_e64                            42       40       -2
v_pk_fma_f32                            408      411       +3
v_pk_mul_f32                            165      166       +1

########## G4 SPANS ##########
---- R ----
  span   7231-7854     96 mfma  scratch ops inside = 0
  span  11148-11727    84 mfma  scratch ops inside = 0
  mfma total 180 in 2 span(s); scratch ops inside spans = 0  PASS
---- TA ----
  span   7243-7866     96 mfma  scratch ops inside = 0
  span  11166-11745    84 mfma  scratch ops inside = 0
  mfma total 180 in 2 span(s); scratch ops inside spans = 0  PASS
---- TAON ----
  span   7239-7868     96 mfma  scratch ops inside = 0
  span  11146-11725    84 mfma  scratch ops inside = 0
  mfma total 180 in 2 span(s); scratch ops inside spans = 0  PASS
---- TCON ----
  span   7225-7854     96 mfma  scratch ops inside = 0
  span  11125-11704    84 mfma  scratch ops inside = 0
  mfma total 180 in 2 span(s); scratch ops inside spans = 0  PASS
G4 PASS on every TU
```
