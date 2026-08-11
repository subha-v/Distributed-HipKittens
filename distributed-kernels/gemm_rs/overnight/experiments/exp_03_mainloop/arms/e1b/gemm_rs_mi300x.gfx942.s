	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[22:23], s[0:1], 0x60
	s_load_dwordx8 s[24:31], s[0:1], 0x100
	s_load_dwordx8 s[36:43], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
                                        ; implicit-def: $vgpr90 : SGPR spill to VGPR lane
	s_load_dwordx2 s[34:35], s[0:1], 0xf8
	s_load_dwordx2 s[20:21], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s86, s25, 31
	s_lshr_b32 s3, s86, 29
	v_writelane_b32 v90, s4, 0
	s_add_i32 s3, s25, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v90, s5, 1
	v_writelane_b32 v90, s6, 2
	v_writelane_b32 v90, s7, 3
	s_mov_b64 s[6:7], -1
	s_cmp_ge_i32 s2, s31
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_cbranch_scc0 .LBB0_73
; %bb.1:
	s_sub_i32 s14, s2, s31
	s_mov_b32 s15, 0
	s_lshl_b64 s[6:7], s[14:15], 2
	s_add_u32 s6, s42, s6
	s_addc_u32 s7, s43, s7
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB0_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit320
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b64 s[16:17], -1, 0
	s_cmp_eq_u64 s[10:11], 0
	s_cbranch_scc1 .LBB0_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB0_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit323
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_branch .LBB0_8
.LBB0_7:
	s_mov_b64 s[6:7], -1
.LBB0_8:                                ; %Flow829
	s_mov_b64 s[8:9], exec
	v_writelane_b32 v90, s8, 28
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_nop 0
	v_writelane_b32 v90, s9, 29
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB0_72
; %bb.9:                                ; %.critedge
	s_mul_i32 s93, s29, s28
	s_cmp_ge_i32 s14, s93
	s_cbranch_scc1 .LBB0_72
; %bb.10:                               ; %.lr.ph
	s_mul_hi_i32 s13, s33, s26
	s_mul_i32 s12, s33, s26
	s_ashr_i32 s49, s26, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_add_u32 s50, s22, s6
	s_addc_u32 s51, s23, s7
	s_add_u32 s52, s50, s6
	s_addc_u32 s53, s51, s7
	s_add_u32 s54, s52, s6
	s_addc_u32 s55, s53, s7
	s_add_u32 s56, s54, s6
	s_addc_u32 s57, s55, s7
	s_add_u32 s58, s56, s6
	s_addc_u32 s59, s57, s7
	s_add_u32 s60, s58, s6
	s_addc_u32 s61, s59, s7
	s_add_u32 s62, s60, s6
	s_addc_u32 s63, s61, s7
	s_load_dwordx2 s[6:7], s[0:1], 0x120
	s_load_dwordx2 s[80:81], s[0:1], 0x90
	s_mov_b32 s48, s26
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_mbcnt_hi_u32_b32 v4, -1, v4
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[20:21], -1, 0
	s_lshl_b32 s15, s30, 3
	s_add_i32 s3, s6, 64
	s_cmp_lg_u32 s24, 0
	v_writelane_b32 v90, s3, 8
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 10
	s_cmp_lg_u32 s24, 1
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v90, s7, 11
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 20
	s_cmp_lg_u32 s24, 2
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v90, s7, 21
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 4
	s_cmp_lg_u32 s24, 3
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v90, s7, 5
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 16
	s_cmp_lg_u32 s24, 4
	v_mul_lo_u32 v64, s28, v0
	v_writelane_b32 v90, s7, 17
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 6
	s_cmp_lg_u32 s24, 5
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v90, s7, 7
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 14
	s_cmp_lg_u32 s24, 6
	v_cmp_gt_i32_e64 s[10:11], s15, v0
	v_writelane_b32 v90, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 26
	s_cmp_lg_u32 s24, 7
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v90, s7, 27
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s87, s29
	v_cvt_f32_u32_e32 v2, s87
	s_sub_i32 s18, 0, s87
	s_ashr_i32 s3, s29, 31
	s_lshl_b64 s[82:83], s[48:49], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v90, s6, 22
	v_mov_b32_e32 v3, v5
	s_mov_b64 s[98:99], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v90, s7, 23
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_bfrev_b32_e32 v66, 32
	v_readfirstlane_b32 s19, v2
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s88, s19, s18
	s_add_u32 s18, s80, 2
	s_addc_u32 s19, s81, 0
	v_writelane_b32 v90, s18, 24
	v_lshrrev_b32_e32 v2, 3, v0
	s_movk_i32 s89, 0x7fff
	v_writelane_b32 v90, s19, 25
	s_mul_i32 s18, s13, 14
	s_mul_hi_u32 s19, s12, 14
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 14
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	s_add_u32 s18, s18, 2
	s_addc_u32 s19, s19, 0
	v_writelane_b32 v90, s18, 18
	s_mov_b32 s90, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	v_writelane_b32 v90, s19, 19
	s_lshl_b64 s[18:19], s[12:13], 2
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	v_writelane_b32 v90, s18, 12
	s_nop 1
	v_writelane_b32 v90, s19, 13
	s_mul_i32 s18, s13, 12
	s_mul_hi_u32 s19, s12, 12
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 12
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	s_add_u32 s18, s18, 2
	s_addc_u32 s19, s19, 0
	v_writelane_b32 v90, s18, 30
	s_nop 1
	v_writelane_b32 v90, s19, 31
	s_mul_i32 s18, s13, 6
	s_mul_hi_u32 s19, s12, 6
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 6
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	v_writelane_b32 v90, s18, 32
	s_nop 1
	v_writelane_b32 v90, s19, 33
	s_mul_i32 s18, s13, 10
	s_mul_hi_u32 s19, s12, 10
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 10
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	s_add_u32 s94, s18, 2
	s_addc_u32 s95, s19, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	s_add_u32 s96, s22, s12
	s_addc_u32 s97, s23, s13
	s_xor_b64 s[64:65], s[20:21], -1
	s_branch .LBB0_13
.LBB0_11:                               ; %Flow814
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s14, s31
	s_add_i32 s14, s12, 0x130
	s_cmp_ge_i32 s14, s93
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB0_12:                               ; %Flow826
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB0_72
.LBB0_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_17 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
                                        ;       Child Loop BB0_33 Depth 3
	s_abs_i32 s13, s14
	s_mul_hi_u32 s18, s13, s88
	s_mul_i32 s19, s18, s87
	s_ashr_i32 s12, s14, 31
	s_sub_i32 s13, s13, s19
	s_xor_b32 s12, s12, s3
	s_add_i32 s19, s18, 1
	s_sub_i32 s20, s13, s87
	s_cmp_ge_u32 s13, s87
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s13, s20, s13
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s13, s87
	s_cselect_b32 s13, s19, s18
	s_xor_b32 s13, s13, s12
	s_sub_i32 s91, s13, s12
	s_mul_i32 s12, s91, s29
	s_sub_i32 s92, s14, s12
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB0_21
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=1
	v_add_u32_e32 v4, s91, v64
	v_mul_lo_u32 v4, v4, s29
	v_add_u32_e32 v6, s92, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[38:39]
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_21
; %bb.15:                               ; %.lr.ph.i.i.i325.preheader
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[20:21], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB0_17
.LBB0_16:                               ; %Flow822
                                        ;   in Loop: Header=BB0_17 Depth=2
	s_and_b64 s[18:19], exec, s[68:69]
	s_or_b64 s[20:21], s[18:19], s[20:21]
	s_andn2_b64 s[18:19], s[66:67], exec
	s_and_b64 s[44:45], s[72:73], exec
	s_or_b64 s[66:67], s[18:19], s[44:45]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB0_19
.LBB0_17:                               ; %.lr.ph.i.i.i325
                                        ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s70, s70, 1
	s_addc_u32 s71, s71, 0
	v_mov_b64_e32 v[8:9], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[70:71], v[8:9]
	s_mov_b64 s[72:73], -1
	s_or_b64 s[68:69], s[68:69], exec
	s_cbranch_vccnz .LBB0_16
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=2
	s_sleep 4
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[18:19], s[68:69], exec
	s_mov_b64 s[72:73], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[44:45], vcc, exec
	s_or_b64 s[68:69], s[18:19], s[44:45]
	s_branch .LBB0_16
.LBB0_19:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[18:19], s[66:67]
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_21
; %bb.20:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_atomic_or v[6:7], v66
.LBB0_21:                               ; %.critedge489
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[16:17]
	s_mov_b64 s[18:19], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_25
; %bb.22:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_load_dword v4, v[6:7] sc1
.LBB0_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit331
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[18:19], 0, v4
.LBB0_25:                               ; %Flow825
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execz .LBB0_12
; %bb.26:                               ; %.critedge503
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB0_39
; %bb.27:                               ; %.lr.ph.i332.preheader
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_mul_i32 s68, s91, s30
	s_lshl_b32 s70, s92, 6
	s_ashr_i32 s69, s68, 31
	s_ashr_i32 s71, s70, 31
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[68:69]
	v_mov_b64_e32 v[8:9], s[70:71]
	v_mad_u64_u32 v[8:9], s[18:19], s48, v6, v[8:9]
	v_mul_lo_u32 v4, s48, v7
	v_mul_lo_u32 v6, s49, v6
	v_add3_u32 v9, v6, v9, v4
	v_readlane_b32 s18, v90, 24
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s19, v90, 25
	v_lshl_add_u64 v[6:7], s[22:23], 0, v[22:23]
	v_lshl_add_u64 v[10:11], s[50:51], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v90, 18
	v_readlane_b32 s19, v90, 19
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	s_mov_b64 s[72:73], 0
	v_lshl_add_u64 v[12:13], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v90, 12
	v_readlane_b32 s19, v90, 13
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[14:15], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v90, 30
	v_readlane_b32 s19, v90, 31
	s_nop 1
	v_lshl_add_u64 v[16:17], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v90, 32
	v_readlane_b32 s19, v90, 33
	s_nop 1
	v_lshl_add_u64 v[18:19], s[18:19], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	s_branch .LBB0_29
.LBB0_28:                               ; %.critedge.i335
                                        ;   in Loop: Header=BB0_29 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s15, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[82:83]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[82:83]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[82:83]
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[82:83]
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[82:83]
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[82:83]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[82:83]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[82:83]
	s_or_b64 s[72:73], vcc, s[72:73]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	s_andn2_b64 exec, exec, s[72:73]
	s_cbranch_execz .LBB0_39
.LBB0_29:                               ; %.lr.ph.i332
                                        ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_33 Depth 3
	v_lshlrev_b32_e32 v4, 3, v69
	v_and_or_b32 v24, v4, 56, s70
	v_mov_b32_e32 v25, s71
	v_lshl_add_u64 v[26:27], v[24:25], 0, 8
	v_cmp_lt_u64_e32 vcc, s[48:49], v[26:27]
	s_or_b64 s[18:19], s[64:65], vcc
	s_and_saveexec_b64 s[20:21], s[18:19]
	s_xor_b64 s[74:75], exec, s[20:21]
	s_cbranch_execz .LBB0_37
; %bb.30:                               ; %.preheader.i334.preheader
                                        ;   in Loop: Header=BB0_29 Depth=2
	v_and_b32_e32 v4, 56, v68
	v_lshl_add_u64 v[24:25], s[70:71], 0, v[4:5]
	v_lshlrev_b32_e32 v4, 1, v68
	v_and_b32_e32 v4, 0x70, v4
	v_lshl_add_u64 v[26:27], v[6:7], 0, v[4:5]
	v_lshl_add_u64 v[28:29], v[8:9], 0, v[4:5]
	v_lshl_add_u64 v[30:31], v[10:11], 0, v[4:5]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[4:5]
	v_lshl_add_u64 v[34:35], v[14:15], 0, v[4:5]
	v_lshl_add_u64 v[36:37], v[16:17], 0, v[4:5]
	v_lshl_add_u64 v[38:39], v[18:19], 0, v[4:5]
	v_lshl_add_u64 v[40:41], v[20:21], 0, v[4:5]
	v_lshl_add_u64 v[42:43], v[22:23], 0, v[4:5]
	s_mov_b64 s[76:77], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr78_sgpr79
	s_branch .LBB0_33
.LBB0_31:                               ; %Flow816
                                        ;   in Loop: Header=BB0_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[44:45], s[78:79], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[78:79], s[44:45], s[18:19]
.LBB0_32:                               ; %Flow815
                                        ;   in Loop: Header=BB0_33 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[18:19], exec, s[78:79]
	s_or_b64 s[76:77], s[18:19], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB0_36
.LBB0_33:                               ; %.preheader.i334
                                        ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[48:49], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_32
; %bb.34:                               ; %.preheader.i334.1
                                        ;   in Loop: Header=BB0_33 Depth=3
	v_lshl_add_u64 v[48:49], v[26:27], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[30:31], 0, v[44:45]
	global_load_ushort v4, v[48:49], off
	global_load_ushort v72, v[50:51], off
	v_lshl_add_u64 v[52:53], v[34:35], 0, v[44:45]
	global_load_ushort v73, v[52:53], off
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[44:45]
	global_load_ushort v74, v[54:55], off
	v_lshl_add_u64 v[56:57], v[42:43], 0, v[44:45]
	global_load_ushort v75, v[56:57], off
	v_lshl_add_u64 v[58:59], v[40:41], 0, v[44:45]
	global_load_ushort v76, v[58:59], off offset:-2
	v_lshl_add_u64 v[60:61], v[36:37], 0, v[44:45]
	global_load_ushort v77, v[60:61], off offset:-2
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[44:45]
	global_load_ushort v78, v[62:63], off offset:-2
	v_lshl_add_u64 v[70:71], v[24:25], 0, 1
	v_cmp_gt_u64_e32 vcc, s[48:49], v[70:71]
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[44:45]
	s_mov_b64 s[18:19], -1
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v70, 16, v72
	v_add_f32_e32 v4, v4, v70
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v71, 16, v73
	v_add_f32_e32 v4, v4, v71
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v72, 16, v74
	v_add_f32_e32 v4, v4, v72
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v73, 16, v75
	v_add_f32_e32 v4, v4, v73
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v74, 16, v76
	v_add_f32_e32 v4, v4, v74
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v75, 16, v77
	v_add_f32_e32 v4, v4, v75
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v76, 16, v78
	v_add_f32_e32 v4, v4, v76
	v_bfe_u32 v70, v4, 16, 1
	v_add3_u32 v4, v4, v70, s89
	global_store_short_d16_hi v[46:47], v4, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB0_31
; %bb.35:                               ;   in Loop: Header=BB0_33 Depth=3
	global_load_ushort v4, v[50:51], off offset:2
	s_nop 0
	global_load_ushort v48, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v49, v[52:53], off offset:2
	global_load_ushort v50, v[54:55], off offset:2
	global_load_ushort v51, v[56:57], off offset:2
	s_nop 0
	global_load_ushort v52, v[58:59], off
	global_load_ushort v53, v[60:61], off
	global_load_ushort v54, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[24:25], v[24:25], 0, 2
	s_orn2_b64 s[18:19], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v48, 16, v48
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v4, v48, v4
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v4, v4, v49
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v4, v4, v50
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v4, v4, v51
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v53, 16, v53
	v_add_f32_e32 v4, v4, v52
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v4, v4, v53
	v_add_f32_e32 v4, v4, v54
	v_bfe_u32 v48, v4, 16, 1
	v_add3_u32 v4, v4, v48, s89
	global_store_short_d16_hi v[46:47], v4, off
	s_branch .LBB0_31
.LBB0_36:                               ; %Flow817
                                        ;   in Loop: Header=BB0_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB0_37:                               ; %Flow818
                                        ;   in Loop: Header=BB0_29 Depth=2
	s_andn2_saveexec_b64 vcc, s[74:75]
	s_cbranch_execz .LBB0_28
; %bb.38:                               ;   in Loop: Header=BB0_29 Depth=2
	v_lshrrev_b32_e32 v4, 3, v69
	v_lshl_add_u64 v[26:27], v[4:5], 0, s[68:69]
	v_mul_lo_u32 v4, v26, s49
	v_mul_lo_u32 v27, v27, s48
	v_mad_u64_u32 v[24:25], s[18:19], v26, s48, v[24:25]
	v_add3_u32 v25, v27, v25, v4
	v_lshlrev_b64 v[56:57], 1, v[24:25]
	v_lshl_add_u64 v[24:25], s[22:23], 0, v[56:57]
	v_lshl_add_u64 v[28:29], s[50:51], 0, v[56:57]
	global_load_dwordx4 v[24:27], v[24:25], off
	v_lshl_add_u64 v[32:33], s[52:53], 0, v[56:57]
	global_load_dwordx4 v[28:31], v[28:29], off
	v_lshl_add_u64 v[36:37], s[54:55], 0, v[56:57]
	global_load_dwordx4 v[32:35], v[32:33], off
	v_lshl_add_u64 v[40:41], s[56:57], 0, v[56:57]
	global_load_dwordx4 v[36:39], v[36:37], off
	v_lshl_add_u64 v[44:45], s[58:59], 0, v[56:57]
	global_load_dwordx4 v[40:43], v[40:41], off
	v_lshl_add_u64 v[48:49], s[60:61], 0, v[56:57]
	global_load_dwordx4 v[44:47], v[44:45], off
	v_lshl_add_u64 v[52:53], s[62:63], 0, v[56:57]
	global_load_dwordx4 v[48:51], v[48:49], off
	v_lshl_add_u64 v[56:57], s[80:81], 0, v[56:57]
	global_load_dwordx4 v[52:55], v[52:53], off
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v59, 0xffff0000, v24
	v_and_b32_e32 v61, 0xffff0000, v25
	v_lshlrev_b32_e32 v58, 16, v24
	v_lshlrev_b32_e32 v60, 16, v25
	s_waitcnt vmcnt(6)
	v_and_b32_e32 v25, 0xffff0000, v28
	v_lshlrev_b32_e32 v24, 16, v28
	v_and_b32_e32 v63, 0xffff0000, v26
	v_and_b32_e32 v71, 0xffff0000, v27
	v_lshlrev_b32_e32 v62, 16, v26
	v_lshlrev_b32_e32 v70, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v29
	v_lshlrev_b32_e32 v26, 16, v29
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v29, 0xffff0000, v32
	v_lshlrev_b32_e32 v28, 16, v32
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
	v_and_b32_e32 v73, 0xffff0000, v30
	v_and_b32_e32 v75, 0xffff0000, v31
	v_lshlrev_b32_e32 v72, 16, v30
	v_lshlrev_b32_e32 v74, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v33
	v_lshlrev_b32_e32 v30, 16, v33
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v33, 0xffff0000, v36
	v_lshlrev_b32_e32 v32, 16, v36
	v_pk_add_f32 v[26:27], v[26:27], v[60:61]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_and_b32_e32 v77, 0xffff0000, v34
	v_and_b32_e32 v79, 0xffff0000, v35
	v_lshlrev_b32_e32 v76, 16, v34
	v_lshlrev_b32_e32 v78, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v37
	v_lshlrev_b32_e32 v34, 16, v37
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v36, 16, v40
	v_and_b32_e32 v37, 0xffff0000, v40
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[32:33]
	v_and_b32_e32 v81, 0xffff0000, v38
	v_and_b32_e32 v83, 0xffff0000, v39
	v_lshlrev_b32_e32 v80, 16, v38
	v_lshlrev_b32_e32 v82, 16, v39
	v_lshlrev_b32_e32 v38, 16, v41
	v_and_b32_e32 v39, 0xffff0000, v41
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v84, 16, v44
	v_and_b32_e32 v85, 0xffff0000, v44
	v_pk_add_f32 v[26:27], v[26:27], v[34:35]
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	v_lshlrev_b32_e32 v40, 16, v45
	v_and_b32_e32 v41, 0xffff0000, v45
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v44, 16, v48
	v_and_b32_e32 v45, 0xffff0000, v48
	v_pk_add_f32 v[26:27], v[26:27], v[38:39]
	v_pk_add_f32 v[24:25], v[24:25], v[84:85]
	v_lshlrev_b32_e32 v86, 16, v49
	v_and_b32_e32 v87, 0xffff0000, v49
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v88, 16, v52
	v_and_b32_e32 v89, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[88:89]
	v_pk_add_f32 v[26:27], v[26:27], v[48:49]
	v_bfe_u32 v29, v25, 16, 1
	v_bfe_u32 v30, v24, 16, 1
	v_pk_add_f32 v[52:53], v[72:73], v[62:63]
	v_bfe_u32 v4, v27, 16, 1
	v_bfe_u32 v28, v26, 16, 1
	v_add3_u32 v32, v24, v30, s89
	v_add3_u32 v33, v25, v29, s89
	v_pk_add_f32 v[24:25], v[74:75], v[70:71]
	v_add3_u32 v34, v26, v28, s89
	v_add3_u32 v4, v27, v4, s89
	v_pk_add_f32 v[24:25], v[24:25], v[78:79]
	v_pk_add_f32 v[26:27], v[52:53], v[76:77]
	v_pk_add_f32 v[24:25], v[24:25], v[82:83]
	v_pk_add_f32 v[26:27], v[26:27], v[80:81]
	v_lshlrev_b32_e32 v28, 16, v42
	v_lshlrev_b32_e32 v30, 16, v43
	v_and_b32_e32 v29, 0xffff0000, v42
	v_and_b32_e32 v31, 0xffff0000, v43
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v47
	v_lshlrev_b32_e32 v30, 16, v46
	v_and_b32_e32 v29, 0xffff0000, v47
	v_and_b32_e32 v31, 0xffff0000, v46
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v50
	v_lshlrev_b32_e32 v30, 16, v51
	v_and_b32_e32 v29, 0xffff0000, v50
	v_and_b32_e32 v31, 0xffff0000, v51
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v55
	v_lshlrev_b32_e32 v30, 16, v54
	v_and_b32_e32 v29, 0xffff0000, v55
	v_and_b32_e32 v31, 0xffff0000, v54
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_bfe_u32 v30, v27, 16, 1
	v_bfe_u32 v28, v25, 16, 1
	v_bfe_u32 v29, v24, 16, 1
	v_bfe_u32 v31, v26, 16, 1
	v_add3_u32 v26, v26, v31, s89
	v_add3_u32 v30, v27, v30, s89
	v_add3_u32 v24, v24, v29, s89
	v_add3_u32 v25, v25, v28, s89
	v_perm_b32 v27, v25, v24, s90
	v_perm_b32 v26, v30, v26, s90
	v_perm_b32 v25, v4, v34, s90
	v_perm_b32 v24, v33, v32, s90
	global_store_dwordx4 v[56:57], v[24:27], off
	s_branch .LBB0_28
.LBB0_39:                               ; %Flow820
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB0_11
; %bb.40:                               ; %.preheader509
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s28, s24
	v_readlane_b32 s13, v90, 8
	s_add_i32 s12, s91, s12
	s_add_i32 s13, s13, s92
	s_mul_i32 s12, s12, s29
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s18, s38, s12
	s_addc_u32 s19, s39, s13
	v_readlane_b32 s12, v90, 10
	v_readlane_b32 s13, v90, 11
	s_and_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v4, s19
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v6, s[12:13], s18, v6
	s_nop 1
	v_subb_co_u32_e64 v7, s[12:13], v4, v7, s[12:13]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v7, s[12:13]
	v_cndmask_b32_e64 v6, 0, v6, s[12:13]
	s_cbranch_vccz .LBB0_71
; %bb.41:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB0_43
.LBB0_42:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_43:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:16
	v_readlane_b32 s12, v90, 20
	v_readlane_b32 s13, v90, 21
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_45:                               ; %Flow812
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_47:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:24
	v_readlane_b32 s12, v90, 4
	v_readlane_b32 s13, v90, 5
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_49:                               ; %Flow811
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_51
; %bb.50:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_51:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:32
	v_readlane_b32 s12, v90, 16
	v_readlane_b32 s13, v90, 17
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_53:                               ; %Flow810
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_55:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:40
	v_readlane_b32 s12, v90, 6
	v_readlane_b32 s13, v90, 7
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_57
; %bb.56:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_57:                               ; %Flow809
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_59:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:48
	v_readlane_b32 s12, v90, 14
	v_readlane_b32 s13, v90, 15
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_61:                               ; %Flow808
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_63:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:56
	v_readlane_b32 s12, v90, 26
	v_readlane_b32 s13, v90, 27
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_65:                               ; %Flow807
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB0_67:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:64
	v_readlane_b32 s12, v90, 22
	v_readlane_b32 s13, v90, 23
	v_mov_b32_e32 v4, s19
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s18, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB0_69
; %bb.68:                               ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB0_69:                               ; %Flow
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_11
; %bb.70:                               ;   in Loop: Header=BB0_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB0_11
.LBB0_71:                               ;   in Loop: Header=BB0_13 Depth=1
	s_branch .LBB0_42
.LBB0_72:                               ; %Flow830
	v_readlane_b32 s4, v90, 28
	v_readlane_b32 s5, v90, 29
	s_or_b64 exec, exec, s[4:5]
	s_load_dwordx2 s[20:21], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
.LBB0_73:                               ; %Flow872
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB0_169
; %bb.74:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s42, s4
	s_addc_u32 s5, s43, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v90, s8, 4
	s_nop 1
	v_writelane_b32 v90, s9, 5
	s_and_b64 s[8:9], s[6:7], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB0_76
; %bb.75:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB0_76:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[6:7], 0
	s_cselect_b64 s[94:95], -1, 0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_80
; %bb.77:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_79
; %bb.78:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB0_79:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB0_81
	s_branch .LBB0_169
.LBB0_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_169
.LBB0_81:                               ; %.critedge487
	s_lshr_b32 s3, s86, 27
	s_add_i32 s3, s25, s3
	s_ashr_i32 s50, s3, 5
	s_mul_i32 s3, s29, s50
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v90, s3, 6
	s_cbranch_scc1 .LBB0_169
; %bb.82:                               ; %.lr.ph539
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[12:13], src_shared_base
	s_load_dwordx2 s[48:49], s[0:1], 0x80
	s_load_dwordx4 s[16:19], s[0:1], 0x70
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x20
	s_load_dwordx2 s[10:11], s[0:1], 0x30
	s_load_dwordx2 s[14:15], s[0:1], 0x50
	s_cselect_b32 s5, 0, 0
	s_cselect_b32 s4, s13, 0
	s_and_b32 s0, s5, 15
	s_waitcnt lgkmcnt(0)
	s_and_b32 s9, s5, -16
	s_add_u32 s9, s9, 16
	s_mov_b32 s1, 0
	s_addc_u32 s12, s4, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s51, s5, s9
	s_cselect_b32 s0, s4, s12
	s_add_u32 s4, s51, 0x2000
	s_addc_u32 s0, s0, 0
	s_and_b32 s5, s4, -16
	s_and_b32 s0, s4, 15
	s_add_u32 s5, s5, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s86, s4, s5
	s_add_i32 s0, s27, 63
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_i32 s0, s0, s1
	v_lshrrev_b32_e32 v6, 3, v0
	v_and_b32_e32 v14, 56, v2
	s_ashr_i32 s87, s0, 6
	v_mad_u64_u32 v[2:3], s[0:1], v6, s8, v[14:15]
	s_mov_b32 s0, s14
	v_ashrrev_i32_e32 v3, 31, v2
	v_writelane_b32 v90, s0, 8
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[6:7]
	v_lshlrev_b32_e32 v7, 7, v6
	v_writelane_b32 v90, s1, 9
	v_mad_u64_u32 v[2:3], s[0:1], v6, s14, v[14:15]
	v_lshlrev_b32_e32 v15, 4, v0
	s_movk_i32 s0, 0x70
	v_and_or_b32 v36, v15, s0, v7
	v_ashrrev_i32_e32 v3, 31, v2
	v_or_b32_e32 v37, 8, v36
	v_lshl_add_u64 v[18:19], v[2:3], 1, s[10:11]
	v_add_u32_e32 v2, s51, v37
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v38, v3, v2
	v_add_u32_e32 v2, s51, v36
	s_lshl_b32 s89, s29, 2
	s_lshl_b32 s78, s8, 5
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	s_cmp_gt_i32 s27, 0
	v_xor_b32_e32 v39, v3, v2
	v_add_u32_e32 v2, s86, v15
	s_cselect_b64 s[0:1], -1, 0
	v_lshrrev_b32_e32 v3, 4, v2
	v_writelane_b32 v90, s0, 10
	v_and_b32_e32 v3, 0x78, v3
	v_or_b32_e32 v41, 8, v15
	v_writelane_b32 v90, s1, 11
	s_ashr_i32 s1, s20, 31
	s_mov_b32 s0, s20
	v_xor_b32_e32 v40, v3, v2
	v_add_u32_e32 v2, s86, v41
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s0, s38, s0
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s1, s39, s1
	v_xor_b32_e32 v42, v3, v2
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v90, s0, 12
	v_bfe_u32 v4, v0, 6, 2
	v_lshrrev_b32_e32 v5, 8, v0
	v_lshlrev_b32_e32 v8, 7, v2
	v_writelane_b32 v90, s1, 13
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v43, v5, 11, v8
	v_lshl_or_b32 v44, v4, 11, v8
	v_writelane_b32 v90, s0, 14
	v_and_b32_e32 v8, 63, v0
	s_min_i32 s27, s30, 32
	v_writelane_b32 v90, s1, 15
	v_cmp_eq_u32_e64 s[0:1], 0, v8
	s_bfe_i64 s[58:59], s[48:49], 0x200000
	s_cmp_gt_i32 s30, 0
	v_writelane_b32 v90, s0, 16
	s_cselect_b64 s[60:61], -1, 0
	v_lshl_or_b32 v47, v4, 4, v2
	v_writelane_b32 v90, s1, 17
	v_lshrrev_b32_e32 v3, 2, v0
	v_readlane_b32 s8, v90, 0
	v_readlane_b32 s9, v90, 1
	s_cmp_lg_u64 s[8:9], 0
	s_cselect_b64 s[62:63], -1, 0
	s_max_i32 s0, s26, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s21, 0
	s_cselect_b64 s[64:65], -1, 0
	s_abs_i32 s92, s89
	v_cvt_f32_u32_e32 v2, s92
	v_and_b32_e32 v3, 12, v3
	v_lshl_or_b32 v48, v5, 4, v3
	v_lshlrev_b32_e32 v53, 1, v3
	v_rcp_iflag_f32_e32 v3, v2
	s_abs_i32 s93, s30
	v_cvt_f32_u32_e32 v4, s93
	v_readlane_b32 s10, v90, 2
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v3, v3
	v_rcp_iflag_f32_e32 v4, v4
	v_readlane_b32 s11, v90, 3
	v_writelane_b32 v90, s0, 18
	s_lshl_b32 s0, s27, 3
	v_cmp_gt_i32_e64 s[10:11], s0, v0
	v_mad_i64_i32 v[20:21], s[0:1], s48, v6, 0
	v_readfirstlane_b32 s1, v3
	v_mul_f32_e32 v3, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v3, v3
	s_sub_i32 s0, 0, s92
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s80, s1, s0
	s_sub_i32 s0, 0, s93
	v_readfirstlane_b32 s1, v3
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s97, s1, s0
	s_lshr_b32 s0, s97, 27
	s_mul_i32 s1, s0, s93
	s_sub_i32 s1, 32, s1
	s_max_i32 s91, s87, 1
	s_bfe_i32 s79, s29, 0x1001d
	s_ashr_i32 s96, s30, 31
	s_add_i32 s6, s0, 1
	s_sub_i32 s7, s1, s93
	s_cmp_ge_u32 s1, s93
	s_cselect_b32 s0, s6, s0
	s_cselect_b32 s1, s7, s1
	s_add_i32 s6, s0, 1
	s_cmp_ge_u32 s1, s93
	s_cselect_b32 s0, s6, s0
	s_abs_i32 s98, s33
	v_cvt_f32_u32_e32 v4, s98
	v_mov_b32_e32 v23, 0
	v_add_u32_e32 v2, 0, v7
	v_mov_b32_e32 v3, s13
	v_lshlrev_b32_e32 v22, 1, v14
	v_lshl_add_u64 v[24:25], v[2:3], 0, v[22:23]
	v_rcp_iflag_f32_e32 v3, v4
	v_add_u32_e32 v25, v2, v22
	s_xor_b32 s0, s0, s96
	s_sub_i32 s99, s0, s96
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s98
	s_ashr_i32 s25, s33, 31
	v_add3_u32 v58, 0, v22, v7
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s56, s1, s0
	s_cmp_gt_i32 s99, 0
	s_mul_i32 s0, s59, s27
	s_mul_hi_u32 s1, s48, s27
	s_cselect_b64 s[6:7], -1, 0
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s48, s27
	s_lshl_b64 s[70:71], s[0:1], 1
	v_cmp_gt_u32_e64 s[0:1], s99, v0
	v_and_b32_e32 v2, 7, v0
	v_lshlrev_b32_e32 v22, 4, v2
	v_writelane_b32 v90, s0, 20
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v90, s1, 21
	s_mul_i32 s90, s18, s16
	v_readlane_b32 s0, v90, 4
	v_readlane_b32 s1, v90, 5
	v_writelane_b32 v90, s6, 22
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_mov_b64 s[68:69], 0x80
	v_writelane_b32 v90, s7, 23
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v90, s0, 24
	s_mov_b64 s[52:53], 0
	v_cmp_gt_u32_e64 s[4:5], 32, v6
	v_mul_lo_u32 v45, s30, v0
	v_add_u32_e32 v46, -1, v1
	s_mul_i32 s90, s90, s24
	v_lshl_add_u32 v49, v47, 1, 0
	v_or_b32_e32 v50, 1, v48
	v_or_b32_e32 v51, 2, v48
	v_or_b32_e32 v52, 3, v48
	v_or_b32_e32 v54, 32, v53
	v_or_b32_e32 v55, 64, v53
	v_or_b32_e32 v56, 0x60, v53
	v_add_u32_e32 v57, 8, v14
	v_lshl_add_u64 v[26:27], v[18:19], 0, s[68:69]
	v_lshl_add_u32 v59, v0, 1, 0
	v_or_b32_e32 v60, 1, v14
	v_lshl_add_u64 v[28:29], v[20:21], 1, v[22:23]
	s_sub_i32 s57, 0, s33
	v_bfrev_b32_e32 v61, 64
	v_and_b32_e32 v62, 0x100, v2
	s_movk_i32 s49, 0x7fff
	v_writelane_b32 v90, s1, 25
                                        ; implicit-def: $vgpr4_vgpr5
	v_writelane_b32 v90, s80, 26
	s_branch .LBB0_85
.LBB0_83:                               ; %Flow833
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_add_i32 s2, s2, s31
	v_readlane_b32 s0, v90, 6
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB0_84:                               ; %Flow866
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[74:75]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[52:53], s[0:1], s[52:53]
	s_andn2_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB0_169
.LBB0_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_91 Depth 2
                                        ;     Child Loop BB0_107 Depth 2
                                        ;     Child Loop BB0_119 Depth 2
                                        ;       Child Loop BB0_123 Depth 3
                                        ;         Child Loop BB0_144 Depth 4
                                        ;         Child Loop BB0_153 Depth 4
                                        ;         Child Loop BB0_159 Depth 4
                                        ;     Child Loop BB0_165 Depth 2
	s_ashr_i32 s0, s2, 31
	s_xor_b32 s7, s0, s79
	s_abs_i32 s0, s2
	s_mul_hi_u32 s1, s0, s80
	s_mul_i32 s6, s1, s92
	s_sub_i32 s0, s0, s6
	s_add_i32 s6, s1, 1
	s_sub_i32 s8, s0, s92
	s_cmp_ge_u32 s0, s92
	s_cselect_b32 s1, s6, s1
	s_cselect_b32 s0, s8, s0
	s_add_i32 s6, s1, 1
	s_cmp_ge_u32 s0, s92
	s_cselect_b32 s0, s6, s1
	s_xor_b32 s72, s0, s7
	s_sub_i32 s73, s72, s7
	s_lshl_b32 s0, s73, 2
	s_sub_i32 s1, s50, s0
	s_min_i32 s1, s1, 4
	s_abs_i32 s6, s1
	v_cvt_f32_u32_e32 v6, s6
	s_mul_i32 s73, s73, s89
	s_sub_i32 s14, 0, s6
	s_sub_i32 s8, s2, s73
	v_rcp_iflag_f32_e32 v6, v6
	s_xor_b32 s9, s8, s1
	s_ashr_i32 s42, s9, 31
	s_abs_i32 s9, s8
	v_mul_f32_e32 v6, 0x4f7ffffe, v6
	v_cvt_u32_f32_e32 v6, v6
	s_nop 0
	v_readfirstlane_b32 s15, v6
	s_mul_i32 s14, s14, s15
	s_mul_hi_u32 s14, s15, s14
	s_add_i32 s15, s15, s14
	s_mul_hi_u32 s14, s9, s15
	s_mul_i32 s15, s14, s6
	s_sub_i32 s9, s9, s15
	s_add_i32 s15, s14, 1
	s_sub_i32 s16, s9, s6
	s_cmp_ge_u32 s9, s6
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s9, s16, s9
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s9, s6
	s_cselect_b32 s6, s15, s14
	s_xor_b32 s43, s6, s42
	s_sub_i32 s6, s43, s42
	s_mul_i32 s66, s6, s1
	s_sub_i32 s9, s8, s66
	s_add_i32 s9, s9, s0
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_85 Depth=1
	s_mul_i32 s14, s78, s9
	s_ashr_i32 s15, s14, 31
	v_lshl_add_u64 v[2:3], s[14:15], 1, v[16:17]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB0_87:                               ; %_ZN17hk_gemm_rs_mi300x10load_issueITkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEELi512ELi2ELb0ETkNS2_2gl3allENS1_2glIS5_Lin1ELin1ELin1ELin1EJEEETkNS2_5coord4tileENS1_5coordIS8_EEEEvP15HIP_vector_typeIfLj4EERKT3_RKT4_.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s67, s6, 6
	v_readlane_b32 s0, v90, 8
	v_readlane_b32 s1, v90, 9
	s_mul_i32 s0, s67, s0
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[6:7], s[0:1], 1, v[18:19]
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[16:17], s[4:5]
	s_cbranch_execz .LBB0_89
; %bb.88:                               ;   in Loop: Header=BB0_85 Depth=1
	;;#ASMSTART
	ds_write_b64 v39, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v38, v[4:5]

	;;#ASMEND
.LBB0_89:                               ; %_ZN17hk_gemm_rs_mi300x11load_commitILi512ETkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEEEEvRT0_PK15HIP_vector_typeIfLj4EE.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v40, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v42, v[12:13]

	;;#ASMEND
	v_readlane_b32 s12, v90, 10
	v_readlane_b32 s13, v90, 11
	s_andn2_b64 vcc, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB0_101
; %bb.90:                               ; %.lr.ph522.preheader
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_lshl_add_u64 v[30:31], s[0:1], 1, v[26:27]
	s_lshl_b32 s0, s72, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s73
	s_sub_i32 s0, s0, s66
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	s_mul_i32 s0, s78, s0
	v_mov_b32_e32 v6, 0
	s_add_i32 s0, s0, 64
	s_mov_b32 s14, 0
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v9, v6
.LBB0_91:                               ; %.lr.ph522
                                        ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s8, s14, 1
	s_cmp_lt_i32 s8, s87
	s_cselect_b64 s[16:17], -1, 0
	s_cmp_ge_i32 s8, s87
	s_cbranch_scc1 .LBB0_95
; %bb.92:                               ;   in Loop: Header=BB0_91 Depth=2
	s_and_saveexec_b64 s[18:19], s[4:5]
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_91 Depth=2
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[16:17]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB0_94:                               ; %_ZN17hk_gemm_rs_mi300x10load_issueITkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEELi512ELi2ELb0ETkNS2_2gl3allENS1_2glIS5_Lin1ELin1ELin1ELin1EJEEETkNS2_5coord4tileENS1_5coordIS8_EEEEvP15HIP_vector_typeIfLj4EERKT3_RKT4_.exit289
                                        ;   in Loop: Header=BB0_91 Depth=2
	s_or_b64 exec, exec, s[18:19]
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[30:31], off

	;;#ASMEND
.LBB0_95:                               ;   in Loop: Header=BB0_91 Depth=2
	s_and_b32 s1, s14, 1
	s_lshl_b32 s14, s1, 12
	s_add_i32 s14, s51, s14
	v_add_u32_e32 v22, s14, v43
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s86, s1
	v_add_u32_e32 v32, v53, v22
	v_add_u32_e32 v63, s1, v44
	v_lshrrev_b32_e32 v33, 4, v32
	v_add_u32_e32 v34, v54, v22
	v_and_b32_e32 v33, 0x78, v33
	v_lshrrev_b32_e32 v35, 4, v34
	v_add_u32_e32 v64, v53, v63
	v_xor_b32_e32 v32, v33, v32
	v_and_b32_e32 v35, 0x78, v35
	v_lshrrev_b32_e32 v65, 4, v64
	v_add_u32_e32 v66, v54, v63
	;;#ASMSTART
	ds_read_b64 v[32:33], v32 offset:0

	;;#ASMEND
	v_xor_b32_e32 v34, v35, v34
	v_and_b32_e32 v65, 0x78, v65
	v_lshrrev_b32_e32 v67, 4, v66
	;;#ASMSTART
	ds_read_b64 v[34:35], v34 offset:0

	;;#ASMEND
	v_xor_b32_e32 v64, v65, v64
	v_and_b32_e32 v67, 0x78, v67
	;;#ASMSTART
	ds_read_b64 v[64:65], v64 offset:0

	;;#ASMEND
	v_xor_b32_e32 v66, v67, v66
	;;#ASMSTART
	ds_read_b64 v[66:67], v66 offset:0

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_andn2_b64 vcc, exec, s[16:17]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[64:65], v[6:9]
	v_add_u32_e32 v32, v55, v22
	v_lshrrev_b32_e32 v33, 4, v32
	v_add_u32_e32 v22, v56, v22
	;;#ASMSTART
	;;#ASMEND
	v_and_b32_e32 v33, 0x78, v33
	v_mfma_f32_16x16x16_bf16 v[6:9], v[34:35], v[66:67], v[6:9]
	v_lshrrev_b32_e32 v34, 4, v22
	v_xor_b32_e32 v32, v33, v32
	v_and_b32_e32 v34, 0x78, v34
	;;#ASMSTART
	ds_read_b64 v[32:33], v32 offset:0

	;;#ASMEND
	v_xor_b32_e32 v22, v34, v22
	;;#ASMSTART
	ds_read_b64 v[34:35], v22 offset:0

	;;#ASMEND
	v_add_u32_e32 v22, v55, v63
	v_lshrrev_b32_e32 v64, 4, v22
	v_and_b32_e32 v64, 0x78, v64
	v_xor_b32_e32 v22, v64, v22
	;;#ASMSTART
	ds_read_b64 v[64:65], v22 offset:0

	;;#ASMEND
	v_add_u32_e32 v22, v56, v63
	v_lshrrev_b32_e32 v63, 4, v22
	v_and_b32_e32 v63, 0x78, v63
	v_xor_b32_e32 v22, v63, v22
	;;#ASMSTART
	ds_read_b64 v[66:67], v22 offset:0

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[64:65], v[6:9]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[6:9], v[34:35], v[66:67], v[6:9]
	s_cbranch_vccnz .LBB0_99
; %bb.96:                               ;   in Loop: Header=BB0_91 Depth=2
	s_and_b32 s1, s8, 1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[16:17], s[4:5]
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_91 Depth=2
	s_lshl_b32 s14, s1, 12
	s_add_i32 s14, s51, s14
	v_add_u32_e32 v22, s14, v37
	v_lshrrev_b32_e32 v32, 4, v22
	v_and_b32_e32 v32, 0x78, v32
	v_xor_b32_e32 v22, v32, v22
	v_add_u32_e32 v32, s14, v36
	v_lshrrev_b32_e32 v33, 4, v32
	v_and_b32_e32 v33, 0x78, v33
	v_xor_b32_e32 v32, v33, v32
	;;#ASMSTART
	ds_write_b64 v32, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v22, v[4:5]

	;;#ASMEND
.LBB0_98:                               ; %_ZN17hk_gemm_rs_mi300x11load_commitILi512ETkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEEEEvRT0_PK15HIP_vector_typeIfLj4EE.exit303
                                        ;   in Loop: Header=BB0_91 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s86, s1
	v_add_u32_e32 v22, s1, v15
	v_lshrrev_b32_e32 v32, 4, v22
	v_and_b32_e32 v32, 0x78, v32
	v_xor_b32_e32 v22, v32, v22
	v_add_u32_e32 v32, s1, v41
	v_lshrrev_b32_e32 v33, 4, v32
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v33, 0x78, v33
	;;#ASMSTART
	ds_write_b64 v22, v[10:11]

	;;#ASMEND
	v_xor_b32_e32 v32, v33, v32
	;;#ASMSTART
	ds_write_b64 v32, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_99:                               ;   in Loop: Header=BB0_91 Depth=2
	s_add_i32 s0, s0, 64
	s_cmp_eq_u32 s91, s8
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[68:69]
	s_barrier
	s_cbranch_scc1 .LBB0_102
; %bb.100:                              ;   in Loop: Header=BB0_91 Depth=2
	s_mov_b32 s14, s8
	s_branch .LBB0_91
.LBB0_101:                              ;   in Loop: Header=BB0_85 Depth=1
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, v9
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v6, v9
.LBB0_102:                              ; %Flow862
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v90, 20
	v_readlane_b32 s13, v90, 21
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB0_111
; %bb.103:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v90, 14
	v_readlane_b32 s13, v90, 15
	s_and_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB0_111
; %bb.104:                              ;   in Loop: Header=BB0_85 Depth=1
	v_lshl_add_u32 v10, s9, 5, v45
	v_sub_u32_e32 v12, 0, v10
	v_max_i32_e32 v12, v10, v12
	v_mul_hi_u32 v13, v12, s56
	v_mul_lo_u32 v22, v13, s98
	v_sub_u32_e32 v12, v12, v22
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s98, v12
	v_ashrrev_i32_e32 v11, 31, v10
	v_xor_b32_e32 v11, s25, v11
	v_cndmask_b32_e32 v13, v13, v22, vcc
	v_subrev_u32_e32 v22, s98, v12
	v_cndmask_b32_e32 v12, v12, v22, vcc
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s98, v12
	v_readlane_b32 s12, v90, 12
	v_readlane_b32 s13, v90, 13
	v_cndmask_b32_e32 v12, v13, v22, vcc
	v_xor_b32_e32 v12, v12, v11
	v_sub_u32_e32 v11, v12, v11
	v_mul_lo_u32 v12, v11, s33
	v_sub_u32_e32 v10, v10, v12
	v_sub_u32_e32 v13, 0, v10
	v_ashrrev_i32_e32 v12, 31, v10
	v_max_i32_e32 v10, v10, v13
	v_mul_hi_u32 v13, v10, s97
	v_mul_lo_u32 v22, v13, s93
	v_sub_u32_e32 v10, v10, v22
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s93, v10
	v_xor_b32_e32 v12, s96, v12
	s_nop 0
	v_cndmask_b32_e32 v13, v13, v22, vcc
	v_subrev_u32_e32 v22, s93, v10
	v_cndmask_b32_e32 v10, v10, v22, vcc
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s93, v10
	s_nop 1
	v_cndmask_b32_e32 v10, v13, v22, vcc
	v_xor_b32_e32 v10, v10, v12
	v_sub_u32_e32 v10, v10, v12
	v_mad_u64_u32 v[10:11], s[14:15], v11, s28, v[10:11]
	v_mul_lo_u32 v10, v10, s29
	v_add_u32_e32 v10, s6, v10
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[12:13]
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v12, v46
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_111
; %bb.105:                              ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[74:75], 0
                                        ; implicit-def: $sgpr18_sgpr19
                                        ; implicit-def: $sgpr20_sgpr21
	s_branch .LBB0_107
.LBB0_106:                              ; %Flow855
                                        ;   in Loop: Header=BB0_107 Depth=2
	s_and_b64 s[14:15], exec, s[20:21]
	s_or_b64 s[16:17], s[14:15], s[16:17]
	s_andn2_b64 s[14:15], s[18:19], exec
	s_and_b64 s[18:19], s[76:77], exec
	s_or_b64 s[18:19], s[14:15], s[18:19]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_109
.LBB0_107:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s74, s74, 1
	s_addc_u32 s75, s75, 0
	v_mov_b64_e32 v[12:13], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[74:75], v[12:13]
	s_mov_b64 s[76:77], -1
	s_or_b64 s[20:21], s[20:21], exec
	s_cbranch_vccnz .LBB0_106
; %bb.108:                              ;   in Loop: Header=BB0_107 Depth=2
	s_sleep 4
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_andn2_b64 s[14:15], s[20:21], exec
	s_mov_b64 s[76:77], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v12, v46
	s_and_b64 s[20:21], vcc, exec
	s_or_b64 s[20:21], s[14:15], s[20:21]
	s_branch .LBB0_106
.LBB0_109:                              ; %loop.exit.guard805
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[14:15], s[18:19]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB0_111
; %bb.110:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s16, v90, 0
	v_readlane_b32 s18, v90, 2
	v_readlane_b32 s19, v90, 3
	v_readlane_b32 s17, v90, 1
	s_nop 0
	v_mov_b64_e32 v[10:11], s[18:19]
	flat_atomic_or v[10:11], v61
.LBB0_111:                              ; %.critedge488
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[94:95]
	s_mov_b64 s[16:17], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_115
; %bb.112:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v90, 16
	v_mov_b32_e32 v10, 0
	v_readlane_b32 s13, v90, 17
	s_and_saveexec_b64 s[16:17], s[12:13]
	s_cbranch_execz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s44, v90, 0
	v_readlane_b32 s46, v90, 2
	v_readlane_b32 s47, v90, 3
	v_readlane_b32 s45, v90, 1
	s_nop 0
	v_mov_b64_e32 v[10:11], s[46:47]
	flat_load_dword v10, v[10:11] sc1
.LBB0_114:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit306
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v10, v62, v10
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v10, 0x2000000, v10
	v_cmp_eq_u32_e64 s[16:17], 0, v10
.LBB0_115:                              ; %Flow865
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_and_saveexec_b64 s[74:75], s[16:17]
	s_cbranch_execz .LBB0_84
; %bb.116:                              ; %.critedge502
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s0, v90, 22
	v_readlane_b32 s1, v90, 23
	s_mov_b32 s13, s79
	s_mov_b32 s12, s78
	s_mov_b32 s3, s50
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_160
; %bb.117:                              ; %.lr.ph532
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_sub_i32 s18, s26, s67
	s_min_i32 s8, s18, 64
	s_abs_i32 s14, s8
	v_cvt_f32_u32_e32 v12, s14
	s_ashr_i32 s50, s8, 31
	s_sub_i32 s20, 0, s14
	s_lshl_b32 s21, s42, 6
	v_rcp_iflag_f32_e32 v12, v12
	v_or_b32_e32 v10, s67, v47
	v_readlane_b32 s0, v90, 18
	v_cmp_gt_i32_e32 vcc, s26, v10
	v_mul_f32_e32 v12, 0x4f7ffffe, v12
	v_cvt_u32_f32_e32 v12, v12
	v_mov_b32_e32 v11, s0
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_readlane_b32 s44, v90, 0
	v_mul_lo_u32 v13, s20, v12
	s_lshl_b32 s20, s50, 7
	v_subrev_u32_e32 v64, s20, v59
	s_lshl_b32 s20, s43, 6
	s_sub_i32 s42, s20, s21
	s_lshl_b32 s20, s72, 2
	s_add_i32 s20, s2, s20
	s_sub_i32 s20, s20, s73
	s_sub_i32 s20, s20, s66
	s_lshl_b32 s21, s7, 2
	s_sub_i32 s20, s20, s21
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s45, v90, 1
	s_mul_i32 s15, s8, s27
	v_mul_hi_u32 v13, v12, v13
	s_lshl_b32 s20, s20, 5
	s_lshl_b32 s9, s9, 5
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[44:45]
	v_cmp_gt_i32_e64 s[0:1], s15, v0
	v_cmp_lt_i32_e64 s[76:77], s8, v57
	v_cmp_ge_i32_e64 s[16:17], s8, v57
	v_cmp_gt_i32_e64 s[18:19], s18, v14
	s_mov_b32 s54, 0
	v_add_u32_e32 v63, v12, v13
	s_lshl_b32 s55, s8, 1
	s_sub_i32 s88, 0, s8
	s_add_i32 s43, s90, s20
	v_readlane_b32 s46, v90, 2
	v_readlane_b32 s47, v90, 3
	s_branch .LBB0_119
.LBB0_118:                              ; %._crit_edge530
                                        ;   in Loop: Header=BB0_119 Depth=2
	s_add_i32 s54, s54, 1
	s_add_i32 s43, s43, s30
	s_cmp_eq_u32 s54, s99
	s_cbranch_scc1 .LBB0_160
.LBB0_119:                              ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_123 Depth 3
                                        ;         Child Loop BB0_144 Depth 4
                                        ;         Child Loop BB0_153 Depth 4
                                        ;         Child Loop BB0_159 Depth 4
	s_andn2_b64 vcc, exec, s[60:61]
	s_cbranch_vccnz .LBB0_118
; %bb.120:                              ; %.lr.ph529.preheader
                                        ;   in Loop: Header=BB0_119 Depth=2
	v_mov_b64_e32 v[12:13], s[36:37]
	flat_load_dwordx4 v[30:33], v[12:13]
	flat_load_dwordx4 v[66:69], v[12:13] offset:16
	flat_load_dwordx4 v[70:73], v[12:13] offset:32
	flat_load_dwordx4 v[74:77], v[12:13] offset:48
	s_nop 0
	flat_load_dwordx2 v[12:13], v[12:13] offset:64
	s_mul_i32 s44, s54, s30
	s_add_i32 s20, s44, s9
	s_abs_i32 s46, s20
	s_mul_hi_u32 s47, s46, s56
	s_mul_i32 s78, s47, s98
	s_ashr_i32 s21, s20, 31
	s_sub_i32 s46, s46, s78
	s_xor_b32 s21, s21, s25
	s_add_i32 s79, s47, 1
	s_sub_i32 s78, s46, s98
	s_cmp_ge_u32 s46, s98
	s_cselect_b32 s47, s79, s47
	s_cselect_b32 s46, s78, s46
	s_add_i32 s78, s47, 1
	s_cmp_ge_u32 s46, s98
	s_cselect_b32 s46, s78, s47
	s_xor_b32 s46, s46, s21
	s_sub_i32 s46, s46, s21
	s_mul_i32 s47, s46, s33
	s_sub_i32 s78, s20, s47
	s_cmp_eq_u32 s46, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 1
	v_mov_b32_e32 v22, s23
	s_mov_b32 s45, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_cndmask_b32_e32 v33, 0, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 2
	v_sub_co_u32_e64 v30, s[20:21], s22, v30
	v_cndmask_b32_e32 v32, v32, v66, vcc
	s_nop 0
	v_subb_co_u32_e64 v31, s[20:21], v22, v31, s[20:21]
	v_cndmask_b32_e32 v22, v33, v67, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 3
	v_cndmask_b32_e32 v32, v32, v68, vcc
	v_cndmask_b32_e32 v22, v22, v69, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 4
	v_cndmask_b32_e32 v22, v22, v71, vcc
	v_cndmask_b32_e32 v32, v32, v70, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 5
	v_cndmask_b32_e32 v32, v32, v72, vcc
	v_cndmask_b32_e32 v22, v22, v73, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 6
	v_cndmask_b32_e32 v22, v22, v75, vcc
	v_cndmask_b32_e32 v32, v32, v74, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 7
	v_cndmask_b32_e32 v32, v32, v76, vcc
	v_cndmask_b32_e32 v22, v22, v77, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s21, s78
	s_mul_hi_u32 s46, s21, s97
	s_mul_i32 s46, s46, s93
	s_sub_i32 s21, s21, s46
	s_ashr_i32 s20, s78, 31
	s_sub_i32 s46, s21, s93
	s_cmp_ge_u32 s21, s93
	s_cselect_b32 s21, s46, s21
	s_sub_i32 s46, s21, s93
	s_cmp_ge_u32 s21, s93
	s_cselect_b32 s21, s46, s21
	s_add_i32 s46, s78, s90
	s_add_i32 s78, s20, s43
	s_xor_b32 s21, s21, s20
	s_sub_i32 s47, s78, s47
	s_sub_i32 s20, s20, s21
	v_cndmask_b32_e32 v13, v22, v13, vcc
	v_cndmask_b32_e32 v12, v32, v12, vcc
	s_sub_i32 s21, s47, s21
	s_add_i32 s20, s46, s20
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[12:13]
	v_cmp_ne_u64_e32 vcc, 0, v[12:13]
	s_mul_i32 s21, s48, s21
	s_mul_i32 s46, s20, s48
	v_cndmask_b32_e32 v13, 0, v31, vcc
	v_cndmask_b32_e32 v12, 0, v30, vcc
	s_add_i32 s20, s42, s21
	s_add_i32 s46, s46, s67
	v_lshl_add_u64 v[30:31], v[12:13], 0, v[28:29]
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s47, s46, 31
	v_lshl_add_u64 v[12:13], s[46:47], 1, v[12:13]
	v_lshl_add_u64 v[30:31], s[20:21], 1, v[30:31]
	s_branch .LBB0_123
.LBB0_121:                              ; %Flow847
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
.LBB0_122:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_add_i32 s45, s45, s27
	s_cmp_ge_i32 s45, s30
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[70:71]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_118
.LBB0_123:                              ; %.lr.ph529
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_119 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB0_144 Depth 4
                                        ;         Child Loop BB0_153 Depth 4
                                        ;         Child Loop BB0_159 Depth 4
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB0_125
; %bb.124:                              ;   in Loop: Header=BB0_123 Depth=3
	flat_load_ushort v22, v[10:11]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v22, 16, v22
	s_branch .LBB0_126
.LBB0_125:                              ;   in Loop: Header=BB0_123 Depth=3
	v_mov_b32_e32 v22, 0
.LBB0_126:                              ;   in Loop: Header=BB0_123 Depth=3
	s_add_i32 s46, s45, s44
	s_add_i32 s47, s46, s27
	v_cmp_le_i32_e32 vcc, s46, v48
	v_cmp_gt_i32_e64 s[20:21], s47, v48
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_123 Depth=3
	v_add_f32_e32 v32, v22, v6
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s49
	v_subrev_u32_e32 v33, s46, v48
	v_lshl_add_u32 v33, v33, 7, v49
	ds_write_b16_d16_hi v33, v32
.LBB0_128:                              ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v50
	v_cmp_gt_i32_e64 s[20:21], s47, v50
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_123 Depth=3
	v_add_f32_e32 v32, v22, v7
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s49
	v_subrev_u32_e32 v33, s46, v50
	v_lshl_add_u32 v33, v33, 7, v49
	ds_write_b16_d16_hi v33, v32
.LBB0_130:                              ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v51
	v_cmp_gt_i32_e64 s[20:21], s47, v51
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_132
; %bb.131:                              ;   in Loop: Header=BB0_123 Depth=3
	v_add_f32_e32 v32, v22, v8
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s49
	v_subrev_u32_e32 v33, s46, v51
	v_lshl_add_u32 v33, v33, 7, v49
	ds_write_b16_d16_hi v33, v32
.LBB0_132:                              ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v52
	v_cmp_gt_i32_e64 s[20:21], s47, v52
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_123 Depth=3
	v_add_f32_e32 v22, v22, v9
	v_bfe_u32 v32, v22, 16, 1
	v_add3_u32 v22, v22, v32, s49
	v_subrev_u32_e32 v32, s46, v52
	v_lshl_add_u32 v32, v32, 7, v49
	ds_write_b16_d16_hi v32, v22
.LBB0_134:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi16ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_mul_i32 s20, s59, s45
	s_mul_hi_u32 s21, s58, s45
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s58, s45
	s_andn2_b64 vcc, exec, s[64:65]
	v_lshl_add_u64 v[32:33], s[20:21], 1, v[12:13]
	s_mov_b64 s[20:21], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_156
; %bb.135:                              ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[80:81], -1
	s_and_saveexec_b64 s[78:79], s[10:11]
	s_cbranch_execz .LBB0_141
; %bb.136:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[82:83], s[76:77]
	s_and_saveexec_b64 s[80:81], s[16:17]
; %bb.137:                              ;   in Loop: Header=BB0_123 Depth=3
	v_lshlrev_b32_e32 v22, 1, v20
	v_lshlrev_b32_e32 v34, 1, v14
	v_add3_u32 v22, v32, v22, v34
	v_or_b32_e32 v22, v24, v22
	v_and_b32_e32 v22, 15, v22
	v_cmp_eq_u32_e32 vcc, 0, v22
	s_andn2_b64 s[46:47], s[76:77], exec
	s_and_b64 s[82:83], vcc, exec
	s_or_b64 s[82:83], s[46:47], s[82:83]
; %bb.138:                              ; %Flow843
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[80:81]
	s_mov_b64 s[80:81], 0
	s_and_saveexec_b64 s[84:85], s[82:83]
; %bb.139:                              ; %.critedge.i
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[80:81], exec
; %bb.140:                              ; %Flow844
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[80:81], s[80:81], exec
.LBB0_141:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[78:79]
	v_cndmask_b32_e64 v22, 0, 1, s[80:81]
	s_nop 0
	v_readfirstlane_b32 s46, v22
	s_bitcmp1_b32 s46, 0
	s_cselect_b64 s[46:47], -1, 0
	s_and_b64 vcc, exec, s[46:47]
	s_cbranch_vccnz .LBB0_146
; %bb.142:                              ;   in Loop: Header=BB0_123 Depth=3
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_145
; %bb.143:                              ; %.lr.ph.i311.preheader
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v34, v64
	v_mov_b32_e32 v22, v0
.LBB0_144:                              ; %.lr.ph.i311
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_119 Depth=2
                                        ;       Parent Loop BB0_123 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v35, v22, v63
	v_mul_lo_u32 v65, v35, s14
	v_sub_u32_e32 v65, v22, v65
	v_add_u32_e32 v66, 1, v35
	v_subrev_u32_e32 v67, s14, v65
	v_cmp_le_u32_e32 vcc, s14, v65
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v66, vcc
	v_cndmask_b32_e32 v65, v65, v67, vcc
	v_add_u32_e32 v66, 1, v35
	v_cmp_le_u32_e32 vcc, s14, v65
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v66, vcc
	v_xor_b32_e32 v35, s50, v35
	v_subrev_u32_e32 v65, s50, v35
	v_mad_u64_u32 v[66:67], s[46:47], s88, v65, v[22:23]
	v_lshlrev_b32_e32 v35, 7, v35
	v_mul_lo_u32 v67, s55, v65
	v_sub_u32_e32 v35, v35, v67
	v_add_u32_e32 v35, v34, v35
	ds_read_u16 v35, v35
	v_mad_i64_i32 v[68:69], s[46:47], s58, v65, 0
	v_add_u32_e32 v22, 0x200, v22
	v_mov_b32_e32 v67, v23
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[32:33]
	v_cmp_le_i32_e32 vcc, s15, v22
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[68:69]
	v_add_u32_e32 v34, 0x400, v34
	s_or_b64 s[78:79], vcc, s[78:79]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[66:67], v35
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_144
.LBB0_145:                              ; %Flow835
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB0_146:                              ; %Flow841
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_155
; %bb.147:                              ;   in Loop: Header=BB0_123 Depth=3
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB0_154
; %bb.148:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_and_saveexec_b64 s[46:47], s[16:17]
	s_xor_b64 s[78:79], exec, s[46:47]
	s_cbranch_execz .LBB0_150
; %bb.149:                              ;   in Loop: Header=BB0_123 Depth=3
	ds_read_b128 v[66:69], v25
	v_lshl_add_u64 v[34:35], v[20:21], 1, v[32:33]
	v_lshlrev_b32_e32 v22, 1, v14
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[22:23]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[34:35], v[66:69]
.LBB0_150:                              ; %Flow838
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_andn2_saveexec_b64 s[46:47], s[78:79]
	s_cbranch_execz .LBB0_154
; %bb.151:                              ; %.preheader.i
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_154
; %bb.152:                              ; %.lr.ph.i313
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b32 s46, 0
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v22, v58
	v_mov_b64_e32 v[34:35], v[30:31]
.LBB0_153:                              ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_119 Depth=2
                                        ;       Parent Loop BB0_123 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v65, v22
	s_add_i32 s47, s46, 1
	v_add_u32_e32 v66, s46, v60
	s_cmp_gt_u32 s46, 6
	v_cmp_le_u32_e32 vcc, s8, v66
	s_cselect_b64 s[80:81], -1, 0
	s_or_b64 s[80:81], s[80:81], vcc
	s_and_b64 s[80:81], exec, s[80:81]
	v_add_u32_e32 v22, 2, v22
	s_mov_b32 s46, s47
	s_waitcnt lgkmcnt(0)
	flat_store_short v[34:35], v65
	s_or_b64 s[78:79], s[80:81], s[78:79]
	v_lshl_add_u64 v[34:35], v[34:35], 0, 2
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_153
.LBB0_154:                              ; %Flow840
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_or_b64 exec, exec, s[20:21]
.LBB0_155:                              ; %Flow842
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[20:21], 0
.LBB0_156:                              ; %Flow848
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_122
; %bb.157:                              ;   in Loop: Header=BB0_123 Depth=3
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_121
; %bb.158:                              ; %.lr.ph.i317.preheader
                                        ;   in Loop: Header=BB0_123 Depth=3
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v34, v64
	v_mov_b32_e32 v22, v0
.LBB0_159:                              ; %.lr.ph.i317
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_119 Depth=2
                                        ;       Parent Loop BB0_123 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v35, v22, v63
	v_mul_lo_u32 v65, v35, s14
	v_sub_u32_e32 v65, v22, v65
	v_add_u32_e32 v66, 1, v35
	v_subrev_u32_e32 v67, s14, v65
	v_cmp_le_u32_e32 vcc, s14, v65
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v66, vcc
	v_cndmask_b32_e32 v65, v65, v67, vcc
	v_add_u32_e32 v66, 1, v35
	v_cmp_le_u32_e32 vcc, s14, v65
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v66, vcc
	v_xor_b32_e32 v35, s50, v35
	v_subrev_u32_e32 v65, s50, v35
	v_mad_u64_u32 v[66:67], s[46:47], s88, v65, v[22:23]
	v_lshlrev_b32_e32 v35, 7, v35
	v_mul_lo_u32 v67, s55, v65
	v_sub_u32_e32 v35, v35, v67
	v_add_u32_e32 v35, v34, v35
	ds_read_u16 v35, v35
	v_mad_i64_i32 v[68:69], s[46:47], s58, v65, 0
	v_add_u32_e32 v22, 0x200, v22
	v_mov_b32_e32 v67, v23
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[32:33]
	v_cmp_le_i32_e32 vcc, s15, v22
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[68:69]
	v_add_u32_e32 v34, 0x400, v34
	s_or_b64 s[78:79], vcc, s[78:79]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[66:67], v35
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_159
	s_branch .LBB0_121
.LBB0_160:                              ; %._crit_edge533
                                        ;   in Loop: Header=BB0_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s8, v90, 4
	v_readlane_b32 s9, v90, 5
	s_and_b64 s[8:9], s[0:1], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB0_162
; %bb.161:                              ;   in Loop: Header=BB0_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB0_162:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[16:17], exec
	v_readlane_b32 s0, v90, 24
	v_readlane_b32 s1, v90, 25
	s_and_b64 s[0:1], s[16:17], s[0:1]
	s_mov_b32 s50, s3
	s_mov_b32 s78, s12
	s_mov_b32 s79, s13
	v_readlane_b32 s80, v90, 26
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB0_83
; %bb.163:                              ; %.lr.ph535
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_lshl_b32 s0, s72, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s73
	s_sub_i32 s0, s0, s66
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s7, s0, 5
	s_mov_b32 s8, s99
	s_branch .LBB0_165
.LBB0_164:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB0_165 Depth=2
	s_add_i32 s8, s8, -1
	s_add_i32 s7, s7, s30
	s_cmp_lg_u32 s8, 0
	s_cbranch_scc0 .LBB0_83
.LBB0_165:                              ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[34:35], s[40:41]
	flat_load_dwordx4 v[6:9], v[34:35]
	flat_load_dwordx4 v[10:13], v[34:35] offset:16
	flat_load_dwordx4 v[30:33], v[34:35] offset:32
	flat_load_dwordx4 v[64:67], v[34:35] offset:48
	s_abs_i32 s1, s7
	flat_load_dwordx2 v[34:35], v[34:35] offset:64
	s_mul_hi_u32 s14, s1, s56
	s_mul_i32 s15, s14, s98
	s_ashr_i32 s0, s7, 31
	s_sub_i32 s1, s1, s15
	s_xor_b32 s0, s0, s25
	s_add_i32 s18, s14, 1
	s_sub_i32 s15, s1, s98
	s_cmp_ge_u32 s1, s98
	s_cselect_b32 s14, s18, s14
	s_cselect_b32 s1, s15, s1
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s1, s98
	s_cselect_b32 s1, s15, s14
	s_xor_b32 s1, s1, s0
	s_sub_i32 s14, s1, s0
	s_mul_i32 s1, s57, s14
	s_add_i32 s1, s7, s1
	s_mul_i32 s0, s14, s33
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s0, s1, s0
	s_add_i32 s0, s7, s0
	s_xor_b32 s0, s0, s1
	s_xor_b32 s15, s1, s96
	s_mul_hi_u32 s1, s0, s97
	s_mul_i32 s18, s1, s93
	s_sub_i32 s0, s0, s18
	s_add_i32 s19, s1, 1
	s_sub_i32 s18, s0, s93
	s_cmp_ge_u32 s0, s93
	s_cselect_b32 s1, s19, s1
	s_cselect_b32 s0, s18, s0
	s_add_i32 s18, s1, 1
	s_cmp_ge_u32 s0, s93
	s_cselect_b32 s0, s18, s1
	s_xor_b32 s0, s0, s15
	s_mul_i32 s9, s28, s24
	s_sub_i32 s0, s0, s15
	s_add_i32 s0, s0, s9
	s_mul_i32 s0, s0, s29
	s_add_i32 s0, s0, s6
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	s_addc_u32 s1, s39, s1
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmp_eq_u32 s14, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 1
	v_mov_b32_e32 v22, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v8, 0, v8, vcc
	v_cndmask_b32_e32 v9, 0, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 2
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e32 v8, v8, v10, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 3
	v_cndmask_b32_e32 v8, v8, v12, vcc
	v_cndmask_b32_e32 v9, v9, v13, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 4
	v_cndmask_b32_e32 v9, v9, v31, vcc
	v_cndmask_b32_e32 v8, v8, v30, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 5
	v_cndmask_b32_e32 v8, v8, v32, vcc
	v_cndmask_b32_e32 v9, v9, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 6
	v_cndmask_b32_e32 v9, v9, v65, vcc
	v_cndmask_b32_e32 v8, v8, v64, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 7
	v_sub_co_u32_e64 v6, s[0:1], s0, v6
	v_cndmask_b32_e32 v8, v8, v66, vcc
	v_cndmask_b32_e32 v9, v9, v67, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v7, s[0:1], v22, v7, s[0:1]
	v_cndmask_b32_e32 v9, v9, v35, vcc
	v_cndmask_b32_e32 v8, v8, v34, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_cmp_lg_u32 s14, s24
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v7, 0, v7, vcc
	v_cndmask_b32_e32 v6, 0, v6, vcc
	s_cbranch_scc1 .LBB0_167
; %bb.166:                              ; %Flow831
                                        ;   in Loop: Header=BB0_165 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_164
	s_branch .LBB0_168
.LBB0_167:                              ;   in Loop: Header=BB0_165 Depth=2
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB0_164
.LBB0_168:                              ;   in Loop: Header=BB0_165 Depth=2
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB0_164
.LBB0_169:                              ; %.critedge262
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 91
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 92
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals,comdat
.Lfunc_end0:
	.size	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals, .Lfunc_end0-_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.num_vgpr, 91
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10492
; TotalNumSgprs: 106
; NumVgprs: 91
; NumAgprs: 0
; TotalNumVgprs: 91
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 91
; AccumOffset: 92
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 22
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[22:23], s[0:1], 0x60
	s_load_dwordx8 s[24:31], s[0:1], 0x100
	s_load_dwordx8 s[36:43], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
                                        ; implicit-def: $vgpr90 : SGPR spill to VGPR lane
	s_load_dwordx2 s[34:35], s[0:1], 0xf8
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s86, s25, 31
	s_lshr_b32 s3, s86, 29
	v_writelane_b32 v90, s4, 0
	s_add_i32 s3, s25, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v90, s5, 1
	v_writelane_b32 v90, s6, 2
	v_writelane_b32 v90, s7, 3
	s_mov_b64 s[6:7], -1
	s_cmp_ge_i32 s2, s31
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_cbranch_scc0 .LBB1_73
; %bb.1:
	s_sub_i32 s16, s2, s31
	s_mov_b32 s17, 0
	s_lshl_b64 s[6:7], s[16:17], 2
	s_add_u32 s6, s42, s6
	s_addc_u32 s7, s43, s7
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB1_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB1_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit319
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_eq_u64 s[10:11], 0
	s_cbranch_scc1 .LBB1_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB1_6
; %bb.5:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB1_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit322
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_branch .LBB1_8
.LBB1_7:
	s_mov_b64 s[6:7], -1
.LBB1_8:                                ; %Flow877
	s_mov_b64 s[8:9], exec
	v_writelane_b32 v90, s8, 24
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_nop 0
	v_writelane_b32 v90, s9, 25
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB1_72
; %bb.9:                                ; %.critedge
	s_mul_i32 s93, s29, s28
	s_cmp_ge_i32 s16, s93
	s_cbranch_scc1 .LBB1_72
; %bb.10:                               ; %.lr.ph
	s_mul_hi_i32 s13, s33, s26
	s_mul_i32 s12, s33, s26
	s_ashr_i32 s49, s26, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_add_u32 s50, s22, s6
	s_addc_u32 s51, s23, s7
	s_add_u32 s52, s50, s6
	s_addc_u32 s53, s51, s7
	s_add_u32 s54, s52, s6
	s_addc_u32 s55, s53, s7
	s_add_u32 s56, s54, s6
	s_addc_u32 s57, s55, s7
	s_add_u32 s58, s56, s6
	s_addc_u32 s59, s57, s7
	s_add_u32 s60, s58, s6
	s_addc_u32 s61, s59, s7
	s_add_u32 s62, s60, s6
	s_addc_u32 s63, s61, s7
	s_load_dwordx2 s[6:7], s[0:1], 0x120
	s_load_dwordx2 s[80:81], s[0:1], 0x90
	s_mov_b32 s48, s26
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_mbcnt_hi_u32_b32 v4, -1, v4
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s17, s30, 3
	s_add_i32 s3, s6, 64
	s_cmp_lg_u32 s24, 0
	v_writelane_b32 v90, s3, 16
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 4
	s_cmp_lg_u32 s24, 1
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v90, s7, 5
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 10
	s_cmp_lg_u32 s24, 2
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v90, s7, 11
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 14
	s_cmp_lg_u32 s24, 3
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v90, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 8
	s_cmp_lg_u32 s24, 4
	v_mul_lo_u32 v64, s28, v0
	v_writelane_b32 v90, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 22
	s_cmp_lg_u32 s24, 5
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v90, s7, 23
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 18
	s_cmp_lg_u32 s24, 6
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v90, s7, 19
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 20
	s_cmp_lg_u32 s24, 7
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v90, s7, 21
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s87, s29
	v_cvt_f32_u32_e32 v2, s87
	s_sub_i32 s20, 0, s87
	s_ashr_i32 s3, s29, 31
	s_lshl_b64 s[82:83], s[48:49], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v90, s6, 12
	v_mov_b32_e32 v3, v5
	s_mov_b64 s[98:99], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v90, s7, 13
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_bfrev_b32_e32 v66, 32
	v_readfirstlane_b32 s21, v2
	s_mul_i32 s20, s20, s21
	s_mul_hi_u32 s20, s21, s20
	s_add_i32 s88, s21, s20
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v90, s20, 6
	v_lshrrev_b32_e32 v2, 3, v0
	s_movk_i32 s89, 0x7fff
	v_writelane_b32 v90, s21, 7
	s_mul_i32 s20, s13, 14
	s_mul_hi_u32 s21, s12, 14
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s12, 14
	s_add_u32 s20, s22, s20
	s_addc_u32 s21, s23, s21
	s_add_u32 s20, s20, 2
	s_addc_u32 s21, s21, 0
	v_writelane_b32 v90, s20, 26
	s_mov_b32 s90, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	v_writelane_b32 v90, s21, 27
	s_lshl_b64 s[20:21], s[12:13], 2
	s_add_u32 s20, s22, s20
	s_addc_u32 s21, s23, s21
	v_writelane_b32 v90, s20, 28
	s_nop 1
	v_writelane_b32 v90, s21, 29
	s_mul_i32 s20, s13, 12
	s_mul_hi_u32 s21, s12, 12
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s12, 12
	s_add_u32 s20, s22, s20
	s_addc_u32 s21, s23, s21
	s_add_u32 s20, s20, 2
	s_addc_u32 s21, s21, 0
	v_writelane_b32 v90, s20, 30
	s_nop 1
	v_writelane_b32 v90, s21, 31
	s_mul_i32 s20, s13, 6
	s_mul_hi_u32 s21, s12, 6
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s12, 6
	s_add_u32 s20, s22, s20
	s_addc_u32 s21, s23, s21
	v_writelane_b32 v90, s20, 32
	s_nop 1
	v_writelane_b32 v90, s21, 33
	s_mul_i32 s20, s13, 10
	s_mul_hi_u32 s21, s12, 10
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s12, 10
	s_add_u32 s20, s22, s20
	s_addc_u32 s21, s23, s21
	s_add_u32 s94, s20, 2
	s_addc_u32 s95, s21, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	s_add_u32 s96, s22, s12
	s_addc_u32 s97, s23, s13
	s_xor_b64 s[64:65], s[14:15], -1
	s_branch .LBB1_13
.LBB1_11:                               ; %Flow862
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s16, s31
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s93
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB1_12:                               ; %Flow874
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB1_72
.LBB1_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_17 Depth 2
                                        ;     Child Loop BB1_29 Depth 2
                                        ;       Child Loop BB1_33 Depth 3
	s_abs_i32 s13, s16
	s_mul_hi_u32 s14, s13, s88
	s_mul_i32 s15, s14, s87
	s_ashr_i32 s12, s16, 31
	s_sub_i32 s13, s13, s15
	s_xor_b32 s12, s12, s3
	s_add_i32 s15, s14, 1
	s_sub_i32 s20, s13, s87
	s_cmp_ge_u32 s13, s87
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s13, s20, s13
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s13, s87
	s_cselect_b32 s13, s15, s14
	s_xor_b32 s13, s13, s12
	s_sub_i32 s91, s13, s12
	s_mul_i32 s12, s91, s29
	s_sub_i32 s92, s16, s12
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB1_21
; %bb.14:                               ;   in Loop: Header=BB1_13 Depth=1
	v_add_u32_e32 v4, s91, v64
	v_mul_lo_u32 v4, v4, s29
	v_add_u32_e32 v6, s92, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[38:39]
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_21
; %bb.15:                               ; %.lr.ph.i.i.i324.preheader
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB1_17
.LBB1_16:                               ; %Flow870
                                        ;   in Loop: Header=BB1_17 Depth=2
	s_and_b64 s[20:21], exec, s[68:69]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[66:67], exec
	s_and_b64 s[66:67], s[72:73], exec
	s_or_b64 s[66:67], s[20:21], s[66:67]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB1_19
.LBB1_17:                               ; %.lr.ph.i.i.i324
                                        ;   Parent Loop BB1_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s70, s70, 1
	s_addc_u32 s71, s71, 0
	v_mov_b64_e32 v[8:9], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[70:71], v[8:9]
	s_mov_b64 s[72:73], -1
	s_or_b64 s[68:69], s[68:69], exec
	s_cbranch_vccnz .LBB1_16
; %bb.18:                               ;   in Loop: Header=BB1_17 Depth=2
	s_sleep 4
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[20:21], s[68:69], exec
	s_mov_b64 s[72:73], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[68:69], vcc, exec
	s_or_b64 s[68:69], s[20:21], s[68:69]
	s_branch .LBB1_16
.LBB1_19:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[66:67]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB1_21
; %bb.20:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_atomic_or v[6:7], v66
.LBB1_21:                               ; %.critedge508
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[18:19]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB1_25
; %bb.22:                               ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b64 s[14:15], s[8:9]
	s_cbranch_execz .LBB1_24
; %bb.23:                               ;   in Loop: Header=BB1_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_load_dword v4, v[6:7] sc1
.LBB1_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit330
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB1_25:                               ; %Flow873
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[14:15]
	s_cbranch_execz .LBB1_12
; %bb.26:                               ; %.critedge522
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB1_39
; %bb.27:                               ; %.lr.ph.i331.preheader
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_mul_i32 s68, s91, s30
	s_lshl_b32 s70, s92, 6
	s_ashr_i32 s69, s68, 31
	s_ashr_i32 s71, s70, 31
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[68:69]
	v_mov_b64_e32 v[8:9], s[70:71]
	v_mad_u64_u32 v[8:9], s[14:15], s48, v6, v[8:9]
	v_mul_lo_u32 v4, s48, v7
	v_mul_lo_u32 v6, s49, v6
	v_add3_u32 v9, v6, v9, v4
	v_readlane_b32 s14, v90, 6
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s15, v90, 7
	v_lshl_add_u64 v[6:7], s[22:23], 0, v[22:23]
	v_lshl_add_u64 v[10:11], s[50:51], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 26
	v_readlane_b32 s15, v90, 27
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	s_mov_b64 s[72:73], 0
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 28
	v_readlane_b32 s15, v90, 29
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 30
	v_readlane_b32 s15, v90, 31
	s_nop 1
	v_lshl_add_u64 v[16:17], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 32
	v_readlane_b32 s15, v90, 33
	s_nop 1
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	s_branch .LBB1_29
.LBB1_28:                               ; %.critedge.i334
                                        ;   in Loop: Header=BB1_29 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s17, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[82:83]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[82:83]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[82:83]
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[82:83]
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[82:83]
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[82:83]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[82:83]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[82:83]
	s_or_b64 s[72:73], vcc, s[72:73]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	s_andn2_b64 exec, exec, s[72:73]
	s_cbranch_execz .LBB1_39
.LBB1_29:                               ; %.lr.ph.i331
                                        ;   Parent Loop BB1_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_33 Depth 3
	v_lshlrev_b32_e32 v4, 3, v69
	v_and_or_b32 v24, v4, 56, s70
	v_mov_b32_e32 v25, s71
	v_lshl_add_u64 v[26:27], v[24:25], 0, 8
	v_cmp_lt_u64_e32 vcc, s[48:49], v[26:27]
	s_or_b64 s[14:15], s[64:65], vcc
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_xor_b64 s[74:75], exec, s[20:21]
	s_cbranch_execz .LBB1_37
; %bb.30:                               ; %.preheader.i333.preheader
                                        ;   in Loop: Header=BB1_29 Depth=2
	v_and_b32_e32 v4, 56, v68
	v_lshl_add_u64 v[24:25], s[70:71], 0, v[4:5]
	v_lshlrev_b32_e32 v4, 1, v68
	v_and_b32_e32 v4, 0x70, v4
	v_lshl_add_u64 v[26:27], v[6:7], 0, v[4:5]
	v_lshl_add_u64 v[28:29], v[8:9], 0, v[4:5]
	v_lshl_add_u64 v[30:31], v[10:11], 0, v[4:5]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[4:5]
	v_lshl_add_u64 v[34:35], v[14:15], 0, v[4:5]
	v_lshl_add_u64 v[36:37], v[16:17], 0, v[4:5]
	v_lshl_add_u64 v[38:39], v[18:19], 0, v[4:5]
	v_lshl_add_u64 v[40:41], v[20:21], 0, v[4:5]
	v_lshl_add_u64 v[42:43], v[22:23], 0, v[4:5]
	s_mov_b64 s[76:77], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr78_sgpr79
	s_branch .LBB1_33
.LBB1_31:                               ; %Flow864
                                        ;   in Loop: Header=BB1_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[78:79], s[78:79], s[20:21]
.LBB1_32:                               ; %Flow863
                                        ;   in Loop: Header=BB1_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[78:79]
	s_or_b64 s[76:77], s[14:15], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB1_36
.LBB1_33:                               ; %.preheader.i333
                                        ;   Parent Loop BB1_13 Depth=1
                                        ;     Parent Loop BB1_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[48:49], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_32
; %bb.34:                               ; %.preheader.i333.1
                                        ;   in Loop: Header=BB1_33 Depth=3
	v_lshl_add_u64 v[48:49], v[26:27], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[30:31], 0, v[44:45]
	global_load_ushort v4, v[48:49], off
	global_load_ushort v72, v[50:51], off
	v_lshl_add_u64 v[52:53], v[34:35], 0, v[44:45]
	global_load_ushort v73, v[52:53], off
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[44:45]
	global_load_ushort v74, v[54:55], off
	v_lshl_add_u64 v[56:57], v[42:43], 0, v[44:45]
	global_load_ushort v75, v[56:57], off
	v_lshl_add_u64 v[58:59], v[40:41], 0, v[44:45]
	global_load_ushort v76, v[58:59], off offset:-2
	v_lshl_add_u64 v[60:61], v[36:37], 0, v[44:45]
	global_load_ushort v77, v[60:61], off offset:-2
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[44:45]
	global_load_ushort v78, v[62:63], off offset:-2
	v_lshl_add_u64 v[70:71], v[24:25], 0, 1
	v_cmp_gt_u64_e32 vcc, s[48:49], v[70:71]
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[44:45]
	s_mov_b64 s[20:21], -1
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v70, 16, v72
	v_add_f32_e32 v4, v4, v70
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v71, 16, v73
	v_add_f32_e32 v4, v4, v71
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v72, 16, v74
	v_add_f32_e32 v4, v4, v72
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v73, 16, v75
	v_add_f32_e32 v4, v4, v73
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v74, 16, v76
	v_add_f32_e32 v4, v4, v74
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v75, 16, v77
	v_add_f32_e32 v4, v4, v75
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v76, 16, v78
	v_add_f32_e32 v4, v4, v76
	v_bfe_u32 v70, v4, 16, 1
	v_add3_u32 v4, v4, v70, s89
	global_store_short_d16_hi v[46:47], v4, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB1_31
; %bb.35:                               ;   in Loop: Header=BB1_33 Depth=3
	global_load_ushort v4, v[50:51], off offset:2
	s_nop 0
	global_load_ushort v48, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v49, v[52:53], off offset:2
	global_load_ushort v50, v[54:55], off offset:2
	global_load_ushort v51, v[56:57], off offset:2
	s_nop 0
	global_load_ushort v52, v[58:59], off
	global_load_ushort v53, v[60:61], off
	global_load_ushort v54, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[24:25], v[24:25], 0, 2
	s_orn2_b64 s[20:21], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v48, 16, v48
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v4, v48, v4
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v4, v4, v49
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v4, v4, v50
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v4, v4, v51
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v53, 16, v53
	v_add_f32_e32 v4, v4, v52
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v4, v4, v53
	v_add_f32_e32 v4, v4, v54
	v_bfe_u32 v48, v4, 16, 1
	v_add3_u32 v4, v4, v48, s89
	global_store_short_d16_hi v[46:47], v4, off
	s_branch .LBB1_31
.LBB1_36:                               ; %Flow865
                                        ;   in Loop: Header=BB1_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB1_37:                               ; %Flow866
                                        ;   in Loop: Header=BB1_29 Depth=2
	s_andn2_saveexec_b64 vcc, s[74:75]
	s_cbranch_execz .LBB1_28
; %bb.38:                               ;   in Loop: Header=BB1_29 Depth=2
	v_lshrrev_b32_e32 v4, 3, v69
	v_lshl_add_u64 v[26:27], v[4:5], 0, s[68:69]
	v_mul_lo_u32 v4, v26, s49
	v_mul_lo_u32 v27, v27, s48
	v_mad_u64_u32 v[24:25], s[14:15], v26, s48, v[24:25]
	v_add3_u32 v25, v27, v25, v4
	v_lshlrev_b64 v[56:57], 1, v[24:25]
	v_lshl_add_u64 v[24:25], s[22:23], 0, v[56:57]
	v_lshl_add_u64 v[28:29], s[50:51], 0, v[56:57]
	global_load_dwordx4 v[24:27], v[24:25], off
	v_lshl_add_u64 v[32:33], s[52:53], 0, v[56:57]
	global_load_dwordx4 v[28:31], v[28:29], off
	v_lshl_add_u64 v[36:37], s[54:55], 0, v[56:57]
	global_load_dwordx4 v[32:35], v[32:33], off
	v_lshl_add_u64 v[40:41], s[56:57], 0, v[56:57]
	global_load_dwordx4 v[36:39], v[36:37], off
	v_lshl_add_u64 v[44:45], s[58:59], 0, v[56:57]
	global_load_dwordx4 v[40:43], v[40:41], off
	v_lshl_add_u64 v[48:49], s[60:61], 0, v[56:57]
	global_load_dwordx4 v[44:47], v[44:45], off
	v_lshl_add_u64 v[52:53], s[62:63], 0, v[56:57]
	global_load_dwordx4 v[48:51], v[48:49], off
	v_lshl_add_u64 v[56:57], s[80:81], 0, v[56:57]
	global_load_dwordx4 v[52:55], v[52:53], off
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v59, 0xffff0000, v24
	v_and_b32_e32 v61, 0xffff0000, v25
	v_lshlrev_b32_e32 v58, 16, v24
	v_lshlrev_b32_e32 v60, 16, v25
	s_waitcnt vmcnt(6)
	v_and_b32_e32 v25, 0xffff0000, v28
	v_lshlrev_b32_e32 v24, 16, v28
	v_and_b32_e32 v63, 0xffff0000, v26
	v_and_b32_e32 v71, 0xffff0000, v27
	v_lshlrev_b32_e32 v62, 16, v26
	v_lshlrev_b32_e32 v70, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v29
	v_lshlrev_b32_e32 v26, 16, v29
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v29, 0xffff0000, v32
	v_lshlrev_b32_e32 v28, 16, v32
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
	v_and_b32_e32 v73, 0xffff0000, v30
	v_and_b32_e32 v75, 0xffff0000, v31
	v_lshlrev_b32_e32 v72, 16, v30
	v_lshlrev_b32_e32 v74, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v33
	v_lshlrev_b32_e32 v30, 16, v33
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v33, 0xffff0000, v36
	v_lshlrev_b32_e32 v32, 16, v36
	v_pk_add_f32 v[26:27], v[26:27], v[60:61]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_and_b32_e32 v77, 0xffff0000, v34
	v_and_b32_e32 v79, 0xffff0000, v35
	v_lshlrev_b32_e32 v76, 16, v34
	v_lshlrev_b32_e32 v78, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v37
	v_lshlrev_b32_e32 v34, 16, v37
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v36, 16, v40
	v_and_b32_e32 v37, 0xffff0000, v40
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[32:33]
	v_and_b32_e32 v81, 0xffff0000, v38
	v_and_b32_e32 v83, 0xffff0000, v39
	v_lshlrev_b32_e32 v80, 16, v38
	v_lshlrev_b32_e32 v82, 16, v39
	v_lshlrev_b32_e32 v38, 16, v41
	v_and_b32_e32 v39, 0xffff0000, v41
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v84, 16, v44
	v_and_b32_e32 v85, 0xffff0000, v44
	v_pk_add_f32 v[26:27], v[26:27], v[34:35]
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	v_lshlrev_b32_e32 v40, 16, v45
	v_and_b32_e32 v41, 0xffff0000, v45
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v44, 16, v48
	v_and_b32_e32 v45, 0xffff0000, v48
	v_pk_add_f32 v[26:27], v[26:27], v[38:39]
	v_pk_add_f32 v[24:25], v[24:25], v[84:85]
	v_lshlrev_b32_e32 v86, 16, v49
	v_and_b32_e32 v87, 0xffff0000, v49
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v88, 16, v52
	v_and_b32_e32 v89, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[88:89]
	v_pk_add_f32 v[26:27], v[26:27], v[48:49]
	v_bfe_u32 v29, v25, 16, 1
	v_bfe_u32 v30, v24, 16, 1
	v_pk_add_f32 v[52:53], v[72:73], v[62:63]
	v_bfe_u32 v4, v27, 16, 1
	v_bfe_u32 v28, v26, 16, 1
	v_add3_u32 v32, v24, v30, s89
	v_add3_u32 v33, v25, v29, s89
	v_pk_add_f32 v[24:25], v[74:75], v[70:71]
	v_add3_u32 v34, v26, v28, s89
	v_add3_u32 v4, v27, v4, s89
	v_pk_add_f32 v[24:25], v[24:25], v[78:79]
	v_pk_add_f32 v[26:27], v[52:53], v[76:77]
	v_pk_add_f32 v[24:25], v[24:25], v[82:83]
	v_pk_add_f32 v[26:27], v[26:27], v[80:81]
	v_lshlrev_b32_e32 v28, 16, v42
	v_lshlrev_b32_e32 v30, 16, v43
	v_and_b32_e32 v29, 0xffff0000, v42
	v_and_b32_e32 v31, 0xffff0000, v43
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v47
	v_lshlrev_b32_e32 v30, 16, v46
	v_and_b32_e32 v29, 0xffff0000, v47
	v_and_b32_e32 v31, 0xffff0000, v46
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v50
	v_lshlrev_b32_e32 v30, 16, v51
	v_and_b32_e32 v29, 0xffff0000, v50
	v_and_b32_e32 v31, 0xffff0000, v51
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v55
	v_lshlrev_b32_e32 v30, 16, v54
	v_and_b32_e32 v29, 0xffff0000, v55
	v_and_b32_e32 v31, 0xffff0000, v54
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_bfe_u32 v30, v27, 16, 1
	v_bfe_u32 v28, v25, 16, 1
	v_bfe_u32 v29, v24, 16, 1
	v_bfe_u32 v31, v26, 16, 1
	v_add3_u32 v26, v26, v31, s89
	v_add3_u32 v30, v27, v30, s89
	v_add3_u32 v24, v24, v29, s89
	v_add3_u32 v25, v25, v28, s89
	v_perm_b32 v27, v25, v24, s90
	v_perm_b32 v26, v30, v26, s90
	v_perm_b32 v25, v4, v34, s90
	v_perm_b32 v24, v33, v32, s90
	global_store_dwordx4 v[56:57], v[24:27], off
	s_branch .LBB1_28
.LBB1_39:                               ; %Flow868
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB1_11
; %bb.40:                               ; %.preheader528
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s28, s24
	v_readlane_b32 s13, v90, 16
	s_add_i32 s12, s91, s12
	s_add_i32 s13, s13, s92
	s_mul_i32 s12, s12, s29
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s38, s12
	s_addc_u32 s15, s39, s13
	v_readlane_b32 s12, v90, 4
	v_readlane_b32 s13, v90, 5
	s_and_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v4, s15
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v6, s[12:13], s14, v6
	s_nop 1
	v_subb_co_u32_e64 v7, s[12:13], v4, v7, s[12:13]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v7, s[12:13]
	v_cndmask_b32_e64 v6, 0, v6, s[12:13]
	s_cbranch_vccz .LBB1_71
; %bb.41:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB1_43
.LBB1_42:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_43:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:16
	v_readlane_b32 s12, v90, 10
	v_readlane_b32 s13, v90, 11
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_45
; %bb.44:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_45:                               ; %Flow860
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_47
; %bb.46:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_47:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:24
	v_readlane_b32 s12, v90, 14
	v_readlane_b32 s13, v90, 15
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_49
; %bb.48:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_49:                               ; %Flow859
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_51
; %bb.50:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_51:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:32
	v_readlane_b32 s12, v90, 8
	v_readlane_b32 s13, v90, 9
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_53
; %bb.52:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_53:                               ; %Flow858
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_55
; %bb.54:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_55:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:40
	v_readlane_b32 s12, v90, 22
	v_readlane_b32 s13, v90, 23
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_57
; %bb.56:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_57:                               ; %Flow857
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_59
; %bb.58:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_59:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:48
	v_readlane_b32 s12, v90, 18
	v_readlane_b32 s13, v90, 19
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_61
; %bb.60:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_61:                               ; %Flow856
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_63
; %bb.62:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_63:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:56
	v_readlane_b32 s12, v90, 20
	v_readlane_b32 s13, v90, 21
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_65
; %bb.64:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_65:                               ; %Flow855
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_67
; %bb.66:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB1_67:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:64
	v_readlane_b32 s12, v90, 12
	v_readlane_b32 s13, v90, 13
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_69
; %bb.68:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_69:                               ; %Flow
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB1_11
; %bb.70:                               ;   in Loop: Header=BB1_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB1_11
.LBB1_71:                               ;   in Loop: Header=BB1_13 Depth=1
	s_branch .LBB1_42
.LBB1_72:                               ; %Flow878
	v_readlane_b32 s4, v90, 24
	v_readlane_b32 s5, v90, 25
	s_or_b64 exec, exec, s[4:5]
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
.LBB1_73:                               ; %Flow918
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB1_171
; %bb.74:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s42, s4
	s_addc_u32 s5, s43, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v90, s8, 4
	s_nop 1
	v_writelane_b32 v90, s9, 5
	s_and_b64 s[8:9], s[6:7], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB1_76
; %bb.75:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB1_76:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[6:7], 0
	s_cselect_b64 s[56:57], -1, 0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB1_80
; %bb.77:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_79
; %bb.78:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB1_79:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB1_81
	s_branch .LBB1_171
.LBB1_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB1_171
.LBB1_81:                               ; %.critedge506
	s_lshr_b32 s3, s86, 26
	s_add_i32 s3, s25, s3
	s_ashr_i32 s3, s3, 6
	s_mul_i32 s4, s29, s3
	s_cmp_ge_i32 s2, s4
	s_cbranch_scc1 .LBB1_171
; %bb.82:                               ; %.lr.ph557
	s_load_dwordx2 s[48:49], s[0:1], 0x80
	s_load_dwordx4 s[8:11], s[0:1], 0x70
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[46:47], s[0:1], 0x20
	s_load_dwordx2 s[6:7], s[0:1], 0x30
	s_load_dwordx2 s[92:93], s[0:1], 0x50
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[16:17], src_shared_base
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s11, 0, 0
	s_cselect_b32 s9, s17, 0
	s_and_b32 s0, s11, 15
	s_and_b32 s12, s11, -16
	s_add_u32 s12, s12, 16
	s_mov_b32 s1, 0
	s_addc_u32 s13, s9, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s51, s11, s12
	s_cselect_b32 s0, s9, s13
	s_add_u32 s9, s51, 0x4000
	s_addc_u32 s0, s0, 0
	s_and_b32 s11, s9, -16
	s_and_b32 s0, s9, 15
	s_add_u32 s11, s11, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s53, s9, s11
	s_add_i32 s0, s27, 63
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_i32 s0, s0, s1
	v_and_b32_e32 v18, 56, v2
	v_lshrrev_b32_e32 v6, 3, v0
	s_ashr_i32 s86, s0, 6
	v_mad_u64_u32 v[2:3], s[0:1], v6, s46, v[18:19]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[20:21], v[2:3], 1, s[4:5]
	v_mad_u64_u32 v[2:3], s[0:1], v6, s92, v[18:19]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b32_e32 v19, 4, v0
	v_lshl_add_u64 v[22:23], v[2:3], 1, s[6:7]
	v_add_u32_e32 v2, s51, v19
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_or_b32_e32 v41, 8, v19
	v_xor_b32_e32 v40, v3, v2
	v_add_u32_e32 v2, s51, v41
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	s_lshl_b32 s25, s29, 2
	v_xor_b32_e32 v42, v3, v2
	v_add_u32_e32 v2, s53, v19
	v_lshrrev_b32_e32 v3, 4, v2
	s_cmp_gt_i32 s27, 0
	v_and_b32_e32 v3, 0x78, v3
	s_cselect_b64 s[78:79], -1, 0
	s_ashr_i32 s1, s14, 31
	s_mov_b32 s0, s14
	v_xor_b32_e32 v43, v3, v2
	v_add_u32_e32 v2, s53, v41
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s0, s38, s0
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s1, s39, s1
	v_xor_b32_e32 v44, v3, v2
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v90, s0, 6
	v_bfe_u32 v4, v0, 6, 2
	v_lshrrev_b32_e32 v5, 8, v0
	v_lshlrev_b32_e32 v7, 7, v2
	v_writelane_b32 v90, s1, 7
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v45, v5, 12, v7
	v_lshl_or_b32 v46, v4, 11, v7
	v_writelane_b32 v90, s0, 8
	v_and_b32_e32 v7, 63, v0
	s_min_i32 s27, s30, 32
	v_writelane_b32 v90, s1, 9
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	s_bfe_i64 s[60:61], s[48:49], 0x200000
	s_cmp_gt_i32 s30, 0
	v_writelane_b32 v90, s0, 10
	s_cselect_b64 s[62:63], -1, 0
	v_lshl_or_b32 v49, v4, 4, v2
	v_writelane_b32 v90, s1, 11
	v_lshrrev_b32_e32 v3, 2, v0
	v_readlane_b32 s4, v90, 0
	v_readlane_b32 s5, v90, 1
	s_cmp_lg_u64 s[4:5], 0
	s_cselect_b64 s[64:65], -1, 0
	s_max_i32 s0, s26, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s15, 0
	s_cselect_b64 s[66:67], -1, 0
	s_abs_i32 s90, s25
	v_cvt_f32_u32_e32 v2, s90
	v_and_b32_e32 v3, 12, v3
	v_lshl_or_b32 v50, v5, 5, v3
	v_lshlrev_b32_e32 v59, 1, v3
	v_rcp_iflag_f32_e32 v3, v2
	s_abs_i32 s91, s30
	v_cvt_f32_u32_e32 v5, s91
	v_readlane_b32 s6, v90, 2
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v3, v3
	v_rcp_iflag_f32_e32 v5, v5
	v_readlane_b32 s7, v90, 3
	v_writelane_b32 v90, s0, 12
	s_lshl_b32 s0, s27, 3
	s_mul_i32 s88, s10, s8
	v_cmp_gt_i32_e64 s[8:9], s0, v0
	v_mad_i64_i32 v[24:25], s[0:1], s48, v6, 0
	v_readfirstlane_b32 s1, v3
	v_mul_f32_e32 v3, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v3, v3
	s_sub_i32 s0, 0, s90
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s11, s1, s0
	s_sub_i32 s0, 0, s91
	v_readfirstlane_b32 s1, v3
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s95, s1, s0
	s_lshr_b32 s0, s95, 26
	s_mul_i32 s1, s0, s91
	s_sub_i32 s1, 64, s1
	s_max_i32 s89, s86, 1
	s_bfe_i32 s47, s29, 0x1001d
	s_ashr_i32 s94, s30, 31
	s_add_i32 s4, s0, 1
	s_sub_i32 s5, s1, s91
	s_cmp_ge_u32 s1, s91
	s_cselect_b32 s0, s4, s0
	s_cselect_b32 s1, s5, s1
	s_add_i32 s4, s0, 1
	s_cmp_ge_u32 s1, s91
	s_cselect_b32 s0, s4, s0
	s_abs_i32 s96, s33
	v_cvt_f32_u32_e32 v5, s96
	v_lshlrev_b32_e32 v4, 7, v6
	v_mov_b32_e32 v27, 0
	v_add_u32_e32 v2, 0, v4
	v_mov_b32_e32 v3, s17
	v_lshlrev_b32_e32 v26, 1, v18
	v_lshl_add_u64 v[28:29], v[2:3], 0, v[26:27]
	v_rcp_iflag_f32_e32 v3, v5
	v_add_u32_e32 v29, v2, v26
	s_xor_b32 s0, s0, s94
	s_sub_i32 s97, s0, s94
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s96
	s_ashr_i32 s98, s33, 31
	v_add3_u32 v64, 0, v26, v4
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s99, s1, s0
	s_cmp_gt_i32 s97, 0
	s_cselect_b64 s[4:5], -1, 0
	s_lshl_b32 s0, s46, 6
	v_writelane_b32 v90, s0, 14
	s_mul_i32 s0, s61, s27
	s_mul_hi_u32 s1, s48, s27
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s48, s27
	s_lshl_b64 s[72:73], s[0:1], 1
	v_cmp_gt_u32_e64 s[0:1], s97, v0
	v_and_b32_e32 v2, 7, v0
	v_lshlrev_b32_e32 v26, 4, v2
	v_writelane_b32 v90, s0, 16
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v90, s1, 17
	s_mov_b64 s[70:71], 0x80
	v_readlane_b32 s0, v90, 4
	v_readlane_b32 s1, v90, 5
	v_writelane_b32 v90, s4, 18
	s_and_b64 s[0:1], s[0:1], s[4:5]
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v90, s5, 19
	v_writelane_b32 v90, s0, 20
	s_mov_b64 s[54:55], 0
	v_mul_lo_u32 v47, s30, v0
	v_add_u32_e32 v48, -1, v1
	s_mul_i32 s88, s88, s24
	v_lshl_add_u32 v51, v49, 1, 0
	v_or_b32_e32 v52, 1, v50
	v_or_b32_e32 v53, 2, v50
	v_or_b32_e32 v54, 3, v50
	v_or_b32_e32 v55, 16, v50
	v_or_b32_e32 v56, 17, v50
	v_or_b32_e32 v57, 18, v50
	v_or_b32_e32 v58, 19, v50
	v_or_b32_e32 v60, 32, v59
	v_or_b32_e32 v61, 64, v59
	v_or_b32_e32 v62, 0x60, v59
	v_add_u32_e32 v63, 8, v18
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[70:71]
	v_lshl_add_u64 v[32:33], v[20:21], 0, s[70:71]
	v_lshl_add_u32 v65, v0, 1, 0
	v_or_b32_e32 v66, 1, v18
	v_lshl_add_u64 v[34:35], v[24:25], 1, v[26:27]
	s_sub_i32 s58, 0, s33
	v_bfrev_b32_e32 v67, 64
	v_and_b32_e32 v68, 0x100, v2
	s_movk_i32 s59, 0x7fff
	v_writelane_b32 v90, s1, 21
	v_writelane_b32 v90, s11, 22
	s_branch .LBB1_85
.LBB1_83:                               ; %Flow881
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_add_i32 s2, s2, s31
	s_mul_i32 s0, s29, s3
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB1_84:                               ; %Flow912
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[76:77]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[54:55], s[0:1], s[54:55]
	s_andn2_b64 exec, exec, s[54:55]
	s_cbranch_execz .LBB1_171
.LBB1_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_87 Depth 2
                                        ;     Child Loop BB1_98 Depth 2
                                        ;     Child Loop BB1_110 Depth 2
                                        ;       Child Loop BB1_114 Depth 3
                                        ;         Child Loop BB1_146 Depth 4
                                        ;         Child Loop BB1_155 Depth 4
                                        ;         Child Loop BB1_161 Depth 4
                                        ;     Child Loop BB1_167 Depth 2
	s_ashr_i32 s0, s2, 31
	s_xor_b32 s4, s0, s47
	s_abs_i32 s0, s2
	s_mul_hi_u32 s1, s0, s11
	s_mul_i32 s5, s1, s90
	s_sub_i32 s0, s0, s5
	s_add_i32 s5, s1, 1
	s_sub_i32 s6, s0, s90
	s_cmp_ge_u32 s0, s90
	s_cselect_b32 s1, s5, s1
	s_cselect_b32 s0, s6, s0
	s_add_i32 s5, s1, 1
	s_cmp_ge_u32 s0, s90
	s_cselect_b32 s0, s5, s1
	s_xor_b32 s5, s0, s4
	s_sub_i32 s74, s5, s4
	s_lshl_b32 s0, s74, 2
	s_sub_i32 s1, s3, s0
	s_min_i32 s1, s1, 4
	s_abs_i32 s6, s1
	v_cvt_f32_u32_e32 v2, s6
	s_mul_i32 s74, s74, s25
	s_sub_i32 s13, 0, s6
	s_sub_i32 s7, s2, s74
	v_rcp_iflag_f32_e32 v2, v2
	s_xor_b32 s12, s7, s1
	s_ashr_i32 s42, s12, 31
	s_abs_i32 s12, s7
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_mov_b32_e32 v5, v27
	v_mov_b32_e32 v4, v27
	v_mov_b32_e32 v9, v27
	v_readfirstlane_b32 s14, v2
	s_mul_i32 s13, s13, s14
	s_mul_hi_u32 s13, s14, s13
	s_add_i32 s14, s14, s13
	s_mul_hi_u32 s13, s12, s14
	s_mul_i32 s14, s13, s6
	s_sub_i32 s12, s12, s14
	s_add_i32 s14, s13, 1
	s_sub_i32 s15, s12, s6
	s_cmp_ge_u32 s12, s6
	s_cselect_b32 s13, s14, s13
	s_cselect_b32 s12, s15, s12
	s_add_i32 s14, s13, 1
	s_cmp_ge_u32 s12, s6
	s_cselect_b32 s6, s14, s13
	s_xor_b32 s43, s6, s42
	s_sub_i32 s49, s43, s42
	s_mul_i32 s75, s49, s1
	s_sub_i32 s1, s7, s75
	s_add_i32 s1, s1, s0
	s_lshl_b32 s68, s1, 6
	s_mul_i32 s0, s68, s46
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s69, s49, 6
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[20:21]
	s_mul_i32 s0, s69, s92
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[2:3], off

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[22:23]
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v40, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v42, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v43, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v44, v[16:17]

	;;#ASMEND
	s_andn2_b64 vcc, exec, s[78:79]
	v_mov_b32_e32 v3, v27
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v8, v27
	v_mov_b32_e32 v7, v27
	v_mov_b32_e32 v6, v27
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB1_93
; %bb.86:                               ; %.lr.ph545.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_lshl_add_u64 v[36:37], s[0:1], 1, v[30:31]
	s_lshl_b32 s0, s5, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s74
	s_sub_i32 s0, s0, s75
	s_lshl_b32 s1, s4, 2
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v90, 14
	s_mul_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v2, 0
	v_lshl_add_u64 v[38:39], s[0:1], 1, v[32:33]
	s_mov_b32 s7, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
.LBB1_87:                               ; %.lr.ph545
                                        ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s6, s7, 1
	s_cmp_lt_i32 s6, s86
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_ge_i32 s6, s86
	s_cbranch_scc1 .LBB1_89
; %bb.88:                               ;   in Loop: Header=BB1_87 Depth=2
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[38:39], off

	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[36:37], off

	;;#ASMEND
.LBB1_89:                               ;   in Loop: Header=BB1_87 Depth=2
	s_lshl_b32 s7, s7, 13
	s_and_b32 s7, s7, 0x2000
	s_add_i32 s12, s51, s7
	v_add_u32_e32 v26, s12, v45
	v_add_u32_e32 v70, v59, v26
	s_add_i32 s7, s53, s7
	v_lshrrev_b32_e32 v71, 4, v70
	v_add_u32_e32 v69, s7, v46
	v_and_b32_e32 v71, 0x78, v71
	v_add_u32_e32 v74, v60, v26
	v_xor_b32_e32 v72, v71, v70
	;;#ASMSTART
	ds_read_b64 v[70:71], v72 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v75, 4, v74
	v_add_u32_e32 v78, v59, v69
	;;#ASMSTART
	ds_read_b64 v[72:73], v72 offset:0x800

	;;#ASMEND
	v_and_b32_e32 v75, 0x78, v75
	v_lshrrev_b32_e32 v79, 4, v78
	v_add_u32_e32 v80, v60, v69
	v_xor_b32_e32 v76, v75, v74
	;;#ASMSTART
	ds_read_b64 v[74:75], v76 offset:0

	;;#ASMEND
	v_and_b32_e32 v79, 0x78, v79
	v_lshrrev_b32_e32 v81, 4, v80
	;;#ASMSTART
	ds_read_b64 v[76:77], v76 offset:0x800

	;;#ASMEND
	v_xor_b32_e32 v78, v79, v78
	v_and_b32_e32 v81, 0x78, v81
	;;#ASMSTART
	ds_read_b64 v[78:79], v78 offset:0

	;;#ASMEND
	v_xor_b32_e32 v80, v81, v80
	;;#ASMSTART
	ds_read_b64 v[80:81], v80 offset:0

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_andn2_b64 vcc, exec, s[0:1]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[70:71], v[78:79], v[6:9]
	v_add_u32_e32 v70, v61, v26
	v_lshrrev_b32_e32 v71, 4, v70
	v_and_b32_e32 v71, 0x78, v71
	v_mfma_f32_16x16x16_bf16 v[2:5], v[72:73], v[78:79], v[2:5]
	v_add_u32_e32 v26, v62, v26
	v_xor_b32_e32 v72, v71, v70
	;;#ASMSTART
	ds_read_b64 v[70:71], v72 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[74:75], v[80:81], v[6:9]
	v_lshrrev_b32_e32 v74, 4, v26
	;;#ASMSTART
	ds_read_b64 v[72:73], v72 offset:0x800

	;;#ASMEND
	v_and_b32_e32 v74, 0x78, v74
	v_xor_b32_e32 v26, v74, v26
	;;#ASMSTART
	ds_read_b64 v[74:75], v26 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[2:5], v[76:77], v[80:81], v[2:5]
	;;#ASMSTART
	ds_read_b64 v[76:77], v26 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v26, v61, v69
	v_lshrrev_b32_e32 v78, 4, v26
	v_and_b32_e32 v78, 0x78, v78
	v_xor_b32_e32 v26, v78, v26
	;;#ASMSTART
	ds_read_b64 v[78:79], v26 offset:0

	;;#ASMEND
	v_add_u32_e32 v26, v62, v69
	v_lshrrev_b32_e32 v69, 4, v26
	v_and_b32_e32 v69, 0x78, v69
	v_xor_b32_e32 v26, v69, v26
	;;#ASMSTART
	ds_read_b64 v[80:81], v26 offset:0

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[70:71], v[78:79], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[72:73], v[78:79], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[74:75], v[80:81], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[76:77], v[80:81], v[2:5]
	s_cbranch_vccnz .LBB1_91
; %bb.90:                               ;   in Loop: Header=BB1_87 Depth=2
	s_lshl_b32 s0, s6, 13
	s_and_b32 s0, s0, 0x2000
	s_add_i32 s1, s51, s0
	v_add_u32_e32 v26, s1, v19
	v_lshrrev_b32_e32 v69, 4, v26
	v_and_b32_e32 v69, 0x78, v69
	v_xor_b32_e32 v26, v69, v26
	v_add_u32_e32 v69, s1, v41
	v_lshrrev_b32_e32 v70, 4, v69
	v_and_b32_e32 v70, 0x78, v70
	s_add_i32 s0, s53, s0
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v69, v70, v69
	;;#ASMSTART
	ds_write_b64 v26, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v26, s0, v19
	;;#ASMSTART
	ds_write_b64 v69, v[12:13]

	;;#ASMEND
	v_lshrrev_b32_e32 v69, 4, v26
	v_and_b32_e32 v69, 0x78, v69
	v_xor_b32_e32 v26, v69, v26
	v_add_u32_e32 v69, s0, v41
	v_lshrrev_b32_e32 v70, 4, v69
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v70, 0x78, v70
	;;#ASMSTART
	ds_write_b64 v26, v[14:15]

	;;#ASMEND
	v_xor_b32_e32 v69, v70, v69
	;;#ASMSTART
	ds_write_b64 v69, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB1_91:                               ;   in Loop: Header=BB1_87 Depth=2
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[70:71]
	s_cmp_eq_u32 s89, s6
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[70:71]
	s_barrier
	s_cbranch_scc1 .LBB1_93
; %bb.92:                               ;   in Loop: Header=BB1_87 Depth=2
	s_mov_b32 s7, s6
	s_branch .LBB1_87
.LBB1_93:                               ; %Flow908
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s6, v90, 16
	v_readlane_b32 s7, v90, 17
	s_and_b64 s[6:7], s[0:1], s[6:7]
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB1_102
; %bb.94:                               ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s6, v90, 8
	v_readlane_b32 s7, v90, 9
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB1_102
; %bb.95:                               ;   in Loop: Header=BB1_85 Depth=1
	v_add_u32_e32 v10, s68, v47
	v_sub_u32_e32 v12, 0, v10
	v_max_i32_e32 v12, v10, v12
	v_mul_hi_u32 v13, v12, s99
	v_mul_lo_u32 v14, v13, s96
	v_sub_u32_e32 v12, v12, v14
	v_add_u32_e32 v14, 1, v13
	v_cmp_le_u32_e32 vcc, s96, v12
	v_ashrrev_i32_e32 v11, 31, v10
	v_xor_b32_e32 v11, s98, v11
	v_cndmask_b32_e32 v13, v13, v14, vcc
	v_subrev_u32_e32 v14, s96, v12
	v_cndmask_b32_e32 v12, v12, v14, vcc
	v_add_u32_e32 v14, 1, v13
	v_cmp_le_u32_e32 vcc, s96, v12
	s_nop 1
	v_cndmask_b32_e32 v12, v13, v14, vcc
	v_xor_b32_e32 v12, v12, v11
	v_sub_u32_e32 v11, v12, v11
	v_mul_lo_u32 v12, v11, s33
	v_sub_u32_e32 v10, v10, v12
	v_sub_u32_e32 v13, 0, v10
	v_ashrrev_i32_e32 v12, 31, v10
	v_max_i32_e32 v10, v10, v13
	v_mul_hi_u32 v13, v10, s95
	v_mul_lo_u32 v14, v13, s91
	v_sub_u32_e32 v10, v10, v14
	v_add_u32_e32 v14, 1, v13
	v_cmp_le_u32_e32 vcc, s91, v10
	v_xor_b32_e32 v12, s94, v12
	s_nop 0
	v_cndmask_b32_e32 v13, v13, v14, vcc
	v_subrev_u32_e32 v14, s91, v10
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_add_u32_e32 v14, 1, v13
	v_cmp_le_u32_e32 vcc, s91, v10
	s_nop 1
	v_cndmask_b32_e32 v10, v13, v14, vcc
	v_xor_b32_e32 v10, v10, v12
	v_sub_u32_e32 v10, v10, v12
	v_mad_u64_u32 v[10:11], s[6:7], v11, s28, v[10:11]
	v_mul_lo_u32 v10, v10, s29
	v_add_u32_e32 v10, s49, v10
	v_readlane_b32 s6, v90, 6
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s7, v90, 7
	s_nop 1
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v12, v48
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_102
; %bb.96:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[20:21], 0
                                        ; implicit-def: $sgpr16_sgpr17
                                        ; implicit-def: $sgpr18_sgpr19
	s_branch .LBB1_98
.LBB1_97:                               ; %Flow903
                                        ;   in Loop: Header=BB1_98 Depth=2
	s_and_b64 s[6:7], exec, s[18:19]
	s_or_b64 s[14:15], s[6:7], s[14:15]
	s_andn2_b64 s[6:7], s[16:17], exec
	s_and_b64 s[12:13], s[76:77], exec
	s_or_b64 s[16:17], s[6:7], s[12:13]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB1_100
.LBB1_98:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s20, s20, 1
	s_addc_u32 s21, s21, 0
	v_mov_b64_e32 v[12:13], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[20:21], v[12:13]
	s_mov_b64 s[76:77], -1
	s_or_b64 s[18:19], s[18:19], exec
	s_cbranch_vccnz .LBB1_97
; %bb.99:                               ;   in Loop: Header=BB1_98 Depth=2
	s_sleep 4
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_andn2_b64 s[6:7], s[18:19], exec
	s_mov_b64 s[76:77], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v12, v48
	s_and_b64 s[12:13], vcc, exec
	s_or_b64 s[18:19], s[6:7], s[12:13]
	s_branch .LBB1_97
.LBB1_100:                              ; %loop.exit.guard853
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB1_102
; %bb.101:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s12, v90, 0
	v_readlane_b32 s14, v90, 2
	v_readlane_b32 s15, v90, 3
	v_readlane_b32 s13, v90, 1
	s_nop 0
	v_mov_b64_e32 v[10:11], s[14:15]
	flat_atomic_or v[10:11], v67
.LBB1_102:                              ; %.critedge507
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[56:57]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB1_106
; %bb.103:                              ;   in Loop: Header=BB1_85 Depth=1
	v_mov_b32_e32 v10, 0
	s_mov_b64 s[14:15], exec
	v_readlane_b32 s6, v90, 10
	v_readlane_b32 s7, v90, 11
	s_and_b64 s[6:7], s[14:15], s[6:7]
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB1_105
; %bb.104:                              ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s16, v90, 0
	v_readlane_b32 s18, v90, 2
	v_readlane_b32 s19, v90, 3
	v_readlane_b32 s17, v90, 1
	s_nop 0
	v_mov_b64_e32 v[10:11], s[18:19]
	flat_load_dword v10, v[10:11] sc1
.LBB1_105:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit304
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v10, v68, v10
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v10, 0x2000000, v10
	v_cmp_eq_u32_e64 s[14:15], 0, v10
.LBB1_106:                              ; %Flow911
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_and_saveexec_b64 s[76:77], s[14:15]
	s_cbranch_execz .LBB1_84
; %bb.107:                              ; %.critedge521
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s0, v90, 18
	v_readlane_b32 s1, v90, 19
	s_mov_b64 s[10:11], s[78:79]
	s_mov_b32 s93, s3
	s_mov_b32 s3, s25
	s_mov_b64 s[44:45], s[56:57]
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB1_162
; %bb.108:                              ; %.lr.ph552
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_sub_i32 s12, s26, s69
	s_min_i32 s25, s12, 64
	s_abs_i32 s7, s25
	v_cvt_f32_u32_e32 v12, s7
	v_or_b32_e32 v10, s69, v49
	v_readlane_b32 s0, v90, 12
	v_cmp_gt_i32_e32 vcc, s26, v10
	v_rcp_iflag_f32_e32 v12, v12
	v_mov_b32_e32 v11, s0
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_readlane_b32 s16, v90, 0
	v_mul_f32_e32 v12, 0x4f7ffffe, v12
	v_cvt_u32_f32_e32 v12, v12
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s17, v90, 1
	v_readlane_b32 s18, v90, 2
	s_sub_i32 s18, 0, s7
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[16:17]
	v_cmp_gt_i32_e64 s[16:17], s12, v18
	s_ashr_i32 s12, s25, 31
	v_readlane_b32 s19, v90, 3
	v_mul_lo_u32 v13, s18, v12
	s_lshl_b32 s18, s12, 7
	v_subrev_u32_e32 v39, s18, v65
	s_lshl_b32 s18, s43, 6
	s_lshl_b32 s19, s42, 6
	s_sub_i32 s57, s18, s19
	s_lshl_b32 s18, s5, 2
	s_add_i32 s18, s2, s18
	s_sub_i32 s18, s18, s74
	s_sub_i32 s18, s18, s75
	s_lshl_b32 s19, s4, 2
	s_sub_i32 s18, s18, s19
	s_mul_i32 s6, s25, s27
	v_mul_hi_u32 v13, v12, v13
	s_lshl_b32 s18, s18, 6
	v_cmp_gt_i32_e64 s[0:1], s6, v0
	v_cmp_lt_i32_e64 s[78:79], s25, v63
	v_cmp_ge_i32_e64 s[14:15], s25, v63
	s_mov_b32 s13, 0
	v_add_u32_e32 v38, v12, v13
	s_lshl_b32 s52, s25, 1
	s_sub_i32 s56, 0, s25
	s_add_i32 s87, s88, s18
	s_branch .LBB1_110
.LBB1_109:                              ; %._crit_edge550
                                        ;   in Loop: Header=BB1_110 Depth=2
	s_add_i32 s13, s13, 1
	s_add_i32 s87, s87, s30
	s_cmp_eq_u32 s13, s97
	s_cbranch_scc1 .LBB1_162
.LBB1_110:                              ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_114 Depth 3
                                        ;         Child Loop BB1_146 Depth 4
                                        ;         Child Loop BB1_155 Depth 4
                                        ;         Child Loop BB1_161 Depth 4
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB1_109
; %bb.111:                              ; %.lr.ph549.preheader
                                        ;   in Loop: Header=BB1_110 Depth=2
	v_mov_b64_e32 v[16:17], s[36:37]
	flat_load_dwordx4 v[12:15], v[16:17]
	flat_load_dwordx4 v[70:73], v[16:17] offset:16
	flat_load_dwordx4 v[74:77], v[16:17] offset:32
	flat_load_dwordx4 v[78:81], v[16:17] offset:48
	s_nop 0
	flat_load_dwordx2 v[16:17], v[16:17] offset:64
	s_mul_i32 s50, s13, s30
	s_add_i32 s18, s50, s68
	s_abs_i32 s20, s18
	s_mul_hi_u32 s21, s20, s99
	s_mul_i32 s43, s21, s96
	s_ashr_i32 s19, s18, 31
	s_sub_i32 s20, s20, s43
	s_xor_b32 s19, s19, s98
	s_add_i32 s80, s21, 1
	s_sub_i32 s43, s20, s96
	s_cmp_ge_u32 s20, s96
	s_cselect_b32 s21, s80, s21
	s_cselect_b32 s20, s43, s20
	s_add_i32 s43, s21, 1
	s_cmp_ge_u32 s20, s96
	s_cselect_b32 s20, s43, s21
	s_xor_b32 s20, s20, s19
	s_sub_i32 s20, s20, s19
	s_mul_i32 s21, s20, s33
	s_sub_i32 s43, s18, s21
	s_cmp_eq_u32 s20, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 1
	v_mov_b32_e32 v26, s23
	s_mov_b32 s42, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v14, 0, v14, vcc
	v_cndmask_b32_e32 v15, 0, v15, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 2
	v_cndmask_b32_e32 v15, v15, v71, vcc
	v_cndmask_b32_e32 v14, v14, v70, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 3
	v_cndmask_b32_e32 v14, v14, v72, vcc
	v_cndmask_b32_e32 v15, v15, v73, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 4
	v_cndmask_b32_e32 v15, v15, v75, vcc
	v_cndmask_b32_e32 v14, v14, v74, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 5
	v_sub_co_u32_e64 v12, s[18:19], s22, v12
	v_cndmask_b32_e32 v14, v14, v76, vcc
	v_cndmask_b32_e32 v15, v15, v77, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 6
	v_subb_co_u32_e64 v13, s[18:19], v26, v13, s[18:19]
	v_cndmask_b32_e32 v15, v15, v79, vcc
	v_cndmask_b32_e32 v14, v14, v78, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 7
	v_cndmask_b32_e32 v14, v14, v80, vcc
	v_cndmask_b32_e32 v15, v15, v81, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s19, s43
	s_mul_hi_u32 s20, s19, s95
	s_mul_i32 s20, s20, s91
	s_sub_i32 s19, s19, s20
	s_ashr_i32 s18, s43, 31
	s_sub_i32 s20, s19, s91
	s_cmp_ge_u32 s19, s91
	s_cselect_b32 s19, s20, s19
	s_sub_i32 s20, s19, s91
	s_cmp_ge_u32 s19, s91
	s_cselect_b32 s19, s20, s19
	s_add_i32 s20, s43, s88
	s_add_i32 s43, s18, s87
	s_xor_b32 s19, s19, s18
	s_sub_i32 s21, s43, s21
	s_sub_i32 s18, s18, s19
	v_cndmask_b32_e32 v15, v15, v17, vcc
	v_cndmask_b32_e32 v14, v14, v16, vcc
	s_sub_i32 s19, s21, s19
	s_add_i32 s18, s20, s18
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_cmp_ne_u64_e32 vcc, 0, v[14:15]
	s_mul_i32 s19, s48, s19
	s_mul_i32 s20, s18, s48
	v_cndmask_b32_e32 v13, 0, v13, vcc
	v_cndmask_b32_e32 v12, 0, v12, vcc
	s_add_i32 s18, s57, s19
	s_add_i32 s20, s20, s69
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[34:35]
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s21, s20, 31
	v_lshl_add_u64 v[12:13], s[20:21], 1, v[12:13]
	v_lshl_add_u64 v[14:15], s[18:19], 1, v[14:15]
	s_branch .LBB1_114
.LBB1_112:                              ; %Flow895
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
.LBB1_113:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_add_i32 s42, s42, s27
	s_cmp_ge_i32 s42, s30
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[72:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB1_109
.LBB1_114:                              ; %.lr.ph549
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_110 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB1_146 Depth 4
                                        ;         Child Loop BB1_155 Depth 4
                                        ;         Child Loop BB1_161 Depth 4
	v_cndmask_b32_e64 v16, 0, 1, s[64:65]
	v_cmp_ne_u32_e64 s[18:19], 1, v16
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB1_116
; %bb.115:                              ;   in Loop: Header=BB1_114 Depth=3
	flat_load_ushort v16, v[10:11]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_branch .LBB1_117
.LBB1_116:                              ;   in Loop: Header=BB1_114 Depth=3
	v_mov_b32_e32 v16, 0
.LBB1_117:                              ;   in Loop: Header=BB1_114 Depth=3
	s_add_i32 s43, s42, s50
	s_add_i32 s80, s43, s27
	v_cmp_le_i32_e32 vcc, s43, v50
	v_cmp_gt_i32_e64 s[20:21], s80, v50
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_119
; %bb.118:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v6
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v50
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_119:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s43, v52
	v_cmp_gt_i32_e64 s[20:21], s80, v52
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_121
; %bb.120:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v7
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v52
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_121:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s43, v53
	v_cmp_gt_i32_e64 s[20:21], s80, v53
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_123
; %bb.122:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v8
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v53
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_123:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s43, v54
	v_cmp_gt_i32_e64 s[20:21], s80, v54
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_125
; %bb.124:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v16, v16, v9
	v_bfe_u32 v17, v16, 16, 1
	v_add3_u32 v16, v16, v17, s59
	v_subrev_u32_e32 v17, s43, v54
	v_lshl_add_u32 v17, v17, 7, v51
	ds_write_b16_d16_hi v17, v16
.LBB1_125:                              ; %.loopexit.i
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccnz .LBB1_127
; %bb.126:                              ;   in Loop: Header=BB1_114 Depth=3
	flat_load_ushort v16, v[10:11]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_branch .LBB1_128
.LBB1_127:                              ;   in Loop: Header=BB1_114 Depth=3
	v_mov_b32_e32 v16, 0
.LBB1_128:                              ;   in Loop: Header=BB1_114 Depth=3
	v_cmp_le_i32_e32 vcc, s43, v55
	v_cmp_gt_i32_e64 s[18:19], s80, v55
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_130
; %bb.129:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v2
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v55
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_130:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s43, v56
	v_cmp_gt_i32_e64 s[18:19], s80, v56
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_132
; %bb.131:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v3
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v56
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_132:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s43, v57
	v_cmp_gt_i32_e64 s[18:19], s80, v57
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_134
; %bb.133:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v17, v16, v4
	v_bfe_u32 v26, v17, 16, 1
	v_add3_u32 v17, v17, v26, s59
	v_subrev_u32_e32 v26, s43, v57
	v_lshl_add_u32 v26, v26, 7, v51
	ds_write_b16_d16_hi v26, v17
.LBB1_134:                              ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s43, v58
	v_cmp_gt_i32_e64 s[18:19], s80, v58
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_136
; %bb.135:                              ;   in Loop: Header=BB1_114 Depth=3
	v_add_f32_e32 v16, v16, v5
	v_bfe_u32 v17, v16, 16, 1
	v_add3_u32 v16, v16, v17, s59
	v_subrev_u32_e32 v17, s43, v58
	v_lshl_add_u32 v17, v17, 7, v51
	ds_write_b16_d16_hi v17, v16
.LBB1_136:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi32ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
	s_mul_i32 s18, s61, s42
	s_mul_hi_u32 s19, s60, s42
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s60, s42
	s_andn2_b64 vcc, exec, s[66:67]
	v_lshl_add_u64 v[16:17], s[18:19], 1, v[12:13]
	s_mov_b64 s[18:19], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB1_158
; %bb.137:                              ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[80:81], -1
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB1_143
; %bb.138:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[82:83], s[78:79]
	s_and_saveexec_b64 s[80:81], s[14:15]
; %bb.139:                              ;   in Loop: Header=BB1_114 Depth=3
	v_lshlrev_b32_e32 v26, 1, v24
	v_lshlrev_b32_e32 v36, 1, v18
	v_add3_u32 v26, v16, v26, v36
	v_or_b32_e32 v26, v28, v26
	v_and_b32_e32 v26, 15, v26
	v_cmp_eq_u32_e32 vcc, 0, v26
	s_andn2_b64 s[82:83], s[78:79], exec
	s_and_b64 s[84:85], vcc, exec
	s_or_b64 s[82:83], s[82:83], s[84:85]
; %bb.140:                              ; %Flow891
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[80:81]
	s_mov_b64 s[80:81], 0
	s_and_saveexec_b64 s[84:85], s[82:83]
; %bb.141:                              ; %.critedge.i
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[80:81], exec
; %bb.142:                              ; %Flow892
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[80:81], s[80:81], exec
.LBB1_143:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cndmask_b32_e64 v26, 0, 1, s[80:81]
	s_nop 0
	v_readfirstlane_b32 s20, v26
	s_bitcmp1_b32 s20, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB1_148
; %bb.144:                              ;   in Loop: Header=BB1_114 Depth=3
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB1_147
; %bb.145:                              ; %.lr.ph.i309.preheader
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v36, v39
	v_mov_b32_e32 v26, v0
.LBB1_146:                              ; %.lr.ph.i309
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_110 Depth=2
                                        ;       Parent Loop BB1_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v37, v26, v38
	v_mul_lo_u32 v69, v37, s7
	v_sub_u32_e32 v69, v26, v69
	v_add_u32_e32 v70, 1, v37
	v_subrev_u32_e32 v71, s7, v69
	v_cmp_le_u32_e32 vcc, s7, v69
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v70, vcc
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_add_u32_e32 v70, 1, v37
	v_cmp_le_u32_e32 vcc, s7, v69
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v70, vcc
	v_xor_b32_e32 v37, s12, v37
	v_subrev_u32_e32 v69, s12, v37
	v_mad_u64_u32 v[70:71], s[80:81], s56, v69, v[26:27]
	v_lshlrev_b32_e32 v37, 7, v37
	v_mul_lo_u32 v71, s52, v69
	v_sub_u32_e32 v37, v37, v71
	v_add_u32_e32 v37, v36, v37
	ds_read_u16 v37, v37
	v_mad_i64_i32 v[72:73], s[80:81], s60, v69, 0
	v_add_u32_e32 v26, 0x200, v26
	v_mov_b32_e32 v71, v27
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[16:17]
	v_cmp_le_i32_e32 vcc, s6, v26
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[72:73]
	v_add_u32_e32 v36, 0x400, v36
	s_or_b64 s[20:21], vcc, s[20:21]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[70:71], v37
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_146
.LBB1_147:                              ; %Flow883
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_148:                              ; %Flow889
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccnz .LBB1_157
; %bb.149:                              ;   in Loop: Header=BB1_114 Depth=3
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB1_156
; %bb.150:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_xor_b64 s[20:21], exec, s[20:21]
	s_cbranch_execz .LBB1_152
; %bb.151:                              ;   in Loop: Header=BB1_114 Depth=3
	ds_read_b128 v[70:73], v29
	v_lshl_add_u64 v[36:37], v[24:25], 1, v[16:17]
	v_lshlrev_b32_e32 v26, 1, v18
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[26:27]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[36:37], v[70:73]
.LBB1_152:                              ; %Flow886
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_andn2_saveexec_b64 s[20:21], s[20:21]
	s_cbranch_execz .LBB1_156
; %bb.153:                              ; %.preheader.i
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB1_156
; %bb.154:                              ; %.lr.ph.i312
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b32 s43, 0
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v26, v64
	v_mov_b64_e32 v[36:37], v[14:15]
.LBB1_155:                              ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_110 Depth=2
                                        ;       Parent Loop BB1_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v69, v26
	s_add_i32 s80, s43, 1
	v_add_u32_e32 v70, s43, v66
	s_cmp_gt_u32 s43, 6
	v_cmp_le_u32_e32 vcc, s25, v70
	s_mov_b32 s43, s80
	s_cselect_b64 s[80:81], -1, 0
	s_or_b64 s[80:81], s[80:81], vcc
	s_and_b64 s[80:81], exec, s[80:81]
	v_add_u32_e32 v26, 2, v26
	s_waitcnt lgkmcnt(0)
	flat_store_short v[36:37], v69
	s_or_b64 s[20:21], s[80:81], s[20:21]
	v_lshl_add_u64 v[36:37], v[36:37], 0, 2
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_155
.LBB1_156:                              ; %Flow888
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_or_b64 exec, exec, s[18:19]
.LBB1_157:                              ; %Flow890
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[18:19], 0
.LBB1_158:                              ; %Flow896
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB1_113
; %bb.159:                              ;   in Loop: Header=BB1_114 Depth=3
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB1_112
; %bb.160:                              ; %.lr.ph.i316.preheader
                                        ;   in Loop: Header=BB1_114 Depth=3
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v36, v39
	v_mov_b32_e32 v26, v0
.LBB1_161:                              ; %.lr.ph.i316
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_110 Depth=2
                                        ;       Parent Loop BB1_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v37, v26, v38
	v_mul_lo_u32 v69, v37, s7
	v_sub_u32_e32 v69, v26, v69
	v_add_u32_e32 v70, 1, v37
	v_subrev_u32_e32 v71, s7, v69
	v_cmp_le_u32_e32 vcc, s7, v69
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v70, vcc
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_add_u32_e32 v70, 1, v37
	v_cmp_le_u32_e32 vcc, s7, v69
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v70, vcc
	v_xor_b32_e32 v37, s12, v37
	v_subrev_u32_e32 v69, s12, v37
	v_mad_u64_u32 v[70:71], s[80:81], s56, v69, v[26:27]
	v_lshlrev_b32_e32 v37, 7, v37
	v_mul_lo_u32 v71, s52, v69
	v_sub_u32_e32 v37, v37, v71
	v_add_u32_e32 v37, v36, v37
	ds_read_u16 v37, v37
	v_mad_i64_i32 v[72:73], s[80:81], s60, v69, 0
	v_add_u32_e32 v26, 0x200, v26
	v_mov_b32_e32 v71, v27
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[16:17]
	v_cmp_le_i32_e32 vcc, s6, v26
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[72:73]
	v_add_u32_e32 v36, 0x400, v36
	s_or_b64 s[20:21], vcc, s[20:21]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[70:71], v37
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_161
	s_branch .LBB1_112
.LBB1_162:                              ; %._crit_edge553
                                        ;   in Loop: Header=BB1_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s6, v90, 4
	v_readlane_b32 s7, v90, 5
	s_and_b64 s[6:7], s[0:1], s[6:7]
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB1_164
; %bb.163:                              ;   in Loop: Header=BB1_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB1_164:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[14:15], exec
	v_readlane_b32 s0, v90, 20
	v_readlane_b32 s1, v90, 21
	s_and_b64 s[0:1], s[14:15], s[0:1]
	s_mov_b64 s[56:57], s[44:45]
	s_mov_b32 s25, s3
	s_mov_b32 s3, s93
	s_mov_b64 s[78:79], s[10:11]
	v_readlane_b32 s11, v90, 22
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB1_83
; %bb.165:                              ; %.lr.ph555.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_lshl_b32 s0, s5, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s74
	s_sub_i32 s0, s0, s75
	s_lshl_b32 s1, s4, 2
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s4, s0, 6
	s_mov_b32 s5, s97
	s_branch .LBB1_167
.LBB1_166:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB1_167 Depth=2
	s_add_i32 s5, s5, -1
	s_add_i32 s4, s4, s30
	s_cmp_lg_u32 s5, 0
	s_cbranch_scc0 .LBB1_83
.LBB1_167:                              ; %.lr.ph555
                                        ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[36:37], s[40:41]
	flat_load_dwordx4 v[2:5], v[36:37]
	flat_load_dwordx4 v[6:9], v[36:37] offset:16
	flat_load_dwordx4 v[10:13], v[36:37] offset:32
	flat_load_dwordx4 v[14:17], v[36:37] offset:48
	s_abs_i32 s1, s4
	flat_load_dwordx2 v[36:37], v[36:37] offset:64
	s_mul_hi_u32 s7, s1, s99
	s_mul_i32 s12, s7, s96
	s_ashr_i32 s0, s4, 31
	s_sub_i32 s1, s1, s12
	s_xor_b32 s0, s0, s98
	s_add_i32 s13, s7, 1
	s_sub_i32 s12, s1, s96
	s_cmp_ge_u32 s1, s96
	s_cselect_b32 s7, s13, s7
	s_cselect_b32 s1, s12, s1
	s_add_i32 s12, s7, 1
	s_cmp_ge_u32 s1, s96
	s_cselect_b32 s1, s12, s7
	s_xor_b32 s1, s1, s0
	s_sub_i32 s7, s1, s0
	s_mul_i32 s1, s58, s7
	s_add_i32 s1, s4, s1
	s_mul_i32 s0, s7, s33
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s0, s1, s0
	s_add_i32 s0, s4, s0
	s_xor_b32 s0, s0, s1
	s_xor_b32 s12, s1, s94
	s_mul_hi_u32 s1, s0, s95
	s_mul_i32 s13, s1, s91
	s_sub_i32 s0, s0, s13
	s_add_i32 s16, s1, 1
	s_sub_i32 s13, s0, s91
	s_cmp_ge_u32 s0, s91
	s_cselect_b32 s1, s16, s1
	s_cselect_b32 s0, s13, s0
	s_add_i32 s13, s1, 1
	s_cmp_ge_u32 s0, s91
	s_cselect_b32 s0, s13, s1
	s_xor_b32 s0, s0, s12
	s_mul_i32 s6, s28, s24
	s_sub_i32 s0, s0, s12
	s_add_i32 s0, s0, s6
	s_mul_i32 s0, s0, s29
	s_add_i32 s0, s0, s49
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	s_addc_u32 s1, s39, s1
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmp_eq_u32 s7, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 1
	v_mov_b32_e32 v26, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v4, 0, v4, vcc
	v_cndmask_b32_e32 v5, 0, v5, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 2
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 3
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_cndmask_b32_e32 v5, v5, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 4
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_cndmask_b32_e32 v4, v4, v10, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 5
	v_cndmask_b32_e32 v4, v4, v12, vcc
	v_cndmask_b32_e32 v5, v5, v13, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 6
	v_cndmask_b32_e32 v5, v5, v15, vcc
	v_cndmask_b32_e32 v4, v4, v14, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 7
	v_sub_co_u32_e64 v2, s[0:1], s0, v2
	v_cndmask_b32_e32 v4, v4, v16, vcc
	v_cndmask_b32_e32 v5, v5, v17, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v3, s[0:1], v26, v3, s[0:1]
	v_cndmask_b32_e32 v5, v5, v37, vcc
	v_cndmask_b32_e32 v4, v4, v36, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_cmp_lg_u32 s7, s24
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB1_169
; %bb.168:                              ; %Flow879
                                        ;   in Loop: Header=BB1_167 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB1_166
	s_branch .LBB1_170
.LBB1_169:                              ;   in Loop: Header=BB1_167 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB1_166
.LBB1_170:                              ;   in Loop: Header=BB1_167 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB1_166
.LBB1_171:                              ; %.critedge262
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 91
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 92
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals,comdat
.Lfunc_end1:
	.size	_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals, .Lfunc_end1-_Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.num_vgpr, 91
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10820
; TotalNumSgprs: 106
; NumVgprs: 91
; NumAgprs: 0
; TotalNumVgprs: 91
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 91
; AccumOffset: 92
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 22
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[56:57], s[0:1], 0x60
	s_load_dwordx8 s[36:43], s[0:1], 0x100
	s_load_dwordx8 s[44:51], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
                                        ; implicit-def: $vgpr164 : SGPR spill to VGPR lane
	s_load_dwordx2 s[58:59], s[0:1], 0xf8
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b32 s88, s2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s89, s37, 31
	s_lshr_b32 s3, s89, 29
	v_writelane_b32 v164, s4, 0
	s_add_i32 s3, s37, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v164, s5, 1
	v_writelane_b32 v164, s6, 2
	v_writelane_b32 v164, s7, 3
	s_cmp_ge_i32 s2, s43
	s_mov_b64 s[4:5], -1
	s_cbranch_scc0 .LBB2_73
; %bb.1:
	s_sub_i32 s16, s88, s43
	s_mov_b32 s17, 0
	s_lshl_b64 s[4:5], s[16:17], 2
	s_add_u32 s6, s50, s4
	s_addc_u32 s7, s51, s5
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB2_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB2_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit341
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_eq_u64 s[10:11], 0
	s_cbranch_scc1 .LBB2_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB2_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit344
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_branch .LBB2_8
.LBB2_7:
	s_mov_b64 s[6:7], -1
.LBB2_8:                                ; %Flow1547
	s_mov_b64 s[2:3], exec
	v_writelane_b32 v164, s2, 37
	s_and_b64 s[6:7], s[2:3], s[6:7]
	s_nop 0
	v_writelane_b32 v164, s3, 38
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB2_72
; %bb.9:                                ; %.critedge
	s_mul_i32 s2, s41, s40
	s_cmp_ge_i32 s16, s2
	s_cbranch_scc1 .LBB2_72
; %bb.10:                               ; %.lr.ph
	s_mul_hi_i32 s13, s33, s38
	s_mul_i32 s12, s33, s38
	s_ashr_i32 s23, s38, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_add_u32 s24, s56, s6
	s_addc_u32 s25, s57, s7
	s_add_u32 s26, s24, s6
	s_addc_u32 s27, s25, s7
	s_add_u32 s28, s26, s6
	s_addc_u32 s29, s27, s7
	s_add_u32 s30, s28, s6
	s_addc_u32 s31, s29, s7
	s_add_u32 s34, s30, s6
	s_addc_u32 s35, s31, s7
	s_add_u32 s60, s34, s6
	s_addc_u32 s61, s35, s7
	s_add_u32 s62, s60, s6
	s_mov_b32 s6, s2
	s_load_dwordx2 s[2:3], s[0:1], 0x120
	s_load_dwordx2 s[80:81], s[0:1], 0x90
	s_addc_u32 s63, s61, s7
	s_mov_b32 s22, s38
	v_mbcnt_lo_u32_b32 v4, -1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s3, 0
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s17, s42, 5
	s_add_i32 s2, s2, 64
	s_cmp_lg_u32 s36, 0
	v_writelane_b32 v164, s2, 4
	s_mov_b32 s2, s6
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 42
	s_cmp_lg_u32 s36, 1
	v_mbcnt_hi_u32_b32 v4, -1, v4
	v_writelane_b32 v164, s7, 43
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 44
	s_cmp_lg_u32 s36, 2
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v164, s7, 45
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 46
	s_cmp_lg_u32 s36, 3
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v164, s7, 47
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 48
	s_cmp_lg_u32 s36, 4
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v164, s7, 49
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 8
	s_cmp_lg_u32 s36, 5
	v_mul_lo_u32 v64, s40, v0
	v_writelane_b32 v164, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 30
	s_cmp_lg_u32 s36, 6
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v164, s7, 31
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v164, s6, 27
	s_cmp_lg_u32 s36, 7
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v164, s7, 28
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s53, s41
	v_cvt_f32_u32_e32 v2, s53
	s_sub_i32 s3, 0, s53
	s_ashr_i32 s54, s41, 31
	s_lshl_b64 s[82:83], s[22:23], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v164, s6, 11
	v_lshlrev_b32_e32 v65, 3, v0
	v_mov_b32_e32 v3, v5
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v164, s7, 12
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	s_mov_b64 s[98:99], 0
	v_readfirstlane_b32 s20, v2
	s_mul_i32 s3, s3, s20
	s_mul_hi_u32 s3, s20, s3
	s_add_i32 s55, s20, s3
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v164, s20, 13
	s_mul_i32 s3, s13, 14
	v_lshrrev_b32_e32 v2, 5, v0
	v_writelane_b32 v164, s21, 14
	s_mul_hi_u32 s20, s12, 14
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 14
	s_add_u32 s3, s56, s3
	s_addc_u32 s20, s57, s20
	s_add_u32 s64, s3, 2
	s_addc_u32 s65, s20, 0
	s_lshl_b64 s[20:21], s[12:13], 2
	v_writelane_b32 v164, s64, 18
	s_add_u32 s20, s56, s20
	s_addc_u32 s21, s57, s21
	v_writelane_b32 v164, s65, 19
	v_writelane_b32 v164, s20, 20
	s_mul_i32 s3, s13, 12
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v164, s21, 21
	s_mul_hi_u32 s20, s12, 12
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 12
	s_add_u32 s3, s56, s3
	s_addc_u32 s20, s57, s20
	s_add_u32 s90, s3, 2
	s_addc_u32 s91, s20, 0
	s_mul_i32 s3, s13, 6
	s_mul_hi_u32 s20, s12, 6
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 6
	s_add_u32 s92, s56, s3
	s_addc_u32 s93, s57, s20
	s_mul_i32 s3, s13, 10
	s_mul_hi_u32 s20, s12, 10
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 10
	s_add_u32 s3, s56, s3
	s_addc_u32 s20, s57, s20
	s_add_u32 s94, s3, 2
	s_addc_u32 s95, s20, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	s_add_u32 s96, s56, s12
	s_addc_u32 s97, s57, s13
	s_xor_b64 s[64:65], s[14:15], -1
	s_movk_i32 s52, 0x7fff
	s_mov_b32 s3, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	s_branch .LBB2_13
.LBB2_11:                               ; %Flow1532
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s16, s43
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s2
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB2_12:                               ; %Flow1544
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB2_72
.LBB2_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_17 Depth 2
                                        ;     Child Loop BB2_29 Depth 2
                                        ;       Child Loop BB2_33 Depth 3
	s_abs_i32 s13, s16
	s_mul_hi_u32 s14, s13, s55
	s_mul_i32 s15, s14, s53
	s_ashr_i32 s12, s16, 31
	s_sub_i32 s13, s13, s15
	s_xor_b32 s12, s12, s54
	s_add_i32 s15, s14, 1
	s_sub_i32 s20, s13, s53
	s_cmp_ge_u32 s13, s53
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s13, s20, s13
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s13, s53
	s_cselect_b32 s13, s15, s14
	s_xor_b32 s13, s13, s12
	s_sub_i32 s86, s13, s12
	s_mul_i32 s12, s86, s41
	s_sub_i32 s87, s16, s12
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB2_21
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=1
	v_add_u32_e32 v4, s86, v64
	v_mul_lo_u32 v4, v4, s41
	v_add_u32_e32 v6, s87, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[46:47]
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_21
; %bb.15:                               ; %.lr.ph.i.i.i346.preheader
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB2_17
.LBB2_16:                               ; %Flow1540
                                        ;   in Loop: Header=BB2_17 Depth=2
	s_and_b64 s[20:21], exec, s[68:69]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[66:67], exec
	s_and_b64 s[66:67], s[72:73], exec
	s_or_b64 s[66:67], s[20:21], s[66:67]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB2_19
.LBB2_17:                               ; %.lr.ph.i.i.i346
                                        ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s70, s70, 1
	s_addc_u32 s71, s71, 0
	v_mov_b64_e32 v[8:9], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[70:71], v[8:9]
	s_mov_b64 s[72:73], -1
	s_or_b64 s[68:69], s[68:69], exec
	s_cbranch_vccnz .LBB2_16
; %bb.18:                               ;   in Loop: Header=BB2_17 Depth=2
	s_sleep 4
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[20:21], s[68:69], exec
	s_mov_b64 s[72:73], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[68:69], vcc, exec
	s_or_b64 s[68:69], s[20:21], s[68:69]
	s_branch .LBB2_16
.LBB2_19:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[66:67]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB2_21
; %bb.20:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_load_dwordx4 s[68:71], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[70:71]
	flat_atomic_or v[6:7], v66
.LBB2_21:                               ; %.critedge662
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[18:19]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB2_25
; %bb.22:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b64 s[14:15], s[8:9]
	s_cbranch_execz .LBB2_24
; %bb.23:                               ;   in Loop: Header=BB2_13 Depth=1
	s_load_dwordx4 s[68:71], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[70:71]
	flat_load_dword v4, v[6:7] sc1
.LBB2_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit352
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB2_25:                               ; %Flow1543
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[14:15]
	s_cbranch_execz .LBB2_12
; %bb.26:                               ; %.critedge680
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB2_39
; %bb.27:                               ; %.lr.ph.i353.preheader
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_mul_i32 s68, s86, s42
	s_lshl_b32 s70, s87, 8
	s_ashr_i32 s69, s68, 31
	s_ashr_i32 s71, s70, 31
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[68:69]
	v_mov_b64_e32 v[8:9], s[70:71]
	v_mad_u64_u32 v[8:9], s[14:15], s22, v6, v[8:9]
	v_mul_lo_u32 v4, s22, v7
	v_mul_lo_u32 v6, s23, v6
	v_add3_u32 v9, v6, v9, v4
	v_readlane_b32 s14, v164, 13
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s15, v164, 14
	v_lshl_add_u64 v[6:7], s[56:57], 0, v[22:23]
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v164, 18
	v_readlane_b32 s15, v164, 19
	v_lshl_add_u64 v[16:17], s[90:91], 0, v[22:23]
	v_lshl_add_u64 v[18:19], s[92:93], 0, v[22:23]
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v164, 20
	v_readlane_b32 s15, v164, 21
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	s_mov_b64 s[72:73], 0
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	s_branch .LBB2_29
.LBB2_28:                               ; %.critedge.i356
                                        ;   in Loop: Header=BB2_29 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s17, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[82:83]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[82:83]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[82:83]
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[82:83]
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[82:83]
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[82:83]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[82:83]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[82:83]
	s_or_b64 s[72:73], vcc, s[72:73]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	s_andn2_b64 exec, exec, s[72:73]
	s_cbranch_execz .LBB2_39
.LBB2_29:                               ; %.lr.ph.i353
                                        ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_33 Depth 3
	v_lshlrev_b32_e32 v4, 3, v69
	v_and_b32_e32 v4, 0xf8, v4
	v_or_b32_e32 v24, s70, v4
	v_mov_b32_e32 v25, s71
	v_lshl_add_u64 v[26:27], v[24:25], 0, 8
	v_cmp_lt_u64_e32 vcc, s[22:23], v[26:27]
	s_or_b64 s[14:15], s[64:65], vcc
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_xor_b64 s[74:75], exec, s[20:21]
	s_cbranch_execz .LBB2_37
; %bb.30:                               ; %.preheader.i355.preheader
                                        ;   in Loop: Header=BB2_29 Depth=2
	v_and_b32_e32 v4, 0xf8, v68
	v_lshl_add_u64 v[24:25], s[70:71], 0, v[4:5]
	v_lshlrev_b32_e32 v4, 1, v68
	v_and_b32_e32 v4, 0x1f0, v4
	v_lshl_add_u64 v[26:27], v[6:7], 0, v[4:5]
	v_lshl_add_u64 v[28:29], v[8:9], 0, v[4:5]
	v_lshl_add_u64 v[30:31], v[10:11], 0, v[4:5]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[4:5]
	v_lshl_add_u64 v[34:35], v[14:15], 0, v[4:5]
	v_lshl_add_u64 v[36:37], v[16:17], 0, v[4:5]
	v_lshl_add_u64 v[38:39], v[18:19], 0, v[4:5]
	v_lshl_add_u64 v[40:41], v[20:21], 0, v[4:5]
	v_lshl_add_u64 v[42:43], v[22:23], 0, v[4:5]
	s_mov_b64 s[76:77], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr78_sgpr79
	s_branch .LBB2_33
.LBB2_31:                               ; %Flow1534
                                        ;   in Loop: Header=BB2_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[78:79], s[78:79], s[20:21]
.LBB2_32:                               ; %Flow1533
                                        ;   in Loop: Header=BB2_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[78:79]
	s_or_b64 s[76:77], s[14:15], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB2_36
.LBB2_33:                               ; %.preheader.i355
                                        ;   Parent Loop BB2_13 Depth=1
                                        ;     Parent Loop BB2_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[22:23], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_32
; %bb.34:                               ; %.preheader.i355.1
                                        ;   in Loop: Header=BB2_33 Depth=3
	v_lshl_add_u64 v[48:49], v[26:27], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[30:31], 0, v[44:45]
	global_load_ushort v4, v[48:49], off
	global_load_ushort v72, v[50:51], off
	v_lshl_add_u64 v[52:53], v[34:35], 0, v[44:45]
	global_load_ushort v73, v[52:53], off
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[44:45]
	global_load_ushort v74, v[54:55], off
	v_lshl_add_u64 v[56:57], v[42:43], 0, v[44:45]
	global_load_ushort v75, v[56:57], off
	v_lshl_add_u64 v[58:59], v[40:41], 0, v[44:45]
	global_load_ushort v76, v[58:59], off offset:-2
	v_lshl_add_u64 v[60:61], v[36:37], 0, v[44:45]
	global_load_ushort v77, v[60:61], off offset:-2
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[44:45]
	global_load_ushort v78, v[62:63], off offset:-2
	v_lshl_add_u64 v[70:71], v[24:25], 0, 1
	v_cmp_gt_u64_e32 vcc, s[22:23], v[70:71]
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[44:45]
	s_mov_b64 s[20:21], -1
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v70, 16, v72
	v_add_f32_e32 v4, v4, v70
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v71, 16, v73
	v_add_f32_e32 v4, v4, v71
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v72, 16, v74
	v_add_f32_e32 v4, v4, v72
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v73, 16, v75
	v_add_f32_e32 v4, v4, v73
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v74, 16, v76
	v_add_f32_e32 v4, v4, v74
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v75, 16, v77
	v_add_f32_e32 v4, v4, v75
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v76, 16, v78
	v_add_f32_e32 v4, v4, v76
	v_bfe_u32 v70, v4, 16, 1
	v_add3_u32 v4, v4, v70, s52
	global_store_short_d16_hi v[46:47], v4, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB2_31
; %bb.35:                               ;   in Loop: Header=BB2_33 Depth=3
	global_load_ushort v4, v[50:51], off offset:2
	s_nop 0
	global_load_ushort v48, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v49, v[52:53], off offset:2
	global_load_ushort v50, v[54:55], off offset:2
	global_load_ushort v51, v[56:57], off offset:2
	s_nop 0
	global_load_ushort v52, v[58:59], off
	global_load_ushort v53, v[60:61], off
	global_load_ushort v54, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[24:25], v[24:25], 0, 2
	s_orn2_b64 s[20:21], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v48, 16, v48
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v4, v48, v4
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v4, v4, v49
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v4, v4, v50
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v4, v4, v51
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v53, 16, v53
	v_add_f32_e32 v4, v4, v52
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v4, v4, v53
	v_add_f32_e32 v4, v4, v54
	v_bfe_u32 v48, v4, 16, 1
	v_add3_u32 v4, v4, v48, s52
	global_store_short_d16_hi v[46:47], v4, off
	s_branch .LBB2_31
.LBB2_36:                               ; %Flow1535
                                        ;   in Loop: Header=BB2_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB2_37:                               ; %Flow1536
                                        ;   in Loop: Header=BB2_29 Depth=2
	s_andn2_saveexec_b64 vcc, s[74:75]
	s_cbranch_execz .LBB2_28
; %bb.38:                               ;   in Loop: Header=BB2_29 Depth=2
	v_lshrrev_b32_e32 v4, 5, v69
	v_lshl_add_u64 v[26:27], v[4:5], 0, s[68:69]
	v_mul_lo_u32 v4, v26, s23
	v_mul_lo_u32 v27, v27, s22
	v_mad_u64_u32 v[24:25], s[14:15], v26, s22, v[24:25]
	v_add3_u32 v25, v27, v25, v4
	v_lshlrev_b64 v[56:57], 1, v[24:25]
	v_lshl_add_u64 v[24:25], s[56:57], 0, v[56:57]
	v_lshl_add_u64 v[28:29], s[24:25], 0, v[56:57]
	global_load_dwordx4 v[24:27], v[24:25], off
	v_lshl_add_u64 v[32:33], s[26:27], 0, v[56:57]
	global_load_dwordx4 v[28:31], v[28:29], off
	v_lshl_add_u64 v[36:37], s[28:29], 0, v[56:57]
	global_load_dwordx4 v[32:35], v[32:33], off
	v_lshl_add_u64 v[40:41], s[30:31], 0, v[56:57]
	global_load_dwordx4 v[36:39], v[36:37], off
	v_lshl_add_u64 v[44:45], s[34:35], 0, v[56:57]
	global_load_dwordx4 v[40:43], v[40:41], off
	v_lshl_add_u64 v[48:49], s[60:61], 0, v[56:57]
	global_load_dwordx4 v[44:47], v[44:45], off
	v_lshl_add_u64 v[52:53], s[62:63], 0, v[56:57]
	global_load_dwordx4 v[48:51], v[48:49], off
	v_lshl_add_u64 v[56:57], s[80:81], 0, v[56:57]
	global_load_dwordx4 v[52:55], v[52:53], off
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v59, 0xffff0000, v24
	v_and_b32_e32 v61, 0xffff0000, v25
	v_lshlrev_b32_e32 v58, 16, v24
	v_lshlrev_b32_e32 v60, 16, v25
	s_waitcnt vmcnt(6)
	v_and_b32_e32 v25, 0xffff0000, v28
	v_lshlrev_b32_e32 v24, 16, v28
	v_and_b32_e32 v63, 0xffff0000, v26
	v_and_b32_e32 v71, 0xffff0000, v27
	v_lshlrev_b32_e32 v62, 16, v26
	v_lshlrev_b32_e32 v70, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v29
	v_lshlrev_b32_e32 v26, 16, v29
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v29, 0xffff0000, v32
	v_lshlrev_b32_e32 v28, 16, v32
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
	v_and_b32_e32 v73, 0xffff0000, v30
	v_and_b32_e32 v75, 0xffff0000, v31
	v_lshlrev_b32_e32 v72, 16, v30
	v_lshlrev_b32_e32 v74, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v33
	v_lshlrev_b32_e32 v30, 16, v33
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v33, 0xffff0000, v36
	v_lshlrev_b32_e32 v32, 16, v36
	v_pk_add_f32 v[26:27], v[26:27], v[60:61]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_and_b32_e32 v77, 0xffff0000, v34
	v_and_b32_e32 v79, 0xffff0000, v35
	v_lshlrev_b32_e32 v76, 16, v34
	v_lshlrev_b32_e32 v78, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v37
	v_lshlrev_b32_e32 v34, 16, v37
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v36, 16, v40
	v_and_b32_e32 v37, 0xffff0000, v40
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[32:33]
	v_and_b32_e32 v81, 0xffff0000, v38
	v_and_b32_e32 v83, 0xffff0000, v39
	v_lshlrev_b32_e32 v80, 16, v38
	v_lshlrev_b32_e32 v82, 16, v39
	v_lshlrev_b32_e32 v38, 16, v41
	v_and_b32_e32 v39, 0xffff0000, v41
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v84, 16, v44
	v_and_b32_e32 v85, 0xffff0000, v44
	v_pk_add_f32 v[26:27], v[26:27], v[34:35]
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	v_lshlrev_b32_e32 v40, 16, v45
	v_and_b32_e32 v41, 0xffff0000, v45
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v44, 16, v48
	v_and_b32_e32 v45, 0xffff0000, v48
	v_pk_add_f32 v[26:27], v[26:27], v[38:39]
	v_pk_add_f32 v[24:25], v[24:25], v[84:85]
	v_lshlrev_b32_e32 v86, 16, v49
	v_and_b32_e32 v87, 0xffff0000, v49
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v88, 16, v52
	v_and_b32_e32 v89, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[88:89]
	v_pk_add_f32 v[26:27], v[26:27], v[48:49]
	v_bfe_u32 v29, v25, 16, 1
	v_bfe_u32 v30, v24, 16, 1
	v_pk_add_f32 v[52:53], v[72:73], v[62:63]
	v_bfe_u32 v4, v27, 16, 1
	v_bfe_u32 v28, v26, 16, 1
	v_add3_u32 v32, v24, v30, s52
	v_add3_u32 v33, v25, v29, s52
	v_pk_add_f32 v[24:25], v[74:75], v[70:71]
	v_add3_u32 v34, v26, v28, s52
	v_add3_u32 v4, v27, v4, s52
	v_pk_add_f32 v[24:25], v[24:25], v[78:79]
	v_pk_add_f32 v[26:27], v[52:53], v[76:77]
	v_pk_add_f32 v[24:25], v[24:25], v[82:83]
	v_pk_add_f32 v[26:27], v[26:27], v[80:81]
	v_lshlrev_b32_e32 v28, 16, v42
	v_lshlrev_b32_e32 v30, 16, v43
	v_and_b32_e32 v29, 0xffff0000, v42
	v_and_b32_e32 v31, 0xffff0000, v43
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v47
	v_lshlrev_b32_e32 v30, 16, v46
	v_and_b32_e32 v29, 0xffff0000, v47
	v_and_b32_e32 v31, 0xffff0000, v46
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v50
	v_lshlrev_b32_e32 v30, 16, v51
	v_and_b32_e32 v29, 0xffff0000, v50
	v_and_b32_e32 v31, 0xffff0000, v51
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v55
	v_lshlrev_b32_e32 v30, 16, v54
	v_and_b32_e32 v29, 0xffff0000, v55
	v_and_b32_e32 v31, 0xffff0000, v54
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_bfe_u32 v30, v27, 16, 1
	v_bfe_u32 v28, v25, 16, 1
	v_bfe_u32 v29, v24, 16, 1
	v_bfe_u32 v31, v26, 16, 1
	v_add3_u32 v26, v26, v31, s52
	v_add3_u32 v30, v27, v30, s52
	v_add3_u32 v24, v24, v29, s52
	v_add3_u32 v25, v25, v28, s52
	v_perm_b32 v27, v25, v24, s3
	v_perm_b32 v26, v30, v26, s3
	v_perm_b32 v25, v4, v34, s3
	v_perm_b32 v24, v33, v32, s3
	global_store_dwordx4 v[56:57], v[24:27], off
	s_branch .LBB2_28
.LBB2_39:                               ; %Flow1538
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB2_11
; %bb.40:                               ; %.preheader689
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s40, s36
	v_readlane_b32 s13, v164, 4
	s_add_i32 s12, s86, s12
	s_add_i32 s13, s13, s87
	s_mul_i32 s12, s12, s41
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s46, s12
	s_addc_u32 s15, s47, s13
	v_readlane_b32 s12, v164, 42
	v_readlane_b32 s13, v164, 43
	s_and_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v4, s15
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v6, s[12:13], s14, v6
	s_nop 1
	v_subb_co_u32_e64 v7, s[12:13], v4, v7, s[12:13]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v7, s[12:13]
	v_cndmask_b32_e64 v6, 0, v6, s[12:13]
	s_cbranch_vccz .LBB2_71
; %bb.41:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB2_43
.LBB2_42:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_43:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:16
	v_readlane_b32 s12, v164, 44
	v_readlane_b32 s13, v164, 45
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_45
; %bb.44:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_45:                               ; %Flow1530
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_47
; %bb.46:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_47:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:24
	v_readlane_b32 s12, v164, 46
	v_readlane_b32 s13, v164, 47
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_49
; %bb.48:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_49:                               ; %Flow1529
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_51
; %bb.50:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_51:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:32
	v_readlane_b32 s12, v164, 48
	v_readlane_b32 s13, v164, 49
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_53
; %bb.52:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_53:                               ; %Flow1528
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_55
; %bb.54:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_55:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:40
	v_readlane_b32 s12, v164, 8
	v_readlane_b32 s13, v164, 9
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_57
; %bb.56:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_57:                               ; %Flow1527
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_59
; %bb.58:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_59:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:48
	v_readlane_b32 s12, v164, 30
	v_readlane_b32 s13, v164, 31
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_61
; %bb.60:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_61:                               ; %Flow1526
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_63
; %bb.62:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_63:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:56
	v_readlane_b32 s12, v164, 27
	v_readlane_b32 s13, v164, 28
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_65
; %bb.64:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_65:                               ; %Flow1525
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_67
; %bb.66:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB2_67:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:64
	v_readlane_b32 s12, v164, 11
	v_readlane_b32 s13, v164, 12
	v_mov_b32_e32 v4, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s14, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB2_69
; %bb.68:                               ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB2_69:                               ; %Flow
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB2_11
; %bb.70:                               ;   in Loop: Header=BB2_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB2_11
.LBB2_71:                               ;   in Loop: Header=BB2_13 Depth=1
	s_branch .LBB2_42
.LBB2_72:                               ; %Flow1548
	v_readlane_b32 s2, v164, 37
	v_readlane_b32 s3, v164, 38
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB2_73:                               ; %Flow1590
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB2_322
; %bb.74:
	s_mov_b32 s4, s88
	s_ashr_i32 s5, s88, 31
	s_mov_b32 s2, s88
	v_writelane_b32 v164, s2, 4
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s50, s4
	v_writelane_b32 v164, s3, 5
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_addc_u32 s5, s51, s5
	s_nop 0
	v_writelane_b32 v164, s2, 6
	s_nop 1
	v_writelane_b32 v164, s3, 7
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB2_76
; %bb.75:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB2_76:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[6:7], 0
	s_cselect_b64 s[2:3], -1, 0
	v_writelane_b32 v164, s2, 8
	s_cmp_eq_u64 s[6:7], 0
	s_nop 0
	v_writelane_b32 v164, s3, 9
	s_cbranch_scc1 .LBB2_80
; %bb.77:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_79
; %bb.78:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB2_79:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB2_81
	s_branch .LBB2_322
.LBB2_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB2_322
.LBB2_81:                               ; %.critedge660
	s_lshr_b32 s2, s89, 25
	s_add_i32 s2, s37, s2
	s_ashr_i32 s51, s2, 7
	s_mul_i32 s4, s41, s51
	v_readlane_b32 s2, v164, 4
	s_cmp_ge_i32 s2, s4
	v_readlane_b32 s3, v164, 5
	v_writelane_b32 v164, s4, 10
	s_cbranch_scc1 .LBB2_322
; %bb.82:                               ; %.lr.ph760
	s_mov_b64 s[2:3], src_shared_base
	s_cmp_lg_u32 0, -1
	s_load_dwordx2 s[60:61], s[0:1], 0x80
	s_load_dwordx4 s[4:7], s[0:1], 0x70
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[24:25], s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[10:11], s[0:1], 0x50
	s_cselect_b32 s2, s3, 0
	s_cselect_b32 s3, 0, 0
	s_and_b32 s0, s3, 15
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s3, -16
	s_add_u32 s5, s5, 16
	s_mov_b32 s1, 0
	s_addc_u32 s7, s2, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s63, s3, s5
	s_cselect_b32 s0, s2, s7
	s_add_u32 s2, s63, 0x4000
	s_addc_u32 s0, s0, 0
	s_and_b32 s3, s2, -16
	s_and_b32 s0, s2, 15
	s_add_u32 s3, s3, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s67, s2, s3
	s_add_i32 s0, s39, 31
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	v_lshlrev_b32_e32 v118, 3, v0
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_and_b32_e32 v2, 24, v118
	v_lshrrev_b32_e32 v7, 2, v0
	s_ashr_i32 s37, s0, 5
	v_mad_u64_u32 v[4:5], s[0:1], v7, s24, v[2:3]
	v_mad_u64_u32 v[80:81], s[0:1], v7, s10, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v4
	s_mov_b32 s0, s10
	v_lshl_add_u64 v[78:79], v[4:5], 1, s[8:9]
	v_or_b32_e32 v4, 0x80, v7
	v_writelane_b32 v164, s0, 11
	v_lshlrev_b32_e32 v119, 4, v0
	s_add_i32 s82, s37, -1
	v_writelane_b32 v164, s1, 12
	v_mad_u64_u32 v[82:83], s[0:1], v4, s10, v[2:3]
	v_add_u32_e32 v2, s63, v119
	v_lshrrev_b32_e32 v4, 4, v2
	s_lshl_b32 s65, s41, 2
	v_and_b32_e32 v4, 56, v4
	v_or_b32_e32 v123, 8, v119
	v_xor_b32_e32 v122, v4, v2
	v_add_u32_e32 v2, s63, v123
	s_cmp_gt_i32 s39, 0
	v_lshrrev_b32_e32 v4, 4, v2
	s_cselect_b64 s[0:1], -1, 0
	v_and_b32_e32 v120, 48, v119
	v_and_b32_e32 v4, 56, v4
	v_writelane_b32 v164, s0, 13
	v_and_b32_e32 v121, 0x1fc0, v119
	v_xor_b32_e32 v124, v4, v2
	v_add_u32_e32 v2, s67, v120
	v_writelane_b32 v164, s1, 14
	s_ashr_i32 s1, s14, 31
	s_mov_b32 s0, s14
	v_add_u32_e32 v5, v2, v121
	s_lshl_b32 s2, s82, 5
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v8, 4, v5
	s_add_u32 s0, s46, s0
	v_add_u32_e32 v4, 8, v2
	v_and_b32_e32 v8, 56, v8
	s_addc_u32 s1, s47, s1
	v_xor_b32_e32 v125, v8, v5
	v_add_u32_e32 v5, v4, v121
	v_writelane_b32 v164, s0, 15
	v_lshrrev_b32_e32 v8, 4, v5
	v_or_b32_e32 v127, 0x2000, v121
	v_writelane_b32 v164, s1, 16
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v2, v2, v127
	s_min_i32 s20, s42, 32
	s_mul_i32 s21, s6, s4
	s_bfe_i64 s[72:73], s[60:61], 0x200000
	v_readlane_b32 s4, v164, 0
	v_xor_b32_e32 v126, v8, v5
	v_lshrrev_b32_e32 v5, 4, v2
	s_cmp_gt_i32 s42, 0
	v_readlane_b32 s5, v164, 1
	v_and_b32_e32 v5, 56, v5
	s_cselect_b64 s[74:75], -1, 0
	s_cmp_lg_u64 s[4:5], 0
	v_xor_b32_e32 v128, v5, v2
	v_add_u32_e32 v2, v4, v127
	s_cselect_b64 s[76:77], -1, 0
	s_max_i32 s0, s38, 1
	v_lshrrev_b32_e32 v4, 4, v2
	s_add_i32 s0, s0, -1
	v_and_b32_e32 v4, 56, v4
	s_cmp_lg_u32 s15, 0
	v_xor_b32_e32 v129, v4, v2
	v_and_b32_e32 v2, 15, v0
	s_cselect_b64 s[78:79], -1, 0
	s_abs_i32 s61, s65
	v_lshlrev_b32_e32 v5, 6, v2
	v_lshl_or_b32 v134, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s61
	s_or_b32 s1, s2, 16
	v_readlane_b32 s6, v164, 2
	v_readlane_b32 s7, v164, 3
	v_writelane_b32 v164, s0, 17
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s39, s2
	s_sub_i32 s2, s39, s1
	s_abs_i32 s39, s42
	v_lshl_or_b32 v131, v3, 12, v5
	v_cvt_f32_u32_e32 v3, s39
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_bfe_i32 s1, s41, 0x1001d
	v_rcp_iflag_f32_e32 v3, v3
	v_writelane_b32 v164, s1, 18
	v_readfirstlane_b32 s3, v2
	s_sub_i32 s1, 0, s61
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s1, s1, s3
	s_mul_hi_u32 s1, s3, s1
	s_add_i32 s1, s3, s1
	v_writelane_b32 v164, s1, 20
	s_sub_i32 s1, 0, s39
	v_readfirstlane_b32 s3, v2
	s_mul_i32 s1, s1, s3
	s_mul_hi_u32 s1, s3, s1
	s_add_i32 s31, s3, s1
	s_lshr_b32 s1, s31, 25
	s_mul_i32 s3, s1, s39
	s_sub_i32 s3, 0x80, s3
	s_lshl_b32 s80, s20, 5
	s_max_i32 s81, s37, 1
	s_ashr_i32 s30, s42, 31
	s_add_i32 s4, s1, 1
	s_sub_i32 s5, s3, s39
	s_cmp_ge_u32 s3, s39
	s_cselect_b32 s1, s4, s1
	s_cselect_b32 s3, s5, s3
	s_add_i32 s4, s1, 1
	s_cmp_ge_u32 s3, s39
	s_cselect_b32 s1, s4, s1
	s_abs_i32 s68, s33
	v_cvt_f32_u32_e32 v2, s68
	v_lshrrev_b32_e32 v6, 8, v0
	v_and_b32_e32 v4, 12, v7
	v_lshl_or_b32 v130, v6, 12, v5
	v_rcp_iflag_f32_e32 v2, v2
	v_or_b32_e32 v5, 1, v4
	v_or_b32_e32 v7, 2, v4
	v_or_b32_e32 v8, 3, v4
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_gt_i32_e64 s[4:5], s0, v4
	v_cmp_gt_i32_e64 s[6:7], s0, v5
	v_cmp_gt_i32_e64 s[8:9], s0, v7
	v_cmp_gt_i32_e64 s[10:11], s0, v8
	s_xor_b32 s0, s1, s30
	s_sub_i32 s69, s0, s30
	s_sub_i32 s0, 0, s68
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_ashr_i32 s83, s33, 31
	s_add_i32 s22, s1, s0
	v_lshrrev_b32_e32 v157, 5, v0
	s_cmp_gt_i32 s69, 0
	v_mad_i64_i32 v[2:3], s[0:1], v157, s60, 0
	s_cselect_b64 s[14:15], -1, 0
	v_readlane_b32 s0, v164, 6
	v_readlane_b32 s1, v164, 7
	v_writelane_b32 v164, s14, 22
	s_and_b64 s[0:1], s[0:1], s[14:15]
	v_and_b32_e32 v9, 63, v0
	v_writelane_b32 v164, s15, 23
	v_writelane_b32 v164, s0, 24
	v_and_b32_e32 v10, 31, v0
	v_lshlrev_b32_e32 v86, 4, v10
	v_writelane_b32 v164, s1, 25
	s_add_u32 s0, s12, 64
	v_writelane_b32 v164, s0, 26
	v_writelane_b32 v164, s12, 27
	s_addc_u32 s0, s13, 0
	s_mul_hi_u32 s1, s60, s20
	v_writelane_b32 v164, s13, 28
	v_writelane_b32 v164, s0, 29
	s_mul_i32 s0, s73, s20
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s20
	s_lshl_b64 s[84:85], s[0:1], 1
	s_mov_b32 s0, s24
	v_writelane_b32 v164, s0, 30
	v_mov_b32_e32 v87, 0
	v_lshl_add_u64 v[88:89], v[2:3], 1, v[86:87]
	v_writelane_b32 v164, s1, 31
	s_lshl_b32 s0, s24, 7
	v_writelane_b32 v164, s0, 32
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_lshl_or_b32 v135, v6, 6, v4
	v_writelane_b32 v164, s0, 33
	v_lshlrev_b32_e32 v6, 9, v157
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v164, s1, 34
	v_cmp_eq_u32_e64 s[0:1], 0, v9
	v_lshlrev_b32_e32 v155, 1, v4
	v_or_b32_e32 v3, v6, v86
	v_writelane_b32 v164, s0, 35
	v_lshlrev_b32_e32 v2, 2, v2
	v_ashrrev_i32_e32 v81, 31, v80
	v_writelane_b32 v164, s1, 36
	v_cmp_gt_u32_e64 s[0:1], s69, v0
	v_ashrrev_i32_e32 v83, 31, v82
	v_mul_lo_u32 v132, s42, v0
	v_writelane_b32 v164, s0, 37
	v_add_u32_e32 v133, -1, v1
	s_mul_i32 s21, s21, s36
	v_writelane_b32 v164, s1, 38
	v_writelane_b32 v164, s51, 39
	v_lshl_add_u32 v136, v134, 1, 0
	v_or_b32_e32 v137, 1, v135
	v_or_b32_e32 v138, 2, v135
	v_or_b32_e32 v139, 3, v135
	v_or_b32_e32 v140, 16, v134
	v_or_b32_e32 v141, 32, v134
	v_or_b32_e32 v142, 48, v134
	v_or_b32_e32 v143, 16, v135
	v_or_b32_e32 v144, 17, v135
	v_or_b32_e32 v145, 18, v135
	v_or_b32_e32 v146, 19, v135
	v_or_b32_e32 v147, 32, v135
	v_or_b32_e32 v148, 33, v135
	v_or_b32_e32 v149, 34, v135
	v_or_b32_e32 v150, 35, v135
	v_or_b32_e32 v151, 48, v135
	v_or_b32_e32 v152, 49, v135
	v_or_b32_e32 v153, 50, v135
	v_or_b32_e32 v154, 51, v135
	v_or_b32_e32 v156, 32, v155
	v_lshl_add_u64 v[84:85], v[78:79], 0, 64
	v_add_u32_e32 v158, 0, v6
	v_lshl_add_u32 v159, v0, 1, 0
	v_lshl_or_b32 v160, v10, 3, 1
	v_bfrev_b32_e32 v161, 64
	v_add_u32_e32 v162, 0, v3
	v_and_b32_e32 v163, 0x100, v2
	v_cmp_gt_i32_e64 s[12:13], s2, v4
	v_cmp_gt_i32_e64 s[14:15], s2, v5
	v_cmp_gt_i32_e64 s[16:17], s2, v7
	v_cmp_gt_i32_e64 s[18:19], s2, v8
	s_sub_i32 s23, 0, s33
	s_movk_i32 s62, 0x7fff
	s_mov_b64 s[86:87], 0
	v_cmp_gt_i32_e64 s[24:25], s80, v0
	s_lshl_b64 s[88:89], s[72:73], 5
	v_writelane_b32 v164, s65, 40
	v_writelane_b32 v164, s61, 41
	s_branch .LBB2_85
.LBB2_83:                               ; %Flow1551
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_readlane_b32 s0, v164, 4
	s_add_i32 s2, s0, s43
	v_readlane_b32 s1, v164, 5
	s_mov_b32 s0, s2
	v_writelane_b32 v164, s0, 4
	s_nop 1
	v_writelane_b32 v164, s1, 5
	s_nop 0
	v_readlane_b32 s0, v164, 10
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB2_84:                               ; %Flow1584
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[86:87], s[0:1], s[86:87]
	s_andn2_b64 exec, exec, s[86:87]
	s_cbranch_execz .LBB2_322
.LBB2_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_87 Depth 2
                                        ;     Child Loop BB2_102 Depth 2
                                        ;     Child Loop BB2_114 Depth 2
                                        ;       Child Loop BB2_118 Depth 3
                                        ;         Child Loop BB2_312 Depth 4
                                        ;         Child Loop BB2_272 Depth 4
                                        ;         Child Loop BB2_295 Depth 4
                                        ;         Child Loop BB2_302 Depth 4
                                        ;           Child Loop BB2_307 Depth 5
                                        ;     Child Loop BB2_318 Depth 2
	v_readlane_b32 s0, v164, 4
	v_readlane_b32 s1, v164, 5
	s_mov_b32 s28, s0
	s_ashr_i32 s0, s0, 31
	v_readlane_b32 s1, v164, 18
	s_xor_b32 s26, s0, s1
	s_abs_i32 s0, s28
	v_readlane_b32 s1, v164, 20
	s_mul_hi_u32 s1, s0, s1
	s_mul_i32 s2, s1, s61
	s_sub_i32 s0, s0, s2
	s_add_i32 s2, s1, 1
	s_sub_i32 s3, s0, s61
	s_cmp_ge_u32 s0, s61
	s_cselect_b32 s1, s2, s1
	s_cselect_b32 s0, s3, s0
	s_add_i32 s2, s1, 1
	s_cmp_ge_u32 s0, s61
	s_cselect_b32 s0, s2, s1
	s_xor_b32 s0, s0, s26
	v_writelane_b32 v164, s26, 42
	v_writelane_b32 v164, s0, 44
	s_sub_i32 s0, s0, s26
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s2, s51, s1
	s_min_i32 s3, s2, 4
	s_abs_i32 s26, s3
	v_cvt_f32_u32_e32 v2, s26
	s_mul_i32 s0, s0, s65
	v_writelane_b32 v164, s0, 46
	s_sub_i32 s0, s28, s0
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s28, 0, s26
	s_abs_i32 s27, s0
	s_xor_b32 s2, s0, s3
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_ashr_i32 s2, s2, 31
	v_mov_b32_e32 v5, v87
	v_mov_b32_e32 v4, v87
	v_readfirstlane_b32 s29, v2
	s_mul_i32 s28, s28, s29
	s_mul_hi_u32 s28, s29, s28
	s_add_i32 s29, s29, s28
	s_mul_hi_u32 s28, s27, s29
	s_mul_i32 s29, s28, s26
	s_sub_i32 s27, s27, s29
	s_add_i32 s29, s28, 1
	s_sub_i32 s34, s27, s26
	s_cmp_ge_u32 s27, s26
	s_cselect_b32 s28, s29, s28
	s_cselect_b32 s27, s34, s27
	s_add_i32 s29, s28, 1
	s_cmp_ge_u32 s27, s26
	s_cselect_b32 s26, s29, s28
	s_xor_b32 s50, s26, s2
	s_sub_i32 s66, s50, s2
	s_mul_i32 s3, s66, s3
	s_sub_i32 s0, s0, s3
	v_writelane_b32 v164, s3, 48
	s_add_i32 s0, s0, s1
	s_lshl_b32 s27, s0, 7
	v_readlane_b32 s0, v164, 30
	v_readlane_b32 s1, v164, 31
	s_mul_i32 s0, s27, s0
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[78:79]
	s_lshl_b32 s64, s66, 8
	v_readlane_b32 s0, v164, 11
	v_readlane_b32 s1, v164, 12
	s_mul_i32 s0, s64, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v164, 27
	v_readlane_b32 s29, v164, 28
	s_add_u32 s28, s28, s0
	;;#ASMSTART
	global_load_dwordx4 v[66:69], v[2:3], off

	;;#ASMEND
	s_addc_u32 s29, s29, s1
	v_lshl_add_u64 v[2:3], v[80:81], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[70:73], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[82:83], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[74:77], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v122, v[66:67]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v124, v[68:69]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v125, v[70:71]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v126, v[72:73]

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v128, v[74:75]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v129, v[76:77]

	;;#ASMEND
	v_readlane_b32 s28, v164, 13
	v_readlane_b32 s29, v164, 14
	s_andn2_b64 vcc, exec, s[28:29]
	v_mov_b32_e32 v3, v87
	v_mov_b32_e32 v2, v87
	v_mov_b32_e32 v9, v87
	v_mov_b32_e32 v8, v87
	v_mov_b32_e32 v7, v87
	v_mov_b32_e32 v6, v87
	v_mov_b32_e32 v13, v87
	v_mov_b32_e32 v12, v87
	v_mov_b32_e32 v11, v87
	v_mov_b32_e32 v10, v87
	v_mov_b32_e32 v17, v87
	v_mov_b32_e32 v16, v87
	v_mov_b32_e32 v15, v87
	v_mov_b32_e32 v14, v87
	v_mov_b32_e32 v21, v87
	v_mov_b32_e32 v20, v87
	v_mov_b32_e32 v19, v87
	v_mov_b32_e32 v18, v87
	v_mov_b32_e32 v25, v87
	v_mov_b32_e32 v24, v87
	v_mov_b32_e32 v23, v87
	v_mov_b32_e32 v22, v87
	v_mov_b32_e32 v29, v87
	v_mov_b32_e32 v28, v87
	v_mov_b32_e32 v27, v87
	v_mov_b32_e32 v26, v87
	v_mov_b32_e32 v33, v87
	v_mov_b32_e32 v32, v87
	v_mov_b32_e32 v31, v87
	v_mov_b32_e32 v30, v87
	v_mov_b32_e32 v37, v87
	v_mov_b32_e32 v36, v87
	v_mov_b32_e32 v35, v87
	v_mov_b32_e32 v34, v87
	v_mov_b32_e32 v41, v87
	v_mov_b32_e32 v40, v87
	v_mov_b32_e32 v39, v87
	v_mov_b32_e32 v38, v87
	v_mov_b32_e32 v45, v87
	v_mov_b32_e32 v44, v87
	v_mov_b32_e32 v43, v87
	v_mov_b32_e32 v42, v87
	v_mov_b32_e32 v49, v87
	v_mov_b32_e32 v48, v87
	v_mov_b32_e32 v47, v87
	v_mov_b32_e32 v46, v87
	v_mov_b32_e32 v53, v87
	v_mov_b32_e32 v52, v87
	v_mov_b32_e32 v51, v87
	v_mov_b32_e32 v50, v87
	v_mov_b32_e32 v57, v87
	v_mov_b32_e32 v56, v87
	v_mov_b32_e32 v55, v87
	v_mov_b32_e32 v54, v87
	v_mov_b32_e32 v61, v87
	v_mov_b32_e32 v60, v87
	v_mov_b32_e32 v59, v87
	v_mov_b32_e32 v58, v87
	v_mov_b32_e32 v65, v87
	v_mov_b32_e32 v64, v87
	v_mov_b32_e32 v63, v87
	v_mov_b32_e32 v62, v87
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB2_97
; %bb.86:                               ; %.lr.ph734.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s3, v164, 26
	s_add_u32 s28, s3, s0
	v_readlane_b32 s0, v164, 29
	s_addc_u32 s29, s0, s1
	v_readlane_b32 s0, v164, 44
	s_lshl_b32 s0, s0, 2
	v_readlane_b32 s34, v164, 4
	s_add_i32 s0, s34, s0
	v_readlane_b32 s1, v164, 46
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 48
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 42
	s_lshl_b32 s1, s1, 2
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 32
	s_mul_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v2, 0
	v_lshl_add_u64 v[90:91], s[0:1], 1, v[84:85]
	s_mov_b32 s0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_readlane_b32 s35, v164, 5
.LBB2_87:                               ; %.lr.ph734
                                        ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s3, s0, 1
	s_cmp_lt_i32 s3, s37
	s_cselect_b64 s[34:35], -1, 0
	s_cmp_ge_i32 s3, s37
	s_cbranch_scc1 .LBB2_89
; %bb.88:                               ;   in Loop: Header=BB2_87 Depth=2
	;;#ASMSTART
	global_load_dwordx4 v[66:69], v[90:91], off

	;;#ASMEND
	v_lshl_add_u64 v[70:71], v[80:81], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[70:73], v[70:71], off

	;;#ASMEND
	v_lshl_add_u64 v[74:75], v[82:83], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[74:77], v[74:75], off

	;;#ASMEND
.LBB2_89:                               ;   in Loop: Header=BB2_87 Depth=2
	s_and_b32 s1, s0, 1
	s_lshl_b32 s26, s1, 13
	s_add_i32 s26, s63, s26
	v_add_u32_e32 v108, s26, v130
	v_add_u32_e32 v92, v108, v155
	s_lshl_b32 s1, s1, 14
	v_lshrrev_b32_e32 v93, 4, v92
	s_add_i32 s1, s67, s1
	v_and_b32_e32 v93, 56, v93
	v_add_u32_e32 v86, s1, v131
	v_xor_b32_e32 v92, v93, v92
	;;#ASMSTART
	ds_read_b64 v[106:107], v92 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[104:105], v92 offset:0x400

	;;#ASMEND
	v_add_u32_e32 v94, v86, v155
	;;#ASMSTART
	ds_read_b64 v[100:101], v92 offset:0x800

	;;#ASMEND
	v_lshrrev_b32_e32 v95, 4, v94
	;;#ASMSTART
	ds_read_b64 v[92:93], v92 offset:0xc00

	;;#ASMEND
	v_and_b32_e32 v95, 56, v95
	v_xor_b32_e32 v94, v95, v94
	;;#ASMSTART
	ds_read_b64 v[102:103], v94 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v94 offset:0x400

	;;#ASMEND
	s_cmp_eq_u32 s82, s0
	;;#ASMSTART
	ds_read_b64 v[96:97], v94 offset:0x800

	;;#ASMEND
	s_cselect_b64 s[90:91], -1, 0
	s_cmp_lg_u32 s82, s0
	;;#ASMSTART
	ds_read_b64 v[94:95], v94 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB2_91
; %bb.90:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB2_87 Depth=2
	s_or_b64 s[0:1], s[10:11], s[8:9]
	s_or_b64 s[0:1], s[0:1], s[6:7]
	v_cndmask_b32_e64 v109, 0, v106, s[4:5]
	v_and_b32_e32 v110, 0xffff0000, v107
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v110, v110, v107, s[8:9]
	v_cndmask_b32_sdwa v106, v109, v106, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v107, v110, v107, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v109, 0, v104, s[4:5]
	v_and_b32_e32 v110, 0xffff0000, v105
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v110, v110, v105, s[8:9]
	v_cndmask_b32_sdwa v104, v109, v104, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v105, v110, v105, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v109, 0, v100, s[4:5]
	v_and_b32_e32 v110, 0xffff0000, v101
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v110, v110, v101, s[8:9]
	v_cndmask_b32_sdwa v100, v109, v100, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v101, v110, v101, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v109, 0, v92, s[4:5]
	v_and_b32_e32 v110, 0xffff0000, v93
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v110, v110, v93, s[8:9]
	v_cndmask_b32_sdwa v92, v109, v92, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v93, v110, v93, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB2_91:                               ;   in Loop: Header=BB2_87 Depth=2
	v_mfma_f32_16x16x16_bf16 v[62:65], v[106:107], v[102:103], v[62:65]
	v_add_u32_e32 v86, v86, v156
	v_lshrrev_b32_e32 v112, 4, v86
	v_and_b32_e32 v112, 56, v112
	v_mfma_f32_16x16x16_bf16 v[58:61], v[106:107], v[98:99], v[58:61]
	v_xor_b32_e32 v86, v112, v86
	s_andn2_b64 vcc, exec, s[90:91]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[106:107], v[96:97], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[106:107], v[94:95], v[50:53]
	v_add_u32_e32 v106, v108, v156
	v_lshrrev_b32_e32 v107, 4, v106
	v_and_b32_e32 v107, 56, v107
	v_mfma_f32_16x16x16_bf16 v[46:49], v[104:105], v[102:103], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[104:105], v[98:99], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[104:105], v[96:97], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[104:105], v[94:95], v[34:37]
	v_xor_b32_e32 v104, v107, v106
	;;#ASMSTART
	ds_read_b64 v[110:111], v104 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[108:109], v104 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[106:107], v104 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[104:105], v104 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[30:33], v[100:101], v[102:103], v[30:33]
	;;#ASMSTART
	ds_read_b64 v[112:113], v86 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[114:115], v86 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[116:117], v86 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[26:29], v[100:101], v[98:99], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[100:101], v[96:97], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[100:101], v[94:95], v[18:21]
	;;#ASMSTART
	ds_read_b64 v[100:101], v86 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[14:17], v[92:93], v[102:103], v[14:17]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[10:13], v[92:93], v[98:99], v[10:13]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[92:93], v[96:97], v[6:9]
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[2:5], v[92:93], v[94:95], v[2:5]
	s_cbranch_vccnz .LBB2_93
; %bb.92:                               ; %.loopexit.i.1
                                        ;   in Loop: Header=BB2_87 Depth=2
	s_or_b64 s[0:1], s[18:19], s[16:17]
	s_or_b64 s[0:1], s[0:1], s[14:15]
	v_cndmask_b32_e64 v86, 0, v110, s[12:13]
	v_and_b32_e32 v92, 0xffff0000, v111
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v92, v92, v111, s[16:17]
	v_cndmask_b32_sdwa v110, v86, v110, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v111, v92, v111, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v86, 0, v108, s[12:13]
	v_and_b32_e32 v92, 0xffff0000, v109
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v92, v92, v109, s[16:17]
	v_cndmask_b32_sdwa v108, v86, v108, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v109, v92, v109, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v86, 0, v106, s[12:13]
	v_and_b32_e32 v92, 0xffff0000, v107
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v92, v92, v107, s[16:17]
	v_cndmask_b32_sdwa v106, v86, v106, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v107, v92, v107, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v86, 0, v104, s[12:13]
	v_and_b32_e32 v92, 0xffff0000, v105
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v92, v92, v105, s[16:17]
	v_cndmask_b32_sdwa v104, v86, v104, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v105, v92, v105, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB2_93:                               ;   in Loop: Header=BB2_87 Depth=2
	v_mfma_f32_16x16x16_bf16 v[62:65], v[110:111], v[112:113], v[62:65]
	s_andn2_b64 vcc, exec, s[34:35]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[110:111], v[114:115], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[110:111], v[116:117], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[110:111], v[100:101], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[108:109], v[112:113], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[108:109], v[114:115], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[108:109], v[116:117], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[108:109], v[100:101], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[106:107], v[112:113], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[106:107], v[114:115], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[106:107], v[116:117], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[106:107], v[100:101], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[104:105], v[112:113], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[104:105], v[114:115], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[104:105], v[116:117], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[104:105], v[100:101], v[2:5]
	s_cbranch_vccnz .LBB2_95
; %bb.94:                               ;   in Loop: Header=BB2_87 Depth=2
	s_and_b32 s0, s3, 1
	s_lshl_b32 s1, s0, 13
	s_add_i32 s1, s63, s1
	v_add_u32_e32 v86, s1, v119
	v_lshrrev_b32_e32 v92, 4, v86
	v_and_b32_e32 v92, 56, v92
	v_xor_b32_e32 v86, v92, v86
	v_add_u32_e32 v92, s1, v123
	s_lshl_b32 s0, s0, 14
	v_lshrrev_b32_e32 v93, 4, v92
	s_add_i32 s0, s67, s0
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v93, 56, v93
	;;#ASMSTART
	ds_write_b64 v86, v[66:67]

	;;#ASMEND
	v_add_u32_e32 v86, s0, v120
	v_xor_b32_e32 v92, v93, v92
	v_add_u32_e32 v93, v86, v121
	v_lshrrev_b32_e32 v94, 4, v93
	s_or_b32 s0, s0, 8
	v_and_b32_e32 v94, 56, v94
	;;#ASMSTART
	ds_write_b64 v92, v[68:69]

	;;#ASMEND
	v_add_u32_e32 v92, s0, v120
	v_xor_b32_e32 v93, v94, v93
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v93, v[70:71]

	;;#ASMEND
	v_add_u32_e32 v93, v92, v121
	v_lshrrev_b32_e32 v94, 4, v93
	v_and_b32_e32 v94, 56, v94
	v_xor_b32_e32 v93, v94, v93
	v_add_u32_e32 v86, v86, v127
	;;#ASMSTART
	ds_write_b64 v93, v[72:73]

	;;#ASMEND
	v_lshrrev_b32_e32 v93, 4, v86
	v_and_b32_e32 v93, 56, v93
	v_xor_b32_e32 v86, v93, v86
	;;#ASMSTART
	ds_write_b64 v86, v[74:75]

	;;#ASMEND
	v_add_u32_e32 v86, v92, v127
	v_lshrrev_b32_e32 v92, 4, v86
	v_and_b32_e32 v92, 56, v92
	v_xor_b32_e32 v86, v92, v86
	;;#ASMSTART
	ds_write_b64 v86, v[76:77]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB2_95:                               ;   in Loop: Header=BB2_87 Depth=2
	s_add_u32 s28, s28, 64
	s_addc_u32 s29, s29, 0
	s_cmp_eq_u32 s81, s3
	v_lshl_add_u64 v[90:91], v[90:91], 0, 64
	s_barrier
	s_cbranch_scc1 .LBB2_97
; %bb.96:                               ;   in Loop: Header=BB2_87 Depth=2
	s_mov_b32 s0, s3
	s_branch .LBB2_87
.LBB2_97:                               ; %Flow1580
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v164, 37
	v_readlane_b32 s29, v164, 38
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB2_106
; %bb.98:                               ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s28, v164, 33
	v_readlane_b32 s29, v164, 34
	s_and_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB2_106
; %bb.99:                               ;   in Loop: Header=BB2_85 Depth=1
	v_add_u32_e32 v66, s27, v132
	v_sub_u32_e32 v68, 0, v66
	v_max_i32_e32 v68, v66, v68
	v_mul_hi_u32 v69, v68, s22
	v_mul_lo_u32 v70, v69, s68
	v_sub_u32_e32 v68, v68, v70
	v_add_u32_e32 v70, 1, v69
	v_cmp_le_u32_e32 vcc, s68, v68
	v_ashrrev_i32_e32 v67, 31, v66
	v_xor_b32_e32 v67, s83, v67
	v_cndmask_b32_e32 v69, v69, v70, vcc
	v_subrev_u32_e32 v70, s68, v68
	v_cndmask_b32_e32 v68, v68, v70, vcc
	v_add_u32_e32 v70, 1, v69
	v_cmp_le_u32_e32 vcc, s68, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v69, v70, vcc
	v_xor_b32_e32 v68, v68, v67
	v_sub_u32_e32 v67, v68, v67
	v_mul_lo_u32 v68, v67, s33
	v_sub_u32_e32 v66, v66, v68
	v_sub_u32_e32 v69, 0, v66
	v_ashrrev_i32_e32 v68, 31, v66
	v_max_i32_e32 v66, v66, v69
	v_mul_hi_u32 v69, v66, s31
	v_mul_lo_u32 v70, v69, s39
	v_sub_u32_e32 v66, v66, v70
	v_add_u32_e32 v70, 1, v69
	v_cmp_le_u32_e32 vcc, s39, v66
	v_xor_b32_e32 v68, s30, v68
	s_nop 0
	v_cndmask_b32_e32 v69, v69, v70, vcc
	v_subrev_u32_e32 v70, s39, v66
	v_cndmask_b32_e32 v66, v66, v70, vcc
	v_add_u32_e32 v70, 1, v69
	v_cmp_le_u32_e32 vcc, s39, v66
	s_nop 1
	v_cndmask_b32_e32 v66, v69, v70, vcc
	v_xor_b32_e32 v66, v66, v68
	v_sub_u32_e32 v66, v66, v68
	v_mad_u64_u32 v[66:67], s[28:29], v67, s40, v[66:67]
	v_mul_lo_u32 v66, v66, s41
	v_add_u32_e32 v66, s66, v66
	v_readlane_b32 s28, v164, 15
	v_ashrrev_i32_e32 v67, 31, v66
	v_readlane_b32 s29, v164, 16
	s_nop 1
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[28:29]
	flat_load_dword v68, v[66:67] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v68, v133
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_106
; %bb.100:                              ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_mov_b64 s[28:29], 0
	s_mov_b64 s[92:93], 0
                                        ; implicit-def: $sgpr34_sgpr35
                                        ; implicit-def: $sgpr90_sgpr91
	s_branch .LBB2_102
.LBB2_101:                              ; %Flow1575
                                        ;   in Loop: Header=BB2_102 Depth=2
	s_and_b64 s[52:53], exec, s[90:91]
	s_or_b64 s[28:29], s[52:53], s[28:29]
	s_andn2_b64 s[34:35], s[34:35], exec
	s_and_b64 s[52:53], s[94:95], exec
	s_or_b64 s[34:35], s[34:35], s[52:53]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB2_104
.LBB2_102:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s92, s92, 1
	s_addc_u32 s93, s93, 0
	v_mov_b64_e32 v[68:69], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[92:93], v[68:69]
	s_mov_b64 s[94:95], -1
	s_or_b64 s[90:91], s[90:91], exec
	s_cbranch_vccnz .LBB2_101
; %bb.103:                              ;   in Loop: Header=BB2_102 Depth=2
	s_sleep 4
	flat_load_dword v68, v[66:67] offset:256 sc0 sc1
	s_andn2_b64 s[52:53], s[90:91], exec
	s_mov_b64 s[94:95], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v68, v133
	s_and_b64 s[54:55], vcc, exec
	s_or_b64 s[90:91], s[52:53], s[54:55]
	s_branch .LBB2_101
.LBB2_104:                              ; %loop.exit.guard1523
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_and_saveexec_b64 s[28:29], s[34:35]
	s_xor_b64 s[28:29], exec, s[28:29]
	s_cbranch_execz .LBB2_106
; %bb.105:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s52, v164, 0
	v_readlane_b32 s54, v164, 2
	v_readlane_b32 s55, v164, 3
	v_readlane_b32 s53, v164, 1
	s_nop 0
	v_mov_b64_e32 v[66:67], s[54:55]
	flat_atomic_or v[66:67], v161
.LBB2_106:                              ; %.critedge661
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s28, v164, 8
	v_readlane_b32 s29, v164, 9
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[28:29]
	s_mov_b64 s[28:29], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB2_110
; %bb.107:                              ;   in Loop: Header=BB2_85 Depth=1
	v_mov_b32_e32 v66, 0
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s34, v164, 35
	v_readlane_b32 s35, v164, 36
	s_and_b64 s[34:35], s[28:29], s[34:35]
	s_mov_b64 exec, s[34:35]
	s_cbranch_execz .LBB2_109
; %bb.108:                              ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s52, v164, 0
	v_readlane_b32 s54, v164, 2
	v_readlane_b32 s55, v164, 3
	v_readlane_b32 s53, v164, 1
	s_nop 0
	v_mov_b64_e32 v[66:67], s[54:55]
	flat_load_dword v66, v[66:67] sc1
.LBB2_109:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit324
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v66, v163, v66
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v66, 0x2000000, v66
	v_cmp_eq_u32_e64 s[28:29], 0, v66
.LBB2_110:                              ; %Flow1583
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_and_saveexec_b64 s[52:53], s[28:29]
	s_cbranch_execz .LBB2_84
; %bb.111:                              ; %.critedge679
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_writelane_b32 v164, s52, 50
	s_nop 1
	v_writelane_b32 v164, s53, 51
	s_nop 0
	v_readlane_b32 s0, v164, 22
	v_readlane_b32 s1, v164, 23
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_313
; %bb.112:                              ; %.lr.ph755
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_sub_i32 s65, s38, s64
	s_min_i32 s52, s65, 0x100
	s_abs_i32 s53, s52
	v_cvt_f32_u32_e32 v74, s53
	v_readlane_b32 s0, v164, 17
	s_ashr_i32 s55, s52, 31
	s_lshl_b32 s1, s2, 8
	v_rcp_iflag_f32_e32 v74, v74
	v_mov_b32_e32 v72, s0
	s_sub_i32 s0, 0, s53
	v_or_b32_e32 v66, s64, v134
	v_mul_f32_e32 v74, 0x4f7ffffe, v74
	v_cvt_u32_f32_e32 v74, v74
	v_readlane_b32 s34, v164, 4
	v_cmp_gt_i32_e32 vcc, s38, v66
	v_or_b32_e32 v68, s64, v140
	v_mul_lo_u32 v75, s0, v74
	s_lshl_b32 s0, s55, 9
	v_subrev_u32_e32 v97, s0, v159
	s_lshl_b32 s0, s50, 8
	s_sub_i32 s61, s0, s1
	v_readlane_b32 s0, v164, 44
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s34, s0
	v_readlane_b32 s1, v164, 46
	v_cndmask_b32_e32 v66, v72, v66, vcc
	v_cmp_gt_i32_e32 vcc, s38, v68
	v_or_b32_e32 v70, s64, v141
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 48
	v_cndmask_b32_e32 v68, v72, v68, vcc
	v_cmp_gt_i32_e32 vcc, s38, v70
	v_or_b32_e32 v73, s64, v142
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 42
	v_cndmask_b32_e32 v70, v72, v70, vcc
	v_cmp_gt_i32_e32 vcc, s38, v73
	s_lshl_b32 s1, s1, 2
	v_readlane_b32 s92, v164, 0
	v_cndmask_b32_e32 v72, v72, v73, vcc
	s_sub_i32 s0, s0, s1
	v_ashrrev_i32_e32 v67, 31, v66
	v_readlane_b32 s93, v164, 1
	v_ashrrev_i32_e32 v69, 31, v68
	v_ashrrev_i32_e32 v71, 31, v70
	v_ashrrev_i32_e32 v73, 31, v72
	s_mul_i32 s54, s52, s20
	v_mul_hi_u32 v75, v74, v75
	s_lshl_b32 s0, s0, 7
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[92:93]
	v_lshl_add_u64 v[68:69], v[68:69], 1, s[92:93]
	v_lshl_add_u64 v[70:71], v[70:71], 1, s[92:93]
	v_lshl_add_u64 v[72:73], v[72:73], 1, s[92:93]
	v_cmp_gt_i32_e64 s[28:29], s54, v0
	s_mov_b32 s51, 0
	v_add_u32_e32 v96, v74, v75
	s_lshl_b32 s3, s52, 1
	s_sub_i32 s26, 0, s52
	s_add_i32 s50, s21, s0
	v_readlane_b32 s94, v164, 2
	v_readlane_b32 s95, v164, 3
	v_readlane_b32 s35, v164, 5
	s_branch .LBB2_114
.LBB2_113:                              ; %._crit_edge753
                                        ;   in Loop: Header=BB2_114 Depth=2
	s_add_i32 s51, s51, 1
	s_add_i32 s50, s50, s42
	s_cmp_eq_u32 s51, s69
	s_cbranch_scc1 .LBB2_313
.LBB2_114:                              ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_118 Depth 3
                                        ;         Child Loop BB2_312 Depth 4
                                        ;         Child Loop BB2_272 Depth 4
                                        ;         Child Loop BB2_295 Depth 4
                                        ;         Child Loop BB2_302 Depth 4
                                        ;           Child Loop BB2_307 Depth 5
	s_andn2_b64 vcc, exec, s[74:75]
	s_cbranch_vccnz .LBB2_113
; %bb.115:                              ; %.lr.ph752.preheader
                                        ;   in Loop: Header=BB2_114 Depth=2
	v_mov_b64_e32 v[94:95], s[44:45]
	flat_load_dwordx4 v[74:77], v[94:95]
	flat_load_dwordx4 v[90:93], v[94:95] offset:16
	flat_load_dwordx4 v[98:101], v[94:95] offset:32
	flat_load_dwordx4 v[102:105], v[94:95] offset:48
	s_nop 0
	flat_load_dwordx2 v[94:95], v[94:95] offset:64
	s_mul_i32 s2, s51, s42
	s_add_i32 s0, s2, s27
	s_abs_i32 s34, s0
	s_mul_hi_u32 s35, s34, s22
	s_mul_i32 s70, s35, s68
	s_ashr_i32 s1, s0, 31
	s_sub_i32 s34, s34, s70
	s_xor_b32 s1, s1, s83
	s_add_i32 s71, s35, 1
	s_sub_i32 s70, s34, s68
	s_cmp_ge_u32 s34, s68
	s_cselect_b32 s35, s71, s35
	s_cselect_b32 s34, s70, s34
	s_add_i32 s70, s35, 1
	s_cmp_ge_u32 s34, s68
	s_cselect_b32 s34, s70, s35
	s_xor_b32 s34, s34, s1
	s_sub_i32 s34, s34, s1
	s_mul_i32 s35, s34, s33
	s_sub_i32 s70, s0, s35
	s_cmp_eq_u32 s34, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 1
	v_mov_b32_e32 v86, s57
	s_mov_b32 s90, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v76, 0, v76, vcc
	v_cndmask_b32_e32 v77, 0, v77, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 2
	v_cndmask_b32_e32 v77, v77, v91, vcc
	v_cndmask_b32_e32 v76, v76, v90, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 3
	v_cndmask_b32_e32 v76, v76, v92, vcc
	v_cndmask_b32_e32 v77, v77, v93, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 4
	v_cndmask_b32_e32 v77, v77, v99, vcc
	v_cndmask_b32_e32 v76, v76, v98, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 5
	v_sub_co_u32_e64 v74, s[0:1], s56, v74
	v_cndmask_b32_e32 v76, v76, v100, vcc
	v_cndmask_b32_e32 v77, v77, v101, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 6
	v_subb_co_u32_e64 v75, s[0:1], v86, v75, s[0:1]
	v_cndmask_b32_e32 v77, v77, v103, vcc
	v_cndmask_b32_e32 v76, v76, v102, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 7
	v_cndmask_b32_e32 v76, v76, v104, vcc
	v_cndmask_b32_e32 v77, v77, v105, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s1, s70
	s_mul_hi_u32 s34, s1, s31
	s_mul_i32 s34, s34, s39
	s_sub_i32 s1, s1, s34
	s_ashr_i32 s0, s70, 31
	s_sub_i32 s34, s1, s39
	s_cmp_ge_u32 s1, s39
	s_cselect_b32 s1, s34, s1
	s_sub_i32 s34, s1, s39
	s_cmp_ge_u32 s1, s39
	s_cselect_b32 s1, s34, s1
	s_add_i32 s34, s70, s21
	s_add_i32 s70, s0, s50
	s_xor_b32 s1, s1, s0
	s_sub_i32 s35, s70, s35
	s_sub_i32 s0, s0, s1
	v_cndmask_b32_e32 v77, v77, v95, vcc
	v_cndmask_b32_e32 v76, v76, v94, vcc
	s_sub_i32 s1, s35, s1
	s_add_i32 s0, s34, s0
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	v_cmp_ne_u64_e32 vcc, 0, v[76:77]
	s_mul_i32 s1, s60, s1
	s_mul_i32 s34, s0, s60
	v_cndmask_b32_e32 v75, 0, v75, vcc
	v_cndmask_b32_e32 v74, 0, v74, vcc
	s_add_i32 s0, s61, s1
	s_add_i32 s34, s34, s64
	v_lshl_add_u64 v[76:77], v[74:75], 0, v[88:89]
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s35, s34, 31
	v_lshl_add_u64 v[74:75], s[34:35], 1, v[74:75]
	v_lshl_add_u64 v[76:77], s[0:1], 1, v[76:77]
	s_branch .LBB2_118
.LBB2_116:                              ; %Flow1567
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB2_117:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_add_i32 s90, s90, s20
	s_cmp_ge_i32 s90, s42
	v_lshl_add_u64 v[76:77], v[76:77], 0, s[84:85]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB2_113
.LBB2_118:                              ; %.lr.ph752
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB2_312 Depth 4
                                        ;         Child Loop BB2_272 Depth 4
                                        ;         Child Loop BB2_295 Depth 4
                                        ;         Child Loop BB2_302 Depth 4
                                        ;           Child Loop BB2_307 Depth 5
	v_cndmask_b32_e64 v86, 0, 1, s[76:77]
	v_cmp_ne_u32_e64 s[0:1], 1, v86
	s_andn2_b64 vcc, exec, s[76:77]
	s_cbranch_vccnz .LBB2_120
; %bb.119:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[66:67]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_branch .LBB2_121
.LBB2_120:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
.LBB2_121:                              ;   in Loop: Header=BB2_118 Depth=3
	s_add_i32 s91, s90, s2
	s_add_i32 s98, s91, s20
	v_cmp_le_i32_e32 vcc, s91, v135
	v_cmp_gt_i32_e64 s[34:35], s98, v135
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_123
; %bb.122:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v62
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v135
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_123:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v137
	v_cmp_gt_i32_e64 s[34:35], s98, v137
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_125
; %bb.124:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v63
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v137
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_125:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v138
	v_cmp_gt_i32_e64 s[34:35], s98, v138
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_127
; %bb.126:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v64
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v138
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_127:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v139
	v_cmp_gt_i32_e64 s[34:35], s98, v139
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_129
; %bb.128:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v65
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v139
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86
.LBB2_129:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_286
; %bb.130:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[68:69]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_132
.LBB2_131:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v58
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v135
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_132:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_149
; %bb.133:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_150
.LBB2_134:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_151
.LBB2_135:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_152
.LBB2_136:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[70:71]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_138
.LBB2_137:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v54
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v135
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_138:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_153
; %bb.139:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_154
.LBB2_140:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_155
.LBB2_141:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_156
.LBB2_142:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[72:73]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_144
.LBB2_143:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v50
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v135
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_144:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_157
; %bb.145:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_158
.LBB2_146:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_159
.LBB2_147:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_160
.LBB2_148:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[66:67]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_branch .LBB2_161
.LBB2_149:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v59
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v137
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_134
.LBB2_150:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v60
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v138
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_135
.LBB2_151:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v61
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v139
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_136
.LBB2_152:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_137
	s_branch .LBB2_138
.LBB2_153:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v55
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v137
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_140
.LBB2_154:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v56
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v138
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_141
.LBB2_155:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v57
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v139
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_142
.LBB2_156:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_143
	s_branch .LBB2_144
.LBB2_157:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v51
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v137
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_146
.LBB2_158:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v52
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v138
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_147
.LBB2_159:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v53
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v139
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_148
.LBB2_160:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
.LBB2_161:                              ;   in Loop: Header=BB2_118 Depth=3
	v_cmp_le_i32_e32 vcc, s91, v143
	v_cmp_gt_i32_e64 s[34:35], s98, v143
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_163
; %bb.162:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v46
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v143
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_163:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v144
	v_cmp_gt_i32_e64 s[34:35], s98, v144
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_165
; %bb.164:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v47
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v144
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_165:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v145
	v_cmp_gt_i32_e64 s[34:35], s98, v145
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_167
; %bb.166:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v48
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v145
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_167:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v146
	v_cmp_gt_i32_e64 s[34:35], s98, v146
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_169
; %bb.168:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v49
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v146
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86
.LBB2_169:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_287
; %bb.170:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[68:69]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_172
.LBB2_171:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v42
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v143
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_172:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_189
; %bb.173:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_190
.LBB2_174:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_191
.LBB2_175:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_192
.LBB2_176:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[70:71]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_178
.LBB2_177:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v38
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v143
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_178:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_193
; %bb.179:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_194
.LBB2_180:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_195
.LBB2_181:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_196
.LBB2_182:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[72:73]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_184
.LBB2_183:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v34
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v143
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_184:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_197
; %bb.185:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_198
.LBB2_186:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_199
.LBB2_187:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_200
.LBB2_188:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[66:67]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_branch .LBB2_201
.LBB2_189:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v43
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v144
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_174
.LBB2_190:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v44
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v145
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_175
.LBB2_191:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v45
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v146
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_176
.LBB2_192:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_177
	s_branch .LBB2_178
.LBB2_193:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v39
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v144
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_180
.LBB2_194:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v40
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v145
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_181
.LBB2_195:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v41
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v146
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_182
.LBB2_196:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_183
	s_branch .LBB2_184
.LBB2_197:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v35
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v144
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_186
.LBB2_198:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v36
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v145
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_187
.LBB2_199:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v37
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v146
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_188
.LBB2_200:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
.LBB2_201:                              ;   in Loop: Header=BB2_118 Depth=3
	v_cmp_le_i32_e32 vcc, s91, v147
	v_cmp_gt_i32_e64 s[34:35], s98, v147
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_203
; %bb.202:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v30
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v147
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_203:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v148
	v_cmp_gt_i32_e64 s[34:35], s98, v148
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_205
; %bb.204:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v31
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v148
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_205:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v149
	v_cmp_gt_i32_e64 s[34:35], s98, v149
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_207
; %bb.206:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v32
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v149
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_207:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v150
	v_cmp_gt_i32_e64 s[34:35], s98, v150
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_209
; %bb.208:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v33
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v150
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86
.LBB2_209:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_288
; %bb.210:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[68:69]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_212
.LBB2_211:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v26
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v147
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_212:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_229
; %bb.213:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_230
.LBB2_214:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_231
.LBB2_215:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_232
.LBB2_216:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[70:71]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_218
.LBB2_217:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v22
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v147
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_218:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_233
; %bb.219:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_234
.LBB2_220:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_235
.LBB2_221:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_236
.LBB2_222:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[72:73]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_224
.LBB2_223:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v18
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v147
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_224:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_237
; %bb.225:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_238
.LBB2_226:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_239
.LBB2_227:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_240
.LBB2_228:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[66:67]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_branch .LBB2_241
.LBB2_229:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v27
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v148
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_214
.LBB2_230:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v28
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v149
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_215
.LBB2_231:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v29
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v150
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_216
.LBB2_232:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_217
	s_branch .LBB2_218
.LBB2_233:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v23
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v148
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_220
.LBB2_234:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v24
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v149
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_221
.LBB2_235:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v25
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v150
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_222
.LBB2_236:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_223
	s_branch .LBB2_224
.LBB2_237:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v19
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v148
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_226
.LBB2_238:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v20
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v149
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_227
.LBB2_239:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v21
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v150
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_228
.LBB2_240:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
.LBB2_241:                              ;   in Loop: Header=BB2_118 Depth=3
	v_cmp_le_i32_e32 vcc, s91, v151
	v_cmp_gt_i32_e64 s[34:35], s98, v151
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_243
; %bb.242:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v14
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v151
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_243:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v152
	v_cmp_gt_i32_e64 s[34:35], s98, v152
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_245
; %bb.244:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v15
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v152
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_245:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v153
	v_cmp_gt_i32_e64 s[34:35], s98, v153
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_247
; %bb.246:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v16
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v153
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90
.LBB2_247:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s91, v154
	v_cmp_gt_i32_e64 s[34:35], s98, v154
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_249
; %bb.248:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v17
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v154
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86
.LBB2_249:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_289
; %bb.250:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[68:69]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_252
.LBB2_251:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v10
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v151
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_252:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_276
; %bb.253:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_277
.LBB2_254:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_278
.LBB2_255:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_279
.LBB2_256:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[70:71]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_258
.LBB2_257:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v6
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v151
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_258:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_280
; %bb.259:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_281
.LBB2_260:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_282
.LBB2_261:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_283
.LBB2_262:                              ;   in Loop: Header=BB2_118 Depth=3
	flat_load_ushort v86, v[72:73]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v86
	s_and_saveexec_b64 s[0:1], s[92:93]
	s_cbranch_execz .LBB2_264
.LBB2_263:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v2
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v151
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_264:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execnz .LBB2_284
; %bb.265:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execnz .LBB2_285
.LBB2_266:                              ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[34:35]
	s_cbranch_execz .LBB2_268
.LBB2_267:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v5
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v154
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:96
.LBB2_268:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi64ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s73, s90
	s_mul_hi_u32 s1, s72, s90
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s72, s90
	s_andn2_b64 vcc, exec, s[78:79]
	v_lshl_add_u64 v[90:91], s[0:1], 1, v[74:75]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB2_290
; %bb.269:                              ;   in Loop: Header=BB2_118 Depth=3
	s_mov_b64 s[34:35], -1
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB2_292
; %bb.270:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v86, v118
	v_mov_b32_e32 v92, v157
	v_mov_b32_e32 v93, v158
	v_mov_b32_e32 v94, v0
                                        ; implicit-def: $sgpr92_sgpr93
                                        ; implicit-def: $sgpr94_sgpr95
	s_branch .LBB2_272
.LBB2_271:                              ; %Flow1562
                                        ;   in Loop: Header=BB2_272 Depth=4
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 s[70:71], exec, s[98:99]
	s_or_b64 s[34:35], s[70:71], s[34:35]
	s_andn2_b64 s[70:71], s[92:93], exec
	s_and_b64 s[92:93], s[94:95], exec
	s_or_b64 s[92:93], s[70:71], s[92:93]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB2_291
.LBB2_272:                              ; %.lr.ph.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ;       Parent Loop BB2_118 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v95, 0xf8, v86
	v_add_u32_e32 v98, 8, v95
	v_cmp_lt_i32_e64 s[96:97], s52, v98
	v_cmp_ge_i32_e32 vcc, s52, v98
	s_and_saveexec_b64 s[70:71], vcc
	s_cbranch_execz .LBB2_274
; %bb.273:                              ;   in Loop: Header=BB2_272 Depth=4
	v_mul_lo_u32 v98, s72, v92
	v_lshlrev_b32_e32 v98, 1, v98
	v_lshlrev_b32_e32 v95, 1, v95
	v_add3_u32 v98, v90, v98, v95
	v_add_u32_e32 v95, v93, v95
	v_or_b32_e32 v95, v95, v98
	v_and_b32_e32 v95, 15, v95
	v_cmp_eq_u32_e32 vcc, 0, v95
	s_andn2_b64 s[96:97], s[96:97], exec
	s_and_b64 s[98:99], vcc, exec
	s_or_b64 s[96:97], s[96:97], s[98:99]
.LBB2_274:                              ; %Flow1561
                                        ;   in Loop: Header=BB2_272 Depth=4
	s_or_b64 exec, exec, s[70:71]
	s_mov_b64 s[98:99], -1
	s_andn2_b64 s[94:95], s[94:95], exec
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_271
; %bb.275:                              ; %.critedge.i
                                        ;   in Loop: Header=BB2_272 Depth=4
	v_add_u32_e32 v94, 0x200, v94
	v_cmp_le_i32_e32 vcc, s80, v94
	v_add_u32_e32 v93, 0x2000, v93
	v_add_u32_e32 v92, 16, v92
	v_add_u32_e32 v86, 0x1000, v86
	s_or_b64 s[94:95], s[94:95], exec
	s_orn2_b64 s[98:99], vcc, exec
	s_branch .LBB2_271
.LBB2_276:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v11
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v152
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_254
.LBB2_277:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v12
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v153
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_255
.LBB2_278:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v13
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v154
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_256
.LBB2_279:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_257
	s_branch .LBB2_258
.LBB2_280:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v7
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v152
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_260
.LBB2_281:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v8
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v153
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_261
.LBB2_282:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v86, v86, v9
	v_bfe_u32 v90, v86, 16, 1
	v_add3_u32 v86, v86, v90, s62
	v_subrev_u32_e32 v90, s91, v154
	v_lshl_add_u32 v90, v90, 9, v136
	ds_write_b16_d16_hi v90, v86 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_262
.LBB2_283:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[0:1], s[92:93]
	s_cbranch_execnz .LBB2_263
	s_branch .LBB2_264
.LBB2_284:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v3
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v152
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB2_266
.LBB2_285:                              ;   in Loop: Header=BB2_118 Depth=3
	v_add_f32_e32 v90, v86, v4
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s91, v153
	v_lshl_add_u32 v91, v91, 9, v136
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[34:35]
	s_cbranch_execnz .LBB2_267
	s_branch .LBB2_268
.LBB2_286:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_131
	s_branch .LBB2_132
.LBB2_287:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_171
	s_branch .LBB2_172
.LBB2_288:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_211
	s_branch .LBB2_212
.LBB2_289:                              ;   in Loop: Header=BB2_118 Depth=3
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_251
	s_branch .LBB2_252
.LBB2_290:                              ;   in Loop: Header=BB2_118 Depth=3
	s_cbranch_execz .LBB2_117
	s_branch .LBB2_310
.LBB2_291:                              ; %Flow1563
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[34:35]
	s_orn2_b64 s[34:35], s[92:93], exec
.LBB2_292:                              ; %Flow1564
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v86, 0, 1, s[34:35]
	s_nop 0
	v_readfirstlane_b32 s0, v86
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[34:35], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB2_297
; %bb.293:                              ;   in Loop: Header=BB2_118 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB2_296
; %bb.294:                              ; %.lr.ph.i331.preheader
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v92, v97
	v_mov_b32_e32 v86, v0
.LBB2_295:                              ; %.lr.ph.i331
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ;       Parent Loop BB2_118 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v93, v86, v96
	v_mul_lo_u32 v94, v93, s53
	v_sub_u32_e32 v94, v86, v94
	v_add_u32_e32 v95, 1, v93
	v_subrev_u32_e32 v98, s53, v94
	v_cmp_le_u32_e32 vcc, s53, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v94, v94, v98, vcc
	v_add_u32_e32 v95, 1, v93
	v_cmp_le_u32_e32 vcc, s53, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_xor_b32_e32 v93, s55, v93
	v_subrev_u32_e32 v98, s55, v93
	v_mad_u64_u32 v[94:95], s[70:71], s26, v98, v[86:87]
	v_lshlrev_b32_e32 v93, 9, v93
	v_mul_lo_u32 v95, s3, v98
	v_sub_u32_e32 v93, v93, v95
	v_add_u32_e32 v93, v92, v93
	ds_read_u16 v93, v93
	v_mad_i64_i32 v[98:99], s[70:71], s72, v98, 0
	v_add_u32_e32 v86, 0x200, v86
	v_mov_b32_e32 v95, v87
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[90:91]
	v_cmp_le_i32_e32 vcc, s54, v86
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[98:99]
	v_add_u32_e32 v92, 0x400, v92
	s_or_b64 s[34:35], vcc, s[34:35]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v93
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB2_295
.LBB2_296:                              ; %Flow1553
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_297:                              ; %Flow1559
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_309
; %bb.298:                              ;   in Loop: Header=BB2_118 Depth=3
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB2_308
; %bb.299:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v98, v162
	v_mov_b64_e32 v[92:93], v[76:77]
	v_mov_b32_e32 v99, v0
	s_branch .LBB2_302
.LBB2_300:                              ; %Flow1555
                                        ;   in Loop: Header=BB2_302 Depth=4
	s_or_b64 exec, exec, s[94:95]
.LBB2_301:                              ; %.loopexit.i333
                                        ;   in Loop: Header=BB2_302 Depth=4
	s_or_b64 exec, exec, s[92:93]
	v_add_u32_e32 v99, 0x200, v99
	v_cmp_le_i32_e32 vcc, s80, v99
	v_lshl_add_u64 v[92:93], v[92:93], 0, s[88:89]
	s_or_b64 s[34:35], vcc, s[34:35]
	v_add_u32_e32 v98, 0x2000, v98
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB2_308
.LBB2_302:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ;       Parent Loop BB2_118 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB2_307 Depth 5
	v_lshlrev_b32_e32 v86, 3, v99
	v_and_b32_e32 v86, 0xf8, v86
	v_add_u32_e32 v94, 8, v86
	v_cmp_ge_i32_e32 vcc, s52, v94
	s_and_saveexec_b64 s[70:71], vcc
	s_xor_b64 s[92:93], exec, s[70:71]
	s_cbranch_execz .LBB2_304
; %bb.303:                              ;   in Loop: Header=BB2_302 Depth=4
	v_lshrrev_b32_e32 v94, 5, v99
	v_lshlrev_b32_e32 v86, 1, v86
	v_lshlrev_b32_e32 v95, 9, v94
	v_add3_u32 v95, 0, v95, v86
	ds_read_b128 v[100:103], v95
	v_mad_i64_i32 v[94:95], s[70:71], s72, v94, 0
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[90:91]
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[86:87]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[94:95], v[100:103]
                                        ; implicit-def: $vgpr86
.LBB2_304:                              ; %Flow1556
                                        ;   in Loop: Header=BB2_302 Depth=4
	s_andn2_saveexec_b64 s[92:93], s[92:93]
	s_cbranch_execz .LBB2_301
; %bb.305:                              ; %.preheader.i
                                        ;   in Loop: Header=BB2_302 Depth=4
	v_cmp_gt_i32_e32 vcc, s65, v86
	s_and_saveexec_b64 s[94:95], vcc
	s_cbranch_execz .LBB2_300
; %bb.306:                              ; %.lr.ph.i334
                                        ;   in Loop: Header=BB2_302 Depth=4
	s_mov_b32 s70, 0
	s_mov_b64 s[96:97], 0
	v_mov_b32_e32 v86, v98
	v_mov_b64_e32 v[94:95], v[92:93]
.LBB2_307:                              ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ;       Parent Loop BB2_118 Depth=3
                                        ;         Parent Loop BB2_302 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v100, v86
	s_add_i32 s71, s70, 1
	v_add_u32_e32 v101, s70, v160
	s_cmp_gt_u32 s70, 6
	v_cmp_le_u32_e32 vcc, s52, v101
	s_cselect_b64 s[98:99], -1, 0
	s_or_b64 s[98:99], s[98:99], vcc
	s_and_b64 s[98:99], exec, s[98:99]
	v_add_u32_e32 v86, 2, v86
	s_mov_b32 s70, s71
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v100
	s_or_b64 s[96:97], s[98:99], s[96:97]
	v_lshl_add_u64 v[94:95], v[94:95], 0, 2
	s_andn2_b64 exec, exec, s[96:97]
	s_cbranch_execnz .LBB2_307
	s_branch .LBB2_300
.LBB2_308:                              ; %Flow1558
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB2_309:                              ; %Flow1560
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_branch .LBB2_117
.LBB2_310:                              ;   in Loop: Header=BB2_118 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB2_116
; %bb.311:                              ; %.lr.ph.i338.preheader
                                        ;   in Loop: Header=BB2_118 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v92, v97
	v_mov_b32_e32 v86, v0
.LBB2_312:                              ; %.lr.ph.i338
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_114 Depth=2
                                        ;       Parent Loop BB2_118 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v93, v86, v96
	v_mul_lo_u32 v94, v93, s53
	v_sub_u32_e32 v94, v86, v94
	v_add_u32_e32 v95, 1, v93
	v_subrev_u32_e32 v98, s53, v94
	v_cmp_le_u32_e32 vcc, s53, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v94, v94, v98, vcc
	v_add_u32_e32 v95, 1, v93
	v_cmp_le_u32_e32 vcc, s53, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_xor_b32_e32 v93, s55, v93
	v_subrev_u32_e32 v98, s55, v93
	v_mad_u64_u32 v[94:95], s[70:71], s26, v98, v[86:87]
	v_lshlrev_b32_e32 v93, 9, v93
	v_mul_lo_u32 v95, s3, v98
	v_sub_u32_e32 v93, v93, v95
	v_add_u32_e32 v93, v92, v93
	ds_read_u16 v93, v93
	v_mad_i64_i32 v[98:99], s[70:71], s72, v98, 0
	v_add_u32_e32 v86, 0x200, v86
	v_mov_b32_e32 v95, v87
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[90:91]
	v_cmp_le_i32_e32 vcc, s54, v86
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[98:99]
	v_add_u32_e32 v92, 0x400, v92
	s_or_b64 s[34:35], vcc, s[34:35]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v93
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB2_312
	s_branch .LBB2_116
.LBB2_313:                              ; %._crit_edge756
                                        ;   in Loop: Header=BB2_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s2, v164, 6
	v_readlane_b32 s3, v164, 7
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_mov_b64 exec, s[2:3]
	s_cbranch_execz .LBB2_315
; %bb.314:                              ;   in Loop: Header=BB2_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB2_315:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s0, v164, 24
	v_readlane_b32 s1, v164, 25
	v_readlane_b32 s52, v164, 50
	s_and_b64 s[0:1], s[28:29], s[0:1]
	v_readlane_b32 s51, v164, 39
	v_readlane_b32 s65, v164, 40
	v_readlane_b32 s61, v164, 41
	v_readlane_b32 s53, v164, 51
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_83
; %bb.316:                              ; %.lr.ph758.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s0, v164, 44
	s_lshl_b32 s0, s0, 2
	v_readlane_b32 s2, v164, 4
	s_add_i32 s0, s2, s0
	v_readlane_b32 s1, v164, 46
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 48
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v164, 42
	s_lshl_b32 s1, s1, 2
	v_readlane_b32 s3, v164, 5
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s2, s0, 7
	s_mov_b32 s3, s69
	s_branch .LBB2_318
.LBB2_317:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB2_318 Depth=2
	s_add_i32 s3, s3, -1
	s_add_i32 s2, s2, s42
	s_cmp_lg_u32 s3, 0
	s_cbranch_scc0 .LBB2_83
.LBB2_318:                              ; %.lr.ph758
                                        ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[18:19], s[48:49]
	flat_load_dwordx4 v[2:5], v[18:19]
	flat_load_dwordx4 v[6:9], v[18:19] offset:16
	flat_load_dwordx4 v[10:13], v[18:19] offset:32
	flat_load_dwordx4 v[14:17], v[18:19] offset:48
	s_abs_i32 s1, s2
	flat_load_dwordx2 v[18:19], v[18:19] offset:64
	s_mul_hi_u32 s27, s1, s22
	s_mul_i32 s34, s27, s68
	s_ashr_i32 s0, s2, 31
	s_sub_i32 s1, s1, s34
	s_xor_b32 s0, s0, s83
	s_add_i32 s35, s27, 1
	s_sub_i32 s34, s1, s68
	s_cmp_ge_u32 s1, s68
	s_cselect_b32 s27, s35, s27
	s_cselect_b32 s1, s34, s1
	s_add_i32 s34, s27, 1
	s_cmp_ge_u32 s1, s68
	s_cselect_b32 s1, s34, s27
	s_xor_b32 s1, s1, s0
	s_sub_i32 s27, s1, s0
	s_mul_i32 s1, s23, s27
	s_add_i32 s1, s2, s1
	s_mul_i32 s0, s27, s33
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s0, s1, s0
	s_add_i32 s0, s2, s0
	s_xor_b32 s0, s0, s1
	s_xor_b32 s34, s1, s30
	s_mul_hi_u32 s1, s0, s31
	s_mul_i32 s35, s1, s39
	s_sub_i32 s0, s0, s35
	s_add_i32 s50, s1, 1
	s_sub_i32 s35, s0, s39
	s_cmp_ge_u32 s0, s39
	s_cselect_b32 s1, s50, s1
	s_cselect_b32 s0, s35, s0
	s_add_i32 s35, s1, 1
	s_cmp_ge_u32 s0, s39
	s_cselect_b32 s0, s35, s1
	s_xor_b32 s0, s0, s34
	s_mul_i32 s26, s40, s36
	s_sub_i32 s0, s0, s34
	s_add_i32 s0, s0, s26
	s_mul_i32 s0, s0, s41
	s_add_i32 s0, s0, s66
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s46, s0
	s_addc_u32 s1, s47, s1
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmp_eq_u32 s27, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 1
	v_mov_b32_e32 v20, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v4, 0, v4, vcc
	v_cndmask_b32_e32 v5, 0, v5, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 2
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 3
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_cndmask_b32_e32 v5, v5, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 4
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_cndmask_b32_e32 v4, v4, v10, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 5
	v_cndmask_b32_e32 v4, v4, v12, vcc
	v_cndmask_b32_e32 v5, v5, v13, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 6
	v_cndmask_b32_e32 v5, v5, v15, vcc
	v_cndmask_b32_e32 v4, v4, v14, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s27, 7
	v_sub_co_u32_e64 v2, s[0:1], s0, v2
	v_cndmask_b32_e32 v4, v4, v16, vcc
	v_cndmask_b32_e32 v5, v5, v17, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v3, s[0:1], v20, v3, s[0:1]
	v_cndmask_b32_e32 v5, v5, v19, vcc
	v_cndmask_b32_e32 v4, v4, v18, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_cmp_lg_u32 s27, s36
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB2_320
; %bb.319:                              ; %Flow1549
                                        ;   in Loop: Header=BB2_318 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_317
	s_branch .LBB2_321
.LBB2_320:                              ;   in Loop: Header=BB2_318 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB2_317
.LBB2_321:                              ;   in Loop: Header=BB2_318 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB2_317
.LBB2_322:                              ; %.critedge270
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 165
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 168
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals,comdat
.Lfunc_end2:
	.size	_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals, .Lfunc_end2-_Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.num_vgpr, 165
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16908
; TotalNumSgprs: 106
; NumVgprs: 165
; NumAgprs: 0
; TotalNumVgprs: 165
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 165
; AccumOffset: 168
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 41
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[18:19], s[0:1], 0x60
	s_load_dwordx8 s[20:27], s[0:1], 0x100
	s_load_dwordx8 s[36:43], s[0:1], 0xc0
	s_load_dwordx4 s[28:31], s[0:1], 0xe0
	s_load_dwordx2 s[34:35], s[0:1], 0xf8
	s_load_dwordx2 s[16:17], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s88, s21, 31
	s_lshr_b32 s3, s88, 29
	s_add_i32 s3, s21, s3
	s_ashr_i32 s33, s3, 3
	s_cmp_ge_i32 s2, s27
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr245 : SGPR spill to VGPR lane
	s_cbranch_scc0 .LBB3_72
; %bb.1:
	s_sub_i32 s16, s2, s27
	s_mov_b32 s17, 0
	s_lshl_b64 s[4:5], s[16:17], 2
	s_add_u32 s6, s42, s4
	s_addc_u32 s7, s43, s5
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB3_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit355
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_cmp_lg_u64 s[30:31], 0
	s_cselect_b64 s[44:45], -1, 0
	s_cmp_eq_u64 s[30:31], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB3_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB3_6
; %bb.5:
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_load_dword v2, v[2:3] sc1
.LBB3_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit358
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_and_saveexec_b64 s[46:47], s[6:7]
	s_cbranch_execnz .LBB3_8
	s_branch .LBB3_71
.LBB3_7:
	s_mov_b64 s[6:7], -1
	s_and_saveexec_b64 s[46:47], s[6:7]
	s_cbranch_execz .LBB3_71
.LBB3_8:                                ; %.critedge
	s_mul_i32 s95, s25, s24
	s_cmp_ge_i32 s16, s95
	s_cbranch_scc1 .LBB3_71
; %bb.9:                                ; %.lr.ph
	s_mul_hi_i32 s13, s33, s22
	s_mul_i32 s12, s33, s22
	s_ashr_i32 s51, s22, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_add_u32 s52, s18, s6
	s_addc_u32 s53, s19, s7
	s_add_u32 s54, s52, s6
	s_addc_u32 s55, s53, s7
	s_add_u32 s56, s54, s6
	s_addc_u32 s57, s55, s7
	s_add_u32 s58, s56, s6
	s_addc_u32 s59, s57, s7
	s_add_u32 s60, s58, s6
	s_addc_u32 s61, s59, s7
	s_add_u32 s62, s60, s6
	s_addc_u32 s63, s61, s7
	s_add_u32 s64, s62, s6
	s_load_dwordx2 s[48:49], s[0:1], 0x90
	s_addc_u32 s65, s63, s7
	s_load_dwordx2 s[6:7], s[0:1], 0x120
	s_mov_b32 s50, s22
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v21, 0
	v_mul_lo_u32 v64, s24, v0
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s17, s26, 5
	s_add_i32 s3, s6, 64
	s_cmp_lg_u32 s20, 0
	v_writelane_b32 v245, s3, 2
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 31
	s_cmp_lg_u32 s20, 1
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v245, s7, 32
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 6
	s_cmp_lg_u32 s20, 2
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v245, s7, 7
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 8
	s_cmp_lg_u32 s20, 3
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v245, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 17
	s_cmp_lg_u32 s20, 4
	v_lshrrev_b32_e32 v18, 5, v0
	v_writelane_b32 v245, s7, 18
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 0
	s_cmp_lg_u32 s20, 5
	v_mov_b32_e32 v19, v21
	v_writelane_b32 v245, s7, 1
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 14
	s_cmp_lg_u32 s20, 6
	s_mov_b64 s[98:99], 0
	v_writelane_b32 v245, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v245, s6, 23
	s_cmp_lg_u32 s20, 7
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v245, s7, 24
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s89, s25
	v_cvt_f32_u32_e32 v2, s89
	s_sub_i32 s66, 0, s89
	s_ashr_i32 s3, s25, 31
	s_lshl_b64 s[82:83], s[50:51], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v245, s6, 25
	s_movk_i32 s91, 0x7fff
	s_mov_b32 s92, 0x7060302
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v245, s7, 26
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_readfirstlane_b32 s67, v2
	s_mul_i32 s66, s66, s67
	s_mul_hi_u32 s66, s67, s66
	s_add_i32 s90, s67, s66
	s_add_u32 s66, s48, 2
	s_addc_u32 s67, s49, 0
	v_writelane_b32 v245, s66, 27
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v245, s67, 28
	s_mul_i32 s66, s13, 14
	s_mul_hi_u32 s67, s12, 14
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 14
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	v_writelane_b32 v245, s66, 29
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v67, 0x100, v2
	v_writelane_b32 v245, s67, 30
	s_lshl_b64 s[66:67], s[12:13], 2
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	v_writelane_b32 v245, s66, 33
	s_nop 1
	v_writelane_b32 v245, s67, 34
	s_mul_i32 s66, s13, 12
	s_mul_hi_u32 s67, s12, 12
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 12
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	v_writelane_b32 v245, s66, 4
	s_nop 1
	v_writelane_b32 v245, s67, 5
	s_mul_i32 s66, s13, 6
	s_mul_hi_u32 s67, s12, 6
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 6
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	v_writelane_b32 v245, s66, 35
	s_nop 1
	v_writelane_b32 v245, s67, 36
	s_mul_i32 s66, s13, 10
	s_mul_hi_u32 s67, s12, 10
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 10
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	v_writelane_b32 v245, s66, 12
	s_add_u32 s96, s18, s12
	s_addc_u32 s97, s19, s13
	v_writelane_b32 v245, s67, 13
	s_xor_b64 s[66:67], s[14:15], -1
	s_branch .LBB3_12
.LBB3_10:                               ; %Flow2264
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_sub_i32 s12, s16, s27
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s95
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB3_11:                               ; %Flow2276
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB3_71
.LBB3_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_16 Depth 2
                                        ;     Child Loop BB3_28 Depth 2
                                        ;       Child Loop BB3_32 Depth 3
	s_abs_i32 s13, s16
	s_mul_hi_u32 s14, s13, s90
	s_mul_i32 s15, s14, s89
	s_ashr_i32 s12, s16, 31
	s_sub_i32 s13, s13, s15
	s_xor_b32 s12, s12, s3
	s_add_i32 s15, s14, 1
	s_sub_i32 s68, s13, s89
	s_cmp_ge_u32 s13, s89
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s13, s68, s13
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s13, s89
	s_cselect_b32 s13, s15, s14
	s_xor_b32 s13, s13, s12
	s_sub_i32 s93, s13, s12
	s_mul_i32 s12, s93, s25
	s_sub_i32 s94, s16, s12
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB3_20
; %bb.13:                               ;   in Loop: Header=BB3_12 Depth=1
	v_add_u32_e32 v2, s93, v64
	v_mul_lo_u32 v2, v2, s25
	v_add_u32_e32 v2, s94, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[38:39]
	flat_load_dword v4, v[2:3] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB3_20
; %bb.14:                               ; %.lr.ph.i.i.i360.preheader
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[72:73], 0
                                        ; implicit-def: $sgpr68_sgpr69
                                        ; implicit-def: $sgpr70_sgpr71
	s_branch .LBB3_16
.LBB3_15:                               ; %Flow2272
                                        ;   in Loop: Header=BB3_16 Depth=2
	s_and_b64 s[76:77], exec, s[70:71]
	s_or_b64 s[14:15], s[76:77], s[14:15]
	s_andn2_b64 s[68:69], s[68:69], exec
	s_and_b64 s[74:75], s[74:75], exec
	s_or_b64 s[68:69], s[68:69], s[74:75]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_18
.LBB3_16:                               ; %.lr.ph.i.i.i360
                                        ;   Parent Loop BB3_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s72, s72, 1
	s_addc_u32 s73, s73, 0
	v_mov_b64_e32 v[4:5], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[72:73], v[4:5]
	s_mov_b64 s[74:75], -1
	s_or_b64 s[70:71], s[70:71], exec
	s_cbranch_vccnz .LBB3_15
; %bb.17:                               ;   in Loop: Header=BB3_16 Depth=2
	s_sleep 4
	flat_load_dword v4, v[2:3] offset:256 sc0 sc1
	s_andn2_b64 s[70:71], s[70:71], exec
	s_mov_b64 s[74:75], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[76:77], vcc, exec
	s_or_b64 s[70:71], s[70:71], s[76:77]
	s_branch .LBB3_15
.LBB3_18:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[68:69]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB3_20
; %bb.19:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_atomic_or v[2:3], v66
.LBB3_20:                               ; %.critedge833
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[44:45]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB3_24
; %bb.21:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b32_e32 v2, 0
	s_and_saveexec_b64 s[14:15], s[8:9]
	s_cbranch_execz .LBB3_23
; %bb.22:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_load_dword v2, v[2:3] sc1
.LBB3_23:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit366
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v67, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x4000000, v2
	v_cmp_eq_u32_e64 s[14:15], 0, v2
.LBB3_24:                               ; %Flow2275
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_and_saveexec_b64 s[68:69], s[14:15]
	s_cbranch_execz .LBB3_11
; %bb.25:                               ; %.critedge855
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB3_38
; %bb.26:                               ; %.lr.ph.i367.preheader
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_mul_i32 s70, s93, s26
	s_lshl_b32 s72, s94, 8
	s_ashr_i32 s71, s70, 31
	s_ashr_i32 s73, s72, 31
	v_lshl_add_u64 v[2:3], v[18:19], 0, s[70:71]
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mad_u64_u32 v[4:5], s[14:15], s50, v2, v[4:5]
	v_mul_lo_u32 v3, s50, v3
	v_mul_lo_u32 v2, s51, v2
	v_add3_u32 v5, v2, v5, v3
	v_readlane_b32 s14, v245, 27
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_readlane_b32 s15, v245, 28
	v_lshl_add_u64 v[22:23], s[18:19], 0, v[2:3]
	v_lshl_add_u64 v[26:27], s[52:53], 0, v[2:3]
	v_lshl_add_u64 v[24:25], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v245, 29
	v_readlane_b32 s15, v245, 30
	v_lshl_add_u64 v[38:39], s[96:97], 0, v[2:3]
	s_mov_b64 s[74:75], 0
	v_lshl_add_u64 v[28:29], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v245, 33
	v_readlane_b32 s15, v245, 34
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[30:31], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v245, 4
	v_readlane_b32 s15, v245, 5
	s_nop 1
	v_lshl_add_u64 v[32:33], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v245, 35
	v_readlane_b32 s15, v245, 36
	s_nop 1
	v_lshl_add_u64 v[34:35], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v245, 12
	v_readlane_b32 s15, v245, 13
	s_nop 1
	v_lshl_add_u64 v[36:37], s[14:15], 0, v[2:3]
	s_branch .LBB3_28
.LBB3_27:                               ; %.critedge.i370
                                        ;   in Loop: Header=BB3_28 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s17, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[82:83]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[82:83]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[82:83]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[82:83]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[82:83]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[82:83]
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[82:83]
	s_or_b64 s[74:75], vcc, s[74:75]
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[82:83]
	s_andn2_b64 exec, exec, s[74:75]
	s_cbranch_execz .LBB3_38
.LBB3_28:                               ; %.lr.ph.i367
                                        ;   Parent Loop BB3_12 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_32 Depth 3
	v_lshlrev_b32_e32 v2, 3, v69
	v_and_b32_e32 v2, 0xf8, v2
	v_or_b32_e32 v2, s72, v2
	v_mov_b32_e32 v3, s73
	v_lshl_add_u64 v[4:5], v[2:3], 0, 8
	v_cmp_lt_u64_e32 vcc, s[50:51], v[4:5]
	s_or_b64 s[14:15], s[66:67], vcc
	s_and_saveexec_b64 s[76:77], s[14:15]
	s_xor_b64 s[76:77], exec, s[76:77]
	s_cbranch_execz .LBB3_36
; %bb.29:                               ; %.preheader.i369.preheader
                                        ;   in Loop: Header=BB3_28 Depth=2
	v_and_b32_e32 v20, 0xf8, v68
	v_lshlrev_b32_e32 v4, 1, v68
	v_lshl_add_u64 v[2:3], s[72:73], 0, v[20:21]
	v_and_b32_e32 v20, 0x1f0, v4
	v_lshl_add_u64 v[4:5], v[22:23], 0, v[20:21]
	v_lshl_add_u64 v[6:7], v[24:25], 0, v[20:21]
	v_lshl_add_u64 v[8:9], v[26:27], 0, v[20:21]
	v_lshl_add_u64 v[10:11], v[28:29], 0, v[20:21]
	v_lshl_add_u64 v[12:13], v[30:31], 0, v[20:21]
	v_lshl_add_u64 v[14:15], v[32:33], 0, v[20:21]
	v_lshl_add_u64 v[16:17], v[34:35], 0, v[20:21]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[20:21]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[20:21]
	s_mov_b64 s[78:79], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr80_sgpr81
	s_branch .LBB3_32
.LBB3_30:                               ; %Flow2266
                                        ;   in Loop: Header=BB3_32 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_b64 s[84:85], s[86:87], exec
	s_or_b64 s[80:81], s[80:81], s[84:85]
.LBB3_31:                               ; %Flow2265
                                        ;   in Loop: Header=BB3_32 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[80:81]
	s_or_b64 s[78:79], s[14:15], s[78:79]
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execz .LBB3_35
.LBB3_32:                               ; %.preheader.i369
                                        ;   Parent Loop BB3_12 Depth=1
                                        ;     Parent Loop BB3_28 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[50:51], v[2:3]
	s_or_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB3_31
; %bb.33:                               ; %.preheader.i369.1
                                        ;   in Loop: Header=BB3_32 Depth=3
	v_lshl_add_u64 v[46:47], v[4:5], 0, v[44:45]
	v_lshl_add_u64 v[48:49], v[8:9], 0, v[44:45]
	global_load_ushort v20, v[46:47], off
	global_load_ushort v50, v[48:49], off
	v_lshl_add_u64 v[62:63], v[10:11], 0, v[44:45]
	v_lshl_add_u64 v[70:71], v[2:3], 0, 1
	v_cmp_gt_u64_e32 vcc, s[50:51], v[70:71]
	s_mov_b64 s[86:87], -1
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v20, 16, v20
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v20, v20, v50
	v_lshl_add_u64 v[50:51], v[12:13], 0, v[44:45]
	global_load_ushort v52, v[50:51], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v20, v20, v52
	v_lshl_add_u64 v[52:53], v[16:17], 0, v[44:45]
	global_load_ushort v54, v[52:53], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v20, v20, v54
	v_lshl_add_u64 v[54:55], v[42:43], 0, v[44:45]
	global_load_ushort v56, v[54:55], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v56, 16, v56
	v_add_f32_e32 v20, v20, v56
	v_lshl_add_u64 v[56:57], v[40:41], 0, v[44:45]
	global_load_ushort v58, v[56:57], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v58, 16, v58
	v_add_f32_e32 v20, v20, v58
	v_lshl_add_u64 v[58:59], v[14:15], 0, v[44:45]
	global_load_ushort v60, v[58:59], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v60, 16, v60
	v_add_f32_e32 v20, v20, v60
	global_load_ushort v60, v[62:63], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v60, 16, v60
	v_add_f32_e32 v20, v20, v60
	v_bfe_u32 v60, v20, 16, 1
	v_add3_u32 v20, v20, v60, s91
	v_lshl_add_u64 v[60:61], v[6:7], 0, v[44:45]
	global_store_short_d16_hi v[60:61], v20, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB3_30
; %bb.34:                               ;   in Loop: Header=BB3_32 Depth=3
	global_load_ushort v20, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v46, v[46:47], off offset:2
	s_nop 0
	global_load_ushort v47, v[50:51], off offset:2
	global_load_ushort v48, v[52:53], off offset:2
	global_load_ushort v49, v[54:55], off offset:2
	s_nop 0
	global_load_ushort v50, v[56:57], off
	global_load_ushort v51, v[58:59], off
	global_load_ushort v52, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[2:3], v[2:3], 0, 2
	s_orn2_b64 s[86:87], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v20, 16, v20
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v46, 16, v46
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v47, 16, v47
	v_add_f32_e32 v20, v46, v20
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v48, 16, v48
	v_add_f32_e32 v20, v20, v47
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v20, v20, v48
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v20, v20, v49
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v20, v20, v50
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v20, v20, v51
	v_add_f32_e32 v20, v20, v52
	v_bfe_u32 v46, v20, 16, 1
	v_add3_u32 v20, v20, v46, s91
	global_store_short_d16_hi v[60:61], v20, off
	s_branch .LBB3_30
.LBB3_35:                               ; %Flow2267
                                        ;   in Loop: Header=BB3_28 Depth=2
	s_or_b64 exec, exec, s[78:79]
                                        ; implicit-def: $vgpr2_vgpr3
.LBB3_36:                               ; %Flow2268
                                        ;   in Loop: Header=BB3_28 Depth=2
	s_andn2_saveexec_b64 vcc, s[76:77]
	s_cbranch_execz .LBB3_27
; %bb.37:                               ;   in Loop: Header=BB3_28 Depth=2
	v_lshrrev_b32_e32 v20, 5, v69
	v_lshl_add_u64 v[4:5], v[20:21], 0, s[70:71]
	v_mul_lo_u32 v6, v4, s51
	v_mul_lo_u32 v5, v5, s50
	v_mad_u64_u32 v[2:3], s[14:15], v4, s50, v[2:3]
	v_add3_u32 v3, v5, v3, v6
	v_lshlrev_b64 v[40:41], 1, v[2:3]
	v_lshl_add_u64 v[2:3], s[18:19], 0, v[40:41]
	v_lshl_add_u64 v[6:7], s[52:53], 0, v[40:41]
	v_lshl_add_u64 v[10:11], s[54:55], 0, v[40:41]
	v_lshl_add_u64 v[14:15], s[56:57], 0, v[40:41]
	global_load_dwordx4 v[2:5], v[2:3], off
	v_lshl_add_u64 v[58:59], s[58:59], 0, v[40:41]
	global_load_dwordx4 v[6:9], v[6:7], off
	v_lshl_add_u64 v[60:61], s[60:61], 0, v[40:41]
	global_load_dwordx4 v[10:13], v[10:11], off
	v_lshl_add_u64 v[62:63], s[62:63], 0, v[40:41]
	global_load_dwordx4 v[14:17], v[14:15], off
	v_lshl_add_u64 v[70:71], s[64:65], 0, v[40:41]
	v_lshl_add_u64 v[40:41], s[48:49], 0, v[40:41]
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v73, 0xffff0000, v2
	v_and_b32_e32 v75, 0xffff0000, v3
	v_and_b32_e32 v55, 0xffff0000, v4
	v_and_b32_e32 v43, 0xffff0000, v5
	v_lshlrev_b32_e32 v72, 16, v2
	v_lshlrev_b32_e32 v74, 16, v3
	v_lshlrev_b32_e32 v54, 16, v4
	v_lshlrev_b32_e32 v42, 16, v5
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v77, 0xffff0000, v6
	v_and_b32_e32 v79, 0xffff0000, v7
	v_and_b32_e32 v57, 0xffff0000, v8
	v_and_b32_e32 v45, 0xffff0000, v9
	v_lshlrev_b32_e32 v76, 16, v6
	v_lshlrev_b32_e32 v78, 16, v7
	v_lshlrev_b32_e32 v56, 16, v8
	v_lshlrev_b32_e32 v44, 16, v9
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v81, 0xffff0000, v10
	v_and_b32_e32 v83, 0xffff0000, v11
	v_and_b32_e32 v47, 0xffff0000, v12
	v_and_b32_e32 v51, 0xffff0000, v13
	v_lshlrev_b32_e32 v80, 16, v10
	v_lshlrev_b32_e32 v82, 16, v11
	v_lshlrev_b32_e32 v46, 16, v12
	v_lshlrev_b32_e32 v50, 16, v13
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v85, 0xffff0000, v14
	v_and_b32_e32 v87, 0xffff0000, v15
	v_and_b32_e32 v53, 0xffff0000, v16
	v_and_b32_e32 v49, 0xffff0000, v17
	v_lshlrev_b32_e32 v84, 16, v14
	v_lshlrev_b32_e32 v86, 16, v15
	v_lshlrev_b32_e32 v52, 16, v16
	v_lshlrev_b32_e32 v48, 16, v17
	global_load_dwordx4 v[14:17], v[58:59], off
	global_load_dwordx4 v[10:13], v[60:61], off
	global_load_dwordx4 v[6:9], v[62:63], off
	global_load_dwordx4 v[2:5], v[70:71], off
	v_pk_add_f32 v[58:59], v[76:77], v[72:73]
	v_pk_add_f32 v[60:61], v[78:79], v[74:75]
	v_pk_add_f32 v[58:59], v[58:59], v[80:81]
	v_pk_add_f32 v[60:61], v[60:61], v[82:83]
	v_pk_add_f32 v[58:59], v[58:59], v[84:85]
	v_pk_add_f32 v[60:61], v[60:61], v[86:87]
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v62, 16, v14
	v_lshlrev_b32_e32 v70, 16, v15
	v_and_b32_e32 v63, 0xffff0000, v14
	v_and_b32_e32 v71, 0xffff0000, v15
	v_pk_add_f32 v[14:15], v[60:61], v[70:71]
	v_pk_add_f32 v[58:59], v[58:59], v[62:63]
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v60, 16, v11
	v_lshlrev_b32_e32 v62, 16, v10
	v_and_b32_e32 v61, 0xffff0000, v11
	v_and_b32_e32 v63, 0xffff0000, v10
	v_pk_add_f32 v[10:11], v[58:59], v[62:63]
	v_pk_add_f32 v[14:15], v[14:15], v[60:61]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v58, 16, v6
	v_lshlrev_b32_e32 v60, 16, v7
	v_and_b32_e32 v59, 0xffff0000, v6
	v_and_b32_e32 v61, 0xffff0000, v7
	v_pk_add_f32 v[6:7], v[14:15], v[60:61]
	v_pk_add_f32 v[10:11], v[10:11], v[58:59]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v14, 16, v3
	v_lshlrev_b32_e32 v58, 16, v2
	v_and_b32_e32 v15, 0xffff0000, v3
	v_and_b32_e32 v59, 0xffff0000, v2
	v_pk_add_f32 v[2:3], v[10:11], v[58:59]
	v_pk_add_f32 v[6:7], v[6:7], v[14:15]
	v_bfe_u32 v14, v3, 16, 1
	v_bfe_u32 v10, v7, 16, 1
	v_bfe_u32 v11, v6, 16, 1
	v_bfe_u32 v15, v2, 16, 1
	v_add3_u32 v20, v2, v15, s91
	v_add3_u32 v58, v3, v14, s91
	v_add3_u32 v59, v6, v11, s91
	v_add3_u32 v60, v7, v10, s91
	v_pk_add_f32 v[2:3], v[56:57], v[54:55]
	v_pk_add_f32 v[6:7], v[44:45], v[42:43]
	v_pk_add_f32 v[2:3], v[2:3], v[46:47]
	v_pk_add_f32 v[6:7], v[6:7], v[50:51]
	v_pk_add_f32 v[2:3], v[2:3], v[52:53]
	v_pk_add_f32 v[6:7], v[6:7], v[48:49]
	v_lshlrev_b32_e32 v10, 16, v16
	v_lshlrev_b32_e32 v14, 16, v17
	v_and_b32_e32 v11, 0xffff0000, v16
	v_and_b32_e32 v15, 0xffff0000, v17
	v_pk_add_f32 v[6:7], v[6:7], v[14:15]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_lshlrev_b32_e32 v10, 16, v13
	v_lshlrev_b32_e32 v14, 16, v12
	v_and_b32_e32 v11, 0xffff0000, v13
	v_and_b32_e32 v15, 0xffff0000, v12
	v_pk_add_f32 v[2:3], v[2:3], v[14:15]
	v_pk_add_f32 v[6:7], v[6:7], v[10:11]
	v_lshlrev_b32_e32 v10, 16, v8
	v_lshlrev_b32_e32 v12, 16, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_and_b32_e32 v13, 0xffff0000, v9
	v_pk_add_f32 v[6:7], v[6:7], v[12:13]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_lshlrev_b32_e32 v8, 16, v5
	v_lshlrev_b32_e32 v10, 16, v4
	v_and_b32_e32 v9, 0xffff0000, v5
	v_and_b32_e32 v11, 0xffff0000, v4
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_pk_add_f32 v[4:5], v[6:7], v[8:9]
	v_bfe_u32 v8, v3, 16, 1
	v_bfe_u32 v6, v5, 16, 1
	v_bfe_u32 v7, v4, 16, 1
	v_bfe_u32 v9, v2, 16, 1
	v_add3_u32 v2, v2, v9, s91
	v_add3_u32 v3, v3, v8, s91
	v_add3_u32 v4, v4, v7, s91
	v_add3_u32 v5, v5, v6, s91
	v_perm_b32 v5, v5, v4, s92
	v_perm_b32 v4, v3, v2, s92
	v_perm_b32 v3, v60, v59, s92
	v_perm_b32 v2, v58, v20, s92
	global_store_dwordx4 v[40:41], v[2:5], off
	s_branch .LBB3_27
.LBB3_38:                               ; %Flow2270
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[70:71], s[4:5]
	s_cbranch_execz .LBB3_10
; %bb.39:                               ; %.preheader861
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx4 v[2:5], v[2:3]
	s_mul_i32 s12, s24, s20
	s_add_i32 s12, s93, s12
	v_readlane_b32 s13, v245, 2
	s_mul_i32 s12, s12, s25
	s_add_i32 s13, s13, s94
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s38, s12
	s_addc_u32 s15, s39, s13
	v_mov_b32_e32 v6, s15
	v_readlane_b32 s12, v245, 31
	v_readlane_b32 s13, v245, 32
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e32 v2, vcc, s14, v2
	s_nop 1
	v_subb_co_u32_e32 v3, vcc, v6, v3, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_nop 1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_and_b64 vcc, exec, s[12:13]
	s_cbranch_vccz .LBB3_70
; %bb.40:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB3_42
.LBB3_41:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_42:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:16
	v_readlane_b32 s12, v245, 6
	v_readlane_b32 s13, v245, 7
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_44
; %bb.43:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_44:                               ; %Flow2262
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_46
; %bb.45:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_46:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:24
	v_readlane_b32 s12, v245, 8
	v_readlane_b32 s13, v245, 9
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_48
; %bb.47:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_48:                               ; %Flow2261
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_50
; %bb.49:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_50:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:32
	v_readlane_b32 s12, v245, 17
	v_readlane_b32 s13, v245, 18
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_52
; %bb.51:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_52:                               ; %Flow2260
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_54
; %bb.53:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_54:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:40
	v_readlane_b32 s12, v245, 0
	v_readlane_b32 s13, v245, 1
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_56
; %bb.55:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_56:                               ; %Flow2259
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_58
; %bb.57:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_58:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:48
	v_readlane_b32 s12, v245, 14
	v_readlane_b32 s13, v245, 15
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_60
; %bb.59:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_60:                               ; %Flow2258
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_62
; %bb.61:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_62:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:56
	v_readlane_b32 s12, v245, 23
	v_readlane_b32 s13, v245, 24
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_64
; %bb.63:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_64:                               ; %Flow2257
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_66
; %bb.65:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB3_66:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:64
	v_readlane_b32 s12, v245, 25
	v_readlane_b32 s13, v245, 26
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB3_68
; %bb.67:                               ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB3_68:                               ; %Flow
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_10
; %bb.69:                               ;   in Loop: Header=BB3_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB3_10
.LBB3_70:                               ;   in Loop: Header=BB3_12 Depth=1
	s_branch .LBB3_41
.LBB3_71:                               ; %Flow2280
	s_or_b64 exec, exec, s[46:47]
	s_load_dwordx2 s[16:17], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB3_72:                               ; %Flow2322
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB3_481
; %bb.73:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s42, s4
	s_addc_u32 s5, s43, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v245, s8, 0
	s_nop 1
	v_writelane_b32 v245, s9, 1
	s_and_b64 s[8:9], s[6:7], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB3_75
; %bb.74:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB3_75:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_cmp_lg_u64 s[30:31], 0
	s_cselect_b64 s[4:5], -1, 0
	v_writelane_b32 v245, s4, 2
	s_cmp_eq_u64 s[30:31], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_writelane_b32 v245, s5, 3
	s_cbranch_scc1 .LBB3_79
; %bb.76:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_78
; %bb.77:
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_load_dword v2, v[2:3] sc1
.LBB3_78:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB3_80
	s_branch .LBB3_481
.LBB3_79:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB3_481
.LBB3_80:                               ; %.critedge831
	s_lshr_b32 s3, s88, 24
	s_add_i32 s3, s21, s3
	s_ashr_i32 s48, s3, 8
	s_mul_i32 s3, s25, s48
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v245, s3, 4
	s_cbranch_scc1 .LBB3_481
; %bb.81:                               ; %.lr.ph982
	s_mov_b64 s[4:5], src_shared_base
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s4, s5, 0
	s_cselect_b32 s5, 0, 0
	s_load_dwordx2 s[44:45], s[0:1], 0x80
	s_load_dwordx4 s[8:11], s[0:1], 0x70
	s_load_dwordx2 s[46:47], s[0:1], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x20
	s_load_dwordx2 s[50:51], s[0:1], 0x30
	s_load_dwordx2 s[14:15], s[0:1], 0x50
	s_and_b32 s0, s5, 15
	s_and_b32 s6, s5, -16
	s_add_u32 s6, s6, 16
	s_mov_b32 s1, 0
	s_addc_u32 s7, s4, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s49, s5, s6
	s_cselect_b32 s0, s4, s7
	s_add_u32 s4, s49, 0x8000
	s_addc_u32 s0, s0, 0
	s_and_b32 s5, s4, -16
	s_and_b32 s0, s4, 15
	s_add_u32 s5, s5, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s53, s4, s5
	s_add_i32 s0, s23, 31
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	v_lshlrev_b32_e32 v160, 3, v0
	v_lshrrev_b32_e32 v5, 2, v0
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_and_b32_e32 v2, 24, v160
	v_or_b32_e32 v6, 0x80, v5
	s_ashr_i32 s88, s0, 5
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[146:147], s[0:1], v5, s12, v[2:3]
	v_mad_u64_u32 v[148:149], s[0:1], v6, s12, v[2:3]
	v_mad_u64_u32 v[150:151], s[0:1], v5, s14, v[2:3]
	s_mov_b32 s0, s14
	s_nop 0
	v_writelane_b32 v245, s0, 6
	s_lshl_b32 s43, s25, 2
	s_cmp_gt_i32 s23, 0
	v_writelane_b32 v245, s1, 7
	v_mad_u64_u32 v[152:153], s[0:1], v6, s14, v[2:3]
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v161, 48, v2
	v_and_b32_e32 v162, 0x1fc0, v2
	v_add_u32_e32 v2, s49, v161
	v_add_u32_e32 v7, v2, v162
	v_lshrrev_b32_e32 v8, 4, v7
	v_add_u32_e32 v6, 8, v2
	v_and_b32_e32 v8, 56, v8
	v_xor_b32_e32 v163, v8, v7
	v_add_u32_e32 v7, v6, v162
	v_lshrrev_b32_e32 v8, 4, v7
	v_or_b32_e32 v165, 0x2000, v162
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v2, v2, v165
	v_xor_b32_e32 v164, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	v_and_b32_e32 v7, 56, v7
	v_xor_b32_e32 v166, v7, v2
	v_add_u32_e32 v2, v6, v165
	v_lshrrev_b32_e32 v6, 4, v2
	v_and_b32_e32 v6, 56, v6
	v_xor_b32_e32 v167, v6, v2
	v_add_u32_e32 v2, s53, v161
	v_add_u32_e32 v7, v2, v162
	v_lshrrev_b32_e32 v8, 4, v7
	v_add_u32_e32 v6, 8, v2
	v_and_b32_e32 v8, 56, v8
	v_xor_b32_e32 v168, v8, v7
	v_add_u32_e32 v7, v6, v162
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v2, v2, v165
	s_cselect_b64 s[0:1], -1, 0
	v_xor_b32_e32 v169, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	v_writelane_b32 v245, s0, 8
	v_and_b32_e32 v7, 56, v7
	v_xor_b32_e32 v170, v7, v2
	v_writelane_b32 v245, s1, 9
	s_ashr_i32 s1, s16, 31
	s_mov_b32 s0, s16
	v_add_u32_e32 v2, v6, v165
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v6, 4, v2
	s_add_u32 s0, s38, s0
	v_and_b32_e32 v6, 56, v6
	s_addc_u32 s1, s39, s1
	v_xor_b32_e32 v171, v6, v2
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v245, s0, 10
	v_lshrrev_b32_e32 v4, 8, v0
	v_lshlrev_b32_e32 v6, 6, v2
	v_writelane_b32 v245, s1, 11
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v172, v4, 13, v6
	v_lshl_or_b32 v174, v3, 12, v6
	v_writelane_b32 v245, s0, 12
	v_and_b32_e32 v6, 63, v0
	s_min_i32 s23, s26, 32
	s_bfe_i64 s[60:61], s[44:45], 0x200000
	v_writelane_b32 v245, s1, 13
	v_cmp_eq_u32_e64 s[0:1], 0, v6
	s_cmp_gt_i32 s26, 0
	s_cselect_b64 s[62:63], -1, 0
	v_writelane_b32 v245, s0, 14
	s_cmp_lg_u64 s[28:29], 0
	s_cselect_b64 s[64:65], -1, 0
	v_writelane_b32 v245, s1, 15
	s_max_i32 s0, s22, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s17, 0
	s_cselect_b64 s[66:67], -1, 0
	s_abs_i32 s91, s43
	v_lshl_or_b32 v177, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s91
	s_abs_i32 s94, s26
	v_cvt_f32_u32_e32 v3, s94
	v_writelane_b32 v245, s0, 16
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, 0, s91
	v_rcp_iflag_f32_e32 v3, v3
	s_lshl_b32 s92, s23, 5
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_max_i32 s93, s88, 1
	s_bfe_i32 s52, s25, 0x1001d
	s_ashr_i32 s97, s26, 31
	v_readfirstlane_b32 s1, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s82, s1, s0
	s_sub_i32 s0, 0, s94
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s98, s1, s0
	s_lshr_b32 s0, s98, 24
	s_mul_i32 s1, s0, s94
	s_sub_i32 s1, 0x100, s1
	s_add_i32 s4, s0, 1
	s_sub_i32 s5, s1, s94
	s_cmp_ge_u32 s1, s94
	s_cselect_b32 s0, s4, s0
	s_cselect_b32 s1, s5, s1
	s_add_i32 s4, s0, 1
	s_cmp_ge_u32 s1, s94
	s_cselect_b32 s0, s4, s0
	s_abs_i32 s99, s33
	v_cvt_f32_u32_e32 v2, s99
	s_xor_b32 s0, s0, s97
	s_sub_i32 s21, s0, s97
	v_cmp_gt_u32_e64 s[0:1], s21, v0
	v_rcp_iflag_f32_e32 v2, v2
	s_ashr_i32 s58, s33, 31
	v_writelane_b32 v245, s0, 17
	v_lshrrev_b32_e32 v207, 5, v0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v245, s1, 18
	s_sub_i32 s0, 0, s99
	v_and_b32_e32 v5, 12, v5
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s59, s1, s0
	s_cmp_gt_i32 s21, 0
	v_mad_i64_i32 v[2:3], s[0:1], v207, s44, 0
	s_cselect_b64 s[4:5], -1, 0
	v_readlane_b32 s0, v245, 0
	v_readlane_b32 s1, v245, 1
	v_writelane_b32 v245, s4, 19
	s_and_b64 s[0:1], s[0:1], s[4:5]
	v_lshl_or_b32 v178, v4, 7, v5
	v_writelane_b32 v245, s5, 20
	v_writelane_b32 v245, s0, 21
	v_and_b32_e32 v4, 31, v0
	v_lshlrev_b32_e32 v154, 4, v4
	v_writelane_b32 v245, s1, 22
	s_add_u32 s0, s50, 64
	v_writelane_b32 v245, s0, 23
	s_addc_u32 s0, s51, 0
	v_writelane_b32 v245, s0, 25
	s_add_u32 s0, s46, 64
	v_writelane_b32 v245, s0, 27
	s_addc_u32 s0, s47, 0
	v_mov_b32_e32 v155, 0
	v_writelane_b32 v245, s0, 29
	s_mov_b32 s0, s12
	v_lshl_add_u64 v[156:157], v[2:3], 1, v[154:155]
	v_lshlrev_b32_e32 v2, 9, v207
	v_writelane_b32 v245, s0, 31
	v_add_u32_e32 v216, 0, v2
	v_or_b32_e32 v2, v2, v154
	v_writelane_b32 v245, s1, 32
	s_lshl_b32 s0, s12, 8
	v_add_u32_e32 v217, 0, v2
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_lshrrev_b32_e32 v7, 1, v0
	v_writelane_b32 v245, s0, 33
	s_mul_i32 s0, s61, s23
	s_mul_hi_u32 s1, s44, s23
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_and_b32_e32 v173, 24, v7
	s_mul_i32 s90, s10, s8
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s44, s23
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v245, s43, 35
	s_mov_b64 s[54:55], 0
	v_ashrrev_i32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v149, 31, v148
	v_ashrrev_i32_e32 v151, 31, v150
	v_ashrrev_i32_e32 v153, 31, v152
	v_mul_lo_u32 v175, s26, v0
	v_add_u32_e32 v176, -1, v1
	s_mul_i32 s90, s90, s20
	v_lshl_add_u32 v179, v177, 1, 0
	v_or_b32_e32 v180, 1, v178
	v_or_b32_e32 v181, 2, v178
	v_or_b32_e32 v182, 3, v178
	v_or_b32_e32 v183, 16, v177
	v_or_b32_e32 v184, 32, v177
	v_or_b32_e32 v185, 48, v177
	v_or_b32_e32 v186, 16, v178
	v_or_b32_e32 v187, 17, v178
	v_or_b32_e32 v188, 18, v178
	v_or_b32_e32 v189, 19, v178
	v_or_b32_e32 v190, 32, v178
	v_or_b32_e32 v191, 33, v178
	v_or_b32_e32 v192, 34, v178
	v_or_b32_e32 v193, 35, v178
	v_or_b32_e32 v194, 48, v178
	v_or_b32_e32 v195, 49, v178
	v_or_b32_e32 v196, 50, v178
	v_or_b32_e32 v197, 51, v178
	v_or_b32_e32 v198, 64, v178
	v_or_b32_e32 v199, 0x41, v178
	v_or_b32_e32 v200, 0x42, v178
	v_or_b32_e32 v201, 0x43, v178
	v_or_b32_e32 v202, 0x50, v178
	v_or_b32_e32 v203, 0x51, v178
	v_or_b32_e32 v204, 0x52, v178
	v_or_b32_e32 v205, 0x53, v178
	v_or_b32_e32 v206, 0x60, v178
	v_cmp_gt_i32_e64 s[8:9], s92, v0
	v_or_b32_e32 v208, 0x61, v178
	v_or_b32_e32 v209, 0x62, v178
	v_or_b32_e32 v210, 0x63, v178
	v_or_b32_e32 v211, 0x70, v178
	v_or_b32_e32 v212, 0x71, v178
	v_lshl_or_b32 v213, v4, 3, 1
	v_or_b32_e32 v214, 0x72, v178
	v_or_b32_e32 v215, 0x73, v178
	s_lshl_b64 s[72:73], s[0:1], 1
	s_lshl_b64 s[74:75], s[60:61], 5
	s_sub_i32 s68, 0, s33
	v_and_b32_e32 v218, 0x100, v2
	v_or_b32_e32 v219, 32, v173
	s_movk_i32 s69, 0x7fff
	v_lshl_add_u32 v220, v0, 1, 0
	v_bfrev_b32_e32 v221, 64
	v_writelane_b32 v245, s52, 37
	v_writelane_b32 v245, s82, 38
	s_branch .LBB3_84
.LBB3_82:                               ; %Flow2283
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s2, s2, s27
	v_readlane_b32 s0, v245, 4
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB3_83:                               ; %Flow2316
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[76:77]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[54:55], s[0:1], s[54:55]
	s_andn2_b64 exec, exec, s[54:55]
	s_cbranch_execz .LBB3_481
.LBB3_84:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_86 Depth 2
                                        ;     Child Loop BB3_97 Depth 2
                                        ;     Child Loop BB3_109 Depth 2
                                        ;       Child Loop BB3_113 Depth 3
                                        ;         Child Loop BB3_471 Depth 4
                                        ;         Child Loop BB3_427 Depth 4
                                        ;         Child Loop BB3_454 Depth 4
                                        ;         Child Loop BB3_461 Depth 4
                                        ;           Child Loop BB3_466 Depth 5
                                        ;     Child Loop BB3_477 Depth 2
	s_ashr_i32 s0, s2, 31
	s_xor_b32 s7, s0, s52
	s_abs_i32 s0, s2
	s_mul_hi_u32 s1, s0, s82
	s_mul_i32 s4, s1, s91
	s_sub_i32 s0, s0, s4
	s_add_i32 s4, s1, 1
	s_sub_i32 s5, s0, s91
	s_cmp_ge_u32 s0, s91
	s_cselect_b32 s1, s4, s1
	s_cselect_b32 s0, s5, s0
	s_add_i32 s4, s1, 1
	s_cmp_ge_u32 s0, s91
	s_cselect_b32 s0, s4, s1
	s_xor_b32 s45, s0, s7
	s_sub_i32 s4, s45, s7
	s_lshl_b32 s0, s4, 2
	s_sub_i32 s1, s48, s0
	s_min_i32 s1, s1, 4
	s_abs_i32 s5, s1
	v_cvt_f32_u32_e32 v2, s5
	s_mul_i32 s4, s4, s43
	s_sub_i32 s12, 0, s5
	s_sub_i32 s10, s2, s4
	v_rcp_iflag_f32_e32 v2, v2
	s_xor_b32 s6, s10, s1
	s_ashr_i32 s11, s6, 31
	s_abs_i32 s6, s10
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_mov_b32_e32 v5, v155
	v_mov_b32_e32 v4, v155
	v_mov_b32_e32 v9, v155
	v_readfirstlane_b32 s13, v2
	s_mul_i32 s12, s12, s13
	s_mul_hi_u32 s12, s13, s12
	s_add_i32 s13, s13, s12
	s_mul_hi_u32 s12, s6, s13
	s_mul_i32 s13, s12, s5
	s_sub_i32 s6, s6, s13
	s_add_i32 s13, s12, 1
	s_sub_i32 s14, s6, s5
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s12, s13, s12
	s_cselect_b32 s6, s14, s6
	s_add_i32 s13, s12, 1
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s5, s13, s12
	s_xor_b32 s42, s5, s11
	s_sub_i32 s6, s42, s11
	s_mul_i32 s5, s6, s1
	s_sub_i32 s1, s10, s5
	s_add_i32 s1, s1, s0
	s_lshl_b32 s70, s1, 8
	v_readlane_b32 s0, v245, 31
	v_readlane_b32 s1, v245, 32
	s_mul_i32 s0, s70, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s46, s0
	s_addc_u32 s1, s47, s1
	v_lshl_add_u64 v[2:3], v[146:147], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[130:133], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[148:149], 1, s[0:1]
	s_lshl_b32 s71, s6, 8
	v_readlane_b32 s0, v245, 6
	v_readlane_b32 s1, v245, 7
	s_mul_i32 s0, s71, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s12, s50, s0
	;;#ASMSTART
	global_load_dwordx4 v[134:137], v[2:3], off

	;;#ASMEND
	s_addc_u32 s13, s51, s1
	v_lshl_add_u64 v[2:3], v[150:151], 1, s[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[138:141], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[152:153], 1, s[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v163, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v164, v[132:133]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v166, v[134:135]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v167, v[136:137]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v168, v[138:139]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v169, v[140:141]

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v170, v[142:143]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v171, v[144:145]

	;;#ASMEND
	v_readlane_b32 s12, v245, 8
	v_readlane_b32 s13, v245, 9
	s_andn2_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v3, v155
	v_mov_b32_e32 v2, v155
	v_mov_b32_e32 v8, v155
	v_mov_b32_e32 v7, v155
	v_mov_b32_e32 v6, v155
	v_mov_b32_e32 v13, v155
	v_mov_b32_e32 v12, v155
	v_mov_b32_e32 v11, v155
	v_mov_b32_e32 v10, v155
	v_mov_b32_e32 v17, v155
	v_mov_b32_e32 v16, v155
	v_mov_b32_e32 v15, v155
	v_mov_b32_e32 v14, v155
	v_mov_b32_e32 v21, v155
	v_mov_b32_e32 v20, v155
	v_mov_b32_e32 v19, v155
	v_mov_b32_e32 v18, v155
	v_mov_b32_e32 v25, v155
	v_mov_b32_e32 v24, v155
	v_mov_b32_e32 v23, v155
	v_mov_b32_e32 v22, v155
	v_mov_b32_e32 v29, v155
	v_mov_b32_e32 v28, v155
	v_mov_b32_e32 v27, v155
	v_mov_b32_e32 v26, v155
	v_mov_b32_e32 v33, v155
	v_mov_b32_e32 v32, v155
	v_mov_b32_e32 v31, v155
	v_mov_b32_e32 v30, v155
	v_mov_b32_e32 v37, v155
	v_mov_b32_e32 v36, v155
	v_mov_b32_e32 v35, v155
	v_mov_b32_e32 v34, v155
	v_mov_b32_e32 v41, v155
	v_mov_b32_e32 v40, v155
	v_mov_b32_e32 v39, v155
	v_mov_b32_e32 v38, v155
	v_mov_b32_e32 v45, v155
	v_mov_b32_e32 v44, v155
	v_mov_b32_e32 v43, v155
	v_mov_b32_e32 v42, v155
	v_mov_b32_e32 v49, v155
	v_mov_b32_e32 v48, v155
	v_mov_b32_e32 v47, v155
	v_mov_b32_e32 v46, v155
	v_mov_b32_e32 v53, v155
	v_mov_b32_e32 v52, v155
	v_mov_b32_e32 v51, v155
	v_mov_b32_e32 v50, v155
	v_mov_b32_e32 v57, v155
	v_mov_b32_e32 v56, v155
	v_mov_b32_e32 v55, v155
	v_mov_b32_e32 v54, v155
	v_mov_b32_e32 v61, v155
	v_mov_b32_e32 v60, v155
	v_mov_b32_e32 v59, v155
	v_mov_b32_e32 v58, v155
	v_mov_b32_e32 v65, v155
	v_mov_b32_e32 v64, v155
	v_mov_b32_e32 v63, v155
	v_mov_b32_e32 v62, v155
	v_mov_b32_e32 v69, v155
	v_mov_b32_e32 v68, v155
	v_mov_b32_e32 v67, v155
	v_mov_b32_e32 v66, v155
	v_mov_b32_e32 v73, v155
	v_mov_b32_e32 v72, v155
	v_mov_b32_e32 v71, v155
	v_mov_b32_e32 v70, v155
	v_mov_b32_e32 v77, v155
	v_mov_b32_e32 v76, v155
	v_mov_b32_e32 v75, v155
	v_mov_b32_e32 v74, v155
	v_mov_b32_e32 v81, v155
	v_mov_b32_e32 v80, v155
	v_mov_b32_e32 v79, v155
	v_mov_b32_e32 v78, v155
	v_mov_b32_e32 v85, v155
	v_mov_b32_e32 v84, v155
	v_mov_b32_e32 v83, v155
	v_mov_b32_e32 v82, v155
	v_mov_b32_e32 v89, v155
	v_mov_b32_e32 v88, v155
	v_mov_b32_e32 v87, v155
	v_mov_b32_e32 v86, v155
	v_mov_b32_e32 v93, v155
	v_mov_b32_e32 v92, v155
	v_mov_b32_e32 v91, v155
	v_mov_b32_e32 v90, v155
	v_mov_b32_e32 v97, v155
	v_mov_b32_e32 v96, v155
	v_mov_b32_e32 v95, v155
	v_mov_b32_e32 v94, v155
	v_mov_b32_e32 v101, v155
	v_mov_b32_e32 v100, v155
	v_mov_b32_e32 v99, v155
	v_mov_b32_e32 v98, v155
	v_mov_b32_e32 v105, v155
	v_mov_b32_e32 v104, v155
	v_mov_b32_e32 v103, v155
	v_mov_b32_e32 v102, v155
	v_mov_b32_e32 v109, v155
	v_mov_b32_e32 v108, v155
	v_mov_b32_e32 v107, v155
	v_mov_b32_e32 v106, v155
	v_mov_b32_e32 v113, v155
	v_mov_b32_e32 v112, v155
	v_mov_b32_e32 v111, v155
	v_mov_b32_e32 v110, v155
	v_mov_b32_e32 v117, v155
	v_mov_b32_e32 v116, v155
	v_mov_b32_e32 v115, v155
	v_mov_b32_e32 v114, v155
	v_mov_b32_e32 v121, v155
	v_mov_b32_e32 v120, v155
	v_mov_b32_e32 v119, v155
	v_mov_b32_e32 v118, v155
	v_mov_b32_e32 v125, v155
	v_mov_b32_e32 v124, v155
	v_mov_b32_e32 v123, v155
	v_mov_b32_e32 v122, v155
	v_mov_b32_e32 v129, v155
	v_mov_b32_e32 v128, v155
	v_mov_b32_e32 v127, v155
	v_mov_b32_e32 v126, v155
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB3_92
; %bb.85:                               ; %.lr.ph940.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s3, v245, 23
	s_add_u32 s0, s3, s0
	v_readlane_b32 s3, v245, 25
	s_addc_u32 s1, s3, s1
	s_lshl_b32 s10, s45, 2
	s_add_i32 s10, s2, s10
	s_sub_i32 s10, s10, s4
	s_sub_i32 s10, s10, s5
	s_lshl_b32 s12, s7, 2
	s_sub_i32 s10, s10, s12
	v_readlane_b32 s3, v245, 33
	s_mul_i32 s12, s3, s10
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 1
	v_readlane_b32 s3, v245, 27
	s_add_u32 s12, s3, s12
	v_readlane_b32 s3, v245, 29
	v_mov_b32_e32 v2, 0
	s_addc_u32 s13, s3, s13
	s_mov_b32 s10, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v99, v2
	v_mov_b32_e32 v100, v2
	v_mov_b32_e32 v101, v2
	v_mov_b32_e32 v102, v2
	v_mov_b32_e32 v103, v2
	v_mov_b32_e32 v104, v2
	v_mov_b32_e32 v105, v2
	v_mov_b32_e32 v106, v2
	v_mov_b32_e32 v107, v2
	v_mov_b32_e32 v108, v2
	v_mov_b32_e32 v109, v2
	v_mov_b32_e32 v110, v2
	v_mov_b32_e32 v111, v2
	v_mov_b32_e32 v112, v2
	v_mov_b32_e32 v113, v2
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v115, v2
	v_mov_b32_e32 v116, v2
	v_mov_b32_e32 v117, v2
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v119, v2
	v_mov_b32_e32 v120, v2
	v_mov_b32_e32 v121, v2
	v_mov_b32_e32 v122, v2
	v_mov_b32_e32 v123, v2
	v_mov_b32_e32 v124, v2
	v_mov_b32_e32 v125, v2
	v_mov_b32_e32 v126, v2
	v_mov_b32_e32 v127, v2
	v_mov_b32_e32 v128, v2
	v_mov_b32_e32 v129, v2
.LBB3_86:                               ; %.lr.ph940
                                        ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s16, s10, 1
	s_cmp_lt_i32 s16, s88
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_ge_i32 s16, s88
	s_cbranch_scc1 .LBB3_88
; %bb.87:                               ;   in Loop: Header=BB3_86 Depth=2
	v_lshl_add_u64 v[130:131], v[146:147], 1, s[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[130:133], v[130:131], off

	;;#ASMEND
	v_lshl_add_u64 v[134:135], v[148:149], 1, s[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[134:137], v[134:135], off

	;;#ASMEND
	v_lshl_add_u64 v[138:139], v[150:151], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[138:141], v[138:139], off

	;;#ASMEND
	v_lshl_add_u64 v[142:143], v[152:153], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
.LBB3_88:                               ;   in Loop: Header=BB3_86 Depth=2
	s_lshl_b32 s10, s10, 14
	s_and_b32 s10, s10, 0x4000
	s_add_i32 s17, s49, s10
	v_add_u32_e32 v154, s17, v172
	v_add_u32_e32 v158, v154, v173
	v_lshrrev_b32_e32 v159, 4, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v234, v159, v158
	;;#ASMSTART
	ds_read_b64 v[158:159], v234 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v234 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v234 offset:0x800

	;;#ASMEND
	s_add_i32 s10, s53, s10
	;;#ASMSTART
	ds_read_b64 v[226:227], v234 offset:0xc00

	;;#ASMEND
	v_add_u32_e32 v244, s10, v174
	;;#ASMSTART
	ds_read_b64 v[228:229], v234 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[230:231], v234 offset:0x1400

	;;#ASMEND
	v_add_u32_e32 v236, v244, v173
	;;#ASMSTART
	ds_read_b64 v[232:233], v234 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v237, 4, v236
	;;#ASMSTART
	ds_read_b64 v[234:235], v234 offset:0x1c00

	;;#ASMEND
	v_and_b32_e32 v237, 56, v237
	v_xor_b32_e32 v242, v237, v236
	;;#ASMSTART
	ds_read_b64 v[236:237], v242 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[238:239], v242 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[240:241], v242 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[242:243], v242 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_add_u32_e32 v154, v154, v219
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_andn2_b64 vcc, exec, s[14:15]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[158:159], v[236:237], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[158:159], v[238:239], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[158:159], v[240:241], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[158:159], v[242:243], v[114:117]
	v_lshrrev_b32_e32 v158, 4, v154
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v154, v158, v154
	;;#ASMSTART
	ds_read_b64 v[158:159], v154 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[110:113], v[222:223], v[236:237], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[222:223], v[238:239], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[222:223], v[240:241], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[222:223], v[242:243], v[98:101]
	;;#ASMSTART
	ds_read_b64 v[222:223], v154 offset:0x400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[94:97], v[224:225], v[236:237], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[224:225], v[238:239], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[224:225], v[240:241], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[224:225], v[242:243], v[82:85]
	;;#ASMSTART
	ds_read_b64 v[224:225], v154 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[78:81], v[226:227], v[236:237], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[226:227], v[238:239], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[226:227], v[240:241], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[226:227], v[242:243], v[66:69]
	;;#ASMSTART
	ds_read_b64 v[226:227], v154 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[62:65], v[228:229], v[236:237], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[228:229], v[238:239], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[228:229], v[240:241], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[228:229], v[242:243], v[50:53]
	;;#ASMSTART
	ds_read_b64 v[228:229], v154 offset:0x1000

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[46:49], v[230:231], v[236:237], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[230:231], v[238:239], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[230:231], v[240:241], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[230:231], v[242:243], v[34:37]
	;;#ASMSTART
	ds_read_b64 v[230:231], v154 offset:0x1400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[30:33], v[232:233], v[236:237], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[232:233], v[238:239], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[232:233], v[240:241], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[232:233], v[242:243], v[18:21]
	;;#ASMSTART
	ds_read_b64 v[232:233], v154 offset:0x1800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[14:17], v[234:235], v[236:237], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[234:235], v[238:239], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[234:235], v[240:241], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[234:235], v[242:243], v[2:5]
	;;#ASMSTART
	ds_read_b64 v[234:235], v154 offset:0x1c00

	;;#ASMEND
	v_add_u32_e32 v154, v244, v219
	v_lshrrev_b32_e32 v236, 4, v154
	v_and_b32_e32 v236, 56, v236
	v_xor_b32_e32 v154, v236, v154
	;;#ASMSTART
	ds_read_b64 v[236:237], v154 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[238:239], v154 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[240:241], v154 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[242:243], v154 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[126:129], v[158:159], v[236:237], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[158:159], v[238:239], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[158:159], v[240:241], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[158:159], v[242:243], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[222:223], v[236:237], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[222:223], v[238:239], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[222:223], v[240:241], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[222:223], v[242:243], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[224:225], v[236:237], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[224:225], v[238:239], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[224:225], v[240:241], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[224:225], v[242:243], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[226:227], v[236:237], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[226:227], v[238:239], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[226:227], v[240:241], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[226:227], v[242:243], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[228:229], v[236:237], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[228:229], v[238:239], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[228:229], v[240:241], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[228:229], v[242:243], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[230:231], v[236:237], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[230:231], v[238:239], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[230:231], v[240:241], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[230:231], v[242:243], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[232:233], v[236:237], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[232:233], v[238:239], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[232:233], v[240:241], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[232:233], v[242:243], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[234:235], v[236:237], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[234:235], v[238:239], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[234:235], v[240:241], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[234:235], v[242:243], v[2:5]
	s_cbranch_vccnz .LBB3_90
; %bb.89:                               ;   in Loop: Header=BB3_86 Depth=2
	s_lshl_b32 s10, s16, 14
	s_and_b32 s10, s10, 0x4000
	s_add_i32 s14, s49, s10
	v_add_u32_e32 v154, s14, v161
	v_add_u32_e32 v159, v154, v162
	v_lshrrev_b32_e32 v222, 4, v159
	s_or_b32 s14, s14, 8
	v_and_b32_e32 v222, 56, v222
	v_add_u32_e32 v158, s14, v161
	v_xor_b32_e32 v159, v222, v159
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v159, v[130:131]

	;;#ASMEND
	v_add_u32_e32 v159, v158, v162
	v_lshrrev_b32_e32 v222, 4, v159
	v_and_b32_e32 v222, 56, v222
	v_xor_b32_e32 v159, v222, v159
	v_add_u32_e32 v154, v154, v165
	;;#ASMSTART
	ds_write_b64 v159, v[132:133]

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v154
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v154, v159, v154
	;;#ASMSTART
	ds_write_b64 v154, v[134:135]

	;;#ASMEND
	v_add_u32_e32 v154, v158, v165
	v_lshrrev_b32_e32 v158, 4, v154
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v154, v158, v154
	s_add_i32 s10, s53, s10
	;;#ASMSTART
	ds_write_b64 v154, v[136:137]

	;;#ASMEND
	v_add_u32_e32 v154, s10, v161
	v_add_u32_e32 v159, v154, v162
	v_lshrrev_b32_e32 v222, 4, v159
	s_or_b32 s10, s10, 8
	v_and_b32_e32 v222, 56, v222
	v_add_u32_e32 v158, s10, v161
	v_xor_b32_e32 v159, v222, v159
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v159, v[138:139]

	;;#ASMEND
	v_add_u32_e32 v159, v158, v162
	v_lshrrev_b32_e32 v222, 4, v159
	v_and_b32_e32 v222, 56, v222
	v_xor_b32_e32 v159, v222, v159
	v_add_u32_e32 v154, v154, v165
	;;#ASMSTART
	ds_write_b64 v159, v[140:141]

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v154
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v154, v159, v154
	;;#ASMSTART
	ds_write_b64 v154, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v154, v158, v165
	v_lshrrev_b32_e32 v158, 4, v154
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v154, v158, v154
	;;#ASMSTART
	ds_write_b64 v154, v[144:145]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB3_90:                               ;   in Loop: Header=BB3_86 Depth=2
	s_add_u32 s0, s0, 64
	s_addc_u32 s1, s1, 0
	s_add_u32 s12, s12, 64
	s_addc_u32 s13, s13, 0
	s_cmp_eq_u32 s93, s16
	s_barrier
	s_cbranch_scc1 .LBB3_92
; %bb.91:                               ;   in Loop: Header=BB3_86 Depth=2
	s_mov_b32 s10, s16
	s_branch .LBB3_86
.LBB3_92:                               ; %Flow2312
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s12, v245, 17
	v_readlane_b32 s13, v245, 18
	s_and_b64 s[12:13], s[0:1], s[12:13]
	s_mov_b64 exec, s[12:13]
	s_cbranch_execz .LBB3_101
; %bb.93:                               ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s12, v245, 12
	v_readlane_b32 s13, v245, 13
	s_and_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB3_101
; %bb.94:                               ;   in Loop: Header=BB3_84 Depth=1
	v_add_u32_e32 v130, s70, v175
	v_sub_u32_e32 v132, 0, v130
	v_max_i32_e32 v132, v130, v132
	v_mul_hi_u32 v133, v132, s59
	v_mul_lo_u32 v134, v133, s99
	v_sub_u32_e32 v132, v132, v134
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s99, v132
	v_ashrrev_i32_e32 v131, 31, v130
	v_xor_b32_e32 v131, s58, v131
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s99, v132
	v_cndmask_b32_e32 v132, v132, v134, vcc
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s99, v132
	s_nop 1
	v_cndmask_b32_e32 v132, v133, v134, vcc
	v_xor_b32_e32 v132, v132, v131
	v_sub_u32_e32 v131, v132, v131
	v_mul_lo_u32 v132, v131, s33
	v_sub_u32_e32 v130, v130, v132
	v_sub_u32_e32 v133, 0, v130
	v_ashrrev_i32_e32 v132, 31, v130
	v_max_i32_e32 v130, v130, v133
	v_mul_hi_u32 v133, v130, s98
	v_mul_lo_u32 v134, v133, s94
	v_sub_u32_e32 v130, v130, v134
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s94, v130
	v_xor_b32_e32 v132, s97, v132
	s_nop 0
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s94, v130
	v_cndmask_b32_e32 v130, v130, v134, vcc
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s94, v130
	s_nop 1
	v_cndmask_b32_e32 v130, v133, v134, vcc
	v_xor_b32_e32 v130, v130, v132
	v_sub_u32_e32 v130, v130, v132
	v_mad_u64_u32 v[130:131], s[12:13], v131, s24, v[130:131]
	v_mul_lo_u32 v130, v130, s25
	v_add_u32_e32 v130, s6, v130
	v_readlane_b32 s12, v245, 10
	v_ashrrev_i32_e32 v131, 31, v130
	v_readlane_b32 s13, v245, 11
	s_nop 1
	v_lshl_add_u64 v[130:131], v[130:131], 2, s[12:13]
	flat_load_dword v132, v[130:131] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v132, v176
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB3_101
; %bb.95:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_mov_b64 s[12:13], 0
	s_mov_b64 s[78:79], 0
                                        ; implicit-def: $sgpr14_sgpr15
                                        ; implicit-def: $sgpr76_sgpr77
	s_branch .LBB3_97
.LBB3_96:                               ; %Flow2307
                                        ;   in Loop: Header=BB3_97 Depth=2
	s_and_b64 s[16:17], exec, s[76:77]
	s_or_b64 s[12:13], s[16:17], s[12:13]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], s[80:81], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB3_99
.LBB3_97:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s78, s78, 1
	s_addc_u32 s79, s79, 0
	v_mov_b64_e32 v[132:133], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[78:79], v[132:133]
	s_mov_b64 s[80:81], -1
	s_or_b64 s[76:77], s[76:77], exec
	s_cbranch_vccnz .LBB3_96
; %bb.98:                               ;   in Loop: Header=BB3_97 Depth=2
	s_sleep 4
	flat_load_dword v132, v[130:131] offset:256 sc0 sc1
	s_andn2_b64 s[16:17], s[76:77], exec
	s_mov_b64 s[80:81], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v132, v176
	s_and_b64 s[56:57], vcc, exec
	s_or_b64 s[76:77], s[16:17], s[56:57]
	s_branch .LBB3_96
.LBB3_99:                               ; %loop.exit.guard2255
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execz .LBB3_101
; %bb.100:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b64_e32 v[130:131], s[30:31]
	flat_atomic_or v[130:131], v221
.LBB3_101:                              ; %.critedge832
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s12, v245, 2
	v_readlane_b32 s13, v245, 3
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB3_105
; %bb.102:                              ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b32_e32 v130, 0
	s_mov_b64 s[12:13], exec
	v_readlane_b32 s14, v245, 14
	v_readlane_b32 s15, v245, 15
	s_and_b64 s[14:15], s[12:13], s[14:15]
	s_mov_b64 exec, s[14:15]
	s_cbranch_execz .LBB3_104
; %bb.103:                              ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b64_e32 v[130:131], s[30:31]
	flat_load_dword v130, v[130:131] sc1
.LBB3_104:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit340
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v130, v218, v130
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v130, 0x2000000, v130
	v_cmp_eq_u32_e64 s[12:13], 0, v130
.LBB3_105:                              ; %Flow2315
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_and_saveexec_b64 s[76:77], s[12:13]
	s_cbranch_execz .LBB3_83
; %bb.106:                              ; %.critedge854
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s0, v245, 19
	v_readlane_b32 s1, v245, 20
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB3_472
; %bb.107:                              ; %.lr.ph977
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_sub_i32 s16, s22, s71
	s_min_i32 s17, s16, 0x100
	s_abs_i32 s56, s17
	v_cvt_f32_u32_e32 v138, s56
	v_readlane_b32 s0, v245, 16
	s_ashr_i32 s95, s17, 31
	v_or_b32_e32 v130, s71, v177
	v_rcp_iflag_f32_e32 v138, v138
	v_mov_b32_e32 v136, s0
	s_sub_i32 s0, 0, s56
	s_lshl_b32 s1, s11, 8
	v_mul_f32_e32 v138, 0x4f7ffffe, v138
	v_cvt_u32_f32_e32 v138, v138
	v_cmp_gt_i32_e32 vcc, s22, v130
	v_or_b32_e32 v132, s71, v183
	v_or_b32_e32 v134, s71, v184
	v_mul_lo_u32 v139, s0, v138
	s_lshl_b32 s0, s95, 9
	v_subrev_u32_e32 v223, s0, v220
	s_lshl_b32 s0, s42, 8
	s_sub_i32 s11, s0, s1
	s_lshl_b32 s0, s45, 2
	v_cndmask_b32_e32 v130, v136, v130, vcc
	v_cmp_gt_i32_e32 vcc, s22, v132
	s_add_i32 s0, s2, s0
	v_or_b32_e32 v137, s71, v185
	v_cndmask_b32_e32 v132, v136, v132, vcc
	v_cmp_gt_i32_e32 vcc, s22, v134
	s_sub_i32 s0, s0, s4
	s_sub_i32 s0, s0, s5
	v_cndmask_b32_e32 v134, v136, v134, vcc
	v_cmp_gt_i32_e32 vcc, s22, v137
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	v_cndmask_b32_e32 v136, v136, v137, vcc
	v_ashrrev_i32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v137, 31, v136
	s_mul_i32 s57, s17, s23
	v_mul_hi_u32 v139, v138, v139
	s_lshl_b32 s0, s0, 8
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[28:29]
	v_lshl_add_u64 v[132:133], v[132:133], 1, s[28:29]
	v_lshl_add_u64 v[134:135], v[134:135], 1, s[28:29]
	v_lshl_add_u64 v[136:137], v[136:137], 1, s[28:29]
	v_cmp_gt_i32_e64 s[12:13], s57, v0
	s_mov_b32 s96, 0
	v_add_u32_e32 v222, v138, v139
	s_lshl_b32 s89, s17, 1
	s_sub_i32 s10, 0, s17
	s_add_i32 s52, s90, s0
	s_branch .LBB3_109
.LBB3_108:                              ; %._crit_edge975
                                        ;   in Loop: Header=BB3_109 Depth=2
	s_add_i32 s96, s96, 1
	s_add_i32 s52, s52, s26
	s_cmp_eq_u32 s96, s21
	s_cbranch_scc1 .LBB3_472
.LBB3_109:                              ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_113 Depth 3
                                        ;         Child Loop BB3_471 Depth 4
                                        ;         Child Loop BB3_427 Depth 4
                                        ;         Child Loop BB3_454 Depth 4
                                        ;         Child Loop BB3_461 Depth 4
                                        ;           Child Loop BB3_466 Depth 5
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB3_108
; %bb.110:                              ; %.lr.ph974.preheader
                                        ;   in Loop: Header=BB3_109 Depth=2
	v_mov_b64_e32 v[158:159], s[36:37]
	flat_load_dwordx4 v[138:141], v[158:159]
	flat_load_dwordx4 v[142:145], v[158:159] offset:16
	flat_load_dwordx4 v[224:227], v[158:159] offset:32
	flat_load_dwordx4 v[228:231], v[158:159] offset:48
	s_nop 0
	flat_load_dwordx2 v[158:159], v[158:159] offset:64
	s_mul_i32 s42, s96, s26
	s_add_i32 s1, s42, s70
	s_abs_i32 s14, s1
	s_mul_hi_u32 s15, s14, s59
	s_mul_i32 s43, s15, s99
	s_ashr_i32 s0, s1, 31
	s_sub_i32 s14, s14, s43
	s_xor_b32 s0, s0, s58
	s_add_i32 s43, s15, 1
	s_sub_i32 s78, s14, s99
	s_cmp_ge_u32 s14, s99
	s_cselect_b32 s15, s43, s15
	s_cselect_b32 s14, s78, s14
	s_add_i32 s43, s15, 1
	s_cmp_ge_u32 s14, s99
	s_cselect_b32 s14, s43, s15
	s_xor_b32 s14, s14, s0
	s_sub_i32 s14, s14, s0
	s_mul_i32 s0, s14, s33
	s_sub_i32 s1, s1, s0
	s_cmp_eq_u32 s14, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 1
	s_mov_b32 s43, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v140, 0, v140, vcc
	v_cndmask_b32_e32 v141, 0, v141, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 2
	v_cndmask_b32_e32 v141, v141, v143, vcc
	v_cndmask_b32_e32 v140, v140, v142, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 3
	v_cndmask_b32_e32 v140, v140, v144, vcc
	v_cndmask_b32_e32 v141, v141, v145, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 4
	v_cndmask_b32_e32 v141, v141, v225, vcc
	v_cndmask_b32_e32 v140, v140, v224, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 5
	v_cndmask_b32_e32 v140, v140, v226, vcc
	v_cndmask_b32_e32 v141, v141, v227, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 6
	v_cndmask_b32_e32 v141, v141, v229, vcc
	v_cndmask_b32_e32 v140, v140, v228, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 7
	v_cndmask_b32_e32 v140, v140, v230, vcc
	v_cndmask_b32_e32 v141, v141, v231, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s14, s1
	s_mul_hi_u32 s15, s14, s98
	s_mul_i32 s15, s15, s94
	s_sub_i32 s14, s14, s15
	s_ashr_i32 s78, s1, 31
	s_sub_i32 s15, s14, s94
	s_cmp_ge_u32 s14, s94
	s_cselect_b32 s14, s15, s14
	s_sub_i32 s15, s14, s94
	s_cmp_ge_u32 s14, s94
	s_cselect_b32 s14, s15, s14
	s_xor_b32 s79, s14, s78
	s_add_i32 s1, s1, s90
	s_sub_i32 s14, s78, s79
	s_add_i32 s78, s78, s52
	v_cndmask_b32_e32 v141, v141, v159, vcc
	v_cndmask_b32_e32 v140, v140, v158, vcc
	v_sub_co_u32_e32 v138, vcc, s18, v138
	v_mov_b32_e32 v142, s19
	s_add_i32 s1, s1, s14
	s_sub_i32 s0, s78, s0
	v_subb_co_u32_e32 v139, vcc, v142, v139, vcc
	s_mul_i32 s1, s1, s44
	s_sub_i32 s0, s0, s79
	v_lshl_add_u64 v[138:139], v[138:139], 0, v[140:141]
	v_cmp_ne_u64_e32 vcc, 0, v[140:141]
	s_add_i32 s14, s1, s71
	s_mul_i32 s0, s44, s0
	v_cndmask_b32_e32 v141, 0, v139, vcc
	v_cndmask_b32_e32 v140, 0, v138, vcc
	s_ashr_i32 s15, s14, 31
	s_add_i32 s0, s11, s0
	v_lshl_add_u64 v[138:139], s[14:15], 1, v[140:141]
	v_lshl_add_u64 v[140:141], v[140:141], 0, v[156:157]
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[140:141], s[0:1], 1, v[140:141]
	s_branch .LBB3_113
.LBB3_111:                              ; %Flow2299
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB3_112:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_add_i32 s43, s43, s23
	s_cmp_ge_i32 s43, s26
	v_lshl_add_u64 v[140:141], v[140:141], 0, s[72:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB3_108
.LBB3_113:                              ; %.lr.ph974
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB3_471 Depth 4
                                        ;         Child Loop BB3_427 Depth 4
                                        ;         Child Loop BB3_454 Depth 4
                                        ;         Child Loop BB3_461 Depth 4
                                        ;           Child Loop BB3_466 Depth 5
	v_cndmask_b32_e64 v142, 0, 1, s[64:65]
	v_cmp_ne_u32_e64 s[14:15], 1, v142
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB3_115
; %bb.114:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_116
.LBB3_115:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_116:                              ;   in Loop: Header=BB3_113 Depth=3
	s_add_i32 s86, s43, s42
	s_add_i32 s87, s86, s23
	v_cmp_le_i32_e32 vcc, s86, v178
	v_cmp_gt_i32_e64 s[0:1], s87, v178
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_118
; %bb.117:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v126
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v178
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_118:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v180
	v_cmp_gt_i32_e64 s[0:1], s87, v180
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_120
; %bb.119:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v127
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v180
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_120:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v181
	v_cmp_gt_i32_e64 s[0:1], s87, v181
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_122
; %bb.121:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v128
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v181
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_122:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v182
	v_cmp_gt_i32_e64 s[0:1], s87, v182
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_124
; %bb.123:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v129
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v182
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_124:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_441
; %bb.125:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_127
.LBB3_126:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v122
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v178
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_127:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_144
; %bb.128:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_145
.LBB3_129:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_146
.LBB3_130:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_147
.LBB3_131:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_133
.LBB3_132:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v118
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v178
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_133:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_148
; %bb.134:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_149
.LBB3_135:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_150
.LBB3_136:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_151
.LBB3_137:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_139
.LBB3_138:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v114
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v178
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_139:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_152
; %bb.140:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_153
.LBB3_141:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_154
.LBB3_142:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_155
.LBB3_143:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_156
.LBB3_144:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v123
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v180
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_129
.LBB3_145:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v124
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v181
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_130
.LBB3_146:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v125
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v182
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_131
.LBB3_147:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_132
	s_branch .LBB3_133
.LBB3_148:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v119
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v180
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_135
.LBB3_149:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v120
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v181
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_136
.LBB3_150:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v121
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v182
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_137
.LBB3_151:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_138
	s_branch .LBB3_139
.LBB3_152:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v115
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v180
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_141
.LBB3_153:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v116
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v181
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_142
.LBB3_154:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v117
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v182
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_143
.LBB3_155:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_156:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v186
	v_cmp_gt_i32_e64 s[0:1], s87, v186
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_158
; %bb.157:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v110
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v186
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_158:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v187
	v_cmp_gt_i32_e64 s[0:1], s87, v187
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_160
; %bb.159:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v111
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v187
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_160:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v188
	v_cmp_gt_i32_e64 s[0:1], s87, v188
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_162
; %bb.161:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v112
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v188
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_162:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v189
	v_cmp_gt_i32_e64 s[0:1], s87, v189
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_164
; %bb.163:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v113
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v189
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_164:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_442
; %bb.165:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_167
.LBB3_166:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v106
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v186
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_167:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_184
; %bb.168:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_185
.LBB3_169:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_186
.LBB3_170:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_187
.LBB3_171:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_173
.LBB3_172:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v102
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v186
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_173:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_188
; %bb.174:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_189
.LBB3_175:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_190
.LBB3_176:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_191
.LBB3_177:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_179
.LBB3_178:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v98
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v186
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_179:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_192
; %bb.180:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_193
.LBB3_181:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_194
.LBB3_182:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_195
.LBB3_183:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_196
.LBB3_184:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v107
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v187
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_169
.LBB3_185:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v108
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v188
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_170
.LBB3_186:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v109
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v189
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_171
.LBB3_187:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_172
	s_branch .LBB3_173
.LBB3_188:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v103
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v187
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_175
.LBB3_189:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v104
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v188
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_176
.LBB3_190:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v105
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v189
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_177
.LBB3_191:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_178
	s_branch .LBB3_179
.LBB3_192:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v99
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v187
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_181
.LBB3_193:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v100
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v188
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_182
.LBB3_194:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v101
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v189
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_183
.LBB3_195:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_196:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v190
	v_cmp_gt_i32_e64 s[0:1], s87, v190
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_198
; %bb.197:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v94
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v190
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_198:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v191
	v_cmp_gt_i32_e64 s[0:1], s87, v191
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_200
; %bb.199:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v95
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v191
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_200:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v192
	v_cmp_gt_i32_e64 s[0:1], s87, v192
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_202
; %bb.201:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v96
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v192
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_202:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v193
	v_cmp_gt_i32_e64 s[0:1], s87, v193
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_204
; %bb.203:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v97
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v193
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_204:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_443
; %bb.205:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_207
.LBB3_206:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v90
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v190
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_207:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_224
; %bb.208:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_225
.LBB3_209:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_226
.LBB3_210:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_227
.LBB3_211:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_213
.LBB3_212:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v86
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v190
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_213:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_228
; %bb.214:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_229
.LBB3_215:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_230
.LBB3_216:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_231
.LBB3_217:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_219
.LBB3_218:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v82
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v190
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_219:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_232
; %bb.220:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_233
.LBB3_221:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_234
.LBB3_222:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_235
.LBB3_223:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_236
.LBB3_224:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v91
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v191
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_209
.LBB3_225:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v92
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v192
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_210
.LBB3_226:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v93
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v193
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_211
.LBB3_227:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_212
	s_branch .LBB3_213
.LBB3_228:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v87
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v191
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_215
.LBB3_229:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v88
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v192
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_216
.LBB3_230:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v89
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v193
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_217
.LBB3_231:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_218
	s_branch .LBB3_219
.LBB3_232:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v83
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v191
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_221
.LBB3_233:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v84
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v192
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_222
.LBB3_234:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v85
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v193
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_223
.LBB3_235:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_236:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v194
	v_cmp_gt_i32_e64 s[0:1], s87, v194
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_238
; %bb.237:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v78
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v194
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_238:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v195
	v_cmp_gt_i32_e64 s[0:1], s87, v195
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_240
; %bb.239:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v79
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v195
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_240:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v196
	v_cmp_gt_i32_e64 s[0:1], s87, v196
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_242
; %bb.241:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v80
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v196
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_242:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v197
	v_cmp_gt_i32_e64 s[0:1], s87, v197
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_244
; %bb.243:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v81
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v197
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_244:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_444
; %bb.245:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_247
.LBB3_246:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v74
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v194
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_247:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_264
; %bb.248:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_265
.LBB3_249:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_266
.LBB3_250:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_267
.LBB3_251:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_253
.LBB3_252:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v70
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v194
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_253:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_268
; %bb.254:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_269
.LBB3_255:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_270
.LBB3_256:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_271
.LBB3_257:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_259
.LBB3_258:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v66
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v194
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_259:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_272
; %bb.260:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_273
.LBB3_261:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_274
.LBB3_262:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_275
.LBB3_263:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_276
.LBB3_264:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v75
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v195
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_249
.LBB3_265:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v76
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v196
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_250
.LBB3_266:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v77
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v197
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_251
.LBB3_267:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_252
	s_branch .LBB3_253
.LBB3_268:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v71
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v195
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_255
.LBB3_269:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v72
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v196
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_256
.LBB3_270:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v73
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v197
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_257
.LBB3_271:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_258
	s_branch .LBB3_259
.LBB3_272:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v67
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v195
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_261
.LBB3_273:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v68
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v196
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_262
.LBB3_274:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v69
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v197
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_263
.LBB3_275:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_276:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v198
	v_cmp_gt_i32_e64 s[0:1], s87, v198
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_278
; %bb.277:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v62
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v198
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_278:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v199
	v_cmp_gt_i32_e64 s[0:1], s87, v199
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_280
; %bb.279:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v63
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v199
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_280:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v200
	v_cmp_gt_i32_e64 s[0:1], s87, v200
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_282
; %bb.281:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v64
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v200
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_282:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v201
	v_cmp_gt_i32_e64 s[0:1], s87, v201
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_284
; %bb.283:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v65
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v201
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_284:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_445
; %bb.285:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_287
.LBB3_286:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v58
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v198
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_287:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_304
; %bb.288:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_305
.LBB3_289:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_306
.LBB3_290:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_307
.LBB3_291:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_293
.LBB3_292:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v54
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v198
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_293:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_308
; %bb.294:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_309
.LBB3_295:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_310
.LBB3_296:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_311
.LBB3_297:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_299
.LBB3_298:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v50
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v198
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_299:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_312
; %bb.300:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_313
.LBB3_301:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_314
.LBB3_302:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_315
.LBB3_303:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_316
.LBB3_304:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v59
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v199
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_289
.LBB3_305:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v60
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v200
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_290
.LBB3_306:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v61
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v201
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_291
.LBB3_307:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_292
	s_branch .LBB3_293
.LBB3_308:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v55
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v199
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_295
.LBB3_309:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v56
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v200
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_296
.LBB3_310:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v57
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v201
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_297
.LBB3_311:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_298
	s_branch .LBB3_299
.LBB3_312:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v51
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v199
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_301
.LBB3_313:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v52
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v200
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_302
.LBB3_314:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v53
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v201
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_303
.LBB3_315:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_316:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v202
	v_cmp_gt_i32_e64 s[0:1], s87, v202
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_318
; %bb.317:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v46
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v202
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_318:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v203
	v_cmp_gt_i32_e64 s[0:1], s87, v203
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_320
; %bb.319:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v47
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v203
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_320:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v204
	v_cmp_gt_i32_e64 s[0:1], s87, v204
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_322
; %bb.321:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v48
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v204
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_322:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v205
	v_cmp_gt_i32_e64 s[0:1], s87, v205
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_324
; %bb.323:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v49
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v205
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_324:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_446
; %bb.325:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_327
.LBB3_326:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v42
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v202
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_327:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_344
; %bb.328:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_345
.LBB3_329:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_346
.LBB3_330:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_347
.LBB3_331:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_333
.LBB3_332:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v38
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v202
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_333:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_348
; %bb.334:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_349
.LBB3_335:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_350
.LBB3_336:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_351
.LBB3_337:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_339
.LBB3_338:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v34
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v202
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_339:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_352
; %bb.340:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_353
.LBB3_341:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_354
.LBB3_342:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_355
.LBB3_343:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_356
.LBB3_344:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v43
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v203
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_329
.LBB3_345:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v44
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v204
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_330
.LBB3_346:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v45
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v205
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_331
.LBB3_347:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_332
	s_branch .LBB3_333
.LBB3_348:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v39
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v203
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_335
.LBB3_349:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v40
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v204
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_336
.LBB3_350:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v41
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v205
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_337
.LBB3_351:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_338
	s_branch .LBB3_339
.LBB3_352:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v35
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v203
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_341
.LBB3_353:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v36
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v204
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_342
.LBB3_354:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v37
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v205
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_343
.LBB3_355:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_356:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v206
	v_cmp_gt_i32_e64 s[0:1], s87, v206
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_358
; %bb.357:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v30
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v206
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_358:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v208
	v_cmp_gt_i32_e64 s[0:1], s87, v208
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_360
; %bb.359:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v31
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v208
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_360:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v209
	v_cmp_gt_i32_e64 s[0:1], s87, v209
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_362
; %bb.361:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v32
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v209
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_362:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v210
	v_cmp_gt_i32_e64 s[0:1], s87, v210
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_364
; %bb.363:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v33
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v210
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_364:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_447
; %bb.365:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_367
.LBB3_366:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v26
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v206
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_367:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_384
; %bb.368:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_385
.LBB3_369:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_386
.LBB3_370:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_387
.LBB3_371:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_373
.LBB3_372:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v22
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v206
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_373:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_388
; %bb.374:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_389
.LBB3_375:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_390
.LBB3_376:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_391
.LBB3_377:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_379
.LBB3_378:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v18
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v206
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_379:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_392
; %bb.380:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_393
.LBB3_381:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_394
.LBB3_382:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_395
.LBB3_383:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB3_396
.LBB3_384:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v27
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v208
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_369
.LBB3_385:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v28
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v209
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_370
.LBB3_386:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v29
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v210
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_371
.LBB3_387:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_372
	s_branch .LBB3_373
.LBB3_388:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v23
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v208
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_375
.LBB3_389:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v24
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v209
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_376
.LBB3_390:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v25
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v210
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_377
.LBB3_391:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_378
	s_branch .LBB3_379
.LBB3_392:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v19
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v208
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_381
.LBB3_393:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v20
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v209
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_382
.LBB3_394:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v21
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v210
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_383
.LBB3_395:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
.LBB3_396:                              ;   in Loop: Header=BB3_113 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v211
	v_cmp_gt_i32_e64 s[0:1], s87, v211
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_398
; %bb.397:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v14
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v211
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_398:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v212
	v_cmp_gt_i32_e64 s[0:1], s87, v212
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_400
; %bb.399:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v15
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v212
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_400:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v214
	v_cmp_gt_i32_e64 s[0:1], s87, v214
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_402
; %bb.401:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v16
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v214
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143
.LBB3_402:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v215
	v_cmp_gt_i32_e64 s[0:1], s87, v215
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_404
; %bb.403:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v17
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v215
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142
.LBB3_404:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_448
; %bb.405:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_407
.LBB3_406:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v10
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v211
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB3_407:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_431
; %bb.408:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_432
.LBB3_409:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_433
.LBB3_410:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_434
.LBB3_411:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_413
.LBB3_412:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v6
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v211
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB3_413:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_435
; %bb.414:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_436
.LBB3_415:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_437
.LBB3_416:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_438
.LBB3_417:                              ;   in Loop: Header=BB3_113 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[14:15], s[78:79]
	s_cbranch_execz .LBB3_419
.LBB3_418:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v2
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v211
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB3_419:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[80:81]
	s_cbranch_execnz .LBB3_439
; %bb.420:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[82:83]
	s_cbranch_execnz .LBB3_440
.LBB3_421:                              ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB3_423
.LBB3_422:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v5
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v215
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:96
.LBB3_423:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi128ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_mul_i32 s0, s61, s43
	s_mul_hi_u32 s1, s60, s43
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s43
	v_lshl_add_u64 v[142:143], s[0:1], 1, v[138:139]
	s_andn2_b64 vcc, exec, s[66:67]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB3_449
; %bb.424:                              ;   in Loop: Header=BB3_113 Depth=3
	s_mov_b64 s[14:15], -1
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_451
; %bb.425:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v144, v160
	v_mov_b32_e32 v145, v207
	v_mov_b32_e32 v154, v216
	v_mov_b32_e32 v158, v0
                                        ; implicit-def: $sgpr78_sgpr79
                                        ; implicit-def: $sgpr80_sgpr81
	s_branch .LBB3_427
.LBB3_426:                              ; %Flow2294
                                        ;   in Loop: Header=BB3_427 Depth=4
	s_or_b64 exec, exec, s[86:87]
	s_and_b64 s[82:83], exec, s[84:85]
	s_or_b64 s[14:15], s[82:83], s[14:15]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[82:83], s[80:81], exec
	s_or_b64 s[78:79], s[78:79], s[82:83]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_450
.LBB3_427:                              ; %.lr.ph.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ;       Parent Loop BB3_113 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v159, 0xf8, v144
	v_add_u32_e32 v224, 8, v159
	v_cmp_lt_i32_e64 s[82:83], s17, v224
	v_cmp_ge_i32_e32 vcc, s17, v224
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB3_429
; %bb.428:                              ;   in Loop: Header=BB3_427 Depth=4
	v_mul_lo_u32 v224, s60, v145
	v_lshlrev_b32_e32 v224, 1, v224
	v_lshlrev_b32_e32 v159, 1, v159
	v_add3_u32 v224, v142, v224, v159
	v_add_u32_e32 v159, v154, v159
	v_or_b32_e32 v159, v159, v224
	v_and_b32_e32 v159, 15, v159
	v_cmp_eq_u32_e32 vcc, 0, v159
	s_andn2_b64 s[82:83], s[82:83], exec
	s_and_b64 s[86:87], vcc, exec
	s_or_b64 s[82:83], s[82:83], s[86:87]
.LBB3_429:                              ; %Flow2293
                                        ;   in Loop: Header=BB3_427 Depth=4
	s_or_b64 exec, exec, s[84:85]
	s_mov_b64 s[84:85], -1
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[86:87], s[82:83]
	s_cbranch_execz .LBB3_426
; %bb.430:                              ; %.critedge.i
                                        ;   in Loop: Header=BB3_427 Depth=4
	v_add_u32_e32 v158, 0x200, v158
	v_cmp_le_i32_e32 vcc, s92, v158
	v_add_u32_e32 v154, 0x2000, v154
	v_add_u32_e32 v145, 16, v145
	v_add_u32_e32 v144, 0x1000, v144
	s_or_b64 s[80:81], s[80:81], exec
	s_orn2_b64 s[84:85], vcc, exec
	s_branch .LBB3_426
.LBB3_431:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v11
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v212
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_409
.LBB3_432:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v12
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v214
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_410
.LBB3_433:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v13
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v215
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_411
.LBB3_434:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_412
	s_branch .LBB3_413
.LBB3_435:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v7
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v212
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_415
.LBB3_436:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v8
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v214
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_416
.LBB3_437:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v142, v142, v9
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s69
	v_subrev_u32_e32 v143, s86, v215
	v_lshl_add_u32 v143, v143, 9, v179
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_417
.LBB3_438:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[14:15], s[78:79]
	s_cbranch_execnz .LBB3_418
	s_branch .LBB3_419
.LBB3_439:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v3
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v212
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[82:83]
	s_cbranch_execz .LBB3_421
.LBB3_440:                              ;   in Loop: Header=BB3_113 Depth=3
	v_add_f32_e32 v143, v142, v4
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s69
	v_subrev_u32_e32 v144, s86, v214
	v_lshl_add_u32 v144, v144, 9, v179
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execnz .LBB3_422
	s_branch .LBB3_423
.LBB3_441:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_126
	s_branch .LBB3_127
.LBB3_442:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_166
	s_branch .LBB3_167
.LBB3_443:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_206
	s_branch .LBB3_207
.LBB3_444:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_246
	s_branch .LBB3_247
.LBB3_445:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_286
	s_branch .LBB3_287
.LBB3_446:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_326
	s_branch .LBB3_327
.LBB3_447:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_366
	s_branch .LBB3_367
.LBB3_448:                              ;   in Loop: Header=BB3_113 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_406
	s_branch .LBB3_407
.LBB3_449:                              ;   in Loop: Header=BB3_113 Depth=3
	s_cbranch_execz .LBB3_112
	s_branch .LBB3_469
.LBB3_450:                              ; %Flow2295
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_orn2_b64 s[14:15], s[78:79], exec
.LBB3_451:                              ; %Flow2296
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v144, 0, 1, s[14:15]
	s_nop 0
	v_readfirstlane_b32 s0, v144
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[14:15], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_456
; %bb.452:                              ;   in Loop: Header=BB3_113 Depth=3
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB3_455
; %bb.453:                              ; %.lr.ph.i346.preheader
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v145, v223
	v_mov_b32_e32 v144, v0
.LBB3_454:                              ; %.lr.ph.i346
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ;       Parent Loop BB3_113 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v154, v144, v222
	v_mul_lo_u32 v158, v154, s56
	v_sub_u32_e32 v158, v144, v158
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_subrev_u32_e32 v159, s56, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_xor_b32_e32 v154, s95, v154
	v_subrev_u32_e32 v224, s95, v154
	v_mad_u64_u32 v[158:159], s[78:79], s10, v224, v[144:145]
	v_lshlrev_b32_e32 v154, 9, v154
	v_mul_lo_u32 v159, s89, v224
	v_sub_u32_e32 v154, v154, v159
	v_add_u32_e32 v154, v145, v154
	ds_read_u16 v154, v154
	v_mad_i64_i32 v[224:225], s[78:79], s60, v224, 0
	v_add_u32_e32 v144, 0x200, v144
	v_mov_b32_e32 v159, v155
	v_lshl_add_u64 v[224:225], v[224:225], 1, v[142:143]
	v_cmp_le_i32_e32 vcc, s57, v144
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[224:225]
	v_add_u32_e32 v145, 0x400, v145
	s_or_b64 s[14:15], vcc, s[14:15]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v154
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_454
.LBB3_455:                              ; %Flow2285
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_456:                              ; %Flow2291
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB3_468
; %bb.457:                              ;   in Loop: Header=BB3_113 Depth=3
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_467
; %bb.458:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v224, v217
	v_mov_b64_e32 v[144:145], v[140:141]
	v_mov_b32_e32 v225, v0
	s_branch .LBB3_461
.LBB3_459:                              ; %Flow2287
                                        ;   in Loop: Header=BB3_461 Depth=4
	s_or_b64 exec, exec, s[80:81]
.LBB3_460:                              ; %.loopexit.i
                                        ;   in Loop: Header=BB3_461 Depth=4
	s_or_b64 exec, exec, s[78:79]
	v_add_u32_e32 v225, 0x200, v225
	v_cmp_le_i32_e32 vcc, s92, v225
	v_lshl_add_u64 v[144:145], v[144:145], 0, s[74:75]
	s_or_b64 s[14:15], vcc, s[14:15]
	v_add_u32_e32 v224, 0x2000, v224
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_467
.LBB3_461:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ;       Parent Loop BB3_113 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB3_466 Depth 5
	v_lshlrev_b32_e32 v154, 3, v225
	v_and_b32_e32 v154, 0xf8, v154
	v_add_u32_e32 v158, 8, v154
	v_cmp_ge_i32_e32 vcc, s17, v158
	s_and_saveexec_b64 s[78:79], vcc
	s_xor_b64 s[78:79], exec, s[78:79]
	s_cbranch_execz .LBB3_463
; %bb.462:                              ;   in Loop: Header=BB3_461 Depth=4
	v_lshrrev_b32_e32 v226, 5, v225
	v_mad_i64_i32 v[158:159], s[80:81], s60, v226, 0
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[142:143]
	v_lshlrev_b32_e32 v154, 1, v154
	v_lshlrev_b32_e32 v226, 9, v226
	v_lshl_add_u64 v[158:159], v[158:159], 0, v[154:155]
	v_add3_u32 v154, 0, v226, v154
	ds_read_b128 v[226:229], v154
                                        ; implicit-def: $vgpr154
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[158:159], v[226:229]
.LBB3_463:                              ; %Flow2288
                                        ;   in Loop: Header=BB3_461 Depth=4
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB3_460
; %bb.464:                              ; %.preheader.i
                                        ;   in Loop: Header=BB3_461 Depth=4
	v_cmp_gt_i32_e32 vcc, s16, v154
	s_and_saveexec_b64 s[80:81], vcc
	s_cbranch_execz .LBB3_459
; %bb.465:                              ; %.lr.ph.i348
                                        ;   in Loop: Header=BB3_461 Depth=4
	s_mov_b32 s84, 0
	s_mov_b64 s[82:83], 0
	v_mov_b32_e32 v154, v224
	v_mov_b64_e32 v[158:159], v[144:145]
.LBB3_466:                              ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ;       Parent Loop BB3_113 Depth=3
                                        ;         Parent Loop BB3_461 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v226, v154
	s_add_i32 s3, s84, 1
	s_cmp_gt_u32 s84, 6
	s_cselect_b64 s[86:87], -1, 0
	v_add_u32_e32 v154, 2, v154
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v226
	v_add_u32_e32 v226, s84, v213
	v_cmp_le_u32_e32 vcc, s17, v226
	s_or_b64 s[84:85], s[86:87], vcc
	s_and_b64 s[84:85], exec, s[84:85]
	v_lshl_add_u64 v[158:159], v[158:159], 0, 2
	s_or_b64 s[82:83], s[84:85], s[82:83]
	s_mov_b32 s84, s3
	s_andn2_b64 exec, exec, s[82:83]
	s_cbranch_execnz .LBB3_466
	s_branch .LBB3_459
.LBB3_467:                              ; %Flow2290
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB3_468:                              ; %Flow2292
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_branch .LBB3_112
.LBB3_469:                              ;   in Loop: Header=BB3_113 Depth=3
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB3_111
; %bb.470:                              ; %.lr.ph.i352.preheader
                                        ;   in Loop: Header=BB3_113 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v145, v223
	v_mov_b32_e32 v144, v0
.LBB3_471:                              ; %.lr.ph.i352
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_109 Depth=2
                                        ;       Parent Loop BB3_113 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v154, v144, v222
	v_mul_lo_u32 v158, v154, s56
	v_sub_u32_e32 v158, v144, v158
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_subrev_u32_e32 v159, s56, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_xor_b32_e32 v154, s95, v154
	v_subrev_u32_e32 v224, s95, v154
	v_mad_u64_u32 v[158:159], s[78:79], s10, v224, v[144:145]
	v_lshlrev_b32_e32 v154, 9, v154
	v_mul_lo_u32 v159, s89, v224
	v_sub_u32_e32 v154, v154, v159
	v_add_u32_e32 v154, v145, v154
	ds_read_u16 v154, v154
	v_mad_i64_i32 v[224:225], s[78:79], s60, v224, 0
	v_add_u32_e32 v144, 0x200, v144
	v_mov_b32_e32 v159, v155
	v_lshl_add_u64 v[224:225], v[224:225], 1, v[142:143]
	v_cmp_le_i32_e32 vcc, s57, v144
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[224:225]
	v_add_u32_e32 v145, 0x400, v145
	s_or_b64 s[14:15], vcc, s[14:15]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v154
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_471
	s_branch .LBB3_111
.LBB3_472:                              ; %._crit_edge978
                                        ;   in Loop: Header=BB3_84 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s10, v245, 0
	v_readlane_b32 s11, v245, 1
	s_and_b64 s[10:11], s[0:1], s[10:11]
	s_mov_b64 exec, s[10:11]
	s_cbranch_execz .LBB3_474
; %bb.473:                              ;   in Loop: Header=BB3_84 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB3_474:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s10, v245, 21
	v_readlane_b32 s11, v245, 22
	s_and_b64 s[10:11], s[0:1], s[10:11]
	v_readlane_b32 s43, v245, 35
	v_readlane_b32 s52, v245, 37
	v_readlane_b32 s82, v245, 38
	s_mov_b64 exec, s[10:11]
	s_cbranch_execz .LBB3_82
; %bb.475:                              ; %.lr.ph980.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_lshl_b32 s10, s45, 2
	s_add_i32 s10, s2, s10
	s_sub_i32 s4, s10, s4
	s_sub_i32 s4, s4, s5
	s_lshl_b32 s5, s7, 2
	s_sub_i32 s4, s4, s5
	s_lshl_b32 s4, s4, 8
	s_mov_b32 s5, s21
	s_branch .LBB3_477
.LBB3_476:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB3_477 Depth=2
	s_add_i32 s5, s5, -1
	s_add_i32 s4, s4, s26
	s_cmp_lg_u32 s5, 0
	s_cbranch_scc0 .LBB3_82
.LBB3_477:                              ; %.lr.ph980
                                        ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[8:9], s[40:41]
	flat_load_dwordx4 v[2:5], v[8:9]
	s_abs_i32 s7, s4
	s_mul_hi_u32 s10, s7, s59
	s_mul_i32 s11, s10, s99
	s_ashr_i32 s3, s4, 31
	s_sub_i32 s7, s7, s11
	s_xor_b32 s3, s3, s58
	s_add_i32 s11, s10, 1
	s_sub_i32 s12, s7, s99
	s_cmp_ge_u32 s7, s99
	s_cselect_b32 s10, s11, s10
	s_cselect_b32 s7, s12, s7
	s_add_i32 s11, s10, 1
	s_cmp_ge_u32 s7, s99
	s_cselect_b32 s7, s11, s10
	s_xor_b32 s7, s7, s3
	s_sub_i32 s7, s7, s3
	s_mul_i32 s10, s68, s7
	s_add_i32 s10, s4, s10
	s_mul_i32 s3, s7, s33
	s_ashr_i32 s10, s10, 31
	s_sub_i32 s3, s10, s3
	s_add_i32 s3, s4, s3
	s_xor_b32 s3, s3, s10
	s_xor_b32 s11, s10, s97
	s_mul_hi_u32 s10, s3, s98
	s_mul_i32 s12, s10, s94
	s_sub_i32 s3, s3, s12
	s_add_i32 s12, s10, 1
	s_sub_i32 s13, s3, s94
	s_cmp_ge_u32 s3, s94
	s_cselect_b32 s10, s12, s10
	s_cselect_b32 s3, s13, s3
	s_add_i32 s12, s10, 1
	s_cmp_ge_u32 s3, s94
	s_cselect_b32 s3, s12, s10
	s_xor_b32 s3, s3, s11
	s_sub_i32 s3, s3, s11
	s_mul_i32 s10, s24, s20
	s_add_i32 s3, s3, s10
	s_mul_i32 s3, s3, s25
	s_add_i32 s10, s3, s6
	s_ashr_i32 s11, s10, 31
	s_lshl_b64 s[10:11], s[10:11], 2
	s_add_u32 s3, s38, s10
	s_addc_u32 s10, s39, s11
	s_add_u32 s11, s3, 0x100
	s_addc_u32 s10, s10, 0
	s_cmp_eq_u32 s7, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 1
	s_mov_b64 s[12:13], -1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v10, 0, v4, vcc
	v_cndmask_b32_e32 v11, 0, v5, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:16
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v10, v4, v6, vcc
	v_cndmask_b32_e32 v11, v5, v7, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:32
	s_cmp_eq_u32 s7, 3
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v10, v4, v6, vcc
	v_cndmask_b32_e32 v11, v5, v7, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:48
	s_cmp_eq_u32 s7, 5
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 6
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v6, v4, v6, vcc
	v_cndmask_b32_e32 v7, v5, v7, vcc
	flat_load_dwordx2 v[4:5], v[8:9] offset:64
	s_cmp_eq_u32 s7, 7
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s7, s20
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v7, v5, vcc
	v_cndmask_b32_e32 v4, v6, v4, vcc
	v_sub_co_u32_e32 v2, vcc, s11, v2
	v_mov_b32_e32 v6, s10
	s_nop 0
	v_subb_co_u32_e32 v3, vcc, v6, v3, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_nop 1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB3_479
; %bb.478:                              ; %Flow2281
                                        ;   in Loop: Header=BB3_477 Depth=2
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_476
	s_branch .LBB3_480
.LBB3_479:                              ;   in Loop: Header=BB3_477 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB3_476
.LBB3_480:                              ;   in Loop: Header=BB3_477 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB3_476
.LBB3_481:                              ; %.critedge262
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 246
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 248
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals,comdat
.Lfunc_end3:
	.size	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals, .Lfunc_end3-_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.num_vgpr, 246
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 21632
; TotalNumSgprs: 106
; NumVgprs: 246
; NumAgprs: 0
; TotalNumVgprs: 246
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 246
; AccumOffset: 248
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 61
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[56:57], s[0:1], 0x60
	s_load_dwordx8 s[36:43], s[0:1], 0x100
	s_load_dwordx8 s[44:51], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
                                        ; implicit-def: $vgpr247 : SGPR spill to VGPR lane
	s_load_dwordx2 s[58:59], s[0:1], 0xf8
	s_load_dwordx2 s[18:19], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s52, s37, 31
	s_lshr_b32 s3, s52, 29
	v_writelane_b32 v247, s4, 0
	s_add_i32 s3, s37, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v247, s5, 1
	v_writelane_b32 v247, s6, 2
	v_writelane_b32 v247, s7, 3
	s_cmp_ge_i32 s2, s43
	s_mov_b64 s[4:5], -1
	s_cbranch_scc0 .LBB4_72
; %bb.1:
	s_sub_i32 s16, s2, s43
	s_mov_b32 s17, 0
	s_lshl_b64 s[4:5], s[16:17], 2
	s_add_u32 s6, s50, s4
	s_addc_u32 s7, s51, s5
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB4_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB4_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit365
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_eq_u64 s[10:11], 0
	s_cbranch_scc1 .LBB4_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB4_6
; %bb.5:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB4_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit368
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execnz .LBB4_8
	s_branch .LBB4_71
.LBB4_7:
	s_mov_b64 s[6:7], -1
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execz .LBB4_71
.LBB4_8:                                ; %.critedge
	s_mul_i32 s91, s41, s40
	s_cmp_ge_i32 s16, s91
	s_cbranch_scc1 .LBB4_71
; %bb.9:                                ; %.lr.ph
	s_mul_hi_i32 s13, s33, s38
	s_mul_i32 s12, s33, s38
	s_ashr_i32 s25, s38, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_add_u32 s26, s56, s6
	s_addc_u32 s27, s57, s7
	s_add_u32 s28, s26, s6
	s_addc_u32 s29, s27, s7
	s_add_u32 s30, s28, s6
	s_addc_u32 s31, s29, s7
	s_add_u32 s34, s30, s6
	s_addc_u32 s35, s31, s7
	s_add_u32 s60, s34, s6
	s_addc_u32 s61, s35, s7
	s_add_u32 s62, s60, s6
	s_addc_u32 s63, s61, s7
	s_add_u32 s64, s62, s6
	s_load_dwordx2 s[22:23], s[0:1], 0x90
	s_addc_u32 s65, s63, s7
	s_load_dwordx2 s[6:7], s[0:1], 0x120
	s_mov_b32 s24, s38
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v21, 0
	v_mul_lo_u32 v64, s40, v0
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s17, s42, 5
	s_add_i32 s3, s6, 64
	s_cmp_lg_u32 s36, 0
	v_writelane_b32 v247, s3, 45
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 46
	s_cmp_lg_u32 s36, 1
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v247, s7, 47
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 48
	s_cmp_lg_u32 s36, 2
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v247, s7, 49
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 50
	s_cmp_lg_u32 s36, 3
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v247, s7, 51
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 29
	s_cmp_lg_u32 s36, 4
	v_lshrrev_b32_e32 v18, 5, v0
	v_writelane_b32 v247, s7, 30
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 32
	s_cmp_lg_u32 s36, 5
	v_mov_b32_e32 v19, v21
	v_writelane_b32 v247, s7, 33
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 25
	s_cmp_lg_u32 s36, 6
	s_mov_b64 s[98:99], 0
	v_writelane_b32 v247, s7, 26
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v247, s6, 7
	s_cmp_lg_u32 s36, 7
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v247, s7, 8
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s54, s41
	v_cvt_f32_u32_e32 v2, s54
	s_sub_i32 s3, 0, s54
	s_ashr_i32 s55, s41, 31
	s_lshl_b64 s[82:83], s[24:25], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v247, s6, 9
	s_mov_b32 s88, 0x7060302
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v247, s7, 10
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_readfirstlane_b32 s53, v2
	s_mul_i32 s3, s3, s53
	s_mul_hi_u32 s3, s53, s3
	s_add_i32 s53, s53, s3
	s_add_u32 s66, s22, 2
	s_addc_u32 s67, s23, 0
	v_writelane_b32 v247, s66, 16
	s_mul_i32 s3, s13, 14
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_writelane_b32 v247, s67, 17
	s_mul_hi_u32 s66, s12, 14
	s_add_i32 s66, s66, s3
	s_mul_i32 s3, s12, 14
	s_add_u32 s3, s56, s3
	s_addc_u32 s66, s57, s66
	s_add_u32 s68, s3, 2
	s_addc_u32 s69, s66, 0
	s_lshl_b64 s[66:67], s[12:13], 2
	v_writelane_b32 v247, s68, 14
	s_add_u32 s66, s56, s66
	s_addc_u32 s67, s57, s67
	v_writelane_b32 v247, s69, 15
	v_writelane_b32 v247, s66, 18
	s_mul_i32 s3, s13, 12
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v247, s67, 19
	s_mul_hi_u32 s66, s12, 12
	s_add_i32 s66, s66, s3
	s_mul_i32 s3, s12, 12
	s_add_u32 s3, s56, s3
	s_addc_u32 s66, s57, s66
	s_add_u32 s68, s3, 2
	s_addc_u32 s69, s66, 0
	s_mul_i32 s3, s13, 6
	s_mul_hi_u32 s66, s12, 6
	s_add_i32 s66, s66, s3
	s_mul_i32 s3, s12, 6
	s_add_u32 s92, s56, s3
	s_addc_u32 s93, s57, s66
	s_mul_i32 s3, s13, 10
	s_mul_hi_u32 s66, s12, 10
	s_add_i32 s66, s66, s3
	s_mul_i32 s3, s12, 10
	s_add_u32 s3, s56, s3
	s_addc_u32 s66, s57, s66
	s_add_u32 s94, s3, 2
	s_addc_u32 s95, s66, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	s_add_u32 s96, s56, s12
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v247, s68, 39
	s_addc_u32 s97, s57, s13
	s_xor_b64 s[66:67], s[14:15], -1
	s_movk_i32 s3, 0x7fff
	v_and_b32_e32 v67, 0x100, v2
	v_writelane_b32 v247, s69, 40
	s_branch .LBB4_12
.LBB4_10:                               ; %Flow2283
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_sub_i32 s12, s16, s43
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s91
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB4_11:                               ; %Flow2295
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB4_71
.LBB4_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_16 Depth 2
                                        ;     Child Loop BB4_28 Depth 2
                                        ;       Child Loop BB4_32 Depth 3
	s_abs_i32 s13, s16
	s_mul_hi_u32 s14, s13, s53
	s_mul_i32 s15, s14, s54
	s_ashr_i32 s12, s16, 31
	s_sub_i32 s13, s13, s15
	s_xor_b32 s12, s12, s55
	s_add_i32 s15, s14, 1
	s_sub_i32 s68, s13, s54
	s_cmp_ge_u32 s13, s54
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s13, s68, s13
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s13, s54
	s_cselect_b32 s13, s15, s14
	s_xor_b32 s13, s13, s12
	s_sub_i32 s89, s13, s12
	s_mul_i32 s12, s89, s41
	s_sub_i32 s90, s16, s12
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB4_20
; %bb.13:                               ;   in Loop: Header=BB4_12 Depth=1
	v_add_u32_e32 v2, s89, v64
	v_mul_lo_u32 v2, v2, s41
	v_add_u32_e32 v2, s90, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[46:47]
	flat_load_dword v4, v[2:3] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB4_20
; %bb.14:                               ; %.lr.ph.i.i.i370.preheader
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[72:73], 0
                                        ; implicit-def: $sgpr68_sgpr69
                                        ; implicit-def: $sgpr70_sgpr71
	s_branch .LBB4_16
.LBB4_15:                               ; %Flow2291
                                        ;   in Loop: Header=BB4_16 Depth=2
	s_and_b64 s[76:77], exec, s[70:71]
	s_or_b64 s[14:15], s[76:77], s[14:15]
	s_andn2_b64 s[68:69], s[68:69], exec
	s_and_b64 s[74:75], s[74:75], exec
	s_or_b64 s[68:69], s[68:69], s[74:75]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB4_18
.LBB4_16:                               ; %.lr.ph.i.i.i370
                                        ;   Parent Loop BB4_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s72, s72, 1
	s_addc_u32 s73, s73, 0
	v_mov_b64_e32 v[4:5], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[72:73], v[4:5]
	s_mov_b64 s[74:75], -1
	s_or_b64 s[70:71], s[70:71], exec
	s_cbranch_vccnz .LBB4_15
; %bb.17:                               ;   in Loop: Header=BB4_16 Depth=2
	s_sleep 4
	flat_load_dword v4, v[2:3] offset:256 sc0 sc1
	s_andn2_b64 s[70:71], s[70:71], exec
	s_mov_b64 s[74:75], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[76:77], vcc, exec
	s_or_b64 s[70:71], s[70:71], s[76:77]
	s_branch .LBB4_15
.LBB4_18:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[68:69]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB4_20
; %bb.19:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_load_dwordx4 s[68:71], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[70:71]
	flat_atomic_or v[2:3], v66
.LBB4_20:                               ; %.critedge849
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[18:19]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB4_24
; %bb.21:                               ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b32_e32 v2, 0
	s_and_saveexec_b64 s[14:15], s[8:9]
	s_cbranch_execz .LBB4_23
; %bb.22:                               ;   in Loop: Header=BB4_12 Depth=1
	s_load_dwordx4 s[68:71], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[70:71]
	flat_load_dword v2, v[2:3] sc1
.LBB4_23:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit376
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v67, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x4000000, v2
	v_cmp_eq_u32_e64 s[14:15], 0, v2
.LBB4_24:                               ; %Flow2294
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_and_saveexec_b64 s[68:69], s[14:15]
	s_cbranch_execz .LBB4_11
; %bb.25:                               ; %.critedge871
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB4_38
; %bb.26:                               ; %.lr.ph.i377.preheader
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_mul_i32 s70, s89, s42
	s_lshl_b32 s72, s90, 8
	s_ashr_i32 s71, s70, 31
	s_ashr_i32 s73, s72, 31
	v_lshl_add_u64 v[2:3], v[18:19], 0, s[70:71]
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mad_u64_u32 v[4:5], s[14:15], s24, v2, v[4:5]
	v_mul_lo_u32 v3, s24, v3
	v_mul_lo_u32 v2, s25, v2
	v_add3_u32 v5, v2, v5, v3
	v_readlane_b32 s14, v247, 16
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_readlane_b32 s15, v247, 17
	v_lshl_add_u64 v[22:23], s[56:57], 0, v[2:3]
	v_lshl_add_u64 v[26:27], s[26:27], 0, v[2:3]
	v_lshl_add_u64 v[24:25], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v247, 14
	v_readlane_b32 s15, v247, 15
	v_lshl_add_u64 v[34:35], s[92:93], 0, v[2:3]
	v_lshl_add_u64 v[36:37], s[94:95], 0, v[2:3]
	v_lshl_add_u64 v[28:29], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v247, 18
	v_readlane_b32 s15, v247, 19
	v_lshl_add_u64 v[38:39], s[96:97], 0, v[2:3]
	s_mov_b64 s[74:75], 0
	v_lshl_add_u64 v[30:31], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v247, 39
	v_readlane_b32 s15, v247, 40
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[32:33], s[14:15], 0, v[2:3]
	s_branch .LBB4_28
.LBB4_27:                               ; %.critedge.i380
                                        ;   in Loop: Header=BB4_28 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s17, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[82:83]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[82:83]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[82:83]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[82:83]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[82:83]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[82:83]
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[82:83]
	s_or_b64 s[74:75], vcc, s[74:75]
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[82:83]
	s_andn2_b64 exec, exec, s[74:75]
	s_cbranch_execz .LBB4_38
.LBB4_28:                               ; %.lr.ph.i377
                                        ;   Parent Loop BB4_12 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB4_32 Depth 3
	v_lshlrev_b32_e32 v2, 3, v69
	v_and_b32_e32 v2, 0xf8, v2
	v_or_b32_e32 v2, s72, v2
	v_mov_b32_e32 v3, s73
	v_lshl_add_u64 v[4:5], v[2:3], 0, 8
	v_cmp_lt_u64_e32 vcc, s[24:25], v[4:5]
	s_or_b64 s[14:15], s[66:67], vcc
	s_and_saveexec_b64 s[76:77], s[14:15]
	s_xor_b64 s[76:77], exec, s[76:77]
	s_cbranch_execz .LBB4_36
; %bb.29:                               ; %.preheader.i379.preheader
                                        ;   in Loop: Header=BB4_28 Depth=2
	v_and_b32_e32 v20, 0xf8, v68
	v_lshlrev_b32_e32 v4, 1, v68
	v_lshl_add_u64 v[2:3], s[72:73], 0, v[20:21]
	v_and_b32_e32 v20, 0x1f0, v4
	v_lshl_add_u64 v[4:5], v[22:23], 0, v[20:21]
	v_lshl_add_u64 v[6:7], v[24:25], 0, v[20:21]
	v_lshl_add_u64 v[8:9], v[26:27], 0, v[20:21]
	v_lshl_add_u64 v[10:11], v[28:29], 0, v[20:21]
	v_lshl_add_u64 v[12:13], v[30:31], 0, v[20:21]
	v_lshl_add_u64 v[14:15], v[32:33], 0, v[20:21]
	v_lshl_add_u64 v[16:17], v[34:35], 0, v[20:21]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[20:21]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[20:21]
	s_mov_b64 s[78:79], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr80_sgpr81
	s_branch .LBB4_32
.LBB4_30:                               ; %Flow2285
                                        ;   in Loop: Header=BB4_32 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_b64 s[84:85], s[86:87], exec
	s_or_b64 s[80:81], s[80:81], s[84:85]
.LBB4_31:                               ; %Flow2284
                                        ;   in Loop: Header=BB4_32 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[80:81]
	s_or_b64 s[78:79], s[14:15], s[78:79]
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execz .LBB4_35
.LBB4_32:                               ; %.preheader.i379
                                        ;   Parent Loop BB4_12 Depth=1
                                        ;     Parent Loop BB4_28 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[24:25], v[2:3]
	s_or_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_31
; %bb.33:                               ; %.preheader.i379.1
                                        ;   in Loop: Header=BB4_32 Depth=3
	v_lshl_add_u64 v[46:47], v[4:5], 0, v[44:45]
	v_lshl_add_u64 v[48:49], v[8:9], 0, v[44:45]
	global_load_ushort v20, v[46:47], off
	global_load_ushort v50, v[48:49], off
	v_lshl_add_u64 v[62:63], v[10:11], 0, v[44:45]
	v_lshl_add_u64 v[70:71], v[2:3], 0, 1
	v_cmp_gt_u64_e32 vcc, s[24:25], v[70:71]
	s_mov_b64 s[86:87], -1
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v20, 16, v20
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v20, v20, v50
	v_lshl_add_u64 v[50:51], v[12:13], 0, v[44:45]
	global_load_ushort v52, v[50:51], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v20, v20, v52
	v_lshl_add_u64 v[52:53], v[16:17], 0, v[44:45]
	global_load_ushort v54, v[52:53], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v20, v20, v54
	v_lshl_add_u64 v[54:55], v[42:43], 0, v[44:45]
	global_load_ushort v56, v[54:55], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v56, 16, v56
	v_add_f32_e32 v20, v20, v56
	v_lshl_add_u64 v[56:57], v[40:41], 0, v[44:45]
	global_load_ushort v58, v[56:57], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v58, 16, v58
	v_add_f32_e32 v20, v20, v58
	v_lshl_add_u64 v[58:59], v[14:15], 0, v[44:45]
	global_load_ushort v60, v[58:59], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v60, 16, v60
	v_add_f32_e32 v20, v20, v60
	global_load_ushort v60, v[62:63], off offset:-2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v60, 16, v60
	v_add_f32_e32 v20, v20, v60
	v_bfe_u32 v60, v20, 16, 1
	v_add3_u32 v20, v20, v60, s3
	v_lshl_add_u64 v[60:61], v[6:7], 0, v[44:45]
	global_store_short_d16_hi v[60:61], v20, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB4_30
; %bb.34:                               ;   in Loop: Header=BB4_32 Depth=3
	global_load_ushort v20, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v46, v[46:47], off offset:2
	s_nop 0
	global_load_ushort v47, v[50:51], off offset:2
	global_load_ushort v48, v[52:53], off offset:2
	global_load_ushort v49, v[54:55], off offset:2
	s_nop 0
	global_load_ushort v50, v[56:57], off
	global_load_ushort v51, v[58:59], off
	global_load_ushort v52, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[2:3], v[2:3], 0, 2
	s_orn2_b64 s[86:87], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v20, 16, v20
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v46, 16, v46
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v47, 16, v47
	v_add_f32_e32 v20, v46, v20
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v48, 16, v48
	v_add_f32_e32 v20, v20, v47
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v20, v20, v48
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v20, v20, v49
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v20, v20, v50
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v20, v20, v51
	v_add_f32_e32 v20, v20, v52
	v_bfe_u32 v46, v20, 16, 1
	v_add3_u32 v20, v20, v46, s3
	global_store_short_d16_hi v[60:61], v20, off
	s_branch .LBB4_30
.LBB4_35:                               ; %Flow2286
                                        ;   in Loop: Header=BB4_28 Depth=2
	s_or_b64 exec, exec, s[78:79]
                                        ; implicit-def: $vgpr2_vgpr3
.LBB4_36:                               ; %Flow2287
                                        ;   in Loop: Header=BB4_28 Depth=2
	s_andn2_saveexec_b64 vcc, s[76:77]
	s_cbranch_execz .LBB4_27
; %bb.37:                               ;   in Loop: Header=BB4_28 Depth=2
	v_lshrrev_b32_e32 v20, 5, v69
	v_lshl_add_u64 v[4:5], v[20:21], 0, s[70:71]
	v_mul_lo_u32 v6, v4, s25
	v_mul_lo_u32 v5, v5, s24
	v_mad_u64_u32 v[2:3], s[14:15], v4, s24, v[2:3]
	v_add3_u32 v3, v5, v3, v6
	v_lshlrev_b64 v[40:41], 1, v[2:3]
	v_lshl_add_u64 v[2:3], s[56:57], 0, v[40:41]
	v_lshl_add_u64 v[6:7], s[26:27], 0, v[40:41]
	v_lshl_add_u64 v[10:11], s[28:29], 0, v[40:41]
	v_lshl_add_u64 v[14:15], s[30:31], 0, v[40:41]
	global_load_dwordx4 v[2:5], v[2:3], off
	v_lshl_add_u64 v[58:59], s[34:35], 0, v[40:41]
	global_load_dwordx4 v[6:9], v[6:7], off
	v_lshl_add_u64 v[60:61], s[60:61], 0, v[40:41]
	global_load_dwordx4 v[10:13], v[10:11], off
	v_lshl_add_u64 v[62:63], s[62:63], 0, v[40:41]
	global_load_dwordx4 v[14:17], v[14:15], off
	v_lshl_add_u64 v[70:71], s[64:65], 0, v[40:41]
	v_lshl_add_u64 v[40:41], s[22:23], 0, v[40:41]
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v73, 0xffff0000, v2
	v_and_b32_e32 v75, 0xffff0000, v3
	v_and_b32_e32 v55, 0xffff0000, v4
	v_and_b32_e32 v43, 0xffff0000, v5
	v_lshlrev_b32_e32 v72, 16, v2
	v_lshlrev_b32_e32 v74, 16, v3
	v_lshlrev_b32_e32 v54, 16, v4
	v_lshlrev_b32_e32 v42, 16, v5
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v77, 0xffff0000, v6
	v_and_b32_e32 v79, 0xffff0000, v7
	v_and_b32_e32 v57, 0xffff0000, v8
	v_and_b32_e32 v45, 0xffff0000, v9
	v_lshlrev_b32_e32 v76, 16, v6
	v_lshlrev_b32_e32 v78, 16, v7
	v_lshlrev_b32_e32 v56, 16, v8
	v_lshlrev_b32_e32 v44, 16, v9
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v81, 0xffff0000, v10
	v_and_b32_e32 v83, 0xffff0000, v11
	v_and_b32_e32 v47, 0xffff0000, v12
	v_and_b32_e32 v51, 0xffff0000, v13
	v_lshlrev_b32_e32 v80, 16, v10
	v_lshlrev_b32_e32 v82, 16, v11
	v_lshlrev_b32_e32 v46, 16, v12
	v_lshlrev_b32_e32 v50, 16, v13
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v85, 0xffff0000, v14
	v_and_b32_e32 v87, 0xffff0000, v15
	v_and_b32_e32 v53, 0xffff0000, v16
	v_and_b32_e32 v49, 0xffff0000, v17
	v_lshlrev_b32_e32 v84, 16, v14
	v_lshlrev_b32_e32 v86, 16, v15
	v_lshlrev_b32_e32 v52, 16, v16
	v_lshlrev_b32_e32 v48, 16, v17
	global_load_dwordx4 v[14:17], v[58:59], off
	global_load_dwordx4 v[10:13], v[60:61], off
	global_load_dwordx4 v[6:9], v[62:63], off
	global_load_dwordx4 v[2:5], v[70:71], off
	v_pk_add_f32 v[58:59], v[76:77], v[72:73]
	v_pk_add_f32 v[60:61], v[78:79], v[74:75]
	v_pk_add_f32 v[58:59], v[58:59], v[80:81]
	v_pk_add_f32 v[60:61], v[60:61], v[82:83]
	v_pk_add_f32 v[58:59], v[58:59], v[84:85]
	v_pk_add_f32 v[60:61], v[60:61], v[86:87]
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v62, 16, v14
	v_lshlrev_b32_e32 v70, 16, v15
	v_and_b32_e32 v63, 0xffff0000, v14
	v_and_b32_e32 v71, 0xffff0000, v15
	v_pk_add_f32 v[14:15], v[60:61], v[70:71]
	v_pk_add_f32 v[58:59], v[58:59], v[62:63]
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v60, 16, v11
	v_lshlrev_b32_e32 v62, 16, v10
	v_and_b32_e32 v61, 0xffff0000, v11
	v_and_b32_e32 v63, 0xffff0000, v10
	v_pk_add_f32 v[10:11], v[58:59], v[62:63]
	v_pk_add_f32 v[14:15], v[14:15], v[60:61]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v58, 16, v6
	v_lshlrev_b32_e32 v60, 16, v7
	v_and_b32_e32 v59, 0xffff0000, v6
	v_and_b32_e32 v61, 0xffff0000, v7
	v_pk_add_f32 v[6:7], v[14:15], v[60:61]
	v_pk_add_f32 v[10:11], v[10:11], v[58:59]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v14, 16, v3
	v_lshlrev_b32_e32 v58, 16, v2
	v_and_b32_e32 v15, 0xffff0000, v3
	v_and_b32_e32 v59, 0xffff0000, v2
	v_pk_add_f32 v[2:3], v[10:11], v[58:59]
	v_pk_add_f32 v[6:7], v[6:7], v[14:15]
	v_bfe_u32 v14, v3, 16, 1
	v_bfe_u32 v10, v7, 16, 1
	v_bfe_u32 v11, v6, 16, 1
	v_bfe_u32 v15, v2, 16, 1
	v_add3_u32 v20, v2, v15, s3
	v_add3_u32 v58, v3, v14, s3
	v_add3_u32 v59, v6, v11, s3
	v_add3_u32 v60, v7, v10, s3
	v_pk_add_f32 v[2:3], v[56:57], v[54:55]
	v_pk_add_f32 v[6:7], v[44:45], v[42:43]
	v_pk_add_f32 v[2:3], v[2:3], v[46:47]
	v_pk_add_f32 v[6:7], v[6:7], v[50:51]
	v_pk_add_f32 v[2:3], v[2:3], v[52:53]
	v_pk_add_f32 v[6:7], v[6:7], v[48:49]
	v_lshlrev_b32_e32 v10, 16, v16
	v_lshlrev_b32_e32 v14, 16, v17
	v_and_b32_e32 v11, 0xffff0000, v16
	v_and_b32_e32 v15, 0xffff0000, v17
	v_pk_add_f32 v[6:7], v[6:7], v[14:15]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_lshlrev_b32_e32 v10, 16, v13
	v_lshlrev_b32_e32 v14, 16, v12
	v_and_b32_e32 v11, 0xffff0000, v13
	v_and_b32_e32 v15, 0xffff0000, v12
	v_pk_add_f32 v[2:3], v[2:3], v[14:15]
	v_pk_add_f32 v[6:7], v[6:7], v[10:11]
	v_lshlrev_b32_e32 v10, 16, v8
	v_lshlrev_b32_e32 v12, 16, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_and_b32_e32 v13, 0xffff0000, v9
	v_pk_add_f32 v[6:7], v[6:7], v[12:13]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_lshlrev_b32_e32 v8, 16, v5
	v_lshlrev_b32_e32 v10, 16, v4
	v_and_b32_e32 v9, 0xffff0000, v5
	v_and_b32_e32 v11, 0xffff0000, v4
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_pk_add_f32 v[4:5], v[6:7], v[8:9]
	v_bfe_u32 v8, v3, 16, 1
	v_bfe_u32 v6, v5, 16, 1
	v_bfe_u32 v7, v4, 16, 1
	v_bfe_u32 v9, v2, 16, 1
	v_add3_u32 v2, v2, v9, s3
	v_add3_u32 v3, v3, v8, s3
	v_add3_u32 v4, v4, v7, s3
	v_add3_u32 v5, v5, v6, s3
	v_perm_b32 v5, v5, v4, s88
	v_perm_b32 v4, v3, v2, s88
	v_perm_b32 v3, v60, v59, s88
	v_perm_b32 v2, v58, v20, s88
	global_store_dwordx4 v[40:41], v[2:5], off
	s_branch .LBB4_27
.LBB4_38:                               ; %Flow2289
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[70:71], s[4:5]
	s_cbranch_execz .LBB4_10
; %bb.39:                               ; %.preheader880
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx4 v[2:5], v[2:3]
	s_mul_i32 s12, s40, s36
	s_add_i32 s12, s89, s12
	v_readlane_b32 s13, v247, 45
	s_mul_i32 s12, s12, s41
	s_add_i32 s13, s13, s90
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s46, s12
	s_addc_u32 s15, s47, s13
	v_mov_b32_e32 v6, s15
	v_readlane_b32 s12, v247, 46
	v_readlane_b32 s13, v247, 47
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e32 v2, vcc, s14, v2
	s_nop 1
	v_subb_co_u32_e32 v3, vcc, v6, v3, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_nop 1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_and_b64 vcc, exec, s[12:13]
	s_cbranch_vccz .LBB4_70
; %bb.40:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB4_42
.LBB4_41:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_42:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:16
	v_readlane_b32 s12, v247, 48
	v_readlane_b32 s13, v247, 49
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_44
; %bb.43:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_44:                               ; %Flow2281
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_46
; %bb.45:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_46:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:24
	v_readlane_b32 s12, v247, 50
	v_readlane_b32 s13, v247, 51
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_48
; %bb.47:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_48:                               ; %Flow2280
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_50
; %bb.49:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_50:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:32
	v_readlane_b32 s12, v247, 29
	v_readlane_b32 s13, v247, 30
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_52
; %bb.51:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_52:                               ; %Flow2279
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_54
; %bb.53:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_54:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:40
	v_readlane_b32 s12, v247, 32
	v_readlane_b32 s13, v247, 33
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_56
; %bb.55:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_56:                               ; %Flow2278
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_58
; %bb.57:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_58:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:48
	v_readlane_b32 s12, v247, 25
	v_readlane_b32 s13, v247, 26
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_60
; %bb.59:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_60:                               ; %Flow2277
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_62
; %bb.61:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_62:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:56
	v_readlane_b32 s12, v247, 7
	v_readlane_b32 s13, v247, 8
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_64
; %bb.63:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_64:                               ; %Flow2276
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_66
; %bb.65:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
.LBB4_66:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx2 v[4:5], v[2:3]
	s_nop 0
	flat_load_dwordx2 v[2:3], v[2:3] offset:64
	v_readlane_b32 s12, v247, 9
	v_readlane_b32 s13, v247, 10
	v_mov_b32_e32 v6, s15
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v4, s[12:13], s14, v4
	s_nop 1
	v_subb_co_u32_e64 v5, s[12:13], v6, v5, s[12:13]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_cmp_ne_u64_e64 s[12:13], 0, v[2:3]
	s_nop 1
	v_cndmask_b32_e64 v3, 0, v5, s[12:13]
	v_cndmask_b32_e64 v2, 0, v4, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB4_68
; %bb.67:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[2:3], v1 sc0 sc1
.LBB4_68:                               ; %Flow
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB4_10
; %bb.69:                               ;   in Loop: Header=BB4_12 Depth=1
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB4_10
.LBB4_70:                               ;   in Loop: Header=BB4_12 Depth=1
	s_branch .LBB4_41
.LBB4_71:                               ; %Flow2299
	s_or_b64 exec, exec, s[20:21]
	s_load_dwordx2 s[18:19], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB4_72:                               ; %Flow2341
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB4_485
; %bb.73:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s50, s4
	s_addc_u32 s5, s51, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v247, s8, 4
	s_nop 1
	v_writelane_b32 v247, s9, 5
	s_and_b64 s[8:9], s[6:7], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB4_75
; %bb.74:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB4_75:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[6:7], 0
	s_cselect_b64 s[66:67], -1, 0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB4_79
; %bb.76:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB4_78
; %bb.77:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB4_78:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB4_80
	s_branch .LBB4_485
.LBB4_79:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB4_485
.LBB4_80:                               ; %.critedge847
	s_lshr_b32 s3, s52, 24
	s_add_i32 s3, s37, s3
	s_ashr_i32 s63, s3, 8
	s_mul_i32 s3, s41, s63
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v247, s3, 6
	s_cbranch_scc1 .LBB4_485
; %bb.81:                               ; %.lr.ph1001
	s_load_dwordx2 s[60:61], s[0:1], 0x80
	s_load_dwordx4 s[4:7], s[0:1], 0x70
	s_load_dwordx2 s[12:13], s[0:1], 0x0
	s_load_dwordx2 s[14:15], s[0:1], 0x20
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_load_dwordx2 s[10:11], s[0:1], 0x50
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[8:9], src_shared_base
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s5, 0, 0
	s_cselect_b32 s3, s9, 0
	s_and_b32 s0, s5, 15
	s_and_b32 s7, s5, -16
	s_add_u32 s7, s7, 16
	s_mov_b32 s1, 0
	s_addc_u32 s8, s3, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s65, s5, s7
	s_cselect_b32 s0, s3, s8
	s_add_u32 s3, s65, 0x8000
	s_addc_u32 s0, s0, 0
	s_and_b32 s5, s3, -16
	s_and_b32 s0, s3, 15
	s_add_u32 s5, s5, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s69, s3, s5
	s_add_i32 s0, s39, 31
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	v_lshlrev_b32_e32 v182, 3, v0
	v_lshrrev_b32_e32 v5, 2, v0
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_and_b32_e32 v2, 24, v182
	v_or_b32_e32 v6, 0x80, v5
	s_ashr_i32 s37, s0, 5
	v_mad_u64_u32 v[146:147], s[0:1], v5, s14, v[2:3]
	v_mad_u64_u32 v[148:149], s[0:1], v6, s14, v[2:3]
	v_mad_u64_u32 v[150:151], s[0:1], v5, s10, v[2:3]
	s_mov_b32 s0, s10
	s_nop 0
	v_writelane_b32 v247, s0, 7
	s_add_i32 s84, s37, -1
	s_lshl_b32 s5, s41, 2
	v_writelane_b32 v247, s1, 8
	v_mad_u64_u32 v[152:153], s[0:1], v6, s10, v[2:3]
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v183, 48, v2
	v_and_b32_e32 v184, 0x1fc0, v2
	v_add_u32_e32 v2, s65, v183
	v_add_u32_e32 v7, v2, v184
	v_lshrrev_b32_e32 v8, 4, v7
	v_add_u32_e32 v6, 8, v2
	v_and_b32_e32 v8, 56, v8
	v_xor_b32_e32 v185, v8, v7
	v_add_u32_e32 v7, v6, v184
	v_lshrrev_b32_e32 v8, 4, v7
	v_or_b32_e32 v187, 0x2000, v184
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v2, v2, v187
	v_xor_b32_e32 v186, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	v_and_b32_e32 v7, 56, v7
	v_xor_b32_e32 v188, v7, v2
	v_add_u32_e32 v2, v6, v187
	s_cmp_gt_i32 s39, 0
	v_lshrrev_b32_e32 v6, 4, v2
	s_cselect_b64 s[0:1], -1, 0
	v_and_b32_e32 v6, 56, v6
	v_writelane_b32 v247, s0, 9
	v_xor_b32_e32 v189, v6, v2
	v_add_u32_e32 v2, s69, v183
	v_writelane_b32 v247, s1, 10
	s_ashr_i32 s1, s18, 31
	s_mov_b32 s0, s18
	v_add_u32_e32 v7, v2, v184
	s_lshl_b32 s3, s84, 5
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v8, 4, v7
	s_add_u32 s0, s46, s0
	v_add_u32_e32 v6, 8, v2
	v_and_b32_e32 v8, 56, v8
	s_addc_u32 s1, s47, s1
	v_xor_b32_e32 v190, v8, v7
	v_add_u32_e32 v7, v6, v184
	v_writelane_b32 v247, s0, 11
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 56, v8
	v_writelane_b32 v247, s1, 12
	v_add_u32_e32 v2, v2, v187
	s_min_i32 s20, s42, 32
	s_mul_i32 s21, s6, s4
	s_bfe_i64 s[74:75], s[60:61], 0x200000
	s_mov_b32 s61, s5
	v_readlane_b32 s4, v247, 0
	v_xor_b32_e32 v191, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	s_cmp_gt_i32 s42, 0
	v_readlane_b32 s5, v247, 1
	v_and_b32_e32 v7, 56, v7
	s_cselect_b64 s[76:77], -1, 0
	s_cmp_lg_u64 s[4:5], 0
	v_xor_b32_e32 v192, v7, v2
	v_add_u32_e32 v2, v6, v187
	s_cselect_b64 s[78:79], -1, 0
	s_max_i32 s0, s38, 1
	v_lshrrev_b32_e32 v6, 4, v2
	s_add_i32 s0, s0, -1
	v_and_b32_e32 v6, 56, v6
	s_cmp_lg_u32 s19, 0
	v_xor_b32_e32 v193, v6, v2
	v_and_b32_e32 v2, 15, v0
	s_cselect_b64 s[80:81], -1, 0
	s_abs_i32 s4, s61
	v_lshlrev_b32_e32 v6, 6, v2
	v_lshl_or_b32 v198, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s4
	s_or_b32 s1, s3, 16
	v_readlane_b32 s6, v247, 2
	v_readlane_b32 s7, v247, 3
	v_writelane_b32 v247, s0, 13
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s39, s3
	s_sub_i32 s3, s39, s1
	s_abs_i32 s39, s42
	v_lshl_or_b32 v195, v3, 12, v6
	v_cvt_f32_u32_e32 v3, s39
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_bfe_i32 s1, s41, 0x1001d
	v_rcp_iflag_f32_e32 v3, v3
	v_writelane_b32 v247, s1, 14
	v_writelane_b32 v247, s4, 16
	s_sub_i32 s1, 0, s4
	v_readfirstlane_b32 s4, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s1, s1, s4
	s_mul_hi_u32 s1, s4, s1
	s_add_i32 s1, s4, s1
	v_writelane_b32 v247, s1, 18
	s_sub_i32 s1, 0, s39
	v_readfirstlane_b32 s4, v2
	s_mul_i32 s1, s1, s4
	s_mul_hi_u32 s1, s4, s1
	s_add_i32 s35, s4, s1
	s_lshr_b32 s1, s35, 24
	s_mul_i32 s4, s1, s39
	s_sub_i32 s4, 0x100, s4
	s_lshl_b32 s82, s20, 5
	s_max_i32 s83, s37, 1
	s_ashr_i32 s34, s42, 31
	s_add_i32 s5, s1, 1
	s_sub_i32 s6, s4, s39
	s_cmp_ge_u32 s4, s39
	s_cselect_b32 s1, s5, s1
	s_cselect_b32 s4, s6, s4
	s_add_i32 s5, s1, 1
	s_cmp_ge_u32 s4, s39
	s_cselect_b32 s1, s5, s1
	s_abs_i32 s70, s33
	v_cvt_f32_u32_e32 v2, s70
	s_sub_i32 s9, 0, s70
	v_lshrrev_b32_e32 v4, 8, v0
	v_and_b32_e32 v5, 12, v5
	v_rcp_iflag_f32_e32 v2, v2
	v_lshl_or_b32 v194, v4, 13, v6
	v_or_b32_e32 v6, 1, v5
	v_or_b32_e32 v7, 2, v5
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_or_b32_e32 v8, 3, v5
	v_cmp_gt_i32_e64 s[4:5], s0, v6
	v_cmp_gt_i32_e64 s[6:7], s0, v7
	v_readfirstlane_b32 s8, v2
	s_mul_i32 s9, s9, s8
	s_mul_hi_u32 s9, s8, s9
	s_add_i32 s71, s8, s9
	v_cmp_gt_i32_e64 s[8:9], s0, v5
	v_cmp_gt_i32_e64 s[10:11], s0, v8
	s_xor_b32 s0, s1, s34
	v_lshrrev_b32_e32 v233, 5, v0
	s_sub_i32 s22, s0, s34
	s_ashr_i32 s23, s33, 31
	s_cmp_gt_i32 s22, 0
	v_mad_i64_i32 v[2:3], s[0:1], v233, s60, 0
	s_cselect_b64 s[18:19], -1, 0
	v_readlane_b32 s0, v247, 4
	v_readlane_b32 s1, v247, 5
	v_writelane_b32 v247, s18, 20
	s_and_b64 s[0:1], s[0:1], s[18:19]
	v_and_b32_e32 v9, 63, v0
	v_writelane_b32 v247, s19, 21
	v_writelane_b32 v247, s0, 22
	v_lshl_or_b32 v199, v4, 7, v5
	v_and_b32_e32 v4, 31, v0
	v_writelane_b32 v247, s1, 23
	s_add_u32 s0, s16, 64
	v_writelane_b32 v247, s0, 24
	v_writelane_b32 v247, s16, 25
	s_addc_u32 s0, s17, 0
	v_lshlrev_b32_e32 v154, 4, v4
	v_writelane_b32 v247, s17, 26
	v_writelane_b32 v247, s0, 27
	s_add_u32 s0, s12, 64
	v_writelane_b32 v247, s0, 28
	v_writelane_b32 v247, s12, 29
	s_addc_u32 s0, s13, 0
	v_lshlrev_b32_e32 v10, 9, v233
	v_writelane_b32 v247, s13, 30
	v_writelane_b32 v247, s0, 31
	s_mov_b32 s0, s14
	v_writelane_b32 v247, s0, 32
	v_mov_b32_e32 v155, 0
	v_lshl_add_u64 v[156:157], v[2:3], 1, v[154:155]
	v_writelane_b32 v247, s1, 33
	s_lshl_b32 s0, s14, 8
	v_writelane_b32 v247, s0, 34
	s_mul_i32 s0, s75, s20
	s_mul_hi_u32 s1, s60, s20
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s20
	s_lshl_b64 s[86:87], s[0:1], 1
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_or_b32_e32 v2, v10, v154
	v_add_u32_e32 v241, 0, v2
	v_writelane_b32 v247, s0, 35
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v247, s1, 36
	v_cmp_eq_u32_e64 s[0:1], 0, v9
	v_lshlrev_b32_e32 v232, 1, v5
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v247, s0, 37
	v_ashrrev_i32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v149, 31, v148
	v_writelane_b32 v247, s1, 38
	v_cmp_gt_u32_e64 s[0:1], s22, v0
	v_ashrrev_i32_e32 v151, 31, v150
	v_ashrrev_i32_e32 v153, 31, v152
	v_writelane_b32 v247, s0, 39
	v_mul_lo_u32 v196, s42, v0
	v_add_u32_e32 v197, -1, v1
	v_writelane_b32 v247, s1, 40
	v_writelane_b32 v247, s66, 41
	s_mul_i32 s21, s21, s36
	v_lshl_add_u32 v200, v198, 1, 0
	v_writelane_b32 v247, s67, 42
	v_or_b32_e32 v201, 1, v199
	v_or_b32_e32 v202, 2, v199
	v_or_b32_e32 v203, 3, v199
	v_or_b32_e32 v204, 16, v198
	v_or_b32_e32 v205, 32, v198
	v_or_b32_e32 v206, 48, v198
	v_or_b32_e32 v207, 16, v199
	v_or_b32_e32 v208, 17, v199
	v_or_b32_e32 v209, 18, v199
	v_or_b32_e32 v210, 19, v199
	v_or_b32_e32 v211, 32, v199
	v_or_b32_e32 v212, 33, v199
	v_or_b32_e32 v213, 34, v199
	v_or_b32_e32 v214, 35, v199
	v_or_b32_e32 v215, 48, v199
	v_or_b32_e32 v216, 49, v199
	v_or_b32_e32 v217, 50, v199
	v_or_b32_e32 v218, 51, v199
	v_or_b32_e32 v219, 64, v199
	v_or_b32_e32 v220, 0x41, v199
	v_or_b32_e32 v221, 0x42, v199
	v_or_b32_e32 v222, 0x43, v199
	v_or_b32_e32 v223, 0x50, v199
	v_or_b32_e32 v224, 0x51, v199
	v_or_b32_e32 v225, 0x52, v199
	v_or_b32_e32 v226, 0x53, v199
	v_or_b32_e32 v227, 0x60, v199
	v_or_b32_e32 v228, 0x61, v199
	v_or_b32_e32 v229, 0x62, v199
	v_or_b32_e32 v230, 0x63, v199
	v_or_b32_e32 v231, 0x70, v199
	v_or_b32_e32 v234, 0x71, v199
	v_or_b32_e32 v235, 0x72, v199
	v_or_b32_e32 v236, 0x73, v199
	v_or_b32_e32 v237, 32, v232
	v_lshl_or_b32 v238, v4, 3, 1
	v_add_u32_e32 v239, 0, v10
	v_lshl_add_u32 v240, v0, 1, 0
	s_sub_i32 s85, 0, s33
	v_and_b32_e32 v242, 0x100, v2
	v_bfrev_b32_e32 v243, 64
	s_movk_i32 s26, 0x7fff
	v_cmp_gt_i32_e64 s[12:13], s3, v6
	v_cmp_gt_i32_e64 s[14:15], s3, v7
	v_cmp_gt_i32_e64 s[16:17], s3, v5
	v_cmp_gt_i32_e64 s[18:19], s3, v8
	s_mov_b64 s[88:89], 0
	v_cmp_gt_i32_e64 s[24:25], s82, v0
	s_lshl_b64 s[90:91], s[74:75], 5
	v_writelane_b32 v247, s63, 43
	v_writelane_b32 v247, s61, 44
	s_branch .LBB4_84
.LBB4_82:                               ; %Flow2302
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s2, s2, s43
	v_readlane_b32 s0, v247, 6
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	v_readlane_b32 s30, v247, 52
	s_orn2_b64 s[0:1], s[0:1], exec
	v_readlane_b32 s31, v247, 53
.LBB4_83:                               ; %Flow2335
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[30:31]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[88:89], s[0:1], s[88:89]
	s_andn2_b64 exec, exec, s[88:89]
	s_cbranch_execz .LBB4_485
.LBB4_84:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_86 Depth 2
                                        ;     Child Loop BB4_101 Depth 2
                                        ;     Child Loop BB4_113 Depth 2
                                        ;       Child Loop BB4_117 Depth 3
                                        ;         Child Loop BB4_475 Depth 4
                                        ;         Child Loop BB4_431 Depth 4
                                        ;         Child Loop BB4_458 Depth 4
                                        ;         Child Loop BB4_465 Depth 4
                                        ;           Child Loop BB4_470 Depth 5
                                        ;     Child Loop BB4_481 Depth 2
	s_ashr_i32 s0, s2, 31
	v_readlane_b32 s1, v247, 14
	s_xor_b32 s29, s0, s1
	s_abs_i32 s0, s2
	v_readlane_b32 s1, v247, 18
	s_mul_hi_u32 s1, s0, s1
	v_readlane_b32 s28, v247, 16
	s_mul_i32 s3, s1, s28
	s_sub_i32 s0, s0, s3
	s_add_i32 s3, s1, 1
	s_sub_i32 s27, s0, s28
	s_cmp_ge_u32 s0, s28
	s_cselect_b32 s1, s3, s1
	s_cselect_b32 s0, s27, s0
	s_add_i32 s3, s1, 1
	s_cmp_ge_u32 s0, s28
	s_cselect_b32 s0, s3, s1
	s_xor_b32 s0, s0, s29
	v_writelane_b32 v247, s29, 45
	v_writelane_b32 v247, s0, 46
	s_sub_i32 s0, s0, s29
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s3, s63, s1
	s_min_i32 s28, s3, 4
	s_abs_i32 s27, s28
	v_cvt_f32_u32_e32 v2, s27
	s_sub_i32 s30, 0, s27
	s_mul_i32 s0, s0, s61
	v_writelane_b32 v247, s0, 48
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s2, s0
	s_abs_i32 s29, s0
	s_xor_b32 s3, s0, s28
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_ashr_i32 s3, s3, 31
	v_mov_b32_e32 v5, v155
	v_mov_b32_e32 v4, v155
	v_readfirstlane_b32 s31, v2
	s_mul_i32 s30, s30, s31
	s_mul_hi_u32 s30, s31, s30
	s_add_i32 s31, s31, s30
	s_mul_hi_u32 s30, s29, s31
	s_mul_i32 s31, s30, s27
	s_sub_i32 s29, s29, s31
	s_add_i32 s31, s30, 1
	s_sub_i32 s50, s29, s27
	s_cmp_ge_u32 s29, s27
	s_cselect_b32 s30, s31, s30
	s_cselect_b32 s29, s50, s29
	s_add_i32 s31, s30, 1
	s_cmp_ge_u32 s29, s27
	s_cselect_b32 s27, s31, s30
	s_xor_b32 s51, s27, s3
	s_sub_i32 s27, s51, s3
	s_mul_i32 s28, s27, s28
	s_sub_i32 s0, s0, s28
	v_writelane_b32 v247, s28, 50
	s_add_i32 s0, s0, s1
	s_lshl_b32 s64, s0, 8
	v_readlane_b32 s0, v247, 32
	v_readlane_b32 s1, v247, 33
	s_mul_i32 s0, s64, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v247, 29
	v_readlane_b32 s29, v247, 30
	s_add_u32 s0, s28, s0
	s_addc_u32 s1, s29, s1
	v_lshl_add_u64 v[2:3], v[146:147], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[130:133], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[148:149], 1, s[0:1]
	s_lshl_b32 s62, s27, 8
	v_readlane_b32 s0, v247, 7
	v_readlane_b32 s1, v247, 8
	s_mul_i32 s0, s62, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v247, 25
	v_readlane_b32 s29, v247, 26
	s_add_u32 s28, s28, s0
	;;#ASMSTART
	global_load_dwordx4 v[134:137], v[2:3], off

	;;#ASMEND
	s_addc_u32 s29, s29, s1
	v_lshl_add_u64 v[2:3], v[150:151], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[138:141], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[152:153], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v185, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v186, v[132:133]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v188, v[134:135]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v189, v[136:137]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v190, v[138:139]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v191, v[140:141]

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v192, v[142:143]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v193, v[144:145]

	;;#ASMEND
	v_readlane_b32 s28, v247, 9
	v_readlane_b32 s29, v247, 10
	s_andn2_b64 vcc, exec, s[28:29]
	v_mov_b32_e32 v3, v155
	v_mov_b32_e32 v2, v155
	v_mov_b32_e32 v9, v155
	v_mov_b32_e32 v8, v155
	v_mov_b32_e32 v7, v155
	v_mov_b32_e32 v6, v155
	v_mov_b32_e32 v13, v155
	v_mov_b32_e32 v12, v155
	v_mov_b32_e32 v11, v155
	v_mov_b32_e32 v10, v155
	v_mov_b32_e32 v17, v155
	v_mov_b32_e32 v16, v155
	v_mov_b32_e32 v15, v155
	v_mov_b32_e32 v14, v155
	v_mov_b32_e32 v21, v155
	v_mov_b32_e32 v20, v155
	v_mov_b32_e32 v19, v155
	v_mov_b32_e32 v18, v155
	v_mov_b32_e32 v25, v155
	v_mov_b32_e32 v24, v155
	v_mov_b32_e32 v23, v155
	v_mov_b32_e32 v22, v155
	v_mov_b32_e32 v29, v155
	v_mov_b32_e32 v28, v155
	v_mov_b32_e32 v27, v155
	v_mov_b32_e32 v26, v155
	v_mov_b32_e32 v33, v155
	v_mov_b32_e32 v32, v155
	v_mov_b32_e32 v31, v155
	v_mov_b32_e32 v30, v155
	v_mov_b32_e32 v37, v155
	v_mov_b32_e32 v36, v155
	v_mov_b32_e32 v35, v155
	v_mov_b32_e32 v34, v155
	v_mov_b32_e32 v41, v155
	v_mov_b32_e32 v40, v155
	v_mov_b32_e32 v39, v155
	v_mov_b32_e32 v38, v155
	v_mov_b32_e32 v45, v155
	v_mov_b32_e32 v44, v155
	v_mov_b32_e32 v43, v155
	v_mov_b32_e32 v42, v155
	v_mov_b32_e32 v49, v155
	v_mov_b32_e32 v48, v155
	v_mov_b32_e32 v47, v155
	v_mov_b32_e32 v46, v155
	v_mov_b32_e32 v53, v155
	v_mov_b32_e32 v52, v155
	v_mov_b32_e32 v51, v155
	v_mov_b32_e32 v50, v155
	v_mov_b32_e32 v57, v155
	v_mov_b32_e32 v56, v155
	v_mov_b32_e32 v55, v155
	v_mov_b32_e32 v54, v155
	v_mov_b32_e32 v61, v155
	v_mov_b32_e32 v60, v155
	v_mov_b32_e32 v59, v155
	v_mov_b32_e32 v58, v155
	v_mov_b32_e32 v65, v155
	v_mov_b32_e32 v64, v155
	v_mov_b32_e32 v63, v155
	v_mov_b32_e32 v62, v155
	v_mov_b32_e32 v69, v155
	v_mov_b32_e32 v68, v155
	v_mov_b32_e32 v67, v155
	v_mov_b32_e32 v66, v155
	v_mov_b32_e32 v73, v155
	v_mov_b32_e32 v72, v155
	v_mov_b32_e32 v71, v155
	v_mov_b32_e32 v70, v155
	v_mov_b32_e32 v77, v155
	v_mov_b32_e32 v76, v155
	v_mov_b32_e32 v75, v155
	v_mov_b32_e32 v74, v155
	v_mov_b32_e32 v81, v155
	v_mov_b32_e32 v80, v155
	v_mov_b32_e32 v79, v155
	v_mov_b32_e32 v78, v155
	v_mov_b32_e32 v85, v155
	v_mov_b32_e32 v84, v155
	v_mov_b32_e32 v83, v155
	v_mov_b32_e32 v82, v155
	v_mov_b32_e32 v89, v155
	v_mov_b32_e32 v88, v155
	v_mov_b32_e32 v87, v155
	v_mov_b32_e32 v86, v155
	v_mov_b32_e32 v93, v155
	v_mov_b32_e32 v92, v155
	v_mov_b32_e32 v91, v155
	v_mov_b32_e32 v90, v155
	v_mov_b32_e32 v97, v155
	v_mov_b32_e32 v96, v155
	v_mov_b32_e32 v95, v155
	v_mov_b32_e32 v94, v155
	v_mov_b32_e32 v101, v155
	v_mov_b32_e32 v100, v155
	v_mov_b32_e32 v99, v155
	v_mov_b32_e32 v98, v155
	v_mov_b32_e32 v105, v155
	v_mov_b32_e32 v104, v155
	v_mov_b32_e32 v103, v155
	v_mov_b32_e32 v102, v155
	v_mov_b32_e32 v109, v155
	v_mov_b32_e32 v108, v155
	v_mov_b32_e32 v107, v155
	v_mov_b32_e32 v106, v155
	v_mov_b32_e32 v113, v155
	v_mov_b32_e32 v112, v155
	v_mov_b32_e32 v111, v155
	v_mov_b32_e32 v110, v155
	v_mov_b32_e32 v117, v155
	v_mov_b32_e32 v116, v155
	v_mov_b32_e32 v115, v155
	v_mov_b32_e32 v114, v155
	v_mov_b32_e32 v121, v155
	v_mov_b32_e32 v120, v155
	v_mov_b32_e32 v119, v155
	v_mov_b32_e32 v118, v155
	v_mov_b32_e32 v125, v155
	v_mov_b32_e32 v124, v155
	v_mov_b32_e32 v123, v155
	v_mov_b32_e32 v122, v155
	v_mov_b32_e32 v129, v155
	v_mov_b32_e32 v128, v155
	v_mov_b32_e32 v127, v155
	v_mov_b32_e32 v126, v155
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB4_96
; %bb.85:                               ; %.lr.ph959.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v247, 24
	s_add_u32 s28, s28, s0
	v_readlane_b32 s0, v247, 27
	s_addc_u32 s29, s0, s1
	v_readlane_b32 s0, v247, 46
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s2, s0
	v_readlane_b32 s1, v247, 48
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v247, 50
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v247, 45
	s_lshl_b32 s1, s1, 2
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v247, 34
	s_mul_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s30, v247, 28
	s_add_u32 s30, s30, s0
	v_readlane_b32 s0, v247, 31
	v_mov_b32_e32 v2, 0
	s_addc_u32 s31, s0, s1
	s_mov_b32 s0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v99, v2
	v_mov_b32_e32 v100, v2
	v_mov_b32_e32 v101, v2
	v_mov_b32_e32 v102, v2
	v_mov_b32_e32 v103, v2
	v_mov_b32_e32 v104, v2
	v_mov_b32_e32 v105, v2
	v_mov_b32_e32 v106, v2
	v_mov_b32_e32 v107, v2
	v_mov_b32_e32 v108, v2
	v_mov_b32_e32 v109, v2
	v_mov_b32_e32 v110, v2
	v_mov_b32_e32 v111, v2
	v_mov_b32_e32 v112, v2
	v_mov_b32_e32 v113, v2
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v115, v2
	v_mov_b32_e32 v116, v2
	v_mov_b32_e32 v117, v2
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v119, v2
	v_mov_b32_e32 v120, v2
	v_mov_b32_e32 v121, v2
	v_mov_b32_e32 v122, v2
	v_mov_b32_e32 v123, v2
	v_mov_b32_e32 v124, v2
	v_mov_b32_e32 v125, v2
	v_mov_b32_e32 v126, v2
	v_mov_b32_e32 v127, v2
	v_mov_b32_e32 v128, v2
	v_mov_b32_e32 v129, v2
.LBB4_86:                               ; %.lr.ph959
                                        ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s50, s0, 1
	s_cmp_lt_i32 s50, s37
	s_cselect_b64 s[92:93], -1, 0
	s_cmp_ge_i32 s50, s37
	s_cbranch_scc1 .LBB4_88
; %bb.87:                               ;   in Loop: Header=BB4_86 Depth=2
	v_lshl_add_u64 v[130:131], v[146:147], 1, s[30:31]
	;;#ASMSTART
	global_load_dwordx4 v[130:133], v[130:131], off

	;;#ASMEND
	v_lshl_add_u64 v[134:135], v[148:149], 1, s[30:31]
	;;#ASMSTART
	global_load_dwordx4 v[134:137], v[134:135], off

	;;#ASMEND
	v_lshl_add_u64 v[138:139], v[150:151], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[138:141], v[138:139], off

	;;#ASMEND
	v_lshl_add_u64 v[142:143], v[152:153], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
.LBB4_88:                               ;   in Loop: Header=BB4_86 Depth=2
	s_lshl_b32 s1, s0, 14
	s_and_b32 s1, s1, 0x4000
	s_add_i32 s52, s65, s1
	v_add_u32_e32 v244, s52, v194
	v_add_u32_e32 v158, v244, v232
	v_lshrrev_b32_e32 v159, 4, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	;;#ASMSTART
	ds_read_b64 v[180:181], v158 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[178:179], v158 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v158 offset:0x800

	;;#ASMEND
	s_add_i32 s1, s69, s1
	;;#ASMSTART
	ds_read_b64 v[174:175], v158 offset:0xc00

	;;#ASMEND
	v_add_u32_e32 v154, s1, v195
	;;#ASMSTART
	ds_read_b64 v[172:173], v158 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v158 offset:0x1400

	;;#ASMEND
	v_add_u32_e32 v164, v154, v232
	;;#ASMSTART
	ds_read_b64 v[160:161], v158 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v165, 4, v164
	;;#ASMSTART
	ds_read_b64 v[158:159], v158 offset:0x1c00

	;;#ASMEND
	v_and_b32_e32 v165, 56, v165
	v_xor_b32_e32 v170, v165, v164
	;;#ASMSTART
	ds_read_b64 v[164:165], v170 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[166:167], v170 offset:0x400

	;;#ASMEND
	s_cmp_eq_u32 s84, s0
	;;#ASMSTART
	ds_read_b64 v[168:169], v170 offset:0x800

	;;#ASMEND
	s_cselect_b64 s[94:95], -1, 0
	s_cmp_lg_u32 s84, s0
	;;#ASMSTART
	ds_read_b64 v[170:171], v170 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB4_90
; %bb.89:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB4_86 Depth=2
	s_or_b64 s[0:1], s[10:11], s[6:7]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_cndmask_b32_e64 v245, 0, v180, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v181
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v181, s[6:7]
	v_cndmask_b32_sdwa v180, v245, v180, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v181, v246, v181, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v178, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v179
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v179, s[6:7]
	v_cndmask_b32_sdwa v178, v245, v178, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v179, v246, v179, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v176, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v177
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v177, s[6:7]
	v_cndmask_b32_sdwa v176, v245, v176, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v177, v246, v177, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v174, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v175
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v175, s[6:7]
	v_cndmask_b32_sdwa v174, v245, v174, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v175, v246, v175, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v172, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v173
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v173, s[6:7]
	v_cndmask_b32_sdwa v172, v245, v172, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v173, v246, v173, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v162, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v163
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v163, s[6:7]
	v_cndmask_b32_sdwa v162, v245, v162, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v163, v246, v163, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v160, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v161
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v161, s[6:7]
	v_cndmask_b32_sdwa v160, v245, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v161, v246, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v245, 0, v158, s[8:9]
	v_and_b32_e32 v246, 0xffff0000, v159
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v246, v246, v159, s[6:7]
	v_cndmask_b32_sdwa v158, v245, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v159, v246, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB4_90:                               ;   in Loop: Header=BB4_86 Depth=2
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[14:17], v[158:159], v[164:165], v[14:17]
	v_add_u32_e32 v154, v154, v237
	s_andn2_b64 vcc, exec, s[94:95]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[158:159], v[166:167], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[158:159], v[168:169], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[158:159], v[170:171], v[2:5]
	v_add_u32_e32 v158, v244, v237
	v_lshrrev_b32_e32 v159, 4, v158
	v_and_b32_e32 v159, 56, v159
	v_mfma_f32_16x16x16_bf16 v[114:117], v[180:181], v[170:171], v[114:117]
	v_xor_b32_e32 v158, v159, v158
	v_mfma_f32_16x16x16_bf16 v[98:101], v[178:179], v[170:171], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[176:177], v[170:171], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[174:175], v[170:171], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[172:173], v[170:171], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[162:163], v[170:171], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[160:161], v[170:171], v[18:21]
	;;#ASMSTART
	ds_read_b64 v[170:171], v158 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[62:65], v[172:173], v[164:165], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[172:173], v[166:167], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[172:173], v[168:169], v[54:57]
	;;#ASMSTART
	ds_read_b64 v[172:173], v158 offset:0x400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[118:121], v[180:181], v[168:169], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[178:179], v[168:169], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[176:177], v[168:169], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[174:175], v[168:169], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[162:163], v[168:169], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[160:161], v[168:169], v[22:25]
	;;#ASMSTART
	ds_read_b64 v[168:169], v158 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[126:129], v[180:181], v[164:165], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[178:179], v[164:165], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[176:177], v[164:165], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[174:175], v[164:165], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[162:163], v[164:165], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[160:161], v[164:165], v[30:33]
	;;#ASMSTART
	ds_read_b64 v[164:165], v158 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[122:125], v[180:181], v[166:167], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[178:179], v[166:167], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[176:177], v[166:167], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[174:175], v[166:167], v[74:77]
	v_lshrrev_b32_e32 v174, 4, v154
	v_and_b32_e32 v174, 56, v174
	v_xor_b32_e32 v154, v174, v154
	v_mfma_f32_16x16x16_bf16 v[42:45], v[162:163], v[166:167], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[160:161], v[166:167], v[26:29]
	;;#ASMSTART
	ds_read_b64 v[166:167], v158 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v158 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v158 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v158 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[174:175], v154 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[178:179], v154 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[180:181], v154 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v154 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_vccnz .LBB4_92
; %bb.91:                               ; %.loopexit.i.1
                                        ;   in Loop: Header=BB4_86 Depth=2
	s_or_b64 s[0:1], s[18:19], s[14:15]
	s_or_b64 s[0:1], s[0:1], s[12:13]
	v_cndmask_b32_e64 v154, 0, v170, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v171
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v171, s[14:15]
	v_cndmask_b32_sdwa v170, v154, v170, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v171, v244, v171, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v172, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v173
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v173, s[14:15]
	v_cndmask_b32_sdwa v172, v154, v172, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v173, v244, v173, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v168, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v169
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v169, s[14:15]
	v_cndmask_b32_sdwa v168, v154, v168, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v169, v244, v169, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v164, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v165
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v165, s[14:15]
	v_cndmask_b32_sdwa v164, v154, v164, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v165, v244, v165, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v166, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v167
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v167, s[14:15]
	v_cndmask_b32_sdwa v166, v154, v166, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v167, v244, v167, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v162, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v163
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v163, s[14:15]
	v_cndmask_b32_sdwa v162, v154, v162, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v163, v244, v163, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v160, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v161
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v161, s[14:15]
	v_cndmask_b32_sdwa v160, v154, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v161, v244, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v154, 0, v158, s[16:17]
	v_and_b32_e32 v244, 0xffff0000, v159
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v244, v244, v159, s[14:15]
	v_cndmask_b32_sdwa v158, v154, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v159, v244, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB4_92:                               ;   in Loop: Header=BB4_86 Depth=2
	v_mfma_f32_16x16x16_bf16 v[126:129], v[170:171], v[174:175], v[126:129]
	s_andn2_b64 vcc, exec, s[92:93]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[170:171], v[178:179], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[170:171], v[180:181], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[170:171], v[176:177], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[172:173], v[174:175], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[172:173], v[178:179], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[172:173], v[180:181], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[172:173], v[176:177], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[168:169], v[174:175], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[168:169], v[178:179], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[168:169], v[180:181], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[168:169], v[176:177], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[164:165], v[174:175], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[164:165], v[178:179], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[164:165], v[180:181], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[164:165], v[176:177], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[166:167], v[174:175], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[166:167], v[178:179], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[166:167], v[180:181], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[166:167], v[176:177], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[162:163], v[174:175], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[162:163], v[178:179], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[162:163], v[180:181], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[162:163], v[176:177], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[160:161], v[174:175], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[160:161], v[178:179], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[160:161], v[180:181], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[160:161], v[176:177], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[158:159], v[174:175], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[158:159], v[178:179], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[158:159], v[180:181], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[158:159], v[176:177], v[2:5]
	s_cbranch_vccnz .LBB4_94
; %bb.93:                               ;   in Loop: Header=BB4_86 Depth=2
	s_lshl_b32 s0, s50, 14
	s_and_b32 s0, s0, 0x4000
	s_add_i32 s1, s65, s0
	v_add_u32_e32 v154, s1, v183
	v_add_u32_e32 v159, v154, v184
	v_lshrrev_b32_e32 v160, 4, v159
	s_or_b32 s1, s1, 8
	v_and_b32_e32 v160, 56, v160
	v_add_u32_e32 v158, s1, v183
	v_xor_b32_e32 v159, v160, v159
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v159, v[130:131]

	;;#ASMEND
	v_add_u32_e32 v159, v158, v184
	v_lshrrev_b32_e32 v160, 4, v159
	v_and_b32_e32 v160, 56, v160
	v_xor_b32_e32 v159, v160, v159
	v_add_u32_e32 v154, v154, v187
	;;#ASMSTART
	ds_write_b64 v159, v[132:133]

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v154
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v154, v159, v154
	;;#ASMSTART
	ds_write_b64 v154, v[134:135]

	;;#ASMEND
	v_add_u32_e32 v154, v158, v187
	v_lshrrev_b32_e32 v158, 4, v154
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v154, v158, v154
	s_add_i32 s0, s69, s0
	;;#ASMSTART
	ds_write_b64 v154, v[136:137]

	;;#ASMEND
	v_add_u32_e32 v154, s0, v183
	v_add_u32_e32 v159, v154, v184
	v_lshrrev_b32_e32 v160, 4, v159
	s_or_b32 s0, s0, 8
	v_and_b32_e32 v160, 56, v160
	v_add_u32_e32 v158, s0, v183
	v_xor_b32_e32 v159, v160, v159
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v159, v[138:139]

	;;#ASMEND
	v_add_u32_e32 v159, v158, v184
	v_lshrrev_b32_e32 v160, 4, v159
	v_and_b32_e32 v160, 56, v160
	v_xor_b32_e32 v159, v160, v159
	v_add_u32_e32 v154, v154, v187
	;;#ASMSTART
	ds_write_b64 v159, v[140:141]

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v154
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v154, v159, v154
	;;#ASMSTART
	ds_write_b64 v154, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v154, v158, v187
	v_lshrrev_b32_e32 v158, 4, v154
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v154, v158, v154
	;;#ASMSTART
	ds_write_b64 v154, v[144:145]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB4_94:                               ;   in Loop: Header=BB4_86 Depth=2
	s_add_u32 s28, s28, 64
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 64
	s_addc_u32 s31, s31, 0
	s_cmp_eq_u32 s83, s50
	s_barrier
	s_cbranch_scc1 .LBB4_96
; %bb.95:                               ;   in Loop: Header=BB4_86 Depth=2
	s_mov_b32 s0, s50
	s_branch .LBB4_86
.LBB4_96:                               ; %Flow2331
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v247, 39
	v_readlane_b32 s29, v247, 40
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_105
; %bb.97:                               ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v247, 35
	v_readlane_b32 s29, v247, 36
	s_and_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB4_105
; %bb.98:                               ;   in Loop: Header=BB4_84 Depth=1
	v_add_u32_e32 v130, s64, v196
	v_sub_u32_e32 v132, 0, v130
	v_max_i32_e32 v132, v130, v132
	v_mul_hi_u32 v133, v132, s71
	v_mul_lo_u32 v134, v133, s70
	v_sub_u32_e32 v132, v132, v134
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s70, v132
	v_ashrrev_i32_e32 v131, 31, v130
	v_xor_b32_e32 v131, s23, v131
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s70, v132
	v_cndmask_b32_e32 v132, v132, v134, vcc
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s70, v132
	s_nop 1
	v_cndmask_b32_e32 v132, v133, v134, vcc
	v_xor_b32_e32 v132, v132, v131
	v_sub_u32_e32 v131, v132, v131
	v_mul_lo_u32 v132, v131, s33
	v_sub_u32_e32 v130, v130, v132
	v_sub_u32_e32 v133, 0, v130
	v_ashrrev_i32_e32 v132, 31, v130
	v_max_i32_e32 v130, v130, v133
	v_mul_hi_u32 v133, v130, s35
	v_mul_lo_u32 v134, v133, s39
	v_sub_u32_e32 v130, v130, v134
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s39, v130
	v_xor_b32_e32 v132, s34, v132
	s_nop 0
	v_cndmask_b32_e32 v133, v133, v134, vcc
	v_subrev_u32_e32 v134, s39, v130
	v_cndmask_b32_e32 v130, v130, v134, vcc
	v_add_u32_e32 v134, 1, v133
	v_cmp_le_u32_e32 vcc, s39, v130
	s_nop 1
	v_cndmask_b32_e32 v130, v133, v134, vcc
	v_xor_b32_e32 v130, v130, v132
	v_sub_u32_e32 v130, v130, v132
	v_mad_u64_u32 v[130:131], s[28:29], v131, s40, v[130:131]
	v_mul_lo_u32 v130, v130, s41
	v_add_u32_e32 v130, s27, v130
	v_readlane_b32 s28, v247, 11
	v_ashrrev_i32_e32 v131, 31, v130
	v_readlane_b32 s29, v247, 12
	s_nop 1
	v_lshl_add_u64 v[130:131], v[130:131], 2, s[28:29]
	flat_load_dword v132, v[130:131] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v132, v197
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB4_105
; %bb.99:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_mov_b64 s[28:29], 0
	s_mov_b64 s[94:95], 0
                                        ; implicit-def: $sgpr30_sgpr31
                                        ; implicit-def: $sgpr92_sgpr93
	s_branch .LBB4_101
.LBB4_100:                              ; %Flow2326
                                        ;   in Loop: Header=BB4_101 Depth=2
	s_and_b64 s[52:53], exec, s[92:93]
	s_or_b64 s[28:29], s[52:53], s[28:29]
	s_andn2_b64 s[30:31], s[30:31], exec
	s_and_b64 s[52:53], s[96:97], exec
	s_or_b64 s[30:31], s[30:31], s[52:53]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB4_103
.LBB4_101:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s94, s94, 1
	s_addc_u32 s95, s95, 0
	v_mov_b64_e32 v[132:133], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[94:95], v[132:133]
	s_mov_b64 s[96:97], -1
	s_or_b64 s[92:93], s[92:93], exec
	s_cbranch_vccnz .LBB4_100
; %bb.102:                              ;   in Loop: Header=BB4_101 Depth=2
	s_sleep 4
	flat_load_dword v132, v[130:131] offset:256 sc0 sc1
	s_andn2_b64 s[52:53], s[92:93], exec
	s_mov_b64 s[96:97], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v132, v197
	s_and_b64 s[54:55], vcc, exec
	s_or_b64 s[92:93], s[52:53], s[54:55]
	s_branch .LBB4_100
.LBB4_103:                              ; %loop.exit.guard2274
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_xor_b64 s[28:29], exec, s[28:29]
	s_cbranch_execz .LBB4_105
; %bb.104:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v247, 0
	v_readlane_b32 s30, v247, 2
	v_readlane_b32 s31, v247, 3
	v_readlane_b32 s29, v247, 1
	s_nop 0
	v_mov_b64_e32 v[130:131], s[30:31]
	flat_atomic_or v[130:131], v243
.LBB4_105:                              ; %.critedge848
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[66:67]
	s_mov_b64 s[28:29], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB4_109
; %bb.106:                              ;   in Loop: Header=BB4_84 Depth=1
	v_mov_b32_e32 v130, 0
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s30, v247, 37
	v_readlane_b32 s31, v247, 38
	s_and_b64 s[30:31], s[28:29], s[30:31]
	s_mov_b64 exec, s[30:31]
	s_cbranch_execz .LBB4_108
; %bb.107:                              ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s52, v247, 0
	v_readlane_b32 s54, v247, 2
	v_readlane_b32 s55, v247, 3
	v_readlane_b32 s53, v247, 1
	s_nop 0
	v_mov_b64_e32 v[130:131], s[54:55]
	flat_load_dword v130, v[130:131] sc1
.LBB4_108:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit348
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v130, v242, v130
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v130, 0x2000000, v130
	v_cmp_eq_u32_e64 s[28:29], 0, v130
.LBB4_109:                              ; %Flow2334
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_and_saveexec_b64 s[30:31], s[28:29]
	s_cbranch_execz .LBB4_83
; %bb.110:                              ; %.critedge870
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_writelane_b32 v247, s30, 52
	s_nop 1
	v_writelane_b32 v247, s31, 53
	s_nop 0
	v_readlane_b32 s0, v247, 20
	v_readlane_b32 s1, v247, 21
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB4_476
; %bb.111:                              ; %.lr.ph996
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_sub_i32 s63, s38, s62
	s_min_i32 s66, s63, 0x100
	s_abs_i32 s67, s66
	v_cvt_f32_u32_e32 v138, s67
	v_readlane_b32 s0, v247, 13
	s_ashr_i32 s53, s66, 31
	s_lshl_b32 s1, s3, 8
	v_rcp_iflag_f32_e32 v138, v138
	v_mov_b32_e32 v136, s0
	s_sub_i32 s0, 0, s67
	v_or_b32_e32 v130, s62, v198
	v_mul_f32_e32 v138, 0x4f7ffffe, v138
	v_cvt_u32_f32_e32 v138, v138
	v_cmp_gt_i32_e32 vcc, s38, v130
	v_or_b32_e32 v132, s62, v204
	v_or_b32_e32 v134, s62, v205
	v_mul_lo_u32 v139, s0, v138
	s_lshl_b32 s0, s53, 9
	v_subrev_u32_e32 v161, s0, v240
	s_lshl_b32 s0, s51, 8
	s_sub_i32 s51, s0, s1
	v_readlane_b32 s0, v247, 46
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s2, s0
	v_readlane_b32 s1, v247, 48
	v_cndmask_b32_e32 v130, v136, v130, vcc
	v_cmp_gt_i32_e32 vcc, s38, v132
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v247, 50
	v_cndmask_b32_e32 v132, v136, v132, vcc
	v_cmp_gt_i32_e32 vcc, s38, v134
	v_or_b32_e32 v137, s62, v206
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v247, 45
	v_cndmask_b32_e32 v134, v136, v134, vcc
	v_cmp_gt_i32_e32 vcc, s38, v137
	s_lshl_b32 s1, s1, 2
	v_readlane_b32 s28, v247, 0
	v_cndmask_b32_e32 v136, v136, v137, vcc
	s_sub_i32 s0, s0, s1
	v_ashrrev_i32_e32 v131, 31, v130
	v_readlane_b32 s29, v247, 1
	v_ashrrev_i32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v137, 31, v136
	s_mul_i32 s52, s66, s20
	v_mul_hi_u32 v139, v138, v139
	s_lshl_b32 s0, s0, 8
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[28:29]
	v_lshl_add_u64 v[132:133], v[132:133], 1, s[28:29]
	v_lshl_add_u64 v[134:135], v[134:135], 1, s[28:29]
	v_lshl_add_u64 v[136:137], v[136:137], 1, s[28:29]
	v_cmp_gt_i32_e64 s[28:29], s52, v0
	s_mov_b32 s54, 0
	v_add_u32_e32 v160, v138, v139
	s_lshl_b32 s55, s66, 1
	s_sub_i32 s50, 0, s66
	s_add_i32 s3, s21, s0
	v_readlane_b32 s30, v247, 2
	v_readlane_b32 s31, v247, 3
	s_branch .LBB4_113
.LBB4_112:                              ; %._crit_edge994
                                        ;   in Loop: Header=BB4_113 Depth=2
	s_add_i32 s54, s54, 1
	s_add_i32 s3, s3, s42
	s_cmp_eq_u32 s54, s22
	s_cbranch_scc1 .LBB4_476
.LBB4_113:                              ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB4_117 Depth 3
                                        ;         Child Loop BB4_475 Depth 4
                                        ;         Child Loop BB4_431 Depth 4
                                        ;         Child Loop BB4_458 Depth 4
                                        ;         Child Loop BB4_465 Depth 4
                                        ;           Child Loop BB4_470 Depth 5
	s_andn2_b64 vcc, exec, s[76:77]
	s_cbranch_vccnz .LBB4_112
; %bb.114:                              ; %.lr.ph993.preheader
                                        ;   in Loop: Header=BB4_113 Depth=2
	v_mov_b64_e32 v[158:159], s[44:45]
	flat_load_dwordx4 v[138:141], v[158:159]
	flat_load_dwordx4 v[142:145], v[158:159] offset:16
	flat_load_dwordx4 v[162:165], v[158:159] offset:32
	flat_load_dwordx4 v[166:169], v[158:159] offset:48
	s_nop 0
	flat_load_dwordx2 v[158:159], v[158:159] offset:64
	s_mul_i32 s61, s54, s42
	s_add_i32 s1, s61, s64
	s_abs_i32 s30, s1
	s_mul_hi_u32 s31, s30, s71
	s_mul_i32 s68, s31, s70
	s_ashr_i32 s0, s1, 31
	s_sub_i32 s30, s30, s68
	s_xor_b32 s0, s0, s23
	s_add_i32 s68, s31, 1
	s_sub_i32 s72, s30, s70
	s_cmp_ge_u32 s30, s70
	s_cselect_b32 s31, s68, s31
	s_cselect_b32 s30, s72, s30
	s_add_i32 s68, s31, 1
	s_cmp_ge_u32 s30, s70
	s_cselect_b32 s30, s68, s31
	s_xor_b32 s30, s30, s0
	s_sub_i32 s30, s30, s0
	s_mul_i32 s0, s30, s33
	s_sub_i32 s1, s1, s0
	s_cmp_eq_u32 s30, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 1
	s_mov_b32 s92, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v140, 0, v140, vcc
	v_cndmask_b32_e32 v141, 0, v141, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 2
	v_cndmask_b32_e32 v141, v141, v143, vcc
	v_cndmask_b32_e32 v140, v140, v142, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 3
	v_cndmask_b32_e32 v140, v140, v144, vcc
	v_cndmask_b32_e32 v141, v141, v145, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 4
	v_cndmask_b32_e32 v141, v141, v163, vcc
	v_cndmask_b32_e32 v140, v140, v162, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 5
	v_cndmask_b32_e32 v140, v140, v164, vcc
	v_cndmask_b32_e32 v141, v141, v165, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 6
	v_cndmask_b32_e32 v141, v141, v167, vcc
	v_cndmask_b32_e32 v140, v140, v166, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 7
	v_cndmask_b32_e32 v140, v140, v168, vcc
	v_cndmask_b32_e32 v141, v141, v169, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s30, s1
	s_mul_hi_u32 s31, s30, s35
	s_mul_i32 s31, s31, s39
	s_sub_i32 s30, s30, s31
	s_ashr_i32 s68, s1, 31
	s_sub_i32 s31, s30, s39
	s_cmp_ge_u32 s30, s39
	s_cselect_b32 s30, s31, s30
	s_sub_i32 s31, s30, s39
	s_cmp_ge_u32 s30, s39
	s_cselect_b32 s30, s31, s30
	s_xor_b32 s72, s30, s68
	s_add_i32 s1, s1, s21
	s_sub_i32 s30, s68, s72
	s_add_i32 s68, s68, s3
	v_cndmask_b32_e32 v141, v141, v159, vcc
	v_cndmask_b32_e32 v140, v140, v158, vcc
	v_sub_co_u32_e32 v138, vcc, s56, v138
	v_mov_b32_e32 v142, s57
	s_add_i32 s1, s1, s30
	s_sub_i32 s0, s68, s0
	v_subb_co_u32_e32 v139, vcc, v142, v139, vcc
	s_mul_i32 s1, s1, s60
	s_sub_i32 s0, s0, s72
	v_lshl_add_u64 v[138:139], v[138:139], 0, v[140:141]
	v_cmp_ne_u64_e32 vcc, 0, v[140:141]
	s_add_i32 s30, s1, s62
	s_mul_i32 s0, s60, s0
	v_cndmask_b32_e32 v141, 0, v139, vcc
	v_cndmask_b32_e32 v140, 0, v138, vcc
	s_ashr_i32 s31, s30, 31
	s_add_i32 s0, s51, s0
	v_lshl_add_u64 v[138:139], s[30:31], 1, v[140:141]
	v_lshl_add_u64 v[140:141], v[140:141], 0, v[156:157]
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[140:141], s[0:1], 1, v[140:141]
	s_branch .LBB4_117
.LBB4_115:                              ; %Flow2318
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB4_116:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_add_i32 s92, s92, s20
	s_cmp_ge_i32 s92, s42
	v_lshl_add_u64 v[140:141], v[140:141], 0, s[86:87]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB4_112
.LBB4_117:                              ; %.lr.ph993
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB4_475 Depth 4
                                        ;         Child Loop BB4_431 Depth 4
                                        ;         Child Loop BB4_458 Depth 4
                                        ;         Child Loop BB4_465 Depth 4
                                        ;           Child Loop BB4_470 Depth 5
	v_cndmask_b32_e64 v142, 0, 1, s[78:79]
	v_cmp_ne_u32_e64 s[30:31], 1, v142
	s_andn2_b64 vcc, exec, s[78:79]
	s_cbranch_vccnz .LBB4_119
; %bb.118:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_120
.LBB4_119:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_120:                              ;   in Loop: Header=BB4_117 Depth=3
	s_add_i32 s93, s92, s61
	s_add_i32 s68, s93, s20
	v_cmp_le_i32_e32 vcc, s93, v199
	v_cmp_gt_i32_e64 s[0:1], s68, v199
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_122
; %bb.121:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v126
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v199
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_122:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v201
	v_cmp_gt_i32_e64 s[0:1], s68, v201
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_124
; %bb.123:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v127
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v201
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_124:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v202
	v_cmp_gt_i32_e64 s[0:1], s68, v202
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_126
; %bb.125:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v128
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v202
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_126:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v203
	v_cmp_gt_i32_e64 s[0:1], s68, v203
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_128
; %bb.127:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v129
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v203
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_128:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_445
; %bb.129:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_131
.LBB4_130:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v122
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v199
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_131:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_148
; %bb.132:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_149
.LBB4_133:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_150
.LBB4_134:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_151
.LBB4_135:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_137
.LBB4_136:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v118
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v199
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_137:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_152
; %bb.138:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_153
.LBB4_139:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_154
.LBB4_140:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_155
.LBB4_141:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_143
.LBB4_142:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v114
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v199
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_143:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_156
; %bb.144:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_157
.LBB4_145:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_158
.LBB4_146:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_159
.LBB4_147:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_160
.LBB4_148:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v123
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v201
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_133
.LBB4_149:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v124
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v202
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_134
.LBB4_150:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v125
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v203
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_135
.LBB4_151:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_136
	s_branch .LBB4_137
.LBB4_152:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v119
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v201
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_139
.LBB4_153:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v120
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v202
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_140
.LBB4_154:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v121
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v203
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_141
.LBB4_155:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_142
	s_branch .LBB4_143
.LBB4_156:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v115
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v201
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_145
.LBB4_157:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v116
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v202
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_146
.LBB4_158:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v117
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v203
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_147
.LBB4_159:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_160:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v207
	v_cmp_gt_i32_e64 s[0:1], s68, v207
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_162
; %bb.161:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v110
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v207
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_162:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v208
	v_cmp_gt_i32_e64 s[0:1], s68, v208
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_164
; %bb.163:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v111
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v208
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_164:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v209
	v_cmp_gt_i32_e64 s[0:1], s68, v209
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_166
; %bb.165:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v112
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v209
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_166:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v210
	v_cmp_gt_i32_e64 s[0:1], s68, v210
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_168
; %bb.167:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v113
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v210
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_168:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_446
; %bb.169:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_171
.LBB4_170:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v106
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v207
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_171:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_188
; %bb.172:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_189
.LBB4_173:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_190
.LBB4_174:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_191
.LBB4_175:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_177
.LBB4_176:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v102
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v207
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_177:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_192
; %bb.178:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_193
.LBB4_179:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_194
.LBB4_180:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_195
.LBB4_181:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_183
.LBB4_182:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v98
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v207
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_183:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_196
; %bb.184:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_197
.LBB4_185:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_198
.LBB4_186:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_199
.LBB4_187:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_200
.LBB4_188:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v107
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v208
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_173
.LBB4_189:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v108
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v209
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_174
.LBB4_190:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v109
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v210
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_175
.LBB4_191:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_176
	s_branch .LBB4_177
.LBB4_192:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v103
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v208
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_179
.LBB4_193:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v104
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v209
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_180
.LBB4_194:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v105
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v210
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_181
.LBB4_195:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_182
	s_branch .LBB4_183
.LBB4_196:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v99
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v208
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_185
.LBB4_197:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v100
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v209
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_186
.LBB4_198:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v101
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v210
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_187
.LBB4_199:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_200:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v211
	v_cmp_gt_i32_e64 s[0:1], s68, v211
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_202
; %bb.201:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v94
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v211
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_202:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v212
	v_cmp_gt_i32_e64 s[0:1], s68, v212
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_204
; %bb.203:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v95
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v212
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_204:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v213
	v_cmp_gt_i32_e64 s[0:1], s68, v213
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_206
; %bb.205:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v96
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v213
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_206:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v214
	v_cmp_gt_i32_e64 s[0:1], s68, v214
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_208
; %bb.207:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v97
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v214
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_208:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_447
; %bb.209:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_211
.LBB4_210:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v90
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v211
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_211:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_228
; %bb.212:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_229
.LBB4_213:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_230
.LBB4_214:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_231
.LBB4_215:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_217
.LBB4_216:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v86
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v211
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_217:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_232
; %bb.218:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_233
.LBB4_219:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_234
.LBB4_220:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_235
.LBB4_221:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_223
.LBB4_222:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v82
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v211
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_223:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_236
; %bb.224:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_237
.LBB4_225:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_238
.LBB4_226:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_239
.LBB4_227:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_240
.LBB4_228:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v91
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v212
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_213
.LBB4_229:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v92
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v213
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_214
.LBB4_230:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v93
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v214
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_215
.LBB4_231:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_216
	s_branch .LBB4_217
.LBB4_232:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v87
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v212
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_219
.LBB4_233:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v88
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v213
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_220
.LBB4_234:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v89
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v214
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_221
.LBB4_235:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_222
	s_branch .LBB4_223
.LBB4_236:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v83
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v212
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_225
.LBB4_237:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v84
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v213
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_226
.LBB4_238:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v85
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v214
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_227
.LBB4_239:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_240:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v215
	v_cmp_gt_i32_e64 s[0:1], s68, v215
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_242
; %bb.241:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v78
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v215
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_242:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v216
	v_cmp_gt_i32_e64 s[0:1], s68, v216
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_244
; %bb.243:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v79
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v216
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_244:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v217
	v_cmp_gt_i32_e64 s[0:1], s68, v217
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_246
; %bb.245:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v80
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v217
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_246:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v218
	v_cmp_gt_i32_e64 s[0:1], s68, v218
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_248
; %bb.247:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v81
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v218
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_248:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_448
; %bb.249:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_251
.LBB4_250:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v74
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v215
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_251:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_268
; %bb.252:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_269
.LBB4_253:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_270
.LBB4_254:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_271
.LBB4_255:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_257
.LBB4_256:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v70
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v215
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_257:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_272
; %bb.258:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_273
.LBB4_259:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_274
.LBB4_260:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_275
.LBB4_261:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_263
.LBB4_262:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v66
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v215
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_263:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_276
; %bb.264:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_277
.LBB4_265:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_278
.LBB4_266:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_279
.LBB4_267:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_280
.LBB4_268:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v75
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v216
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_253
.LBB4_269:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v76
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v217
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_254
.LBB4_270:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v77
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v218
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_255
.LBB4_271:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_256
	s_branch .LBB4_257
.LBB4_272:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v71
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v216
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_259
.LBB4_273:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v72
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v217
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_260
.LBB4_274:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v73
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v218
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_261
.LBB4_275:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_262
	s_branch .LBB4_263
.LBB4_276:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v67
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v216
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_265
.LBB4_277:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v68
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v217
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_266
.LBB4_278:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v69
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v218
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_267
.LBB4_279:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_280:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v219
	v_cmp_gt_i32_e64 s[0:1], s68, v219
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_282
; %bb.281:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v62
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v219
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_282:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v220
	v_cmp_gt_i32_e64 s[0:1], s68, v220
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_284
; %bb.283:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v63
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v220
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_284:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v221
	v_cmp_gt_i32_e64 s[0:1], s68, v221
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_286
; %bb.285:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v64
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v221
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_286:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v222
	v_cmp_gt_i32_e64 s[0:1], s68, v222
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_288
; %bb.287:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v65
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v222
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_288:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_449
; %bb.289:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_291
.LBB4_290:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v58
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v219
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_291:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_308
; %bb.292:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_309
.LBB4_293:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_310
.LBB4_294:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_311
.LBB4_295:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_297
.LBB4_296:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v54
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v219
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_297:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_312
; %bb.298:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_313
.LBB4_299:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_314
.LBB4_300:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_315
.LBB4_301:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_303
.LBB4_302:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v50
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v219
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_303:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_316
; %bb.304:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_317
.LBB4_305:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_318
.LBB4_306:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_319
.LBB4_307:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_320
.LBB4_308:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v59
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v220
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_293
.LBB4_309:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v60
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v221
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_294
.LBB4_310:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v61
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v222
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_295
.LBB4_311:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_296
	s_branch .LBB4_297
.LBB4_312:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v55
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v220
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_299
.LBB4_313:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v56
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v221
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_300
.LBB4_314:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v57
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v222
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_301
.LBB4_315:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_302
	s_branch .LBB4_303
.LBB4_316:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v51
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v220
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_305
.LBB4_317:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v52
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v221
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_306
.LBB4_318:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v53
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v222
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_307
.LBB4_319:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_320:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v223
	v_cmp_gt_i32_e64 s[0:1], s68, v223
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_322
; %bb.321:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v46
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v223
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_322:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v224
	v_cmp_gt_i32_e64 s[0:1], s68, v224
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_324
; %bb.323:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v47
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v224
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_324:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v225
	v_cmp_gt_i32_e64 s[0:1], s68, v225
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_326
; %bb.325:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v48
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v225
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_326:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v226
	v_cmp_gt_i32_e64 s[0:1], s68, v226
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_328
; %bb.327:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v49
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v226
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_328:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_450
; %bb.329:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_331
.LBB4_330:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v42
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v223
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_331:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_348
; %bb.332:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_349
.LBB4_333:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_350
.LBB4_334:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_351
.LBB4_335:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_337
.LBB4_336:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v38
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v223
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_337:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_352
; %bb.338:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_353
.LBB4_339:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_354
.LBB4_340:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_355
.LBB4_341:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_343
.LBB4_342:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v34
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v223
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_343:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_356
; %bb.344:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_357
.LBB4_345:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_358
.LBB4_346:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_359
.LBB4_347:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_360
.LBB4_348:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v43
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v224
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_333
.LBB4_349:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v44
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v225
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_334
.LBB4_350:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v45
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v226
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_335
.LBB4_351:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_336
	s_branch .LBB4_337
.LBB4_352:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v39
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v224
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_339
.LBB4_353:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v40
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v225
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_340
.LBB4_354:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v41
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v226
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_341
.LBB4_355:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_342
	s_branch .LBB4_343
.LBB4_356:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v35
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v224
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_345
.LBB4_357:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v36
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v225
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_346
.LBB4_358:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v37
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v226
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_347
.LBB4_359:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_360:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v227
	v_cmp_gt_i32_e64 s[0:1], s68, v227
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_362
; %bb.361:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v30
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v227
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_362:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v228
	v_cmp_gt_i32_e64 s[0:1], s68, v228
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_364
; %bb.363:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v31
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v228
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_364:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v229
	v_cmp_gt_i32_e64 s[0:1], s68, v229
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_366
; %bb.365:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v32
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v229
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_366:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v230
	v_cmp_gt_i32_e64 s[0:1], s68, v230
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_368
; %bb.367:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v33
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v230
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_368:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_451
; %bb.369:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_371
.LBB4_370:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v26
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v227
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_371:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_388
; %bb.372:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_389
.LBB4_373:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_390
.LBB4_374:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_391
.LBB4_375:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_377
.LBB4_376:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v22
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v227
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_377:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_392
; %bb.378:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_393
.LBB4_379:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_394
.LBB4_380:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_395
.LBB4_381:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_383
.LBB4_382:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v18
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v227
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_383:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_396
; %bb.384:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_397
.LBB4_385:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_398
.LBB4_386:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_399
.LBB4_387:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[130:131]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_branch .LBB4_400
.LBB4_388:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v27
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v228
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_373
.LBB4_389:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v28
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v229
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_374
.LBB4_390:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v29
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v230
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_375
.LBB4_391:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_376
	s_branch .LBB4_377
.LBB4_392:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v23
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v228
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_379
.LBB4_393:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v24
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v229
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_380
.LBB4_394:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v25
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v230
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_381
.LBB4_395:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_382
	s_branch .LBB4_383
.LBB4_396:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v19
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v228
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_385
.LBB4_397:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v20
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v229
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_386
.LBB4_398:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v21
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v230
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_387
.LBB4_399:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
.LBB4_400:                              ;   in Loop: Header=BB4_117 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v231
	v_cmp_gt_i32_e64 s[0:1], s68, v231
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_402
; %bb.401:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v14
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v231
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_402:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v234
	v_cmp_gt_i32_e64 s[0:1], s68, v234
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_404
; %bb.403:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v15
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v234
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_404:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v235
	v_cmp_gt_i32_e64 s[0:1], s68, v235
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_406
; %bb.405:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v16
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v235
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143
.LBB4_406:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v236
	v_cmp_gt_i32_e64 s[0:1], s68, v236
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_408
; %bb.407:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v17
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v236
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142
.LBB4_408:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_452
; %bb.409:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[132:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_411
.LBB4_410:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v10
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v231
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
.LBB4_411:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_435
; %bb.412:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_436
.LBB4_413:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_437
.LBB4_414:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_438
.LBB4_415:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[134:135]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_417
.LBB4_416:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v6
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v231
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
.LBB4_417:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_439
; %bb.418:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_440
.LBB4_419:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_441
.LBB4_420:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_442
.LBB4_421:                              ;   in Loop: Header=BB4_117 Depth=3
	flat_load_ushort v142, v[136:137]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v142, 16, v142
	s_and_saveexec_b64 s[30:31], s[94:95]
	s_cbranch_execz .LBB4_423
.LBB4_422:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v2
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v231
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
.LBB4_423:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[96:97]
	s_cbranch_execnz .LBB4_443
; %bb.424:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[98:99]
	s_cbranch_execnz .LBB4_444
.LBB4_425:                              ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB4_427
.LBB4_426:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v5
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v236
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:96
.LBB4_427:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi128ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_mul_i32 s0, s75, s92
	s_mul_hi_u32 s1, s74, s92
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s74, s92
	v_lshl_add_u64 v[142:143], s[0:1], 1, v[138:139]
	s_andn2_b64 vcc, exec, s[80:81]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB4_453
; %bb.428:                              ;   in Loop: Header=BB4_117 Depth=3
	s_mov_b64 s[30:31], -1
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB4_455
; %bb.429:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v144, v182
	v_mov_b32_e32 v145, v233
	v_mov_b32_e32 v154, v239
	v_mov_b32_e32 v158, v0
                                        ; implicit-def: $sgpr94_sgpr95
                                        ; implicit-def: $sgpr96_sgpr97
	s_branch .LBB4_431
.LBB4_430:                              ; %Flow2313
                                        ;   in Loop: Header=BB4_431 Depth=4
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 s[72:73], exec, vcc
	s_or_b64 s[30:31], s[72:73], s[30:31]
	s_andn2_b64 s[72:73], s[94:95], exec
	s_and_b64 s[94:95], s[96:97], exec
	s_or_b64 s[94:95], s[72:73], s[94:95]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB4_454
.LBB4_431:                              ; %.lr.ph.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ;       Parent Loop BB4_117 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v159, 0xf8, v144
	v_add_u32_e32 v162, 8, v159
	v_cmp_lt_i32_e64 s[98:99], s66, v162
	v_cmp_ge_i32_e32 vcc, s66, v162
	s_and_saveexec_b64 s[72:73], vcc
	s_cbranch_execz .LBB4_433
; %bb.432:                              ;   in Loop: Header=BB4_431 Depth=4
	v_mul_lo_u32 v162, s74, v145
	v_lshlrev_b32_e32 v162, 1, v162
	v_lshlrev_b32_e32 v159, 1, v159
	v_add3_u32 v162, v142, v162, v159
	v_add_u32_e32 v159, v154, v159
	v_or_b32_e32 v159, v159, v162
	v_and_b32_e32 v159, 15, v159
	v_cmp_eq_u32_e32 vcc, 0, v159
	s_andn2_b64 s[98:99], s[98:99], exec
	s_and_b64 vcc, vcc, exec
	s_or_b64 s[98:99], s[98:99], vcc
.LBB4_433:                              ; %Flow2312
                                        ;   in Loop: Header=BB4_431 Depth=4
	s_or_b64 exec, exec, s[72:73]
	s_mov_b64 vcc, -1
	s_andn2_b64 s[96:97], s[96:97], exec
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_430
; %bb.434:                              ; %.critedge.i
                                        ;   in Loop: Header=BB4_431 Depth=4
	v_add_u32_e32 v158, 0x200, v158
	v_cmp_le_i32_e32 vcc, s82, v158
	v_add_u32_e32 v154, 0x2000, v154
	v_add_u32_e32 v145, 16, v145
	v_add_u32_e32 v144, 0x1000, v144
	s_or_b64 s[96:97], s[96:97], exec
	s_orn2_b64 vcc, vcc, exec
	s_branch .LBB4_430
.LBB4_435:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v11
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v234
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_413
.LBB4_436:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v12
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v235
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_414
.LBB4_437:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v13
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v236
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_415
.LBB4_438:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_416
	s_branch .LBB4_417
.LBB4_439:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v7
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v234
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_419
.LBB4_440:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v8
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v235
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_420
.LBB4_441:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v142, v142, v9
	v_bfe_u32 v143, v142, 16, 1
	v_add3_u32 v142, v142, v143, s26
	v_subrev_u32_e32 v143, s93, v236
	v_lshl_add_u32 v143, v143, 9, v200
	ds_write_b16_d16_hi v143, v142 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_421
.LBB4_442:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[30:31], s[94:95]
	s_cbranch_execnz .LBB4_422
	s_branch .LBB4_423
.LBB4_443:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v3
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v234
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[98:99]
	s_cbranch_execz .LBB4_425
.LBB4_444:                              ;   in Loop: Header=BB4_117 Depth=3
	v_add_f32_e32 v143, v142, v4
	v_bfe_u32 v144, v143, 16, 1
	v_add3_u32 v143, v143, v144, s26
	v_subrev_u32_e32 v144, s93, v235
	v_lshl_add_u32 v144, v144, 9, v200
	ds_write_b16_d16_hi v144, v143 offset:96
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execnz .LBB4_426
	s_branch .LBB4_427
.LBB4_445:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_130
	s_branch .LBB4_131
.LBB4_446:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_170
	s_branch .LBB4_171
.LBB4_447:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_210
	s_branch .LBB4_211
.LBB4_448:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_250
	s_branch .LBB4_251
.LBB4_449:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_290
	s_branch .LBB4_291
.LBB4_450:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_330
	s_branch .LBB4_331
.LBB4_451:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_370
	s_branch .LBB4_371
.LBB4_452:                              ;   in Loop: Header=BB4_117 Depth=3
	v_mov_b32_e32 v142, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_410
	s_branch .LBB4_411
.LBB4_453:                              ;   in Loop: Header=BB4_117 Depth=3
	s_cbranch_execz .LBB4_116
	s_branch .LBB4_473
.LBB4_454:                              ; %Flow2314
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_orn2_b64 s[30:31], s[94:95], exec
.LBB4_455:                              ; %Flow2315
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v144, 0, 1, s[30:31]
	s_nop 0
	v_readfirstlane_b32 s0, v144
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[30:31], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_460
; %bb.456:                              ;   in Loop: Header=BB4_117 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB4_459
; %bb.457:                              ; %.lr.ph.i355.preheader
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v145, v161
	v_mov_b32_e32 v144, v0
.LBB4_458:                              ; %.lr.ph.i355
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ;       Parent Loop BB4_117 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v154, v144, v160
	v_mul_lo_u32 v158, v154, s67
	v_sub_u32_e32 v158, v144, v158
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_subrev_u32_e32 v159, s67, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_xor_b32_e32 v154, s53, v154
	v_subrev_u32_e32 v162, s53, v154
	v_mad_u64_u32 v[158:159], s[72:73], s50, v162, v[144:145]
	v_lshlrev_b32_e32 v154, 9, v154
	v_mul_lo_u32 v159, s55, v162
	v_sub_u32_e32 v154, v154, v159
	v_add_u32_e32 v154, v145, v154
	ds_read_u16 v154, v154
	v_mad_i64_i32 v[162:163], s[72:73], s74, v162, 0
	v_add_u32_e32 v144, 0x200, v144
	v_mov_b32_e32 v159, v155
	v_lshl_add_u64 v[162:163], v[162:163], 1, v[142:143]
	v_cmp_le_i32_e32 vcc, s52, v144
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[162:163]
	v_add_u32_e32 v145, 0x400, v145
	s_or_b64 s[30:31], vcc, s[30:31]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v154
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB4_458
.LBB4_459:                              ; %Flow2304
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB4_460:                              ; %Flow2310
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB4_472
; %bb.461:                              ;   in Loop: Header=BB4_117 Depth=3
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB4_471
; %bb.462:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v162, v241
	v_mov_b64_e32 v[144:145], v[140:141]
	v_mov_b32_e32 v163, v0
	s_branch .LBB4_465
.LBB4_463:                              ; %Flow2306
                                        ;   in Loop: Header=BB4_465 Depth=4
	s_or_b64 exec, exec, s[96:97]
.LBB4_464:                              ; %.loopexit.i357
                                        ;   in Loop: Header=BB4_465 Depth=4
	s_or_b64 exec, exec, s[94:95]
	v_add_u32_e32 v163, 0x200, v163
	v_cmp_le_i32_e32 vcc, s82, v163
	v_lshl_add_u64 v[144:145], v[144:145], 0, s[90:91]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_add_u32_e32 v162, 0x2000, v162
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB4_471
.LBB4_465:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ;       Parent Loop BB4_117 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB4_470 Depth 5
	v_lshlrev_b32_e32 v154, 3, v163
	v_and_b32_e32 v154, 0xf8, v154
	v_add_u32_e32 v158, 8, v154
	v_cmp_ge_i32_e32 vcc, s66, v158
	s_and_saveexec_b64 s[72:73], vcc
	s_xor_b64 s[94:95], exec, s[72:73]
	s_cbranch_execz .LBB4_467
; %bb.466:                              ;   in Loop: Header=BB4_465 Depth=4
	v_lshrrev_b32_e32 v164, 5, v163
	v_mad_i64_i32 v[158:159], s[72:73], s74, v164, 0
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[142:143]
	v_lshlrev_b32_e32 v154, 1, v154
	v_lshlrev_b32_e32 v164, 9, v164
	v_lshl_add_u64 v[158:159], v[158:159], 0, v[154:155]
	v_add3_u32 v154, 0, v164, v154
	ds_read_b128 v[164:167], v154
                                        ; implicit-def: $vgpr154
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[158:159], v[164:167]
.LBB4_467:                              ; %Flow2307
                                        ;   in Loop: Header=BB4_465 Depth=4
	s_andn2_saveexec_b64 s[94:95], s[94:95]
	s_cbranch_execz .LBB4_464
; %bb.468:                              ; %.preheader.i
                                        ;   in Loop: Header=BB4_465 Depth=4
	v_cmp_gt_i32_e32 vcc, s63, v154
	s_and_saveexec_b64 s[96:97], vcc
	s_cbranch_execz .LBB4_463
; %bb.469:                              ; %.lr.ph.i358
                                        ;   in Loop: Header=BB4_465 Depth=4
	s_mov_b32 s68, 0
	s_mov_b64 s[98:99], 0
	v_mov_b32_e32 v154, v162
	v_mov_b64_e32 v[158:159], v[144:145]
.LBB4_470:                              ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ;       Parent Loop BB4_117 Depth=3
                                        ;         Parent Loop BB4_465 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v164, v154
	s_add_i32 s93, s68, 1
	s_cmp_gt_u32 s68, 6
	s_cselect_b64 s[72:73], -1, 0
	v_add_u32_e32 v154, 2, v154
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v164
	v_add_u32_e32 v164, s68, v238
	v_cmp_le_u32_e32 vcc, s66, v164
	s_or_b64 s[72:73], s[72:73], vcc
	s_and_b64 s[72:73], exec, s[72:73]
	v_lshl_add_u64 v[158:159], v[158:159], 0, 2
	s_or_b64 s[98:99], s[72:73], s[98:99]
	s_mov_b32 s68, s93
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execnz .LBB4_470
	s_branch .LBB4_463
.LBB4_471:                              ; %Flow2309
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB4_472:                              ; %Flow2311
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_branch .LBB4_116
.LBB4_473:                              ;   in Loop: Header=BB4_117 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB4_115
; %bb.474:                              ; %.lr.ph.i362.preheader
                                        ;   in Loop: Header=BB4_117 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v145, v161
	v_mov_b32_e32 v144, v0
.LBB4_475:                              ; %.lr.ph.i362
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_113 Depth=2
                                        ;       Parent Loop BB4_117 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v154, v144, v160
	v_mul_lo_u32 v158, v154, s67
	v_sub_u32_e32 v158, v144, v158
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_subrev_u32_e32 v159, s67, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v154
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v154, v154, v159, vcc
	v_xor_b32_e32 v154, s53, v154
	v_subrev_u32_e32 v162, s53, v154
	v_mad_u64_u32 v[158:159], s[72:73], s50, v162, v[144:145]
	v_lshlrev_b32_e32 v154, 9, v154
	v_mul_lo_u32 v159, s55, v162
	v_sub_u32_e32 v154, v154, v159
	v_add_u32_e32 v154, v145, v154
	ds_read_u16 v154, v154
	v_mad_i64_i32 v[162:163], s[72:73], s74, v162, 0
	v_add_u32_e32 v144, 0x200, v144
	v_mov_b32_e32 v159, v155
	v_lshl_add_u64 v[162:163], v[162:163], 1, v[142:143]
	v_cmp_le_i32_e32 vcc, s52, v144
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[162:163]
	v_add_u32_e32 v145, 0x400, v145
	s_or_b64 s[30:31], vcc, s[30:31]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v154
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB4_475
	s_branch .LBB4_115
.LBB4_476:                              ; %._crit_edge997
                                        ;   in Loop: Header=BB4_84 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v247, 4
	v_readlane_b32 s29, v247, 5
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_478
; %bb.477:                              ;   in Loop: Header=BB4_84 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB4_478:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v247, 22
	v_readlane_b32 s29, v247, 23
	v_readlane_b32 s66, v247, 41
	s_and_b64 s[28:29], s[0:1], s[28:29]
	v_readlane_b32 s67, v247, 42
	v_readlane_b32 s63, v247, 43
	v_readlane_b32 s61, v247, 44
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_82
; %bb.479:                              ; %.lr.ph999.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s3, v247, 46
	s_lshl_b32 s3, s3, 2
	s_add_i32 s3, s2, s3
	v_readlane_b32 s28, v247, 48
	s_sub_i32 s3, s3, s28
	v_readlane_b32 s28, v247, 50
	s_sub_i32 s3, s3, s28
	v_readlane_b32 s28, v247, 45
	s_lshl_b32 s28, s28, 2
	s_sub_i32 s3, s3, s28
	s_lshl_b32 s3, s3, 8
	s_mov_b32 s30, s22
	s_branch .LBB4_481
.LBB4_480:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB4_481 Depth=2
	s_add_i32 s30, s30, -1
	s_add_i32 s3, s3, s42
	s_cmp_lg_u32 s30, 0
	s_cbranch_scc0 .LBB4_82
.LBB4_481:                              ; %.lr.ph999
                                        ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[8:9], s[48:49]
	flat_load_dwordx4 v[2:5], v[8:9]
	s_abs_i32 s29, s3
	s_mul_hi_u32 s31, s29, s71
	s_mul_i32 s50, s31, s70
	s_ashr_i32 s28, s3, 31
	s_sub_i32 s29, s29, s50
	s_xor_b32 s28, s28, s23
	s_add_i32 s50, s31, 1
	s_sub_i32 s51, s29, s70
	s_cmp_ge_u32 s29, s70
	s_cselect_b32 s31, s50, s31
	s_cselect_b32 s29, s51, s29
	s_add_i32 s50, s31, 1
	s_cmp_ge_u32 s29, s70
	s_cselect_b32 s29, s50, s31
	s_xor_b32 s29, s29, s28
	s_sub_i32 s31, s29, s28
	s_mul_i32 s29, s85, s31
	s_add_i32 s29, s3, s29
	s_mul_i32 s28, s31, s33
	s_ashr_i32 s29, s29, 31
	s_sub_i32 s28, s29, s28
	s_add_i32 s28, s3, s28
	s_xor_b32 s28, s28, s29
	s_xor_b32 s50, s29, s34
	s_mul_hi_u32 s29, s28, s35
	s_mul_i32 s51, s29, s39
	s_sub_i32 s28, s28, s51
	s_add_i32 s51, s29, 1
	s_sub_i32 s52, s28, s39
	s_cmp_ge_u32 s28, s39
	s_cselect_b32 s29, s51, s29
	s_cselect_b32 s28, s52, s28
	s_add_i32 s51, s29, 1
	s_cmp_ge_u32 s28, s39
	s_cselect_b32 s28, s51, s29
	s_xor_b32 s28, s28, s50
	s_sub_i32 s28, s28, s50
	s_mul_i32 s29, s40, s36
	s_add_i32 s28, s28, s29
	s_mul_i32 s28, s28, s41
	s_add_i32 s28, s28, s27
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[28:29], s[28:29], 2
	s_add_u32 s28, s46, s28
	s_addc_u32 s50, s47, s29
	s_add_u32 s29, s28, 0x100
	s_addc_u32 s28, s50, 0
	s_cmp_eq_u32 s31, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s31, 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v10, 0, v4, vcc
	v_cndmask_b32_e32 v11, 0, v5, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:16
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s31, 2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v10, v4, v6, vcc
	v_cndmask_b32_e32 v11, v5, v7, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:32
	s_cmp_eq_u32 s31, 3
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s31, 4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v10, v4, v6, vcc
	v_cndmask_b32_e32 v11, v5, v7, vcc
	flat_load_dwordx4 v[4:7], v[8:9] offset:48
	s_cmp_eq_u32 s31, 5
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s31, 6
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v4, v10, v4, vcc
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v6, v4, v6, vcc
	v_cndmask_b32_e32 v7, v5, v7, vcc
	flat_load_dwordx2 v[4:5], v[8:9] offset:64
	s_cmp_eq_u32 s31, 7
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s31, s36
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v5, v7, v5, vcc
	v_cndmask_b32_e32 v4, v6, v4, vcc
	v_sub_co_u32_e32 v2, vcc, s29, v2
	v_mov_b32_e32 v6, s28
	s_nop 0
	v_subb_co_u32_e32 v3, vcc, v6, v3, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_mov_b64 s[28:29], -1
	s_nop 0
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB4_483
; %bb.482:                              ; %Flow2300
                                        ;   in Loop: Header=BB4_481 Depth=2
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB4_480
	s_branch .LBB4_484
.LBB4_483:                              ;   in Loop: Header=BB4_481 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB4_480
.LBB4_484:                              ;   in Loop: Header=BB4_481 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB4_480
.LBB4_485:                              ; %.critedge270
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 248
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 248
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals,comdat
.Lfunc_end4:
	.size	_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals, .Lfunc_end4-_Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.num_vgpr, 248
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 22948
; TotalNumSgprs: 106
; NumVgprs: 248
; NumAgprs: 0
; TotalNumVgprs: 248
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 248
; AccumOffset: 248
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 61
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[14:15], s[0:1], 0x120
                                        ; implicit-def: $vgpr90 : SGPR spill to VGPR lane
	s_mov_b32 s86, s2
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v90, s4, 0
	s_nop 1
	v_writelane_b32 v90, s5, 1
	s_load_dwordx8 s[56:63], s[0:1], 0x100
	s_load_dwordx8 s[64:71], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s33, s57, 31
	s_lshr_b32 s3, s33, 29
	v_writelane_b32 v90, s4, 2
	s_add_i32 s3, s57, s3
	s_ashr_i32 s72, s3, 3
	v_writelane_b32 v90, s5, 3
	v_writelane_b32 v90, s6, 4
	v_writelane_b32 v90, s7, 5
	s_load_dwordx2 s[4:5], s[0:1], 0xf8
	s_mov_b64 s[6:7], -1
	s_cmp_ge_i32 s2, s63
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v90, s4, 6
	s_nop 1
	v_writelane_b32 v90, s5, 7
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	v_writelane_b32 v90, s72, 8
	s_cbranch_scc0 .LBB5_73
; %bb.1:
	s_sub_i32 s16, s86, s63
	s_mov_b32 s17, 0
	s_lshl_b64 s[6:7], s[16:17], 2
	s_add_u32 s6, s70, s6
	s_addc_u32 s7, s71, s7
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB5_3
; %bb.2:
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[6:7]
	flat_atomic_add v[2:3], v1 offset:1216
.LBB5_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit329
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	flat_load_dword v1, v[2:3] offset:1216 sc1
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[10:11], 0
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_eq_u64 s[10:11], 0
	s_cbranch_scc1 .LBB5_7
; %bb.4:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB5_6
; %bb.5:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB5_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit332
	s_or_b64 exec, exec, s[6:7]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[6:7], 0, v2
	s_branch .LBB5_8
.LBB5_7:
	s_mov_b64 s[6:7], -1
.LBB5_8:                                ; %Flow843
	s_mov_b64 s[2:3], exec
	v_writelane_b32 v90, s2, 59
	s_and_b64 s[6:7], s[2:3], s[6:7]
	s_nop 0
	v_writelane_b32 v90, s3, 60
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB5_72
; %bb.9:                                ; %.critedge
	s_mul_i32 s87, s61, s60
	s_cmp_ge_i32 s16, s87
	s_cbranch_scc1 .LBB5_72
; %bb.10:                               ; %.lr.ph
	s_load_dwordx2 s[40:41], s[0:1], 0x60
	s_load_dwordx2 s[80:81], s[0:1], 0x90
	v_readlane_b32 s2, v90, 8
	s_mul_hi_i32 s13, s2, s58
	s_mul_i32 s12, s2, s58
	s_ashr_i32 s23, s58, 31
	s_lshl_b64 s[6:7], s[12:13], 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s24, s40, s6
	s_load_dwordx2 s[2:3], s[0:1], 0x120
	s_addc_u32 s25, s41, s7
	s_add_u32 s26, s24, s6
	s_addc_u32 s27, s25, s7
	s_add_u32 s28, s26, s6
	s_addc_u32 s29, s27, s7
	s_add_u32 s30, s28, s6
	s_addc_u32 s31, s29, s7
	s_add_u32 s34, s30, s6
	s_addc_u32 s35, s31, s7
	s_add_u32 s36, s34, s6
	s_addc_u32 s37, s35, s7
	s_add_u32 s38, s36, s6
	s_addc_u32 s39, s37, s7
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s3, 0
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s17, s62, 3
	s_add_i32 s2, s2, 64
	s_cmp_lg_u32 s56, 0
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 18
	s_cmp_lg_u32 s56, 1
	s_mov_b32 s22, s58
	v_writelane_b32 v90, s7, 19
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 21
	s_cmp_lg_u32 s56, 2
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_writelane_b32 v90, s7, 22
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 23
	s_cmp_lg_u32 s56, 3
	v_mbcnt_hi_u32_b32 v4, -1, v4
	v_writelane_b32 v90, s7, 24
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 28
	s_cmp_lg_u32 s56, 4
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v90, s7, 29
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 30
	s_cmp_lg_u32 s56, 5
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v90, s7, 31
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 16
	s_cmp_lg_u32 s56, 6
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v90, s7, 17
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 9
	s_cmp_lg_u32 s56, 7
	v_mul_lo_u32 v64, s60, v0
	v_writelane_b32 v90, s7, 10
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s73, s61
	v_cvt_f32_u32_e32 v2, s73
	s_sub_i32 s3, 0, s73
	s_ashr_i32 s74, s61, 31
	s_lshl_b64 s[82:83], s[22:23], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v90, s6, 51
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v90, s7, 52
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_lshlrev_b32_e32 v65, 3, v0
	v_readfirstlane_b32 s20, v2
	s_mul_i32 s3, s3, s20
	s_mul_hi_u32 s3, s20, s3
	s_add_i32 s75, s20, s3
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v90, s20, 53
	s_mul_i32 s3, s13, 14
	v_lshrrev_b32_e32 v2, 3, v0
	v_writelane_b32 v90, s21, 54
	s_mul_hi_u32 s20, s12, 14
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 14
	s_add_u32 s3, s40, s3
	s_addc_u32 s20, s41, s20
	s_add_u32 s42, s3, 2
	s_addc_u32 s43, s20, 0
	s_lshl_b64 s[20:21], s[12:13], 2
	s_add_u32 s88, s40, s20
	s_mul_i32 s3, s13, 12
	s_mul_hi_u32 s20, s12, 12
	s_addc_u32 s89, s41, s21
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 12
	s_add_u32 s3, s40, s3
	s_addc_u32 s20, s41, s20
	s_add_u32 s90, s3, 2
	s_addc_u32 s91, s20, 0
	s_mul_i32 s3, s13, 6
	s_mul_hi_u32 s20, s12, 6
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 6
	s_add_u32 s92, s40, s3
	s_addc_u32 s93, s41, s20
	s_mul_i32 s3, s13, 10
	s_mul_hi_u32 s20, s12, 10
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 10
	s_add_u32 s3, s40, s3
	s_addc_u32 s20, s41, s20
	s_add_u32 s94, s3, 2
	s_addc_u32 s95, s20, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	s_add_u32 s96, s40, s12
	v_mov_b32_e32 v3, v5
	v_writelane_b32 v90, s42, 55
	s_addc_u32 s97, s41, s13
	s_mov_b64 s[98:99], 0
	v_bfrev_b32_e32 v66, 32
	s_xor_b64 s[40:41], s[14:15], -1
	s_movk_i32 s78, 0x7fff
	s_mov_b32 s79, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	v_writelane_b32 v90, s43, 56
	s_branch .LBB5_13
.LBB5_11:                               ; %Flow828
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_sub_i32 s3, s16, s63
	s_add_i32 s16, s3, 0x130
	s_cmp_ge_i32 s16, s87
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB5_12:                               ; %Flow840
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[42:43]
	s_and_b64 s[12:13], exec, s[12:13]
	s_or_b64 s[98:99], s[12:13], s[98:99]
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execz .LBB5_72
.LBB5_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_17 Depth 2
                                        ;     Child Loop BB5_29 Depth 2
                                        ;       Child Loop BB5_33 Depth 3
	s_abs_i32 s12, s16
	s_mul_hi_u32 s13, s12, s75
	s_mul_i32 s14, s13, s73
	s_ashr_i32 s3, s16, 31
	s_sub_i32 s12, s12, s14
	s_xor_b32 s3, s3, s74
	s_add_i32 s14, s13, 1
	s_sub_i32 s15, s12, s73
	s_cmp_ge_u32 s12, s73
	s_cselect_b32 s13, s14, s13
	s_cselect_b32 s12, s15, s12
	s_add_i32 s14, s13, 1
	s_cmp_ge_u32 s12, s73
	s_cselect_b32 s12, s14, s13
	s_xor_b32 s12, s12, s3
	s_sub_i32 s72, s12, s3
	s_mul_i32 s3, s72, s61
	s_sub_i32 s3, s16, s3
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB5_21
; %bb.14:                               ;   in Loop: Header=BB5_13 Depth=1
	v_add_u32_e32 v4, s72, v64
	v_mul_lo_u32 v4, v4, s61
	v_add_u32_e32 v6, s3, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[66:67]
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v4, v1
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB5_21
; %bb.15:                               ; %.lr.ph.i.i.i334.preheader
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[46:47], 0
                                        ; implicit-def: $sgpr42_sgpr43
                                        ; implicit-def: $sgpr44_sgpr45
	s_branch .LBB5_17
.LBB5_16:                               ; %Flow836
                                        ;   in Loop: Header=BB5_17 Depth=2
	s_and_b64 s[20:21], exec, s[44:45]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[42:43], exec
	s_and_b64 s[42:43], s[48:49], exec
	s_or_b64 s[42:43], s[20:21], s[42:43]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB5_19
.LBB5_17:                               ; %.lr.ph.i.i.i334
                                        ;   Parent Loop BB5_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_load_dwordx2 s[20:21], s[0:1], 0xf8
	s_add_u32 s46, s46, 1
	s_addc_u32 s47, s47, 0
	s_mov_b64 s[48:49], -1
	s_or_b64 s[44:45], s[44:45], exec
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[8:9], s[20:21]
	v_cmp_gt_u64_e32 vcc, s[46:47], v[8:9]
	s_cbranch_vccnz .LBB5_16
; %bb.18:                               ;   in Loop: Header=BB5_17 Depth=2
	s_sleep 4
	flat_load_dword v4, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[20:21], s[44:45], exec
	s_mov_b64 s[48:49], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v4, v1
	s_and_b64 s[44:45], vcc, exec
	s_or_b64 s[44:45], s[20:21], s[44:45]
	s_branch .LBB5_16
.LBB5_19:                               ; %loop.exit.guard
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[42:43]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB5_21
; %bb.20:                               ; %_ZN17hk_gemm_rs_mi300x15wait_band_epochEPKjjmPii.exit
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_atomic_or v[6:7], v66
.LBB5_21:                               ; %.critedge500
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_andn2_b64 vcc, exec, s[18:19]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB5_25
; %bb.22:                               ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b64 s[14:15], s[8:9]
	s_cbranch_execz .LBB5_24
; %bb.23:                               ;   in Loop: Header=BB5_13 Depth=1
	s_load_dwordx4 s[44:47], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_load_dword v4, v[6:7] sc1
.LBB5_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit340
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB5_25:                               ; %Flow839
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_and_saveexec_b64 s[42:43], s[14:15]
	s_cbranch_execz .LBB5_12
; %bb.26:                               ; %.critedge514
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB5_39
; %bb.27:                               ; %.lr.ph.i341.preheader
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_mul_i32 s44, s72, s62
	s_lshl_b32 s46, s3, 6
	s_ashr_i32 s45, s44, 31
	s_ashr_i32 s47, s46, 31
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[44:45]
	v_mov_b64_e32 v[8:9], s[46:47]
	v_mad_u64_u32 v[8:9], s[14:15], s22, v6, v[8:9]
	s_load_dwordx2 s[14:15], s[0:1], 0x60
	v_mul_lo_u32 v4, s22, v7
	v_mul_lo_u32 v6, s23, v6
	v_add3_u32 v9, v6, v9, v4
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 53
	v_readlane_b32 s15, v90, 54
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[22:23]
	v_lshl_add_u64 v[14:15], s[88:89], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v90, 55
	v_readlane_b32 s15, v90, 56
	v_lshl_add_u64 v[16:17], s[90:91], 0, v[22:23]
	v_lshl_add_u64 v[18:19], s[92:93], 0, v[22:23]
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[22:23]
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	s_mov_b64 s[48:49], 0
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	s_branch .LBB5_29
.LBB5_28:                               ; %.critedge.i344
                                        ;   in Loop: Header=BB5_29 Depth=2
	s_or_b64 exec, exec, vcc
	v_add_u32_e32 v69, 0x200, v69
	v_cmp_le_i32_e32 vcc, s17, v69
	v_add_u32_e32 v68, 0x1000, v68
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[82:83]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[82:83]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[82:83]
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[82:83]
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[82:83]
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[82:83]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[82:83]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[82:83]
	s_or_b64 s[48:49], vcc, s[48:49]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[82:83]
	s_andn2_b64 exec, exec, s[48:49]
	s_cbranch_execz .LBB5_39
.LBB5_29:                               ; %.lr.ph.i341
                                        ;   Parent Loop BB5_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_33 Depth 3
	v_lshlrev_b32_e32 v4, 3, v69
	v_and_or_b32 v24, v4, 56, s46
	v_mov_b32_e32 v25, s47
	v_lshl_add_u64 v[26:27], v[24:25], 0, 8
	v_cmp_lt_u64_e32 vcc, s[22:23], v[26:27]
	s_or_b64 s[14:15], s[40:41], vcc
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_xor_b64 s[50:51], exec, s[20:21]
	s_cbranch_execz .LBB5_37
; %bb.30:                               ; %.preheader.i343.preheader
                                        ;   in Loop: Header=BB5_29 Depth=2
	v_and_b32_e32 v4, 56, v68
	v_lshl_add_u64 v[24:25], s[46:47], 0, v[4:5]
	v_lshlrev_b32_e32 v4, 1, v68
	v_and_b32_e32 v4, 0x70, v4
	v_lshl_add_u64 v[26:27], v[6:7], 0, v[4:5]
	v_lshl_add_u64 v[28:29], v[8:9], 0, v[4:5]
	v_lshl_add_u64 v[30:31], v[10:11], 0, v[4:5]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[4:5]
	v_lshl_add_u64 v[34:35], v[14:15], 0, v[4:5]
	v_lshl_add_u64 v[36:37], v[16:17], 0, v[4:5]
	v_lshl_add_u64 v[38:39], v[18:19], 0, v[4:5]
	v_lshl_add_u64 v[40:41], v[20:21], 0, v[4:5]
	v_lshl_add_u64 v[42:43], v[22:23], 0, v[4:5]
	s_mov_b64 s[52:53], 0
	v_mov_b64_e32 v[44:45], 0
                                        ; implicit-def: $sgpr54_sgpr55
	s_branch .LBB5_33
.LBB5_31:                               ; %Flow830
                                        ;   in Loop: Header=BB5_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[54:55], s[54:55], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[54:55], s[54:55], s[20:21]
.LBB5_32:                               ; %Flow829
                                        ;   in Loop: Header=BB5_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[54:55]
	s_or_b64 s[52:53], s[14:15], s[52:53]
	s_andn2_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB5_36
.LBB5_33:                               ; %.preheader.i343
                                        ;   Parent Loop BB5_13 Depth=1
                                        ;     Parent Loop BB5_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[22:23], v[24:25]
	s_or_b64 s[54:55], s[54:55], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB5_32
; %bb.34:                               ; %.preheader.i343.1
                                        ;   in Loop: Header=BB5_33 Depth=3
	v_lshl_add_u64 v[48:49], v[26:27], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[30:31], 0, v[44:45]
	global_load_ushort v4, v[48:49], off
	global_load_ushort v72, v[50:51], off
	v_lshl_add_u64 v[52:53], v[34:35], 0, v[44:45]
	global_load_ushort v73, v[52:53], off
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[44:45]
	global_load_ushort v74, v[54:55], off
	v_lshl_add_u64 v[56:57], v[42:43], 0, v[44:45]
	global_load_ushort v75, v[56:57], off
	v_lshl_add_u64 v[58:59], v[40:41], 0, v[44:45]
	global_load_ushort v76, v[58:59], off offset:-2
	v_lshl_add_u64 v[60:61], v[36:37], 0, v[44:45]
	global_load_ushort v77, v[60:61], off offset:-2
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[44:45]
	global_load_ushort v78, v[62:63], off offset:-2
	v_lshl_add_u64 v[70:71], v[24:25], 0, 1
	v_cmp_gt_u64_e32 vcc, s[22:23], v[70:71]
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[44:45]
	s_mov_b64 s[20:21], -1
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v70, 16, v72
	v_add_f32_e32 v4, v4, v70
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v71, 16, v73
	v_add_f32_e32 v4, v4, v71
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v72, 16, v74
	v_add_f32_e32 v4, v4, v72
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v73, 16, v75
	v_add_f32_e32 v4, v4, v73
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v74, 16, v76
	v_add_f32_e32 v4, v4, v74
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v75, 16, v77
	v_add_f32_e32 v4, v4, v75
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v76, 16, v78
	v_add_f32_e32 v4, v4, v76
	v_bfe_u32 v70, v4, 16, 1
	v_add3_u32 v4, v4, v70, s78
	global_store_short_d16_hi v[46:47], v4, off offset:-2
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB5_31
; %bb.35:                               ;   in Loop: Header=BB5_33 Depth=3
	global_load_ushort v4, v[50:51], off offset:2
	s_nop 0
	global_load_ushort v48, v[48:49], off offset:2
	s_nop 0
	global_load_ushort v49, v[52:53], off offset:2
	global_load_ushort v50, v[54:55], off offset:2
	global_load_ushort v51, v[56:57], off offset:2
	s_nop 0
	global_load_ushort v52, v[58:59], off
	global_load_ushort v53, v[60:61], off
	global_load_ushort v54, v[62:63], off
	v_lshl_add_u64 v[44:45], v[44:45], 0, 4
	v_cmp_eq_u32_e32 vcc, 16, v44
	v_lshl_add_u64 v[24:25], v[24:25], 0, 2
	s_orn2_b64 s[20:21], vcc, exec
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v4, 16, v4
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v48, 16, v48
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v49, 16, v49
	v_add_f32_e32 v4, v48, v4
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v50, 16, v50
	v_add_f32_e32 v4, v4, v49
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v51, 16, v51
	v_add_f32_e32 v4, v4, v50
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v52, 16, v52
	v_add_f32_e32 v4, v4, v51
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v53, 16, v53
	v_add_f32_e32 v4, v4, v52
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v54
	v_add_f32_e32 v4, v4, v53
	v_add_f32_e32 v4, v4, v54
	v_bfe_u32 v48, v4, 16, 1
	v_add3_u32 v4, v4, v48, s78
	global_store_short_d16_hi v[46:47], v4, off
	s_branch .LBB5_31
.LBB5_36:                               ; %Flow831
                                        ;   in Loop: Header=BB5_29 Depth=2
	s_or_b64 exec, exec, s[52:53]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB5_37:                               ; %Flow832
                                        ;   in Loop: Header=BB5_29 Depth=2
	s_andn2_saveexec_b64 vcc, s[50:51]
	s_cbranch_execz .LBB5_28
; %bb.38:                               ;   in Loop: Header=BB5_29 Depth=2
	v_lshrrev_b32_e32 v4, 3, v69
	v_lshl_add_u64 v[26:27], v[4:5], 0, s[44:45]
	v_mad_u64_u32 v[24:25], s[14:15], v26, s22, v[24:25]
	s_load_dwordx2 s[14:15], s[0:1], 0x60
	v_mul_lo_u32 v4, v26, s23
	v_mul_lo_u32 v27, v27, s22
	v_add3_u32 v25, v27, v25, v4
	v_lshlrev_b64 v[56:57], 1, v[24:25]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[24:25], s[14:15], 0, v[56:57]
	v_lshl_add_u64 v[28:29], s[24:25], 0, v[56:57]
	global_load_dwordx4 v[24:27], v[24:25], off
	v_lshl_add_u64 v[32:33], s[26:27], 0, v[56:57]
	global_load_dwordx4 v[28:31], v[28:29], off
	v_lshl_add_u64 v[36:37], s[28:29], 0, v[56:57]
	global_load_dwordx4 v[32:35], v[32:33], off
	v_lshl_add_u64 v[40:41], s[30:31], 0, v[56:57]
	global_load_dwordx4 v[36:39], v[36:37], off
	v_lshl_add_u64 v[44:45], s[34:35], 0, v[56:57]
	global_load_dwordx4 v[40:43], v[40:41], off
	v_lshl_add_u64 v[48:49], s[36:37], 0, v[56:57]
	global_load_dwordx4 v[44:47], v[44:45], off
	v_lshl_add_u64 v[52:53], s[38:39], 0, v[56:57]
	global_load_dwordx4 v[48:51], v[48:49], off
	v_lshl_add_u64 v[56:57], s[80:81], 0, v[56:57]
	global_load_dwordx4 v[52:55], v[52:53], off
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v59, 0xffff0000, v24
	v_and_b32_e32 v61, 0xffff0000, v25
	v_lshlrev_b32_e32 v58, 16, v24
	v_lshlrev_b32_e32 v60, 16, v25
	s_waitcnt vmcnt(6)
	v_and_b32_e32 v25, 0xffff0000, v28
	v_lshlrev_b32_e32 v24, 16, v28
	v_and_b32_e32 v63, 0xffff0000, v26
	v_and_b32_e32 v71, 0xffff0000, v27
	v_lshlrev_b32_e32 v62, 16, v26
	v_lshlrev_b32_e32 v70, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v29
	v_lshlrev_b32_e32 v26, 16, v29
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v29, 0xffff0000, v32
	v_lshlrev_b32_e32 v28, 16, v32
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
	v_and_b32_e32 v73, 0xffff0000, v30
	v_and_b32_e32 v75, 0xffff0000, v31
	v_lshlrev_b32_e32 v72, 16, v30
	v_lshlrev_b32_e32 v74, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v33
	v_lshlrev_b32_e32 v30, 16, v33
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v33, 0xffff0000, v36
	v_lshlrev_b32_e32 v32, 16, v36
	v_pk_add_f32 v[26:27], v[26:27], v[60:61]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_and_b32_e32 v77, 0xffff0000, v34
	v_and_b32_e32 v79, 0xffff0000, v35
	v_lshlrev_b32_e32 v76, 16, v34
	v_lshlrev_b32_e32 v78, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v37
	v_lshlrev_b32_e32 v34, 16, v37
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v36, 16, v40
	v_and_b32_e32 v37, 0xffff0000, v40
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[32:33]
	v_and_b32_e32 v81, 0xffff0000, v38
	v_and_b32_e32 v83, 0xffff0000, v39
	v_lshlrev_b32_e32 v80, 16, v38
	v_lshlrev_b32_e32 v82, 16, v39
	v_lshlrev_b32_e32 v38, 16, v41
	v_and_b32_e32 v39, 0xffff0000, v41
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v84, 16, v44
	v_and_b32_e32 v85, 0xffff0000, v44
	v_pk_add_f32 v[26:27], v[26:27], v[34:35]
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	v_lshlrev_b32_e32 v40, 16, v45
	v_and_b32_e32 v41, 0xffff0000, v45
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v44, 16, v48
	v_and_b32_e32 v45, 0xffff0000, v48
	v_pk_add_f32 v[26:27], v[26:27], v[38:39]
	v_pk_add_f32 v[24:25], v[24:25], v[84:85]
	v_lshlrev_b32_e32 v86, 16, v49
	v_and_b32_e32 v87, 0xffff0000, v49
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v88, 16, v52
	v_and_b32_e32 v89, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[88:89]
	v_pk_add_f32 v[26:27], v[26:27], v[48:49]
	v_bfe_u32 v29, v25, 16, 1
	v_bfe_u32 v30, v24, 16, 1
	v_pk_add_f32 v[52:53], v[72:73], v[62:63]
	v_bfe_u32 v4, v27, 16, 1
	v_bfe_u32 v28, v26, 16, 1
	v_add3_u32 v32, v24, v30, s78
	v_add3_u32 v33, v25, v29, s78
	v_pk_add_f32 v[24:25], v[74:75], v[70:71]
	v_add3_u32 v34, v26, v28, s78
	v_add3_u32 v4, v27, v4, s78
	v_pk_add_f32 v[24:25], v[24:25], v[78:79]
	v_pk_add_f32 v[26:27], v[52:53], v[76:77]
	v_pk_add_f32 v[24:25], v[24:25], v[82:83]
	v_pk_add_f32 v[26:27], v[26:27], v[80:81]
	v_lshlrev_b32_e32 v28, 16, v42
	v_lshlrev_b32_e32 v30, 16, v43
	v_and_b32_e32 v29, 0xffff0000, v42
	v_and_b32_e32 v31, 0xffff0000, v43
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v47
	v_lshlrev_b32_e32 v30, 16, v46
	v_and_b32_e32 v29, 0xffff0000, v47
	v_and_b32_e32 v31, 0xffff0000, v46
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v50
	v_lshlrev_b32_e32 v30, 16, v51
	v_and_b32_e32 v29, 0xffff0000, v50
	v_and_b32_e32 v31, 0xffff0000, v51
	v_pk_add_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	v_lshlrev_b32_e32 v28, 16, v55
	v_lshlrev_b32_e32 v30, 16, v54
	v_and_b32_e32 v29, 0xffff0000, v55
	v_and_b32_e32 v31, 0xffff0000, v54
	v_pk_add_f32 v[26:27], v[26:27], v[30:31]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_bfe_u32 v30, v27, 16, 1
	v_bfe_u32 v28, v25, 16, 1
	v_bfe_u32 v29, v24, 16, 1
	v_bfe_u32 v31, v26, 16, 1
	v_add3_u32 v26, v26, v31, s78
	v_add3_u32 v30, v27, v30, s78
	v_add3_u32 v24, v24, v29, s78
	v_add3_u32 v25, v25, v28, s78
	v_perm_b32 v27, v25, v24, s79
	v_perm_b32 v26, v30, v26, s79
	v_perm_b32 v25, v4, v34, s79
	v_perm_b32 v24, v33, v32, s79
	global_store_dwordx4 v[56:57], v[24:27], off
	s_branch .LBB5_28
.LBB5_39:                               ; %Flow834
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[44:45], s[4:5]
	s_cbranch_execz .LBB5_11
; %bb.40:                               ; %.preheader523
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s60, s56
	s_add_i32 s12, s72, s12
	s_add_i32 s3, s2, s3
	s_mul_i32 s12, s12, s61
	s_add_i32 s12, s3, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s3, s66, s12
	s_addc_u32 s14, s67, s13
	v_readlane_b32 s12, v90, 18
	v_readlane_b32 s13, v90, 19
	s_and_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v4, s14
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v6, s[12:13], s3, v6
	s_nop 1
	v_subb_co_u32_e64 v7, s[12:13], v4, v7, s[12:13]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v7, s[12:13]
	v_cndmask_b32_e64 v6, 0, v6, s[12:13]
	s_cbranch_vccz .LBB5_71
; %bb.41:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB5_43
.LBB5_42:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_43:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:16
	v_readlane_b32 s12, v90, 21
	v_readlane_b32 s13, v90, 22
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_45
; %bb.44:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_45:                               ; %Flow826
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_47
; %bb.46:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_47:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.1
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:24
	v_readlane_b32 s12, v90, 23
	v_readlane_b32 s13, v90, 24
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_49
; %bb.48:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_49:                               ; %Flow825
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_51
; %bb.50:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_51:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.2
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:32
	v_readlane_b32 s12, v90, 28
	v_readlane_b32 s13, v90, 29
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_53
; %bb.52:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_53:                               ; %Flow824
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_55
; %bb.54:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_55:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.3
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:40
	v_readlane_b32 s12, v90, 30
	v_readlane_b32 s13, v90, 31
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_57
; %bb.56:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_57:                               ; %Flow823
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_59
; %bb.58:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_59:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.4
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:48
	v_readlane_b32 s12, v90, 16
	v_readlane_b32 s13, v90, 17
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_61
; %bb.60:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_61:                               ; %Flow822
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_63
; %bb.62:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_63:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.5
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:56
	v_readlane_b32 s12, v90, 9
	v_readlane_b32 s13, v90, 10
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_65
; %bb.64:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_65:                               ; %Flow821
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_67
; %bb.66:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
.LBB5_67:                               ; %_ZN17hk_gemm_rs_mi300x20publish_reuse_creditEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit.6
                                        ;   in Loop: Header=BB5_13 Depth=1
	v_mov_b64_e32 v[6:7], s[68:69]
	flat_load_dwordx2 v[8:9], v[6:7]
	s_nop 0
	flat_load_dwordx2 v[6:7], v[6:7] offset:64
	v_readlane_b32 s12, v90, 51
	v_readlane_b32 s13, v90, 52
	v_mov_b32_e32 v4, s14
	s_andn2_b64 vcc, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_co_u32_e64 v8, s[12:13], s3, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[12:13], v4, v9, s[12:13]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]
	s_nop 1
	v_cndmask_b32_e64 v7, 0, v9, s[12:13]
	v_cndmask_b32_e64 v6, 0, v8, s[12:13]
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB5_69
; %bb.68:                               ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB5_69:                               ; %Flow
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB5_11
; %bb.70:                               ;   in Loop: Header=BB5_13 Depth=1
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB5_11
.LBB5_71:                               ;   in Loop: Header=BB5_13 Depth=1
	s_branch .LBB5_42
.LBB5_72:                               ; %Flow844
	v_readlane_b32 s2, v90, 59
	v_readlane_b32 s3, v90, 60
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
	v_readlane_b32 s72, v90, 8
.LBB5_73:                               ; %Flow886
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB5_173
; %bb.74:
	s_ashr_i32 s87, s86, 31
	s_mov_b32 s2, s86
	v_writelane_b32 v90, s2, 9
	s_lshl_b64 s[4:5], s[86:87], 2
	s_add_u32 s4, s70, s4
	v_writelane_b32 v90, s3, 10
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_addc_u32 s5, s71, s5
	s_nop 0
	v_writelane_b32 v90, s2, 11
	s_nop 1
	v_writelane_b32 v90, s3, 12
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB5_76
; %bb.75:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, 1
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_atomic_add v[2:3], v1
.LBB5_76:                               ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit
	s_or_b64 exec, exec, s[6:7]
	v_mov_b64_e32 v[2:3], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	flat_load_dword v1, v[2:3] sc1
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lg_u64 s[6:7], 0
	s_cselect_b64 s[76:77], -1, 0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB5_80
; %bb.77:
	v_and_b32_e32 v3, 63, v0
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB5_79
; %bb.78:
	s_load_dwordx4 s[8:11], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_load_dword v2, v[2:3] sc1
.LBB5_79:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit
	s_or_b64 exec, exec, s[4:5]
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v3, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x6000000, v2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB5_81
	s_branch .LBB5_173
.LBB5_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB5_173
.LBB5_81:                               ; %.critedge498
	s_lshr_b32 s2, s33, 27
	s_add_i32 s2, s57, s2
	s_ashr_i32 s82, s2, 5
	s_mul_i32 s4, s61, s82
	v_readlane_b32 s2, v90, 9
	s_cmp_ge_i32 s2, s4
	v_readlane_b32 s3, v90, 10
	v_writelane_b32 v90, s4, 13
	s_cbranch_scc1 .LBB5_173
; %bb.82:                               ; %.lr.ph553
	s_load_dwordx2 s[80:81], s[0:1], 0x80
	s_load_dwordx4 s[4:7], s[0:1], 0x70
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x20
	s_load_dwordx2 s[10:11], s[0:1], 0x30
	s_load_dwordx2 s[84:85], s[0:1], 0x50
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[12:13], src_shared_base
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s5, 0, 0
	s_cselect_b32 s3, s13, 0
	s_and_b32 s0, s5, 15
	s_and_b32 s7, s5, -16
	s_add_u32 s7, s7, 16
	s_mov_b32 s1, 0
	s_addc_u32 s12, s3, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s57, s5, s7
	s_cselect_b32 s0, s3, s12
	s_add_u32 s3, s57, 0x2000
	s_addc_u32 s0, s0, 0
	s_and_b32 s5, s3, -16
	s_and_b32 s0, s3, 15
	s_add_u32 s5, s5, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s70, s3, s5
	s_add_i32 s0, s59, 63
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_i32 s0, s0, s1
	v_lshrrev_b32_e32 v6, 3, v0
	v_and_b32_e32 v14, 56, v2
	s_ashr_i32 s71, s0, 6
	v_mad_u64_u32 v[2:3], s[0:1], v6, s2, v[14:15]
	v_ashrrev_i32_e32 v3, 31, v2
	s_add_i32 s3, s71, -1
	s_lshl_b32 s83, s61, 2
	s_lshl_b32 s85, s2, 5
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[8:9]
	v_mad_u64_u32 v[2:3], s[0:1], v6, s84, v[14:15]
	v_lshlrev_b32_e32 v15, 4, v0
	v_lshlrev_b32_e32 v7, 7, v6
	s_movk_i32 s0, 0x70
	s_cmp_gt_i32 s59, 0
	v_and_or_b32 v40, v15, s0, v7
	s_cselect_b64 s[90:91], -1, 0
	s_ashr_i32 s1, s14, 31
	s_mov_b32 s0, s14
	s_lshl_b32 s2, s3, 6
	s_lshl_b64 s[0:1], s[0:1], 2
	v_ashrrev_i32_e32 v3, 31, v2
	v_or_b32_e32 v41, 8, v40
	s_add_u32 s0, s66, s0
	v_lshl_add_u64 v[18:19], v[2:3], 1, s[10:11]
	v_add_u32_e32 v2, s57, v41
	s_addc_u32 s1, s67, s1
	v_lshrrev_b32_e32 v3, 4, v2
	v_writelane_b32 v90, s0, 14
	v_and_b32_e32 v3, 0x78, v3
	s_min_i32 s73, s62, 32
	v_writelane_b32 v90, s1, 15
	s_mul_i32 s0, s6, s4
	s_bfe_i64 s[88:89], s[80:81], 0x200000
	v_xor_b32_e32 v42, v3, v2
	v_add_u32_e32 v2, s57, v40
	s_mul_i32 s0, s0, s56
	s_cmp_gt_i32 s62, 0
	v_lshrrev_b32_e32 v3, 4, v2
	v_writelane_b32 v90, s0, 16
	s_cselect_b64 s[0:1], -1, 0
	v_and_b32_e32 v3, 0x78, v3
	v_writelane_b32 v90, s0, 18
	v_xor_b32_e32 v43, v3, v2
	v_add_u32_e32 v2, s70, v15
	v_writelane_b32 v90, s1, 19
	v_lshrrev_b32_e32 v3, 4, v2
	v_readlane_b32 s4, v90, 2
	v_and_b32_e32 v3, 0x78, v3
	v_or_b32_e32 v45, 8, v15
	v_readlane_b32 s5, v90, 3
	v_xor_b32_e32 v44, v3, v2
	v_add_u32_e32 v2, s70, v45
	s_cmp_lg_u64 s[4:5], 0
	v_lshrrev_b32_e32 v3, 4, v2
	s_cselect_b64 s[92:93], -1, 0
	s_max_i32 s0, s58, 1
	v_and_b32_e32 v3, 0x78, v3
	s_add_i32 s0, s0, -1
	v_xor_b32_e32 v46, v3, v2
	v_lshrrev_b32_e32 v3, 2, v0
	v_readlane_b32 s6, v90, 4
	s_cmp_lg_u32 s15, 0
	v_and_b32_e32 v8, 12, v3
	v_readlane_b32 s7, v90, 5
	v_writelane_b32 v90, s0, 20
	s_cselect_b64 s[94:95], -1, 0
	s_lshl_b32 s0, s73, 3
	s_sub_i32 s6, s59, s2
	v_bfe_u32 v4, v0, 6, 2
	v_and_b32_e32 v2, 15, v0
	v_cmp_gt_i32_e64 s[4:5], s0, v0
	v_cmp_gt_i32_e64 s[0:1], s6, v8
	s_abs_i32 s8, s83
	v_lshlrev_b32_e32 v3, 7, v2
	v_or_b32_e32 v10, 1, v8
	v_lshl_or_b32 v51, v4, 4, v2
	v_writelane_b32 v90, s0, 21
	v_cvt_f32_u32_e32 v2, s8
	v_lshrrev_b32_e32 v5, 8, v0
	v_writelane_b32 v90, s1, 22
	v_cmp_gt_i32_e64 s[0:1], s6, v10
	v_lshl_or_b32 v47, v5, 11, v3
	v_lshl_or_b32 v48, v4, 11, v3
	v_writelane_b32 v90, s0, 23
	v_rcp_iflag_f32_e32 v3, v2
	s_max_i32 s81, s71, 1
	v_writelane_b32 v90, s1, 24
	s_or_b32 s0, s2, 32
	s_sub_i32 s7, s59, s0
	s_abs_i32 s59, s62
	v_cvt_f32_u32_e32 v4, s59
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v3, v3
	v_mad_i64_i32 v[20:21], s[0:1], s80, v6, 0
	v_rcp_iflag_f32_e32 v4, v4
	s_bfe_i32 s0, s61, 0x1001d
	v_readfirstlane_b32 s1, v3
	v_writelane_b32 v90, s0, 25
	v_mul_f32_e32 v3, 0x4f7ffffe, v4
	s_sub_i32 s0, 0, s8
	v_cvt_u32_f32_e32 v3, v3
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	v_writelane_b32 v90, s8, 26
	s_add_i32 s0, s1, s0
	v_writelane_b32 v90, s0, 27
	s_sub_i32 s0, 0, s59
	v_readfirstlane_b32 s1, v3
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s46, s1, s0
	s_lshr_b32 s0, s46, 27
	s_mul_i32 s1, s0, s59
	s_sub_i32 s1, 32, s1
	s_ashr_i32 s9, s62, 31
	s_add_i32 s2, s0, 1
	s_sub_i32 s8, s1, s59
	s_cmp_ge_u32 s1, s59
	s_cselect_b32 s0, s2, s0
	s_cselect_b32 s1, s8, s1
	s_add_i32 s2, s0, 1
	s_cmp_ge_u32 s1, s59
	s_cselect_b32 s0, s2, s0
	s_abs_i32 s47, s72
	v_cvt_f32_u32_e32 v4, s47
	v_mov_b32_e32 v23, 0
	v_add_u32_e32 v2, 0, v7
	v_mov_b32_e32 v3, s13
	v_lshlrev_b32_e32 v22, 1, v14
	v_lshl_add_u64 v[24:25], v[2:3], 0, v[22:23]
	v_rcp_iflag_f32_e32 v3, v4
	v_add_u32_e32 v25, v2, v22
	s_xor_b32 s0, s0, s9
	s_sub_i32 s75, s0, s9
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s47
	s_ashr_i32 s2, s72, 31
	v_writelane_b32 v90, s9, 28
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s96, s1, s0
	s_cmp_gt_i32 s75, 0
	s_mul_i32 s0, s89, s73
	s_mul_hi_u32 s1, s80, s73
	s_cselect_b64 s[8:9], -1, 0
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s80, s73
	s_lshl_b64 s[86:87], s[0:1], 1
	s_sub_i32 s0, 0, s72
	v_writelane_b32 v90, s0, 30
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_and_b32_e32 v32, 63, v0
	v_and_b32_e32 v2, 7, v0
	v_writelane_b32 v90, s0, 32
	v_add3_u32 v62, 0, v22, v7
	v_lshlrev_b32_e32 v22, 4, v2
	v_writelane_b32 v90, s1, 33
	v_cmp_eq_u32_e64 s[0:1], 0, v32
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v90, s0, 34
	v_or_b32_e32 v9, 16, v8
	v_or_b32_e32 v11, 2, v8
	v_writelane_b32 v90, s1, 35
	v_cmp_gt_u32_e64 s[0:1], s75, v0
	v_or_b32_e32 v12, 3, v8
	v_or_b32_e32 v13, 17, v8
	v_writelane_b32 v90, s0, 36
	v_or_b32_e32 v30, 18, v8
	v_or_b32_e32 v31, 19, v8
	v_writelane_b32 v90, s1, 37
	v_lshl_or_b32 v52, v5, 4, v8
	v_readlane_b32 s0, v90, 11
	v_readlane_b32 s1, v90, 12
	v_writelane_b32 v90, s8, 38
	s_and_b64 s[0:1], s[0:1], s[8:9]
	v_lshlrev_b32_e32 v57, 1, v8
	v_writelane_b32 v90, s9, 39
	v_writelane_b32 v90, s0, 40
	s_mov_b64 s[98:99], 0x80
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v90, s1, 41
	v_writelane_b32 v90, s76, 42
	v_mul_lo_u32 v49, s62, v0
	v_add_u32_e32 v50, -1, v1
	v_writelane_b32 v90, s77, 43
	v_writelane_b32 v90, s82, 44
	v_writelane_b32 v90, s84, 45
	v_lshl_add_u32 v53, v51, 1, 0
	v_or_b32_e32 v54, 1, v52
	v_writelane_b32 v90, s85, 46
	v_writelane_b32 v90, s83, 47
	v_writelane_b32 v90, s85, 48
	v_or_b32_e32 v55, 2, v52
	v_or_b32_e32 v56, 3, v52
	v_lshlrev_b32_e32 v58, 1, v9
	v_cmp_gt_i32_e64 s[10:11], s6, v11
	v_or_b32_e32 v59, 64, v57
	v_or_b32_e32 v60, 0x60, v57
	v_add_u32_e32 v61, 8, v14
	v_lshl_add_u64 v[26:27], v[18:19], 0, s[98:99]
	v_lshl_add_u32 v63, v0, 1, 0
	v_or_b32_e32 v64, 1, v14
	v_lshl_add_u64 v[28:29], v[20:21], 1, v[22:23]
	v_bfrev_b32_e32 v65, 64
	v_and_b32_e32 v66, 0x100, v2
	v_cmp_gt_i32_e64 s[12:13], s6, v12
	v_cmp_gt_i32_e64 s[14:15], s6, v9
	v_cmp_gt_i32_e64 s[16:17], s6, v13
	v_cmp_gt_i32_e64 s[18:19], s6, v30
	v_cmp_gt_i32_e64 s[20:21], s6, v31
	v_cmp_gt_i32_e64 s[22:23], s7, v8
	v_cmp_gt_i32_e64 s[24:25], s7, v10
	v_cmp_gt_i32_e64 s[26:27], s7, v11
	v_cmp_gt_i32_e64 s[28:29], s7, v12
	v_cmp_gt_i32_e64 s[30:31], s7, v9
	v_cmp_gt_i32_e64 s[34:35], s7, v13
	v_cmp_gt_i32_e64 s[36:37], s7, v30
	v_cmp_gt_i32_e64 s[38:39], s7, v31
	s_movk_i32 s74, 0x7fff
	s_mov_b64 s[44:45], 0
	v_cmp_gt_u32_e64 s[40:41], 32, v6
	v_writelane_b32 v90, s90, 49
                                        ; implicit-def: $vgpr4_vgpr5
	s_nop 1
	v_writelane_b32 v90, s91, 50
	s_branch .LBB5_85
.LBB5_83:                               ; %Flow847
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_readlane_b32 s0, v90, 9
	s_add_i32 s6, s0, s63
	v_readlane_b32 s1, v90, 10
	s_mov_b32 s0, s6
	v_writelane_b32 v90, s0, 9
	s_nop 1
	v_writelane_b32 v90, s1, 10
	s_nop 0
	v_readlane_b32 s0, v90, 13
	s_cmp_ge_i32 s6, s0
	s_cselect_b64 s[0:1], -1, 0
	v_readlane_b32 s44, v90, 51
	s_orn2_b64 s[0:1], s[0:1], exec
	v_readlane_b32 s45, v90, 52
.LBB5_84:                               ; %Flow880
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s6, v90, 55
	v_readlane_b32 s7, v90, 56
	s_or_b64 exec, exec, s[6:7]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[44:45], s[0:1], s[44:45]
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execz .LBB5_173
.LBB5_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_91 Depth 2
                                        ;     Child Loop BB5_111 Depth 2
                                        ;     Child Loop BB5_123 Depth 2
                                        ;       Child Loop BB5_127 Depth 3
                                        ;         Child Loop BB5_163 Depth 4
                                        ;         Child Loop BB5_148 Depth 4
                                        ;         Child Loop BB5_157 Depth 4
                                        ;     Child Loop BB5_169 Depth 2
	v_writelane_b32 v90, s44, 51
	s_nop 1
	v_writelane_b32 v90, s45, 52
	s_nop 0
	v_readlane_b32 s0, v90, 9
	v_readlane_b32 s1, v90, 10
	s_mov_b32 s42, s0
	s_ashr_i32 s0, s0, 31
	v_readlane_b32 s1, v90, 25
	s_xor_b32 s97, s0, s1
	s_abs_i32 s0, s42
	v_readlane_b32 s1, v90, 27
	s_mul_hi_u32 s1, s0, s1
	v_readlane_b32 s8, v90, 26
	s_mul_i32 s6, s1, s8
	s_sub_i32 s0, s0, s6
	s_add_i32 s6, s1, 1
	s_sub_i32 s7, s0, s8
	s_cmp_ge_u32 s0, s8
	s_cselect_b32 s1, s6, s1
	s_cselect_b32 s0, s7, s0
	s_add_i32 s6, s1, 1
	s_cmp_ge_u32 s0, s8
	s_cselect_b32 s0, s6, s1
	s_xor_b32 s52, s0, s97
	s_sub_i32 s0, s52, s97
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s6, s82, s1
	s_min_i32 s8, s6, 4
	s_abs_i32 s7, s8
	v_cvt_f32_u32_e32 v6, s7
	s_mul_i32 s53, s0, s83
	s_sub_i32 s0, s42, s53
	s_sub_i32 s33, 0, s7
	v_rcp_iflag_f32_e32 v6, v6
	s_abs_i32 s9, s0
	s_xor_b32 s6, s0, s8
	s_ashr_i32 s6, s6, 31
	v_mul_f32_e32 v6, 0x4f7ffffe, v6
	v_cvt_u32_f32_e32 v6, v6
	s_nop 0
	v_readfirstlane_b32 s42, v6
	s_mul_i32 s33, s33, s42
	s_mul_hi_u32 s33, s42, s33
	s_add_i32 s42, s42, s33
	s_mul_hi_u32 s33, s9, s42
	s_mul_i32 s42, s33, s7
	s_sub_i32 s9, s9, s42
	s_add_i32 s42, s33, 1
	s_sub_i32 s43, s9, s7
	s_cmp_ge_u32 s9, s7
	s_cselect_b32 s33, s42, s33
	s_cselect_b32 s9, s43, s9
	s_add_i32 s42, s33, 1
	s_cmp_ge_u32 s9, s7
	s_cselect_b32 s7, s42, s33
	s_xor_b32 s7, s7, s6
	s_sub_i32 s78, s7, s6
	s_mul_i32 s8, s78, s8
	v_writelane_b32 v90, s8, 53
	s_sub_i32 s8, s0, s8
	s_add_i32 s8, s8, s1
	s_and_saveexec_b64 s[0:1], s[40:41]
	s_cbranch_execz .LBB5_87
; %bb.86:                               ;   in Loop: Header=BB5_85 Depth=1
	s_mul_i32 s42, s85, s8
	s_ashr_i32 s43, s42, 31
	v_lshl_add_u64 v[2:3], s[42:43], 1, v[16:17]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB5_87:                               ; %_ZN17hk_gemm_rs_mi300x10load_issueITkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEELi512ELi2ELb0ETkNS2_2gl3allENS1_2glIS5_Lin1ELin1ELin1ELin1EJEEETkNS2_5coord4tileENS1_5coordIS8_EEEEvP15HIP_vector_typeIfLj4EERKT3_RKT4_.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s79, s78, 6
	s_mul_i32 s0, s79, s84
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[6:7], s[0:1], 1, v[18:19]
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], s[40:41]
	s_cbranch_execz .LBB5_89
; %bb.88:                               ;   in Loop: Header=BB5_85 Depth=1
	;;#ASMSTART
	ds_write_b64 v43, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v42, v[4:5]

	;;#ASMEND
.LBB5_89:                               ; %_ZN17hk_gemm_rs_mi300x11load_commitILi512ETkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEEEEvRT0_PK15HIP_vector_typeIfLj4EE.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v44, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v46, v[12:13]

	;;#ASMEND
	s_andn2_b64 vcc, exec, s[90:91]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB5_105
; %bb.90:                               ; %.lr.ph536.preheader
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_lshl_add_u64 v[30:31], s[0:1], 1, v[26:27]
	s_lshl_b32 s0, s52, 2
	v_readlane_b32 s42, v90, 9
	s_add_i32 s0, s42, s0
	s_sub_i32 s0, s0, s53
	v_readlane_b32 s1, v90, 53
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s1, s97, 2
	s_sub_i32 s0, s0, s1
	s_mul_i32 s0, s85, s0
	v_mov_b32_e32 v6, 0
	s_add_i32 s0, s0, 64
	s_mov_b32 s33, 0
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v9, v6
	v_readlane_b32 s43, v90, 10
.LBB5_91:                               ; %.lr.ph536
                                        ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s9, s33, 1
	s_cmp_lt_i32 s9, s71
	s_cselect_b64 s[48:49], -1, 0
	s_cmp_ge_i32 s9, s71
	s_cbranch_scc1 .LBB5_95
; %bb.92:                               ;   in Loop: Header=BB5_91 Depth=2
	s_and_saveexec_b64 s[44:45], s[40:41]
	s_cbranch_execz .LBB5_94
; %bb.93:                               ;   in Loop: Header=BB5_91 Depth=2
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[16:17]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB5_94:                               ; %_ZN17hk_gemm_rs_mi300x10load_issueITkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEELi512ELi2ELb0ETkNS2_2gl3allENS1_2glIS5_Lin1ELin1ELin1ELin1EJEEETkNS2_5coord4tileENS1_5coordIS8_EEEEvP15HIP_vector_typeIfLj4EERKT3_RKT4_.exit297
                                        ;   in Loop: Header=BB5_91 Depth=2
	s_or_b64 exec, exec, s[44:45]
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[30:31], off

	;;#ASMEND
.LBB5_95:                               ;   in Loop: Header=BB5_91 Depth=2
	s_and_b32 s1, s33, 1
	s_lshl_b32 s42, s1, 12
	s_add_i32 s42, s57, s42
	v_add_u32_e32 v67, s42, v47
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s70, s1
	v_add_u32_e32 v32, v57, v67
	v_add_u32_e32 v22, s1, v48
	v_lshrrev_b32_e32 v33, 4, v32
	v_add_u32_e32 v34, v58, v67
	v_and_b32_e32 v33, 0x78, v33
	v_lshrrev_b32_e32 v35, 4, v34
	v_add_u32_e32 v36, v57, v22
	v_xor_b32_e32 v32, v33, v32
	v_and_b32_e32 v35, 0x78, v35
	v_lshrrev_b32_e32 v37, 4, v36
	v_add_u32_e32 v38, v58, v22
	;;#ASMSTART
	ds_read_b64 v[32:33], v32 offset:0

	;;#ASMEND
	v_xor_b32_e32 v34, v35, v34
	v_and_b32_e32 v37, 0x78, v37
	v_lshrrev_b32_e32 v39, 4, v38
	;;#ASMSTART
	ds_read_b64 v[34:35], v34 offset:0

	;;#ASMEND
	v_xor_b32_e32 v36, v37, v36
	v_and_b32_e32 v39, 0x78, v39
	s_cmp_eq_u32 s3, s33
	;;#ASMSTART
	ds_read_b64 v[36:37], v36 offset:0

	;;#ASMEND
	v_xor_b32_e32 v38, v39, v38
	s_cselect_b64 s[50:51], -1, 0
	s_cmp_lg_u32 s3, s33
	;;#ASMSTART
	ds_read_b64 v[38:39], v38 offset:0

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB5_97
; %bb.96:                               ; %.preheader.1.i
                                        ;   in Loop: Header=BB5_91 Depth=2
	v_readlane_b32 s42, v90, 21
	v_readlane_b32 s43, v90, 22
	v_and_b32_e32 v69, 0xffff0000, v33
	v_readlane_b32 s44, v90, 23
	v_cndmask_b32_e64 v68, 0, v32, s[42:43]
	v_cndmask_b32_e64 v69, v69, v33, s[10:11]
	s_or_b64 s[42:43], s[12:13], s[10:11]
	v_readlane_b32 s45, v90, 24
	v_and_b32_e32 v69, 0xffff, v69
	s_or_b64 vcc, s[42:43], s[44:45]
	s_or_b64 s[42:43], s[20:21], s[18:19]
	v_cndmask_b32_sdwa v32, v68, v32, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v33, v69, v33, s[12:13]
	v_cndmask_b32_e64 v68, 0, v34, s[14:15]
	v_and_b32_e32 v69, 0xffff0000, v35
	s_or_b64 vcc, s[42:43], s[16:17]
	v_cndmask_b32_e64 v69, v69, v35, s[18:19]
	v_cndmask_b32_sdwa v34, v68, v34, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[20:21]
	v_cndmask_b32_sdwa v35, v69, v35, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB5_97:                               ;   in Loop: Header=BB5_91 Depth=2
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[36:37], v[6:9]
	v_add_u32_e32 v68, v59, v67
	v_lshrrev_b32_e32 v69, 4, v68
	v_and_b32_e32 v32, 0x78, v69
	v_mfma_f32_16x16x16_bf16 v[6:9], v[34:35], v[38:39], v[6:9]
	v_add_u32_e32 v34, v60, v67
	v_lshrrev_b32_e32 v35, 4, v34
	v_add_u32_e32 v36, v59, v22
	v_xor_b32_e32 v32, v32, v68
	v_and_b32_e32 v35, 0x78, v35
	v_lshrrev_b32_e32 v37, 4, v36
	;;#ASMSTART
	ds_read_b64 v[32:33], v32 offset:0

	;;#ASMEND
	v_xor_b32_e32 v34, v35, v34
	v_and_b32_e32 v37, 0x78, v37
	;;#ASMSTART
	ds_read_b64 v[34:35], v34 offset:0

	;;#ASMEND
	v_xor_b32_e32 v36, v37, v36
	v_add_u32_e32 v22, v60, v22
	;;#ASMSTART
	ds_read_b64 v[38:39], v36 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v36, 4, v22
	v_and_b32_e32 v36, 0x78, v36
	v_xor_b32_e32 v22, v36, v22
	;;#ASMSTART
	ds_read_b64 v[36:37], v22 offset:0

	;;#ASMEND
	s_andn2_b64 vcc, exec, s[50:51]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_vccnz .LBB5_99
; %bb.98:                               ; %.preheader.1.i.1
                                        ;   in Loop: Header=BB5_91 Depth=2
	v_and_b32_e32 v67, 0xffff0000, v33
	v_cndmask_b32_e64 v67, v67, v33, s[26:27]
	s_or_b64 s[42:43], s[28:29], s[26:27]
	v_cndmask_b32_e64 v22, 0, v32, s[22:23]
	v_and_b32_e32 v67, 0xffff, v67
	s_or_b64 vcc, s[42:43], s[24:25]
	s_or_b64 s[42:43], s[38:39], s[36:37]
	v_cndmask_b32_sdwa v32, v22, v32, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v33, v67, v33, s[28:29]
	v_cndmask_b32_e64 v22, 0, v34, s[30:31]
	v_and_b32_e32 v67, 0xffff0000, v35
	s_or_b64 vcc, s[42:43], s[34:35]
	v_cndmask_b32_e64 v67, v67, v35, s[36:37]
	v_cndmask_b32_sdwa v34, v22, v34, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[38:39]
	v_cndmask_b32_sdwa v35, v67, v35, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB5_99:                               ;   in Loop: Header=BB5_91 Depth=2
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[38:39], v[6:9]
	s_andn2_b64 vcc, exec, s[48:49]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[34:35], v[36:37], v[6:9]
	s_cbranch_vccnz .LBB5_103
; %bb.100:                              ;   in Loop: Header=BB5_91 Depth=2
	s_and_b32 s1, s9, 1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], s[40:41]
	s_cbranch_execz .LBB5_102
; %bb.101:                              ;   in Loop: Header=BB5_91 Depth=2
	s_lshl_b32 s33, s1, 12
	s_add_i32 s33, s57, s33
	v_add_u32_e32 v22, s33, v41
	v_lshrrev_b32_e32 v32, 4, v22
	v_and_b32_e32 v32, 0x78, v32
	v_xor_b32_e32 v22, v32, v22
	v_add_u32_e32 v32, s33, v40
	v_lshrrev_b32_e32 v33, 4, v32
	v_and_b32_e32 v33, 0x78, v33
	v_xor_b32_e32 v32, v33, v32
	;;#ASMSTART
	ds_write_b64 v32, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v22, v[4:5]

	;;#ASMEND
.LBB5_102:                              ; %_ZN17hk_gemm_rs_mi300x11load_commitILi512ETkN7kittens5ducks2st3allENS1_2stI14__hip_bfloat16Li32ELi64ENS2_9st_layout3rowEEEEEvRT0_PK15HIP_vector_typeIfLj4EE.exit311
                                        ;   in Loop: Header=BB5_91 Depth=2
	s_or_b64 exec, exec, s[44:45]
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s70, s1
	v_add_u32_e32 v22, s1, v15
	v_lshrrev_b32_e32 v32, 4, v22
	v_and_b32_e32 v32, 0x78, v32
	v_xor_b32_e32 v22, v32, v22
	v_add_u32_e32 v32, s1, v45
	v_lshrrev_b32_e32 v33, 4, v32
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v33, 0x78, v33
	;;#ASMSTART
	ds_write_b64 v22, v[10:11]

	;;#ASMEND
	v_xor_b32_e32 v32, v33, v32
	;;#ASMSTART
	ds_write_b64 v32, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB5_103:                              ;   in Loop: Header=BB5_91 Depth=2
	s_add_i32 s0, s0, 64
	s_cmp_eq_u32 s81, s9
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[98:99]
	s_barrier
	s_cbranch_scc1 .LBB5_106
; %bb.104:                              ;   in Loop: Header=BB5_91 Depth=2
	s_mov_b32 s33, s9
	s_branch .LBB5_91
.LBB5_105:                              ;   in Loop: Header=BB5_85 Depth=1
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, v9
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v6, v9
.LBB5_106:                              ; %Flow876
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s42, v90, 36
	v_readlane_b32 s43, v90, 37
	s_and_b64 s[42:43], s[0:1], s[42:43]
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_115
; %bb.107:                              ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s42, v90, 32
	v_readlane_b32 s43, v90, 33
	s_and_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB5_115
; %bb.108:                              ;   in Loop: Header=BB5_85 Depth=1
	v_lshl_add_u32 v10, s8, 5, v49
	v_sub_u32_e32 v12, 0, v10
	v_max_i32_e32 v12, v10, v12
	v_mul_hi_u32 v13, v12, s96
	v_mul_lo_u32 v22, v13, s47
	v_sub_u32_e32 v12, v12, v22
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s47, v12
	v_ashrrev_i32_e32 v11, 31, v10
	v_xor_b32_e32 v11, s2, v11
	v_cndmask_b32_e32 v13, v13, v22, vcc
	v_subrev_u32_e32 v22, s47, v12
	v_cndmask_b32_e32 v12, v12, v22, vcc
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s47, v12
	v_readlane_b32 s9, v90, 28
	s_nop 0
	v_cndmask_b32_e32 v12, v13, v22, vcc
	v_xor_b32_e32 v12, v12, v11
	v_sub_u32_e32 v11, v12, v11
	v_mul_lo_u32 v12, v11, s72
	v_sub_u32_e32 v10, v10, v12
	v_sub_u32_e32 v13, 0, v10
	v_ashrrev_i32_e32 v12, 31, v10
	v_max_i32_e32 v10, v10, v13
	v_mul_hi_u32 v13, v10, s46
	v_mul_lo_u32 v22, v13, s59
	v_sub_u32_e32 v10, v10, v22
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s59, v10
	v_xor_b32_e32 v12, s9, v12
	s_nop 0
	v_cndmask_b32_e32 v13, v13, v22, vcc
	v_subrev_u32_e32 v22, s59, v10
	v_cndmask_b32_e32 v10, v10, v22, vcc
	v_add_u32_e32 v22, 1, v13
	v_cmp_le_u32_e32 vcc, s59, v10
	s_nop 1
	v_cndmask_b32_e32 v10, v13, v22, vcc
	v_xor_b32_e32 v10, v10, v12
	v_sub_u32_e32 v10, v10, v12
	v_mad_u64_u32 v[10:11], s[42:43], v11, s60, v[10:11]
	v_mul_lo_u32 v10, v10, s61
	v_add_u32_e32 v10, s78, v10
	v_readlane_b32 s42, v90, 14
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s43, v90, 15
	s_nop 1
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[42:43]
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v12, v50
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB5_115
; %bb.109:                              ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b32 s33, s53
	s_mov_b32 s9, s52
	s_mov_b64 s[44:45], 0
	s_mov_b64 s[52:53], 0
                                        ; implicit-def: $sgpr48_sgpr49
                                        ; implicit-def: $sgpr50_sgpr51
	s_branch .LBB5_111
.LBB5_110:                              ; %Flow869
                                        ;   in Loop: Header=BB5_111 Depth=2
	s_and_b64 s[42:43], exec, s[50:51]
	s_or_b64 s[44:45], s[42:43], s[44:45]
	s_andn2_b64 s[42:43], s[48:49], exec
	s_and_b64 s[48:49], s[54:55], exec
	s_or_b64 s[48:49], s[42:43], s[48:49]
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execz .LBB5_113
.LBB5_111:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_readlane_b32 s42, v90, 6
	s_add_u32 s52, s52, 1
	v_readlane_b32 s43, v90, 7
	s_addc_u32 s53, s53, 0
	s_mov_b64 s[54:55], -1
	v_mov_b64_e32 v[12:13], s[42:43]
	v_cmp_gt_u64_e32 vcc, s[52:53], v[12:13]
	s_or_b64 s[50:51], s[50:51], exec
	s_cbranch_vccnz .LBB5_110
; %bb.112:                              ;   in Loop: Header=BB5_111 Depth=2
	s_sleep 4
	flat_load_dword v12, v[10:11] offset:256 sc0 sc1
	s_andn2_b64 s[42:43], s[50:51], exec
	s_mov_b64 s[54:55], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v12, v50
	s_and_b64 s[50:51], vcc, exec
	s_or_b64 s[50:51], s[42:43], s[50:51]
	s_branch .LBB5_110
.LBB5_113:                              ; %loop.exit.guard819
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[42:43], s[48:49]
	s_mov_b32 s52, s9
	s_mov_b32 s53, s33
	s_xor_b64 s[42:43], exec, s[42:43]
	s_cbranch_execz .LBB5_115
; %bb.114:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s48, v90, 2
	v_readlane_b32 s50, v90, 4
	v_readlane_b32 s51, v90, 5
	v_readlane_b32 s49, v90, 3
	s_nop 0
	v_mov_b64_e32 v[10:11], s[50:51]
	flat_atomic_or v[10:11], v65
.LBB5_115:                              ; %.critedge499
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[76:77]
	s_mov_b64 s[44:45], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB5_119
; %bb.116:                              ;   in Loop: Header=BB5_85 Depth=1
	v_mov_b32_e32 v10, 0
	s_mov_b64 s[44:45], exec
	v_readlane_b32 s42, v90, 34
	v_readlane_b32 s43, v90, 35
	s_and_b64 s[42:43], s[44:45], s[42:43]
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_118
; %bb.117:                              ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s48, v90, 2
	v_readlane_b32 s50, v90, 4
	v_readlane_b32 s51, v90, 5
	v_readlane_b32 s49, v90, 3
	s_nop 0
	v_mov_b64_e32 v[10:11], s[50:51]
	flat_load_dword v10, v[10:11] sc1
.LBB5_118:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit314
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v10, v66, v10
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v10, 0x2000000, v10
	v_cmp_eq_u32_e64 s[44:45], 0, v10
.LBB5_119:                              ; %Flow879
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b64 s[42:43], exec
	v_writelane_b32 v90, s42, 55
	s_nop 1
	v_writelane_b32 v90, s43, 56
	s_and_b64 s[42:43], s[42:43], s[44:45]
	v_readlane_b32 s44, v90, 51
	v_readlane_b32 s45, v90, 52
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_84
; %bb.120:                              ; %.critedge513
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_writelane_b32 v90, s53, 57
	v_writelane_b32 v90, s52, 58
	v_writelane_b32 v90, s97, 59
	s_nop 0
	v_readlane_b32 s0, v90, 38
	v_readlane_b32 s1, v90, 39
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB5_164
; %bb.121:                              ; %.lr.ph546
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_sub_i32 s9, s58, s79
	s_min_i32 s72, s9, 64
	s_abs_i32 s33, s72
	v_cvt_f32_u32_e32 v12, s33
	s_lshl_b32 s43, s8, 5
	s_ashr_i32 s97, s72, 31
	s_sub_i32 s8, 0, s33
	v_rcp_iflag_f32_e32 v12, v12
	s_lshl_b32 s7, s7, 6
	s_lshl_b32 s6, s6, 6
	s_sub_i32 s91, s7, s6
	v_mul_f32_e32 v12, 0x4f7ffffe, v12
	v_cvt_u32_f32_e32 v12, v12
	v_readlane_b32 s6, v90, 58
	v_cmp_gt_i32_e64 s[52:53], s9, v14
	s_lshl_b32 s6, s6, 2
	v_mul_lo_u32 v13, s8, v12
	s_lshl_b32 s8, s97, 7
	v_subrev_u32_e32 v37, s8, v63
	v_readlane_b32 s8, v90, 9
	s_add_i32 s6, s8, s6
	v_readlane_b32 s7, v90, 57
	s_sub_i32 s6, s6, s7
	v_readlane_b32 s7, v90, 53
	v_or_b32_e32 v10, s79, v51
	v_readlane_b32 s0, v90, 20
	s_sub_i32 s6, s6, s7
	v_readlane_b32 s7, v90, 59
	v_mov_b32_e32 v11, s0
	v_cmp_gt_i32_e32 vcc, s58, v10
	s_lshl_b32 s7, s7, 2
	v_readlane_b32 s48, v90, 2
	v_cndmask_b32_e32 v10, v11, v10, vcc
	s_sub_i32 s6, s6, s7
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s49, v90, 3
	v_readlane_b32 s50, v90, 4
	v_readlane_b32 s51, v90, 5
	s_mul_i32 s42, s72, s73
	v_mul_hi_u32 v13, v12, v13
	s_lshl_b32 s6, s6, 5
	v_readlane_b32 s7, v90, 16
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[48:49]
	v_cmp_gt_i32_e64 s[0:1], s42, v0
	v_cmp_lt_i32_e64 s[48:49], s72, v61
	v_cmp_ge_i32_e64 s[50:51], s72, v61
	s_mov_b32 s76, 0
	v_add_u32_e32 v36, v12, v13
	s_lshl_b32 s77, s72, 1
	s_sub_i32 s90, 0, s72
	s_add_i32 s6, s7, s6
	v_readlane_b32 s9, v90, 10
	s_branch .LBB5_123
.LBB5_122:                              ; %._crit_edge544
                                        ;   in Loop: Header=BB5_123 Depth=2
	s_add_i32 s76, s76, 1
	s_add_i32 s6, s6, s62
	s_cmp_eq_u32 s76, s75
	s_cbranch_scc1 .LBB5_164
.LBB5_123:                              ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_127 Depth 3
                                        ;         Child Loop BB5_163 Depth 4
                                        ;         Child Loop BB5_148 Depth 4
                                        ;         Child Loop BB5_157 Depth 4
	v_readlane_b32 s8, v90, 18
	v_readlane_b32 s9, v90, 19
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB5_122
; %bb.124:                              ; %.lr.ph543.preheader
                                        ;   in Loop: Header=BB5_123 Depth=2
	v_mov_b64_e32 v[12:13], s[64:65]
	flat_load_dwordx4 v[30:33], v[12:13]
	flat_load_dwordx4 v[68:71], v[12:13] offset:16
	flat_load_dwordx4 v[72:75], v[12:13] offset:32
	flat_load_dwordx4 v[76:79], v[12:13] offset:48
	s_nop 0
	flat_load_dwordx2 v[12:13], v[12:13] offset:64
	s_mul_i32 s7, s76, s62
	s_add_i32 s9, s7, s43
	s_abs_i32 s45, s9
	s_mul_hi_u32 s54, s45, s96
	s_mul_i32 s55, s54, s47
	s_ashr_i32 s44, s9, 31
	s_sub_i32 s45, s45, s55
	s_xor_b32 s44, s44, s2
	s_add_i32 s82, s54, 1
	s_sub_i32 s55, s45, s47
	s_cmp_ge_u32 s45, s47
	s_cselect_b32 s54, s82, s54
	s_cselect_b32 s45, s55, s45
	s_add_i32 s55, s54, 1
	s_cmp_ge_u32 s45, s47
	s_cselect_b32 s45, s55, s54
	s_xor_b32 s45, s45, s44
	s_sub_i32 s44, s45, s44
	v_readlane_b32 s45, v90, 8
	s_mul_i32 s45, s44, s45
	s_sub_i32 s9, s9, s45
	v_readlane_b32 s84, v90, 0
	s_cmp_eq_u32 s44, 0
	v_readlane_b32 s85, v90, 1
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 1
	v_mov_b32_e32 v22, s85
	s_mov_b32 s8, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_cndmask_b32_e32 v33, 0, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 2
	v_sub_co_u32_e64 v30, s[54:55], s84, v30
	v_cndmask_b32_e32 v32, v32, v68, vcc
	s_nop 0
	v_subb_co_u32_e64 v31, s[54:55], v22, v31, s[54:55]
	v_cndmask_b32_e32 v22, v33, v69, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 3
	v_cndmask_b32_e32 v32, v32, v70, vcc
	v_cndmask_b32_e32 v22, v22, v71, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 4
	v_cndmask_b32_e32 v22, v22, v73, vcc
	v_cndmask_b32_e32 v32, v32, v72, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 5
	v_cndmask_b32_e32 v32, v32, v74, vcc
	v_cndmask_b32_e32 v22, v22, v75, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 6
	v_cndmask_b32_e32 v22, v22, v77, vcc
	v_cndmask_b32_e32 v32, v32, v76, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 7
	v_cndmask_b32_e32 v32, v32, v78, vcc
	v_cndmask_b32_e32 v22, v22, v79, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s54, s9
	s_mul_hi_u32 s55, s54, s46
	s_mul_i32 s55, s55, s59
	s_sub_i32 s54, s54, s55
	s_ashr_i32 s44, s9, 31
	s_sub_i32 s55, s54, s59
	s_cmp_ge_u32 s54, s59
	s_cselect_b32 s54, s55, s54
	s_sub_i32 s55, s54, s59
	s_cmp_ge_u32 s54, s59
	s_cselect_b32 s54, s55, s54
	v_readlane_b32 s55, v90, 16
	s_add_i32 s9, s9, s55
	s_add_i32 s55, s44, s6
	s_xor_b32 s54, s54, s44
	s_sub_i32 s45, s55, s45
	s_sub_i32 s44, s44, s54
	v_cndmask_b32_e32 v13, v22, v13, vcc
	v_cndmask_b32_e32 v12, v32, v12, vcc
	s_sub_i32 s45, s45, s54
	s_add_i32 s9, s9, s44
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[12:13]
	v_cmp_ne_u64_e32 vcc, 0, v[12:13]
	s_mul_i32 s44, s80, s45
	s_mul_i32 s9, s9, s80
	v_cndmask_b32_e32 v13, 0, v31, vcc
	v_cndmask_b32_e32 v12, 0, v30, vcc
	s_add_i32 s44, s91, s44
	s_add_i32 s54, s9, s79
	v_lshl_add_u64 v[30:31], v[12:13], 0, v[28:29]
	s_ashr_i32 s45, s44, 31
	s_ashr_i32 s55, s54, 31
	v_lshl_add_u64 v[12:13], s[54:55], 1, v[12:13]
	v_lshl_add_u64 v[30:31], s[44:45], 1, v[30:31]
	s_branch .LBB5_127
.LBB5_125:                              ; %Flow861
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[54:55]
.LBB5_126:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_add_i32 s8, s8, s73
	s_cmp_ge_i32 s8, s62
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[86:87]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB5_122
.LBB5_127:                              ; %.lr.ph543
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_123 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB5_163 Depth 4
                                        ;         Child Loop BB5_148 Depth 4
                                        ;         Child Loop BB5_157 Depth 4
	s_andn2_b64 vcc, exec, s[92:93]
	s_cbranch_vccnz .LBB5_129
; %bb.128:                              ;   in Loop: Header=BB5_127 Depth=3
	flat_load_ushort v22, v[10:11]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v22, 16, v22
	s_branch .LBB5_130
.LBB5_129:                              ;   in Loop: Header=BB5_127 Depth=3
	v_mov_b32_e32 v22, 0
.LBB5_130:                              ;   in Loop: Header=BB5_127 Depth=3
	s_add_i32 s9, s8, s7
	s_add_i32 s82, s9, s73
	v_cmp_le_i32_e32 vcc, s9, v52
	v_cmp_gt_i32_e64 s[54:55], s82, v52
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_132
; %bb.131:                              ;   in Loop: Header=BB5_127 Depth=3
	v_add_f32_e32 v32, v22, v6
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s74
	v_subrev_u32_e32 v33, s9, v52
	v_lshl_add_u32 v33, v33, 7, v53
	ds_write_b16_d16_hi v33, v32
.LBB5_132:                              ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s9, v54
	v_cmp_gt_i32_e64 s[54:55], s82, v54
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_134
; %bb.133:                              ;   in Loop: Header=BB5_127 Depth=3
	v_add_f32_e32 v32, v22, v7
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s74
	v_subrev_u32_e32 v33, s9, v54
	v_lshl_add_u32 v33, v33, 7, v53
	ds_write_b16_d16_hi v33, v32
.LBB5_134:                              ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s9, v55
	v_cmp_gt_i32_e64 s[54:55], s82, v55
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_136
; %bb.135:                              ;   in Loop: Header=BB5_127 Depth=3
	v_add_f32_e32 v32, v22, v8
	v_bfe_u32 v33, v32, 16, 1
	v_add3_u32 v32, v32, v33, s74
	v_subrev_u32_e32 v33, s9, v55
	v_lshl_add_u32 v33, v33, 7, v53
	ds_write_b16_d16_hi v33, v32
.LBB5_136:                              ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s9, v56
	v_cmp_gt_i32_e64 s[54:55], s82, v56
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_138
; %bb.137:                              ;   in Loop: Header=BB5_127 Depth=3
	v_add_f32_e32 v22, v22, v9
	v_bfe_u32 v32, v22, 16, 1
	v_add3_u32 v22, v22, v32, s74
	v_subrev_u32_e32 v32, s9, v56
	v_lshl_add_u32 v32, v32, 7, v53
	ds_write_b16_d16_hi v32, v22
.LBB5_138:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi16ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[44:45]
	s_mul_i32 s9, s89, s8
	s_mul_hi_u32 s44, s88, s8
	s_add_i32 s45, s44, s9
	s_mul_i32 s44, s88, s8
	s_andn2_b64 vcc, exec, s[94:95]
	v_lshl_add_u64 v[32:33], s[44:45], 1, v[12:13]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB5_160
; %bb.139:                              ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b64 s[54:55], -1
	s_mov_b64 s[82:83], -1
	s_and_saveexec_b64 s[44:45], s[4:5]
	s_cbranch_execz .LBB5_145
; %bb.140:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b64 vcc, s[48:49]
	s_and_saveexec_b64 s[82:83], s[50:51]
; %bb.141:                              ;   in Loop: Header=BB5_127 Depth=3
	v_lshlrev_b32_e32 v22, 1, v20
	v_lshlrev_b32_e32 v34, 1, v14
	v_add3_u32 v22, v32, v22, v34
	v_or_b32_e32 v22, v24, v22
	v_and_b32_e32 v22, 15, v22
	v_cmp_eq_u32_e32 vcc, 0, v22
	s_andn2_b64 s[84:85], s[48:49], exec
	s_and_b64 vcc, vcc, exec
	s_or_b64 vcc, s[84:85], vcc
; %bb.142:                              ; %Flow857
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[82:83]
	s_mov_b64 s[82:83], 0
	s_and_saveexec_b64 s[84:85], vcc
; %bb.143:                              ; %.critedge.i
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b64 s[82:83], exec
; %bb.144:                              ; %Flow858
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[82:83], s[82:83], exec
.LBB5_145:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cndmask_b32_e64 v22, 0, 1, s[82:83]
	s_nop 0
	v_readfirstlane_b32 s9, v22
	s_bitcmp1_b32 s9, 0
	s_cselect_b64 s[44:45], -1, 0
	s_and_b64 vcc, exec, s[44:45]
	s_cbranch_vccnz .LBB5_150
; %bb.146:                              ;   in Loop: Header=BB5_127 Depth=3
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB5_149
; %bb.147:                              ; %.lr.ph.i320.preheader
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v34, v37
	v_mov_b32_e32 v22, v0
.LBB5_148:                              ; %.lr.ph.i320
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_123 Depth=2
                                        ;       Parent Loop BB5_127 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v35, v22, v36
	v_mul_lo_u32 v38, v35, s33
	v_sub_u32_e32 v38, v22, v38
	v_add_u32_e32 v39, 1, v35
	v_subrev_u32_e32 v67, s33, v38
	v_cmp_le_u32_e32 vcc, s33, v38
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cndmask_b32_e32 v38, v38, v67, vcc
	v_add_u32_e32 v39, 1, v35
	v_cmp_le_u32_e32 vcc, s33, v38
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_xor_b32_e32 v35, s97, v35
	v_subrev_u32_e32 v67, s97, v35
	v_mad_u64_u32 v[38:39], s[82:83], s90, v67, v[22:23]
	v_lshlrev_b32_e32 v35, 7, v35
	v_mul_lo_u32 v39, s77, v67
	v_sub_u32_e32 v35, v35, v39
	v_add_u32_e32 v35, v34, v35
	ds_read_u16 v35, v35
	v_mad_i64_i32 v[68:69], s[82:83], s88, v67, 0
	v_add_u32_e32 v22, 0x200, v22
	v_mov_b32_e32 v39, v23
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[32:33]
	v_cmp_le_i32_e32 vcc, s42, v22
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[68:69]
	v_add_u32_e32 v34, 0x400, v34
	s_or_b64 s[44:45], vcc, s[44:45]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[38:39], v35
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_148
.LBB5_149:                              ; %Flow849
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[54:55]
	s_mov_b64 s[54:55], 0
.LBB5_150:                              ; %Flow855
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_andn2_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB5_159
; %bb.151:                              ;   in Loop: Header=BB5_127 Depth=3
	s_and_saveexec_b64 s[54:55], s[4:5]
	s_cbranch_execz .LBB5_158
; %bb.152:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_and_saveexec_b64 s[44:45], s[50:51]
	s_xor_b64 s[44:45], exec, s[44:45]
	s_cbranch_execz .LBB5_154
; %bb.153:                              ;   in Loop: Header=BB5_127 Depth=3
	ds_read_b128 v[68:71], v25
	v_lshl_add_u64 v[34:35], v[20:21], 1, v[32:33]
	v_lshlrev_b32_e32 v22, 1, v14
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[22:23]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[34:35], v[68:71]
.LBB5_154:                              ; %Flow852
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_andn2_saveexec_b64 s[44:45], s[44:45]
	s_cbranch_execz .LBB5_158
; %bb.155:                              ; %.preheader.i
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_and_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB5_158
; %bb.156:                              ; %.lr.ph.i322
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b32 s9, 0
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v22, v62
	v_mov_b64_e32 v[34:35], v[30:31]
.LBB5_157:                              ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_123 Depth=2
                                        ;       Parent Loop BB5_127 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v38, v22
	s_add_i32 s82, s9, 1
	v_add_u32_e32 v39, s9, v64
	s_cmp_gt_u32 s9, 6
	v_cmp_le_u32_e32 vcc, s72, v39
	s_mov_b32 s9, s82
	s_cselect_b64 s[82:83], -1, 0
	s_or_b64 s[82:83], s[82:83], vcc
	s_and_b64 s[82:83], exec, s[82:83]
	v_add_u32_e32 v22, 2, v22
	s_waitcnt lgkmcnt(0)
	flat_store_short v[34:35], v38
	s_or_b64 s[44:45], s[82:83], s[44:45]
	v_lshl_add_u64 v[34:35], v[34:35], 0, 2
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_157
.LBB5_158:                              ; %Flow854
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_or_b64 exec, exec, s[54:55]
.LBB5_159:                              ; %Flow856
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_branch .LBB5_126
.LBB5_160:                              ;   in Loop: Header=BB5_127 Depth=3
	s_cbranch_execz .LBB5_126
; %bb.161:                              ;   in Loop: Header=BB5_127 Depth=3
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB5_125
; %bb.162:                              ; %.lr.ph.i326.preheader
                                        ;   in Loop: Header=BB5_127 Depth=3
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v34, v37
	v_mov_b32_e32 v22, v0
.LBB5_163:                              ; %.lr.ph.i326
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_123 Depth=2
                                        ;       Parent Loop BB5_127 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v35, v22, v36
	v_mul_lo_u32 v38, v35, s33
	v_sub_u32_e32 v38, v22, v38
	v_add_u32_e32 v39, 1, v35
	v_subrev_u32_e32 v67, s33, v38
	v_cmp_le_u32_e32 vcc, s33, v38
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cndmask_b32_e32 v38, v38, v67, vcc
	v_add_u32_e32 v39, 1, v35
	v_cmp_le_u32_e32 vcc, s33, v38
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_xor_b32_e32 v35, s97, v35
	v_subrev_u32_e32 v67, s97, v35
	v_mad_u64_u32 v[38:39], s[82:83], s90, v67, v[22:23]
	v_lshlrev_b32_e32 v35, 7, v35
	v_mul_lo_u32 v39, s77, v67
	v_sub_u32_e32 v35, v35, v39
	v_add_u32_e32 v35, v34, v35
	ds_read_u16 v35, v35
	v_mad_i64_i32 v[68:69], s[82:83], s88, v67, 0
	v_add_u32_e32 v22, 0x200, v22
	v_mov_b32_e32 v39, v23
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[32:33]
	v_cmp_le_i32_e32 vcc, s42, v22
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[68:69]
	v_add_u32_e32 v34, 0x400, v34
	s_or_b64 s[44:45], vcc, s[44:45]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[38:39], v35
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_163
	s_branch .LBB5_125
.LBB5_164:                              ; %._crit_edge547
                                        ;   in Loop: Header=BB5_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s6, v90, 11
	v_readlane_b32 s7, v90, 12
	s_and_b64 s[6:7], s[0:1], s[6:7]
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB5_166
; %bb.165:                              ;   in Loop: Header=BB5_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB5_166:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[48:49], exec
	v_readlane_b32 s0, v90, 40
	v_readlane_b32 s84, v90, 45
	v_readlane_b32 s1, v90, 41
	v_readlane_b32 s76, v90, 42
	v_readlane_b32 s85, v90, 46
	v_readlane_b32 s90, v90, 49
	s_and_b64 s[0:1], s[48:49], s[0:1]
	v_readlane_b32 s72, v90, 8
	v_readlane_b32 s77, v90, 43
	v_readlane_b32 s82, v90, 44
	v_readlane_b32 s83, v90, 47
	v_readlane_b32 s85, v90, 48
	v_readlane_b32 s91, v90, 50
	v_readlane_b32 s8, v90, 59
	v_readlane_b32 s6, v90, 58
	v_readlane_b32 s9, v90, 57
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB5_83
; %bb.167:                              ; %.lr.ph549
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_lshl_b32 s0, s6, 2
	v_readlane_b32 s6, v90, 9
	s_add_i32 s0, s6, s0
	s_sub_i32 s0, s0, s9
	v_readlane_b32 s1, v90, 53
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s1, s8, 2
	v_readlane_b32 s7, v90, 10
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s6, s0, 5
	s_mov_b32 s7, s75
	s_branch .LBB5_169
.LBB5_168:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB5_169 Depth=2
	s_add_i32 s7, s7, -1
	s_add_i32 s6, s6, s62
	s_cmp_lg_u32 s7, 0
	s_cbranch_scc0 .LBB5_83
.LBB5_169:                              ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[38:39], s[68:69]
	flat_load_dwordx4 v[6:9], v[38:39]
	flat_load_dwordx4 v[10:13], v[38:39] offset:16
	flat_load_dwordx4 v[30:33], v[38:39] offset:32
	flat_load_dwordx4 v[34:37], v[38:39] offset:48
	s_abs_i32 s1, s6
	flat_load_dwordx2 v[38:39], v[38:39] offset:64
	s_mul_hi_u32 s9, s1, s96
	s_mul_i32 s33, s9, s47
	s_ashr_i32 s0, s6, 31
	s_sub_i32 s1, s1, s33
	s_xor_b32 s0, s0, s2
	s_add_i32 s42, s9, 1
	s_sub_i32 s33, s1, s47
	s_cmp_ge_u32 s1, s47
	s_cselect_b32 s9, s42, s9
	s_cselect_b32 s1, s33, s1
	s_add_i32 s33, s9, 1
	s_cmp_ge_u32 s1, s47
	s_cselect_b32 s1, s33, s9
	s_xor_b32 s1, s1, s0
	s_sub_i32 s9, s1, s0
	v_readlane_b32 s1, v90, 30
	s_mul_i32 s1, s1, s9
	s_add_i32 s1, s6, s1
	s_mul_i32 s0, s9, s72
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s0, s1, s0
	s_add_i32 s0, s6, s0
	v_readlane_b32 s33, v90, 28
	s_xor_b32 s0, s0, s1
	s_xor_b32 s33, s1, s33
	s_mul_hi_u32 s1, s0, s46
	s_mul_i32 s42, s1, s59
	s_sub_i32 s0, s0, s42
	s_add_i32 s43, s1, 1
	s_sub_i32 s42, s0, s59
	s_cmp_ge_u32 s0, s59
	s_cselect_b32 s1, s43, s1
	s_cselect_b32 s0, s42, s0
	s_add_i32 s42, s1, 1
	s_cmp_ge_u32 s0, s59
	s_cselect_b32 s0, s42, s1
	s_xor_b32 s0, s0, s33
	s_mul_i32 s8, s60, s56
	s_sub_i32 s0, s0, s33
	s_add_i32 s0, s0, s8
	s_mul_i32 s0, s0, s61
	s_add_i32 s0, s0, s78
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s66, s0
	s_addc_u32 s1, s67, s1
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmp_eq_u32 s9, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 1
	v_mov_b32_e32 v22, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v8, 0, v8, vcc
	v_cndmask_b32_e32 v9, 0, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 2
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e32 v8, v8, v10, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 3
	v_cndmask_b32_e32 v8, v8, v12, vcc
	v_cndmask_b32_e32 v9, v9, v13, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 4
	v_cndmask_b32_e32 v9, v9, v31, vcc
	v_cndmask_b32_e32 v8, v8, v30, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 5
	v_cndmask_b32_e32 v8, v8, v32, vcc
	v_cndmask_b32_e32 v9, v9, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 6
	v_cndmask_b32_e32 v9, v9, v35, vcc
	v_cndmask_b32_e32 v8, v8, v34, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s9, 7
	v_sub_co_u32_e64 v6, s[0:1], s0, v6
	v_cndmask_b32_e32 v8, v8, v36, vcc
	v_cndmask_b32_e32 v9, v9, v37, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v7, s[0:1], v22, v7, s[0:1]
	v_cndmask_b32_e32 v9, v9, v39, vcc
	v_cndmask_b32_e32 v8, v8, v38, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_cmp_lg_u32 s9, s56
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v7, 0, v7, vcc
	v_cndmask_b32_e32 v6, 0, v6, vcc
	s_cbranch_scc1 .LBB5_171
; %bb.170:                              ; %Flow845
                                        ;   in Loop: Header=BB5_169 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB5_168
	s_branch .LBB5_172
.LBB5_171:                              ;   in Loop: Header=BB5_169 Depth=2
	flat_store_dword v[6:7], v1 sc0 sc1
	s_cbranch_execnz .LBB5_168
.LBB5_172:                              ;   in Loop: Header=BB5_169 Depth=2
	flat_store_dword v[6:7], v1 sc1
	s_branch .LBB5_168
.LBB5_173:                              ; %.critedge270
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 91
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 92
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,comdat
.Lfunc_end5:
	.size	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals, .Lfunc_end5-_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
                                        ; -- End function
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.num_vgpr, 91
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.num_agpr, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.numbered_sgpr, 100
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.num_named_barrier, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.private_seg_size, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.uses_vcc, 1
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.uses_flat_scratch, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.has_dyn_sized_stack, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.has_recursion, 0
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11576
; TotalNumSgprs: 106
; NumVgprs: 91
; NumAgprs: 0
; TotalNumVgprs: 91
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 91
; AccumOffset: 92
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 22
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_8db5d4376d51495b,@object ; @__hip_cuid_8db5d4376d51495b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_8db5d4376d51495b
__hip_cuid_8db5d4376d51495b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8db5d4376d51495b, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_8db5d4376d51495b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 54
    .symbol:         _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     91
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 50
    .symbol:         _Z21gemm_rs_mi300x_kernelILi64ELi64ELi64ELb0EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     91
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 73
    .symbol:         _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     165
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 61
    .symbol:         _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     246
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 73
    .symbol:         _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     248
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           320
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 80
    .symbol:         _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     91
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
