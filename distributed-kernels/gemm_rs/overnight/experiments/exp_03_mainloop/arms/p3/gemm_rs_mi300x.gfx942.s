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
                                        ; implicit-def: $vgpr88 : SGPR spill to VGPR lane
	s_load_dwordx2 s[34:35], s[0:1], 0xf8
	s_load_dwordx2 s[20:21], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s86, s25, 31
	s_lshr_b32 s3, s86, 29
	v_writelane_b32 v88, s4, 0
	s_add_i32 s3, s25, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v88, s5, 1
	v_writelane_b32 v88, s6, 2
	v_writelane_b32 v88, s7, 3
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
.LBB0_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit273
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
.LBB0_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit276
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
.LBB0_8:                                ; %Flow742
	s_mov_b64 s[8:9], exec
	v_writelane_b32 v88, s8, 28
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_nop 0
	v_writelane_b32 v88, s9, 29
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
	v_writelane_b32 v88, s3, 8
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 10
	s_cmp_lg_u32 s24, 1
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v88, s7, 11
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 20
	s_cmp_lg_u32 s24, 2
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v88, s7, 21
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 4
	s_cmp_lg_u32 s24, 3
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v88, s7, 5
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 16
	s_cmp_lg_u32 s24, 4
	v_mul_lo_u32 v64, s28, v0
	v_writelane_b32 v88, s7, 17
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 6
	s_cmp_lg_u32 s24, 5
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v88, s7, 7
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 14
	s_cmp_lg_u32 s24, 6
	v_cmp_gt_i32_e64 s[10:11], s15, v0
	v_writelane_b32 v88, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 26
	s_cmp_lg_u32 s24, 7
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v88, s7, 27
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s87, s29
	v_cvt_f32_u32_e32 v2, s87
	s_sub_i32 s18, 0, s87
	s_ashr_i32 s3, s29, 31
	s_lshl_b64 s[82:83], s[48:49], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v88, s6, 22
	v_mov_b32_e32 v3, v5
	s_mov_b64 s[98:99], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v88, s7, 23
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_bfrev_b32_e32 v66, 32
	v_readfirstlane_b32 s19, v2
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s88, s19, s18
	s_add_u32 s18, s80, 2
	s_addc_u32 s19, s81, 0
	v_writelane_b32 v88, s18, 24
	v_lshrrev_b32_e32 v2, 3, v0
	s_movk_i32 s89, 0x7fff
	v_writelane_b32 v88, s19, 25
	s_mul_i32 s18, s13, 14
	s_mul_hi_u32 s19, s12, 14
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 14
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	s_add_u32 s18, s18, 2
	s_addc_u32 s19, s19, 0
	v_writelane_b32 v88, s18, 18
	s_mov_b32 s90, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	v_writelane_b32 v88, s19, 19
	s_lshl_b64 s[18:19], s[12:13], 2
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	v_writelane_b32 v88, s18, 12
	s_nop 1
	v_writelane_b32 v88, s19, 13
	s_mul_i32 s18, s13, 12
	s_mul_hi_u32 s19, s12, 12
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 12
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	s_add_u32 s18, s18, 2
	s_addc_u32 s19, s19, 0
	v_writelane_b32 v88, s18, 30
	s_nop 1
	v_writelane_b32 v88, s19, 31
	s_mul_i32 s18, s13, 6
	s_mul_hi_u32 s19, s12, 6
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s12, 6
	s_add_u32 s18, s22, s18
	s_addc_u32 s19, s23, s19
	v_writelane_b32 v88, s18, 32
	s_nop 1
	v_writelane_b32 v88, s19, 33
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
.LBB0_11:                               ; %Flow727
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s14, s31
	s_add_i32 s14, s12, 0x130
	s_cmp_ge_i32 s14, s93
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB0_12:                               ; %Flow739
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
; %bb.15:                               ; %.lr.ph.i.i.i278.preheader
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_mov_b64 s[20:21], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB0_17
.LBB0_16:                               ; %Flow735
                                        ;   in Loop: Header=BB0_17 Depth=2
	s_and_b64 s[18:19], exec, s[68:69]
	s_or_b64 s[20:21], s[18:19], s[20:21]
	s_andn2_b64 s[18:19], s[66:67], exec
	s_and_b64 s[44:45], s[72:73], exec
	s_or_b64 s[66:67], s[18:19], s[44:45]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB0_19
.LBB0_17:                               ; %.lr.ph.i.i.i278
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
.LBB0_21:                               ; %.critedge417
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
.LBB0_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit284
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[18:19], 0, v4
.LBB0_25:                               ; %Flow738
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execz .LBB0_12
; %bb.26:                               ; %.critedge425
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB0_39
; %bb.27:                               ; %.lr.ph.i285.preheader
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
	v_readlane_b32 s18, v88, 24
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s19, v88, 25
	v_lshl_add_u64 v[6:7], s[22:23], 0, v[22:23]
	v_lshl_add_u64 v[10:11], s[50:51], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v88, 18
	v_readlane_b32 s19, v88, 19
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	s_mov_b64 s[72:73], 0
	v_lshl_add_u64 v[12:13], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v88, 12
	v_readlane_b32 s19, v88, 13
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[14:15], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v88, 30
	v_readlane_b32 s19, v88, 31
	s_nop 1
	v_lshl_add_u64 v[16:17], s[18:19], 0, v[22:23]
	v_readlane_b32 s18, v88, 32
	v_readlane_b32 s19, v88, 33
	s_nop 1
	v_lshl_add_u64 v[18:19], s[18:19], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	s_branch .LBB0_29
.LBB0_28:                               ; %.critedge.i288
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
.LBB0_29:                               ; %.lr.ph.i285
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
; %bb.30:                               ; %.preheader.i287.preheader
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
.LBB0_31:                               ; %Flow729
                                        ;   in Loop: Header=BB0_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[44:45], s[78:79], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[78:79], s[44:45], s[18:19]
.LBB0_32:                               ; %Flow728
                                        ;   in Loop: Header=BB0_33 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[18:19], exec, s[78:79]
	s_or_b64 s[76:77], s[18:19], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB0_36
.LBB0_33:                               ; %.preheader.i287
                                        ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[48:49], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_32
; %bb.34:                               ; %.preheader.i287.1
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
.LBB0_36:                               ; %Flow730
                                        ;   in Loop: Header=BB0_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB0_37:                               ; %Flow731
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
	v_lshlrev_b32_e32 v58, 16, v52
	v_and_b32_e32 v59, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
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
.LBB0_39:                               ; %Flow733
                                        ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB0_11
; %bb.40:                               ; %.preheader431
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s28, s24
	v_readlane_b32 s13, v88, 8
	s_add_i32 s12, s91, s12
	s_add_i32 s13, s13, s92
	s_mul_i32 s12, s12, s29
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s18, s38, s12
	s_addc_u32 s19, s39, s13
	v_readlane_b32 s12, v88, 10
	v_readlane_b32 s13, v88, 11
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
	v_readlane_b32 s12, v88, 20
	v_readlane_b32 s13, v88, 21
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
.LBB0_45:                               ; %Flow725
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
	v_readlane_b32 s12, v88, 4
	v_readlane_b32 s13, v88, 5
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
.LBB0_49:                               ; %Flow724
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
	v_readlane_b32 s12, v88, 16
	v_readlane_b32 s13, v88, 17
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
.LBB0_53:                               ; %Flow723
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
	v_readlane_b32 s12, v88, 6
	v_readlane_b32 s13, v88, 7
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
.LBB0_57:                               ; %Flow722
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
	v_readlane_b32 s12, v88, 14
	v_readlane_b32 s13, v88, 15
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
.LBB0_61:                               ; %Flow721
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
	v_readlane_b32 s12, v88, 26
	v_readlane_b32 s13, v88, 27
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
.LBB0_65:                               ; %Flow720
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
	v_readlane_b32 s12, v88, 22
	v_readlane_b32 s13, v88, 23
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
.LBB0_72:                               ; %Flow743
	v_readlane_b32 s4, v88, 28
	v_readlane_b32 s5, v88, 29
	s_or_b64 exec, exec, s[4:5]
	s_load_dwordx2 s[20:21], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
.LBB0_73:                               ; %Flow784
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB0_166
; %bb.74:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s42, s4
	s_addc_u32 s5, s43, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v88, s8, 4
	s_nop 1
	v_writelane_b32 v88, s9, 5
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
	s_branch .LBB0_166
.LBB0_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_166
.LBB0_81:                               ; %.critedge415
	s_lshr_b32 s3, s86, 27
	s_add_i32 s3, s25, s3
	s_ashr_i32 s50, s3, 5
	s_mul_i32 s3, s29, s50
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v88, s3, 6
	s_cbranch_scc1 .LBB0_166
; %bb.82:                               ; %.lr.ph453
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
	v_and_b32_e32 v10, 56, v2
	s_ashr_i32 s87, s0, 6
	v_lshlrev_b32_e32 v6, 4, v0
	v_lshlrev_b32_e32 v12, 1, v10
	s_movk_i32 s0, 0xf80
	v_and_or_b32 v11, v6, s0, v12
	v_add_u32_e32 v7, s51, v11
	v_or_b32_e32 v32, 8, v11
	v_lshrrev_b32_e32 v2, 4, v7
	v_add_u32_e32 v9, s51, v32
	v_and_b32_e32 v8, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v9
	s_movk_i32 s0, 0x100
	v_lshrrev_b32_e32 v20, 3, v0
	v_and_b32_e32 v13, 0x78, v2
	v_cmp_gt_u32_e64 s[4:5], s0, v0
	v_mad_u64_u32 v[2:3], s[0:1], v20, s8, v[10:11]
	s_movk_i32 s0, 0x1f80
	s_nop 0
	v_and_or_b32 v35, v6, s0, v12
	v_ashrrev_i32_e32 v3, 31, v2
	v_add_u32_e32 v6, s86, v35
	v_or_b32_e32 v36, 8, v35
	v_lshl_add_u64 v[14:15], v[2:3], 1, s[6:7]
	v_xor_b32_e32 v34, v8, v7
	v_lshrrev_b32_e32 v2, 4, v6
	v_add_u32_e32 v8, s86, v36
	s_mov_b32 s0, s14
	s_lshl_b32 s3, s29, 2
	s_lshl_b32 s80, s8, 5
	v_and_b32_e32 v7, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v8
	v_writelane_b32 v88, s0, 8
	v_xor_b32_e32 v33, v13, v9
	v_and_b32_e32 v9, 0x78, v2
	v_writelane_b32 v88, s1, 9
	v_mad_u64_u32 v[2:3], s[0:1], v20, s14, v[10:11]
	s_cmp_gt_i32 s27, 0
	s_cselect_b64 s[0:1], -1, 0
	v_writelane_b32 v88, s0, 10
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[10:11]
	v_writelane_b32 v88, s1, 11
	s_ashr_i32 s1, s20, 31
	s_mov_b32 s0, s20
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	s_addc_u32 s1, s39, s1
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v88, s0, 12
	v_bfe_u32 v4, v0, 6, 2
	v_lshrrev_b32_e32 v5, 8, v0
	v_xor_b32_e32 v38, v7, v6
	v_lshlrev_b32_e32 v6, 7, v2
	v_writelane_b32 v88, s1, 13
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v39, v5, 11, v6
	v_lshl_or_b32 v40, v4, 11, v6
	v_writelane_b32 v88, s0, 14
	v_and_b32_e32 v6, 63, v0
	s_min_i32 s27, s30, 32
	v_writelane_b32 v88, s1, 15
	v_cmp_eq_u32_e64 s[0:1], 0, v6
	s_bfe_i64 s[58:59], s[48:49], 0x200000
	s_cmp_gt_i32 s30, 0
	v_writelane_b32 v88, s0, 16
	s_cselect_b64 s[60:61], -1, 0
	v_lshl_or_b32 v43, v4, 4, v2
	v_writelane_b32 v88, s1, 17
	v_lshrrev_b32_e32 v3, 2, v0
	v_readlane_b32 s8, v88, 0
	v_readlane_b32 s9, v88, 1
	s_cmp_lg_u64 s[8:9], 0
	s_cselect_b64 s[62:63], -1, 0
	s_max_i32 s0, s26, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s21, 0
	s_cselect_b64 s[64:65], -1, 0
	s_abs_i32 s92, s3
	v_cvt_f32_u32_e32 v2, s92
	v_and_b32_e32 v3, 12, v3
	s_abs_i32 s93, s30
	v_lshl_or_b32 v44, v5, 4, v3
	v_rcp_iflag_f32_e32 v2, v2
	v_lshlrev_b32_e32 v49, 1, v3
	v_cvt_f32_u32_e32 v3, s93
	v_readlane_b32 s10, v88, 2
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_rcp_iflag_f32_e32 v3, v3
	v_readlane_b32 s11, v88, 3
	v_writelane_b32 v88, s0, 18
	s_lshl_b32 s0, s27, 3
	v_cmp_gt_i32_e64 s[10:11], s0, v0
	v_mad_i64_i32 v[18:19], s[0:1], s48, v20, 0
	v_readfirstlane_b32 s1, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s92
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s89, s1, s0
	s_sub_i32 s0, 0, s93
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s97, s1, s0
	s_lshr_b32 s0, s97, 27
	s_mul_i32 s1, s0, s93
	s_sub_i32 s1, 32, s1
	s_max_i32 s91, s87, 1
	s_bfe_i32 s78, s29, 0x1001d
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
	v_cvt_f32_u32_e32 v5, s98
	v_lshlrev_b32_e32 v4, 7, v20
	v_mov_b32_e32 v13, 0
	v_add_u32_e32 v2, 0, v4
	v_mov_b32_e32 v3, s13
	v_lshl_add_u64 v[20:21], v[2:3], 0, v[12:13]
	v_rcp_iflag_f32_e32 v3, v5
	v_add_u32_e32 v21, v2, v12
	s_xor_b32 s0, s0, s96
	s_sub_i32 s99, s0, s96
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s98
	s_ashr_i32 s25, s33, 31
	v_add3_u32 v54, 0, v12, v4
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
	v_lshlrev_b32_e32 v12, 4, v2
	v_writelane_b32 v88, s0, 20
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v88, s1, 21
	s_mul_i32 s90, s18, s16
	v_readlane_b32 s0, v88, 4
	v_readlane_b32 s1, v88, 5
	v_writelane_b32 v88, s6, 22
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_mov_b64 s[68:69], 0x80
	v_writelane_b32 v88, s7, 23
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v88, s0, 24
	s_mov_b64 s[52:53], 0
	v_xor_b32_e32 v37, v9, v8
	v_mul_lo_u32 v41, s30, v0
	v_add_u32_e32 v42, -1, v1
	s_mul_i32 s90, s90, s24
	v_lshl_add_u32 v45, v43, 1, 0
	v_or_b32_e32 v46, 1, v44
	v_or_b32_e32 v47, 2, v44
	v_or_b32_e32 v48, 3, v44
	v_or_b32_e32 v50, 32, v49
	v_or_b32_e32 v51, 64, v49
	v_or_b32_e32 v52, 0x60, v49
	v_add_u32_e32 v53, 8, v10
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[68:69]
	v_lshl_add_u32 v55, v0, 1, 0
	v_or_b32_e32 v56, 1, v10
	v_lshl_add_u64 v[24:25], v[18:19], 1, v[12:13]
	s_sub_i32 s57, 0, s33
	v_bfrev_b32_e32 v57, 64
	v_and_b32_e32 v58, 0x100, v2
	s_movk_i32 s49, 0x7fff
	v_writelane_b32 v88, s1, 25
	v_writelane_b32 v88, s80, 26
	s_branch .LBB0_85
.LBB0_83:                               ; %Flow746
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_add_i32 s2, s2, s31
	v_readlane_b32 s0, v88, 6
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB0_84:                               ; %Flow778
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[74:75]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[52:53], s[0:1], s[52:53]
	s_andn2_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB0_166
.LBB0_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_96 Depth 2
                                        ;     Child Loop BB0_104 Depth 2
                                        ;     Child Loop BB0_116 Depth 2
                                        ;       Child Loop BB0_120 Depth 3
                                        ;         Child Loop BB0_141 Depth 4
                                        ;         Child Loop BB0_150 Depth 4
                                        ;         Child Loop BB0_156 Depth 4
                                        ;     Child Loop BB0_162 Depth 2
	s_ashr_i32 s0, s2, 31
	s_xor_b32 s7, s0, s78
	s_abs_i32 s0, s2
	s_mul_hi_u32 s1, s0, s89
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
	v_cvt_f32_u32_e32 v2, s6
	s_mul_i32 s73, s73, s3
	s_sub_i32 s14, 0, s6
	s_sub_i32 s8, s2, s73
	v_rcp_iflag_f32_e32 v2, v2
	s_xor_b32 s9, s8, s1
	s_ashr_i32 s42, s9, 31
	s_abs_i32 s9, s8
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_nop 0
	v_readfirstlane_b32 s15, v2
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
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_85 Depth=1
	s_mul_i32 s14, s80, s9
	s_ashr_i32 s15, s14, 31
	v_lshl_add_u64 v[2:3], s[14:15], 1, v[14:15]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB0_87:                               ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_89
; %bb.88:                               ;   in Loop: Header=BB0_85 Depth=1
	;;#ASMSTART
	ds_write_b64 v34, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v33, v[4:5]

	;;#ASMEND
.LBB0_89:                               ; %_ZN7kittens5groupILi8EE4loadITkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi64ENS3_9st_layout3rowEEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEETkNS3_5coord4tileENS_5coordIS9_EEEEvRT_RKT0_RKT1_.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s67, s6, 6
	v_readlane_b32 s0, v88, 8
	s_mul_i32 s16, s67, s0
	s_ashr_i32 s17, s16, 31
	v_lshl_add_u64 v[2:3], s[16:17], 1, v[16:17]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_readlane_b32 s1, v88, 9
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_readlane_b32 s0, v88, 10
	;;#ASMSTART
	ds_write_b64 v38, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v37, v[4:5]

	;;#ASMEND
	v_readlane_b32 s1, v88, 11
	s_andn2_b64 vcc, exec, s[0:1]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB0_98
; %bb.90:                               ; %.lr.ph440.preheader
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_lshl_b32 s0, s72, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s73
	s_sub_i32 s0, s0, s66
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	s_mul_i32 s0, s80, s0
	v_mov_b32_e32 v2, 0
	s_add_i32 s0, s0, 64
	v_lshl_add_u64 v[26:27], s[16:17], 1, v[22:23]
	s_mov_b32 s14, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	s_add_i32 s8, s14, 1
	s_cmp_ge_i32 s8, s87
	s_cbranch_scc1 .LBB0_96
.LBB0_91:                               ;   in Loop: Header=BB0_85 Depth=1
                                        ; implicit-def: $vgpr8_vgpr9
	s_and_saveexec_b64 s[16:17], s[4:5]
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_85 Depth=1
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[6:7], s[0:1], 1, v[14:15]
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
.LBB0_93:                               ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_b32 s1, s8, 1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[16:17], s[4:5]
	s_cbranch_execz .LBB0_95
; %bb.94:                               ;   in Loop: Header=BB0_85 Depth=1
	s_lshl_b32 s15, s1, 12
	s_add_i32 s15, s51, s15
	v_add_u32_e32 v12, s15, v11
	v_lshrrev_b32_e32 v28, 4, v12
	v_add_u32_e32 v29, s15, v32
	v_and_b32_e32 v28, 0x78, v28
	v_lshrrev_b32_e32 v30, 4, v29
	v_and_b32_e32 v30, 0x78, v30
	v_xor_b32_e32 v12, v28, v12
	;;#ASMSTART
	ds_write_b64 v12, v[6:7]

	;;#ASMEND
	v_xor_b32_e32 v29, v30, v29
	;;#ASMSTART
	ds_write_b64 v29, v[8:9]

	;;#ASMEND
.LBB0_95:                               ; %_ZN7kittens5groupILi8EE4loadITkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi64ENS3_9st_layout3rowEEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEETkNS3_5coord4tileENS_5coordIS9_EEEEvRT_RKT0_RKT1_.exit318
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s86, s1
	v_add_u32_e32 v12, s1, v35
	v_lshrrev_b32_e32 v6, 4, v12
	v_add_u32_e32 v29, s1, v36
	v_and_b32_e32 v28, 0x78, v6
	v_lshrrev_b32_e32 v6, 4, v29
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v30, 0x78, v6
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[26:27], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v12, v28, v12
	;;#ASMSTART
	ds_write_b64 v12, v[6:7]

	;;#ASMEND
	v_xor_b32_e32 v29, v30, v29
	;;#ASMSTART
	ds_write_b64 v29, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_96:                               ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_and_b32 s1, s14, 1
	s_lshl_b32 s14, s1, 12
	s_add_i32 s14, s51, s14
	v_add_u32_e32 v12, s14, v39
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s86, s1
	v_add_u32_e32 v6, v49, v12
	v_add_u32_e32 v59, s1, v40
	v_lshrrev_b32_e32 v7, 4, v6
	v_add_u32_e32 v8, v50, v12
	v_and_b32_e32 v7, 0x78, v7
	v_lshrrev_b32_e32 v9, 4, v8
	v_add_u32_e32 v28, v49, v59
	v_xor_b32_e32 v6, v7, v6
	v_and_b32_e32 v9, 0x78, v9
	v_lshrrev_b32_e32 v29, 4, v28
	v_add_u32_e32 v30, v50, v59
	;;#ASMSTART
	ds_read_b64 v[6:7], v6 offset:0

	;;#ASMEND
	v_xor_b32_e32 v8, v9, v8
	v_and_b32_e32 v29, 0x78, v29
	v_lshrrev_b32_e32 v31, 4, v30
	;;#ASMSTART
	ds_read_b64 v[8:9], v8 offset:0

	;;#ASMEND
	v_xor_b32_e32 v28, v29, v28
	v_and_b32_e32 v31, 0x78, v31
	;;#ASMSTART
	ds_read_b64 v[28:29], v28 offset:0

	;;#ASMEND
	v_xor_b32_e32 v30, v31, v30
	;;#ASMSTART
	ds_read_b64 v[30:31], v30 offset:0

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
	s_add_i32 s0, s0, 64
	v_mfma_f32_16x16x16_bf16 v[2:5], v[6:7], v[28:29], v[2:5]
	v_add_u32_e32 v6, v51, v12
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v7, 4, v6
	v_mfma_f32_16x16x16_bf16 v[2:5], v[8:9], v[30:31], v[2:5]
	v_add_u32_e32 v8, v52, v12
	v_and_b32_e32 v7, 0x78, v7
	v_lshrrev_b32_e32 v9, 4, v8
	v_add_u32_e32 v12, v51, v59
	v_xor_b32_e32 v6, v7, v6
	v_and_b32_e32 v9, 0x78, v9
	v_lshrrev_b32_e32 v28, 4, v12
	;;#ASMSTART
	ds_read_b64 v[6:7], v6 offset:0

	;;#ASMEND
	v_xor_b32_e32 v8, v9, v8
	v_and_b32_e32 v28, 0x78, v28
	;;#ASMSTART
	ds_read_b64 v[8:9], v8 offset:0

	;;#ASMEND
	v_xor_b32_e32 v12, v28, v12
	;;#ASMSTART
	ds_read_b64 v[28:29], v12 offset:0

	;;#ASMEND
	v_add_u32_e32 v12, v52, v59
	v_lshrrev_b32_e32 v30, 4, v12
	v_and_b32_e32 v30, 0x78, v30
	v_xor_b32_e32 v12, v30, v12
	;;#ASMSTART
	ds_read_b64 v[30:31], v12 offset:0

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
	s_cmp_eq_u32 s91, s8
	v_mfma_f32_16x16x16_bf16 v[2:5], v[6:7], v[28:29], v[2:5]
	;;#ASMSTART
	;;#ASMEND
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[68:69]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[8:9], v[30:31], v[2:5]
	s_barrier
	s_cbranch_scc1 .LBB0_99
; %bb.97:                               ;   in Loop: Header=BB0_96 Depth=2
	s_mov_b32 s14, s8
	s_add_i32 s8, s14, 1
	s_cmp_ge_i32 s8, s87
	s_cbranch_scc0 .LBB0_91
	s_branch .LBB0_96
.LBB0_98:                               ;   in Loop: Header=BB0_85 Depth=1
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v3, v5
	v_mov_b32_e32 v2, v5
.LBB0_99:                               ; %Flow774
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v88, 20
	v_readlane_b32 s13, v88, 21
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB0_108
; %bb.100:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v88, 14
	v_readlane_b32 s13, v88, 15
	s_and_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB0_108
; %bb.101:                              ;   in Loop: Header=BB0_85 Depth=1
	v_lshl_add_u32 v6, s9, 5, v41
	v_sub_u32_e32 v8, 0, v6
	v_max_i32_e32 v8, v6, v8
	v_mul_hi_u32 v9, v8, s56
	v_mul_lo_u32 v12, v9, s98
	v_sub_u32_e32 v8, v8, v12
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s98, v8
	v_ashrrev_i32_e32 v7, 31, v6
	v_xor_b32_e32 v7, s25, v7
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_subrev_u32_e32 v12, s98, v8
	v_cndmask_b32_e32 v8, v8, v12, vcc
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s98, v8
	v_readlane_b32 s12, v88, 12
	v_readlane_b32 s13, v88, 13
	v_cndmask_b32_e32 v8, v9, v12, vcc
	v_xor_b32_e32 v8, v8, v7
	v_sub_u32_e32 v7, v8, v7
	v_mul_lo_u32 v8, v7, s33
	v_sub_u32_e32 v6, v6, v8
	v_sub_u32_e32 v9, 0, v6
	v_ashrrev_i32_e32 v8, 31, v6
	v_max_i32_e32 v6, v6, v9
	v_mul_hi_u32 v9, v6, s97
	v_mul_lo_u32 v12, v9, s93
	v_sub_u32_e32 v6, v6, v12
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s93, v6
	v_xor_b32_e32 v8, s96, v8
	s_nop 0
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_subrev_u32_e32 v12, s93, v6
	v_cndmask_b32_e32 v6, v6, v12, vcc
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s93, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v9, v12, vcc
	v_xor_b32_e32 v6, v6, v8
	v_sub_u32_e32 v6, v6, v8
	v_mad_u64_u32 v[6:7], s[14:15], v7, s28, v[6:7]
	v_mul_lo_u32 v6, v6, s29
	v_add_u32_e32 v6, s6, v6
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[12:13]
	flat_load_dword v8, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v8, v42
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_108
; %bb.102:                              ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[74:75], 0
                                        ; implicit-def: $sgpr18_sgpr19
                                        ; implicit-def: $sgpr20_sgpr21
	s_branch .LBB0_104
.LBB0_103:                              ; %Flow768
                                        ;   in Loop: Header=BB0_104 Depth=2
	s_and_b64 s[14:15], exec, s[20:21]
	s_or_b64 s[16:17], s[14:15], s[16:17]
	s_andn2_b64 s[14:15], s[18:19], exec
	s_and_b64 s[18:19], s[76:77], exec
	s_or_b64 s[18:19], s[14:15], s[18:19]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_106
.LBB0_104:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s74, s74, 1
	s_addc_u32 s75, s75, 0
	v_mov_b64_e32 v[8:9], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[74:75], v[8:9]
	s_mov_b64 s[76:77], -1
	s_or_b64 s[20:21], s[20:21], exec
	s_cbranch_vccnz .LBB0_103
; %bb.105:                              ;   in Loop: Header=BB0_104 Depth=2
	s_sleep 4
	flat_load_dword v8, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[14:15], s[20:21], exec
	s_mov_b64 s[76:77], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v8, v42
	s_and_b64 s[20:21], vcc, exec
	s_or_b64 s[20:21], s[14:15], s[20:21]
	s_branch .LBB0_103
.LBB0_106:                              ; %loop.exit.guard718
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[14:15], s[18:19]
	s_xor_b64 s[14:15], exec, s[14:15]
	s_cbranch_execz .LBB0_108
; %bb.107:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s16, v88, 0
	v_readlane_b32 s18, v88, 2
	v_readlane_b32 s19, v88, 3
	v_readlane_b32 s17, v88, 1
	s_nop 0
	v_mov_b64_e32 v[6:7], s[18:19]
	flat_atomic_or v[6:7], v57
.LBB0_108:                              ; %.critedge416
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[94:95]
	s_mov_b64 s[16:17], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_112
; %bb.109:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s12, v88, 16
	v_mov_b32_e32 v6, 0
	v_readlane_b32 s13, v88, 17
	s_and_saveexec_b64 s[16:17], s[12:13]
	s_cbranch_execz .LBB0_111
; %bb.110:                              ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s44, v88, 0
	v_readlane_b32 s46, v88, 2
	v_readlane_b32 s47, v88, 3
	v_readlane_b32 s45, v88, 1
	s_nop 0
	v_mov_b64_e32 v[6:7], s[46:47]
	flat_load_dword v6, v[6:7] sc1
.LBB0_111:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit262
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v6, v58, v6
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v6, 0x2000000, v6
	v_cmp_eq_u32_e64 s[16:17], 0, v6
.LBB0_112:                              ; %Flow777
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_and_saveexec_b64 s[74:75], s[16:17]
	s_cbranch_execz .LBB0_84
; %bb.113:                              ; %.critedge424
                                        ;   in Loop: Header=BB0_85 Depth=1
	v_readlane_b32 s0, v88, 22
	v_readlane_b32 s1, v88, 23
	s_mov_b32 s13, s78
	s_mov_b32 s12, s3
	s_mov_b32 s3, s50
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_157
; %bb.114:                              ; %.lr.ph448
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_sub_i32 s18, s26, s67
	s_min_i32 s8, s18, 64
	s_abs_i32 s14, s8
	v_cvt_f32_u32_e32 v8, s14
	s_ashr_i32 s50, s8, 31
	s_sub_i32 s20, 0, s14
	s_lshl_b32 s21, s42, 6
	v_rcp_iflag_f32_e32 v8, v8
	v_or_b32_e32 v6, s67, v43
	v_readlane_b32 s0, v88, 18
	v_cmp_gt_i32_e32 vcc, s26, v6
	v_mul_f32_e32 v8, 0x4f7ffffe, v8
	v_cvt_u32_f32_e32 v8, v8
	v_mov_b32_e32 v7, s0
	v_cndmask_b32_e32 v6, v7, v6, vcc
	v_readlane_b32 s44, v88, 0
	v_mul_lo_u32 v9, s20, v8
	s_lshl_b32 s20, s50, 7
	v_subrev_u32_e32 v60, s20, v55
	s_lshl_b32 s20, s43, 6
	s_sub_i32 s42, s20, s21
	s_lshl_b32 s20, s72, 2
	s_add_i32 s20, s2, s20
	s_sub_i32 s20, s20, s73
	s_sub_i32 s20, s20, s66
	s_lshl_b32 s21, s7, 2
	s_sub_i32 s20, s20, s21
	v_ashrrev_i32_e32 v7, 31, v6
	v_readlane_b32 s45, v88, 1
	s_mul_i32 s15, s8, s27
	v_mul_hi_u32 v9, v8, v9
	s_lshl_b32 s20, s20, 5
	s_lshl_b32 s9, s9, 5
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[44:45]
	v_cmp_gt_i32_e64 s[0:1], s15, v0
	v_cmp_lt_i32_e64 s[76:77], s8, v53
	v_cmp_ge_i32_e64 s[16:17], s8, v53
	v_cmp_gt_i32_e64 s[18:19], s18, v10
	s_mov_b32 s54, 0
	v_add_u32_e32 v59, v8, v9
	s_lshl_b32 s55, s8, 1
	s_sub_i32 s88, 0, s8
	s_add_i32 s43, s90, s20
	v_readlane_b32 s46, v88, 2
	v_readlane_b32 s47, v88, 3
	s_branch .LBB0_116
.LBB0_115:                              ; %._crit_edge446
                                        ;   in Loop: Header=BB0_116 Depth=2
	s_add_i32 s54, s54, 1
	s_add_i32 s43, s43, s30
	s_cmp_eq_u32 s54, s99
	s_cbranch_scc1 .LBB0_157
.LBB0_116:                              ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_120 Depth 3
                                        ;         Child Loop BB0_141 Depth 4
                                        ;         Child Loop BB0_150 Depth 4
                                        ;         Child Loop BB0_156 Depth 4
	s_andn2_b64 vcc, exec, s[60:61]
	s_cbranch_vccnz .LBB0_115
; %bb.117:                              ; %.lr.ph445.preheader
                                        ;   in Loop: Header=BB0_116 Depth=2
	v_mov_b64_e32 v[8:9], s[36:37]
	flat_load_dwordx4 v[26:29], v[8:9]
	flat_load_dwordx4 v[62:65], v[8:9] offset:16
	flat_load_dwordx4 v[66:69], v[8:9] offset:32
	flat_load_dwordx4 v[70:73], v[8:9] offset:48
	s_nop 0
	flat_load_dwordx2 v[8:9], v[8:9] offset:64
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
	v_mov_b32_e32 v12, s23
	s_mov_b32 s45, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v28, 0, v28, vcc
	v_cndmask_b32_e32 v29, 0, v29, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 2
	v_sub_co_u32_e64 v26, s[20:21], s22, v26
	v_cndmask_b32_e32 v28, v28, v62, vcc
	s_nop 0
	v_subb_co_u32_e64 v27, s[20:21], v12, v27, s[20:21]
	v_cndmask_b32_e32 v12, v29, v63, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 3
	v_cndmask_b32_e32 v28, v28, v64, vcc
	v_cndmask_b32_e32 v12, v12, v65, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 4
	v_cndmask_b32_e32 v12, v12, v67, vcc
	v_cndmask_b32_e32 v28, v28, v66, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 5
	v_cndmask_b32_e32 v28, v28, v68, vcc
	v_cndmask_b32_e32 v12, v12, v69, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 6
	v_cndmask_b32_e32 v12, v12, v71, vcc
	v_cndmask_b32_e32 v28, v28, v70, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s46, 7
	v_cndmask_b32_e32 v28, v28, v72, vcc
	v_cndmask_b32_e32 v12, v12, v73, vcc
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
	v_cndmask_b32_e32 v9, v12, v9, vcc
	v_cndmask_b32_e32 v8, v28, v8, vcc
	s_sub_i32 s21, s47, s21
	s_add_i32 s20, s46, s20
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[8:9]
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_mul_i32 s21, s48, s21
	s_mul_i32 s46, s20, s48
	v_cndmask_b32_e32 v9, 0, v27, vcc
	v_cndmask_b32_e32 v8, 0, v26, vcc
	s_add_i32 s20, s42, s21
	s_add_i32 s46, s46, s67
	v_lshl_add_u64 v[26:27], v[8:9], 0, v[24:25]
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s47, s46, 31
	v_lshl_add_u64 v[8:9], s[46:47], 1, v[8:9]
	v_lshl_add_u64 v[26:27], s[20:21], 1, v[26:27]
	s_branch .LBB0_120
.LBB0_118:                              ; %Flow760
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
.LBB0_119:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_add_i32 s45, s45, s27
	s_cmp_ge_i32 s45, s30
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[70:71]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_115
.LBB0_120:                              ; %.lr.ph445
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_116 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB0_141 Depth 4
                                        ;         Child Loop BB0_150 Depth 4
                                        ;         Child Loop BB0_156 Depth 4
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB0_122
; %bb.121:                              ;   in Loop: Header=BB0_120 Depth=3
	flat_load_ushort v12, v[6:7]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	s_branch .LBB0_123
.LBB0_122:                              ;   in Loop: Header=BB0_120 Depth=3
	v_mov_b32_e32 v12, 0
.LBB0_123:                              ;   in Loop: Header=BB0_120 Depth=3
	s_add_i32 s46, s45, s44
	s_add_i32 s47, s46, s27
	v_cmp_le_i32_e32 vcc, s46, v44
	v_cmp_gt_i32_e64 s[20:21], s47, v44
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_125
; %bb.124:                              ;   in Loop: Header=BB0_120 Depth=3
	v_add_f32_e32 v28, v12, v2
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s49
	v_subrev_u32_e32 v29, s46, v44
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB0_125:                              ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v46
	v_cmp_gt_i32_e64 s[20:21], s47, v46
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_127
; %bb.126:                              ;   in Loop: Header=BB0_120 Depth=3
	v_add_f32_e32 v28, v12, v3
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s49
	v_subrev_u32_e32 v29, s46, v46
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB0_127:                              ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v47
	v_cmp_gt_i32_e64 s[20:21], s47, v47
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_129
; %bb.128:                              ;   in Loop: Header=BB0_120 Depth=3
	v_add_f32_e32 v28, v12, v4
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s49
	v_subrev_u32_e32 v29, s46, v47
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB0_129:                              ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s46, v48
	v_cmp_gt_i32_e64 s[20:21], s47, v48
	s_and_b64 s[78:79], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[78:79]
	s_cbranch_execz .LBB0_131
; %bb.130:                              ;   in Loop: Header=BB0_120 Depth=3
	v_add_f32_e32 v12, v12, v5
	v_bfe_u32 v28, v12, 16, 1
	v_add3_u32 v12, v12, v28, s49
	v_subrev_u32_e32 v28, s46, v48
	v_lshl_add_u32 v28, v28, 7, v45
	ds_write_b16_d16_hi v28, v12
.LBB0_131:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi16ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_mul_i32 s20, s59, s45
	s_mul_hi_u32 s21, s58, s45
	s_add_i32 s21, s21, s20
	s_mul_i32 s20, s58, s45
	s_andn2_b64 vcc, exec, s[64:65]
	v_lshl_add_u64 v[28:29], s[20:21], 1, v[8:9]
	s_mov_b64 s[20:21], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_153
; %bb.132:                              ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[80:81], -1
	s_and_saveexec_b64 s[78:79], s[10:11]
	s_cbranch_execz .LBB0_138
; %bb.133:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[82:83], s[76:77]
	s_and_saveexec_b64 s[80:81], s[16:17]
; %bb.134:                              ;   in Loop: Header=BB0_120 Depth=3
	v_lshlrev_b32_e32 v12, 1, v18
	v_lshlrev_b32_e32 v30, 1, v10
	v_add3_u32 v12, v28, v12, v30
	v_or_b32_e32 v12, v20, v12
	v_and_b32_e32 v12, 15, v12
	v_cmp_eq_u32_e32 vcc, 0, v12
	s_andn2_b64 s[46:47], s[76:77], exec
	s_and_b64 s[82:83], vcc, exec
	s_or_b64 s[82:83], s[46:47], s[82:83]
; %bb.135:                              ; %Flow756
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[80:81]
	s_mov_b64 s[80:81], 0
	s_and_saveexec_b64 s[84:85], s[82:83]
; %bb.136:                              ; %.critedge.i
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[80:81], exec
; %bb.137:                              ; %Flow757
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[80:81], s[80:81], exec
.LBB0_138:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[78:79]
	v_cndmask_b32_e64 v12, 0, 1, s[80:81]
	s_nop 0
	v_readfirstlane_b32 s46, v12
	s_bitcmp1_b32 s46, 0
	s_cselect_b64 s[46:47], -1, 0
	s_and_b64 vcc, exec, s[46:47]
	s_cbranch_vccnz .LBB0_143
; %bb.139:                              ;   in Loop: Header=BB0_120 Depth=3
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_142
; %bb.140:                              ; %.lr.ph.i264.preheader
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v30, v60
	v_mov_b32_e32 v12, v0
.LBB0_141:                              ; %.lr.ph.i264
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_116 Depth=2
                                        ;       Parent Loop BB0_120 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v31, v12, v59
	v_mul_lo_u32 v61, v31, s14
	v_sub_u32_e32 v61, v12, v61
	v_add_u32_e32 v62, 1, v31
	v_subrev_u32_e32 v63, s14, v61
	v_cmp_le_u32_e32 vcc, s14, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_add_u32_e32 v62, 1, v31
	v_cmp_le_u32_e32 vcc, s14, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_xor_b32_e32 v31, s50, v31
	v_subrev_u32_e32 v61, s50, v31
	v_mad_u64_u32 v[62:63], s[46:47], s88, v61, v[12:13]
	v_lshlrev_b32_e32 v31, 7, v31
	v_mul_lo_u32 v63, s55, v61
	v_sub_u32_e32 v31, v31, v63
	v_add_u32_e32 v31, v30, v31
	ds_read_u16 v31, v31
	v_mad_i64_i32 v[64:65], s[46:47], s58, v61, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v63, v13
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]
	v_cmp_le_i32_e32 vcc, s15, v12
	v_lshl_add_u64 v[62:63], v[62:63], 1, v[64:65]
	v_add_u32_e32 v30, 0x400, v30
	s_or_b64 s[78:79], vcc, s[78:79]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[62:63], v31
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_141
.LBB0_142:                              ; %Flow748
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB0_143:                              ; %Flow754
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_152
; %bb.144:                              ;   in Loop: Header=BB0_120 Depth=3
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB0_151
; %bb.145:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_and_saveexec_b64 s[46:47], s[16:17]
	s_xor_b64 s[78:79], exec, s[46:47]
	s_cbranch_execz .LBB0_147
; %bb.146:                              ;   in Loop: Header=BB0_120 Depth=3
	ds_read_b128 v[62:65], v21
	v_lshl_add_u64 v[30:31], v[18:19], 1, v[28:29]
	v_lshlrev_b32_e32 v12, 1, v10
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[12:13]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[30:31], v[62:65]
.LBB0_147:                              ; %Flow751
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_andn2_saveexec_b64 s[46:47], s[78:79]
	s_cbranch_execz .LBB0_151
; %bb.148:                              ; %.preheader.i
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_151
; %bb.149:                              ; %.lr.ph.i266
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b32 s46, 0
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v12, v54
	v_mov_b64_e32 v[30:31], v[26:27]
.LBB0_150:                              ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_116 Depth=2
                                        ;       Parent Loop BB0_120 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v61, v12
	s_add_i32 s47, s46, 1
	v_add_u32_e32 v62, s46, v56
	s_cmp_gt_u32 s46, 6
	v_cmp_le_u32_e32 vcc, s8, v62
	s_cselect_b64 s[80:81], -1, 0
	s_or_b64 s[80:81], s[80:81], vcc
	s_and_b64 s[80:81], exec, s[80:81]
	v_add_u32_e32 v12, 2, v12
	s_mov_b32 s46, s47
	s_waitcnt lgkmcnt(0)
	flat_store_short v[30:31], v61
	s_or_b64 s[78:79], s[80:81], s[78:79]
	v_lshl_add_u64 v[30:31], v[30:31], 0, 2
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_150
.LBB0_151:                              ; %Flow753
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_or_b64 exec, exec, s[20:21]
.LBB0_152:                              ; %Flow755
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[20:21], 0
.LBB0_153:                              ; %Flow761
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_119
; %bb.154:                              ;   in Loop: Header=BB0_120 Depth=3
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_118
; %bb.155:                              ; %.lr.ph.i270.preheader
                                        ;   in Loop: Header=BB0_120 Depth=3
	s_mov_b64 s[78:79], 0
	v_mov_b32_e32 v30, v60
	v_mov_b32_e32 v12, v0
.LBB0_156:                              ; %.lr.ph.i270
                                        ;   Parent Loop BB0_85 Depth=1
                                        ;     Parent Loop BB0_116 Depth=2
                                        ;       Parent Loop BB0_120 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v31, v12, v59
	v_mul_lo_u32 v61, v31, s14
	v_sub_u32_e32 v61, v12, v61
	v_add_u32_e32 v62, 1, v31
	v_subrev_u32_e32 v63, s14, v61
	v_cmp_le_u32_e32 vcc, s14, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_add_u32_e32 v62, 1, v31
	v_cmp_le_u32_e32 vcc, s14, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_xor_b32_e32 v31, s50, v31
	v_subrev_u32_e32 v61, s50, v31
	v_mad_u64_u32 v[62:63], s[46:47], s88, v61, v[12:13]
	v_lshlrev_b32_e32 v31, 7, v31
	v_mul_lo_u32 v63, s55, v61
	v_sub_u32_e32 v31, v31, v63
	v_add_u32_e32 v31, v30, v31
	ds_read_u16 v31, v31
	v_mad_i64_i32 v[64:65], s[46:47], s58, v61, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v63, v13
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]
	v_cmp_le_i32_e32 vcc, s15, v12
	v_lshl_add_u64 v[62:63], v[62:63], 1, v[64:65]
	v_add_u32_e32 v30, 0x400, v30
	s_or_b64 s[78:79], vcc, s[78:79]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[62:63], v31
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execnz .LBB0_156
	s_branch .LBB0_118
.LBB0_157:                              ; %._crit_edge449
                                        ;   in Loop: Header=BB0_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s8, v88, 4
	v_readlane_b32 s9, v88, 5
	s_and_b64 s[8:9], s[0:1], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB0_159
; %bb.158:                              ;   in Loop: Header=BB0_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB0_159:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[16:17], exec
	v_readlane_b32 s0, v88, 24
	v_readlane_b32 s1, v88, 25
	s_and_b64 s[0:1], s[16:17], s[0:1]
	s_mov_b32 s50, s3
	s_mov_b32 s3, s12
	s_mov_b32 s78, s13
	v_readlane_b32 s80, v88, 26
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB0_83
; %bb.160:                              ; %.lr.ph451
                                        ;   in Loop: Header=BB0_85 Depth=1
	s_lshl_b32 s0, s72, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s73
	s_sub_i32 s0, s0, s66
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s7, s0, 5
	s_mov_b32 s8, s99
	s_branch .LBB0_162
.LBB0_161:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB0_162 Depth=2
	s_add_i32 s8, s8, -1
	s_add_i32 s7, s7, s30
	s_cmp_lg_u32 s8, 0
	s_cbranch_scc0 .LBB0_83
.LBB0_162:                              ;   Parent Loop BB0_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[30:31], s[40:41]
	flat_load_dwordx4 v[2:5], v[30:31]
	flat_load_dwordx4 v[6:9], v[30:31] offset:16
	flat_load_dwordx4 v[26:29], v[30:31] offset:32
	flat_load_dwordx4 v[60:63], v[30:31] offset:48
	s_abs_i32 s1, s7
	flat_load_dwordx2 v[30:31], v[30:31] offset:64
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
	v_mov_b32_e32 v12, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v4, 0, v4, vcc
	v_cndmask_b32_e32 v5, 0, v5, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 2
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 3
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_cndmask_b32_e32 v5, v5, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 4
	v_cndmask_b32_e32 v5, v5, v27, vcc
	v_cndmask_b32_e32 v4, v4, v26, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 5
	v_cndmask_b32_e32 v4, v4, v28, vcc
	v_cndmask_b32_e32 v5, v5, v29, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 6
	v_cndmask_b32_e32 v5, v5, v61, vcc
	v_cndmask_b32_e32 v4, v4, v60, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 7
	v_sub_co_u32_e64 v2, s[0:1], s0, v2
	v_cndmask_b32_e32 v4, v4, v62, vcc
	v_cndmask_b32_e32 v5, v5, v63, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v3, s[0:1], v12, v3, s[0:1]
	v_cndmask_b32_e32 v5, v5, v31, vcc
	v_cndmask_b32_e32 v4, v4, v30, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_cmp_lg_u32 s14, s24
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB0_164
; %bb.163:                              ; %Flow744
                                        ;   in Loop: Header=BB0_162 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_161
	s_branch .LBB0_165
.LBB0_164:                              ;   in Loop: Header=BB0_162 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB0_161
.LBB0_165:                              ;   in Loop: Header=BB0_162 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB0_161
.LBB0_166:                              ; %.critedge259
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
		.amdhsa_next_free_vgpr 89
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
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb0EEv14mi300x_globals.num_vgpr, 89
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
; codeLenInByte = 10500
; TotalNumSgprs: 106
; NumVgprs: 89
; NumAgprs: 0
; TotalNumVgprs: 89
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 89
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
.LBB1_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit274
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
.LBB1_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit277
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
.LBB1_8:                                ; %Flow793
	s_mov_b64 s[8:9], exec
	v_writelane_b32 v90, s8, 22
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_nop 0
	v_writelane_b32 v90, s9, 23
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
	v_writelane_b32 v90, s3, 4
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 10
	s_cmp_lg_u32 s24, 1
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v90, s7, 11
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 14
	s_cmp_lg_u32 s24, 2
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v90, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 8
	s_cmp_lg_u32 s24, 3
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v90, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 16
	s_cmp_lg_u32 s24, 4
	v_mul_lo_u32 v64, s28, v0
	v_writelane_b32 v90, s7, 17
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 20
	s_cmp_lg_u32 s24, 5
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v90, s7, 21
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 18
	s_cmp_lg_u32 s24, 6
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v90, s7, 19
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v90, s6, 12
	s_cmp_lg_u32 s24, 7
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v90, s7, 13
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s87, s29
	v_cvt_f32_u32_e32 v2, s87
	s_sub_i32 s20, 0, s87
	s_ashr_i32 s3, s29, 31
	s_lshl_b64 s[82:83], s[48:49], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v90, s6, 6
	v_mov_b32_e32 v3, v5
	s_mov_b64 s[98:99], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v90, s7, 7
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_bfrev_b32_e32 v66, 32
	v_readfirstlane_b32 s21, v2
	s_mul_i32 s20, s20, s21
	s_mul_hi_u32 s20, s21, s20
	s_add_i32 s88, s21, s20
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v90, s20, 24
	v_lshrrev_b32_e32 v2, 3, v0
	s_movk_i32 s89, 0x7fff
	v_writelane_b32 v90, s21, 25
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
.LBB1_11:                               ; %Flow778
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s16, s31
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s93
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB1_12:                               ; %Flow790
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
; %bb.15:                               ; %.lr.ph.i.i.i279.preheader
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB1_17
.LBB1_16:                               ; %Flow786
                                        ;   in Loop: Header=BB1_17 Depth=2
	s_and_b64 s[20:21], exec, s[68:69]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[66:67], exec
	s_and_b64 s[66:67], s[72:73], exec
	s_or_b64 s[66:67], s[20:21], s[66:67]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB1_19
.LBB1_17:                               ; %.lr.ph.i.i.i279
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
.LBB1_21:                               ; %.critedge435
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
.LBB1_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit285
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB1_25:                               ; %Flow789
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[14:15]
	s_cbranch_execz .LBB1_12
; %bb.26:                               ; %.critedge443
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB1_39
; %bb.27:                               ; %.lr.ph.i286.preheader
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
	v_readlane_b32 s14, v90, 24
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s15, v90, 25
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
.LBB1_28:                               ; %.critedge.i289
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
.LBB1_29:                               ; %.lr.ph.i286
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
; %bb.30:                               ; %.preheader.i288.preheader
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
.LBB1_31:                               ; %Flow780
                                        ;   in Loop: Header=BB1_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[78:79], s[78:79], s[20:21]
.LBB1_32:                               ; %Flow779
                                        ;   in Loop: Header=BB1_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[78:79]
	s_or_b64 s[76:77], s[14:15], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB1_36
.LBB1_33:                               ; %.preheader.i288
                                        ;   Parent Loop BB1_13 Depth=1
                                        ;     Parent Loop BB1_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[48:49], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_32
; %bb.34:                               ; %.preheader.i288.1
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
.LBB1_36:                               ; %Flow781
                                        ;   in Loop: Header=BB1_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB1_37:                               ; %Flow782
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
.LBB1_39:                               ; %Flow784
                                        ;   in Loop: Header=BB1_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB1_11
; %bb.40:                               ; %.preheader449
                                        ;   in Loop: Header=BB1_13 Depth=1
	v_mov_b64_e32 v[6:7], s[40:41]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s28, s24
	v_readlane_b32 s13, v90, 4
	s_add_i32 s12, s91, s12
	s_add_i32 s13, s13, s92
	s_mul_i32 s12, s12, s29
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s38, s12
	s_addc_u32 s15, s39, s13
	v_readlane_b32 s12, v90, 10
	v_readlane_b32 s13, v90, 11
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
	s_cbranch_vccnz .LBB1_45
; %bb.44:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_45:                               ; %Flow776
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
	s_cbranch_vccnz .LBB1_49
; %bb.48:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_49:                               ; %Flow775
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
	v_readlane_b32 s12, v90, 16
	v_readlane_b32 s13, v90, 17
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
.LBB1_53:                               ; %Flow774
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
	s_cbranch_vccnz .LBB1_57
; %bb.56:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_57:                               ; %Flow773
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
.LBB1_61:                               ; %Flow772
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
	s_cbranch_vccnz .LBB1_65
; %bb.64:                               ;   in Loop: Header=BB1_13 Depth=1
	s_mov_b64 s[12:13], 0
	flat_store_dword v[6:7], v1 sc0 sc1
.LBB1_65:                               ; %Flow771
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
	v_readlane_b32 s12, v90, 6
	v_readlane_b32 s13, v90, 7
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
.LBB1_72:                               ; %Flow794
	v_readlane_b32 s4, v90, 22
	v_readlane_b32 s5, v90, 23
	s_or_b64 exec, exec, s[4:5]
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
.LBB1_73:                               ; %Flow834
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB1_168
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
	s_branch .LBB1_168
.LBB1_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB1_168
.LBB1_81:                               ; %.critedge433
	s_lshr_b32 s3, s86, 26
	s_add_i32 s3, s25, s3
	s_ashr_i32 s52, s3, 6
	s_mul_i32 s4, s29, s52
	s_cmp_ge_i32 s2, s4
	s_cbranch_scc1 .LBB1_168
; %bb.82:                               ; %.lr.ph474
	s_load_dwordx2 s[48:49], s[0:1], 0x80
	s_load_dwordx4 s[8:11], s[0:1], 0x70
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[50:51], s[0:1], 0x20
	s_load_dwordx2 s[6:7], s[0:1], 0x30
	s_load_dwordx2 s[44:45], s[0:1], 0x50
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
	v_and_b32_e32 v10, 56, v2
	s_ashr_i32 s86, s0, 6
	v_lshlrev_b32_e32 v2, 4, v0
	v_lshlrev_b32_e32 v12, 1, v10
	s_movk_i32 s0, 0x1f80
	v_and_or_b32 v11, v2, s0, v12
	v_add_u32_e32 v6, s51, v11
	v_or_b32_e32 v38, 8, v11
	v_lshrrev_b32_e32 v2, 4, v6
	v_add_u32_e32 v8, s51, v38
	v_and_b32_e32 v7, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v8
	v_lshrrev_b32_e32 v20, 3, v0
	v_and_b32_e32 v9, 0x78, v2
	v_mad_u64_u32 v[2:3], s[0:1], v20, s50, v[10:11]
	v_ashrrev_i32_e32 v3, 31, v2
	v_xor_b32_e32 v40, v7, v6
	v_add_u32_e32 v6, s53, v11
	v_lshl_add_u64 v[14:15], v[2:3], 1, s[4:5]
	v_xor_b32_e32 v39, v9, v8
	v_lshrrev_b32_e32 v2, 4, v6
	v_add_u32_e32 v8, s53, v38
	s_lshl_b32 s25, s29, 2
	v_and_b32_e32 v7, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v8
	v_and_b32_e32 v9, 0x78, v2
	v_mad_u64_u32 v[2:3], s[0:1], v20, s44, v[10:11]
	s_cmp_gt_i32 s27, 0
	s_cselect_b64 s[92:93], -1, 0
	s_ashr_i32 s1, s14, 31
	s_mov_b32 s0, s14
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	v_ashrrev_i32_e32 v3, 31, v2
	s_addc_u32 s1, s39, s1
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[6:7]
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v90, s0, 6
	v_bfe_u32 v4, v0, 6, 2
	v_lshrrev_b32_e32 v5, 8, v0
	v_xor_b32_e32 v42, v7, v6
	v_lshlrev_b32_e32 v6, 7, v2
	v_writelane_b32 v90, s1, 7
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v43, v5, 12, v6
	v_lshl_or_b32 v44, v4, 11, v6
	v_writelane_b32 v90, s0, 8
	v_and_b32_e32 v6, 63, v0
	s_min_i32 s27, s30, 32
	v_writelane_b32 v90, s1, 9
	v_cmp_eq_u32_e64 s[0:1], 0, v6
	s_bfe_i64 s[60:61], s[48:49], 0x200000
	s_cmp_gt_i32 s30, 0
	v_writelane_b32 v90, s0, 10
	s_cselect_b64 s[62:63], -1, 0
	v_lshl_or_b32 v47, v4, 4, v2
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
	s_abs_i32 s91, s30
	v_lshl_or_b32 v48, v5, 5, v3
	v_rcp_iflag_f32_e32 v2, v2
	v_lshlrev_b32_e32 v57, 1, v3
	v_cvt_f32_u32_e32 v3, s91
	v_readlane_b32 s6, v90, 2
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_rcp_iflag_f32_e32 v3, v3
	v_readlane_b32 s7, v90, 3
	v_writelane_b32 v90, s0, 12
	s_lshl_b32 s0, s27, 3
	s_mul_i32 s88, s10, s8
	v_cmp_gt_i32_e64 s[8:9], s0, v0
	v_mad_i64_i32 v[18:19], s[0:1], s48, v20, 0
	v_readfirstlane_b32 s1, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s90
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s3, s1, s0
	s_sub_i32 s0, 0, s91
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s95, s1, s0
	s_lshr_b32 s0, s95, 26
	s_mul_i32 s1, s0, s91
	s_sub_i32 s1, 64, s1
	s_max_i32 s89, s86, 1
	s_bfe_i32 s11, s29, 0x1001d
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
	v_lshlrev_b32_e32 v4, 7, v20
	v_mov_b32_e32 v13, 0
	v_add_u32_e32 v2, 0, v4
	v_mov_b32_e32 v3, s17
	v_lshl_add_u64 v[20:21], v[2:3], 0, v[12:13]
	v_rcp_iflag_f32_e32 v3, v5
	v_add_u32_e32 v21, v2, v12
	s_xor_b32 s0, s0, s94
	s_sub_i32 s97, s0, s94
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s96
	s_ashr_i32 s98, s33, 31
	v_add3_u32 v62, 0, v12, v4
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s99, s1, s0
	s_cmp_gt_i32 s97, 0
	s_cselect_b64 s[4:5], -1, 0
	s_lshl_b32 s0, s50, 6
	v_writelane_b32 v90, s0, 14
	s_mul_i32 s0, s61, s27
	s_mul_hi_u32 s1, s48, s27
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s48, s27
	s_lshl_b64 s[72:73], s[0:1], 1
	v_readlane_b32 s0, v90, 4
	v_and_b32_e32 v2, 7, v0
	v_readlane_b32 s1, v90, 5
	v_writelane_b32 v90, s4, 16
	v_lshlrev_b32_e32 v12, 4, v2
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_writelane_b32 v90, s5, 17
	s_and_b64 s[0:1], s[0:1], s[4:5]
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v90, s0, 18
	s_mov_b64 s[70:71], 0x80
	v_lshlrev_b32_e32 v2, 2, v2
	v_cmp_gt_u32_e64 s[78:79], s97, v0
	v_writelane_b32 v90, s1, 19
	s_mov_b64 s[54:55], 0
	v_xor_b32_e32 v41, v9, v8
	v_mul_lo_u32 v45, s30, v0
	v_add_u32_e32 v46, -1, v1
	s_mul_i32 s88, s88, s24
	v_lshl_add_u32 v49, v47, 1, 0
	v_or_b32_e32 v50, 1, v48
	v_or_b32_e32 v51, 2, v48
	v_or_b32_e32 v52, 3, v48
	v_or_b32_e32 v53, 16, v48
	v_or_b32_e32 v54, 17, v48
	v_or_b32_e32 v55, 18, v48
	v_or_b32_e32 v56, 19, v48
	v_or_b32_e32 v58, 32, v57
	v_or_b32_e32 v59, 64, v57
	v_or_b32_e32 v60, 0x60, v57
	v_add_u32_e32 v61, 8, v10
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[70:71]
	v_lshl_add_u64 v[24:25], v[14:15], 0, s[70:71]
	v_lshl_add_u32 v63, v0, 1, 0
	v_or_b32_e32 v64, 1, v10
	v_lshl_add_u64 v[26:27], v[18:19], 1, v[12:13]
	s_sub_i32 s58, 0, s33
	v_bfrev_b32_e32 v65, 64
	v_and_b32_e32 v66, 0x100, v2
	s_movk_i32 s59, 0x7fff
	v_writelane_b32 v90, s78, 20
	s_nop 1
	v_writelane_b32 v90, s79, 21
	s_branch .LBB1_85
.LBB1_83:                               ; %Flow797
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_add_i32 s2, s2, s31
	s_mul_i32 s0, s29, s52
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB1_84:                               ; %Flow828
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[76:77]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[54:55], s[0:1], s[54:55]
	s_andn2_b64 exec, exec, s[54:55]
	s_cbranch_execz .LBB1_168
.LBB1_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_88 Depth 2
                                        ;     Child Loop BB1_95 Depth 2
                                        ;     Child Loop BB1_107 Depth 2
                                        ;       Child Loop BB1_111 Depth 3
                                        ;         Child Loop BB1_143 Depth 4
                                        ;         Child Loop BB1_152 Depth 4
                                        ;         Child Loop BB1_158 Depth 4
                                        ;     Child Loop BB1_164 Depth 2
	s_ashr_i32 s0, s2, 31
	s_xor_b32 s4, s0, s11
	s_abs_i32 s0, s2
	s_mul_hi_u32 s1, s0, s3
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
	s_sub_i32 s1, s52, s0
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
	v_mov_b32_e32 v9, v13
	v_mov_b32_e32 v8, v13
	v_mov_b32_e32 v7, v13
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
	s_mul_i32 s0, s68, s50
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s69, s49, 6
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[14:15]
	s_mul_i32 s0, s69, s44
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v40, v[2:3]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[16:17]
	;;#ASMSTART
	ds_write_b64 v39, v[4:5]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_andn2_b64 vcc, exec, s[92:93]
	;;#ASMSTART
	ds_write_b64 v42, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v41, v[4:5]

	;;#ASMEND
	v_mov_b32_e32 v5, v13
	v_mov_b32_e32 v4, v13
	v_mov_b32_e32 v3, v13
	v_mov_b32_e32 v2, v13
	v_mov_b32_e32 v6, v13
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB1_90
; %bb.86:                               ; %.lr.ph462.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_lshl_add_u64 v[28:29], s[0:1], 1, v[22:23]
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
	v_lshl_add_u64 v[30:31], s[0:1], 1, v[24:25]
	s_mov_b32 s1, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	s_add_i32 s0, s1, 1
	s_cmp_ge_i32 s0, s86
	s_cbranch_scc1 .LBB1_88
.LBB1_87:                               ;   in Loop: Header=BB1_85 Depth=1
	s_lshl_b32 s6, s0, 13
	s_and_b32 s6, s6, 0x2000
	s_add_i32 s7, s51, s6
	v_add_u32_e32 v12, s7, v11
	v_lshrrev_b32_e32 v32, 4, v12
	v_add_u32_e32 v37, s7, v38
	v_and_b32_e32 v36, 0x78, v32
	v_lshrrev_b32_e32 v32, 4, v37
	v_and_b32_e32 v67, 0x78, v32
	v_xor_b32_e32 v12, v36, v12
	s_add_i32 s6, s53, s6
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[30:31], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v37, v67, v37
	;;#ASMSTART
	ds_write_b64 v12, v[32:33]

	;;#ASMEND
	v_add_u32_e32 v12, s6, v11
	;;#ASMSTART
	ds_write_b64 v37, v[34:35]

	;;#ASMEND
	v_lshrrev_b32_e32 v32, 4, v12
	v_add_u32_e32 v37, s6, v38
	v_and_b32_e32 v36, 0x78, v32
	v_lshrrev_b32_e32 v32, 4, v37
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v67, 0x78, v32
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[28:29], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v12, v36, v12
	;;#ASMSTART
	ds_write_b64 v12, v[32:33]

	;;#ASMEND
	v_xor_b32_e32 v37, v67, v37
	;;#ASMSTART
	ds_write_b64 v37, v[34:35]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB1_88:                               ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_lshl_b32 s1, s1, 13
	s_and_b32 s1, s1, 0x2000
	s_add_i32 s6, s51, s1
	v_add_u32_e32 v12, s6, v43
	v_add_u32_e32 v32, v57, v12
	s_add_i32 s1, s53, s1
	v_lshrrev_b32_e32 v33, 4, v32
	v_add_u32_e32 v67, s1, v44
	v_and_b32_e32 v33, 0x78, v33
	v_add_u32_e32 v36, v58, v12
	v_xor_b32_e32 v34, v33, v32
	;;#ASMSTART
	ds_read_b64 v[32:33], v34 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v37, 4, v36
	v_add_u32_e32 v70, v57, v67
	;;#ASMSTART
	ds_read_b64 v[34:35], v34 offset:0x800

	;;#ASMEND
	v_and_b32_e32 v37, 0x78, v37
	v_lshrrev_b32_e32 v71, 4, v70
	v_add_u32_e32 v72, v58, v67
	v_xor_b32_e32 v68, v37, v36
	;;#ASMSTART
	ds_read_b64 v[36:37], v68 offset:0

	;;#ASMEND
	v_and_b32_e32 v71, 0x78, v71
	v_lshrrev_b32_e32 v73, 4, v72
	;;#ASMSTART
	ds_read_b64 v[68:69], v68 offset:0x800

	;;#ASMEND
	v_xor_b32_e32 v70, v71, v70
	v_and_b32_e32 v73, 0x78, v73
	;;#ASMSTART
	ds_read_b64 v[70:71], v70 offset:0

	;;#ASMEND
	v_xor_b32_e32 v72, v73, v72
	;;#ASMSTART
	ds_read_b64 v[72:73], v72 offset:0

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
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[70:71]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[70:71], v[6:9]
	v_add_u32_e32 v32, v59, v12
	v_lshrrev_b32_e32 v33, 4, v32
	v_and_b32_e32 v33, 0x78, v33
	v_mfma_f32_16x16x16_bf16 v[2:5], v[34:35], v[70:71], v[2:5]
	v_add_u32_e32 v12, v60, v12
	v_xor_b32_e32 v34, v33, v32
	;;#ASMSTART
	ds_read_b64 v[32:33], v34 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[36:37], v[72:73], v[6:9]
	v_lshrrev_b32_e32 v36, 4, v12
	;;#ASMSTART
	ds_read_b64 v[34:35], v34 offset:0x800

	;;#ASMEND
	v_and_b32_e32 v36, 0x78, v36
	v_xor_b32_e32 v12, v36, v12
	;;#ASMSTART
	ds_read_b64 v[36:37], v12 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[2:5], v[68:69], v[72:73], v[2:5]
	;;#ASMSTART
	ds_read_b64 v[68:69], v12 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v12, v59, v67
	v_lshrrev_b32_e32 v70, 4, v12
	v_and_b32_e32 v70, 0x78, v70
	v_xor_b32_e32 v12, v70, v12
	;;#ASMSTART
	ds_read_b64 v[70:71], v12 offset:0

	;;#ASMEND
	v_add_u32_e32 v12, v60, v67
	v_lshrrev_b32_e32 v67, 4, v12
	v_and_b32_e32 v67, 0x78, v67
	v_xor_b32_e32 v12, v67, v12
	;;#ASMSTART
	ds_read_b64 v[72:73], v12 offset:0

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
	s_cmp_eq_u32 s89, s0
	v_mfma_f32_16x16x16_bf16 v[6:9], v[32:33], v[70:71], v[6:9]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[70:71]
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[2:5], v[34:35], v[70:71], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[36:37], v[72:73], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[68:69], v[72:73], v[2:5]
	s_cbranch_scc1 .LBB1_90
; %bb.89:                               ;   in Loop: Header=BB1_88 Depth=2
	s_mov_b32 s1, s0
	s_add_i32 s0, s1, 1
	s_cmp_ge_i32 s0, s86
	s_cbranch_scc0 .LBB1_87
	s_branch .LBB1_88
.LBB1_90:                               ; %Flow824
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB1_99
; %bb.91:                               ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s6, v90, 8
	v_readlane_b32 s7, v90, 9
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB1_99
; %bb.92:                               ;   in Loop: Header=BB1_85 Depth=1
	v_add_u32_e32 v12, s68, v45
	v_sub_u32_e32 v29, 0, v12
	v_max_i32_e32 v29, v12, v29
	v_mul_hi_u32 v30, v29, s99
	v_mul_lo_u32 v31, v30, s96
	v_sub_u32_e32 v29, v29, v31
	v_add_u32_e32 v31, 1, v30
	v_cmp_le_u32_e32 vcc, s96, v29
	v_ashrrev_i32_e32 v28, 31, v12
	v_xor_b32_e32 v28, s98, v28
	v_cndmask_b32_e32 v30, v30, v31, vcc
	v_subrev_u32_e32 v31, s96, v29
	v_cndmask_b32_e32 v29, v29, v31, vcc
	v_add_u32_e32 v31, 1, v30
	v_cmp_le_u32_e32 vcc, s96, v29
	s_nop 1
	v_cndmask_b32_e32 v29, v30, v31, vcc
	v_xor_b32_e32 v29, v29, v28
	v_sub_u32_e32 v28, v29, v28
	v_mul_lo_u32 v29, v28, s33
	v_sub_u32_e32 v12, v12, v29
	v_sub_u32_e32 v30, 0, v12
	v_ashrrev_i32_e32 v29, 31, v12
	v_max_i32_e32 v12, v12, v30
	v_mul_hi_u32 v30, v12, s95
	v_mul_lo_u32 v31, v30, s91
	v_sub_u32_e32 v12, v12, v31
	v_add_u32_e32 v31, 1, v30
	v_cmp_le_u32_e32 vcc, s91, v12
	v_xor_b32_e32 v29, s94, v29
	s_nop 0
	v_cndmask_b32_e32 v30, v30, v31, vcc
	v_subrev_u32_e32 v31, s91, v12
	v_cndmask_b32_e32 v12, v12, v31, vcc
	v_add_u32_e32 v31, 1, v30
	v_cmp_le_u32_e32 vcc, s91, v12
	s_nop 1
	v_cndmask_b32_e32 v12, v30, v31, vcc
	v_xor_b32_e32 v12, v12, v29
	v_sub_u32_e32 v12, v12, v29
	v_mad_u64_u32 v[28:29], s[6:7], v28, s28, v[12:13]
	v_mul_lo_u32 v12, v28, s29
	v_add_u32_e32 v28, s49, v12
	v_readlane_b32 s6, v90, 6
	v_ashrrev_i32_e32 v29, 31, v28
	v_readlane_b32 s7, v90, 7
	s_nop 1
	v_lshl_add_u64 v[28:29], v[28:29], 2, s[6:7]
	flat_load_dword v12, v[28:29] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v12, v46
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_99
; %bb.93:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[20:21], 0
                                        ; implicit-def: $sgpr16_sgpr17
                                        ; implicit-def: $sgpr18_sgpr19
	s_branch .LBB1_95
.LBB1_94:                               ; %Flow819
                                        ;   in Loop: Header=BB1_95 Depth=2
	s_and_b64 s[6:7], exec, s[18:19]
	s_or_b64 s[14:15], s[6:7], s[14:15]
	s_andn2_b64 s[6:7], s[16:17], exec
	s_and_b64 s[12:13], s[76:77], exec
	s_or_b64 s[16:17], s[6:7], s[12:13]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB1_97
.LBB1_95:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s20, s20, 1
	s_addc_u32 s21, s21, 0
	v_mov_b64_e32 v[30:31], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[20:21], v[30:31]
	s_mov_b64 s[76:77], -1
	s_or_b64 s[18:19], s[18:19], exec
	s_cbranch_vccnz .LBB1_94
; %bb.96:                               ;   in Loop: Header=BB1_95 Depth=2
	s_sleep 4
	flat_load_dword v12, v[28:29] offset:256 sc0 sc1
	s_andn2_b64 s[6:7], s[18:19], exec
	s_mov_b64 s[76:77], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v12, v46
	s_and_b64 s[12:13], vcc, exec
	s_or_b64 s[18:19], s[6:7], s[12:13]
	s_branch .LBB1_94
.LBB1_97:                               ; %loop.exit.guard769
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB1_99
; %bb.98:                               ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s12, v90, 0
	v_readlane_b32 s14, v90, 2
	v_readlane_b32 s15, v90, 3
	v_readlane_b32 s13, v90, 1
	s_nop 0
	v_mov_b64_e32 v[28:29], s[14:15]
	flat_atomic_or v[28:29], v65
.LBB1_99:                               ; %.critedge434
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[56:57]
	s_mov_b64 s[14:15], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB1_103
; %bb.100:                              ;   in Loop: Header=BB1_85 Depth=1
	v_mov_b32_e32 v12, 0
	s_mov_b64 s[14:15], exec
	v_readlane_b32 s6, v90, 10
	v_readlane_b32 s7, v90, 11
	s_and_b64 s[6:7], s[14:15], s[6:7]
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB1_102
; %bb.101:                              ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s16, v90, 0
	v_readlane_b32 s18, v90, 2
	v_readlane_b32 s19, v90, 3
	v_readlane_b32 s17, v90, 1
	s_nop 0
	v_mov_b64_e32 v[28:29], s[18:19]
	flat_load_dword v12, v[28:29] sc1
.LBB1_102:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit262
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v12, v66, v12
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v12, 0x2000000, v12
	v_cmp_eq_u32_e64 s[14:15], 0, v12
.LBB1_103:                              ; %Flow827
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_and_saveexec_b64 s[76:77], s[14:15]
	s_cbranch_execz .LBB1_84
; %bb.104:                              ; %.critedge442
                                        ;   in Loop: Header=BB1_85 Depth=1
	v_readlane_b32 s0, v90, 16
	v_readlane_b32 s1, v90, 17
	s_mov_b32 s10, s50
	s_mov_b32 s50, s52
	s_mov_b32 s47, s25
	s_mov_b32 s46, s44
	s_mov_b64 s[44:45], s[56:57]
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB1_159
; %bb.105:                              ; %.lr.ph469
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_sub_i32 s12, s26, s69
	s_min_i32 s25, s12, 64
	v_or_b32_e32 v12, s69, v47
	v_readlane_b32 s0, v90, 12
	v_cmp_gt_i32_e32 vcc, s26, v12
	s_abs_i32 s7, s25
	v_mov_b32_e32 v28, s0
	v_cndmask_b32_e32 v28, v28, v12, vcc
	v_cvt_f32_u32_e32 v12, s7
	v_readlane_b32 s16, v90, 0
	v_ashrrev_i32_e32 v29, 31, v28
	v_readlane_b32 s17, v90, 1
	v_rcp_iflag_f32_e32 v12, v12
	v_readlane_b32 s18, v90, 2
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[16:17]
	v_cmp_gt_i32_e64 s[16:17], s12, v10
	v_mul_f32_e32 v12, 0x4f7ffffe, v12
	v_cvt_u32_f32_e32 v12, v12
	s_ashr_i32 s12, s25, 31
	s_sub_i32 s18, 0, s7
	v_readlane_b32 s19, v90, 3
	v_mul_lo_u32 v30, s18, v12
	s_lshl_b32 s18, s12, 7
	v_subrev_u32_e32 v68, s18, v63
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
	v_mul_hi_u32 v30, v12, v30
	s_lshl_b32 s18, s18, 6
	v_cmp_gt_i32_e64 s[0:1], s6, v0
	v_cmp_lt_i32_e64 s[78:79], s25, v61
	v_cmp_ge_i32_e64 s[14:15], s25, v61
	s_mov_b32 s13, 0
	v_add_u32_e32 v67, v12, v30
	s_lshl_b32 s52, s25, 1
	s_sub_i32 s56, 0, s25
	s_add_i32 s87, s88, s18
	s_branch .LBB1_107
.LBB1_106:                              ; %._crit_edge467
                                        ;   in Loop: Header=BB1_107 Depth=2
	s_add_i32 s13, s13, 1
	s_add_i32 s87, s87, s30
	s_cmp_eq_u32 s13, s97
	s_cbranch_scc1 .LBB1_159
.LBB1_107:                              ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_111 Depth 3
                                        ;         Child Loop BB1_143 Depth 4
                                        ;         Child Loop BB1_152 Depth 4
                                        ;         Child Loop BB1_158 Depth 4
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB1_106
; %bb.108:                              ; %.lr.ph466.preheader
                                        ;   in Loop: Header=BB1_107 Depth=2
	v_mov_b64_e32 v[78:79], s[36:37]
	flat_load_dwordx4 v[30:33], v[78:79]
	flat_load_dwordx4 v[34:37], v[78:79] offset:16
	flat_load_dwordx4 v[70:73], v[78:79] offset:32
	flat_load_dwordx4 v[74:77], v[78:79] offset:48
	s_nop 0
	flat_load_dwordx2 v[78:79], v[78:79] offset:64
	s_mul_i32 s42, s13, s30
	s_add_i32 s18, s42, s68
	s_abs_i32 s20, s18
	s_mul_hi_u32 s21, s20, s99
	s_mul_i32 s80, s21, s96
	s_ashr_i32 s19, s18, 31
	s_sub_i32 s20, s20, s80
	s_xor_b32 s19, s19, s98
	s_add_i32 s81, s21, 1
	s_sub_i32 s80, s20, s96
	s_cmp_ge_u32 s20, s96
	s_cselect_b32 s21, s81, s21
	s_cselect_b32 s20, s80, s20
	s_add_i32 s80, s21, 1
	s_cmp_ge_u32 s20, s96
	s_cselect_b32 s20, s80, s21
	s_xor_b32 s20, s20, s19
	s_sub_i32 s20, s20, s19
	s_mul_i32 s21, s20, s33
	s_sub_i32 s80, s18, s21
	s_cmp_eq_u32 s20, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 1
	v_mov_b32_e32 v12, s23
	s_mov_b32 s43, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_cndmask_b32_e32 v33, 0, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 2
	v_sub_co_u32_e64 v30, s[18:19], s22, v30
	v_cndmask_b32_e32 v32, v32, v34, vcc
	s_nop 0
	v_subb_co_u32_e64 v31, s[18:19], v12, v31, s[18:19]
	v_cndmask_b32_e32 v12, v33, v35, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 3
	v_cndmask_b32_e32 v32, v32, v36, vcc
	v_cndmask_b32_e32 v12, v12, v37, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 4
	v_cndmask_b32_e32 v12, v12, v71, vcc
	v_cndmask_b32_e32 v32, v32, v70, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 5
	v_cndmask_b32_e32 v32, v32, v72, vcc
	v_cndmask_b32_e32 v12, v12, v73, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 6
	v_cndmask_b32_e32 v12, v12, v75, vcc
	v_cndmask_b32_e32 v32, v32, v74, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s20, 7
	v_cndmask_b32_e32 v32, v32, v76, vcc
	v_cndmask_b32_e32 v12, v12, v77, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s19, s80
	s_mul_hi_u32 s20, s19, s95
	s_mul_i32 s20, s20, s91
	s_sub_i32 s19, s19, s20
	s_ashr_i32 s18, s80, 31
	s_sub_i32 s20, s19, s91
	s_cmp_ge_u32 s19, s91
	s_cselect_b32 s19, s20, s19
	s_sub_i32 s20, s19, s91
	s_cmp_ge_u32 s19, s91
	s_cselect_b32 s19, s20, s19
	s_add_i32 s20, s80, s88
	s_add_i32 s80, s18, s87
	s_xor_b32 s19, s19, s18
	s_sub_i32 s21, s80, s21
	s_sub_i32 s18, s18, s19
	v_cndmask_b32_e32 v33, v12, v79, vcc
	v_cndmask_b32_e32 v32, v32, v78, vcc
	s_sub_i32 s19, s21, s19
	s_add_i32 s18, s20, s18
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[32:33]
	v_cmp_ne_u64_e32 vcc, 0, v[32:33]
	s_mul_i32 s19, s48, s19
	s_mul_i32 s20, s18, s48
	v_cndmask_b32_e32 v31, 0, v31, vcc
	v_cndmask_b32_e32 v30, 0, v30, vcc
	s_add_i32 s18, s57, s19
	s_add_i32 s20, s20, s69
	v_lshl_add_u64 v[32:33], v[30:31], 0, v[26:27]
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s21, s20, 31
	v_lshl_add_u64 v[30:31], s[20:21], 1, v[30:31]
	v_lshl_add_u64 v[32:33], s[18:19], 1, v[32:33]
	s_branch .LBB1_111
.LBB1_109:                              ; %Flow811
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
.LBB1_110:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_add_i32 s43, s43, s27
	s_cmp_ge_i32 s43, s30
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[72:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB1_106
.LBB1_111:                              ; %.lr.ph466
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_107 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB1_143 Depth 4
                                        ;         Child Loop BB1_152 Depth 4
                                        ;         Child Loop BB1_158 Depth 4
	v_cndmask_b32_e64 v12, 0, 1, s[64:65]
	v_cmp_ne_u32_e64 s[18:19], 1, v12
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB1_113
; %bb.112:                              ;   in Loop: Header=BB1_111 Depth=3
	flat_load_ushort v12, v[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	s_branch .LBB1_114
.LBB1_113:                              ;   in Loop: Header=BB1_111 Depth=3
	v_mov_b32_e32 v12, 0
.LBB1_114:                              ;   in Loop: Header=BB1_111 Depth=3
	s_add_i32 s80, s43, s42
	s_add_i32 s81, s80, s27
	v_cmp_le_i32_e32 vcc, s80, v48
	v_cmp_gt_i32_e64 s[20:21], s81, v48
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_116
; %bb.115:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v6
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v48
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_116:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s80, v50
	v_cmp_gt_i32_e64 s[20:21], s81, v50
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_118
; %bb.117:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v7
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v50
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_118:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s80, v51
	v_cmp_gt_i32_e64 s[20:21], s81, v51
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_120
; %bb.119:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v8
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v51
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_120:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cmp_le_i32_e32 vcc, s80, v52
	v_cmp_gt_i32_e64 s[20:21], s81, v52
	s_and_b64 s[82:83], vcc, s[20:21]
	s_and_saveexec_b64 s[20:21], s[82:83]
	s_cbranch_execz .LBB1_122
; %bb.121:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v12, v12, v9
	v_bfe_u32 v34, v12, 16, 1
	v_add3_u32 v12, v12, v34, s59
	v_subrev_u32_e32 v34, s80, v52
	v_lshl_add_u32 v34, v34, 7, v49
	ds_write_b16_d16_hi v34, v12
.LBB1_122:                              ; %.loopexit.i
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccnz .LBB1_124
; %bb.123:                              ;   in Loop: Header=BB1_111 Depth=3
	flat_load_ushort v12, v[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	s_branch .LBB1_125
.LBB1_124:                              ;   in Loop: Header=BB1_111 Depth=3
	v_mov_b32_e32 v12, 0
.LBB1_125:                              ;   in Loop: Header=BB1_111 Depth=3
	v_cmp_le_i32_e32 vcc, s80, v53
	v_cmp_gt_i32_e64 s[18:19], s81, v53
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_127
; %bb.126:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v2
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v53
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_127:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s80, v54
	v_cmp_gt_i32_e64 s[18:19], s81, v54
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_129
; %bb.128:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v3
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v54
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_129:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s80, v55
	v_cmp_gt_i32_e64 s[18:19], s81, v55
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_131
; %bb.130:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v34, v12, v4
	v_bfe_u32 v35, v34, 16, 1
	v_add3_u32 v34, v34, v35, s59
	v_subrev_u32_e32 v35, s80, v55
	v_lshl_add_u32 v35, v35, 7, v49
	ds_write_b16_d16_hi v35, v34
.LBB1_131:                              ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_i32_e32 vcc, s80, v56
	v_cmp_gt_i32_e64 s[18:19], s81, v56
	s_and_b64 s[20:21], vcc, s[18:19]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_133
; %bb.132:                              ;   in Loop: Header=BB1_111 Depth=3
	v_add_f32_e32 v12, v12, v5
	v_bfe_u32 v34, v12, 16, 1
	v_add3_u32 v12, v12, v34, s59
	v_subrev_u32_e32 v34, s80, v56
	v_lshl_add_u32 v34, v34, 7, v49
	ds_write_b16_d16_hi v34, v12
.LBB1_133:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi32ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
	s_mul_i32 s18, s61, s43
	s_mul_hi_u32 s19, s60, s43
	s_add_i32 s19, s19, s18
	s_mul_i32 s18, s60, s43
	s_andn2_b64 vcc, exec, s[66:67]
	v_lshl_add_u64 v[34:35], s[18:19], 1, v[30:31]
	s_mov_b64 s[18:19], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB1_155
; %bb.134:                              ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[80:81], -1
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB1_140
; %bb.135:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[82:83], s[78:79]
	s_and_saveexec_b64 s[80:81], s[14:15]
; %bb.136:                              ;   in Loop: Header=BB1_111 Depth=3
	v_lshlrev_b32_e32 v12, 1, v18
	v_lshlrev_b32_e32 v36, 1, v10
	v_add3_u32 v12, v34, v12, v36
	v_or_b32_e32 v12, v20, v12
	v_and_b32_e32 v12, 15, v12
	v_cmp_eq_u32_e32 vcc, 0, v12
	s_andn2_b64 s[82:83], s[78:79], exec
	s_and_b64 s[84:85], vcc, exec
	s_or_b64 s[82:83], s[82:83], s[84:85]
; %bb.137:                              ; %Flow807
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[80:81]
	s_mov_b64 s[80:81], 0
	s_and_saveexec_b64 s[84:85], s[82:83]
; %bb.138:                              ; %.critedge.i
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[80:81], exec
; %bb.139:                              ; %Flow808
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[80:81], s[80:81], exec
.LBB1_140:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[20:21]
	v_cndmask_b32_e64 v12, 0, 1, s[80:81]
	s_nop 0
	v_readfirstlane_b32 s20, v12
	s_bitcmp1_b32 s20, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB1_145
; %bb.141:                              ;   in Loop: Header=BB1_111 Depth=3
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB1_144
; %bb.142:                              ; %.lr.ph.i264.preheader
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v36, v68
	v_mov_b32_e32 v12, v0
.LBB1_143:                              ; %.lr.ph.i264
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_107 Depth=2
                                        ;       Parent Loop BB1_111 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v37, v12, v67
	v_mul_lo_u32 v69, v37, s7
	v_sub_u32_e32 v69, v12, v69
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
	v_mad_u64_u32 v[70:71], s[80:81], s56, v69, v[12:13]
	v_lshlrev_b32_e32 v37, 7, v37
	v_mul_lo_u32 v71, s52, v69
	v_sub_u32_e32 v37, v37, v71
	v_add_u32_e32 v37, v36, v37
	ds_read_u16 v37, v37
	v_mad_i64_i32 v[72:73], s[80:81], s60, v69, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v71, v13
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[34:35]
	v_cmp_le_i32_e32 vcc, s6, v12
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[72:73]
	v_add_u32_e32 v36, 0x400, v36
	s_or_b64 s[20:21], vcc, s[20:21]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[70:71], v37
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_143
.LBB1_144:                              ; %Flow799
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_145:                              ; %Flow805
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccnz .LBB1_154
; %bb.146:                              ;   in Loop: Header=BB1_111 Depth=3
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB1_153
; %bb.147:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_xor_b64 s[20:21], exec, s[20:21]
	s_cbranch_execz .LBB1_149
; %bb.148:                              ;   in Loop: Header=BB1_111 Depth=3
	ds_read_b128 v[70:73], v21
	v_lshl_add_u64 v[36:37], v[18:19], 1, v[34:35]
	v_lshlrev_b32_e32 v12, 1, v10
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[12:13]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[36:37], v[70:73]
.LBB1_149:                              ; %Flow802
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_andn2_saveexec_b64 s[20:21], s[20:21]
	s_cbranch_execz .LBB1_153
; %bb.150:                              ; %.preheader.i
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB1_153
; %bb.151:                              ; %.lr.ph.i267
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b32 s80, 0
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v12, v62
	v_mov_b64_e32 v[36:37], v[32:33]
.LBB1_152:                              ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_107 Depth=2
                                        ;       Parent Loop BB1_111 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v69, v12
	s_add_i32 s81, s80, 1
	v_add_u32_e32 v70, s80, v64
	s_cmp_gt_u32 s80, 6
	v_cmp_le_u32_e32 vcc, s25, v70
	s_cselect_b64 s[82:83], -1, 0
	s_or_b64 s[82:83], s[82:83], vcc
	s_and_b64 s[82:83], exec, s[82:83]
	v_add_u32_e32 v12, 2, v12
	s_mov_b32 s80, s81
	s_waitcnt lgkmcnt(0)
	flat_store_short v[36:37], v69
	s_or_b64 s[20:21], s[82:83], s[20:21]
	v_lshl_add_u64 v[36:37], v[36:37], 0, 2
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_152
.LBB1_153:                              ; %Flow804
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_or_b64 exec, exec, s[18:19]
.LBB1_154:                              ; %Flow806
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[18:19], 0
.LBB1_155:                              ; %Flow812
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB1_110
; %bb.156:                              ;   in Loop: Header=BB1_111 Depth=3
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB1_109
; %bb.157:                              ; %.lr.ph.i271.preheader
                                        ;   in Loop: Header=BB1_111 Depth=3
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v36, v68
	v_mov_b32_e32 v12, v0
.LBB1_158:                              ; %.lr.ph.i271
                                        ;   Parent Loop BB1_85 Depth=1
                                        ;     Parent Loop BB1_107 Depth=2
                                        ;       Parent Loop BB1_111 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v37, v12, v67
	v_mul_lo_u32 v69, v37, s7
	v_sub_u32_e32 v69, v12, v69
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
	v_mad_u64_u32 v[70:71], s[80:81], s56, v69, v[12:13]
	v_lshlrev_b32_e32 v37, 7, v37
	v_mul_lo_u32 v71, s52, v69
	v_sub_u32_e32 v37, v37, v71
	v_add_u32_e32 v37, v36, v37
	ds_read_u16 v37, v37
	v_mad_i64_i32 v[72:73], s[80:81], s60, v69, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v71, v13
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[34:35]
	v_cmp_le_i32_e32 vcc, s6, v12
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[72:73]
	v_add_u32_e32 v36, 0x400, v36
	s_or_b64 s[20:21], vcc, s[20:21]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[70:71], v37
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB1_158
	s_branch .LBB1_109
.LBB1_159:                              ; %._crit_edge470
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
	s_cbranch_execz .LBB1_161
; %bb.160:                              ;   in Loop: Header=BB1_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB1_161:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[14:15], exec
	v_readlane_b32 s0, v90, 18
	v_readlane_b32 s1, v90, 19
	v_readlane_b32 s78, v90, 20
	s_and_b64 s[0:1], s[14:15], s[0:1]
	s_mov_b64 s[56:57], s[44:45]
	s_mov_b32 s44, s46
	s_mov_b32 s25, s47
	s_mov_b32 s52, s50
	s_mov_b32 s50, s10
	v_readlane_b32 s79, v90, 21
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB1_83
; %bb.162:                              ; %.lr.ph472.preheader
                                        ;   in Loop: Header=BB1_85 Depth=1
	s_lshl_b32 s0, s5, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s74
	s_sub_i32 s0, s0, s75
	s_lshl_b32 s1, s4, 2
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s4, s0, 6
	s_mov_b32 s5, s97
	s_branch .LBB1_164
.LBB1_163:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB1_164 Depth=2
	s_add_i32 s5, s5, -1
	s_add_i32 s4, s4, s30
	s_cmp_lg_u32 s5, 0
	s_cbranch_scc0 .LBB1_83
.LBB1_164:                              ; %.lr.ph472
                                        ;   Parent Loop BB1_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[36:37], s[40:41]
	flat_load_dwordx4 v[2:5], v[36:37]
	flat_load_dwordx4 v[6:9], v[36:37] offset:16
	flat_load_dwordx4 v[28:31], v[36:37] offset:32
	flat_load_dwordx4 v[32:35], v[36:37] offset:48
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
	v_mov_b32_e32 v12, s1
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
	v_cndmask_b32_e32 v5, v5, v29, vcc
	v_cndmask_b32_e32 v4, v4, v28, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 5
	v_cndmask_b32_e32 v4, v4, v30, vcc
	v_cndmask_b32_e32 v5, v5, v31, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 6
	v_cndmask_b32_e32 v5, v5, v33, vcc
	v_cndmask_b32_e32 v4, v4, v32, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s7, 7
	v_sub_co_u32_e64 v2, s[0:1], s0, v2
	v_cndmask_b32_e32 v4, v4, v34, vcc
	v_cndmask_b32_e32 v5, v5, v35, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v3, s[0:1], v12, v3, s[0:1]
	v_cndmask_b32_e32 v5, v5, v37, vcc
	v_cndmask_b32_e32 v4, v4, v36, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_cmp_lg_u32 s7, s24
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB1_166
; %bb.165:                              ; %Flow795
                                        ;   in Loop: Header=BB1_164 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB1_163
	s_branch .LBB1_167
.LBB1_166:                              ;   in Loop: Header=BB1_164 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB1_163
.LBB1_167:                              ;   in Loop: Header=BB1_164 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB1_163
.LBB1_168:                              ; %.critedge259
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
; codeLenInByte = 10816
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
                                        ; implicit-def: $vgpr152 : SGPR spill to VGPR lane
	s_load_dwordx2 s[58:59], s[0:1], 0xf8
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b32 s88, s2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s89, s37, 31
	s_lshr_b32 s3, s89, 29
	v_writelane_b32 v152, s4, 0
	s_add_i32 s3, s37, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v152, s5, 1
	v_writelane_b32 v152, s6, 2
	v_writelane_b32 v152, s7, 3
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
.LBB2_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit283
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
.LBB2_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit286
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
.LBB2_8:                                ; %Flow1428
	s_mov_b64 s[2:3], exec
	v_writelane_b32 v152, s2, 39
	s_and_b64 s[6:7], s[2:3], s[6:7]
	s_nop 0
	v_writelane_b32 v152, s3, 40
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
	v_writelane_b32 v152, s2, 4
	s_mov_b32 s2, s6
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 49
	s_cmp_lg_u32 s36, 1
	v_mbcnt_hi_u32_b32 v4, -1, v4
	v_writelane_b32 v152, s7, 50
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 47
	s_cmp_lg_u32 s36, 2
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v152, s7, 48
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 45
	s_cmp_lg_u32 s36, 3
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v152, s7, 46
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 8
	s_cmp_lg_u32 s36, 4
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v152, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 32
	s_cmp_lg_u32 s36, 5
	v_mul_lo_u32 v64, s40, v0
	v_writelane_b32 v152, s7, 33
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 29
	s_cmp_lg_u32 s36, 6
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v152, s7, 30
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v152, s6, 11
	s_cmp_lg_u32 s36, 7
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v152, s7, 12
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s53, s41
	v_cvt_f32_u32_e32 v2, s53
	s_sub_i32 s3, 0, s53
	s_ashr_i32 s54, s41, 31
	s_lshl_b64 s[82:83], s[22:23], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v152, s6, 13
	v_lshlrev_b32_e32 v65, 3, v0
	v_mov_b32_e32 v3, v5
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v152, s7, 14
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	s_mov_b64 s[98:99], 0
	v_readfirstlane_b32 s20, v2
	s_mul_i32 s3, s3, s20
	s_mul_hi_u32 s3, s20, s3
	s_add_i32 s55, s20, s3
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v152, s20, 20
	s_mul_i32 s3, s13, 14
	v_lshrrev_b32_e32 v2, 5, v0
	v_writelane_b32 v152, s21, 21
	s_mul_hi_u32 s20, s12, 14
	s_add_i32 s20, s20, s3
	s_mul_i32 s3, s12, 14
	s_add_u32 s3, s56, s3
	s_addc_u32 s20, s57, s20
	s_add_u32 s64, s3, 2
	s_addc_u32 s65, s20, 0
	s_lshl_b64 s[20:21], s[12:13], 2
	v_writelane_b32 v152, s64, 18
	s_add_u32 s20, s56, s20
	s_addc_u32 s21, s57, s21
	v_writelane_b32 v152, s65, 19
	v_writelane_b32 v152, s20, 22
	s_mul_i32 s3, s13, 12
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v152, s21, 23
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
.LBB2_11:                               ; %Flow1413
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[68:69]
	s_sub_i32 s12, s16, s43
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s2
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB2_12:                               ; %Flow1425
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
; %bb.15:                               ; %.lr.ph.i.i.i288.preheader
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[70:71], 0
                                        ; implicit-def: $sgpr66_sgpr67
                                        ; implicit-def: $sgpr68_sgpr69
	s_branch .LBB2_17
.LBB2_16:                               ; %Flow1421
                                        ;   in Loop: Header=BB2_17 Depth=2
	s_and_b64 s[20:21], exec, s[68:69]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[66:67], exec
	s_and_b64 s[66:67], s[72:73], exec
	s_or_b64 s[66:67], s[20:21], s[66:67]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB2_19
.LBB2_17:                               ; %.lr.ph.i.i.i288
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
.LBB2_21:                               ; %.critedge560
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
.LBB2_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit294
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB2_25:                               ; %Flow1424
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_and_saveexec_b64 s[66:67], s[14:15]
	s_cbranch_execz .LBB2_12
; %bb.26:                               ; %.critedge568
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB2_39
; %bb.27:                               ; %.lr.ph.i295.preheader
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
	v_readlane_b32 s14, v152, 20
	v_lshlrev_b64 v[22:23], 1, v[8:9]
	v_readlane_b32 s15, v152, 21
	v_lshl_add_u64 v[6:7], s[56:57], 0, v[22:23]
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v152, 18
	v_readlane_b32 s15, v152, 19
	v_lshl_add_u64 v[16:17], s[90:91], 0, v[22:23]
	v_lshl_add_u64 v[18:19], s[92:93], 0, v[22:23]
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v152, 22
	v_readlane_b32 s15, v152, 23
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	s_mov_b64 s[72:73], 0
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	s_branch .LBB2_29
.LBB2_28:                               ; %.critedge.i298
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
.LBB2_29:                               ; %.lr.ph.i295
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
; %bb.30:                               ; %.preheader.i297.preheader
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
.LBB2_31:                               ; %Flow1415
                                        ;   in Loop: Header=BB2_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[78:79], s[78:79], s[20:21]
.LBB2_32:                               ; %Flow1414
                                        ;   in Loop: Header=BB2_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[78:79]
	s_or_b64 s[76:77], s[14:15], s[76:77]
	s_andn2_b64 exec, exec, s[76:77]
	s_cbranch_execz .LBB2_36
.LBB2_33:                               ; %.preheader.i297
                                        ;   Parent Loop BB2_13 Depth=1
                                        ;     Parent Loop BB2_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[22:23], v[24:25]
	s_or_b64 s[78:79], s[78:79], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_32
; %bb.34:                               ; %.preheader.i297.1
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
.LBB2_36:                               ; %Flow1416
                                        ;   in Loop: Header=BB2_29 Depth=2
	s_or_b64 exec, exec, s[76:77]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB2_37:                               ; %Flow1417
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
.LBB2_39:                               ; %Flow1419
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[68:69], s[4:5]
	s_cbranch_execz .LBB2_11
; %bb.40:                               ; %.preheader577
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b64_e32 v[6:7], s[48:49]
	flat_load_dwordx4 v[6:9], v[6:7]
	s_mul_i32 s12, s40, s36
	v_readlane_b32 s13, v152, 4
	s_add_i32 s12, s86, s12
	s_add_i32 s13, s13, s87
	s_mul_i32 s12, s12, s41
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s46, s12
	s_addc_u32 s15, s47, s13
	v_readlane_b32 s12, v152, 49
	v_readlane_b32 s13, v152, 50
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
	v_readlane_b32 s12, v152, 47
	v_readlane_b32 s13, v152, 48
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
.LBB2_45:                               ; %Flow1411
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
	v_readlane_b32 s12, v152, 45
	v_readlane_b32 s13, v152, 46
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
.LBB2_49:                               ; %Flow1410
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
	v_readlane_b32 s12, v152, 8
	v_readlane_b32 s13, v152, 9
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
.LBB2_53:                               ; %Flow1409
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
	v_readlane_b32 s12, v152, 32
	v_readlane_b32 s13, v152, 33
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
.LBB2_57:                               ; %Flow1408
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
	v_readlane_b32 s12, v152, 29
	v_readlane_b32 s13, v152, 30
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
.LBB2_61:                               ; %Flow1407
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
	v_readlane_b32 s12, v152, 11
	v_readlane_b32 s13, v152, 12
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
.LBB2_65:                               ; %Flow1406
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
	v_readlane_b32 s12, v152, 13
	v_readlane_b32 s13, v152, 14
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
.LBB2_72:                               ; %Flow1429
	v_readlane_b32 s2, v152, 39
	v_readlane_b32 s3, v152, 40
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[14:15], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB2_73:                               ; %Flow1471
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB2_319
; %bb.74:
	s_mov_b32 s4, s88
	s_ashr_i32 s5, s88, 31
	s_mov_b32 s2, s88
	v_writelane_b32 v152, s2, 4
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s50, s4
	v_writelane_b32 v152, s3, 5
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_addc_u32 s5, s51, s5
	s_nop 0
	v_writelane_b32 v152, s2, 6
	s_nop 1
	v_writelane_b32 v152, s3, 7
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
	v_writelane_b32 v152, s2, 8
	s_cmp_eq_u64 s[6:7], 0
	s_nop 0
	v_writelane_b32 v152, s3, 9
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
	s_branch .LBB2_319
.LBB2_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB2_319
.LBB2_81:                               ; %.critedge558
	s_lshr_b32 s2, s89, 25
	s_add_i32 s2, s37, s2
	s_ashr_i32 s51, s2, 7
	s_mul_i32 s4, s41, s51
	v_readlane_b32 s2, v152, 4
	s_cmp_ge_i32 s2, s4
	v_readlane_b32 s3, v152, 5
	v_writelane_b32 v152, s4, 10
	s_cbranch_scc1 .LBB2_319
; %bb.82:                               ; %.lr.ph642
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
	v_lshlrev_b32_e32 v106, 3, v0
	s_and_b32 s0, s2, 15
	s_add_u32 s3, s3, 16
	v_and_b32_e32 v2, 24, v106
	v_lshlrev_b32_e32 v4, 4, v0
	s_cmp_eq_u64 s[0:1], 0
	v_and_b32_e32 v107, 0x1fc0, v4
	v_lshlrev_b32_e32 v108, 1, v2
	s_cselect_b32 s67, s2, s3
	s_add_i32 s0, s39, 31
	v_or_b32_e32 v109, v108, v107
	s_ashr_i32 s1, s0, 31
	v_add_u32_e32 v7, s63, v109
	v_or_b32_e32 v110, 8, v109
	s_lshr_b32 s1, s1, 27
	v_lshrrev_b32_e32 v4, 4, v7
	v_add_u32_e32 v9, s63, v110
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_and_b32_e32 v8, 56, v4
	v_lshrrev_b32_e32 v4, 4, v9
	v_lshrrev_b32_e32 v11, 2, v0
	s_ashr_i32 s82, s0, 5
	v_and_b32_e32 v10, 56, v4
	v_mad_u64_u32 v[4:5], s[0:1], v11, s24, v[2:3]
	v_mad_u64_u32 v[68:69], s[0:1], v11, s10, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v4
	s_mov_b32 s0, s10
	s_add_i32 s37, s82, -1
	s_lshl_b32 s53, s41, 2
	v_lshl_add_u64 v[66:67], v[4:5], 1, s[8:9]
	v_or_b32_e32 v4, 0x80, v11
	v_writelane_b32 v152, s0, 11
	s_cmp_gt_i32 s39, 0
	v_xor_b32_e32 v112, v8, v7
	v_writelane_b32 v152, s1, 12
	v_mad_u64_u32 v[70:71], s[0:1], v4, s10, v[2:3]
	s_cselect_b64 s[0:1], -1, 0
	s_nop 0
	v_writelane_b32 v152, s0, 13
	v_add_u32_e32 v2, s67, v108
	v_add_u32_e32 v5, v2, v107
	v_writelane_b32 v152, s1, 14
	s_ashr_i32 s1, s14, 31
	s_mov_b32 s0, s14
	s_lshl_b32 s2, s37, 5
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v7, 4, v5
	s_add_u32 s0, s46, s0
	v_add_u32_e32 v4, 8, v2
	v_and_b32_e32 v7, 56, v7
	s_addc_u32 s1, s47, s1
	v_xor_b32_e32 v113, v7, v5
	v_add_u32_e32 v5, v4, v107
	v_writelane_b32 v152, s0, 15
	v_lshrrev_b32_e32 v7, 4, v5
	v_or_b32_e32 v115, 0x2000, v107
	v_writelane_b32 v152, s1, 16
	v_and_b32_e32 v7, 56, v7
	v_add_u32_e32 v2, v2, v115
	s_min_i32 s20, s42, 32
	s_mul_i32 s21, s6, s4
	s_bfe_i64 s[72:73], s[60:61], 0x200000
	v_readlane_b32 s4, v152, 0
	v_xor_b32_e32 v114, v7, v5
	v_lshrrev_b32_e32 v5, 4, v2
	s_cmp_gt_i32 s42, 0
	v_readlane_b32 s5, v152, 1
	v_and_b32_e32 v5, 56, v5
	s_cselect_b64 s[74:75], -1, 0
	s_cmp_lg_u64 s[4:5], 0
	v_xor_b32_e32 v116, v5, v2
	v_add_u32_e32 v2, v4, v115
	s_cselect_b64 s[76:77], -1, 0
	s_max_i32 s0, s38, 1
	v_lshrrev_b32_e32 v4, 4, v2
	s_add_i32 s0, s0, -1
	v_and_b32_e32 v4, 56, v4
	s_cmp_lg_u32 s15, 0
	v_xor_b32_e32 v117, v4, v2
	v_and_b32_e32 v2, 15, v0
	s_cselect_b64 s[78:79], -1, 0
	s_abs_i32 s3, s53
	v_lshlrev_b32_e32 v5, 6, v2
	v_lshl_or_b32 v122, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s3
	s_or_b32 s1, s2, 16
	v_readlane_b32 s6, v152, 2
	v_readlane_b32 s7, v152, 3
	v_writelane_b32 v152, s0, 17
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s39, s2
	s_sub_i32 s2, s39, s1
	s_abs_i32 s39, s42
	v_lshl_or_b32 v119, v3, 12, v5
	v_cvt_f32_u32_e32 v3, s39
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_bfe_i32 s1, s41, 0x1001d
	v_rcp_iflag_f32_e32 v3, v3
	v_writelane_b32 v152, s1, 18
	v_writelane_b32 v152, s3, 20
	s_sub_i32 s1, 0, s3
	v_readfirstlane_b32 s3, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s1, s1, s3
	s_mul_hi_u32 s1, s3, s1
	s_add_i32 s1, s3, s1
	v_writelane_b32 v152, s1, 22
	s_sub_i32 s1, 0, s39
	v_readfirstlane_b32 s3, v2
	s_mul_i32 s1, s1, s3
	s_mul_hi_u32 s1, s3, s1
	s_add_i32 s31, s3, s1
	s_lshr_b32 s1, s31, 25
	s_mul_i32 s3, s1, s39
	s_sub_i32 s3, 0x80, s3
	s_lshl_b32 s80, s20, 5
	s_max_i32 s81, s82, 1
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
	v_and_b32_e32 v4, 12, v11
	v_lshl_or_b32 v118, v6, 12, v5
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
	v_lshrrev_b32_e32 v145, 5, v0
	s_cmp_gt_i32 s69, 0
	v_mad_i64_i32 v[2:3], s[0:1], v145, s60, 0
	s_cselect_b64 s[14:15], -1, 0
	v_readlane_b32 s0, v152, 6
	v_readlane_b32 s1, v152, 7
	v_writelane_b32 v152, s14, 24
	s_and_b64 s[0:1], s[0:1], s[14:15]
	v_xor_b32_e32 v111, v10, v9
	v_writelane_b32 v152, s15, 25
	v_writelane_b32 v152, s0, 26
	v_and_b32_e32 v9, 63, v0
	v_and_b32_e32 v10, 31, v0
	v_writelane_b32 v152, s1, 27
	s_add_u32 s0, s12, 64
	v_writelane_b32 v152, s0, 28
	v_writelane_b32 v152, s12, 29
	s_addc_u32 s0, s13, 0
	s_mul_hi_u32 s1, s60, s20
	v_writelane_b32 v152, s13, 30
	v_writelane_b32 v152, s0, 31
	s_mul_i32 s0, s73, s20
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s20
	s_lshl_b64 s[84:85], s[0:1], 1
	s_mov_b32 s0, s24
	v_writelane_b32 v152, s0, 32
	v_lshlrev_b32_e32 v74, 4, v10
	v_mov_b32_e32 v75, 0
	v_writelane_b32 v152, s1, 33
	s_lshl_b32 s0, s24, 7
	v_writelane_b32 v152, s0, 34
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_add_u64 v[76:77], v[2:3], 1, v[74:75]
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_writelane_b32 v152, s0, 35
	v_lshl_or_b32 v123, v6, 6, v4
	v_lshlrev_b32_e32 v6, 9, v145
	v_writelane_b32 v152, s1, 36
	v_cmp_eq_u32_e64 s[0:1], 0, v9
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_lshlrev_b32_e32 v143, 1, v4
	v_writelane_b32 v152, s0, 37
	v_or_b32_e32 v3, v6, v74
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v152, s1, 38
	v_cmp_gt_u32_e64 s[0:1], s69, v0
	v_ashrrev_i32_e32 v69, 31, v68
	v_ashrrev_i32_e32 v71, 31, v70
	v_writelane_b32 v152, s0, 39
	v_mul_lo_u32 v120, s42, v0
	v_add_u32_e32 v121, -1, v1
	v_writelane_b32 v152, s1, 40
	s_mul_i32 s21, s21, s36
	v_lshl_add_u32 v124, v122, 1, 0
	v_or_b32_e32 v125, 1, v123
	v_or_b32_e32 v126, 2, v123
	v_or_b32_e32 v127, 3, v123
	v_or_b32_e32 v128, 16, v122
	v_or_b32_e32 v129, 32, v122
	v_or_b32_e32 v130, 48, v122
	v_or_b32_e32 v131, 16, v123
	v_or_b32_e32 v132, 17, v123
	v_or_b32_e32 v133, 18, v123
	v_or_b32_e32 v134, 19, v123
	v_or_b32_e32 v135, 32, v123
	v_or_b32_e32 v136, 33, v123
	v_or_b32_e32 v137, 34, v123
	v_or_b32_e32 v138, 35, v123
	v_or_b32_e32 v139, 48, v123
	v_or_b32_e32 v140, 49, v123
	v_or_b32_e32 v141, 50, v123
	v_or_b32_e32 v142, 51, v123
	v_or_b32_e32 v144, 32, v143
	v_lshl_add_u64 v[72:73], v[66:67], 0, 64
	v_add_u32_e32 v146, 0, v6
	v_lshl_add_u32 v147, v0, 1, 0
	v_lshl_or_b32 v148, v10, 3, 1
	v_bfrev_b32_e32 v149, 64
	v_add_u32_e32 v150, 0, v3
	v_and_b32_e32 v151, 0x100, v2
	v_cmp_gt_i32_e64 s[12:13], s2, v4
	v_cmp_gt_i32_e64 s[14:15], s2, v5
	v_cmp_gt_i32_e64 s[16:17], s2, v7
	v_cmp_gt_i32_e64 s[18:19], s2, v8
	s_sub_i32 s23, 0, s33
	s_movk_i32 s62, 0x7fff
	s_mov_b64 s[86:87], 0
	v_cmp_gt_i32_e64 s[24:25], s80, v0
	s_lshl_b64 s[88:89], s[72:73], 5
	v_writelane_b32 v152, s51, 41
	v_writelane_b32 v152, s53, 42
	s_branch .LBB2_85
.LBB2_83:                               ; %Flow1432
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_readlane_b32 s0, v152, 4
	s_add_i32 s2, s0, s43
	v_readlane_b32 s1, v152, 5
	s_mov_b32 s0, s2
	v_writelane_b32 v152, s0, 4
	s_nop 1
	v_writelane_b32 v152, s1, 5
	s_nop 0
	v_readlane_b32 s0, v152, 10
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB2_84:                               ; %Flow1465
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[86:87], s[0:1], s[86:87]
	s_andn2_b64 exec, exec, s[86:87]
	s_cbranch_execz .LBB2_319
.LBB2_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_88 Depth 2
                                        ;     Child Loop BB2_99 Depth 2
                                        ;     Child Loop BB2_111 Depth 2
                                        ;       Child Loop BB2_115 Depth 3
                                        ;         Child Loop BB2_309 Depth 4
                                        ;         Child Loop BB2_269 Depth 4
                                        ;         Child Loop BB2_292 Depth 4
                                        ;         Child Loop BB2_299 Depth 4
                                        ;           Child Loop BB2_304 Depth 5
                                        ;     Child Loop BB2_315 Depth 2
	v_readlane_b32 s0, v152, 4
	v_readlane_b32 s1, v152, 5
	s_mov_b32 s28, s0
	s_ashr_i32 s0, s0, 31
	v_readlane_b32 s1, v152, 18
	s_xor_b32 s2, s0, s1
	s_abs_i32 s0, s28
	v_readlane_b32 s1, v152, 22
	s_mul_hi_u32 s1, s0, s1
	v_readlane_b32 s27, v152, 20
	s_mul_i32 s3, s1, s27
	s_sub_i32 s0, s0, s3
	s_add_i32 s3, s1, 1
	s_sub_i32 s26, s0, s27
	s_cmp_ge_u32 s0, s27
	s_cselect_b32 s1, s3, s1
	s_cselect_b32 s0, s26, s0
	s_add_i32 s3, s1, 1
	s_cmp_ge_u32 s0, s27
	s_cselect_b32 s0, s3, s1
	s_xor_b32 s54, s0, s2
	s_sub_i32 s0, s54, s2
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s3, s51, s1
	s_min_i32 s3, s3, 4
	s_abs_i32 s26, s3
	v_cvt_f32_u32_e32 v2, s26
	s_mul_i32 s55, s0, s53
	s_sub_i32 s0, s28, s55
	s_sub_i32 s28, 0, s26
	v_rcp_iflag_f32_e32 v2, v2
	s_xor_b32 s27, s0, s3
	s_ashr_i32 s50, s27, 31
	s_abs_i32 s27, s0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_mov_b32_e32 v25, v75
	v_mov_b32_e32 v24, v75
	v_mov_b32_e32 v23, v75
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
	s_xor_b32 s61, s26, s50
	s_sub_i32 s66, s61, s50
	s_mul_i32 s64, s66, s3
	s_sub_i32 s0, s0, s64
	s_add_i32 s0, s0, s1
	s_lshl_b32 s27, s0, 7
	v_readlane_b32 s0, v152, 32
	v_readlane_b32 s1, v152, 33
	s_mul_i32 s0, s27, s0
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[2:3], s[0:1], 1, v[66:67]
	s_lshl_b32 s52, s66, 8
	v_readlane_b32 s0, v152, 11
	v_readlane_b32 s1, v152, 12
	s_mul_i32 s0, s52, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v152, 29
	v_readlane_b32 s29, v152, 30
	s_add_u32 s28, s28, s0
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	s_addc_u32 s29, s29, s1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v112, v[2:3]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[68:69], 1, s[28:29]
	;;#ASMSTART
	ds_write_b64 v111, v[4:5]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], v[70:71], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v113, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v114, v[4:5]

	;;#ASMEND
	v_readlane_b32 s28, v152, 13
	;;#ASMSTART
	ds_write_b64 v116, v[6:7]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v117, v[8:9]

	;;#ASMEND
	v_readlane_b32 s29, v152, 14
	s_andn2_b64 vcc, exec, s[28:29]
	v_mov_b32_e32 v22, v75
	v_mov_b32_e32 v5, v75
	v_mov_b32_e32 v4, v75
	v_mov_b32_e32 v3, v75
	v_mov_b32_e32 v2, v75
	v_mov_b32_e32 v9, v75
	v_mov_b32_e32 v8, v75
	v_mov_b32_e32 v7, v75
	v_mov_b32_e32 v6, v75
	v_mov_b32_e32 v13, v75
	v_mov_b32_e32 v12, v75
	v_mov_b32_e32 v11, v75
	v_mov_b32_e32 v10, v75
	v_mov_b32_e32 v17, v75
	v_mov_b32_e32 v16, v75
	v_mov_b32_e32 v15, v75
	v_mov_b32_e32 v14, v75
	v_mov_b32_e32 v21, v75
	v_mov_b32_e32 v20, v75
	v_mov_b32_e32 v19, v75
	v_mov_b32_e32 v18, v75
	v_mov_b32_e32 v29, v75
	v_mov_b32_e32 v28, v75
	v_mov_b32_e32 v27, v75
	v_mov_b32_e32 v26, v75
	v_mov_b32_e32 v33, v75
	v_mov_b32_e32 v32, v75
	v_mov_b32_e32 v31, v75
	v_mov_b32_e32 v30, v75
	v_mov_b32_e32 v37, v75
	v_mov_b32_e32 v36, v75
	v_mov_b32_e32 v35, v75
	v_mov_b32_e32 v34, v75
	v_mov_b32_e32 v41, v75
	v_mov_b32_e32 v40, v75
	v_mov_b32_e32 v39, v75
	v_mov_b32_e32 v38, v75
	v_mov_b32_e32 v45, v75
	v_mov_b32_e32 v44, v75
	v_mov_b32_e32 v43, v75
	v_mov_b32_e32 v42, v75
	v_mov_b32_e32 v49, v75
	v_mov_b32_e32 v48, v75
	v_mov_b32_e32 v47, v75
	v_mov_b32_e32 v46, v75
	v_mov_b32_e32 v53, v75
	v_mov_b32_e32 v52, v75
	v_mov_b32_e32 v51, v75
	v_mov_b32_e32 v50, v75
	v_mov_b32_e32 v57, v75
	v_mov_b32_e32 v56, v75
	v_mov_b32_e32 v55, v75
	v_mov_b32_e32 v54, v75
	v_mov_b32_e32 v61, v75
	v_mov_b32_e32 v60, v75
	v_mov_b32_e32 v59, v75
	v_mov_b32_e32 v58, v75
	v_mov_b32_e32 v65, v75
	v_mov_b32_e32 v64, v75
	v_mov_b32_e32 v63, v75
	v_mov_b32_e32 v62, v75
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB2_94
; %bb.86:                               ; %.lr.ph616.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s3, v152, 28
	s_add_u32 s28, s3, s0
	v_readlane_b32 s0, v152, 31
	s_addc_u32 s29, s0, s1
	s_lshl_b32 s0, s54, 2
	v_readlane_b32 s34, v152, 4
	s_add_i32 s0, s34, s0
	s_sub_i32 s0, s0, s55
	s_sub_i32 s0, s0, s64
	s_lshl_b32 s1, s2, 2
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v152, 34
	s_mul_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v22, 0
	v_lshl_add_u64 v[78:79], s[0:1], 1, v[72:73]
	s_mov_b32 s0, 0
	v_mov_b32_e32 v23, v22
	v_mov_b32_e32 v24, v22
	v_mov_b32_e32 v25, v22
	v_mov_b32_e32 v2, v22
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v4, v22
	v_mov_b32_e32 v5, v22
	v_mov_b32_e32 v6, v22
	v_mov_b32_e32 v7, v22
	v_mov_b32_e32 v8, v22
	v_mov_b32_e32 v9, v22
	v_mov_b32_e32 v10, v22
	v_mov_b32_e32 v11, v22
	v_mov_b32_e32 v12, v22
	v_mov_b32_e32 v13, v22
	v_mov_b32_e32 v14, v22
	v_mov_b32_e32 v15, v22
	v_mov_b32_e32 v16, v22
	v_mov_b32_e32 v17, v22
	v_mov_b32_e32 v18, v22
	v_mov_b32_e32 v19, v22
	v_mov_b32_e32 v20, v22
	v_mov_b32_e32 v21, v22
	v_mov_b32_e32 v26, v22
	v_mov_b32_e32 v27, v22
	v_mov_b32_e32 v28, v22
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v30, v22
	v_mov_b32_e32 v31, v22
	v_mov_b32_e32 v32, v22
	v_mov_b32_e32 v33, v22
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v36, v22
	v_mov_b32_e32 v37, v22
	v_mov_b32_e32 v38, v22
	v_mov_b32_e32 v39, v22
	v_mov_b32_e32 v40, v22
	v_mov_b32_e32 v41, v22
	v_mov_b32_e32 v42, v22
	v_mov_b32_e32 v43, v22
	v_mov_b32_e32 v44, v22
	v_mov_b32_e32 v45, v22
	v_mov_b32_e32 v46, v22
	v_mov_b32_e32 v47, v22
	v_mov_b32_e32 v48, v22
	v_mov_b32_e32 v49, v22
	v_mov_b32_e32 v50, v22
	v_mov_b32_e32 v51, v22
	v_mov_b32_e32 v52, v22
	v_mov_b32_e32 v53, v22
	v_mov_b32_e32 v54, v22
	v_mov_b32_e32 v55, v22
	v_mov_b32_e32 v56, v22
	v_mov_b32_e32 v57, v22
	v_mov_b32_e32 v58, v22
	v_mov_b32_e32 v59, v22
	v_mov_b32_e32 v60, v22
	v_mov_b32_e32 v61, v22
	v_mov_b32_e32 v62, v22
	v_mov_b32_e32 v63, v22
	v_mov_b32_e32 v64, v22
	v_mov_b32_e32 v65, v22
	v_readlane_b32 s35, v152, 5
	s_add_i32 s3, s0, 1
	s_cmp_ge_i32 s3, s82
	s_cbranch_scc1 .LBB2_88
.LBB2_87:                               ;   in Loop: Header=BB2_85 Depth=1
	s_and_b32 s1, s3, 1
	s_lshl_b32 s26, s1, 13
	s_add_i32 s26, s63, s26
	v_add_u32_e32 v74, s26, v109
	v_lshrrev_b32_e32 v80, 4, v74
	v_and_b32_e32 v84, 56, v80
	v_add_u32_e32 v85, s26, v110
	s_lshl_b32 s1, s1, 14
	v_lshrrev_b32_e32 v80, 4, v85
	v_xor_b32_e32 v74, v84, v74
	s_add_i32 s1, s67, s1
	v_and_b32_e32 v86, 56, v80
	;;#ASMSTART
	global_load_dwordx4 v[80:83], v[78:79], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v85, v86, v85
	;;#ASMSTART
	ds_write_b64 v74, v[80:81]

	;;#ASMEND
	v_add_u32_e32 v74, s1, v108
	v_add_u32_e32 v89, v74, v107
	v_lshl_add_u64 v[80:81], v[68:69], 1, s[28:29]
	s_or_b32 s26, s1, 8
	v_lshrrev_b32_e32 v90, 4, v89
	;;#ASMSTART
	ds_write_b64 v85, v[82:83]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[80:83], v[80:81], off

	;;#ASMEND
	v_lshl_add_u64 v[84:85], v[70:71], 1, s[28:29]
	v_add_u32_e32 v88, s26, v108
	v_and_b32_e32 v90, 56, v90
	;;#ASMSTART
	global_load_dwordx4 v[84:87], v[84:85], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v89, v90, v89
	;;#ASMSTART
	ds_write_b64 v89, v[80:81]

	;;#ASMEND
	v_add_u32_e32 v80, v88, v107
	v_lshrrev_b32_e32 v81, 4, v80
	v_and_b32_e32 v81, 56, v81
	v_xor_b32_e32 v80, v81, v80
	v_add_u32_e32 v74, v74, v115
	;;#ASMSTART
	ds_write_b64 v80, v[82:83]

	;;#ASMEND
	v_lshrrev_b32_e32 v80, 4, v74
	v_and_b32_e32 v80, 56, v80
	v_xor_b32_e32 v74, v80, v74
	;;#ASMSTART
	ds_write_b64 v74, v[84:85]

	;;#ASMEND
	v_add_u32_e32 v74, v88, v115
	v_lshrrev_b32_e32 v80, 4, v74
	v_and_b32_e32 v80, 56, v80
	v_xor_b32_e32 v74, v80, v74
	;;#ASMSTART
	ds_write_b64 v74, v[86:87]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB2_88:                               ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_and_b32 s1, s0, 1
	s_lshl_b32 s26, s1, 13
	s_add_i32 s26, s63, s26
	v_add_u32_e32 v96, s26, v118
	v_add_u32_e32 v80, v96, v143
	s_lshl_b32 s1, s1, 14
	v_lshrrev_b32_e32 v81, 4, v80
	s_add_i32 s1, s67, s1
	v_and_b32_e32 v81, 56, v81
	v_add_u32_e32 v74, s1, v119
	v_xor_b32_e32 v80, v81, v80
	;;#ASMSTART
	ds_read_b64 v[94:95], v80 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v80 offset:0x400

	;;#ASMEND
	v_add_u32_e32 v82, v74, v143
	;;#ASMSTART
	ds_read_b64 v[88:89], v80 offset:0x800

	;;#ASMEND
	v_lshrrev_b32_e32 v83, 4, v82
	;;#ASMSTART
	ds_read_b64 v[80:81], v80 offset:0xc00

	;;#ASMEND
	v_and_b32_e32 v83, 56, v83
	v_xor_b32_e32 v82, v83, v82
	;;#ASMSTART
	ds_read_b64 v[90:91], v82 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[86:87], v82 offset:0x400

	;;#ASMEND
	s_cmp_eq_u32 s37, s0
	;;#ASMSTART
	ds_read_b64 v[84:85], v82 offset:0x800

	;;#ASMEND
	s_cselect_b64 s[34:35], -1, 0
	s_cmp_lg_u32 s37, s0
	;;#ASMSTART
	ds_read_b64 v[82:83], v82 offset:0xc00

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
	s_cbranch_scc1 .LBB2_90
; %bb.89:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB2_88 Depth=2
	s_or_b64 s[0:1], s[10:11], s[8:9]
	s_or_b64 s[0:1], s[0:1], s[6:7]
	v_cndmask_b32_e64 v97, 0, v94, s[4:5]
	v_and_b32_e32 v98, 0xffff0000, v95
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v98, v98, v95, s[8:9]
	v_cndmask_b32_sdwa v94, v97, v94, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v95, v98, v95, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v97, 0, v92, s[4:5]
	v_and_b32_e32 v98, 0xffff0000, v93
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v98, v98, v93, s[8:9]
	v_cndmask_b32_sdwa v92, v97, v92, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v93, v98, v93, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v97, 0, v88, s[4:5]
	v_and_b32_e32 v98, 0xffff0000, v89
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v98, v98, v89, s[8:9]
	v_cndmask_b32_sdwa v88, v97, v88, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v89, v98, v89, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v97, 0, v80, s[4:5]
	v_and_b32_e32 v98, 0xffff0000, v81
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v98, v98, v81, s[8:9]
	v_cndmask_b32_sdwa v80, v97, v80, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v81, v98, v81, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB2_90:                               ;   in Loop: Header=BB2_88 Depth=2
	v_mfma_f32_16x16x16_bf16 v[62:65], v[94:95], v[90:91], v[62:65]
	v_add_u32_e32 v74, v74, v144
	s_andn2_b64 vcc, exec, s[34:35]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[94:95], v[86:87], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[94:95], v[84:85], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[94:95], v[82:83], v[50:53]
	v_add_u32_e32 v94, v96, v144
	v_lshrrev_b32_e32 v95, 4, v94
	v_and_b32_e32 v95, 56, v95
	v_mfma_f32_16x16x16_bf16 v[46:49], v[92:93], v[90:91], v[46:49]
	v_lshrrev_b32_e32 v96, 4, v74
	v_and_b32_e32 v96, 56, v96
	v_xor_b32_e32 v74, v96, v74
	v_mfma_f32_16x16x16_bf16 v[42:45], v[92:93], v[86:87], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[92:93], v[84:85], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[92:93], v[82:83], v[34:37]
	v_xor_b32_e32 v92, v95, v94
	;;#ASMSTART
	ds_read_b64 v[100:101], v92 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v92 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[94:95], v92 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v92 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[30:33], v[88:89], v[90:91], v[30:33]
	;;#ASMSTART
	ds_read_b64 v[102:103], v74 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[104:105], v74 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[96:97], v74 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[26:29], v[88:89], v[86:87], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[88:89], v[84:85], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[88:89], v[82:83], v[14:17]
	;;#ASMSTART
	ds_read_b64 v[88:89], v74 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[10:13], v[80:81], v[90:91], v[10:13]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[80:81], v[86:87], v[6:9]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[2:5], v[80:81], v[84:85], v[2:5]
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[22:25], v[80:81], v[82:83], v[22:25]
	s_cbranch_vccnz .LBB2_92
; %bb.91:                               ; %.loopexit.i.1
                                        ;   in Loop: Header=BB2_88 Depth=2
	s_or_b64 s[0:1], s[18:19], s[16:17]
	s_or_b64 s[0:1], s[0:1], s[14:15]
	v_cndmask_b32_e64 v74, 0, v100, s[12:13]
	v_and_b32_e32 v80, 0xffff0000, v101
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v80, v80, v101, s[16:17]
	v_cndmask_b32_sdwa v100, v74, v100, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v101, v80, v101, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v74, 0, v98, s[12:13]
	v_and_b32_e32 v80, 0xffff0000, v99
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v80, v80, v99, s[16:17]
	v_cndmask_b32_sdwa v98, v74, v98, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v99, v80, v99, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v74, 0, v94, s[12:13]
	v_and_b32_e32 v80, 0xffff0000, v95
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v80, v80, v95, s[16:17]
	v_cndmask_b32_sdwa v94, v74, v94, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v95, v80, v95, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v74, 0, v92, s[12:13]
	v_and_b32_e32 v80, 0xffff0000, v93
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v80, v80, v93, s[16:17]
	v_cndmask_b32_sdwa v92, v74, v92, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v93, v80, v93, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB2_92:                               ;   in Loop: Header=BB2_88 Depth=2
	v_mfma_f32_16x16x16_bf16 v[62:65], v[100:101], v[102:103], v[62:65]
	s_add_u32 s28, s28, 64
	s_addc_u32 s29, s29, 0
	v_lshl_add_u64 v[78:79], v[78:79], 0, 64
	v_mfma_f32_16x16x16_bf16 v[58:61], v[100:101], v[104:105], v[58:61]
	s_cmp_eq_u32 s81, s3
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[54:57], v[100:101], v[96:97], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[100:101], v[88:89], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[98:99], v[102:103], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[98:99], v[104:105], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[98:99], v[96:97], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[98:99], v[88:89], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[94:95], v[102:103], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[94:95], v[104:105], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[94:95], v[96:97], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[94:95], v[88:89], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[92:93], v[102:103], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[92:93], v[104:105], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[92:93], v[96:97], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[92:93], v[88:89], v[22:25]
	s_cbranch_scc1 .LBB2_94
; %bb.93:                               ;   in Loop: Header=BB2_88 Depth=2
	s_mov_b32 s0, s3
	s_add_i32 s3, s0, 1
	s_cmp_ge_i32 s3, s82
	s_cbranch_scc0 .LBB2_87
	s_branch .LBB2_88
.LBB2_94:                               ; %Flow1461
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v152, 39
	v_readlane_b32 s29, v152, 40
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB2_103
; %bb.95:                               ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s28, v152, 35
	v_readlane_b32 s29, v152, 36
	s_and_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB2_103
; %bb.96:                               ;   in Loop: Header=BB2_85 Depth=1
	v_add_u32_e32 v74, s27, v120
	v_sub_u32_e32 v79, 0, v74
	v_max_i32_e32 v79, v74, v79
	v_mul_hi_u32 v80, v79, s22
	v_mul_lo_u32 v81, v80, s68
	v_sub_u32_e32 v79, v79, v81
	v_add_u32_e32 v81, 1, v80
	v_cmp_le_u32_e32 vcc, s68, v79
	v_ashrrev_i32_e32 v78, 31, v74
	v_xor_b32_e32 v78, s83, v78
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_subrev_u32_e32 v81, s68, v79
	v_cndmask_b32_e32 v79, v79, v81, vcc
	v_add_u32_e32 v81, 1, v80
	v_cmp_le_u32_e32 vcc, s68, v79
	s_nop 1
	v_cndmask_b32_e32 v79, v80, v81, vcc
	v_xor_b32_e32 v79, v79, v78
	v_sub_u32_e32 v78, v79, v78
	v_mul_lo_u32 v79, v78, s33
	v_sub_u32_e32 v74, v74, v79
	v_sub_u32_e32 v80, 0, v74
	v_ashrrev_i32_e32 v79, 31, v74
	v_max_i32_e32 v74, v74, v80
	v_mul_hi_u32 v80, v74, s31
	v_mul_lo_u32 v81, v80, s39
	v_sub_u32_e32 v74, v74, v81
	v_add_u32_e32 v81, 1, v80
	v_cmp_le_u32_e32 vcc, s39, v74
	v_xor_b32_e32 v79, s30, v79
	s_nop 0
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_subrev_u32_e32 v81, s39, v74
	v_cndmask_b32_e32 v74, v74, v81, vcc
	v_add_u32_e32 v81, 1, v80
	v_cmp_le_u32_e32 vcc, s39, v74
	s_nop 1
	v_cndmask_b32_e32 v74, v80, v81, vcc
	v_xor_b32_e32 v74, v74, v79
	v_sub_u32_e32 v74, v74, v79
	v_mad_u64_u32 v[78:79], s[28:29], v78, s40, v[74:75]
	v_mul_lo_u32 v74, v78, s41
	v_add_u32_e32 v78, s66, v74
	v_readlane_b32 s28, v152, 15
	v_ashrrev_i32_e32 v79, 31, v78
	v_readlane_b32 s29, v152, 16
	s_nop 1
	v_lshl_add_u64 v[78:79], v[78:79], 2, s[28:29]
	flat_load_dword v74, v[78:79] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v74, v121
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_103
; %bb.97:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_mov_b32 s70, s64
	s_mov_b32 s26, s55
	s_mov_b32 s3, s54
	s_mov_b64 s[28:29], 0
	s_mov_b64 s[92:93], 0
                                        ; implicit-def: $sgpr34_sgpr35
                                        ; implicit-def: $sgpr90_sgpr91
	s_branch .LBB2_99
.LBB2_98:                               ; %Flow1456
                                        ;   in Loop: Header=BB2_99 Depth=2
	s_and_b64 s[54:55], exec, s[90:91]
	s_or_b64 s[28:29], s[54:55], s[28:29]
	s_andn2_b64 s[34:35], s[34:35], exec
	s_and_b64 s[54:55], s[94:95], exec
	s_or_b64 s[34:35], s[34:35], s[54:55]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB2_101
.LBB2_99:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s92, s92, 1
	s_addc_u32 s93, s93, 0
	v_mov_b64_e32 v[80:81], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[92:93], v[80:81]
	s_mov_b64 s[94:95], -1
	s_or_b64 s[90:91], s[90:91], exec
	s_cbranch_vccnz .LBB2_98
; %bb.100:                              ;   in Loop: Header=BB2_99 Depth=2
	s_sleep 4
	flat_load_dword v74, v[78:79] offset:256 sc0 sc1
	s_andn2_b64 s[54:55], s[90:91], exec
	s_mov_b64 s[94:95], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v74, v121
	s_and_b64 s[64:65], vcc, exec
	s_or_b64 s[90:91], s[54:55], s[64:65]
	s_branch .LBB2_98
.LBB2_101:                              ; %loop.exit.guard1404
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_and_saveexec_b64 s[28:29], s[34:35]
	s_mov_b32 s54, s3
	s_mov_b32 s55, s26
	s_mov_b32 s64, s70
	s_xor_b64 s[28:29], exec, s[28:29]
	s_cbranch_execz .LBB2_103
; %bb.102:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s92, v152, 0
	v_readlane_b32 s94, v152, 2
	v_readlane_b32 s95, v152, 3
	v_readlane_b32 s93, v152, 1
	s_nop 0
	v_mov_b64_e32 v[78:79], s[94:95]
	flat_atomic_or v[78:79], v149
.LBB2_103:                              ; %.critedge559
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s28, v152, 8
	v_readlane_b32 s29, v152, 9
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[28:29]
	s_mov_b64 s[28:29], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB2_107
; %bb.104:                              ;   in Loop: Header=BB2_85 Depth=1
	v_mov_b32_e32 v74, 0
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s34, v152, 37
	v_readlane_b32 s35, v152, 38
	s_and_b64 s[34:35], s[28:29], s[34:35]
	s_mov_b64 exec, s[34:35]
	s_cbranch_execz .LBB2_106
; %bb.105:                              ;   in Loop: Header=BB2_85 Depth=1
	v_readlane_b32 s92, v152, 0
	v_readlane_b32 s94, v152, 2
	v_readlane_b32 s95, v152, 3
	v_readlane_b32 s93, v152, 1
	s_nop 0
	v_mov_b64_e32 v[78:79], s[94:95]
	flat_load_dword v74, v[78:79] sc1
.LBB2_106:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit270
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v74, v151, v74
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v74, 0x2000000, v74
	v_cmp_eq_u32_e64 s[28:29], 0, v74
.LBB2_107:                              ; %Flow1464
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_and_saveexec_b64 s[70:71], s[28:29]
	s_cbranch_execz .LBB2_84
; %bb.108:                              ; %.critedge567
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_writelane_b32 v152, s70, 43
	s_nop 1
	v_writelane_b32 v152, s71, 44
	v_writelane_b32 v152, s64, 45
	v_writelane_b32 v152, s55, 47
	v_writelane_b32 v152, s54, 49
	s_nop 0
	v_readlane_b32 s0, v152, 24
	v_readlane_b32 s1, v152, 25
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_310
; %bb.109:                              ; %.lr.ph637
                                        ;   in Loop: Header=BB2_85 Depth=1
	v_or_b32_e32 v74, s52, v122
	v_readlane_b32 s0, v152, 17
	s_sub_i32 s53, s38, s52
	v_cmp_gt_i32_e32 vcc, s38, v74
	v_mov_b32_e32 v84, s0
	s_min_i32 s54, s53, 0x100
	v_cndmask_b32_e32 v78, v84, v74, vcc
	v_or_b32_e32 v74, s52, v128
	v_cmp_gt_i32_e32 vcc, s38, v74
	s_abs_i32 s55, s54
	v_cvt_f32_u32_e32 v86, s55
	v_cndmask_b32_e32 v80, v84, v74, vcc
	v_or_b32_e32 v74, s52, v129
	v_cmp_gt_i32_e32 vcc, s38, v74
	s_ashr_i32 s65, s54, 31
	s_sub_i32 s0, 0, s55
	v_cndmask_b32_e32 v82, v84, v74, vcc
	v_or_b32_e32 v74, s52, v130
	v_cmp_gt_i32_e32 vcc, s38, v74
	s_lshl_b32 s1, s50, 8
	v_readlane_b32 s34, v152, 4
	v_cndmask_b32_e32 v84, v84, v74, vcc
	v_rcp_iflag_f32_e32 v74, v86
	v_readlane_b32 s92, v152, 0
	v_ashrrev_i32_e32 v79, 31, v78
	v_readlane_b32 s93, v152, 1
	v_mul_f32_e32 v74, 0x4f7ffffe, v74
	v_cvt_u32_f32_e32 v74, v74
	v_ashrrev_i32_e32 v81, 31, v80
	v_ashrrev_i32_e32 v83, 31, v82
	v_ashrrev_i32_e32 v85, 31, v84
	v_mul_lo_u32 v86, s0, v74
	s_lshl_b32 s0, s65, 9
	v_subrev_u32_e32 v97, s0, v147
	s_lshl_b32 s0, s61, 8
	s_sub_i32 s61, s0, s1
	v_readlane_b32 s0, v152, 49
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s34, s0
	v_readlane_b32 s1, v152, 47
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v152, 45
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s1, s2, 2
	s_sub_i32 s0, s0, s1
	s_mul_i32 s64, s54, s20
	v_mul_hi_u32 v86, v74, v86
	s_lshl_b32 s0, s0, 7
	v_lshl_add_u64 v[78:79], v[78:79], 1, s[92:93]
	v_lshl_add_u64 v[80:81], v[80:81], 1, s[92:93]
	v_lshl_add_u64 v[82:83], v[82:83], 1, s[92:93]
	v_lshl_add_u64 v[84:85], v[84:85], 1, s[92:93]
	v_cmp_gt_i32_e64 s[28:29], s64, v0
	s_mov_b32 s51, 0
	v_add_u32_e32 v96, v74, v86
	s_lshl_b32 s3, s54, 1
	s_sub_i32 s26, 0, s54
	s_add_i32 s90, s21, s0
	v_readlane_b32 s94, v152, 2
	v_readlane_b32 s95, v152, 3
	v_readlane_b32 s35, v152, 5
	s_branch .LBB2_111
.LBB2_110:                              ; %._crit_edge635
                                        ;   in Loop: Header=BB2_111 Depth=2
	s_add_i32 s51, s51, 1
	s_add_i32 s90, s90, s42
	s_cmp_eq_u32 s51, s69
	s_cbranch_scc1 .LBB2_310
.LBB2_111:                              ;   Parent Loop BB2_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_115 Depth 3
                                        ;         Child Loop BB2_309 Depth 4
                                        ;         Child Loop BB2_269 Depth 4
                                        ;         Child Loop BB2_292 Depth 4
                                        ;         Child Loop BB2_299 Depth 4
                                        ;           Child Loop BB2_304 Depth 5
	s_andn2_b64 vcc, exec, s[74:75]
	s_cbranch_vccnz .LBB2_110
; %bb.112:                              ; %.lr.ph634.preheader
                                        ;   in Loop: Header=BB2_111 Depth=2
	v_mov_b64_e32 v[94:95], s[44:45]
	flat_load_dwordx4 v[86:89], v[94:95]
	flat_load_dwordx4 v[90:93], v[94:95] offset:16
	flat_load_dwordx4 v[98:101], v[94:95] offset:32
	flat_load_dwordx4 v[102:105], v[94:95] offset:48
	s_nop 0
	flat_load_dwordx2 v[94:95], v[94:95] offset:64
	s_mul_i32 s91, s51, s42
	s_add_i32 s0, s91, s27
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
	v_mov_b32_e32 v74, s57
	s_mov_b32 s50, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v88, 0, v88, vcc
	v_cndmask_b32_e32 v89, 0, v89, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 2
	v_sub_co_u32_e64 v86, s[0:1], s56, v86
	v_cndmask_b32_e32 v88, v88, v90, vcc
	s_nop 0
	v_subb_co_u32_e64 v87, s[0:1], v74, v87, s[0:1]
	v_cndmask_b32_e32 v74, v89, v91, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 3
	v_cndmask_b32_e32 v88, v88, v92, vcc
	v_cndmask_b32_e32 v74, v74, v93, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 4
	v_cndmask_b32_e32 v74, v74, v99, vcc
	v_cndmask_b32_e32 v88, v88, v98, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 5
	v_cndmask_b32_e32 v88, v88, v100, vcc
	v_cndmask_b32_e32 v74, v74, v101, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 6
	v_cndmask_b32_e32 v74, v74, v103, vcc
	v_cndmask_b32_e32 v88, v88, v102, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s34, 7
	v_cndmask_b32_e32 v88, v88, v104, vcc
	v_cndmask_b32_e32 v74, v74, v105, vcc
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
	s_add_i32 s70, s0, s90
	s_xor_b32 s1, s1, s0
	s_sub_i32 s35, s70, s35
	s_sub_i32 s0, s0, s1
	v_cndmask_b32_e32 v89, v74, v95, vcc
	v_cndmask_b32_e32 v88, v88, v94, vcc
	s_sub_i32 s1, s35, s1
	s_add_i32 s0, s34, s0
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[88:89]
	v_cmp_ne_u64_e32 vcc, 0, v[88:89]
	s_mul_i32 s1, s60, s1
	s_mul_i32 s34, s0, s60
	v_cndmask_b32_e32 v87, 0, v87, vcc
	v_cndmask_b32_e32 v86, 0, v86, vcc
	s_add_i32 s0, s61, s1
	s_add_i32 s34, s34, s52
	v_lshl_add_u64 v[88:89], v[86:87], 0, v[76:77]
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s35, s34, 31
	v_lshl_add_u64 v[86:87], s[34:35], 1, v[86:87]
	v_lshl_add_u64 v[88:89], s[0:1], 1, v[88:89]
	s_branch .LBB2_115
.LBB2_113:                              ; %Flow1448
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB2_114:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_add_i32 s50, s50, s20
	s_cmp_ge_i32 s50, s42
	v_lshl_add_u64 v[88:89], v[88:89], 0, s[84:85]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB2_110
.LBB2_115:                              ; %.lr.ph634
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB2_309 Depth 4
                                        ;         Child Loop BB2_269 Depth 4
                                        ;         Child Loop BB2_292 Depth 4
                                        ;         Child Loop BB2_299 Depth 4
                                        ;           Child Loop BB2_304 Depth 5
	v_cndmask_b32_e64 v74, 0, 1, s[76:77]
	v_cmp_ne_u32_e64 s[0:1], 1, v74
	s_andn2_b64 vcc, exec, s[76:77]
	s_cbranch_vccnz .LBB2_117
; %bb.116:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[78:79]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_branch .LBB2_118
.LBB2_117:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
.LBB2_118:                              ;   in Loop: Header=BB2_115 Depth=3
	s_add_i32 s98, s50, s91
	s_add_i32 s99, s98, s20
	v_cmp_le_i32_e32 vcc, s98, v123
	v_cmp_gt_i32_e64 s[34:35], s99, v123
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_120
; %bb.119:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v62
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v123
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_120:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v125
	v_cmp_gt_i32_e64 s[34:35], s99, v125
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_122
; %bb.121:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v63
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v125
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_122:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v126
	v_cmp_gt_i32_e64 s[34:35], s99, v126
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_124
; %bb.123:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v64
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v126
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_124:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v127
	v_cmp_gt_i32_e64 s[34:35], s99, v127
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_126
; %bb.125:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v65
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v127
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74
.LBB2_126:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_283
; %bb.127:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[80:81]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_129
.LBB2_128:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v58
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v123
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_129:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_146
; %bb.130:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_147
.LBB2_131:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_148
.LBB2_132:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_149
.LBB2_133:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[82:83]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_135
.LBB2_134:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v54
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v123
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_135:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_150
; %bb.136:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_151
.LBB2_137:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_152
.LBB2_138:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_153
.LBB2_139:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[84:85]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_141
.LBB2_140:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v50
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v123
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_141:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_154
; %bb.142:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_155
.LBB2_143:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_156
.LBB2_144:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_157
.LBB2_145:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[78:79]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_branch .LBB2_158
.LBB2_146:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v59
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v125
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_131
.LBB2_147:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v60
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v126
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_132
.LBB2_148:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v61
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v127
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_133
.LBB2_149:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_134
	s_branch .LBB2_135
.LBB2_150:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v55
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v125
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_137
.LBB2_151:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v56
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v126
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_138
.LBB2_152:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v57
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v127
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_139
.LBB2_153:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_140
	s_branch .LBB2_141
.LBB2_154:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v51
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v125
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_143
.LBB2_155:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v52
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v126
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_144
.LBB2_156:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v53
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v127
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_145
.LBB2_157:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
.LBB2_158:                              ;   in Loop: Header=BB2_115 Depth=3
	v_cmp_le_i32_e32 vcc, s98, v131
	v_cmp_gt_i32_e64 s[34:35], s99, v131
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_160
; %bb.159:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v46
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v131
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_160:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v132
	v_cmp_gt_i32_e64 s[34:35], s99, v132
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_162
; %bb.161:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v47
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v132
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_162:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v133
	v_cmp_gt_i32_e64 s[34:35], s99, v133
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_164
; %bb.163:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v48
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v133
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_164:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v134
	v_cmp_gt_i32_e64 s[34:35], s99, v134
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_166
; %bb.165:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v49
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v134
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74
.LBB2_166:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_284
; %bb.167:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[80:81]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_169
.LBB2_168:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v42
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v131
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_169:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_186
; %bb.170:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_187
.LBB2_171:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_188
.LBB2_172:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_189
.LBB2_173:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[82:83]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_175
.LBB2_174:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v38
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v131
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_175:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_190
; %bb.176:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_191
.LBB2_177:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_192
.LBB2_178:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_193
.LBB2_179:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[84:85]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_181
.LBB2_180:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v34
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v131
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_181:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_194
; %bb.182:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_195
.LBB2_183:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_196
.LBB2_184:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_197
.LBB2_185:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[78:79]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_branch .LBB2_198
.LBB2_186:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v43
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v132
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_171
.LBB2_187:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v44
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v133
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_172
.LBB2_188:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v45
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v134
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_173
.LBB2_189:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_174
	s_branch .LBB2_175
.LBB2_190:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v39
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v132
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_177
.LBB2_191:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v40
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v133
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_178
.LBB2_192:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v41
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v134
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_179
.LBB2_193:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_180
	s_branch .LBB2_181
.LBB2_194:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v35
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v132
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_183
.LBB2_195:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v36
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v133
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_184
.LBB2_196:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v37
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v134
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_185
.LBB2_197:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
.LBB2_198:                              ;   in Loop: Header=BB2_115 Depth=3
	v_cmp_le_i32_e32 vcc, s98, v135
	v_cmp_gt_i32_e64 s[34:35], s99, v135
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_200
; %bb.199:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v30
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v135
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_200:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v136
	v_cmp_gt_i32_e64 s[34:35], s99, v136
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_202
; %bb.201:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v31
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v136
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_202:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v137
	v_cmp_gt_i32_e64 s[34:35], s99, v137
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_204
; %bb.203:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v32
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v137
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_204:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v138
	v_cmp_gt_i32_e64 s[34:35], s99, v138
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_206
; %bb.205:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v33
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v138
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74
.LBB2_206:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_285
; %bb.207:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[80:81]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_209
.LBB2_208:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v26
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v135
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_209:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_226
; %bb.210:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_227
.LBB2_211:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_228
.LBB2_212:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_229
.LBB2_213:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[82:83]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_215
.LBB2_214:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v18
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v135
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_215:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_230
; %bb.216:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_231
.LBB2_217:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_232
.LBB2_218:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_233
.LBB2_219:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[84:85]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_221
.LBB2_220:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v14
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v135
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_221:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_234
; %bb.222:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_235
.LBB2_223:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_236
.LBB2_224:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_237
.LBB2_225:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[78:79]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_branch .LBB2_238
.LBB2_226:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v27
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v136
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_211
.LBB2_227:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v28
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v137
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_212
.LBB2_228:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v29
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v138
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_213
.LBB2_229:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_214
	s_branch .LBB2_215
.LBB2_230:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v19
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v136
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_217
.LBB2_231:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v20
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v137
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_218
.LBB2_232:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v21
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v138
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_219
.LBB2_233:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_220
	s_branch .LBB2_221
.LBB2_234:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v15
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v136
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_223
.LBB2_235:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v16
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v137
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_224
.LBB2_236:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v17
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v138
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:96
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_225
.LBB2_237:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
.LBB2_238:                              ;   in Loop: Header=BB2_115 Depth=3
	v_cmp_le_i32_e32 vcc, s98, v139
	v_cmp_gt_i32_e64 s[34:35], s99, v139
	s_and_b64 s[92:93], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[92:93]
	s_cbranch_execz .LBB2_240
; %bb.239:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v10
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v139
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_240:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v140
	v_cmp_gt_i32_e64 s[34:35], s99, v140
	s_and_b64 s[94:95], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[94:95]
	s_cbranch_execz .LBB2_242
; %bb.241:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v11
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v140
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_242:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v141
	v_cmp_gt_i32_e64 s[34:35], s99, v141
	s_and_b64 s[96:97], vcc, s[34:35]
	s_and_saveexec_b64 s[34:35], s[96:97]
	s_cbranch_execz .LBB2_244
; %bb.243:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v12
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v141
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90
.LBB2_244:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_le_i32_e32 vcc, s98, v142
	v_cmp_gt_i32_e64 s[34:35], s99, v142
	s_and_b64 s[34:35], vcc, s[34:35]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_246
; %bb.245:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v13
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v142
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74
.LBB2_246:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_286
; %bb.247:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[80:81]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_249
.LBB2_248:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v6
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v139
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
.LBB2_249:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_273
; %bb.250:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_274
.LBB2_251:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_275
.LBB2_252:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_276
.LBB2_253:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[82:83]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execz .LBB2_255
.LBB2_254:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v2
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v139
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
.LBB2_255:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[94:95]
	s_cbranch_execnz .LBB2_277
; %bb.256:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execnz .LBB2_278
.LBB2_257:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execnz .LBB2_279
.LBB2_258:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_280
.LBB2_259:                              ;   in Loop: Header=BB2_115 Depth=3
	flat_load_ushort v74, v[84:85]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v74, 16, v74
	s_and_saveexec_b64 s[0:1], s[92:93]
	s_cbranch_execz .LBB2_261
.LBB2_260:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v22
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v139
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
.LBB2_261:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execnz .LBB2_281
; %bb.262:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execnz .LBB2_282
.LBB2_263:                              ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[34:35]
	s_cbranch_execz .LBB2_265
.LBB2_264:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v25
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v142
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:96
.LBB2_265:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi64ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s73, s50
	s_mul_hi_u32 s1, s72, s50
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s72, s50
	s_andn2_b64 vcc, exec, s[78:79]
	v_lshl_add_u64 v[90:91], s[0:1], 1, v[86:87]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB2_287
; %bb.266:                              ;   in Loop: Header=BB2_115 Depth=3
	s_mov_b64 s[34:35], -1
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB2_289
; %bb.267:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v74, v106
	v_mov_b32_e32 v92, v145
	v_mov_b32_e32 v93, v146
	v_mov_b32_e32 v94, v0
                                        ; implicit-def: $sgpr92_sgpr93
                                        ; implicit-def: $sgpr94_sgpr95
	s_branch .LBB2_269
.LBB2_268:                              ; %Flow1443
                                        ;   in Loop: Header=BB2_269 Depth=4
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 s[70:71], exec, s[98:99]
	s_or_b64 s[34:35], s[70:71], s[34:35]
	s_andn2_b64 s[70:71], s[92:93], exec
	s_and_b64 s[92:93], s[94:95], exec
	s_or_b64 s[92:93], s[70:71], s[92:93]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB2_288
.LBB2_269:                              ; %.lr.ph.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ;       Parent Loop BB2_115 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v95, 0xf8, v74
	v_add_u32_e32 v98, 8, v95
	v_cmp_lt_i32_e64 s[96:97], s54, v98
	v_cmp_ge_i32_e32 vcc, s54, v98
	s_and_saveexec_b64 s[70:71], vcc
	s_cbranch_execz .LBB2_271
; %bb.270:                              ;   in Loop: Header=BB2_269 Depth=4
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
.LBB2_271:                              ; %Flow1442
                                        ;   in Loop: Header=BB2_269 Depth=4
	s_or_b64 exec, exec, s[70:71]
	s_mov_b64 s[98:99], -1
	s_andn2_b64 s[94:95], s[94:95], exec
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_268
; %bb.272:                              ; %.critedge.i
                                        ;   in Loop: Header=BB2_269 Depth=4
	v_add_u32_e32 v94, 0x200, v94
	v_cmp_le_i32_e32 vcc, s80, v94
	v_add_u32_e32 v93, 0x2000, v93
	v_add_u32_e32 v92, 16, v92
	v_add_u32_e32 v74, 0x1000, v74
	s_or_b64 s[94:95], s[94:95], exec
	s_orn2_b64 s[98:99], vcc, exec
	s_branch .LBB2_268
.LBB2_273:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v7
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v140
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_251
.LBB2_274:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v8
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v141
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_252
.LBB2_275:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v9
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v142
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:32
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_253
.LBB2_276:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_254
	s_branch .LBB2_255
.LBB2_277:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v3
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v140
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[96:97]
	s_cbranch_execz .LBB2_257
.LBB2_278:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v4
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v141
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[34:35]
	s_cbranch_execz .LBB2_258
.LBB2_279:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v74, v74, v5
	v_bfe_u32 v90, v74, 16, 1
	v_add3_u32 v74, v74, v90, s62
	v_subrev_u32_e32 v90, s98, v142
	v_lshl_add_u32 v90, v90, 9, v124
	ds_write_b16_d16_hi v90, v74 offset:64
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB2_259
.LBB2_280:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[0:1], s[92:93]
	s_cbranch_execnz .LBB2_260
	s_branch .LBB2_261
.LBB2_281:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v23
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v140
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB2_263
.LBB2_282:                              ;   in Loop: Header=BB2_115 Depth=3
	v_add_f32_e32 v90, v74, v24
	v_bfe_u32 v91, v90, 16, 1
	v_add3_u32 v90, v90, v91, s62
	v_subrev_u32_e32 v91, s98, v141
	v_lshl_add_u32 v91, v91, 9, v124
	ds_write_b16_d16_hi v91, v90 offset:96
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], s[34:35]
	s_cbranch_execnz .LBB2_264
	s_branch .LBB2_265
.LBB2_283:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_128
	s_branch .LBB2_129
.LBB2_284:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_168
	s_branch .LBB2_169
.LBB2_285:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_208
	s_branch .LBB2_209
.LBB2_286:                              ;   in Loop: Header=BB2_115 Depth=3
	v_mov_b32_e32 v74, 0
	s_and_saveexec_b64 s[70:71], s[92:93]
	s_cbranch_execnz .LBB2_248
	s_branch .LBB2_249
.LBB2_287:                              ;   in Loop: Header=BB2_115 Depth=3
	s_cbranch_execz .LBB2_114
	s_branch .LBB2_307
.LBB2_288:                              ; %Flow1444
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[34:35]
	s_orn2_b64 s[34:35], s[92:93], exec
.LBB2_289:                              ; %Flow1445
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v74, 0, 1, s[34:35]
	s_nop 0
	v_readfirstlane_b32 s0, v74
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[34:35], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB2_294
; %bb.290:                              ;   in Loop: Header=BB2_115 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB2_293
; %bb.291:                              ; %.lr.ph.i273.preheader
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v92, v97
	v_mov_b32_e32 v74, v0
.LBB2_292:                              ; %.lr.ph.i273
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ;       Parent Loop BB2_115 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v93, v74, v96
	v_mul_lo_u32 v94, v93, s55
	v_sub_u32_e32 v94, v74, v94
	v_add_u32_e32 v95, 1, v93
	v_subrev_u32_e32 v98, s55, v94
	v_cmp_le_u32_e32 vcc, s55, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v94, v94, v98, vcc
	v_add_u32_e32 v95, 1, v93
	v_cmp_le_u32_e32 vcc, s55, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_xor_b32_e32 v93, s65, v93
	v_subrev_u32_e32 v98, s65, v93
	v_mad_u64_u32 v[94:95], s[70:71], s26, v98, v[74:75]
	v_lshlrev_b32_e32 v93, 9, v93
	v_mul_lo_u32 v95, s3, v98
	v_sub_u32_e32 v93, v93, v95
	v_add_u32_e32 v93, v92, v93
	ds_read_u16 v93, v93
	v_mad_i64_i32 v[98:99], s[70:71], s72, v98, 0
	v_add_u32_e32 v74, 0x200, v74
	v_mov_b32_e32 v95, v75
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[90:91]
	v_cmp_le_i32_e32 vcc, s64, v74
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[98:99]
	v_add_u32_e32 v92, 0x400, v92
	s_or_b64 s[34:35], vcc, s[34:35]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v93
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB2_292
.LBB2_293:                              ; %Flow1434
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_294:                              ; %Flow1440
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_306
; %bb.295:                              ;   in Loop: Header=BB2_115 Depth=3
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB2_305
; %bb.296:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v98, v150
	v_mov_b64_e32 v[92:93], v[88:89]
	v_mov_b32_e32 v99, v0
	s_branch .LBB2_299
.LBB2_297:                              ; %Flow1436
                                        ;   in Loop: Header=BB2_299 Depth=4
	s_or_b64 exec, exec, s[94:95]
.LBB2_298:                              ; %.loopexit.i275
                                        ;   in Loop: Header=BB2_299 Depth=4
	s_or_b64 exec, exec, s[92:93]
	v_add_u32_e32 v99, 0x200, v99
	v_cmp_le_i32_e32 vcc, s80, v99
	v_lshl_add_u64 v[92:93], v[92:93], 0, s[88:89]
	s_or_b64 s[34:35], vcc, s[34:35]
	v_add_u32_e32 v98, 0x2000, v98
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB2_305
.LBB2_299:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ;       Parent Loop BB2_115 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB2_304 Depth 5
	v_lshlrev_b32_e32 v74, 3, v99
	v_and_b32_e32 v74, 0xf8, v74
	v_add_u32_e32 v94, 8, v74
	v_cmp_ge_i32_e32 vcc, s54, v94
	s_and_saveexec_b64 s[70:71], vcc
	s_xor_b64 s[92:93], exec, s[70:71]
	s_cbranch_execz .LBB2_301
; %bb.300:                              ;   in Loop: Header=BB2_299 Depth=4
	v_lshrrev_b32_e32 v94, 5, v99
	v_lshlrev_b32_e32 v74, 1, v74
	v_lshlrev_b32_e32 v95, 9, v94
	v_add3_u32 v95, 0, v95, v74
	ds_read_b128 v[100:103], v95
	v_mad_i64_i32 v[94:95], s[70:71], s72, v94, 0
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[90:91]
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[74:75]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[94:95], v[100:103]
                                        ; implicit-def: $vgpr74
.LBB2_301:                              ; %Flow1437
                                        ;   in Loop: Header=BB2_299 Depth=4
	s_andn2_saveexec_b64 s[92:93], s[92:93]
	s_cbranch_execz .LBB2_298
; %bb.302:                              ; %.preheader.i
                                        ;   in Loop: Header=BB2_299 Depth=4
	v_cmp_gt_i32_e32 vcc, s53, v74
	s_and_saveexec_b64 s[94:95], vcc
	s_cbranch_execz .LBB2_297
; %bb.303:                              ; %.lr.ph.i276
                                        ;   in Loop: Header=BB2_299 Depth=4
	s_mov_b32 s70, 0
	s_mov_b64 s[96:97], 0
	v_mov_b32_e32 v74, v98
	v_mov_b64_e32 v[94:95], v[92:93]
.LBB2_304:                              ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ;       Parent Loop BB2_115 Depth=3
                                        ;         Parent Loop BB2_299 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v100, v74
	s_add_i32 s71, s70, 1
	v_add_u32_e32 v101, s70, v148
	s_cmp_gt_u32 s70, 6
	v_cmp_le_u32_e32 vcc, s54, v101
	s_cselect_b64 s[98:99], -1, 0
	s_or_b64 s[98:99], s[98:99], vcc
	s_and_b64 s[98:99], exec, s[98:99]
	v_add_u32_e32 v74, 2, v74
	s_mov_b32 s70, s71
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v100
	s_or_b64 s[96:97], s[98:99], s[96:97]
	v_lshl_add_u64 v[94:95], v[94:95], 0, 2
	s_andn2_b64 exec, exec, s[96:97]
	s_cbranch_execnz .LBB2_304
	s_branch .LBB2_297
.LBB2_305:                              ; %Flow1439
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB2_306:                              ; %Flow1441
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_branch .LBB2_114
.LBB2_307:                              ;   in Loop: Header=BB2_115 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB2_113
; %bb.308:                              ; %.lr.ph.i280.preheader
                                        ;   in Loop: Header=BB2_115 Depth=3
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v92, v97
	v_mov_b32_e32 v74, v0
.LBB2_309:                              ; %.lr.ph.i280
                                        ;   Parent Loop BB2_85 Depth=1
                                        ;     Parent Loop BB2_111 Depth=2
                                        ;       Parent Loop BB2_115 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v93, v74, v96
	v_mul_lo_u32 v94, v93, s55
	v_sub_u32_e32 v94, v74, v94
	v_add_u32_e32 v95, 1, v93
	v_subrev_u32_e32 v98, s55, v94
	v_cmp_le_u32_e32 vcc, s55, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v94, v94, v98, vcc
	v_add_u32_e32 v95, 1, v93
	v_cmp_le_u32_e32 vcc, s55, v94
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_xor_b32_e32 v93, s65, v93
	v_subrev_u32_e32 v98, s65, v93
	v_mad_u64_u32 v[94:95], s[70:71], s26, v98, v[74:75]
	v_lshlrev_b32_e32 v93, 9, v93
	v_mul_lo_u32 v95, s3, v98
	v_sub_u32_e32 v93, v93, v95
	v_add_u32_e32 v93, v92, v93
	ds_read_u16 v93, v93
	v_mad_i64_i32 v[98:99], s[70:71], s72, v98, 0
	v_add_u32_e32 v74, 0x200, v74
	v_mov_b32_e32 v95, v75
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[90:91]
	v_cmp_le_i32_e32 vcc, s64, v74
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[98:99]
	v_add_u32_e32 v92, 0x400, v92
	s_or_b64 s[34:35], vcc, s[34:35]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[94:95], v93
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB2_309
	s_branch .LBB2_113
.LBB2_310:                              ; %._crit_edge638
                                        ;   in Loop: Header=BB2_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s26, v152, 6
	v_readlane_b32 s27, v152, 7
	s_and_b64 s[26:27], s[0:1], s[26:27]
	s_mov_b64 exec, s[26:27]
	s_cbranch_execz .LBB2_312
; %bb.311:                              ;   in Loop: Header=BB2_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB2_312:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s0, v152, 26
	v_readlane_b32 s1, v152, 27
	v_readlane_b32 s70, v152, 43
	s_and_b64 s[0:1], s[28:29], s[0:1]
	v_readlane_b32 s51, v152, 41
	v_readlane_b32 s53, v152, 42
	v_readlane_b32 s3, v152, 49
	v_readlane_b32 s34, v152, 47
	v_readlane_b32 s35, v152, 45
	v_readlane_b32 s71, v152, 44
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_83
; %bb.313:                              ; %.lr.ph640.preheader
                                        ;   in Loop: Header=BB2_85 Depth=1
	s_lshl_b32 s0, s3, 2
	v_readlane_b32 s26, v152, 4
	s_add_i32 s0, s26, s0
	s_sub_i32 s0, s0, s34
	s_sub_i32 s0, s0, s35
	s_lshl_b32 s1, s2, 2
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s2, s0, 7
	s_mov_b32 s3, s69
	v_readlane_b32 s27, v152, 5
	s_branch .LBB2_315
.LBB2_314:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB2_315 Depth=2
	s_add_i32 s3, s3, -1
	s_add_i32 s2, s2, s42
	s_cmp_lg_u32 s3, 0
	s_cbranch_scc0 .LBB2_83
.LBB2_315:                              ; %.lr.ph640
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
	s_cbranch_scc1 .LBB2_317
; %bb.316:                              ; %Flow1430
                                        ;   in Loop: Header=BB2_315 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB2_314
	s_branch .LBB2_318
.LBB2_317:                              ;   in Loop: Header=BB2_315 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB2_314
.LBB2_318:                              ;   in Loop: Header=BB2_315 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB2_314
.LBB2_319:                              ; %.critedge267
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
		.amdhsa_next_free_vgpr 153
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 156
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
	.set _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.num_vgpr, 153
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
; codeLenInByte = 16880
; TotalNumSgprs: 106
; NumVgprs: 153
; NumAgprs: 0
; TotalNumVgprs: 153
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 19
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 153
; AccumOffset: 156
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 38
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
                                        ; implicit-def: $vgpr232 : SGPR spill to VGPR lane
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
.LBB3_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit273
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
.LBB3_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit276
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
	v_writelane_b32 v232, s3, 2
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 31
	s_cmp_lg_u32 s20, 1
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v232, s7, 32
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 6
	s_cmp_lg_u32 s20, 2
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v232, s7, 7
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 8
	s_cmp_lg_u32 s20, 3
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v232, s7, 9
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 17
	s_cmp_lg_u32 s20, 4
	v_lshrrev_b32_e32 v18, 5, v0
	v_writelane_b32 v232, s7, 18
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 0
	s_cmp_lg_u32 s20, 5
	v_mov_b32_e32 v19, v21
	v_writelane_b32 v232, s7, 1
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 14
	s_cmp_lg_u32 s20, 6
	s_mov_b64 s[98:99], 0
	v_writelane_b32 v232, s7, 15
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 23
	s_cmp_lg_u32 s20, 7
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v232, s7, 24
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s89, s25
	v_cvt_f32_u32_e32 v2, s89
	s_sub_i32 s66, 0, s89
	s_ashr_i32 s3, s25, 31
	s_lshl_b64 s[82:83], s[50:51], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v232, s6, 25
	s_movk_i32 s91, 0x7fff
	s_mov_b32 s92, 0x7060302
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v232, s7, 26
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_readfirstlane_b32 s67, v2
	s_mul_i32 s66, s66, s67
	s_mul_hi_u32 s66, s67, s66
	s_add_i32 s90, s67, s66
	s_add_u32 s66, s48, 2
	s_addc_u32 s67, s49, 0
	v_writelane_b32 v232, s66, 27
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v232, s67, 28
	s_mul_i32 s66, s13, 14
	s_mul_hi_u32 s67, s12, 14
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 14
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	v_writelane_b32 v232, s66, 29
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v67, 0x100, v2
	v_writelane_b32 v232, s67, 30
	s_lshl_b64 s[66:67], s[12:13], 2
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	v_writelane_b32 v232, s66, 33
	s_nop 1
	v_writelane_b32 v232, s67, 34
	s_mul_i32 s66, s13, 12
	s_mul_hi_u32 s67, s12, 12
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 12
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	v_writelane_b32 v232, s66, 4
	s_nop 1
	v_writelane_b32 v232, s67, 5
	s_mul_i32 s66, s13, 6
	s_mul_hi_u32 s67, s12, 6
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 6
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	v_writelane_b32 v232, s66, 35
	s_nop 1
	v_writelane_b32 v232, s67, 36
	s_mul_i32 s66, s13, 10
	s_mul_hi_u32 s67, s12, 10
	s_add_i32 s67, s67, s66
	s_mul_i32 s66, s12, 10
	s_add_u32 s66, s18, s66
	s_addc_u32 s67, s19, s67
	s_add_u32 s66, s66, 2
	s_addc_u32 s67, s67, 0
	s_lshl_b64 s[12:13], s[12:13], 3
	v_writelane_b32 v232, s66, 12
	s_add_u32 s96, s18, s12
	s_addc_u32 s97, s19, s13
	v_writelane_b32 v232, s67, 13
	s_xor_b64 s[66:67], s[14:15], -1
	s_branch .LBB3_12
.LBB3_10:                               ; %Flow2105
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_sub_i32 s12, s16, s27
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s95
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB3_11:                               ; %Flow2117
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
; %bb.14:                               ; %.lr.ph.i.i.i278.preheader
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[72:73], 0
                                        ; implicit-def: $sgpr68_sgpr69
                                        ; implicit-def: $sgpr70_sgpr71
	s_branch .LBB3_16
.LBB3_15:                               ; %Flow2113
                                        ;   in Loop: Header=BB3_16 Depth=2
	s_and_b64 s[76:77], exec, s[70:71]
	s_or_b64 s[14:15], s[76:77], s[14:15]
	s_andn2_b64 s[68:69], s[68:69], exec
	s_and_b64 s[74:75], s[74:75], exec
	s_or_b64 s[68:69], s[68:69], s[74:75]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_18
.LBB3_16:                               ; %.lr.ph.i.i.i278
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
.LBB3_20:                               ; %.critedge697
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
.LBB3_23:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit284
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v67, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x4000000, v2
	v_cmp_eq_u32_e64 s[14:15], 0, v2
.LBB3_24:                               ; %Flow2116
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_and_saveexec_b64 s[68:69], s[14:15]
	s_cbranch_execz .LBB3_11
; %bb.25:                               ; %.critedge705
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB3_38
; %bb.26:                               ; %.lr.ph.i285.preheader
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
	v_readlane_b32 s14, v232, 27
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_readlane_b32 s15, v232, 28
	v_lshl_add_u64 v[22:23], s[18:19], 0, v[2:3]
	v_lshl_add_u64 v[26:27], s[52:53], 0, v[2:3]
	v_lshl_add_u64 v[24:25], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 29
	v_readlane_b32 s15, v232, 30
	v_lshl_add_u64 v[38:39], s[96:97], 0, v[2:3]
	s_mov_b64 s[74:75], 0
	v_lshl_add_u64 v[28:29], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 33
	v_readlane_b32 s15, v232, 34
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[30:31], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 4
	v_readlane_b32 s15, v232, 5
	s_nop 1
	v_lshl_add_u64 v[32:33], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 35
	v_readlane_b32 s15, v232, 36
	s_nop 1
	v_lshl_add_u64 v[34:35], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 12
	v_readlane_b32 s15, v232, 13
	s_nop 1
	v_lshl_add_u64 v[36:37], s[14:15], 0, v[2:3]
	s_branch .LBB3_28
.LBB3_27:                               ; %.critedge.i288
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
.LBB3_28:                               ; %.lr.ph.i285
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
; %bb.29:                               ; %.preheader.i287.preheader
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
.LBB3_30:                               ; %Flow2107
                                        ;   in Loop: Header=BB3_32 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_b64 s[84:85], s[86:87], exec
	s_or_b64 s[80:81], s[80:81], s[84:85]
.LBB3_31:                               ; %Flow2106
                                        ;   in Loop: Header=BB3_32 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[80:81]
	s_or_b64 s[78:79], s[14:15], s[78:79]
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execz .LBB3_35
.LBB3_32:                               ; %.preheader.i287
                                        ;   Parent Loop BB3_12 Depth=1
                                        ;     Parent Loop BB3_28 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[50:51], v[2:3]
	s_or_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB3_31
; %bb.33:                               ; %.preheader.i287.1
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
.LBB3_35:                               ; %Flow2108
                                        ;   in Loop: Header=BB3_28 Depth=2
	s_or_b64 exec, exec, s[78:79]
                                        ; implicit-def: $vgpr2_vgpr3
.LBB3_36:                               ; %Flow2109
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
.LBB3_38:                               ; %Flow2111
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[70:71], s[4:5]
	s_cbranch_execz .LBB3_10
; %bb.39:                               ; %.preheader711
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx4 v[2:5], v[2:3]
	s_mul_i32 s12, s24, s20
	s_add_i32 s12, s93, s12
	v_readlane_b32 s13, v232, 2
	s_mul_i32 s12, s12, s25
	s_add_i32 s13, s13, s94
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s38, s12
	s_addc_u32 s15, s39, s13
	v_mov_b32_e32 v6, s15
	v_readlane_b32 s12, v232, 31
	v_readlane_b32 s13, v232, 32
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
	v_readlane_b32 s12, v232, 6
	v_readlane_b32 s13, v232, 7
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
.LBB3_44:                               ; %Flow2103
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
	v_readlane_b32 s12, v232, 8
	v_readlane_b32 s13, v232, 9
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
.LBB3_48:                               ; %Flow2102
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
	v_readlane_b32 s12, v232, 17
	v_readlane_b32 s13, v232, 18
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
.LBB3_52:                               ; %Flow2101
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
	v_readlane_b32 s12, v232, 0
	v_readlane_b32 s13, v232, 1
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
.LBB3_56:                               ; %Flow2100
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
	v_readlane_b32 s12, v232, 14
	v_readlane_b32 s13, v232, 15
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
.LBB3_60:                               ; %Flow2099
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
	v_readlane_b32 s12, v232, 23
	v_readlane_b32 s13, v232, 24
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
.LBB3_64:                               ; %Flow2098
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
	v_readlane_b32 s12, v232, 25
	v_readlane_b32 s13, v232, 26
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
.LBB3_71:                               ; %Flow2121
	s_or_b64 exec, exec, s[46:47]
	s_load_dwordx2 s[16:17], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB3_72:                               ; %Flow2163
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB3_478
; %bb.73:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s42, s4
	s_addc_u32 s5, s43, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v232, s8, 0
	s_nop 1
	v_writelane_b32 v232, s9, 1
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
	v_writelane_b32 v232, s4, 2
	s_cmp_eq_u64 s[30:31], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_writelane_b32 v232, s5, 3
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
	s_branch .LBB3_478
.LBB3_79:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB3_478
.LBB3_80:                               ; %.critedge695
	s_lshr_b32 s3, s88, 24
	s_add_i32 s3, s21, s3
	s_ashr_i32 s48, s3, 8
	s_mul_i32 s3, s25, s48
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v232, s3, 4
	s_cbranch_scc1 .LBB3_478
; %bb.81:                               ; %.lr.ph824
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
	v_lshlrev_b32_e32 v160, 3, v0
	s_cselect_b32 s49, s5, s6
	v_and_b32_e32 v2, 24, v160
	s_cselect_b32 s0, s4, s7
	s_add_u32 s4, s49, 0x8000
	v_lshlrev_b32_e32 v5, 4, v0
	v_lshlrev_b32_e32 v161, 1, v2
	s_addc_u32 s0, s0, 0
	s_and_b32 s5, s4, -16
	v_and_b32_e32 v162, 0x1fc0, v5
	v_add_u32_e32 v5, s49, v161
	s_and_b32 s0, s4, 15
	s_add_u32 s5, s5, 16
	v_add_u32_e32 v9, v5, v162
	s_cmp_eq_u64 s[0:1], 0
	v_lshrrev_b32_e32 v10, 4, v9
	s_cselect_b32 s53, s4, s5
	s_add_i32 s0, s23, 31
	v_add_u32_e32 v8, 8, v5
	v_and_b32_e32 v10, 56, v10
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v163, v10, v9
	v_add_u32_e32 v9, v8, v162
	s_lshr_b32 s1, s1, 27
	v_lshrrev_b32_e32 v6, 2, v0
	v_lshrrev_b32_e32 v10, 4, v9
	v_or_b32_e32 v165, 0x2000, v162
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_or_b32_e32 v7, 0x80, v6
	v_and_b32_e32 v10, 56, v10
	v_add_u32_e32 v5, v5, v165
	s_ashr_i32 s88, s0, 5
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[130:131], s[0:1], v6, s12, v[2:3]
	v_mad_u64_u32 v[132:133], s[0:1], v7, s12, v[2:3]
	v_xor_b32_e32 v164, v10, v9
	v_lshrrev_b32_e32 v9, 4, v5
	v_mad_u64_u32 v[134:135], s[0:1], v6, s14, v[2:3]
	v_and_b32_e32 v9, 56, v9
	s_mov_b32 s0, s14
	v_xor_b32_e32 v166, v9, v5
	v_add_u32_e32 v5, v8, v165
	v_writelane_b32 v232, s0, 6
	v_lshrrev_b32_e32 v8, 4, v5
	v_and_b32_e32 v8, 56, v8
	v_writelane_b32 v232, s1, 7
	v_mad_u64_u32 v[136:137], s[0:1], v7, s14, v[2:3]
	v_add_u32_e32 v2, s53, v161
	v_add_u32_e32 v7, v2, v162
	v_xor_b32_e32 v167, v8, v5
	v_lshrrev_b32_e32 v8, 4, v7
	v_add_u32_e32 v5, 8, v2
	v_and_b32_e32 v8, 56, v8
	s_lshl_b32 s43, s25, 2
	v_xor_b32_e32 v168, v8, v7
	v_add_u32_e32 v7, v5, v162
	v_lshrrev_b32_e32 v8, 4, v7
	s_cmp_gt_i32 s23, 0
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v2, v2, v165
	s_cselect_b64 s[0:1], -1, 0
	v_xor_b32_e32 v169, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	v_writelane_b32 v232, s0, 8
	v_and_b32_e32 v7, 56, v7
	v_xor_b32_e32 v170, v7, v2
	v_writelane_b32 v232, s1, 9
	s_ashr_i32 s1, s16, 31
	s_mov_b32 s0, s16
	v_add_u32_e32 v2, v5, v165
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v5, 4, v2
	s_add_u32 s0, s38, s0
	v_and_b32_e32 v5, 56, v5
	s_addc_u32 s1, s39, s1
	v_xor_b32_e32 v171, v5, v2
	v_and_b32_e32 v2, 15, v0
	v_writelane_b32 v232, s0, 10
	v_lshrrev_b32_e32 v4, 8, v0
	v_lshlrev_b32_e32 v5, 6, v2
	v_writelane_b32 v232, s1, 11
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_lshl_or_b32 v172, v4, 13, v5
	v_lshl_or_b32 v174, v3, 12, v5
	v_writelane_b32 v232, s0, 12
	v_and_b32_e32 v5, 63, v0
	s_min_i32 s23, s26, 32
	s_bfe_i64 s[60:61], s[44:45], 0x200000
	v_writelane_b32 v232, s1, 13
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	s_cmp_gt_i32 s26, 0
	s_cselect_b64 s[62:63], -1, 0
	v_writelane_b32 v232, s0, 14
	s_cmp_lg_u64 s[28:29], 0
	s_cselect_b64 s[64:65], -1, 0
	v_writelane_b32 v232, s1, 15
	s_max_i32 s0, s22, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s17, 0
	s_cselect_b64 s[66:67], -1, 0
	s_abs_i32 s91, s43
	v_lshl_or_b32 v177, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s91
	s_abs_i32 s94, s26
	v_cvt_f32_u32_e32 v3, s94
	v_writelane_b32 v232, s0, 16
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
	v_writelane_b32 v232, s0, 17
	v_lshrrev_b32_e32 v207, 5, v0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v232, s1, 18
	s_sub_i32 s0, 0, s99
	v_and_b32_e32 v5, 12, v6
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s59, s1, s0
	s_cmp_gt_i32 s21, 0
	v_mad_i64_i32 v[2:3], s[0:1], v207, s44, 0
	s_cselect_b64 s[4:5], -1, 0
	v_readlane_b32 s0, v232, 0
	v_readlane_b32 s1, v232, 1
	v_writelane_b32 v232, s4, 19
	s_and_b64 s[0:1], s[0:1], s[4:5]
	v_lshl_or_b32 v178, v4, 7, v5
	v_writelane_b32 v232, s5, 20
	v_writelane_b32 v232, s0, 21
	v_and_b32_e32 v4, 31, v0
	v_lshlrev_b32_e32 v138, 4, v4
	v_writelane_b32 v232, s1, 22
	s_add_u32 s0, s50, 64
	v_writelane_b32 v232, s0, 23
	s_addc_u32 s0, s51, 0
	v_writelane_b32 v232, s0, 25
	s_add_u32 s0, s46, 64
	v_writelane_b32 v232, s0, 27
	s_addc_u32 s0, s47, 0
	v_mov_b32_e32 v139, 0
	v_writelane_b32 v232, s0, 29
	s_mov_b32 s0, s12
	v_lshl_add_u64 v[140:141], v[2:3], 1, v[138:139]
	v_lshlrev_b32_e32 v2, 9, v207
	v_writelane_b32 v232, s0, 31
	v_add_u32_e32 v216, 0, v2
	v_or_b32_e32 v2, v2, v138
	v_writelane_b32 v232, s1, 32
	s_lshl_b32 s0, s12, 8
	v_add_u32_e32 v217, 0, v2
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_lshrrev_b32_e32 v7, 1, v0
	v_writelane_b32 v232, s0, 33
	s_mul_i32 s0, s61, s23
	s_mul_hi_u32 s1, s44, s23
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_and_b32_e32 v173, 24, v7
	s_mul_i32 s90, s10, s8
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s44, s23
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v232, s43, 35
	s_mov_b64 s[54:55], 0
	v_ashrrev_i32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v137, 31, v136
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
	v_writelane_b32 v232, s52, 37
	v_writelane_b32 v232, s82, 38
	s_branch .LBB3_84
.LBB3_82:                               ; %Flow2124
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s2, s2, s27
	v_readlane_b32 s0, v232, 4
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	s_orn2_b64 s[0:1], s[0:1], exec
.LBB3_83:                               ; %Flow2157
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[76:77]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[54:55], s[0:1], s[54:55]
	s_andn2_b64 exec, exec, s[54:55]
	s_cbranch_execz .LBB3_478
.LBB3_84:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_87 Depth 2
                                        ;     Child Loop BB3_94 Depth 2
                                        ;     Child Loop BB3_106 Depth 2
                                        ;       Child Loop BB3_110 Depth 3
                                        ;         Child Loop BB3_468 Depth 4
                                        ;         Child Loop BB3_424 Depth 4
                                        ;         Child Loop BB3_451 Depth 4
                                        ;         Child Loop BB3_458 Depth 4
                                        ;           Child Loop BB3_463 Depth 5
                                        ;     Child Loop BB3_474 Depth 2
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
	v_mov_b32_e32 v69, v139
	v_mov_b32_e32 v68, v139
	v_mov_b32_e32 v67, v139
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
	v_readlane_b32 s0, v232, 31
	v_readlane_b32 s1, v232, 32
	s_mul_i32 s0, s70, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s46, s0
	s_addc_u32 s1, s47, s1
	v_lshl_add_u64 v[2:3], v[130:131], 1, s[0:1]
	v_lshl_add_u64 v[6:7], v[132:133], 1, s[0:1]
	s_lshl_b32 s71, s6, 8
	v_readlane_b32 s0, v232, 6
	v_readlane_b32 s1, v232, 7
	s_mul_i32 s0, s71, s0
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v163, v[2:3]

	;;#ASMEND
	s_add_u32 s12, s50, s0
	;;#ASMSTART
	ds_write_b64 v164, v[4:5]

	;;#ASMEND
	s_addc_u32 s13, s51, s1
	;;#ASMSTART
	ds_write_b64 v166, v[6:7]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[134:135], 1, s[12:13]
	;;#ASMSTART
	ds_write_b64 v167, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], v[136:137], 1, s[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v168, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v169, v[4:5]

	;;#ASMEND
	v_readlane_b32 s12, v232, 8
	;;#ASMSTART
	ds_write_b64 v170, v[6:7]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v171, v[8:9]

	;;#ASMEND
	v_readlane_b32 s13, v232, 9
	s_andn2_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v66, v139
	v_mov_b32_e32 v73, v139
	v_mov_b32_e32 v72, v139
	v_mov_b32_e32 v71, v139
	v_mov_b32_e32 v70, v139
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v4, v139
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v2, v139
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v8, v139
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v6, v139
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v12, v139
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v10, v139
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v16, v139
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v14, v139
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v20, v139
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v18, v139
	v_mov_b32_e32 v25, v139
	v_mov_b32_e32 v24, v139
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v22, v139
	v_mov_b32_e32 v29, v139
	v_mov_b32_e32 v28, v139
	v_mov_b32_e32 v27, v139
	v_mov_b32_e32 v26, v139
	v_mov_b32_e32 v33, v139
	v_mov_b32_e32 v32, v139
	v_mov_b32_e32 v31, v139
	v_mov_b32_e32 v30, v139
	v_mov_b32_e32 v37, v139
	v_mov_b32_e32 v36, v139
	v_mov_b32_e32 v35, v139
	v_mov_b32_e32 v34, v139
	v_mov_b32_e32 v41, v139
	v_mov_b32_e32 v40, v139
	v_mov_b32_e32 v39, v139
	v_mov_b32_e32 v38, v139
	v_mov_b32_e32 v45, v139
	v_mov_b32_e32 v44, v139
	v_mov_b32_e32 v43, v139
	v_mov_b32_e32 v42, v139
	v_mov_b32_e32 v49, v139
	v_mov_b32_e32 v48, v139
	v_mov_b32_e32 v47, v139
	v_mov_b32_e32 v46, v139
	v_mov_b32_e32 v53, v139
	v_mov_b32_e32 v52, v139
	v_mov_b32_e32 v51, v139
	v_mov_b32_e32 v50, v139
	v_mov_b32_e32 v57, v139
	v_mov_b32_e32 v56, v139
	v_mov_b32_e32 v55, v139
	v_mov_b32_e32 v54, v139
	v_mov_b32_e32 v61, v139
	v_mov_b32_e32 v60, v139
	v_mov_b32_e32 v59, v139
	v_mov_b32_e32 v58, v139
	v_mov_b32_e32 v65, v139
	v_mov_b32_e32 v64, v139
	v_mov_b32_e32 v63, v139
	v_mov_b32_e32 v62, v139
	v_mov_b32_e32 v77, v139
	v_mov_b32_e32 v76, v139
	v_mov_b32_e32 v75, v139
	v_mov_b32_e32 v74, v139
	v_mov_b32_e32 v81, v139
	v_mov_b32_e32 v80, v139
	v_mov_b32_e32 v79, v139
	v_mov_b32_e32 v78, v139
	v_mov_b32_e32 v85, v139
	v_mov_b32_e32 v84, v139
	v_mov_b32_e32 v83, v139
	v_mov_b32_e32 v82, v139
	v_mov_b32_e32 v89, v139
	v_mov_b32_e32 v88, v139
	v_mov_b32_e32 v87, v139
	v_mov_b32_e32 v86, v139
	v_mov_b32_e32 v93, v139
	v_mov_b32_e32 v92, v139
	v_mov_b32_e32 v91, v139
	v_mov_b32_e32 v90, v139
	v_mov_b32_e32 v97, v139
	v_mov_b32_e32 v96, v139
	v_mov_b32_e32 v95, v139
	v_mov_b32_e32 v94, v139
	v_mov_b32_e32 v101, v139
	v_mov_b32_e32 v100, v139
	v_mov_b32_e32 v99, v139
	v_mov_b32_e32 v98, v139
	v_mov_b32_e32 v105, v139
	v_mov_b32_e32 v104, v139
	v_mov_b32_e32 v103, v139
	v_mov_b32_e32 v102, v139
	v_mov_b32_e32 v109, v139
	v_mov_b32_e32 v108, v139
	v_mov_b32_e32 v107, v139
	v_mov_b32_e32 v106, v139
	v_mov_b32_e32 v113, v139
	v_mov_b32_e32 v112, v139
	v_mov_b32_e32 v111, v139
	v_mov_b32_e32 v110, v139
	v_mov_b32_e32 v117, v139
	v_mov_b32_e32 v116, v139
	v_mov_b32_e32 v115, v139
	v_mov_b32_e32 v114, v139
	v_mov_b32_e32 v121, v139
	v_mov_b32_e32 v120, v139
	v_mov_b32_e32 v119, v139
	v_mov_b32_e32 v118, v139
	v_mov_b32_e32 v125, v139
	v_mov_b32_e32 v124, v139
	v_mov_b32_e32 v123, v139
	v_mov_b32_e32 v122, v139
	v_mov_b32_e32 v129, v139
	v_mov_b32_e32 v128, v139
	v_mov_b32_e32 v127, v139
	v_mov_b32_e32 v126, v139
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB3_89
; %bb.85:                               ; %.lr.ph782.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s3, v232, 23
	s_add_u32 s0, s3, s0
	v_readlane_b32 s3, v232, 25
	s_addc_u32 s1, s3, s1
	s_lshl_b32 s10, s45, 2
	s_add_i32 s10, s2, s10
	s_sub_i32 s10, s10, s4
	s_sub_i32 s10, s10, s5
	s_lshl_b32 s12, s7, 2
	s_sub_i32 s10, s10, s12
	v_readlane_b32 s3, v232, 33
	s_mul_i32 s12, s3, s10
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 1
	v_readlane_b32 s3, v232, 27
	s_add_u32 s12, s3, s12
	v_readlane_b32 s3, v232, 29
	v_mov_b32_e32 v66, 0
	s_addc_u32 s13, s3, s13
	s_mov_b32 s10, 0
	v_mov_b32_e32 v67, v66
	v_mov_b32_e32 v68, v66
	v_mov_b32_e32 v69, v66
	v_mov_b32_e32 v70, v66
	v_mov_b32_e32 v71, v66
	v_mov_b32_e32 v72, v66
	v_mov_b32_e32 v73, v66
	v_mov_b32_e32 v2, v66
	v_mov_b32_e32 v3, v66
	v_mov_b32_e32 v4, v66
	v_mov_b32_e32 v5, v66
	v_mov_b32_e32 v6, v66
	v_mov_b32_e32 v7, v66
	v_mov_b32_e32 v8, v66
	v_mov_b32_e32 v9, v66
	v_mov_b32_e32 v10, v66
	v_mov_b32_e32 v11, v66
	v_mov_b32_e32 v12, v66
	v_mov_b32_e32 v13, v66
	v_mov_b32_e32 v14, v66
	v_mov_b32_e32 v15, v66
	v_mov_b32_e32 v16, v66
	v_mov_b32_e32 v17, v66
	v_mov_b32_e32 v18, v66
	v_mov_b32_e32 v19, v66
	v_mov_b32_e32 v20, v66
	v_mov_b32_e32 v21, v66
	v_mov_b32_e32 v22, v66
	v_mov_b32_e32 v23, v66
	v_mov_b32_e32 v24, v66
	v_mov_b32_e32 v25, v66
	v_mov_b32_e32 v26, v66
	v_mov_b32_e32 v27, v66
	v_mov_b32_e32 v28, v66
	v_mov_b32_e32 v29, v66
	v_mov_b32_e32 v30, v66
	v_mov_b32_e32 v31, v66
	v_mov_b32_e32 v32, v66
	v_mov_b32_e32 v33, v66
	v_mov_b32_e32 v34, v66
	v_mov_b32_e32 v35, v66
	v_mov_b32_e32 v36, v66
	v_mov_b32_e32 v37, v66
	v_mov_b32_e32 v38, v66
	v_mov_b32_e32 v39, v66
	v_mov_b32_e32 v40, v66
	v_mov_b32_e32 v41, v66
	v_mov_b32_e32 v42, v66
	v_mov_b32_e32 v43, v66
	v_mov_b32_e32 v44, v66
	v_mov_b32_e32 v45, v66
	v_mov_b32_e32 v46, v66
	v_mov_b32_e32 v47, v66
	v_mov_b32_e32 v48, v66
	v_mov_b32_e32 v49, v66
	v_mov_b32_e32 v50, v66
	v_mov_b32_e32 v51, v66
	v_mov_b32_e32 v52, v66
	v_mov_b32_e32 v53, v66
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v55, v66
	v_mov_b32_e32 v56, v66
	v_mov_b32_e32 v57, v66
	v_mov_b32_e32 v58, v66
	v_mov_b32_e32 v59, v66
	v_mov_b32_e32 v60, v66
	v_mov_b32_e32 v61, v66
	v_mov_b32_e32 v62, v66
	v_mov_b32_e32 v63, v66
	v_mov_b32_e32 v64, v66
	v_mov_b32_e32 v65, v66
	v_mov_b32_e32 v74, v66
	v_mov_b32_e32 v75, v66
	v_mov_b32_e32 v76, v66
	v_mov_b32_e32 v77, v66
	v_mov_b32_e32 v78, v66
	v_mov_b32_e32 v79, v66
	v_mov_b32_e32 v80, v66
	v_mov_b32_e32 v81, v66
	v_mov_b32_e32 v82, v66
	v_mov_b32_e32 v83, v66
	v_mov_b32_e32 v84, v66
	v_mov_b32_e32 v85, v66
	v_mov_b32_e32 v86, v66
	v_mov_b32_e32 v87, v66
	v_mov_b32_e32 v88, v66
	v_mov_b32_e32 v89, v66
	v_mov_b32_e32 v90, v66
	v_mov_b32_e32 v91, v66
	v_mov_b32_e32 v92, v66
	v_mov_b32_e32 v93, v66
	v_mov_b32_e32 v94, v66
	v_mov_b32_e32 v95, v66
	v_mov_b32_e32 v96, v66
	v_mov_b32_e32 v97, v66
	v_mov_b32_e32 v98, v66
	v_mov_b32_e32 v99, v66
	v_mov_b32_e32 v100, v66
	v_mov_b32_e32 v101, v66
	v_mov_b32_e32 v102, v66
	v_mov_b32_e32 v103, v66
	v_mov_b32_e32 v104, v66
	v_mov_b32_e32 v105, v66
	v_mov_b32_e32 v106, v66
	v_mov_b32_e32 v107, v66
	v_mov_b32_e32 v108, v66
	v_mov_b32_e32 v109, v66
	v_mov_b32_e32 v110, v66
	v_mov_b32_e32 v111, v66
	v_mov_b32_e32 v112, v66
	v_mov_b32_e32 v113, v66
	v_mov_b32_e32 v114, v66
	v_mov_b32_e32 v115, v66
	v_mov_b32_e32 v116, v66
	v_mov_b32_e32 v117, v66
	v_mov_b32_e32 v118, v66
	v_mov_b32_e32 v119, v66
	v_mov_b32_e32 v120, v66
	v_mov_b32_e32 v121, v66
	v_mov_b32_e32 v122, v66
	v_mov_b32_e32 v123, v66
	v_mov_b32_e32 v124, v66
	v_mov_b32_e32 v125, v66
	v_mov_b32_e32 v126, v66
	v_mov_b32_e32 v127, v66
	v_mov_b32_e32 v128, v66
	v_mov_b32_e32 v129, v66
	s_add_i32 s14, s10, 1
	s_cmp_ge_i32 s14, s88
	s_cbranch_scc1 .LBB3_87
.LBB3_86:                               ;   in Loop: Header=BB3_84 Depth=1
	s_lshl_b32 s15, s14, 14
	s_and_b32 s15, s15, 0x4000
	s_add_i32 s16, s49, s15
	v_add_u32_e32 v138, s16, v161
	v_add_u32_e32 v151, v138, v162
	v_lshl_add_u64 v[142:143], v[130:131], 1, s[12:13]
	s_or_b32 s17, s16, 8
	v_lshrrev_b32_e32 v152, 4, v151
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
	v_lshl_add_u64 v[146:147], v[132:133], 1, s[12:13]
	v_add_u32_e32 v150, s17, v161
	v_and_b32_e32 v152, 56, v152
	;;#ASMSTART
	global_load_dwordx4 v[146:149], v[146:147], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v151, v152, v151
	;;#ASMSTART
	ds_write_b64 v151, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v142, v150, v162
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v142, v143, v142
	v_add_u32_e32 v138, v138, v165
	;;#ASMSTART
	ds_write_b64 v142, v[144:145]

	;;#ASMEND
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[146:147]

	;;#ASMEND
	v_add_u32_e32 v138, v150, v165
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	s_add_i32 s15, s53, s15
	;;#ASMSTART
	ds_write_b64 v138, v[148:149]

	;;#ASMEND
	v_add_u32_e32 v138, s15, v161
	v_add_u32_e32 v151, v138, v162
	v_lshl_add_u64 v[142:143], v[134:135], 1, s[0:1]
	s_or_b32 s16, s15, 8
	v_lshrrev_b32_e32 v152, 4, v151
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
	v_lshl_add_u64 v[146:147], v[136:137], 1, s[0:1]
	v_add_u32_e32 v150, s16, v161
	v_and_b32_e32 v152, 56, v152
	;;#ASMSTART
	global_load_dwordx4 v[146:149], v[146:147], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v151, v152, v151
	;;#ASMSTART
	ds_write_b64 v151, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v142, v150, v162
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v142, v143, v142
	v_add_u32_e32 v138, v138, v165
	;;#ASMSTART
	ds_write_b64 v142, v[144:145]

	;;#ASMEND
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[146:147]

	;;#ASMEND
	v_add_u32_e32 v138, v150, v165
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[148:149]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB3_87:                               ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_lshl_b32 s10, s10, 14
	s_and_b32 s10, s10, 0x4000
	s_add_i32 s15, s49, s10
	v_add_u32_e32 v138, s15, v172
	v_add_u32_e32 v142, v138, v173
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v156, v143, v142
	;;#ASMSTART
	ds_read_b64 v[142:143], v156 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v156 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v156 offset:0x800

	;;#ASMEND
	s_add_i32 s10, s53, s10
	;;#ASMSTART
	ds_read_b64 v[148:149], v156 offset:0xc00

	;;#ASMEND
	v_add_u32_e32 v228, s10, v174
	;;#ASMSTART
	ds_read_b64 v[150:151], v156 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v156 offset:0x1400

	;;#ASMEND
	v_add_u32_e32 v158, v228, v173
	;;#ASMSTART
	ds_read_b64 v[154:155], v156 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v158
	;;#ASMSTART
	ds_read_b64 v[156:157], v156 offset:0x1c00

	;;#ASMEND
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v226, v159, v158
	;;#ASMSTART
	ds_read_b64 v[158:159], v226 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v226 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v226 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[226:227], v226 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_add_u32_e32 v138, v138, v219
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
	s_add_u32 s0, s0, 64
	v_mfma_f32_16x16x16_bf16 v[126:129], v[142:143], v[158:159], v[126:129]
	s_addc_u32 s1, s1, 0
	s_add_u32 s12, s12, 64
	s_addc_u32 s13, s13, 0
	v_mfma_f32_16x16x16_bf16 v[122:125], v[142:143], v[222:223], v[122:125]
	s_cmp_eq_u32 s93, s14
	v_mfma_f32_16x16x16_bf16 v[118:121], v[142:143], v[224:225], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[142:143], v[226:227], v[114:117]
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_read_b64 v[142:143], v138 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[110:113], v[144:145], v[158:159], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[144:145], v[222:223], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[144:145], v[224:225], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[144:145], v[226:227], v[98:101]
	;;#ASMSTART
	ds_read_b64 v[144:145], v138 offset:0x400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[94:97], v[146:147], v[158:159], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[146:147], v[222:223], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[146:147], v[224:225], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[146:147], v[226:227], v[82:85]
	;;#ASMSTART
	ds_read_b64 v[146:147], v138 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[78:81], v[148:149], v[158:159], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[148:149], v[222:223], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[148:149], v[224:225], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[148:149], v[226:227], v[58:61]
	;;#ASMSTART
	ds_read_b64 v[148:149], v138 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[54:57], v[150:151], v[158:159], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[150:151], v[222:223], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[150:151], v[224:225], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[150:151], v[226:227], v[42:45]
	;;#ASMSTART
	ds_read_b64 v[150:151], v138 offset:0x1000

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[38:41], v[152:153], v[158:159], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[152:153], v[222:223], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[152:153], v[224:225], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[152:153], v[226:227], v[26:29]
	;;#ASMSTART
	ds_read_b64 v[152:153], v138 offset:0x1400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[22:25], v[154:155], v[158:159], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[154:155], v[222:223], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[154:155], v[224:225], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[154:155], v[226:227], v[10:13]
	;;#ASMSTART
	ds_read_b64 v[154:155], v138 offset:0x1800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[6:9], v[156:157], v[158:159], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[156:157], v[222:223], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[156:157], v[224:225], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[156:157], v[226:227], v[66:69]
	;;#ASMSTART
	ds_read_b64 v[156:157], v138 offset:0x1c00

	;;#ASMEND
	v_add_u32_e32 v138, v228, v219
	v_lshrrev_b32_e32 v158, 4, v138
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v138, v158, v138
	;;#ASMSTART
	ds_read_b64 v[158:159], v138 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v138 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v138 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[226:227], v138 offset:0xc00

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
	v_mfma_f32_16x16x16_bf16 v[126:129], v[142:143], v[158:159], v[126:129]
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[122:125], v[142:143], v[222:223], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[142:143], v[224:225], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[142:143], v[226:227], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[144:145], v[158:159], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[144:145], v[222:223], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[144:145], v[224:225], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[144:145], v[226:227], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[146:147], v[158:159], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[146:147], v[222:223], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[146:147], v[224:225], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[146:147], v[226:227], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[148:149], v[158:159], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[148:149], v[222:223], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[148:149], v[224:225], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[148:149], v[226:227], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[150:151], v[158:159], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[150:151], v[222:223], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[150:151], v[224:225], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[150:151], v[226:227], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[152:153], v[158:159], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[152:153], v[222:223], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[152:153], v[224:225], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[152:153], v[226:227], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[154:155], v[158:159], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[154:155], v[222:223], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[154:155], v[224:225], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[154:155], v[226:227], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[156:157], v[158:159], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[156:157], v[222:223], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[156:157], v[224:225], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[156:157], v[226:227], v[66:69]
	s_cbranch_scc1 .LBB3_89
; %bb.88:                               ;   in Loop: Header=BB3_87 Depth=2
	s_mov_b32 s10, s14
	s_add_i32 s14, s10, 1
	s_cmp_ge_i32 s14, s88
	s_cbranch_scc0 .LBB3_86
	s_branch .LBB3_87
.LBB3_89:                               ; %Flow2153
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s12, v232, 17
	v_readlane_b32 s13, v232, 18
	s_and_b64 s[12:13], s[0:1], s[12:13]
	s_mov_b64 exec, s[12:13]
	s_cbranch_execz .LBB3_98
; %bb.90:                               ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s12, v232, 12
	v_readlane_b32 s13, v232, 13
	s_and_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB3_98
; %bb.91:                               ;   in Loop: Header=BB3_84 Depth=1
	v_add_u32_e32 v138, s70, v175
	v_sub_u32_e32 v143, 0, v138
	v_max_i32_e32 v143, v138, v143
	v_mul_hi_u32 v144, v143, s59
	v_mul_lo_u32 v145, v144, s99
	v_sub_u32_e32 v143, v143, v145
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s99, v143
	v_ashrrev_i32_e32 v142, 31, v138
	v_xor_b32_e32 v142, s58, v142
	v_cndmask_b32_e32 v144, v144, v145, vcc
	v_subrev_u32_e32 v145, s99, v143
	v_cndmask_b32_e32 v143, v143, v145, vcc
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s99, v143
	s_nop 1
	v_cndmask_b32_e32 v143, v144, v145, vcc
	v_xor_b32_e32 v143, v143, v142
	v_sub_u32_e32 v142, v143, v142
	v_mul_lo_u32 v143, v142, s33
	v_sub_u32_e32 v138, v138, v143
	v_sub_u32_e32 v144, 0, v138
	v_ashrrev_i32_e32 v143, 31, v138
	v_max_i32_e32 v138, v138, v144
	v_mul_hi_u32 v144, v138, s98
	v_mul_lo_u32 v145, v144, s94
	v_sub_u32_e32 v138, v138, v145
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s94, v138
	v_xor_b32_e32 v143, s97, v143
	s_nop 0
	v_cndmask_b32_e32 v144, v144, v145, vcc
	v_subrev_u32_e32 v145, s94, v138
	v_cndmask_b32_e32 v138, v138, v145, vcc
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s94, v138
	s_nop 1
	v_cndmask_b32_e32 v138, v144, v145, vcc
	v_xor_b32_e32 v138, v138, v143
	v_sub_u32_e32 v138, v138, v143
	v_mad_u64_u32 v[142:143], s[12:13], v142, s24, v[138:139]
	v_mul_lo_u32 v138, v142, s25
	v_add_u32_e32 v142, s6, v138
	v_readlane_b32 s12, v232, 10
	v_ashrrev_i32_e32 v143, 31, v142
	v_readlane_b32 s13, v232, 11
	s_nop 1
	v_lshl_add_u64 v[142:143], v[142:143], 2, s[12:13]
	flat_load_dword v138, v[142:143] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v138, v176
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB3_98
; %bb.92:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_mov_b64 s[12:13], 0
	s_mov_b64 s[78:79], 0
                                        ; implicit-def: $sgpr14_sgpr15
                                        ; implicit-def: $sgpr76_sgpr77
	s_branch .LBB3_94
.LBB3_93:                               ; %Flow2148
                                        ;   in Loop: Header=BB3_94 Depth=2
	s_and_b64 s[16:17], exec, s[76:77]
	s_or_b64 s[12:13], s[16:17], s[12:13]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], s[80:81], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB3_96
.LBB3_94:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s78, s78, 1
	s_addc_u32 s79, s79, 0
	v_mov_b64_e32 v[144:145], s[34:35]
	v_cmp_gt_u64_e32 vcc, s[78:79], v[144:145]
	s_mov_b64 s[80:81], -1
	s_or_b64 s[76:77], s[76:77], exec
	s_cbranch_vccnz .LBB3_93
; %bb.95:                               ;   in Loop: Header=BB3_94 Depth=2
	s_sleep 4
	flat_load_dword v138, v[142:143] offset:256 sc0 sc1
	s_andn2_b64 s[16:17], s[76:77], exec
	s_mov_b64 s[80:81], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v138, v176
	s_and_b64 s[56:57], vcc, exec
	s_or_b64 s[76:77], s[16:17], s[56:57]
	s_branch .LBB3_93
.LBB3_96:                               ; %loop.exit.guard2096
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execz .LBB3_98
; %bb.97:                               ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b64_e32 v[142:143], s[30:31]
	flat_atomic_or v[142:143], v221
.LBB3_98:                               ; %.critedge696
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s12, v232, 2
	v_readlane_b32 s13, v232, 3
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB3_102
; %bb.99:                               ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b32_e32 v138, 0
	s_mov_b64 s[12:13], exec
	v_readlane_b32 s14, v232, 14
	v_readlane_b32 s15, v232, 15
	s_and_b64 s[14:15], s[12:13], s[14:15]
	s_mov_b64 exec, s[14:15]
	s_cbranch_execz .LBB3_101
; %bb.100:                              ;   in Loop: Header=BB3_84 Depth=1
	v_mov_b64_e32 v[142:143], s[30:31]
	flat_load_dword v138, v[142:143] sc1
.LBB3_101:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit262
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v138, v218, v138
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v138, 0x2000000, v138
	v_cmp_eq_u32_e64 s[12:13], 0, v138
.LBB3_102:                              ; %Flow2156
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_and_saveexec_b64 s[76:77], s[12:13]
	s_cbranch_execz .LBB3_83
; %bb.103:                              ; %.critedge704
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_readlane_b32 s0, v232, 19
	v_readlane_b32 s1, v232, 20
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB3_469
; %bb.104:                              ; %.lr.ph819
                                        ;   in Loop: Header=BB3_84 Depth=1
	v_or_b32_e32 v138, s71, v177
	v_readlane_b32 s0, v232, 16
	s_sub_i32 s16, s22, s71
	v_cmp_gt_i32_e32 vcc, s22, v138
	v_mov_b32_e32 v148, s0
	s_min_i32 s17, s16, 0x100
	v_cndmask_b32_e32 v142, v148, v138, vcc
	v_or_b32_e32 v138, s71, v183
	v_cmp_gt_i32_e32 vcc, s22, v138
	s_abs_i32 s56, s17
	v_cvt_f32_u32_e32 v150, s56
	v_cndmask_b32_e32 v144, v148, v138, vcc
	v_or_b32_e32 v138, s71, v184
	v_cmp_gt_i32_e32 vcc, s22, v138
	s_ashr_i32 s95, s17, 31
	s_sub_i32 s0, 0, s56
	v_cndmask_b32_e32 v146, v148, v138, vcc
	v_or_b32_e32 v138, s71, v185
	v_cmp_gt_i32_e32 vcc, s22, v138
	s_lshl_b32 s1, s11, 8
	v_ashrrev_i32_e32 v143, 31, v142
	v_cndmask_b32_e32 v148, v148, v138, vcc
	v_rcp_iflag_f32_e32 v138, v150
	v_ashrrev_i32_e32 v145, 31, v144
	v_ashrrev_i32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v149, 31, v148
	v_mul_f32_e32 v138, 0x4f7ffffe, v138
	v_cvt_u32_f32_e32 v138, v138
	s_mul_i32 s57, s17, s23
	v_lshl_add_u64 v[142:143], v[142:143], 1, s[28:29]
	v_lshl_add_u64 v[144:145], v[144:145], 1, s[28:29]
	v_mul_lo_u32 v150, s0, v138
	s_lshl_b32 s0, s95, 9
	v_subrev_u32_e32 v223, s0, v220
	s_lshl_b32 s0, s42, 8
	s_sub_i32 s11, s0, s1
	s_lshl_b32 s0, s45, 2
	s_add_i32 s0, s2, s0
	s_sub_i32 s0, s0, s4
	s_sub_i32 s0, s0, s5
	s_lshl_b32 s1, s7, 2
	s_sub_i32 s0, s0, s1
	v_mul_hi_u32 v150, v138, v150
	s_lshl_b32 s0, s0, 8
	v_lshl_add_u64 v[146:147], v[146:147], 1, s[28:29]
	v_lshl_add_u64 v[148:149], v[148:149], 1, s[28:29]
	v_cmp_gt_i32_e64 s[12:13], s57, v0
	s_mov_b32 s96, 0
	v_add_u32_e32 v222, v138, v150
	s_lshl_b32 s89, s17, 1
	s_sub_i32 s10, 0, s17
	s_add_i32 s52, s90, s0
	s_branch .LBB3_106
.LBB3_105:                              ; %._crit_edge817
                                        ;   in Loop: Header=BB3_106 Depth=2
	s_add_i32 s96, s96, 1
	s_add_i32 s52, s52, s26
	s_cmp_eq_u32 s96, s21
	s_cbranch_scc1 .LBB3_469
.LBB3_106:                              ;   Parent Loop BB3_84 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_110 Depth 3
                                        ;         Child Loop BB3_468 Depth 4
                                        ;         Child Loop BB3_424 Depth 4
                                        ;         Child Loop BB3_451 Depth 4
                                        ;         Child Loop BB3_458 Depth 4
                                        ;           Child Loop BB3_463 Depth 5
	s_andn2_b64 vcc, exec, s[62:63]
	s_cbranch_vccnz .LBB3_105
; %bb.107:                              ; %.lr.ph816.preheader
                                        ;   in Loop: Header=BB3_106 Depth=2
	v_mov_b64_e32 v[158:159], s[36:37]
	flat_load_dwordx4 v[150:153], v[158:159]
	flat_load_dwordx4 v[154:157], v[158:159] offset:16
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
	v_cndmask_b32_e32 v138, 0, v152, vcc
	v_cndmask_b32_e32 v152, 0, v153, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 2
	v_cndmask_b32_e32 v152, v152, v155, vcc
	v_cndmask_b32_e32 v138, v138, v154, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 3
	v_cndmask_b32_e32 v138, v138, v156, vcc
	v_cndmask_b32_e32 v152, v152, v157, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 4
	v_cndmask_b32_e32 v152, v152, v225, vcc
	v_cndmask_b32_e32 v138, v138, v224, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 5
	v_cndmask_b32_e32 v138, v138, v226, vcc
	v_cndmask_b32_e32 v152, v152, v227, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 6
	v_cndmask_b32_e32 v152, v152, v229, vcc
	v_cndmask_b32_e32 v138, v138, v228, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s14, 7
	v_cndmask_b32_e32 v138, v138, v230, vcc
	v_cndmask_b32_e32 v152, v152, v231, vcc
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
	v_cndmask_b32_e32 v153, v152, v159, vcc
	v_cndmask_b32_e32 v152, v138, v158, vcc
	v_sub_co_u32_e32 v150, vcc, s18, v150
	v_mov_b32_e32 v138, s19
	s_add_i32 s1, s1, s14
	s_sub_i32 s0, s78, s0
	v_subb_co_u32_e32 v151, vcc, v138, v151, vcc
	s_mul_i32 s1, s1, s44
	s_sub_i32 s0, s0, s79
	v_lshl_add_u64 v[150:151], v[150:151], 0, v[152:153]
	v_cmp_ne_u64_e32 vcc, 0, v[152:153]
	s_add_i32 s14, s1, s71
	s_mul_i32 s0, s44, s0
	v_cndmask_b32_e32 v153, 0, v151, vcc
	v_cndmask_b32_e32 v152, 0, v150, vcc
	s_ashr_i32 s15, s14, 31
	s_add_i32 s0, s11, s0
	v_lshl_add_u64 v[150:151], s[14:15], 1, v[152:153]
	v_lshl_add_u64 v[152:153], v[152:153], 0, v[140:141]
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[152:153], s[0:1], 1, v[152:153]
	s_branch .LBB3_110
.LBB3_108:                              ; %Flow2140
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB3_109:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_add_i32 s43, s43, s23
	s_cmp_ge_i32 s43, s26
	v_lshl_add_u64 v[152:153], v[152:153], 0, s[72:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB3_105
.LBB3_110:                              ; %.lr.ph816
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB3_468 Depth 4
                                        ;         Child Loop BB3_424 Depth 4
                                        ;         Child Loop BB3_451 Depth 4
                                        ;         Child Loop BB3_458 Depth 4
                                        ;           Child Loop BB3_463 Depth 5
	v_cndmask_b32_e64 v138, 0, 1, s[64:65]
	v_cmp_ne_u32_e64 s[14:15], 1, v138
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB3_112
; %bb.111:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_113
.LBB3_112:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_113:                              ;   in Loop: Header=BB3_110 Depth=3
	s_add_i32 s86, s43, s42
	s_add_i32 s87, s86, s23
	v_cmp_le_i32_e32 vcc, s86, v178
	v_cmp_gt_i32_e64 s[0:1], s87, v178
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_115
; %bb.114:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v126
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v178
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_115:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v180
	v_cmp_gt_i32_e64 s[0:1], s87, v180
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_117
; %bb.116:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v127
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v180
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_117:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v181
	v_cmp_gt_i32_e64 s[0:1], s87, v181
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_119
; %bb.118:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v128
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v181
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_119:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v182
	v_cmp_gt_i32_e64 s[0:1], s87, v182
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_121
; %bb.120:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v129
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v182
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_121:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_438
; %bb.122:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_124
.LBB3_123:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v122
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v178
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_124:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_141
; %bb.125:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_142
.LBB3_126:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_143
.LBB3_127:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_144
.LBB3_128:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_130
.LBB3_129:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v118
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v178
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_130:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_145
; %bb.131:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_146
.LBB3_132:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_147
.LBB3_133:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_148
.LBB3_134:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_136
.LBB3_135:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v114
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v178
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_136:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_149
; %bb.137:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_150
.LBB3_138:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_151
.LBB3_139:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_152
.LBB3_140:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_153
.LBB3_141:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v123
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v180
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_126
.LBB3_142:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v124
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v181
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_127
.LBB3_143:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v125
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v182
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_128
.LBB3_144:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_129
	s_branch .LBB3_130
.LBB3_145:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v119
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v180
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_132
.LBB3_146:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v120
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v181
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_133
.LBB3_147:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v121
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v182
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_134
.LBB3_148:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_135
	s_branch .LBB3_136
.LBB3_149:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v115
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v180
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_138
.LBB3_150:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v116
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v181
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_139
.LBB3_151:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v117
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v182
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_140
.LBB3_152:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_153:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v186
	v_cmp_gt_i32_e64 s[0:1], s87, v186
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_155
; %bb.154:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v110
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v186
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_155:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v187
	v_cmp_gt_i32_e64 s[0:1], s87, v187
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_157
; %bb.156:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v111
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v187
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_157:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v188
	v_cmp_gt_i32_e64 s[0:1], s87, v188
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_159
; %bb.158:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v112
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v188
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_159:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v189
	v_cmp_gt_i32_e64 s[0:1], s87, v189
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_161
; %bb.160:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v113
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v189
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_161:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_439
; %bb.162:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_164
.LBB3_163:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v106
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v186
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_164:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_181
; %bb.165:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_182
.LBB3_166:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_183
.LBB3_167:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_184
.LBB3_168:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_170
.LBB3_169:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v102
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v186
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_170:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_185
; %bb.171:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_186
.LBB3_172:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_187
.LBB3_173:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_188
.LBB3_174:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_176
.LBB3_175:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v98
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v186
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_176:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_189
; %bb.177:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_190
.LBB3_178:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_191
.LBB3_179:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_192
.LBB3_180:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_193
.LBB3_181:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v107
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v187
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_166
.LBB3_182:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v108
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v188
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_167
.LBB3_183:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v109
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v189
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_168
.LBB3_184:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_169
	s_branch .LBB3_170
.LBB3_185:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v103
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v187
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_172
.LBB3_186:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v104
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v188
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_173
.LBB3_187:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v105
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v189
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_174
.LBB3_188:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_175
	s_branch .LBB3_176
.LBB3_189:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v99
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v187
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_178
.LBB3_190:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v100
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v188
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_179
.LBB3_191:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v101
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v189
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_180
.LBB3_192:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_193:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v190
	v_cmp_gt_i32_e64 s[0:1], s87, v190
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_195
; %bb.194:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v94
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v190
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_195:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v191
	v_cmp_gt_i32_e64 s[0:1], s87, v191
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_197
; %bb.196:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v95
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v191
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_197:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v192
	v_cmp_gt_i32_e64 s[0:1], s87, v192
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_199
; %bb.198:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v96
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v192
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_199:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v193
	v_cmp_gt_i32_e64 s[0:1], s87, v193
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_201
; %bb.200:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v97
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v193
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_201:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_440
; %bb.202:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_204
.LBB3_203:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v90
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v190
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_204:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_221
; %bb.205:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_222
.LBB3_206:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_223
.LBB3_207:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_224
.LBB3_208:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_210
.LBB3_209:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v86
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v190
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_210:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_225
; %bb.211:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_226
.LBB3_212:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_227
.LBB3_213:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_228
.LBB3_214:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_216
.LBB3_215:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v82
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v190
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_216:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_229
; %bb.217:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_230
.LBB3_218:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_231
.LBB3_219:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_232
.LBB3_220:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_233
.LBB3_221:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v91
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v191
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_206
.LBB3_222:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v92
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v192
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_207
.LBB3_223:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v93
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v193
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_208
.LBB3_224:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_209
	s_branch .LBB3_210
.LBB3_225:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v87
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v191
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_212
.LBB3_226:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v88
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v192
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_213
.LBB3_227:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v89
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v193
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_214
.LBB3_228:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_215
	s_branch .LBB3_216
.LBB3_229:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v83
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v191
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_218
.LBB3_230:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v84
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v192
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_219
.LBB3_231:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v85
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v193
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_220
.LBB3_232:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_233:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v194
	v_cmp_gt_i32_e64 s[0:1], s87, v194
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_235
; %bb.234:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v78
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v194
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_235:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v195
	v_cmp_gt_i32_e64 s[0:1], s87, v195
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_237
; %bb.236:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v79
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v195
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_237:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v196
	v_cmp_gt_i32_e64 s[0:1], s87, v196
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_239
; %bb.238:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v80
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v196
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_239:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v197
	v_cmp_gt_i32_e64 s[0:1], s87, v197
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_241
; %bb.240:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v81
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v197
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_241:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_441
; %bb.242:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_244
.LBB3_243:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v74
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v194
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_244:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_261
; %bb.245:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_262
.LBB3_246:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_263
.LBB3_247:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_264
.LBB3_248:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_250
.LBB3_249:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v62
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v194
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_250:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_265
; %bb.251:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_266
.LBB3_252:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_267
.LBB3_253:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_268
.LBB3_254:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_256
.LBB3_255:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v58
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v194
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_256:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_269
; %bb.257:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_270
.LBB3_258:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_271
.LBB3_259:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_272
.LBB3_260:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_273
.LBB3_261:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v75
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v195
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_246
.LBB3_262:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v76
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v196
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_247
.LBB3_263:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v77
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v197
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_248
.LBB3_264:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_249
	s_branch .LBB3_250
.LBB3_265:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v63
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v195
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_252
.LBB3_266:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v64
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v196
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_253
.LBB3_267:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v65
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v197
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_254
.LBB3_268:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_255
	s_branch .LBB3_256
.LBB3_269:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v59
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v195
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_258
.LBB3_270:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v60
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v196
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_259
.LBB3_271:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v61
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v197
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_260
.LBB3_272:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_273:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v198
	v_cmp_gt_i32_e64 s[0:1], s87, v198
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_275
; %bb.274:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v54
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v198
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_275:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v199
	v_cmp_gt_i32_e64 s[0:1], s87, v199
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_277
; %bb.276:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v55
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v199
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_277:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v200
	v_cmp_gt_i32_e64 s[0:1], s87, v200
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_279
; %bb.278:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v56
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v200
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_279:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v201
	v_cmp_gt_i32_e64 s[0:1], s87, v201
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_281
; %bb.280:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v57
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v201
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_281:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_442
; %bb.282:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_284
.LBB3_283:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v50
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v198
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_284:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_301
; %bb.285:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_302
.LBB3_286:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_303
.LBB3_287:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_304
.LBB3_288:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_290
.LBB3_289:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v46
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v198
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_290:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_305
; %bb.291:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_306
.LBB3_292:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_307
.LBB3_293:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_308
.LBB3_294:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_296
.LBB3_295:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v42
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v198
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_296:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_309
; %bb.297:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_310
.LBB3_298:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_311
.LBB3_299:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_312
.LBB3_300:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_313
.LBB3_301:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v51
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v199
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_286
.LBB3_302:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v52
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v200
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_287
.LBB3_303:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v53
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v201
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_288
.LBB3_304:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_289
	s_branch .LBB3_290
.LBB3_305:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v47
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v199
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_292
.LBB3_306:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v48
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v200
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_293
.LBB3_307:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v49
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v201
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_294
.LBB3_308:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_295
	s_branch .LBB3_296
.LBB3_309:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v43
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v199
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_298
.LBB3_310:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v44
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v200
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_299
.LBB3_311:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v45
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v201
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_300
.LBB3_312:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_313:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v202
	v_cmp_gt_i32_e64 s[0:1], s87, v202
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_315
; %bb.314:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v38
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v202
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_315:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v203
	v_cmp_gt_i32_e64 s[0:1], s87, v203
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_317
; %bb.316:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v39
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v203
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_317:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v204
	v_cmp_gt_i32_e64 s[0:1], s87, v204
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_319
; %bb.318:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v40
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v204
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_319:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v205
	v_cmp_gt_i32_e64 s[0:1], s87, v205
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_321
; %bb.320:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v41
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v205
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_321:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_443
; %bb.322:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_324
.LBB3_323:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v34
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v202
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_324:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_341
; %bb.325:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_342
.LBB3_326:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_343
.LBB3_327:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_344
.LBB3_328:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_330
.LBB3_329:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v30
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v202
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_330:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_345
; %bb.331:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_346
.LBB3_332:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_347
.LBB3_333:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_348
.LBB3_334:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_336
.LBB3_335:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v26
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v202
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_336:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_349
; %bb.337:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_350
.LBB3_338:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_351
.LBB3_339:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_352
.LBB3_340:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_353
.LBB3_341:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v35
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v203
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_326
.LBB3_342:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v36
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v204
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_327
.LBB3_343:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v37
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v205
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_328
.LBB3_344:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_329
	s_branch .LBB3_330
.LBB3_345:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v31
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v203
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_332
.LBB3_346:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v32
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v204
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_333
.LBB3_347:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v33
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v205
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_334
.LBB3_348:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_335
	s_branch .LBB3_336
.LBB3_349:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v27
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v203
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_338
.LBB3_350:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v28
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v204
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_339
.LBB3_351:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v29
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v205
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_340
.LBB3_352:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_353:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v206
	v_cmp_gt_i32_e64 s[0:1], s87, v206
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_355
; %bb.354:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v22
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v206
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_355:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v208
	v_cmp_gt_i32_e64 s[0:1], s87, v208
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_357
; %bb.356:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v23
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v208
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_357:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v209
	v_cmp_gt_i32_e64 s[0:1], s87, v209
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_359
; %bb.358:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v24
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v209
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_359:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v210
	v_cmp_gt_i32_e64 s[0:1], s87, v210
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_361
; %bb.360:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v25
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v210
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_361:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_444
; %bb.362:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_364
.LBB3_363:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v18
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v206
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_364:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_381
; %bb.365:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_382
.LBB3_366:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_383
.LBB3_367:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_384
.LBB3_368:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_370
.LBB3_369:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v14
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v206
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_370:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_385
; %bb.371:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_386
.LBB3_372:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_387
.LBB3_373:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_388
.LBB3_374:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_376
.LBB3_375:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v10
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v206
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_376:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[78:79], s[80:81]
	s_cbranch_execnz .LBB3_389
; %bb.377:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execnz .LBB3_390
.LBB3_378:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execnz .LBB3_391
.LBB3_379:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_392
.LBB3_380:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB3_393
.LBB3_381:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v19
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v208
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_366
.LBB3_382:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v20
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v209
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_367
.LBB3_383:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v21
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v210
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_368
.LBB3_384:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_369
	s_branch .LBB3_370
.LBB3_385:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v15
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v208
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_372
.LBB3_386:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v16
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v209
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_373
.LBB3_387:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v17
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v210
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_374
.LBB3_388:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_375
	s_branch .LBB3_376
.LBB3_389:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v11
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v208
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[82:83]
	s_cbranch_execz .LBB3_378
.LBB3_390:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v12
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v209
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[0:1]
	s_cbranch_execz .LBB3_379
.LBB3_391:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v13
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v210
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[78:79]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_380
.LBB3_392:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
.LBB3_393:                              ;   in Loop: Header=BB3_110 Depth=3
	v_cmp_le_i32_e32 vcc, s86, v211
	v_cmp_gt_i32_e64 s[0:1], s87, v211
	s_and_b64 s[78:79], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[78:79]
	s_cbranch_execz .LBB3_395
; %bb.394:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v6
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v211
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_395:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v212
	v_cmp_gt_i32_e64 s[0:1], s87, v212
	s_and_b64 s[80:81], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[80:81]
	s_cbranch_execz .LBB3_397
; %bb.396:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v7
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v212
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_397:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v214
	v_cmp_gt_i32_e64 s[0:1], s87, v214
	s_and_b64 s[82:83], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[82:83]
	s_cbranch_execz .LBB3_399
; %bb.398:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v8
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v214
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154
.LBB3_399:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s86, v215
	v_cmp_gt_i32_e64 s[0:1], s87, v215
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_401
; %bb.400:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v9
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v215
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138
.LBB3_401:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_445
; %bb.402:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_404
.LBB3_403:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v2
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v211
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB3_404:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_428
; %bb.405:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_429
.LBB3_406:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_430
.LBB3_407:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_431
.LBB3_408:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execz .LBB3_410
.LBB3_409:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v70
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v211
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB3_410:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[80:81]
	s_cbranch_execnz .LBB3_432
; %bb.411:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execnz .LBB3_433
.LBB3_412:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execnz .LBB3_434
.LBB3_413:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_435
.LBB3_414:                              ;   in Loop: Header=BB3_110 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[14:15], s[78:79]
	s_cbranch_execz .LBB3_416
.LBB3_415:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v66
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v211
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB3_416:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[80:81]
	s_cbranch_execnz .LBB3_436
; %bb.417:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[82:83]
	s_cbranch_execnz .LBB3_437
.LBB3_418:                              ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB3_420
.LBB3_419:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v69
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v215
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:96
.LBB3_420:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi128ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_mul_i32 s0, s61, s43
	s_mul_hi_u32 s1, s60, s43
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s43
	v_lshl_add_u64 v[154:155], s[0:1], 1, v[150:151]
	s_andn2_b64 vcc, exec, s[66:67]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB3_446
; %bb.421:                              ;   in Loop: Header=BB3_110 Depth=3
	s_mov_b64 s[14:15], -1
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_448
; %bb.422:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v138, v160
	v_mov_b32_e32 v156, v207
	v_mov_b32_e32 v157, v216
	v_mov_b32_e32 v158, v0
                                        ; implicit-def: $sgpr78_sgpr79
                                        ; implicit-def: $sgpr80_sgpr81
	s_branch .LBB3_424
.LBB3_423:                              ; %Flow2135
                                        ;   in Loop: Header=BB3_424 Depth=4
	s_or_b64 exec, exec, s[86:87]
	s_and_b64 s[82:83], exec, s[84:85]
	s_or_b64 s[14:15], s[82:83], s[14:15]
	s_andn2_b64 s[78:79], s[78:79], exec
	s_and_b64 s[82:83], s[80:81], exec
	s_or_b64 s[78:79], s[78:79], s[82:83]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_447
.LBB3_424:                              ; %.lr.ph.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ;       Parent Loop BB3_110 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v159, 0xf8, v138
	v_add_u32_e32 v224, 8, v159
	v_cmp_lt_i32_e64 s[82:83], s17, v224
	v_cmp_ge_i32_e32 vcc, s17, v224
	s_and_saveexec_b64 s[84:85], vcc
	s_cbranch_execz .LBB3_426
; %bb.425:                              ;   in Loop: Header=BB3_424 Depth=4
	v_mul_lo_u32 v224, s60, v156
	v_lshlrev_b32_e32 v224, 1, v224
	v_lshlrev_b32_e32 v159, 1, v159
	v_add3_u32 v224, v154, v224, v159
	v_add_u32_e32 v159, v157, v159
	v_or_b32_e32 v159, v159, v224
	v_and_b32_e32 v159, 15, v159
	v_cmp_eq_u32_e32 vcc, 0, v159
	s_andn2_b64 s[82:83], s[82:83], exec
	s_and_b64 s[86:87], vcc, exec
	s_or_b64 s[82:83], s[82:83], s[86:87]
.LBB3_426:                              ; %Flow2134
                                        ;   in Loop: Header=BB3_424 Depth=4
	s_or_b64 exec, exec, s[84:85]
	s_mov_b64 s[84:85], -1
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[86:87], s[82:83]
	s_cbranch_execz .LBB3_423
; %bb.427:                              ; %.critedge.i
                                        ;   in Loop: Header=BB3_424 Depth=4
	v_add_u32_e32 v158, 0x200, v158
	v_cmp_le_i32_e32 vcc, s92, v158
	v_add_u32_e32 v157, 0x2000, v157
	v_add_u32_e32 v156, 16, v156
	v_add_u32_e32 v138, 0x1000, v138
	s_or_b64 s[80:81], s[80:81], exec
	s_orn2_b64 s[84:85], vcc, exec
	s_branch .LBB3_423
.LBB3_428:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v3
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v212
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_406
.LBB3_429:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v4
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v214
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_407
.LBB3_430:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v5
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v215
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_408
.LBB3_431:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_409
	s_branch .LBB3_410
.LBB3_432:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v71
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v212
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[82:83]
	s_cbranch_execz .LBB3_412
.LBB3_433:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v72
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v214
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_saveexec_b64 s[84:85], s[0:1]
	s_cbranch_execz .LBB3_413
.LBB3_434:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v138, v138, v73
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s69
	v_subrev_u32_e32 v154, s86, v215
	v_lshl_add_u32 v154, v154, 9, v179
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[84:85]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB3_414
.LBB3_435:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[14:15], s[78:79]
	s_cbranch_execnz .LBB3_415
	s_branch .LBB3_416
.LBB3_436:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v67
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v212
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[82:83]
	s_cbranch_execz .LBB3_418
.LBB3_437:                              ;   in Loop: Header=BB3_110 Depth=3
	v_add_f32_e32 v154, v138, v68
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s69
	v_subrev_u32_e32 v155, s86, v214
	v_lshl_add_u32 v155, v155, 9, v179
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execnz .LBB3_419
	s_branch .LBB3_420
.LBB3_438:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_123
	s_branch .LBB3_124
.LBB3_439:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_163
	s_branch .LBB3_164
.LBB3_440:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_203
	s_branch .LBB3_204
.LBB3_441:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_243
	s_branch .LBB3_244
.LBB3_442:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_283
	s_branch .LBB3_284
.LBB3_443:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_323
	s_branch .LBB3_324
.LBB3_444:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_363
	s_branch .LBB3_364
.LBB3_445:                              ;   in Loop: Header=BB3_110 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[84:85], s[78:79]
	s_cbranch_execnz .LBB3_403
	s_branch .LBB3_404
.LBB3_446:                              ;   in Loop: Header=BB3_110 Depth=3
	s_cbranch_execz .LBB3_109
	s_branch .LBB3_466
.LBB3_447:                              ; %Flow2136
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_orn2_b64 s[14:15], s[78:79], exec
.LBB3_448:                              ; %Flow2137
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v138, 0, 1, s[14:15]
	s_nop 0
	v_readfirstlane_b32 s0, v138
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[14:15], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB3_453
; %bb.449:                              ;   in Loop: Header=BB3_110 Depth=3
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB3_452
; %bb.450:                              ; %.lr.ph.i264.preheader
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v156, v223
	v_mov_b32_e32 v138, v0
.LBB3_451:                              ; %.lr.ph.i264
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ;       Parent Loop BB3_110 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v157, v138, v222
	v_mul_lo_u32 v158, v157, s56
	v_sub_u32_e32 v158, v138, v158
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_subrev_u32_e32 v159, s56, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_xor_b32_e32 v157, s95, v157
	v_subrev_u32_e32 v224, s95, v157
	v_mad_u64_u32 v[158:159], s[78:79], s10, v224, v[138:139]
	v_lshlrev_b32_e32 v157, 9, v157
	v_mul_lo_u32 v159, s89, v224
	v_sub_u32_e32 v157, v157, v159
	v_add_u32_e32 v157, v156, v157
	ds_read_u16 v157, v157
	v_mad_i64_i32 v[224:225], s[78:79], s60, v224, 0
	v_add_u32_e32 v138, 0x200, v138
	v_mov_b32_e32 v159, v139
	v_lshl_add_u64 v[224:225], v[224:225], 1, v[154:155]
	v_cmp_le_i32_e32 vcc, s57, v138
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[224:225]
	v_add_u32_e32 v156, 0x400, v156
	s_or_b64 s[14:15], vcc, s[14:15]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v157
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_451
.LBB3_452:                              ; %Flow2126
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_453:                              ; %Flow2132
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB3_465
; %bb.454:                              ;   in Loop: Header=BB3_110 Depth=3
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_464
; %bb.455:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v224, v217
	v_mov_b64_e32 v[156:157], v[152:153]
	v_mov_b32_e32 v225, v0
	s_branch .LBB3_458
.LBB3_456:                              ; %Flow2128
                                        ;   in Loop: Header=BB3_458 Depth=4
	s_or_b64 exec, exec, s[80:81]
.LBB3_457:                              ; %.loopexit.i
                                        ;   in Loop: Header=BB3_458 Depth=4
	s_or_b64 exec, exec, s[78:79]
	v_add_u32_e32 v225, 0x200, v225
	v_cmp_le_i32_e32 vcc, s92, v225
	v_lshl_add_u64 v[156:157], v[156:157], 0, s[74:75]
	s_or_b64 s[14:15], vcc, s[14:15]
	v_add_u32_e32 v224, 0x2000, v224
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_464
.LBB3_458:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ;       Parent Loop BB3_110 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB3_463 Depth 5
	v_lshlrev_b32_e32 v138, 3, v225
	v_and_b32_e32 v138, 0xf8, v138
	v_add_u32_e32 v158, 8, v138
	v_cmp_ge_i32_e32 vcc, s17, v158
	s_and_saveexec_b64 s[78:79], vcc
	s_xor_b64 s[78:79], exec, s[78:79]
	s_cbranch_execz .LBB3_460
; %bb.459:                              ;   in Loop: Header=BB3_458 Depth=4
	v_lshrrev_b32_e32 v226, 5, v225
	v_mad_i64_i32 v[158:159], s[80:81], s60, v226, 0
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[154:155]
	v_lshlrev_b32_e32 v138, 1, v138
	v_lshlrev_b32_e32 v226, 9, v226
	v_lshl_add_u64 v[158:159], v[158:159], 0, v[138:139]
	v_add3_u32 v138, 0, v226, v138
	ds_read_b128 v[226:229], v138
                                        ; implicit-def: $vgpr138
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[158:159], v[226:229]
.LBB3_460:                              ; %Flow2129
                                        ;   in Loop: Header=BB3_458 Depth=4
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB3_457
; %bb.461:                              ; %.preheader.i
                                        ;   in Loop: Header=BB3_458 Depth=4
	v_cmp_gt_i32_e32 vcc, s16, v138
	s_and_saveexec_b64 s[80:81], vcc
	s_cbranch_execz .LBB3_456
; %bb.462:                              ; %.lr.ph.i266
                                        ;   in Loop: Header=BB3_458 Depth=4
	s_mov_b32 s84, 0
	s_mov_b64 s[82:83], 0
	v_mov_b32_e32 v138, v224
	v_mov_b64_e32 v[158:159], v[156:157]
.LBB3_463:                              ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ;       Parent Loop BB3_110 Depth=3
                                        ;         Parent Loop BB3_458 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v226, v138
	s_add_i32 s3, s84, 1
	s_cmp_gt_u32 s84, 6
	s_cselect_b64 s[86:87], -1, 0
	v_add_u32_e32 v138, 2, v138
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
	s_cbranch_execnz .LBB3_463
	s_branch .LBB3_456
.LBB3_464:                              ; %Flow2131
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB3_465:                              ; %Flow2133
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_branch .LBB3_109
.LBB3_466:                              ;   in Loop: Header=BB3_110 Depth=3
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB3_108
; %bb.467:                              ; %.lr.ph.i270.preheader
                                        ;   in Loop: Header=BB3_110 Depth=3
	s_mov_b64 s[14:15], 0
	v_mov_b32_e32 v156, v223
	v_mov_b32_e32 v138, v0
.LBB3_468:                              ; %.lr.ph.i270
                                        ;   Parent Loop BB3_84 Depth=1
                                        ;     Parent Loop BB3_106 Depth=2
                                        ;       Parent Loop BB3_110 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v157, v138, v222
	v_mul_lo_u32 v158, v157, s56
	v_sub_u32_e32 v158, v138, v158
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_subrev_u32_e32 v159, s56, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s56, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_xor_b32_e32 v157, s95, v157
	v_subrev_u32_e32 v224, s95, v157
	v_mad_u64_u32 v[158:159], s[78:79], s10, v224, v[138:139]
	v_lshlrev_b32_e32 v157, 9, v157
	v_mul_lo_u32 v159, s89, v224
	v_sub_u32_e32 v157, v157, v159
	v_add_u32_e32 v157, v156, v157
	ds_read_u16 v157, v157
	v_mad_i64_i32 v[224:225], s[78:79], s60, v224, 0
	v_add_u32_e32 v138, 0x200, v138
	v_mov_b32_e32 v159, v139
	v_lshl_add_u64 v[224:225], v[224:225], 1, v[154:155]
	v_cmp_le_i32_e32 vcc, s57, v138
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[224:225]
	v_add_u32_e32 v156, 0x400, v156
	s_or_b64 s[14:15], vcc, s[14:15]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v157
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_468
	s_branch .LBB3_108
.LBB3_469:                              ; %._crit_edge820
                                        ;   in Loop: Header=BB3_84 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s10, v232, 0
	v_readlane_b32 s11, v232, 1
	s_and_b64 s[10:11], s[0:1], s[10:11]
	s_mov_b64 exec, s[10:11]
	s_cbranch_execz .LBB3_471
; %bb.470:                              ;   in Loop: Header=BB3_84 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB3_471:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s10, v232, 21
	v_readlane_b32 s11, v232, 22
	s_and_b64 s[10:11], s[0:1], s[10:11]
	v_readlane_b32 s43, v232, 35
	v_readlane_b32 s52, v232, 37
	v_readlane_b32 s82, v232, 38
	s_mov_b64 exec, s[10:11]
	s_cbranch_execz .LBB3_82
; %bb.472:                              ; %.lr.ph822.preheader
                                        ;   in Loop: Header=BB3_84 Depth=1
	s_lshl_b32 s10, s45, 2
	s_add_i32 s10, s2, s10
	s_sub_i32 s4, s10, s4
	s_sub_i32 s4, s4, s5
	s_lshl_b32 s5, s7, 2
	s_sub_i32 s4, s4, s5
	s_lshl_b32 s4, s4, 8
	s_mov_b32 s5, s21
	s_branch .LBB3_474
.LBB3_473:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB3_474 Depth=2
	s_add_i32 s5, s5, -1
	s_add_i32 s4, s4, s26
	s_cmp_lg_u32 s5, 0
	s_cbranch_scc0 .LBB3_82
.LBB3_474:                              ; %.lr.ph822
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
	s_cbranch_scc1 .LBB3_476
; %bb.475:                              ; %Flow2122
                                        ;   in Loop: Header=BB3_474 Depth=2
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB3_473
	s_branch .LBB3_477
.LBB3_476:                              ;   in Loop: Header=BB3_474 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB3_473
.LBB3_477:                              ;   in Loop: Header=BB3_474 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB3_473
.LBB3_478:                              ; %.critedge259
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
		.amdhsa_next_free_vgpr 233
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 236
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
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb0EEv14mi300x_globals.num_vgpr, 233
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
; codeLenInByte = 21620
; TotalNumSgprs: 106
; NumVgprs: 233
; NumAgprs: 0
; TotalNumVgprs: 233
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 233
; AccumOffset: 236
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 58
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
                                        ; implicit-def: $vgpr232 : SGPR spill to VGPR lane
	s_load_dwordx2 s[58:59], s[0:1], 0xf8
	s_load_dwordx2 s[18:19], s[0:1], 0x120
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s52, s37, 31
	s_lshr_b32 s3, s52, 29
	v_writelane_b32 v232, s4, 0
	s_add_i32 s3, s37, s3
	s_ashr_i32 s33, s3, 3
	v_writelane_b32 v232, s5, 1
	v_writelane_b32 v232, s6, 2
	v_writelane_b32 v232, s7, 3
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
.LBB4_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit283
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
.LBB4_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit286
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
	v_writelane_b32 v232, s3, 45
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 46
	s_cmp_lg_u32 s36, 1
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_writelane_b32 v232, s7, 47
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 48
	s_cmp_lg_u32 s36, 2
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_writelane_b32 v232, s7, 49
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 50
	s_cmp_lg_u32 s36, 3
	v_lshlrev_b32_e32 v65, 3, v0
	v_writelane_b32 v232, s7, 51
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 29
	s_cmp_lg_u32 s36, 4
	v_lshrrev_b32_e32 v18, 5, v0
	v_writelane_b32 v232, s7, 30
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 32
	s_cmp_lg_u32 s36, 5
	v_mov_b32_e32 v19, v21
	v_writelane_b32 v232, s7, 33
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 25
	s_cmp_lg_u32 s36, 6
	s_mov_b64 s[98:99], 0
	v_writelane_b32 v232, s7, 26
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v232, s6, 7
	s_cmp_lg_u32 s36, 7
	v_bfrev_b32_e32 v66, 32
	v_writelane_b32 v232, s7, 8
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s54, s41
	v_cvt_f32_u32_e32 v2, s54
	s_sub_i32 s3, 0, s54
	s_ashr_i32 s55, s41, 31
	s_lshl_b64 s[82:83], s[24:25], 5
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v232, s6, 9
	s_mov_b32 s88, 0x7060302
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v232, s7, 10
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_readfirstlane_b32 s53, v2
	s_mul_i32 s3, s3, s53
	s_mul_hi_u32 s3, s53, s3
	s_add_i32 s53, s53, s3
	s_add_u32 s66, s22, 2
	s_addc_u32 s67, s23, 0
	v_writelane_b32 v232, s66, 16
	s_mul_i32 s3, s13, 14
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_writelane_b32 v232, s67, 17
	s_mul_hi_u32 s66, s12, 14
	s_add_i32 s66, s66, s3
	s_mul_i32 s3, s12, 14
	s_add_u32 s3, s56, s3
	s_addc_u32 s66, s57, s66
	s_add_u32 s68, s3, 2
	s_addc_u32 s69, s66, 0
	s_lshl_b64 s[66:67], s[12:13], 2
	v_writelane_b32 v232, s68, 14
	s_add_u32 s66, s56, s66
	s_addc_u32 s67, s57, s67
	v_writelane_b32 v232, s69, 15
	v_writelane_b32 v232, s66, 18
	s_mul_i32 s3, s13, 12
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v232, s67, 19
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
	v_writelane_b32 v232, s68, 39
	s_addc_u32 s97, s57, s13
	s_xor_b64 s[66:67], s[14:15], -1
	s_movk_i32 s3, 0x7fff
	v_and_b32_e32 v67, 0x100, v2
	v_writelane_b32 v232, s69, 40
	s_branch .LBB4_12
.LBB4_10:                               ; %Flow2124
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_sub_i32 s12, s16, s43
	s_add_i32 s16, s12, 0x130
	s_cmp_ge_i32 s16, s91
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB4_11:                               ; %Flow2136
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
; %bb.14:                               ; %.lr.ph.i.i.i288.preheader
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[72:73], 0
                                        ; implicit-def: $sgpr68_sgpr69
                                        ; implicit-def: $sgpr70_sgpr71
	s_branch .LBB4_16
.LBB4_15:                               ; %Flow2132
                                        ;   in Loop: Header=BB4_16 Depth=2
	s_and_b64 s[76:77], exec, s[70:71]
	s_or_b64 s[14:15], s[76:77], s[14:15]
	s_andn2_b64 s[68:69], s[68:69], exec
	s_and_b64 s[74:75], s[74:75], exec
	s_or_b64 s[68:69], s[68:69], s[74:75]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB4_18
.LBB4_16:                               ; %.lr.ph.i.i.i288
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
.LBB4_20:                               ; %.critedge713
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
.LBB4_23:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit294
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v2, v67, v2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v2, 0x4000000, v2
	v_cmp_eq_u32_e64 s[14:15], 0, v2
.LBB4_24:                               ; %Flow2135
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_and_saveexec_b64 s[68:69], s[14:15]
	s_cbranch_execz .LBB4_11
; %bb.25:                               ; %.critedge721
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB4_38
; %bb.26:                               ; %.lr.ph.i295.preheader
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
	v_readlane_b32 s14, v232, 16
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_readlane_b32 s15, v232, 17
	v_lshl_add_u64 v[22:23], s[56:57], 0, v[2:3]
	v_lshl_add_u64 v[26:27], s[26:27], 0, v[2:3]
	v_lshl_add_u64 v[24:25], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 14
	v_readlane_b32 s15, v232, 15
	v_lshl_add_u64 v[34:35], s[92:93], 0, v[2:3]
	v_lshl_add_u64 v[36:37], s[94:95], 0, v[2:3]
	v_lshl_add_u64 v[28:29], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 18
	v_readlane_b32 s15, v232, 19
	v_lshl_add_u64 v[38:39], s[96:97], 0, v[2:3]
	s_mov_b64 s[74:75], 0
	v_lshl_add_u64 v[30:31], s[14:15], 0, v[2:3]
	v_readlane_b32 s14, v232, 39
	v_readlane_b32 s15, v232, 40
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	v_lshl_add_u64 v[32:33], s[14:15], 0, v[2:3]
	s_branch .LBB4_28
.LBB4_27:                               ; %.critedge.i298
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
.LBB4_28:                               ; %.lr.ph.i295
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
; %bb.29:                               ; %.preheader.i297.preheader
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
.LBB4_30:                               ; %Flow2126
                                        ;   in Loop: Header=BB4_32 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[80:81], s[80:81], exec
	s_and_b64 s[84:85], s[86:87], exec
	s_or_b64 s[80:81], s[80:81], s[84:85]
.LBB4_31:                               ; %Flow2125
                                        ;   in Loop: Header=BB4_32 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[80:81]
	s_or_b64 s[78:79], s[14:15], s[78:79]
	s_andn2_b64 exec, exec, s[78:79]
	s_cbranch_execz .LBB4_35
.LBB4_32:                               ; %.preheader.i297
                                        ;   Parent Loop BB4_12 Depth=1
                                        ;     Parent Loop BB4_28 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[24:25], v[2:3]
	s_or_b64 s[80:81], s[80:81], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_31
; %bb.33:                               ; %.preheader.i297.1
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
.LBB4_35:                               ; %Flow2127
                                        ;   in Loop: Header=BB4_28 Depth=2
	s_or_b64 exec, exec, s[78:79]
                                        ; implicit-def: $vgpr2_vgpr3
.LBB4_36:                               ; %Flow2128
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
.LBB4_38:                               ; %Flow2130
                                        ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[70:71], s[4:5]
	s_cbranch_execz .LBB4_10
; %bb.39:                               ; %.preheader730
                                        ;   in Loop: Header=BB4_12 Depth=1
	v_mov_b64_e32 v[2:3], s[48:49]
	flat_load_dwordx4 v[2:5], v[2:3]
	s_mul_i32 s12, s40, s36
	s_add_i32 s12, s89, s12
	v_readlane_b32 s13, v232, 45
	s_mul_i32 s12, s12, s41
	s_add_i32 s13, s13, s90
	s_add_i32 s12, s13, s12
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s46, s12
	s_addc_u32 s15, s47, s13
	v_mov_b32_e32 v6, s15
	v_readlane_b32 s12, v232, 46
	v_readlane_b32 s13, v232, 47
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
	v_readlane_b32 s12, v232, 48
	v_readlane_b32 s13, v232, 49
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
.LBB4_44:                               ; %Flow2122
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
	v_readlane_b32 s12, v232, 50
	v_readlane_b32 s13, v232, 51
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
.LBB4_48:                               ; %Flow2121
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
	v_readlane_b32 s12, v232, 29
	v_readlane_b32 s13, v232, 30
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
.LBB4_52:                               ; %Flow2120
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
	v_readlane_b32 s12, v232, 32
	v_readlane_b32 s13, v232, 33
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
.LBB4_56:                               ; %Flow2119
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
	v_readlane_b32 s12, v232, 25
	v_readlane_b32 s13, v232, 26
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
.LBB4_60:                               ; %Flow2118
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
	v_readlane_b32 s12, v232, 7
	v_readlane_b32 s13, v232, 8
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
.LBB4_64:                               ; %Flow2117
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
	v_readlane_b32 s12, v232, 9
	v_readlane_b32 s13, v232, 10
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
.LBB4_71:                               ; %Flow2140
	s_or_b64 exec, exec, s[20:21]
	s_load_dwordx2 s[18:19], s[0:1], 0x120
	s_mov_b64 s[4:5], 0
.LBB4_72:                               ; %Flow2182
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccz .LBB4_482
; %bb.73:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s50, s4
	s_addc_u32 s5, s51, s5
	v_cmp_eq_u32_e64 s[8:9], 0, v0
	s_mov_b64 s[6:7], exec
	s_nop 0
	v_writelane_b32 v232, s8, 4
	s_nop 1
	v_writelane_b32 v232, s9, 5
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
	s_branch .LBB4_482
.LBB4_79:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB4_482
.LBB4_80:                               ; %.critedge711
	s_lshr_b32 s3, s52, 24
	s_add_i32 s3, s37, s3
	s_ashr_i32 s63, s3, 8
	s_mul_i32 s3, s41, s63
	s_cmp_ge_i32 s2, s3
	v_writelane_b32 v232, s3, 6
	s_cbranch_scc1 .LBB4_482
; %bb.81:                               ; %.lr.ph843
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
	v_lshlrev_b32_e32 v166, 3, v0
	s_and_b32 s0, s3, 15
	s_add_u32 s5, s5, 16
	v_and_b32_e32 v2, 24, v166
	s_cmp_eq_u64 s[0:1], 0
	v_lshlrev_b32_e32 v5, 4, v0
	v_lshlrev_b32_e32 v167, 1, v2
	s_cselect_b32 s69, s3, s5
	s_add_i32 s0, s39, 31
	v_and_b32_e32 v168, 0x1fc0, v5
	v_add_u32_e32 v5, s65, v167
	s_ashr_i32 s1, s0, 31
	v_add_u32_e32 v9, v5, v168
	s_lshr_b32 s1, s1, 27
	v_lshrrev_b32_e32 v6, 2, v0
	v_lshrrev_b32_e32 v10, 4, v9
	s_add_i32 s0, s0, s1
	v_bfe_u32 v3, v0, 6, 2
	v_or_b32_e32 v7, 0x80, v6
	v_add_u32_e32 v8, 8, v5
	v_and_b32_e32 v10, 56, v10
	s_ashr_i32 s84, s0, 5
	v_mad_u64_u32 v[130:131], s[0:1], v6, s14, v[2:3]
	v_mad_u64_u32 v[132:133], s[0:1], v7, s14, v[2:3]
	v_xor_b32_e32 v169, v10, v9
	v_add_u32_e32 v9, v8, v168
	v_mad_u64_u32 v[134:135], s[0:1], v6, s10, v[2:3]
	v_lshrrev_b32_e32 v10, 4, v9
	v_or_b32_e32 v171, 0x2000, v168
	s_mov_b32 s0, s10
	s_add_i32 s37, s84, -1
	s_lshl_b32 s5, s41, 2
	v_and_b32_e32 v10, 56, v10
	v_add_u32_e32 v5, v5, v171
	v_writelane_b32 v232, s0, 7
	v_xor_b32_e32 v170, v10, v9
	v_lshrrev_b32_e32 v9, 4, v5
	v_writelane_b32 v232, s1, 8
	v_mad_u64_u32 v[136:137], s[0:1], v7, s10, v[2:3]
	s_cmp_gt_i32 s39, 0
	v_and_b32_e32 v9, 56, v9
	s_cselect_b64 s[0:1], -1, 0
	v_xor_b32_e32 v172, v9, v5
	v_add_u32_e32 v5, v8, v171
	v_writelane_b32 v232, s0, 9
	v_lshrrev_b32_e32 v8, 4, v5
	v_add_u32_e32 v2, s69, v167
	v_writelane_b32 v232, s1, 10
	s_ashr_i32 s1, s18, 31
	s_mov_b32 s0, s18
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v7, v2, v168
	s_lshl_b32 s3, s37, 5
	s_lshl_b64 s[0:1], s[0:1], 2
	v_xor_b32_e32 v173, v8, v5
	v_lshrrev_b32_e32 v8, 4, v7
	s_add_u32 s0, s46, s0
	v_add_u32_e32 v5, 8, v2
	v_and_b32_e32 v8, 56, v8
	s_addc_u32 s1, s47, s1
	v_xor_b32_e32 v174, v8, v7
	v_add_u32_e32 v7, v5, v168
	v_writelane_b32 v232, s0, 11
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 56, v8
	v_writelane_b32 v232, s1, 12
	v_add_u32_e32 v2, v2, v171
	s_min_i32 s20, s42, 32
	s_mul_i32 s21, s6, s4
	s_bfe_i64 s[74:75], s[60:61], 0x200000
	s_mov_b32 s61, s5
	v_readlane_b32 s4, v232, 0
	v_xor_b32_e32 v175, v8, v7
	v_lshrrev_b32_e32 v7, 4, v2
	s_cmp_gt_i32 s42, 0
	v_readlane_b32 s5, v232, 1
	v_and_b32_e32 v7, 56, v7
	s_cselect_b64 s[76:77], -1, 0
	s_cmp_lg_u64 s[4:5], 0
	v_xor_b32_e32 v176, v7, v2
	v_add_u32_e32 v2, v5, v171
	s_cselect_b64 s[78:79], -1, 0
	s_max_i32 s0, s38, 1
	v_lshrrev_b32_e32 v5, 4, v2
	s_add_i32 s0, s0, -1
	v_and_b32_e32 v5, 56, v5
	s_cmp_lg_u32 s19, 0
	v_xor_b32_e32 v177, v5, v2
	v_and_b32_e32 v2, 15, v0
	s_cselect_b64 s[80:81], -1, 0
	s_abs_i32 s4, s61
	v_and_b32_e32 v5, 12, v6
	v_lshlrev_b32_e32 v6, 6, v2
	v_lshl_or_b32 v182, v3, 6, v2
	v_cvt_f32_u32_e32 v2, s4
	s_or_b32 s1, s3, 16
	v_readlane_b32 s6, v232, 2
	v_readlane_b32 s7, v232, 3
	v_writelane_b32 v232, s0, 13
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s39, s3
	s_sub_i32 s3, s39, s1
	s_abs_i32 s39, s42
	v_lshl_or_b32 v179, v3, 12, v6
	v_cvt_f32_u32_e32 v3, s39
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_bfe_i32 s1, s41, 0x1001d
	v_rcp_iflag_f32_e32 v3, v3
	v_writelane_b32 v232, s1, 14
	v_writelane_b32 v232, s4, 16
	s_sub_i32 s1, 0, s4
	v_readfirstlane_b32 s4, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s1, s1, s4
	s_mul_hi_u32 s1, s4, s1
	s_add_i32 s1, s4, s1
	v_writelane_b32 v232, s1, 18
	s_sub_i32 s1, 0, s39
	v_readfirstlane_b32 s4, v2
	s_mul_i32 s1, s1, s4
	s_mul_hi_u32 s1, s4, s1
	s_add_i32 s35, s4, s1
	s_lshr_b32 s1, s35, 24
	s_mul_i32 s4, s1, s39
	s_sub_i32 s4, 0x100, s4
	s_lshl_b32 s82, s20, 5
	s_max_i32 s83, s84, 1
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
	v_lshl_or_b32 v178, v4, 13, v6
	v_rcp_iflag_f32_e32 v2, v2
	v_or_b32_e32 v6, 1, v5
	v_or_b32_e32 v7, 2, v5
	v_or_b32_e32 v8, 3, v5
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_gt_i32_e64 s[4:5], s0, v6
	v_cmp_gt_i32_e64 s[6:7], s0, v7
	v_cmp_gt_i32_e64 s[10:11], s0, v8
	v_readfirstlane_b32 s8, v2
	s_mul_i32 s9, s9, s8
	s_mul_hi_u32 s9, s8, s9
	s_add_i32 s71, s8, s9
	v_cmp_gt_i32_e64 s[8:9], s0, v5
	s_xor_b32 s0, s1, s34
	v_lshrrev_b32_e32 v217, 5, v0
	s_sub_i32 s22, s0, s34
	s_ashr_i32 s23, s33, 31
	s_cmp_gt_i32 s22, 0
	v_mad_i64_i32 v[2:3], s[0:1], v217, s60, 0
	s_cselect_b64 s[18:19], -1, 0
	v_readlane_b32 s0, v232, 4
	v_readlane_b32 s1, v232, 5
	v_writelane_b32 v232, s18, 20
	s_and_b64 s[0:1], s[0:1], s[18:19]
	v_and_b32_e32 v9, 63, v0
	v_writelane_b32 v232, s19, 21
	v_writelane_b32 v232, s0, 22
	v_lshl_or_b32 v183, v4, 7, v5
	v_and_b32_e32 v4, 31, v0
	v_writelane_b32 v232, s1, 23
	s_add_u32 s0, s16, 64
	v_writelane_b32 v232, s0, 24
	v_writelane_b32 v232, s16, 25
	s_addc_u32 s0, s17, 0
	v_lshlrev_b32_e32 v138, 4, v4
	v_writelane_b32 v232, s17, 26
	v_writelane_b32 v232, s0, 27
	s_add_u32 s0, s12, 64
	v_writelane_b32 v232, s0, 28
	v_writelane_b32 v232, s12, 29
	s_addc_u32 s0, s13, 0
	v_lshlrev_b32_e32 v10, 9, v217
	v_writelane_b32 v232, s13, 30
	v_writelane_b32 v232, s0, 31
	s_mov_b32 s0, s14
	v_writelane_b32 v232, s0, 32
	v_mov_b32_e32 v139, 0
	v_lshl_add_u64 v[140:141], v[2:3], 1, v[138:139]
	v_writelane_b32 v232, s1, 33
	s_lshl_b32 s0, s14, 8
	v_writelane_b32 v232, s0, 34
	s_mul_i32 s0, s75, s20
	s_mul_hi_u32 s1, s60, s20
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s60, s20
	s_lshl_b64 s[86:87], s[0:1], 1
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_or_b32_e32 v2, v10, v138
	v_add_u32_e32 v225, 0, v2
	v_writelane_b32 v232, s0, 35
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v232, s1, 36
	v_cmp_eq_u32_e64 s[0:1], 0, v9
	v_lshlrev_b32_e32 v216, 1, v5
	v_lshlrev_b32_e32 v2, 2, v2
	v_writelane_b32 v232, s0, 37
	v_ashrrev_i32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v133, 31, v132
	v_writelane_b32 v232, s1, 38
	v_cmp_gt_u32_e64 s[0:1], s22, v0
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v137, 31, v136
	v_writelane_b32 v232, s0, 39
	v_mul_lo_u32 v180, s42, v0
	v_add_u32_e32 v181, -1, v1
	v_writelane_b32 v232, s1, 40
	v_writelane_b32 v232, s66, 41
	s_mul_i32 s21, s21, s36
	v_lshl_add_u32 v184, v182, 1, 0
	v_writelane_b32 v232, s67, 42
	v_or_b32_e32 v185, 1, v183
	v_or_b32_e32 v186, 2, v183
	v_or_b32_e32 v187, 3, v183
	v_or_b32_e32 v188, 16, v182
	v_or_b32_e32 v189, 32, v182
	v_or_b32_e32 v190, 48, v182
	v_or_b32_e32 v191, 16, v183
	v_or_b32_e32 v192, 17, v183
	v_or_b32_e32 v193, 18, v183
	v_or_b32_e32 v194, 19, v183
	v_or_b32_e32 v195, 32, v183
	v_or_b32_e32 v196, 33, v183
	v_or_b32_e32 v197, 34, v183
	v_or_b32_e32 v198, 35, v183
	v_or_b32_e32 v199, 48, v183
	v_or_b32_e32 v200, 49, v183
	v_or_b32_e32 v201, 50, v183
	v_or_b32_e32 v202, 51, v183
	v_or_b32_e32 v203, 64, v183
	v_or_b32_e32 v204, 0x41, v183
	v_or_b32_e32 v205, 0x42, v183
	v_or_b32_e32 v206, 0x43, v183
	v_or_b32_e32 v207, 0x50, v183
	v_or_b32_e32 v208, 0x51, v183
	v_or_b32_e32 v209, 0x52, v183
	v_or_b32_e32 v210, 0x53, v183
	v_or_b32_e32 v211, 0x60, v183
	v_or_b32_e32 v212, 0x61, v183
	v_or_b32_e32 v213, 0x62, v183
	v_or_b32_e32 v214, 0x63, v183
	v_or_b32_e32 v215, 0x70, v183
	v_or_b32_e32 v218, 0x71, v183
	v_or_b32_e32 v219, 0x72, v183
	v_or_b32_e32 v220, 0x73, v183
	v_or_b32_e32 v221, 32, v216
	v_lshl_or_b32 v222, v4, 3, 1
	v_add_u32_e32 v223, 0, v10
	v_lshl_add_u32 v224, v0, 1, 0
	s_sub_i32 s85, 0, s33
	v_and_b32_e32 v226, 0x100, v2
	v_bfrev_b32_e32 v227, 64
	s_movk_i32 s26, 0x7fff
	v_cmp_gt_i32_e64 s[12:13], s3, v6
	v_cmp_gt_i32_e64 s[14:15], s3, v7
	v_cmp_gt_i32_e64 s[16:17], s3, v5
	v_cmp_gt_i32_e64 s[18:19], s3, v8
	s_mov_b64 s[88:89], 0
	v_cmp_gt_i32_e64 s[24:25], s82, v0
	s_lshl_b64 s[90:91], s[74:75], 5
	v_writelane_b32 v232, s63, 43
	v_writelane_b32 v232, s61, 44
	s_branch .LBB4_84
.LBB4_82:                               ; %Flow2143
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s2, s2, s43
	v_readlane_b32 s0, v232, 6
	s_cmp_ge_i32 s2, s0
	s_cselect_b64 s[0:1], -1, 0
	v_readlane_b32 s30, v232, 52
	s_orn2_b64 s[0:1], s[0:1], exec
	v_readlane_b32 s31, v232, 53
.LBB4_83:                               ; %Flow2176
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[30:31]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[88:89], s[0:1], s[88:89]
	s_andn2_b64 exec, exec, s[88:89]
	s_cbranch_execz .LBB4_482
.LBB4_84:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_87 Depth 2
                                        ;     Child Loop BB4_98 Depth 2
                                        ;     Child Loop BB4_110 Depth 2
                                        ;       Child Loop BB4_114 Depth 3
                                        ;         Child Loop BB4_472 Depth 4
                                        ;         Child Loop BB4_428 Depth 4
                                        ;         Child Loop BB4_455 Depth 4
                                        ;         Child Loop BB4_462 Depth 4
                                        ;           Child Loop BB4_467 Depth 5
                                        ;     Child Loop BB4_478 Depth 2
	s_ashr_i32 s0, s2, 31
	v_readlane_b32 s1, v232, 14
	s_xor_b32 s29, s0, s1
	s_abs_i32 s0, s2
	v_readlane_b32 s1, v232, 18
	s_mul_hi_u32 s1, s0, s1
	v_readlane_b32 s28, v232, 16
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
	v_writelane_b32 v232, s29, 45
	v_writelane_b32 v232, s0, 46
	s_sub_i32 s0, s0, s29
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s3, s63, s1
	s_min_i32 s28, s3, 4
	s_abs_i32 s27, s28
	v_cvt_f32_u32_e32 v2, s27
	s_sub_i32 s30, 0, s27
	s_mul_i32 s0, s0, s61
	v_writelane_b32 v232, s0, 48
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s0, s2, s0
	s_abs_i32 s29, s0
	s_xor_b32 s3, s0, s28
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_ashr_i32 s3, s3, 31
	v_mov_b32_e32 v69, v139
	v_mov_b32_e32 v68, v139
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
	v_writelane_b32 v232, s28, 50
	s_add_i32 s0, s0, s1
	s_lshl_b32 s64, s0, 8
	v_readlane_b32 s0, v232, 32
	v_readlane_b32 s1, v232, 33
	s_mul_i32 s0, s64, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v232, 29
	v_readlane_b32 s29, v232, 30
	s_add_u32 s0, s28, s0
	s_addc_u32 s1, s29, s1
	v_lshl_add_u64 v[2:3], v[130:131], 1, s[0:1]
	v_lshl_add_u64 v[6:7], v[132:133], 1, s[0:1]
	s_lshl_b32 s62, s27, 8
	v_readlane_b32 s0, v232, 7
	v_readlane_b32 s1, v232, 8
	s_mul_i32 s0, s62, s0
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s28, v232, 25
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v169, v[2:3]

	;;#ASMEND
	v_readlane_b32 s29, v232, 26
	s_add_u32 s28, s28, s0
	;;#ASMSTART
	ds_write_b64 v170, v[4:5]

	;;#ASMEND
	s_addc_u32 s29, s29, s1
	;;#ASMSTART
	ds_write_b64 v172, v[6:7]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[134:135], 1, s[28:29]
	;;#ASMSTART
	ds_write_b64 v173, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], v[136:137], 1, s[28:29]
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v174, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v175, v[4:5]

	;;#ASMEND
	v_readlane_b32 s28, v232, 9
	;;#ASMSTART
	ds_write_b64 v176, v[6:7]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v177, v[8:9]

	;;#ASMEND
	v_readlane_b32 s29, v232, 10
	s_andn2_b64 vcc, exec, s[28:29]
	v_mov_b32_e32 v67, v139
	v_mov_b32_e32 v66, v139
	v_mov_b32_e32 v73, v139
	v_mov_b32_e32 v72, v139
	v_mov_b32_e32 v71, v139
	v_mov_b32_e32 v70, v139
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v4, v139
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v2, v139
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v8, v139
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v6, v139
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v12, v139
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v10, v139
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v16, v139
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v14, v139
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v20, v139
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v18, v139
	v_mov_b32_e32 v25, v139
	v_mov_b32_e32 v24, v139
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v22, v139
	v_mov_b32_e32 v29, v139
	v_mov_b32_e32 v28, v139
	v_mov_b32_e32 v27, v139
	v_mov_b32_e32 v26, v139
	v_mov_b32_e32 v33, v139
	v_mov_b32_e32 v32, v139
	v_mov_b32_e32 v31, v139
	v_mov_b32_e32 v30, v139
	v_mov_b32_e32 v37, v139
	v_mov_b32_e32 v36, v139
	v_mov_b32_e32 v35, v139
	v_mov_b32_e32 v34, v139
	v_mov_b32_e32 v41, v139
	v_mov_b32_e32 v40, v139
	v_mov_b32_e32 v39, v139
	v_mov_b32_e32 v38, v139
	v_mov_b32_e32 v45, v139
	v_mov_b32_e32 v44, v139
	v_mov_b32_e32 v43, v139
	v_mov_b32_e32 v42, v139
	v_mov_b32_e32 v49, v139
	v_mov_b32_e32 v48, v139
	v_mov_b32_e32 v47, v139
	v_mov_b32_e32 v46, v139
	v_mov_b32_e32 v53, v139
	v_mov_b32_e32 v52, v139
	v_mov_b32_e32 v51, v139
	v_mov_b32_e32 v50, v139
	v_mov_b32_e32 v57, v139
	v_mov_b32_e32 v56, v139
	v_mov_b32_e32 v55, v139
	v_mov_b32_e32 v54, v139
	v_mov_b32_e32 v61, v139
	v_mov_b32_e32 v60, v139
	v_mov_b32_e32 v59, v139
	v_mov_b32_e32 v58, v139
	v_mov_b32_e32 v65, v139
	v_mov_b32_e32 v64, v139
	v_mov_b32_e32 v63, v139
	v_mov_b32_e32 v62, v139
	v_mov_b32_e32 v77, v139
	v_mov_b32_e32 v76, v139
	v_mov_b32_e32 v75, v139
	v_mov_b32_e32 v74, v139
	v_mov_b32_e32 v81, v139
	v_mov_b32_e32 v80, v139
	v_mov_b32_e32 v79, v139
	v_mov_b32_e32 v78, v139
	v_mov_b32_e32 v85, v139
	v_mov_b32_e32 v84, v139
	v_mov_b32_e32 v83, v139
	v_mov_b32_e32 v82, v139
	v_mov_b32_e32 v89, v139
	v_mov_b32_e32 v88, v139
	v_mov_b32_e32 v87, v139
	v_mov_b32_e32 v86, v139
	v_mov_b32_e32 v93, v139
	v_mov_b32_e32 v92, v139
	v_mov_b32_e32 v91, v139
	v_mov_b32_e32 v90, v139
	v_mov_b32_e32 v97, v139
	v_mov_b32_e32 v96, v139
	v_mov_b32_e32 v95, v139
	v_mov_b32_e32 v94, v139
	v_mov_b32_e32 v101, v139
	v_mov_b32_e32 v100, v139
	v_mov_b32_e32 v99, v139
	v_mov_b32_e32 v98, v139
	v_mov_b32_e32 v105, v139
	v_mov_b32_e32 v104, v139
	v_mov_b32_e32 v103, v139
	v_mov_b32_e32 v102, v139
	v_mov_b32_e32 v109, v139
	v_mov_b32_e32 v108, v139
	v_mov_b32_e32 v107, v139
	v_mov_b32_e32 v106, v139
	v_mov_b32_e32 v113, v139
	v_mov_b32_e32 v112, v139
	v_mov_b32_e32 v111, v139
	v_mov_b32_e32 v110, v139
	v_mov_b32_e32 v117, v139
	v_mov_b32_e32 v116, v139
	v_mov_b32_e32 v115, v139
	v_mov_b32_e32 v114, v139
	v_mov_b32_e32 v121, v139
	v_mov_b32_e32 v120, v139
	v_mov_b32_e32 v119, v139
	v_mov_b32_e32 v118, v139
	v_mov_b32_e32 v125, v139
	v_mov_b32_e32 v124, v139
	v_mov_b32_e32 v123, v139
	v_mov_b32_e32 v122, v139
	v_mov_b32_e32 v129, v139
	v_mov_b32_e32 v128, v139
	v_mov_b32_e32 v127, v139
	v_mov_b32_e32 v126, v139
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB4_93
; %bb.85:                               ; %.lr.ph801.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v232, 24
	s_add_u32 s28, s28, s0
	v_readlane_b32 s0, v232, 27
	s_addc_u32 s29, s0, s1
	v_readlane_b32 s0, v232, 46
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s2, s0
	v_readlane_b32 s1, v232, 48
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v232, 50
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v232, 45
	s_lshl_b32 s1, s1, 2
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v232, 34
	s_mul_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_readlane_b32 s30, v232, 28
	s_add_u32 s30, s30, s0
	v_readlane_b32 s0, v232, 31
	v_mov_b32_e32 v66, 0
	s_addc_u32 s31, s0, s1
	s_mov_b32 s0, 0
	v_mov_b32_e32 v67, v66
	v_mov_b32_e32 v68, v66
	v_mov_b32_e32 v69, v66
	v_mov_b32_e32 v70, v66
	v_mov_b32_e32 v71, v66
	v_mov_b32_e32 v72, v66
	v_mov_b32_e32 v73, v66
	v_mov_b32_e32 v2, v66
	v_mov_b32_e32 v3, v66
	v_mov_b32_e32 v4, v66
	v_mov_b32_e32 v5, v66
	v_mov_b32_e32 v6, v66
	v_mov_b32_e32 v7, v66
	v_mov_b32_e32 v8, v66
	v_mov_b32_e32 v9, v66
	v_mov_b32_e32 v10, v66
	v_mov_b32_e32 v11, v66
	v_mov_b32_e32 v12, v66
	v_mov_b32_e32 v13, v66
	v_mov_b32_e32 v14, v66
	v_mov_b32_e32 v15, v66
	v_mov_b32_e32 v16, v66
	v_mov_b32_e32 v17, v66
	v_mov_b32_e32 v18, v66
	v_mov_b32_e32 v19, v66
	v_mov_b32_e32 v20, v66
	v_mov_b32_e32 v21, v66
	v_mov_b32_e32 v22, v66
	v_mov_b32_e32 v23, v66
	v_mov_b32_e32 v24, v66
	v_mov_b32_e32 v25, v66
	v_mov_b32_e32 v26, v66
	v_mov_b32_e32 v27, v66
	v_mov_b32_e32 v28, v66
	v_mov_b32_e32 v29, v66
	v_mov_b32_e32 v30, v66
	v_mov_b32_e32 v31, v66
	v_mov_b32_e32 v32, v66
	v_mov_b32_e32 v33, v66
	v_mov_b32_e32 v34, v66
	v_mov_b32_e32 v35, v66
	v_mov_b32_e32 v36, v66
	v_mov_b32_e32 v37, v66
	v_mov_b32_e32 v38, v66
	v_mov_b32_e32 v39, v66
	v_mov_b32_e32 v40, v66
	v_mov_b32_e32 v41, v66
	v_mov_b32_e32 v42, v66
	v_mov_b32_e32 v43, v66
	v_mov_b32_e32 v44, v66
	v_mov_b32_e32 v45, v66
	v_mov_b32_e32 v46, v66
	v_mov_b32_e32 v47, v66
	v_mov_b32_e32 v48, v66
	v_mov_b32_e32 v49, v66
	v_mov_b32_e32 v50, v66
	v_mov_b32_e32 v51, v66
	v_mov_b32_e32 v52, v66
	v_mov_b32_e32 v53, v66
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v55, v66
	v_mov_b32_e32 v56, v66
	v_mov_b32_e32 v57, v66
	v_mov_b32_e32 v58, v66
	v_mov_b32_e32 v59, v66
	v_mov_b32_e32 v60, v66
	v_mov_b32_e32 v61, v66
	v_mov_b32_e32 v62, v66
	v_mov_b32_e32 v63, v66
	v_mov_b32_e32 v64, v66
	v_mov_b32_e32 v65, v66
	v_mov_b32_e32 v74, v66
	v_mov_b32_e32 v75, v66
	v_mov_b32_e32 v76, v66
	v_mov_b32_e32 v77, v66
	v_mov_b32_e32 v78, v66
	v_mov_b32_e32 v79, v66
	v_mov_b32_e32 v80, v66
	v_mov_b32_e32 v81, v66
	v_mov_b32_e32 v82, v66
	v_mov_b32_e32 v83, v66
	v_mov_b32_e32 v84, v66
	v_mov_b32_e32 v85, v66
	v_mov_b32_e32 v86, v66
	v_mov_b32_e32 v87, v66
	v_mov_b32_e32 v88, v66
	v_mov_b32_e32 v89, v66
	v_mov_b32_e32 v90, v66
	v_mov_b32_e32 v91, v66
	v_mov_b32_e32 v92, v66
	v_mov_b32_e32 v93, v66
	v_mov_b32_e32 v94, v66
	v_mov_b32_e32 v95, v66
	v_mov_b32_e32 v96, v66
	v_mov_b32_e32 v97, v66
	v_mov_b32_e32 v98, v66
	v_mov_b32_e32 v99, v66
	v_mov_b32_e32 v100, v66
	v_mov_b32_e32 v101, v66
	v_mov_b32_e32 v102, v66
	v_mov_b32_e32 v103, v66
	v_mov_b32_e32 v104, v66
	v_mov_b32_e32 v105, v66
	v_mov_b32_e32 v106, v66
	v_mov_b32_e32 v107, v66
	v_mov_b32_e32 v108, v66
	v_mov_b32_e32 v109, v66
	v_mov_b32_e32 v110, v66
	v_mov_b32_e32 v111, v66
	v_mov_b32_e32 v112, v66
	v_mov_b32_e32 v113, v66
	v_mov_b32_e32 v114, v66
	v_mov_b32_e32 v115, v66
	v_mov_b32_e32 v116, v66
	v_mov_b32_e32 v117, v66
	v_mov_b32_e32 v118, v66
	v_mov_b32_e32 v119, v66
	v_mov_b32_e32 v120, v66
	v_mov_b32_e32 v121, v66
	v_mov_b32_e32 v122, v66
	v_mov_b32_e32 v123, v66
	v_mov_b32_e32 v124, v66
	v_mov_b32_e32 v125, v66
	v_mov_b32_e32 v126, v66
	v_mov_b32_e32 v127, v66
	v_mov_b32_e32 v128, v66
	v_mov_b32_e32 v129, v66
	s_add_i32 s50, s0, 1
	s_cmp_ge_i32 s50, s84
	s_cbranch_scc1 .LBB4_87
.LBB4_86:                               ;   in Loop: Header=BB4_84 Depth=1
	s_lshl_b32 s1, s50, 14
	s_and_b32 s1, s1, 0x4000
	s_add_i32 s52, s65, s1
	v_add_u32_e32 v138, s52, v167
	v_add_u32_e32 v151, v138, v168
	v_lshl_add_u64 v[142:143], v[130:131], 1, s[30:31]
	s_or_b32 s53, s52, 8
	v_lshrrev_b32_e32 v152, 4, v151
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
	v_lshl_add_u64 v[146:147], v[132:133], 1, s[30:31]
	v_add_u32_e32 v150, s53, v167
	v_and_b32_e32 v152, 56, v152
	;;#ASMSTART
	global_load_dwordx4 v[146:149], v[146:147], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v151, v152, v151
	;;#ASMSTART
	ds_write_b64 v151, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v142, v150, v168
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v142, v143, v142
	v_add_u32_e32 v138, v138, v171
	;;#ASMSTART
	ds_write_b64 v142, v[144:145]

	;;#ASMEND
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[146:147]

	;;#ASMEND
	v_add_u32_e32 v138, v150, v171
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	s_add_i32 s1, s69, s1
	;;#ASMSTART
	ds_write_b64 v138, v[148:149]

	;;#ASMEND
	v_add_u32_e32 v138, s1, v167
	v_add_u32_e32 v151, v138, v168
	v_lshl_add_u64 v[142:143], v[134:135], 1, s[28:29]
	s_or_b32 s52, s1, 8
	v_lshrrev_b32_e32 v152, 4, v151
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[142:145], v[142:143], off

	;;#ASMEND
	v_lshl_add_u64 v[146:147], v[136:137], 1, s[28:29]
	v_add_u32_e32 v150, s52, v167
	v_and_b32_e32 v152, 56, v152
	;;#ASMSTART
	global_load_dwordx4 v[146:149], v[146:147], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v151, v152, v151
	;;#ASMSTART
	ds_write_b64 v151, v[142:143]

	;;#ASMEND
	v_add_u32_e32 v142, v150, v168
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v142, v143, v142
	v_add_u32_e32 v138, v138, v171
	;;#ASMSTART
	ds_write_b64 v142, v[144:145]

	;;#ASMEND
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[146:147]

	;;#ASMEND
	v_add_u32_e32 v138, v150, v171
	v_lshrrev_b32_e32 v142, 4, v138
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v138, v142, v138
	;;#ASMSTART
	ds_write_b64 v138, v[148:149]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB4_87:                               ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_lshl_b32 s1, s0, 14
	s_and_b32 s1, s1, 0x4000
	s_add_i32 s52, s65, s1
	v_add_u32_e32 v228, s52, v178
	v_add_u32_e32 v142, v228, v216
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v142, v143, v142
	;;#ASMSTART
	ds_read_b64 v[164:165], v142 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v142 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v142 offset:0x800

	;;#ASMEND
	s_add_i32 s1, s69, s1
	;;#ASMSTART
	ds_read_b64 v[158:159], v142 offset:0xc00

	;;#ASMEND
	v_add_u32_e32 v138, s1, v179
	;;#ASMSTART
	ds_read_b64 v[156:157], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v142 offset:0x1400

	;;#ASMEND
	v_add_u32_e32 v148, v138, v216
	;;#ASMSTART
	ds_read_b64 v[144:145], v142 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v149, 4, v148
	;;#ASMSTART
	ds_read_b64 v[142:143], v142 offset:0x1c00

	;;#ASMEND
	v_and_b32_e32 v149, 56, v149
	v_xor_b32_e32 v154, v149, v148
	;;#ASMSTART
	ds_read_b64 v[148:149], v154 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v154 offset:0x400

	;;#ASMEND
	s_cmp_eq_u32 s37, s0
	;;#ASMSTART
	ds_read_b64 v[152:153], v154 offset:0x800

	;;#ASMEND
	s_cselect_b64 s[92:93], -1, 0
	s_cmp_lg_u32 s37, s0
	;;#ASMSTART
	ds_read_b64 v[154:155], v154 offset:0xc00

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
	s_cbranch_scc1 .LBB4_89
; %bb.88:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB4_87 Depth=2
	s_or_b64 s[0:1], s[10:11], s[6:7]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_cndmask_b32_e64 v229, 0, v164, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v165
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v165, s[6:7]
	v_cndmask_b32_sdwa v164, v229, v164, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v165, v230, v165, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v162, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v163
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v163, s[6:7]
	v_cndmask_b32_sdwa v162, v229, v162, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v163, v230, v163, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v160, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v161
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v161, s[6:7]
	v_cndmask_b32_sdwa v160, v229, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v161, v230, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v158, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v159
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v159, s[6:7]
	v_cndmask_b32_sdwa v158, v229, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v159, v230, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v156, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v157
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v157, s[6:7]
	v_cndmask_b32_sdwa v156, v229, v156, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v157, v230, v157, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v146, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v147
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v147, s[6:7]
	v_cndmask_b32_sdwa v146, v229, v146, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v147, v230, v147, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v144, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v145
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v145, s[6:7]
	v_cndmask_b32_sdwa v144, v229, v144, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v145, v230, v145, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v229, 0, v142, s[8:9]
	v_and_b32_e32 v230, 0xffff0000, v143
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v230, v230, v143, s[6:7]
	v_cndmask_b32_sdwa v142, v229, v142, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[10:11]
	v_cndmask_b32_sdwa v143, v230, v143, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB4_89:                               ;   in Loop: Header=BB4_87 Depth=2
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[6:9], v[142:143], v[148:149], v[6:9]
	v_add_u32_e32 v138, v138, v221
	s_andn2_b64 vcc, exec, s[92:93]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[142:143], v[150:151], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[142:143], v[152:153], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[142:143], v[154:155], v[66:69]
	v_add_u32_e32 v142, v228, v221
	v_lshrrev_b32_e32 v143, 4, v142
	v_and_b32_e32 v143, 56, v143
	v_mfma_f32_16x16x16_bf16 v[78:81], v[158:159], v[148:149], v[78:81]
	v_xor_b32_e32 v142, v143, v142
	v_mfma_f32_16x16x16_bf16 v[74:77], v[158:159], v[150:151], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[158:159], v[152:153], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[158:159], v[154:155], v[58:61]
	;;#ASMSTART
	ds_read_b64 v[158:159], v142 offset:0

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[94:97], v[160:161], v[148:149], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[160:161], v[150:151], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[160:161], v[152:153], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[160:161], v[154:155], v[82:85]
	;;#ASMSTART
	ds_read_b64 v[160:161], v142 offset:0x400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[54:57], v[156:157], v[148:149], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[156:157], v[150:151], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[156:157], v[152:153], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[156:157], v[154:155], v[42:45]
	;;#ASMSTART
	ds_read_b64 v[156:157], v142 offset:0x800

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[118:121], v[164:165], v[152:153], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[162:163], v[152:153], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[146:147], v[152:153], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[144:145], v[152:153], v[14:17]
	;;#ASMSTART
	ds_read_b64 v[152:153], v142 offset:0xc00

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[114:117], v[164:165], v[154:155], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[162:163], v[154:155], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[146:147], v[154:155], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[154:155], v[10:13]
	;;#ASMSTART
	ds_read_b64 v[154:155], v142 offset:0x1000

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[122:125], v[164:165], v[150:151], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[162:163], v[150:151], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[146:147], v[150:151], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[144:145], v[150:151], v[18:21]
	;;#ASMSTART
	ds_read_b64 v[150:151], v142 offset:0x1400

	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[126:129], v[164:165], v[148:149], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[162:163], v[148:149], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[146:147], v[148:149], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[144:145], v[148:149], v[22:25]
	;;#ASMSTART
	ds_read_b64 v[148:149], v142 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v144, 4, v138
	;;#ASMSTART
	ds_read_b64 v[142:143], v142 offset:0x1c00

	;;#ASMEND
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v138, v144, v138
	;;#ASMSTART
	ds_read_b64 v[162:163], v138 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v138 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v138 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v138 offset:0xc00

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
	s_cbranch_vccnz .LBB4_91
; %bb.90:                               ; %.loopexit.i.1
                                        ;   in Loop: Header=BB4_87 Depth=2
	s_or_b64 s[0:1], s[18:19], s[14:15]
	s_or_b64 s[0:1], s[0:1], s[12:13]
	v_cndmask_b32_e64 v138, 0, v158, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v159
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v159, s[14:15]
	v_cndmask_b32_sdwa v158, v138, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v159, v228, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v160, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v161
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v161, s[14:15]
	v_cndmask_b32_sdwa v160, v138, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v161, v228, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v156, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v157
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v157, s[14:15]
	v_cndmask_b32_sdwa v156, v138, v156, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v157, v228, v157, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v152, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v153
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v153, s[14:15]
	v_cndmask_b32_sdwa v152, v138, v152, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v153, v228, v153, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v154, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v155
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v155, s[14:15]
	v_cndmask_b32_sdwa v154, v138, v154, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v155, v228, v155, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v150, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v151
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v151, s[14:15]
	v_cndmask_b32_sdwa v150, v138, v150, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v151, v228, v151, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v148, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v149
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v149, s[14:15]
	v_cndmask_b32_sdwa v148, v138, v148, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v149, v228, v149, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v138, 0, v142, s[16:17]
	v_and_b32_e32 v228, 0xffff0000, v143
	s_mov_b64 vcc, s[0:1]
	v_cndmask_b32_e64 v228, v228, v143, s[14:15]
	v_cndmask_b32_sdwa v142, v138, v142, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[18:19]
	v_cndmask_b32_sdwa v143, v228, v143, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB4_91:                               ;   in Loop: Header=BB4_87 Depth=2
	s_add_u32 s28, s28, 64
	v_mfma_f32_16x16x16_bf16 v[126:129], v[158:159], v[162:163], v[126:129]
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 64
	s_addc_u32 s31, s31, 0
	v_mfma_f32_16x16x16_bf16 v[122:125], v[158:159], v[164:165], v[122:125]
	s_cmp_eq_u32 s83, s50
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[118:121], v[158:159], v[146:147], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[158:159], v[144:145], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[160:161], v[162:163], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[160:161], v[164:165], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[160:161], v[146:147], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[160:161], v[144:145], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[156:157], v[162:163], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[156:157], v[164:165], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[156:157], v[146:147], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[156:157], v[144:145], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[152:153], v[162:163], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[152:153], v[164:165], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[152:153], v[146:147], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[152:153], v[144:145], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[154:155], v[162:163], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[154:155], v[164:165], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[154:155], v[146:147], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[154:155], v[144:145], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[150:151], v[162:163], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[150:151], v[164:165], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[150:151], v[146:147], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[150:151], v[144:145], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[148:149], v[162:163], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[148:149], v[164:165], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[148:149], v[146:147], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[148:149], v[144:145], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[142:143], v[162:163], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[142:143], v[164:165], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[142:143], v[146:147], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[142:143], v[144:145], v[66:69]
	s_cbranch_scc1 .LBB4_93
; %bb.92:                               ;   in Loop: Header=BB4_87 Depth=2
	s_mov_b32 s0, s50
	s_add_i32 s50, s0, 1
	s_cmp_ge_i32 s50, s84
	s_cbranch_scc0 .LBB4_86
	s_branch .LBB4_87
.LBB4_93:                               ; %Flow2172
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v232, 39
	v_readlane_b32 s29, v232, 40
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_102
; %bb.94:                               ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v232, 35
	v_readlane_b32 s29, v232, 36
	s_and_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB4_102
; %bb.95:                               ;   in Loop: Header=BB4_84 Depth=1
	v_add_u32_e32 v138, s64, v180
	v_sub_u32_e32 v143, 0, v138
	v_max_i32_e32 v143, v138, v143
	v_mul_hi_u32 v144, v143, s71
	v_mul_lo_u32 v145, v144, s70
	v_sub_u32_e32 v143, v143, v145
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s70, v143
	v_ashrrev_i32_e32 v142, 31, v138
	v_xor_b32_e32 v142, s23, v142
	v_cndmask_b32_e32 v144, v144, v145, vcc
	v_subrev_u32_e32 v145, s70, v143
	v_cndmask_b32_e32 v143, v143, v145, vcc
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s70, v143
	s_nop 1
	v_cndmask_b32_e32 v143, v144, v145, vcc
	v_xor_b32_e32 v143, v143, v142
	v_sub_u32_e32 v142, v143, v142
	v_mul_lo_u32 v143, v142, s33
	v_sub_u32_e32 v138, v138, v143
	v_sub_u32_e32 v144, 0, v138
	v_ashrrev_i32_e32 v143, 31, v138
	v_max_i32_e32 v138, v138, v144
	v_mul_hi_u32 v144, v138, s35
	v_mul_lo_u32 v145, v144, s39
	v_sub_u32_e32 v138, v138, v145
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s39, v138
	v_xor_b32_e32 v143, s34, v143
	s_nop 0
	v_cndmask_b32_e32 v144, v144, v145, vcc
	v_subrev_u32_e32 v145, s39, v138
	v_cndmask_b32_e32 v138, v138, v145, vcc
	v_add_u32_e32 v145, 1, v144
	v_cmp_le_u32_e32 vcc, s39, v138
	s_nop 1
	v_cndmask_b32_e32 v138, v144, v145, vcc
	v_xor_b32_e32 v138, v138, v143
	v_sub_u32_e32 v138, v138, v143
	v_mad_u64_u32 v[142:143], s[28:29], v142, s40, v[138:139]
	v_mul_lo_u32 v138, v142, s41
	v_add_u32_e32 v142, s27, v138
	v_readlane_b32 s28, v232, 11
	v_ashrrev_i32_e32 v143, 31, v142
	v_readlane_b32 s29, v232, 12
	s_nop 1
	v_lshl_add_u64 v[142:143], v[142:143], 2, s[28:29]
	flat_load_dword v138, v[142:143] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v138, v181
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB4_102
; %bb.96:                               ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_mov_b64 s[28:29], 0
	s_mov_b64 s[94:95], 0
                                        ; implicit-def: $sgpr30_sgpr31
                                        ; implicit-def: $sgpr92_sgpr93
	s_branch .LBB4_98
.LBB4_97:                               ; %Flow2167
                                        ;   in Loop: Header=BB4_98 Depth=2
	s_and_b64 s[52:53], exec, s[92:93]
	s_or_b64 s[28:29], s[52:53], s[28:29]
	s_andn2_b64 s[30:31], s[30:31], exec
	s_and_b64 s[52:53], s[96:97], exec
	s_or_b64 s[30:31], s[30:31], s[52:53]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB4_100
.LBB4_98:                               ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s94, s94, 1
	s_addc_u32 s95, s95, 0
	v_mov_b64_e32 v[144:145], s[58:59]
	v_cmp_gt_u64_e32 vcc, s[94:95], v[144:145]
	s_mov_b64 s[96:97], -1
	s_or_b64 s[92:93], s[92:93], exec
	s_cbranch_vccnz .LBB4_97
; %bb.99:                               ;   in Loop: Header=BB4_98 Depth=2
	s_sleep 4
	flat_load_dword v138, v[142:143] offset:256 sc0 sc1
	s_andn2_b64 s[52:53], s[92:93], exec
	s_mov_b64 s[96:97], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v138, v181
	s_and_b64 s[54:55], vcc, exec
	s_or_b64 s[92:93], s[52:53], s[54:55]
	s_branch .LBB4_97
.LBB4_100:                              ; %loop.exit.guard2115
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_xor_b64 s[28:29], exec, s[28:29]
	s_cbranch_execz .LBB4_102
; %bb.101:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s28, v232, 0
	v_readlane_b32 s30, v232, 2
	v_readlane_b32 s31, v232, 3
	v_readlane_b32 s29, v232, 1
	s_nop 0
	v_mov_b64_e32 v[142:143], s[30:31]
	flat_atomic_or v[142:143], v227
.LBB4_102:                              ; %.critedge712
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[66:67]
	s_mov_b64 s[28:29], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB4_106
; %bb.103:                              ;   in Loop: Header=BB4_84 Depth=1
	v_mov_b32_e32 v138, 0
	s_mov_b64 s[28:29], exec
	v_readlane_b32 s30, v232, 37
	v_readlane_b32 s31, v232, 38
	s_and_b64 s[30:31], s[28:29], s[30:31]
	s_mov_b64 exec, s[30:31]
	s_cbranch_execz .LBB4_105
; %bb.104:                              ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s52, v232, 0
	v_readlane_b32 s54, v232, 2
	v_readlane_b32 s55, v232, 3
	v_readlane_b32 s53, v232, 1
	s_nop 0
	v_mov_b64_e32 v[142:143], s[54:55]
	flat_load_dword v138, v[142:143] sc1
.LBB4_105:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit270
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v138, v226, v138
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v138, 0x2000000, v138
	v_cmp_eq_u32_e64 s[28:29], 0, v138
.LBB4_106:                              ; %Flow2175
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_and_saveexec_b64 s[30:31], s[28:29]
	s_cbranch_execz .LBB4_83
; %bb.107:                              ; %.critedge720
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_writelane_b32 v232, s30, 52
	s_nop 1
	v_writelane_b32 v232, s31, 53
	s_nop 0
	v_readlane_b32 s0, v232, 20
	v_readlane_b32 s1, v232, 21
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB4_473
; %bb.108:                              ; %.lr.ph838
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_or_b32_e32 v138, s62, v182
	v_readlane_b32 s0, v232, 13
	s_sub_i32 s63, s38, s62
	v_cmp_gt_i32_e32 vcc, s38, v138
	v_mov_b32_e32 v148, s0
	s_min_i32 s66, s63, 0x100
	v_cndmask_b32_e32 v142, v148, v138, vcc
	v_or_b32_e32 v138, s62, v188
	v_cmp_gt_i32_e32 vcc, s38, v138
	s_abs_i32 s67, s66
	v_cvt_f32_u32_e32 v150, s67
	v_cndmask_b32_e32 v144, v148, v138, vcc
	v_or_b32_e32 v138, s62, v189
	v_cmp_gt_i32_e32 vcc, s38, v138
	s_ashr_i32 s53, s66, 31
	s_sub_i32 s0, 0, s67
	v_cndmask_b32_e32 v146, v148, v138, vcc
	v_or_b32_e32 v138, s62, v190
	v_cmp_gt_i32_e32 vcc, s38, v138
	s_lshl_b32 s1, s3, 8
	v_readlane_b32 s28, v232, 0
	v_cndmask_b32_e32 v148, v148, v138, vcc
	v_rcp_iflag_f32_e32 v138, v150
	v_ashrrev_i32_e32 v143, 31, v142
	v_readlane_b32 s29, v232, 1
	v_ashrrev_i32_e32 v145, 31, v144
	v_mul_f32_e32 v138, 0x4f7ffffe, v138
	v_cvt_u32_f32_e32 v138, v138
	v_ashrrev_i32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v149, 31, v148
	s_mul_i32 s52, s66, s20
	v_mul_lo_u32 v150, s0, v138
	s_lshl_b32 s0, s53, 9
	v_subrev_u32_e32 v161, s0, v224
	s_lshl_b32 s0, s51, 8
	s_sub_i32 s51, s0, s1
	v_readlane_b32 s0, v232, 46
	s_lshl_b32 s0, s0, 2
	s_add_i32 s0, s2, s0
	v_readlane_b32 s1, v232, 48
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v232, 50
	s_sub_i32 s0, s0, s1
	v_readlane_b32 s1, v232, 45
	s_lshl_b32 s1, s1, 2
	s_sub_i32 s0, s0, s1
	v_mul_hi_u32 v150, v138, v150
	s_lshl_b32 s0, s0, 8
	v_lshl_add_u64 v[142:143], v[142:143], 1, s[28:29]
	v_lshl_add_u64 v[144:145], v[144:145], 1, s[28:29]
	v_lshl_add_u64 v[146:147], v[146:147], 1, s[28:29]
	v_lshl_add_u64 v[148:149], v[148:149], 1, s[28:29]
	v_cmp_gt_i32_e64 s[28:29], s52, v0
	s_mov_b32 s54, 0
	v_add_u32_e32 v160, v138, v150
	s_lshl_b32 s55, s66, 1
	s_sub_i32 s50, 0, s66
	s_add_i32 s3, s21, s0
	v_readlane_b32 s30, v232, 2
	v_readlane_b32 s31, v232, 3
	s_branch .LBB4_110
.LBB4_109:                              ; %._crit_edge836
                                        ;   in Loop: Header=BB4_110 Depth=2
	s_add_i32 s54, s54, 1
	s_add_i32 s3, s3, s42
	s_cmp_eq_u32 s54, s22
	s_cbranch_scc1 .LBB4_473
.LBB4_110:                              ;   Parent Loop BB4_84 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB4_114 Depth 3
                                        ;         Child Loop BB4_472 Depth 4
                                        ;         Child Loop BB4_428 Depth 4
                                        ;         Child Loop BB4_455 Depth 4
                                        ;         Child Loop BB4_462 Depth 4
                                        ;           Child Loop BB4_467 Depth 5
	s_andn2_b64 vcc, exec, s[76:77]
	s_cbranch_vccnz .LBB4_109
; %bb.111:                              ; %.lr.ph835.preheader
                                        ;   in Loop: Header=BB4_110 Depth=2
	v_mov_b64_e32 v[158:159], s[44:45]
	flat_load_dwordx4 v[150:153], v[158:159]
	flat_load_dwordx4 v[154:157], v[158:159] offset:16
	flat_load_dwordx4 v[162:165], v[158:159] offset:32
	flat_load_dwordx4 v[228:231], v[158:159] offset:48
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
	v_cndmask_b32_e32 v138, 0, v152, vcc
	v_cndmask_b32_e32 v152, 0, v153, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 2
	v_cndmask_b32_e32 v152, v152, v155, vcc
	v_cndmask_b32_e32 v138, v138, v154, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 3
	v_cndmask_b32_e32 v138, v138, v156, vcc
	v_cndmask_b32_e32 v152, v152, v157, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 4
	v_cndmask_b32_e32 v152, v152, v163, vcc
	v_cndmask_b32_e32 v138, v138, v162, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 5
	v_cndmask_b32_e32 v138, v138, v164, vcc
	v_cndmask_b32_e32 v152, v152, v165, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 6
	v_cndmask_b32_e32 v152, v152, v229, vcc
	v_cndmask_b32_e32 v138, v138, v228, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s30, 7
	v_cndmask_b32_e32 v138, v138, v230, vcc
	v_cndmask_b32_e32 v152, v152, v231, vcc
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
	v_cndmask_b32_e32 v153, v152, v159, vcc
	v_cndmask_b32_e32 v152, v138, v158, vcc
	v_sub_co_u32_e32 v150, vcc, s56, v150
	v_mov_b32_e32 v138, s57
	s_add_i32 s1, s1, s30
	s_sub_i32 s0, s68, s0
	v_subb_co_u32_e32 v151, vcc, v138, v151, vcc
	s_mul_i32 s1, s1, s60
	s_sub_i32 s0, s0, s72
	v_lshl_add_u64 v[150:151], v[150:151], 0, v[152:153]
	v_cmp_ne_u64_e32 vcc, 0, v[152:153]
	s_add_i32 s30, s1, s62
	s_mul_i32 s0, s60, s0
	v_cndmask_b32_e32 v153, 0, v151, vcc
	v_cndmask_b32_e32 v152, 0, v150, vcc
	s_ashr_i32 s31, s30, 31
	s_add_i32 s0, s51, s0
	v_lshl_add_u64 v[150:151], s[30:31], 1, v[152:153]
	v_lshl_add_u64 v[152:153], v[152:153], 0, v[140:141]
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[152:153], s[0:1], 1, v[152:153]
	s_branch .LBB4_114
.LBB4_112:                              ; %Flow2159
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB4_113:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_add_i32 s92, s92, s20
	s_cmp_ge_i32 s92, s42
	v_lshl_add_u64 v[152:153], v[152:153], 0, s[86:87]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB4_109
.LBB4_114:                              ; %.lr.ph835
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB4_472 Depth 4
                                        ;         Child Loop BB4_428 Depth 4
                                        ;         Child Loop BB4_455 Depth 4
                                        ;         Child Loop BB4_462 Depth 4
                                        ;           Child Loop BB4_467 Depth 5
	v_cndmask_b32_e64 v138, 0, 1, s[78:79]
	v_cmp_ne_u32_e64 s[30:31], 1, v138
	s_andn2_b64 vcc, exec, s[78:79]
	s_cbranch_vccnz .LBB4_116
; %bb.115:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_117
.LBB4_116:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_117:                              ;   in Loop: Header=BB4_114 Depth=3
	s_add_i32 s93, s92, s61
	s_add_i32 s68, s93, s20
	v_cmp_le_i32_e32 vcc, s93, v183
	v_cmp_gt_i32_e64 s[0:1], s68, v183
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_119
; %bb.118:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v126
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v183
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_119:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v185
	v_cmp_gt_i32_e64 s[0:1], s68, v185
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_121
; %bb.120:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v127
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v185
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_121:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v186
	v_cmp_gt_i32_e64 s[0:1], s68, v186
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_123
; %bb.122:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v128
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v186
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_123:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v187
	v_cmp_gt_i32_e64 s[0:1], s68, v187
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_125
; %bb.124:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v129
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v187
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_125:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_442
; %bb.126:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_128
.LBB4_127:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v122
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v183
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_128:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_145
; %bb.129:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_146
.LBB4_130:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_147
.LBB4_131:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_148
.LBB4_132:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_134
.LBB4_133:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v118
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v183
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_134:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_149
; %bb.135:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_150
.LBB4_136:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_151
.LBB4_137:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_152
.LBB4_138:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_140
.LBB4_139:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v114
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v183
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_140:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_153
; %bb.141:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_154
.LBB4_142:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_155
.LBB4_143:                              ; %.preheader.1.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_156
.LBB4_144:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_157
.LBB4_145:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v123
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v185
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_130
.LBB4_146:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v124
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v186
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_131
.LBB4_147:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v125
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v187
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_132
.LBB4_148:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_133
	s_branch .LBB4_134
.LBB4_149:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v119
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v185
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_136
.LBB4_150:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v120
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v186
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_137
.LBB4_151:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v121
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v187
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_138
.LBB4_152:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_139
	s_branch .LBB4_140
.LBB4_153:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v115
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v185
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_142
.LBB4_154:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v116
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v186
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_143
.LBB4_155:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v117
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v187
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_144
.LBB4_156:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_157:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v191
	v_cmp_gt_i32_e64 s[0:1], s68, v191
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_159
; %bb.158:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v110
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v191
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_159:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v192
	v_cmp_gt_i32_e64 s[0:1], s68, v192
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_161
; %bb.160:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v111
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v192
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_161:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v193
	v_cmp_gt_i32_e64 s[0:1], s68, v193
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_163
; %bb.162:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v112
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v193
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_163:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v194
	v_cmp_gt_i32_e64 s[0:1], s68, v194
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_165
; %bb.164:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v113
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v194
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_165:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_443
; %bb.166:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_168
.LBB4_167:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v106
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v191
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_168:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_185
; %bb.169:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_186
.LBB4_170:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_187
.LBB4_171:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_188
.LBB4_172:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_174
.LBB4_173:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v102
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v191
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_174:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_189
; %bb.175:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_190
.LBB4_176:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_191
.LBB4_177:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_192
.LBB4_178:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_180
.LBB4_179:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v98
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v191
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_180:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_193
; %bb.181:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_194
.LBB4_182:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_195
.LBB4_183:                              ; %.preheader.2.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_196
.LBB4_184:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_197
.LBB4_185:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v107
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v192
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_170
.LBB4_186:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v108
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v193
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_171
.LBB4_187:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v109
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v194
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_172
.LBB4_188:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_173
	s_branch .LBB4_174
.LBB4_189:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v103
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v192
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_176
.LBB4_190:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v104
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v193
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_177
.LBB4_191:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v105
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v194
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_178
.LBB4_192:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_179
	s_branch .LBB4_180
.LBB4_193:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v99
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v192
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_182
.LBB4_194:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v100
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v193
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_183
.LBB4_195:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v101
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v194
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_184
.LBB4_196:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_197:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v195
	v_cmp_gt_i32_e64 s[0:1], s68, v195
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_199
; %bb.198:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v94
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v195
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_199:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v196
	v_cmp_gt_i32_e64 s[0:1], s68, v196
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_201
; %bb.200:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v95
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v196
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_201:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v197
	v_cmp_gt_i32_e64 s[0:1], s68, v197
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_203
; %bb.202:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v96
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v197
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_203:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v198
	v_cmp_gt_i32_e64 s[0:1], s68, v198
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_205
; %bb.204:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v97
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v198
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_205:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_444
; %bb.206:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_208
.LBB4_207:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v90
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v195
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_208:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_225
; %bb.209:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_226
.LBB4_210:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_227
.LBB4_211:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_228
.LBB4_212:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_214
.LBB4_213:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v86
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v195
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_214:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_229
; %bb.215:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_230
.LBB4_216:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_231
.LBB4_217:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_232
.LBB4_218:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_220
.LBB4_219:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v82
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v195
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_220:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_233
; %bb.221:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_234
.LBB4_222:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_235
.LBB4_223:                              ; %.preheader.3.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_236
.LBB4_224:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_237
.LBB4_225:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v91
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v196
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_210
.LBB4_226:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v92
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v197
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_211
.LBB4_227:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v93
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v198
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_212
.LBB4_228:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_213
	s_branch .LBB4_214
.LBB4_229:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v87
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v196
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_216
.LBB4_230:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v88
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v197
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_217
.LBB4_231:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v89
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v198
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_218
.LBB4_232:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_219
	s_branch .LBB4_220
.LBB4_233:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v83
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v196
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_222
.LBB4_234:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v84
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v197
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_223
.LBB4_235:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v85
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v198
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_224
.LBB4_236:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_237:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v199
	v_cmp_gt_i32_e64 s[0:1], s68, v199
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_239
; %bb.238:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v78
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v199
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_239:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v200
	v_cmp_gt_i32_e64 s[0:1], s68, v200
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_241
; %bb.240:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v79
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v200
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_241:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v201
	v_cmp_gt_i32_e64 s[0:1], s68, v201
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_243
; %bb.242:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v80
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v201
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_243:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v202
	v_cmp_gt_i32_e64 s[0:1], s68, v202
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_245
; %bb.244:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v81
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v202
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_245:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_445
; %bb.246:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_248
.LBB4_247:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v74
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v199
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_248:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_265
; %bb.249:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_266
.LBB4_250:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_267
.LBB4_251:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_268
.LBB4_252:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_254
.LBB4_253:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v62
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v199
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_254:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_269
; %bb.255:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_270
.LBB4_256:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_271
.LBB4_257:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_272
.LBB4_258:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_260
.LBB4_259:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v58
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v199
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_260:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_273
; %bb.261:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_274
.LBB4_262:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_275
.LBB4_263:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_276
.LBB4_264:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_277
.LBB4_265:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v75
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v200
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_250
.LBB4_266:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v76
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v201
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_251
.LBB4_267:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v77
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v202
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_252
.LBB4_268:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_253
	s_branch .LBB4_254
.LBB4_269:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v63
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v200
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_256
.LBB4_270:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v64
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v201
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_257
.LBB4_271:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v65
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v202
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_258
.LBB4_272:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_259
	s_branch .LBB4_260
.LBB4_273:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v59
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v200
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_262
.LBB4_274:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v60
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v201
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_263
.LBB4_275:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v61
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v202
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_264
.LBB4_276:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_277:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v203
	v_cmp_gt_i32_e64 s[0:1], s68, v203
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_279
; %bb.278:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v54
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v203
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_279:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v204
	v_cmp_gt_i32_e64 s[0:1], s68, v204
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_281
; %bb.280:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v55
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v204
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_281:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v205
	v_cmp_gt_i32_e64 s[0:1], s68, v205
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_283
; %bb.282:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v56
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v205
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_283:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v206
	v_cmp_gt_i32_e64 s[0:1], s68, v206
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_285
; %bb.284:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v57
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v206
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_285:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_446
; %bb.286:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_288
.LBB4_287:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v50
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v203
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_288:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_305
; %bb.289:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_306
.LBB4_290:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_307
.LBB4_291:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_308
.LBB4_292:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_294
.LBB4_293:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v46
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v203
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_294:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_309
; %bb.295:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_310
.LBB4_296:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_311
.LBB4_297:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_312
.LBB4_298:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_300
.LBB4_299:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v42
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v203
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_300:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_313
; %bb.301:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_314
.LBB4_302:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_315
.LBB4_303:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_316
.LBB4_304:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_317
.LBB4_305:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v51
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v204
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_290
.LBB4_306:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v52
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v205
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_291
.LBB4_307:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v53
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v206
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_292
.LBB4_308:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_293
	s_branch .LBB4_294
.LBB4_309:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v47
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v204
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_296
.LBB4_310:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v48
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v205
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_297
.LBB4_311:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v49
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v206
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_298
.LBB4_312:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_299
	s_branch .LBB4_300
.LBB4_313:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v43
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v204
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_302
.LBB4_314:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v44
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v205
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_303
.LBB4_315:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v45
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v206
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_304
.LBB4_316:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_317:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v207
	v_cmp_gt_i32_e64 s[0:1], s68, v207
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_319
; %bb.318:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v38
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v207
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_319:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v208
	v_cmp_gt_i32_e64 s[0:1], s68, v208
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_321
; %bb.320:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v39
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v208
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_321:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v209
	v_cmp_gt_i32_e64 s[0:1], s68, v209
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_323
; %bb.322:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v40
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v209
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_323:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v210
	v_cmp_gt_i32_e64 s[0:1], s68, v210
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_325
; %bb.324:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v41
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v210
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_325:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_447
; %bb.326:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_328
.LBB4_327:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v34
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v207
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_328:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_345
; %bb.329:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_346
.LBB4_330:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_347
.LBB4_331:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_348
.LBB4_332:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_334
.LBB4_333:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v30
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v207
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_334:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_349
; %bb.335:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_350
.LBB4_336:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_351
.LBB4_337:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_352
.LBB4_338:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_340
.LBB4_339:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v26
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v207
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_340:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_353
; %bb.341:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_354
.LBB4_342:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_355
.LBB4_343:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_356
.LBB4_344:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_357
.LBB4_345:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v35
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v208
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_330
.LBB4_346:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v36
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v209
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_331
.LBB4_347:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v37
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v210
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_332
.LBB4_348:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_333
	s_branch .LBB4_334
.LBB4_349:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v31
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v208
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_336
.LBB4_350:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v32
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v209
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_337
.LBB4_351:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v33
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v210
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_338
.LBB4_352:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_339
	s_branch .LBB4_340
.LBB4_353:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v27
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v208
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_342
.LBB4_354:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v28
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v209
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_343
.LBB4_355:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v29
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v210
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_344
.LBB4_356:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_357:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v211
	v_cmp_gt_i32_e64 s[0:1], s68, v211
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_359
; %bb.358:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v22
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v211
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_359:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v212
	v_cmp_gt_i32_e64 s[0:1], s68, v212
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_361
; %bb.360:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v23
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v212
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_361:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v213
	v_cmp_gt_i32_e64 s[0:1], s68, v213
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_363
; %bb.362:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v24
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v213
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_363:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v214
	v_cmp_gt_i32_e64 s[0:1], s68, v214
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_365
; %bb.364:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v25
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v214
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_365:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_448
; %bb.366:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_368
.LBB4_367:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v18
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v211
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_368:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_385
; %bb.369:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_386
.LBB4_370:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_387
.LBB4_371:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_388
.LBB4_372:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_374
.LBB4_373:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v14
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v211
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_374:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_389
; %bb.375:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_390
.LBB4_376:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_391
.LBB4_377:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_392
.LBB4_378:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_380
.LBB4_379:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v10
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v211
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_380:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_393
; %bb.381:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_394
.LBB4_382:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_395
.LBB4_383:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_396
.LBB4_384:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[142:143]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_branch .LBB4_397
.LBB4_385:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v19
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v212
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_370
.LBB4_386:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v20
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v213
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_371
.LBB4_387:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v21
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v214
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_372
.LBB4_388:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_373
	s_branch .LBB4_374
.LBB4_389:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v15
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v212
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_376
.LBB4_390:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v16
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v213
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_377
.LBB4_391:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v17
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v214
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_378
.LBB4_392:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_379
	s_branch .LBB4_380
.LBB4_393:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v11
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v212
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_382
.LBB4_394:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v12
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v213
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_383
.LBB4_395:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v13
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v214
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_384
.LBB4_396:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
.LBB4_397:                              ;   in Loop: Header=BB4_114 Depth=3
	v_cmp_le_i32_e32 vcc, s93, v215
	v_cmp_gt_i32_e64 s[0:1], s68, v215
	s_and_b64 s[94:95], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[94:95]
	s_cbranch_execz .LBB4_399
; %bb.398:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v6
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v215
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_399:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v218
	v_cmp_gt_i32_e64 s[0:1], s68, v218
	s_and_b64 s[96:97], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[96:97]
	s_cbranch_execz .LBB4_401
; %bb.400:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v7
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v218
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_401:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v219
	v_cmp_gt_i32_e64 s[0:1], s68, v219
	s_and_b64 s[98:99], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[98:99]
	s_cbranch_execz .LBB4_403
; %bb.402:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v8
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v219
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154
.LBB4_403:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cmp_le_i32_e32 vcc, s93, v220
	v_cmp_gt_i32_e64 s[0:1], s68, v220
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_405
; %bb.404:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v9
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v220
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138
.LBB4_405:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_449
; %bb.406:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[144:145]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_408
.LBB4_407:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v2
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v215
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
.LBB4_408:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_432
; %bb.409:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_433
.LBB4_410:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_434
.LBB4_411:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_435
.LBB4_412:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[146:147]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execz .LBB4_414
.LBB4_413:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v70
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v215
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
.LBB4_414:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[96:97]
	s_cbranch_execnz .LBB4_436
; %bb.415:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execnz .LBB4_437
.LBB4_416:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execnz .LBB4_438
.LBB4_417:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_439
.LBB4_418:                              ;   in Loop: Header=BB4_114 Depth=3
	flat_load_ushort v138, v[148:149]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v138, 16, v138
	s_and_saveexec_b64 s[30:31], s[94:95]
	s_cbranch_execz .LBB4_420
.LBB4_419:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v66
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v215
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
.LBB4_420:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[96:97]
	s_cbranch_execnz .LBB4_440
; %bb.421:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[98:99]
	s_cbranch_execnz .LBB4_441
.LBB4_422:                              ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB4_424
.LBB4_423:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v69
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v220
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:96
.LBB4_424:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi128ELi64ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_mul_i32 s0, s75, s92
	s_mul_hi_u32 s1, s74, s92
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s74, s92
	v_lshl_add_u64 v[154:155], s[0:1], 1, v[150:151]
	s_andn2_b64 vcc, exec, s[80:81]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB4_450
; %bb.425:                              ;   in Loop: Header=BB4_114 Depth=3
	s_mov_b64 s[30:31], -1
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB4_452
; %bb.426:                              ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v138, v166
	v_mov_b32_e32 v156, v217
	v_mov_b32_e32 v157, v223
	v_mov_b32_e32 v158, v0
                                        ; implicit-def: $sgpr94_sgpr95
                                        ; implicit-def: $sgpr96_sgpr97
	s_branch .LBB4_428
.LBB4_427:                              ; %Flow2154
                                        ;   in Loop: Header=BB4_428 Depth=4
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 s[72:73], exec, vcc
	s_or_b64 s[30:31], s[72:73], s[30:31]
	s_andn2_b64 s[72:73], s[94:95], exec
	s_and_b64 s[94:95], s[96:97], exec
	s_or_b64 s[94:95], s[72:73], s[94:95]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB4_451
.LBB4_428:                              ; %.lr.ph.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ;       Parent Loop BB4_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_and_b32_e32 v159, 0xf8, v138
	v_add_u32_e32 v162, 8, v159
	v_cmp_lt_i32_e64 s[98:99], s66, v162
	v_cmp_ge_i32_e32 vcc, s66, v162
	s_and_saveexec_b64 s[72:73], vcc
	s_cbranch_execz .LBB4_430
; %bb.429:                              ;   in Loop: Header=BB4_428 Depth=4
	v_mul_lo_u32 v162, s74, v156
	v_lshlrev_b32_e32 v162, 1, v162
	v_lshlrev_b32_e32 v159, 1, v159
	v_add3_u32 v162, v154, v162, v159
	v_add_u32_e32 v159, v157, v159
	v_or_b32_e32 v159, v159, v162
	v_and_b32_e32 v159, 15, v159
	v_cmp_eq_u32_e32 vcc, 0, v159
	s_andn2_b64 s[98:99], s[98:99], exec
	s_and_b64 vcc, vcc, exec
	s_or_b64 s[98:99], s[98:99], vcc
.LBB4_430:                              ; %Flow2153
                                        ;   in Loop: Header=BB4_428 Depth=4
	s_or_b64 exec, exec, s[72:73]
	s_mov_b64 vcc, -1
	s_andn2_b64 s[96:97], s[96:97], exec
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_427
; %bb.431:                              ; %.critedge.i
                                        ;   in Loop: Header=BB4_428 Depth=4
	v_add_u32_e32 v158, 0x200, v158
	v_cmp_le_i32_e32 vcc, s82, v158
	v_add_u32_e32 v157, 0x2000, v157
	v_add_u32_e32 v156, 16, v156
	v_add_u32_e32 v138, 0x1000, v138
	s_or_b64 s[96:97], s[96:97], exec
	s_orn2_b64 vcc, vcc, exec
	s_branch .LBB4_427
.LBB4_432:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v3
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v218
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_410
.LBB4_433:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v4
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v219
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_411
.LBB4_434:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v5
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v220
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:32
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_412
.LBB4_435:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_413
	s_branch .LBB4_414
.LBB4_436:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v71
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v218
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[98:99]
	s_cbranch_execz .LBB4_416
.LBB4_437:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v72
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v219
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_saveexec_b64 s[72:73], s[0:1]
	s_cbranch_execz .LBB4_417
.LBB4_438:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v138, v138, v73
	v_bfe_u32 v154, v138, 16, 1
	v_add3_u32 v138, v138, v154, s26
	v_subrev_u32_e32 v154, s93, v220
	v_lshl_add_u32 v154, v154, 9, v184
	ds_write_b16_d16_hi v154, v138 offset:64
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB4_418
.LBB4_439:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[30:31], s[94:95]
	s_cbranch_execnz .LBB4_419
	s_branch .LBB4_420
.LBB4_440:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v67
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v218
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[98:99]
	s_cbranch_execz .LBB4_422
.LBB4_441:                              ;   in Loop: Header=BB4_114 Depth=3
	v_add_f32_e32 v154, v138, v68
	v_bfe_u32 v155, v154, 16, 1
	v_add3_u32 v154, v154, v155, s26
	v_subrev_u32_e32 v155, s93, v219
	v_lshl_add_u32 v155, v155, 9, v184
	ds_write_b16_d16_hi v155, v154 offset:96
	s_or_b64 exec, exec, s[30:31]
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execnz .LBB4_423
	s_branch .LBB4_424
.LBB4_442:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_127
	s_branch .LBB4_128
.LBB4_443:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_167
	s_branch .LBB4_168
.LBB4_444:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_207
	s_branch .LBB4_208
.LBB4_445:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_247
	s_branch .LBB4_248
.LBB4_446:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_287
	s_branch .LBB4_288
.LBB4_447:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_327
	s_branch .LBB4_328
.LBB4_448:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_367
	s_branch .LBB4_368
.LBB4_449:                              ;   in Loop: Header=BB4_114 Depth=3
	v_mov_b32_e32 v138, 0
	s_and_saveexec_b64 s[72:73], s[94:95]
	s_cbranch_execnz .LBB4_407
	s_branch .LBB4_408
.LBB4_450:                              ;   in Loop: Header=BB4_114 Depth=3
	s_cbranch_execz .LBB4_113
	s_branch .LBB4_470
.LBB4_451:                              ; %Flow2155
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[30:31]
	s_orn2_b64 s[30:31], s[94:95], exec
.LBB4_452:                              ; %Flow2156
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	v_cndmask_b32_e64 v138, 0, 1, s[30:31]
	s_nop 0
	v_readfirstlane_b32 s0, v138
	s_bitcmp1_b32 s0, 0
	s_cselect_b64 s[30:31], -1, 0
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB4_457
; %bb.453:                              ;   in Loop: Header=BB4_114 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB4_456
; %bb.454:                              ; %.lr.ph.i273.preheader
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v156, v161
	v_mov_b32_e32 v138, v0
.LBB4_455:                              ; %.lr.ph.i273
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ;       Parent Loop BB4_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v157, v138, v160
	v_mul_lo_u32 v158, v157, s67
	v_sub_u32_e32 v158, v138, v158
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_subrev_u32_e32 v159, s67, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_xor_b32_e32 v157, s53, v157
	v_subrev_u32_e32 v162, s53, v157
	v_mad_u64_u32 v[158:159], s[72:73], s50, v162, v[138:139]
	v_lshlrev_b32_e32 v157, 9, v157
	v_mul_lo_u32 v159, s55, v162
	v_sub_u32_e32 v157, v157, v159
	v_add_u32_e32 v157, v156, v157
	ds_read_u16 v157, v157
	v_mad_i64_i32 v[162:163], s[72:73], s74, v162, 0
	v_add_u32_e32 v138, 0x200, v138
	v_mov_b32_e32 v159, v139
	v_lshl_add_u64 v[162:163], v[162:163], 1, v[154:155]
	v_cmp_le_i32_e32 vcc, s52, v138
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[162:163]
	v_add_u32_e32 v156, 0x400, v156
	s_or_b64 s[30:31], vcc, s[30:31]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v157
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB4_455
.LBB4_456:                              ; %Flow2145
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB4_457:                              ; %Flow2151
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB4_469
; %bb.458:                              ;   in Loop: Header=BB4_114 Depth=3
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB4_468
; %bb.459:                              ; %.lr.ph4.i.preheader
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v162, v225
	v_mov_b64_e32 v[156:157], v[152:153]
	v_mov_b32_e32 v163, v0
	s_branch .LBB4_462
.LBB4_460:                              ; %Flow2147
                                        ;   in Loop: Header=BB4_462 Depth=4
	s_or_b64 exec, exec, s[96:97]
.LBB4_461:                              ; %.loopexit.i275
                                        ;   in Loop: Header=BB4_462 Depth=4
	s_or_b64 exec, exec, s[94:95]
	v_add_u32_e32 v163, 0x200, v163
	v_cmp_le_i32_e32 vcc, s82, v163
	v_lshl_add_u64 v[156:157], v[156:157], 0, s[90:91]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_add_u32_e32 v162, 0x2000, v162
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB4_468
.LBB4_462:                              ; %.lr.ph4.i
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ;       Parent Loop BB4_114 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB4_467 Depth 5
	v_lshlrev_b32_e32 v138, 3, v163
	v_and_b32_e32 v138, 0xf8, v138
	v_add_u32_e32 v158, 8, v138
	v_cmp_ge_i32_e32 vcc, s66, v158
	s_and_saveexec_b64 s[72:73], vcc
	s_xor_b64 s[94:95], exec, s[72:73]
	s_cbranch_execz .LBB4_464
; %bb.463:                              ;   in Loop: Header=BB4_462 Depth=4
	v_lshrrev_b32_e32 v164, 5, v163
	v_mad_i64_i32 v[158:159], s[72:73], s74, v164, 0
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[154:155]
	v_lshlrev_b32_e32 v138, 1, v138
	v_lshlrev_b32_e32 v164, 9, v164
	v_lshl_add_u64 v[158:159], v[158:159], 0, v[138:139]
	v_add3_u32 v138, 0, v164, v138
	ds_read_b128 v[228:231], v138
                                        ; implicit-def: $vgpr138
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[158:159], v[228:231]
.LBB4_464:                              ; %Flow2148
                                        ;   in Loop: Header=BB4_462 Depth=4
	s_andn2_saveexec_b64 s[94:95], s[94:95]
	s_cbranch_execz .LBB4_461
; %bb.465:                              ; %.preheader.i
                                        ;   in Loop: Header=BB4_462 Depth=4
	v_cmp_gt_i32_e32 vcc, s63, v138
	s_and_saveexec_b64 s[96:97], vcc
	s_cbranch_execz .LBB4_460
; %bb.466:                              ; %.lr.ph.i276
                                        ;   in Loop: Header=BB4_462 Depth=4
	s_mov_b32 s68, 0
	s_mov_b64 s[98:99], 0
	v_mov_b32_e32 v138, v162
	v_mov_b64_e32 v[158:159], v[156:157]
.LBB4_467:                              ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ;       Parent Loop BB4_114 Depth=3
                                        ;         Parent Loop BB4_462 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	ds_read_u16 v164, v138
	s_add_i32 s93, s68, 1
	s_cmp_gt_u32 s68, 6
	s_cselect_b64 s[72:73], -1, 0
	v_add_u32_e32 v138, 2, v138
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v164
	v_add_u32_e32 v164, s68, v222
	v_cmp_le_u32_e32 vcc, s66, v164
	s_or_b64 s[72:73], s[72:73], vcc
	s_and_b64 s[72:73], exec, s[72:73]
	v_lshl_add_u64 v[158:159], v[158:159], 0, 2
	s_or_b64 s[98:99], s[72:73], s[98:99]
	s_mov_b32 s68, s93
	s_andn2_b64 exec, exec, s[98:99]
	s_cbranch_execnz .LBB4_467
	s_branch .LBB4_460
.LBB4_468:                              ; %Flow2150
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_or_b64 exec, exec, s[0:1]
.LBB4_469:                              ; %Flow2152
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_branch .LBB4_113
.LBB4_470:                              ;   in Loop: Header=BB4_114 Depth=3
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB4_112
; %bb.471:                              ; %.lr.ph.i280.preheader
                                        ;   in Loop: Header=BB4_114 Depth=3
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v156, v161
	v_mov_b32_e32 v138, v0
.LBB4_472:                              ; %.lr.ph.i280
                                        ;   Parent Loop BB4_84 Depth=1
                                        ;     Parent Loop BB4_110 Depth=2
                                        ;       Parent Loop BB4_114 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v157, v138, v160
	v_mul_lo_u32 v158, v157, s67
	v_sub_u32_e32 v158, v138, v158
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_subrev_u32_e32 v159, s67, v158
	v_cndmask_b32_e32 v158, v158, v159, vcc
	v_add_u32_e32 v159, 1, v157
	v_cmp_le_u32_e32 vcc, s67, v158
	s_nop 1
	v_cndmask_b32_e32 v157, v157, v159, vcc
	v_xor_b32_e32 v157, s53, v157
	v_subrev_u32_e32 v162, s53, v157
	v_mad_u64_u32 v[158:159], s[72:73], s50, v162, v[138:139]
	v_lshlrev_b32_e32 v157, 9, v157
	v_mul_lo_u32 v159, s55, v162
	v_sub_u32_e32 v157, v157, v159
	v_add_u32_e32 v157, v156, v157
	ds_read_u16 v157, v157
	v_mad_i64_i32 v[162:163], s[72:73], s74, v162, 0
	v_add_u32_e32 v138, 0x200, v138
	v_mov_b32_e32 v159, v139
	v_lshl_add_u64 v[162:163], v[162:163], 1, v[154:155]
	v_cmp_le_i32_e32 vcc, s52, v138
	v_lshl_add_u64 v[158:159], v[158:159], 1, v[162:163]
	v_add_u32_e32 v156, 0x400, v156
	s_or_b64 s[30:31], vcc, s[30:31]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[158:159], v157
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB4_472
	s_branch .LBB4_112
.LBB4_473:                              ; %._crit_edge839
                                        ;   in Loop: Header=BB4_84 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v232, 4
	v_readlane_b32 s29, v232, 5
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_475
; %bb.474:                              ;   in Loop: Header=BB4_84 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB4_475:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB4_84 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s28, v232, 22
	v_readlane_b32 s29, v232, 23
	v_readlane_b32 s66, v232, 41
	s_and_b64 s[28:29], s[0:1], s[28:29]
	v_readlane_b32 s67, v232, 42
	v_readlane_b32 s63, v232, 43
	v_readlane_b32 s61, v232, 44
	s_mov_b64 exec, s[28:29]
	s_cbranch_execz .LBB4_82
; %bb.476:                              ; %.lr.ph841.preheader
                                        ;   in Loop: Header=BB4_84 Depth=1
	v_readlane_b32 s3, v232, 46
	s_lshl_b32 s3, s3, 2
	s_add_i32 s3, s2, s3
	v_readlane_b32 s28, v232, 48
	s_sub_i32 s3, s3, s28
	v_readlane_b32 s28, v232, 50
	s_sub_i32 s3, s3, s28
	v_readlane_b32 s28, v232, 45
	s_lshl_b32 s28, s28, 2
	s_sub_i32 s3, s3, s28
	s_lshl_b32 s3, s3, 8
	s_mov_b32 s30, s22
	s_branch .LBB4_478
.LBB4_477:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB4_478 Depth=2
	s_add_i32 s30, s30, -1
	s_add_i32 s3, s3, s42
	s_cmp_lg_u32 s30, 0
	s_cbranch_scc0 .LBB4_82
.LBB4_478:                              ; %.lr.ph841
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
	s_cbranch_scc1 .LBB4_480
; %bb.479:                              ; %Flow2141
                                        ;   in Loop: Header=BB4_478 Depth=2
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB4_477
	s_branch .LBB4_481
.LBB4_480:                              ;   in Loop: Header=BB4_478 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB4_477
.LBB4_481:                              ;   in Loop: Header=BB4_478 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB4_477
.LBB4_482:                              ; %.critedge267
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
		.amdhsa_next_free_vgpr 233
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 236
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
	.set _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals.num_vgpr, 233
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
; codeLenInByte = 22936
; TotalNumSgprs: 106
; NumVgprs: 233
; NumAgprs: 0
; TotalNumVgprs: 233
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 233
; AccumOffset: 236
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 58
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,"axG",@progbits,_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,comdat
	.protected	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals ; -- Begin function _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
	.globl	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
	.p2align	8
	.type	_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals,@function
_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals: ; @_Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals
; %bb.0:
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[16:17], s[0:1], 0x120
                                        ; implicit-def: $vgpr88 : SGPR spill to VGPR lane
	s_mov_b32 s86, s2
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v88, s4, 0
	s_nop 1
	v_writelane_b32 v88, s5, 1
	s_load_dwordx8 s[56:63], s[0:1], 0x100
	s_load_dwordx8 s[64:71], s[0:1], 0xc0
	s_load_dwordx4 s[4:7], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s33, s57, 31
	s_lshr_b32 s3, s33, 29
	v_writelane_b32 v88, s4, 2
	s_add_i32 s3, s57, s3
	s_ashr_i32 s72, s3, 3
	v_writelane_b32 v88, s5, 3
	v_writelane_b32 v88, s6, 4
	v_writelane_b32 v88, s7, 5
	s_load_dwordx2 s[4:5], s[0:1], 0xf8
	s_mov_b64 s[6:7], -1
	s_cmp_ge_i32 s2, s63
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v88, s4, 6
	s_nop 1
	v_writelane_b32 v88, s5, 7
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	v_writelane_b32 v88, s72, 8
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
.LBB5_3:                                ; %_ZN17hk_gemm_rs_mi300x20next_epoch_workgroupEPj.exit282
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
.LBB5_6:                                ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit285
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
.LBB5_8:                                ; %Flow756
	s_mov_b64 s[2:3], exec
	v_writelane_b32 v88, s2, 59
	s_and_b64 s[6:7], s[2:3], s[6:7]
	s_nop 0
	v_writelane_b32 v88, s3, 60
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB5_72
; %bb.9:                                ; %.critedge
	s_mul_i32 s87, s61, s60
	s_cmp_ge_i32 s16, s87
	s_cbranch_scc1 .LBB5_72
; %bb.10:                               ; %.lr.ph
	s_load_dwordx2 s[40:41], s[0:1], 0x60
	s_load_dwordx2 s[80:81], s[0:1], 0x90
	v_readlane_b32 s2, v88, 8
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
	v_writelane_b32 v88, s6, 18
	s_cmp_lg_u32 s56, 1
	s_mov_b32 s22, s58
	v_writelane_b32 v88, s7, 19
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 21
	s_cmp_lg_u32 s56, 2
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_writelane_b32 v88, s7, 22
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 23
	s_cmp_lg_u32 s56, 3
	v_mbcnt_hi_u32_b32 v4, -1, v4
	v_writelane_b32 v88, s7, 24
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 28
	s_cmp_lg_u32 s56, 4
	v_and_b32_e32 v3, 63, v0
	v_writelane_b32 v88, s7, 29
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 30
	s_cmp_lg_u32 s56, 5
	v_mov_b32_e32 v5, 0
	v_writelane_b32 v88, s7, 31
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 16
	s_cmp_lg_u32 s56, 6
	v_lshlrev_b32_e32 v4, 2, v4
	v_writelane_b32 v88, s7, 17
	s_cselect_b64 s[6:7], -1, 0
	v_writelane_b32 v88, s6, 9
	s_cmp_lg_u32 s56, 7
	v_mul_lo_u32 v64, s60, v0
	v_writelane_b32 v88, s7, 10
	s_cselect_b64 s[6:7], -1, 0
	s_abs_i32 s73, s61
	v_cvt_f32_u32_e32 v2, s73
	s_sub_i32 s3, 0, s73
	s_ashr_i32 s74, s61, 31
	s_lshl_b64 s[82:83], s[22:23], 7
	v_rcp_iflag_f32_e32 v2, v2
	v_writelane_b32 v88, s6, 51
	v_cmp_eq_u32_e64 s[8:9], 0, v3
	v_cmp_gt_i32_e64 s[10:11], s17, v0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_writelane_b32 v88, s7, 52
	v_cmp_gt_u32_e64 s[6:7], 8, v0
	v_lshlrev_b32_e32 v65, 3, v0
	v_readfirstlane_b32 s20, v2
	s_mul_i32 s3, s3, s20
	s_mul_hi_u32 s3, s20, s3
	s_add_i32 s75, s20, s3
	s_add_u32 s20, s80, 2
	s_addc_u32 s21, s81, 0
	v_writelane_b32 v88, s20, 53
	s_mul_i32 s3, s13, 14
	v_lshrrev_b32_e32 v2, 3, v0
	v_writelane_b32 v88, s21, 54
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
	v_writelane_b32 v88, s42, 55
	s_addc_u32 s97, s41, s13
	s_mov_b64 s[98:99], 0
	v_bfrev_b32_e32 v66, 32
	s_xor_b64 s[40:41], s[14:15], -1
	s_movk_i32 s78, 0x7fff
	s_mov_b32 s79, 0x7060302
	v_and_b32_e32 v67, 0x100, v4
	v_writelane_b32 v88, s43, 56
	s_branch .LBB5_13
.LBB5_11:                               ; %Flow741
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_sub_i32 s3, s16, s63
	s_add_i32 s16, s3, 0x130
	s_cmp_ge_i32 s16, s87
	s_cselect_b64 s[12:13], -1, 0
	s_orn2_b64 s[12:13], s[12:13], exec
.LBB5_12:                               ; %Flow753
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
; %bb.15:                               ; %.lr.ph.i.i.i287.preheader
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[46:47], 0
                                        ; implicit-def: $sgpr42_sgpr43
                                        ; implicit-def: $sgpr44_sgpr45
	s_branch .LBB5_17
.LBB5_16:                               ; %Flow749
                                        ;   in Loop: Header=BB5_17 Depth=2
	s_and_b64 s[20:21], exec, s[44:45]
	s_or_b64 s[14:15], s[20:21], s[14:15]
	s_andn2_b64 s[20:21], s[42:43], exec
	s_and_b64 s[42:43], s[48:49], exec
	s_or_b64 s[42:43], s[20:21], s[42:43]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB5_19
.LBB5_17:                               ; %.lr.ph.i.i.i287
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
.LBB5_21:                               ; %.critedge428
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
.LBB5_24:                               ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit293
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v4, v67, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, 0x4000000, v4
	v_cmp_eq_u32_e64 s[14:15], 0, v4
.LBB5_25:                               ; %Flow752
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_and_saveexec_b64 s[42:43], s[14:15]
	s_cbranch_execz .LBB5_12
; %bb.26:                               ; %.critedge436
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_barrier
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_cbranch_execz .LBB5_39
; %bb.27:                               ; %.lr.ph.i294.preheader
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
	v_readlane_b32 s14, v88, 53
	v_readlane_b32 s15, v88, 54
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[22:23]
	v_lshl_add_u64 v[14:15], s[88:89], 0, v[22:23]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[22:23]
	v_readlane_b32 s14, v88, 55
	v_readlane_b32 s15, v88, 56
	v_lshl_add_u64 v[16:17], s[90:91], 0, v[22:23]
	v_lshl_add_u64 v[18:19], s[92:93], 0, v[22:23]
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[22:23]
	v_lshl_add_u64 v[20:21], s[94:95], 0, v[22:23]
	v_lshl_add_u64 v[22:23], s[96:97], 0, v[22:23]
	s_mov_b64 s[48:49], 0
	v_mov_b32_e32 v68, v65
	v_mov_b32_e32 v69, v0
	s_branch .LBB5_29
.LBB5_28:                               ; %.critedge.i297
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
.LBB5_29:                               ; %.lr.ph.i294
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
; %bb.30:                               ; %.preheader.i296.preheader
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
.LBB5_31:                               ; %Flow743
                                        ;   in Loop: Header=BB5_33 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_andn2_b64 s[54:55], s[54:55], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[54:55], s[54:55], s[20:21]
.LBB5_32:                               ; %Flow742
                                        ;   in Loop: Header=BB5_33 Depth=3
	s_or_b64 exec, exec, s[14:15]
	s_and_b64 s[14:15], exec, s[54:55]
	s_or_b64 s[52:53], s[14:15], s[52:53]
	s_andn2_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB5_36
.LBB5_33:                               ; %.preheader.i296
                                        ;   Parent Loop BB5_13 Depth=1
                                        ;     Parent Loop BB5_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_cmp_gt_u64_e32 vcc, s[22:23], v[24:25]
	s_or_b64 s[54:55], s[54:55], exec
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB5_32
; %bb.34:                               ; %.preheader.i296.1
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
.LBB5_36:                               ; %Flow744
                                        ;   in Loop: Header=BB5_29 Depth=2
	s_or_b64 exec, exec, s[52:53]
                                        ; implicit-def: $vgpr24_vgpr25
.LBB5_37:                               ; %Flow745
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
	v_lshlrev_b32_e32 v58, 16, v52
	v_and_b32_e32 v59, 0xffff0000, v52
	v_pk_add_f32 v[26:27], v[26:27], v[40:41]
	v_pk_add_f32 v[24:25], v[24:25], v[44:45]
	v_lshlrev_b32_e32 v48, 16, v53
	v_and_b32_e32 v49, 0xffff0000, v53
	v_pk_add_f32 v[26:27], v[26:27], v[86:87]
	v_pk_add_f32 v[24:25], v[24:25], v[58:59]
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
.LBB5_39:                               ; %Flow747
                                        ;   in Loop: Header=BB5_13 Depth=1
	s_or_b64 exec, exec, s[12:13]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[44:45], s[4:5]
	s_cbranch_execz .LBB5_11
; %bb.40:                               ; %.preheader445
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
	v_readlane_b32 s12, v88, 18
	v_readlane_b32 s13, v88, 19
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
	v_readlane_b32 s12, v88, 21
	v_readlane_b32 s13, v88, 22
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
.LBB5_45:                               ; %Flow739
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
	v_readlane_b32 s12, v88, 23
	v_readlane_b32 s13, v88, 24
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
.LBB5_49:                               ; %Flow738
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
	v_readlane_b32 s12, v88, 28
	v_readlane_b32 s13, v88, 29
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
.LBB5_53:                               ; %Flow737
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
	v_readlane_b32 s12, v88, 30
	v_readlane_b32 s13, v88, 31
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
.LBB5_57:                               ; %Flow736
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
	v_readlane_b32 s12, v88, 16
	v_readlane_b32 s13, v88, 17
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
.LBB5_61:                               ; %Flow735
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
	v_readlane_b32 s12, v88, 9
	v_readlane_b32 s13, v88, 10
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
.LBB5_65:                               ; %Flow734
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
	v_readlane_b32 s12, v88, 51
	v_readlane_b32 s13, v88, 52
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
.LBB5_72:                               ; %Flow757
	v_readlane_b32 s2, v88, 59
	v_readlane_b32 s3, v88, 60
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[16:17], s[0:1], 0x120
	s_mov_b64 s[6:7], 0
	v_readlane_b32 s72, v88, 8
.LBB5_73:                               ; %Flow798
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccz .LBB5_170
; %bb.74:
	s_ashr_i32 s87, s86, 31
	s_mov_b32 s2, s86
	v_writelane_b32 v88, s2, 9
	s_lshl_b64 s[4:5], s[86:87], 2
	s_add_u32 s4, s70, s4
	v_writelane_b32 v88, s3, 10
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_addc_u32 s5, s71, s5
	s_nop 0
	v_writelane_b32 v88, s2, 11
	s_nop 1
	v_writelane_b32 v88, s3, 12
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
	s_branch .LBB5_170
.LBB5_80:
	s_mov_b64 s[4:5], -1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB5_170
.LBB5_81:                               ; %.critedge426
	s_lshr_b32 s2, s33, 27
	s_add_i32 s2, s57, s2
	s_ashr_i32 s82, s2, 5
	s_mul_i32 s4, s61, s82
	v_readlane_b32 s2, v88, 9
	s_cmp_ge_i32 s2, s4
	v_readlane_b32 s3, v88, 10
	v_writelane_b32 v88, s4, 13
	s_cbranch_scc1 .LBB5_170
; %bb.82:                               ; %.lr.ph467
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[12:13], src_shared_base
	s_load_dwordx2 s[80:81], s[0:1], 0x80
	s_load_dwordx4 s[8:11], s[0:1], 0x70
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x20
	s_load_dwordx2 s[14:15], s[0:1], 0x30
	s_load_dwordx2 s[84:85], s[0:1], 0x50
	s_cselect_b32 s4, 0, 0
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s3, s13, 0
	s_and_b32 s0, s4, 15
	s_and_b32 s5, s4, -16
	s_add_u32 s5, s5, 16
	s_mov_b32 s1, 0
	s_addc_u32 s9, s3, 0
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s57, s4, s5
	s_cselect_b32 s0, s3, s9
	s_add_u32 s3, s57, 0x2000
	s_addc_u32 s0, s0, 0
	s_and_b32 s4, s3, -16
	s_and_b32 s0, s3, 15
	s_add_u32 s4, s4, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s70, s3, s4
	s_add_i32 s0, s59, 63
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_i32 s0, s0, s1
	v_and_b32_e32 v10, 56, v2
	s_ashr_i32 s3, s0, 6
	v_lshlrev_b32_e32 v6, 4, v0
	v_lshlrev_b32_e32 v12, 1, v10
	s_movk_i32 s0, 0xf80
	v_and_or_b32 v11, v6, s0, v12
	v_add_u32_e32 v7, s57, v11
	v_or_b32_e32 v32, 8, v11
	v_lshrrev_b32_e32 v2, 4, v7
	v_add_u32_e32 v9, s57, v32
	v_and_b32_e32 v8, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v9
	s_movk_i32 s0, 0x100
	v_lshrrev_b32_e32 v20, 3, v0
	v_and_b32_e32 v13, 0x78, v2
	v_cmp_gt_u32_e64 s[4:5], s0, v0
	v_mad_u64_u32 v[2:3], s[0:1], v20, s2, v[10:11]
	s_movk_i32 s0, 0x1f80
	s_nop 0
	v_and_or_b32 v35, v6, s0, v12
	v_ashrrev_i32_e32 v3, 31, v2
	v_add_u32_e32 v6, s70, v35
	v_or_b32_e32 v36, 8, v35
	v_lshl_add_u64 v[14:15], v[2:3], 1, s[6:7]
	v_xor_b32_e32 v34, v8, v7
	v_lshrrev_b32_e32 v2, 4, v6
	v_add_u32_e32 v8, s70, v36
	s_add_i32 s71, s3, -1
	s_lshl_b32 s83, s61, 2
	s_lshl_b32 s85, s2, 5
	v_and_b32_e32 v7, 0x78, v2
	v_lshrrev_b32_e32 v2, 4, v8
	v_xor_b32_e32 v33, v13, v9
	v_and_b32_e32 v9, 0x78, v2
	v_mad_u64_u32 v[2:3], s[0:1], v20, s84, v[10:11]
	s_cmp_gt_i32 s59, 0
	s_cselect_b64 s[90:91], -1, 0
	s_ashr_i32 s1, s16, 31
	s_mov_b32 s0, s16
	s_lshl_b32 s2, s71, 6
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s66, s0
	s_addc_u32 s1, s67, s1
	v_writelane_b32 v88, s0, 14
	s_min_i32 s73, s62, 32
	s_bfe_i64 s[88:89], s[80:81], 0x200000
	v_writelane_b32 v88, s1, 15
	s_mul_i32 s0, s10, s8
	s_mul_i32 s0, s0, s56
	s_cmp_gt_i32 s62, 0
	v_writelane_b32 v88, s0, 16
	s_cselect_b64 s[0:1], -1, 0
	v_writelane_b32 v88, s0, 18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[14:15]
	v_writelane_b32 v88, s1, 19
	v_lshrrev_b32_e32 v3, 2, v0
	v_readlane_b32 s8, v88, 2
	v_readlane_b32 s9, v88, 3
	s_cmp_lg_u64 s[8:9], 0
	s_cselect_b64 s[92:93], -1, 0
	s_max_i32 s0, s58, 1
	s_add_i32 s0, s0, -1
	s_cmp_lg_u32 s17, 0
	v_xor_b32_e32 v38, v7, v6
	v_and_b32_e32 v6, 12, v3
	v_readlane_b32 s10, v88, 4
	v_readlane_b32 s11, v88, 5
	v_writelane_b32 v88, s0, 20
	s_cselect_b64 s[94:95], -1, 0
	s_lshl_b32 s0, s73, 3
	s_sub_i32 s8, s59, s2
	v_bfe_u32 v4, v0, 6, 2
	v_and_b32_e32 v2, 15, v0
	v_cmp_gt_i32_e64 s[6:7], s0, v0
	v_cmp_gt_i32_e64 s[0:1], s8, v6
	s_abs_i32 s10, s83
	v_xor_b32_e32 v37, v9, v8
	v_lshlrev_b32_e32 v3, 7, v2
	v_or_b32_e32 v8, 1, v6
	v_lshl_or_b32 v43, v4, 4, v2
	v_writelane_b32 v88, s0, 21
	v_cvt_f32_u32_e32 v2, s10
	v_lshrrev_b32_e32 v5, 8, v0
	v_writelane_b32 v88, s1, 22
	v_cmp_gt_i32_e64 s[0:1], s8, v8
	v_rcp_iflag_f32_e32 v2, v2
	v_lshl_or_b32 v39, v5, 11, v3
	v_writelane_b32 v88, s0, 23
	v_lshl_or_b32 v40, v4, 11, v3
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_writelane_b32 v88, s1, 24
	s_or_b32 s0, s2, 32
	s_sub_i32 s9, s59, s0
	s_abs_i32 s59, s62
	v_cvt_f32_u32_e32 v3, s59
	v_cvt_u32_f32_e32 v2, v2
	v_mad_i64_i32 v[18:19], s[0:1], s80, v20, 0
	v_rcp_iflag_f32_e32 v3, v3
	s_bfe_i32 s0, s61, 0x1001d
	v_readfirstlane_b32 s1, v2
	v_writelane_b32 v88, s0, 25
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	s_sub_i32 s0, 0, s10
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	v_writelane_b32 v88, s10, 26
	s_add_i32 s0, s1, s0
	v_writelane_b32 v88, s0, 27
	s_sub_i32 s0, 0, s59
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s46, s1, s0
	s_lshr_b32 s0, s46, 27
	s_mul_i32 s1, s0, s59
	s_sub_i32 s1, 32, s1
	s_max_i32 s81, s3, 1
	s_ashr_i32 s11, s62, 31
	s_add_i32 s2, s0, 1
	s_sub_i32 s10, s1, s59
	s_cmp_ge_u32 s1, s59
	s_cselect_b32 s0, s2, s0
	s_cselect_b32 s1, s10, s1
	s_add_i32 s2, s0, 1
	s_cmp_ge_u32 s1, s59
	s_cselect_b32 s0, s2, s0
	s_abs_i32 s47, s72
	v_lshl_or_b32 v44, v5, 4, v6
	v_cvt_f32_u32_e32 v5, s47
	v_lshlrev_b32_e32 v4, 7, v20
	v_mov_b32_e32 v13, 0
	v_add_u32_e32 v2, 0, v4
	v_mov_b32_e32 v3, s13
	v_lshl_add_u64 v[20:21], v[2:3], 0, v[12:13]
	v_rcp_iflag_f32_e32 v3, v5
	v_add_u32_e32 v21, v2, v12
	s_xor_b32 s0, s0, s11
	s_sub_i32 s75, s0, s11
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s0, 0, s47
	s_ashr_i32 s2, s72, 31
	v_writelane_b32 v88, s11, 28
	v_readfirstlane_b32 s1, v2
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s96, s1, s0
	s_cmp_gt_i32 s75, 0
	s_mul_i32 s0, s89, s73
	s_mul_hi_u32 s1, s80, s73
	s_cselect_b64 s[10:11], -1, 0
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s80, s73
	s_lshl_b64 s[86:87], s[0:1], 1
	s_sub_i32 s0, 0, s72
	v_writelane_b32 v88, s0, 30
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[0:1], 1, v1
	v_and_b32_e32 v30, 63, v0
	v_and_b32_e32 v2, 7, v0
	v_writelane_b32 v88, s0, 32
	v_add3_u32 v54, 0, v12, v4
	v_lshlrev_b32_e32 v12, 4, v2
	v_writelane_b32 v88, s1, 33
	v_cmp_eq_u32_e64 s[0:1], 0, v30
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_writelane_b32 v88, s0, 34
	v_or_b32_e32 v7, 16, v6
	v_or_b32_e32 v9, 2, v6
	v_writelane_b32 v88, s1, 35
	v_cmp_gt_u32_e64 s[0:1], s75, v0
	v_or_b32_e32 v26, 3, v6
	v_or_b32_e32 v27, 17, v6
	v_writelane_b32 v88, s0, 36
	v_or_b32_e32 v28, 18, v6
	v_or_b32_e32 v29, 19, v6
	v_writelane_b32 v88, s1, 37
	v_lshlrev_b32_e32 v49, 1, v6
	v_readlane_b32 s0, v88, 11
	v_readlane_b32 s1, v88, 12
	v_writelane_b32 v88, s10, 38
	s_and_b64 s[0:1], s[0:1], s[10:11]
	s_mov_b64 s[98:99], 0x80
	v_writelane_b32 v88, s11, 39
	v_writelane_b32 v88, s0, 40
	v_lshlrev_b32_e32 v2, 2, v2
	v_mul_lo_u32 v41, s62, v0
	v_writelane_b32 v88, s1, 41
	v_writelane_b32 v88, s76, 42
	v_add_u32_e32 v42, -1, v1
	v_lshl_add_u32 v45, v43, 1, 0
	v_writelane_b32 v88, s77, 43
	v_writelane_b32 v88, s82, 44
	v_writelane_b32 v88, s84, 45
	v_or_b32_e32 v46, 1, v44
	v_or_b32_e32 v47, 2, v44
	v_writelane_b32 v88, s85, 46
	v_writelane_b32 v88, s83, 47
	v_writelane_b32 v88, s85, 48
	v_or_b32_e32 v48, 3, v44
	v_lshlrev_b32_e32 v50, 1, v7
	v_or_b32_e32 v51, 64, v49
	v_or_b32_e32 v52, 0x60, v49
	v_add_u32_e32 v53, 8, v10
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[98:99]
	v_lshl_add_u32 v55, v0, 1, 0
	v_or_b32_e32 v56, 1, v10
	v_lshl_add_u64 v[24:25], v[18:19], 1, v[12:13]
	v_bfrev_b32_e32 v57, 64
	v_and_b32_e32 v58, 0x100, v2
	v_cmp_gt_i32_e64 s[12:13], s8, v9
	v_cmp_gt_i32_e64 s[14:15], s8, v26
	v_cmp_gt_i32_e64 s[16:17], s8, v7
	v_cmp_gt_i32_e64 s[18:19], s8, v27
	v_cmp_gt_i32_e64 s[20:21], s8, v28
	v_cmp_gt_i32_e64 s[22:23], s8, v29
	v_cmp_gt_i32_e64 s[24:25], s9, v6
	v_cmp_gt_i32_e64 s[26:27], s9, v8
	v_cmp_gt_i32_e64 s[28:29], s9, v9
	v_cmp_gt_i32_e64 s[30:31], s9, v26
	v_cmp_gt_i32_e64 s[34:35], s9, v7
	v_cmp_gt_i32_e64 s[36:37], s9, v27
	v_cmp_gt_i32_e64 s[38:39], s9, v28
	v_cmp_gt_i32_e64 s[40:41], s9, v29
	s_movk_i32 s74, 0x7fff
	s_mov_b64 s[44:45], 0
	v_writelane_b32 v88, s90, 49
	s_nop 1
	v_writelane_b32 v88, s91, 50
	s_branch .LBB5_85
.LBB5_83:                               ; %Flow760
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_readlane_b32 s0, v88, 9
	s_add_i32 s8, s0, s63
	v_readlane_b32 s1, v88, 10
	s_mov_b32 s0, s8
	v_writelane_b32 v88, s0, 9
	s_nop 1
	v_writelane_b32 v88, s1, 10
	s_nop 0
	v_readlane_b32 s0, v88, 13
	s_cmp_ge_i32 s8, s0
	s_cselect_b64 s[0:1], -1, 0
	v_readlane_b32 s44, v88, 51
	s_orn2_b64 s[0:1], s[0:1], exec
	v_readlane_b32 s45, v88, 52
.LBB5_84:                               ; %Flow792
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s8, v88, 55
	v_readlane_b32 s9, v88, 56
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[44:45], s[0:1], s[44:45]
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execz .LBB5_170
.LBB5_85:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_96 Depth 2
                                        ;     Child Loop BB5_108 Depth 2
                                        ;     Child Loop BB5_120 Depth 2
                                        ;       Child Loop BB5_124 Depth 3
                                        ;         Child Loop BB5_160 Depth 4
                                        ;         Child Loop BB5_145 Depth 4
                                        ;         Child Loop BB5_154 Depth 4
                                        ;     Child Loop BB5_166 Depth 2
	v_writelane_b32 v88, s44, 51
	s_nop 1
	v_writelane_b32 v88, s45, 52
	s_nop 0
	v_readlane_b32 s0, v88, 9
	v_readlane_b32 s1, v88, 10
	s_mov_b32 s42, s0
	s_ashr_i32 s0, s0, 31
	v_readlane_b32 s1, v88, 25
	s_xor_b32 s97, s0, s1
	s_abs_i32 s0, s42
	v_readlane_b32 s1, v88, 27
	s_mul_hi_u32 s1, s0, s1
	v_readlane_b32 s10, v88, 26
	s_mul_i32 s8, s1, s10
	s_sub_i32 s0, s0, s8
	s_add_i32 s8, s1, 1
	s_sub_i32 s9, s0, s10
	s_cmp_ge_u32 s0, s10
	s_cselect_b32 s1, s8, s1
	s_cselect_b32 s0, s9, s0
	s_add_i32 s8, s1, 1
	s_cmp_ge_u32 s0, s10
	s_cselect_b32 s0, s8, s1
	s_xor_b32 s52, s0, s97
	s_sub_i32 s0, s52, s97
	s_lshl_b32 s1, s0, 2
	s_sub_i32 s8, s82, s1
	s_min_i32 s10, s8, 4
	s_abs_i32 s9, s10
	v_cvt_f32_u32_e32 v2, s9
	s_mul_i32 s53, s0, s83
	s_sub_i32 s0, s42, s53
	s_sub_i32 s33, 0, s9
	v_rcp_iflag_f32_e32 v2, v2
	s_abs_i32 s11, s0
	s_xor_b32 s8, s0, s10
	s_ashr_i32 s8, s8, 31
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_nop 0
	v_readfirstlane_b32 s42, v2
	s_mul_i32 s33, s33, s42
	s_mul_hi_u32 s33, s42, s33
	s_add_i32 s42, s42, s33
	s_mul_hi_u32 s33, s11, s42
	s_mul_i32 s42, s33, s9
	s_sub_i32 s11, s11, s42
	s_add_i32 s42, s33, 1
	s_sub_i32 s43, s11, s9
	s_cmp_ge_u32 s11, s9
	s_cselect_b32 s33, s42, s33
	s_cselect_b32 s11, s43, s11
	s_add_i32 s42, s33, 1
	s_cmp_ge_u32 s11, s9
	s_cselect_b32 s9, s42, s33
	s_xor_b32 s9, s9, s8
	s_sub_i32 s78, s9, s8
	s_mul_i32 s10, s78, s10
	v_writelane_b32 v88, s10, 53
	s_sub_i32 s10, s0, s10
	s_add_i32 s10, s10, s1
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB5_87
; %bb.86:                               ;   in Loop: Header=BB5_85 Depth=1
	s_mul_i32 s42, s85, s10
	s_ashr_i32 s43, s42, 31
	v_lshl_add_u64 v[2:3], s[42:43], 1, v[14:15]
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
.LBB5_87:                               ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB5_89
; %bb.88:                               ;   in Loop: Header=BB5_85 Depth=1
	;;#ASMSTART
	ds_write_b64 v34, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v33, v[4:5]

	;;#ASMEND
.LBB5_89:                               ; %_ZN7kittens5groupILi8EE4loadITkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi64ENS3_9st_layout3rowEEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEETkNS3_5coord4tileENS_5coordIS9_EEEEvRT_RKT0_RKT1_.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s79, s78, 6
	s_mul_i32 s44, s79, s84
	s_ashr_i32 s45, s44, 31
	v_lshl_add_u64 v[2:3], s[44:45], 1, v[16:17]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_andn2_b64 vcc, exec, s[90:91]
	;;#ASMSTART
	ds_write_b64 v38, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v37, v[4:5]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB5_102
; %bb.90:                               ; %.lr.ph454.preheader
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_lshl_b32 s0, s52, 2
	v_readlane_b32 s42, v88, 9
	s_add_i32 s0, s42, s0
	s_sub_i32 s0, s0, s53
	v_readlane_b32 s1, v88, 53
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s1, s97, 2
	s_sub_i32 s0, s0, s1
	s_mul_i32 s0, s85, s0
	v_mov_b32_e32 v2, 0
	s_add_i32 s0, s0, 64
	v_lshl_add_u64 v[26:27], s[44:45], 1, v[22:23]
	s_mov_b32 s33, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_readlane_b32 s43, v88, 10
	s_add_i32 s11, s33, 1
	s_cmp_ge_i32 s11, s3
	s_cbranch_scc1 .LBB5_96
.LBB5_91:                               ;   in Loop: Header=BB5_85 Depth=1
                                        ; implicit-def: $vgpr8_vgpr9
	s_and_saveexec_b64 s[44:45], s[4:5]
	s_cbranch_execz .LBB5_93
; %bb.92:                               ;   in Loop: Header=BB5_85 Depth=1
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u64 v[6:7], s[0:1], 1, v[14:15]
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
.LBB5_93:                               ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_b32 s1, s11, 1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], s[4:5]
	s_cbranch_execz .LBB5_95
; %bb.94:                               ;   in Loop: Header=BB5_85 Depth=1
	s_lshl_b32 s42, s1, 12
	s_add_i32 s42, s57, s42
	v_add_u32_e32 v12, s42, v11
	v_lshrrev_b32_e32 v28, 4, v12
	v_add_u32_e32 v29, s42, v32
	v_and_b32_e32 v28, 0x78, v28
	v_lshrrev_b32_e32 v30, 4, v29
	v_and_b32_e32 v30, 0x78, v30
	v_xor_b32_e32 v12, v28, v12
	;;#ASMSTART
	ds_write_b64 v12, v[6:7]

	;;#ASMEND
	v_xor_b32_e32 v29, v30, v29
	;;#ASMSTART
	ds_write_b64 v29, v[8:9]

	;;#ASMEND
.LBB5_95:                               ; %_ZN7kittens5groupILi8EE4loadITkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi64ENS3_9st_layout3rowEEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEETkNS3_5coord4tileENS_5coordIS9_EEEEvRT_RKT0_RKT1_.exit327
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s70, s1
	v_add_u32_e32 v12, s1, v35
	v_lshrrev_b32_e32 v6, 4, v12
	v_add_u32_e32 v29, s1, v36
	v_and_b32_e32 v28, 0x78, v6
	v_lshrrev_b32_e32 v6, 4, v29
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_and_b32_e32 v30, 0x78, v6
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[26:27], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v12, v28, v12
	;;#ASMSTART
	ds_write_b64 v12, v[6:7]

	;;#ASMEND
	v_xor_b32_e32 v29, v30, v29
	;;#ASMSTART
	ds_write_b64 v29, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB5_96:                               ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_and_b32 s1, s33, 1
	s_lshl_b32 s42, s1, 12
	s_add_i32 s42, s57, s42
	v_add_u32_e32 v59, s42, v39
	s_lshl_b32 s1, s1, 13
	s_add_i32 s1, s70, s1
	v_add_u32_e32 v6, v49, v59
	v_add_u32_e32 v12, s1, v40
	v_lshrrev_b32_e32 v7, 4, v6
	v_add_u32_e32 v8, v50, v59
	v_and_b32_e32 v7, 0x78, v7
	v_lshrrev_b32_e32 v9, 4, v8
	v_add_u32_e32 v28, v49, v12
	v_xor_b32_e32 v6, v7, v6
	v_and_b32_e32 v9, 0x78, v9
	v_lshrrev_b32_e32 v29, 4, v28
	v_add_u32_e32 v30, v50, v12
	;;#ASMSTART
	ds_read_b64 v[6:7], v6 offset:0

	;;#ASMEND
	v_xor_b32_e32 v8, v9, v8
	v_and_b32_e32 v29, 0x78, v29
	v_lshrrev_b32_e32 v31, 4, v30
	;;#ASMSTART
	ds_read_b64 v[8:9], v8 offset:0

	;;#ASMEND
	v_xor_b32_e32 v28, v29, v28
	v_and_b32_e32 v31, 0x78, v31
	s_cmp_eq_u32 s71, s33
	;;#ASMSTART
	ds_read_b64 v[28:29], v28 offset:0

	;;#ASMEND
	v_xor_b32_e32 v30, v31, v30
	s_cselect_b64 s[48:49], -1, 0
	s_cmp_lg_u32 s71, s33
	;;#ASMSTART
	ds_read_b64 v[30:31], v30 offset:0

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
	s_cbranch_scc1 .LBB5_98
; %bb.97:                               ; %.preheader.1.i
                                        ;   in Loop: Header=BB5_96 Depth=2
	v_readlane_b32 s42, v88, 21
	v_readlane_b32 s43, v88, 22
	v_and_b32_e32 v61, 0xffff0000, v7
	v_readlane_b32 s44, v88, 23
	v_cndmask_b32_e64 v60, 0, v6, s[42:43]
	v_cndmask_b32_e64 v61, v61, v7, s[12:13]
	s_or_b64 s[42:43], s[14:15], s[12:13]
	v_readlane_b32 s45, v88, 24
	v_and_b32_e32 v61, 0xffff, v61
	s_or_b64 vcc, s[42:43], s[44:45]
	s_or_b64 s[42:43], s[22:23], s[20:21]
	v_cndmask_b32_sdwa v6, v60, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v7, v61, v7, s[14:15]
	v_cndmask_b32_e64 v60, 0, v8, s[16:17]
	v_and_b32_e32 v61, 0xffff0000, v9
	s_or_b64 vcc, s[42:43], s[18:19]
	v_cndmask_b32_e64 v61, v61, v9, s[20:21]
	v_cndmask_b32_sdwa v8, v60, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[22:23]
	v_cndmask_b32_sdwa v9, v61, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB5_98:                               ;   in Loop: Header=BB5_96 Depth=2
	v_add_u32_e32 v60, v51, v59
	v_lshrrev_b32_e32 v61, 4, v60
	v_mfma_f32_16x16x16_bf16 v[2:5], v[6:7], v[28:29], v[2:5]
	v_and_b32_e32 v6, 0x78, v61
	v_xor_b32_e32 v6, v6, v60
	;;#ASMSTART
	ds_read_b64 v[28:29], v6 offset:0

	;;#ASMEND
	v_add_u32_e32 v6, v52, v59
	v_mfma_f32_16x16x16_bf16 v[2:5], v[8:9], v[30:31], v[2:5]
	v_lshrrev_b32_e32 v7, 4, v6
	v_add_u32_e32 v8, v51, v12
	v_and_b32_e32 v7, 0x78, v7
	v_lshrrev_b32_e32 v9, 4, v8
	v_xor_b32_e32 v6, v7, v6
	v_and_b32_e32 v9, 0x78, v9
	;;#ASMSTART
	ds_read_b64 v[6:7], v6 offset:0

	;;#ASMEND
	v_xor_b32_e32 v8, v9, v8
	;;#ASMSTART
	ds_read_b64 v[30:31], v8 offset:0

	;;#ASMEND
	v_add_u32_e32 v8, v52, v12
	v_lshrrev_b32_e32 v9, 4, v8
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v8, v9, v8
	;;#ASMSTART
	ds_read_b64 v[8:9], v8 offset:0

	;;#ASMEND
	s_andn2_b64 vcc, exec, s[48:49]
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
	s_cbranch_vccnz .LBB5_100
; %bb.99:                               ; %.preheader.1.i.1
                                        ;   in Loop: Header=BB5_96 Depth=2
	v_and_b32_e32 v59, 0xffff0000, v29
	v_cndmask_b32_e64 v59, v59, v29, s[28:29]
	s_or_b64 s[42:43], s[30:31], s[28:29]
	v_cndmask_b32_e64 v12, 0, v28, s[24:25]
	v_and_b32_e32 v59, 0xffff, v59
	s_or_b64 vcc, s[42:43], s[26:27]
	s_or_b64 s[42:43], s[40:41], s[38:39]
	v_cndmask_b32_sdwa v28, v12, v28, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v29, v59, v29, s[30:31]
	v_cndmask_b32_e64 v12, 0, v6, s[34:35]
	v_and_b32_e32 v59, 0xffff0000, v7
	s_or_b64 vcc, s[42:43], s[36:37]
	v_cndmask_b32_e64 v59, v59, v7, s[38:39]
	v_cndmask_b32_sdwa v6, v12, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_mov_b64 vcc, s[40:41]
	v_cndmask_b32_sdwa v7, v59, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
.LBB5_100:                              ;   in Loop: Header=BB5_96 Depth=2
	v_mfma_f32_16x16x16_bf16 v[2:5], v[28:29], v[30:31], v[2:5]
	s_add_i32 s0, s0, 64
	s_cmp_eq_u32 s81, s11
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[98:99]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[6:7], v[8:9], v[2:5]
	s_barrier
	s_cbranch_scc1 .LBB5_103
; %bb.101:                              ;   in Loop: Header=BB5_96 Depth=2
	s_mov_b32 s33, s11
	s_add_i32 s11, s33, 1
	s_cmp_ge_i32 s11, s3
	s_cbranch_scc0 .LBB5_91
	s_branch .LBB5_96
.LBB5_102:                              ;   in Loop: Header=BB5_85 Depth=1
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v3, v5
	v_mov_b32_e32 v2, v5
.LBB5_103:                              ; %Flow788
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s42, v88, 36
	v_readlane_b32 s43, v88, 37
	s_and_b64 s[42:43], s[0:1], s[42:43]
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_112
; %bb.104:                              ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s42, v88, 32
	v_readlane_b32 s43, v88, 33
	s_and_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB5_112
; %bb.105:                              ;   in Loop: Header=BB5_85 Depth=1
	v_lshl_add_u32 v6, s10, 5, v41
	v_sub_u32_e32 v8, 0, v6
	v_max_i32_e32 v8, v6, v8
	v_mul_hi_u32 v9, v8, s96
	v_mul_lo_u32 v12, v9, s47
	v_sub_u32_e32 v8, v8, v12
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s47, v8
	v_ashrrev_i32_e32 v7, 31, v6
	v_xor_b32_e32 v7, s2, v7
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_subrev_u32_e32 v12, s47, v8
	v_cndmask_b32_e32 v8, v8, v12, vcc
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s47, v8
	v_readlane_b32 s11, v88, 28
	s_nop 0
	v_cndmask_b32_e32 v8, v9, v12, vcc
	v_xor_b32_e32 v8, v8, v7
	v_sub_u32_e32 v7, v8, v7
	v_mul_lo_u32 v8, v7, s72
	v_sub_u32_e32 v6, v6, v8
	v_sub_u32_e32 v9, 0, v6
	v_ashrrev_i32_e32 v8, 31, v6
	v_max_i32_e32 v6, v6, v9
	v_mul_hi_u32 v9, v6, s46
	v_mul_lo_u32 v12, v9, s59
	v_sub_u32_e32 v6, v6, v12
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s59, v6
	v_xor_b32_e32 v8, s11, v8
	s_nop 0
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_subrev_u32_e32 v12, s59, v6
	v_cndmask_b32_e32 v6, v6, v12, vcc
	v_add_u32_e32 v12, 1, v9
	v_cmp_le_u32_e32 vcc, s59, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v9, v12, vcc
	v_xor_b32_e32 v6, v6, v8
	v_sub_u32_e32 v6, v6, v8
	v_mad_u64_u32 v[6:7], s[42:43], v7, s60, v[6:7]
	v_mul_lo_u32 v6, v6, s61
	v_add_u32_e32 v6, s78, v6
	v_readlane_b32 s42, v88, 14
	v_ashrrev_i32_e32 v7, 31, v6
	v_readlane_b32 s43, v88, 15
	s_nop 1
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[42:43]
	flat_load_dword v8, v[6:7] offset:256 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_u32_e32 vcc, v8, v42
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB5_112
; %bb.106:                              ; %.lr.ph.i.i.i.preheader
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b32 s33, s53
	s_mov_b32 s11, s52
	s_mov_b64 s[44:45], 0
	s_mov_b64 s[52:53], 0
                                        ; implicit-def: $sgpr48_sgpr49
                                        ; implicit-def: $sgpr50_sgpr51
	s_branch .LBB5_108
.LBB5_107:                              ; %Flow782
                                        ;   in Loop: Header=BB5_108 Depth=2
	s_and_b64 s[42:43], exec, s[50:51]
	s_or_b64 s[44:45], s[42:43], s[44:45]
	s_andn2_b64 s[42:43], s[48:49], exec
	s_and_b64 s[48:49], s[54:55], exec
	s_or_b64 s[48:49], s[42:43], s[48:49]
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execz .LBB5_110
.LBB5_108:                              ; %.lr.ph.i.i.i
                                        ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_readlane_b32 s42, v88, 6
	s_add_u32 s52, s52, 1
	v_readlane_b32 s43, v88, 7
	s_addc_u32 s53, s53, 0
	s_mov_b64 s[54:55], -1
	v_mov_b64_e32 v[8:9], s[42:43]
	v_cmp_gt_u64_e32 vcc, s[52:53], v[8:9]
	s_or_b64 s[50:51], s[50:51], exec
	s_cbranch_vccnz .LBB5_107
; %bb.109:                              ;   in Loop: Header=BB5_108 Depth=2
	s_sleep 4
	flat_load_dword v8, v[6:7] offset:256 sc0 sc1
	s_andn2_b64 s[42:43], s[50:51], exec
	s_mov_b64 s[54:55], 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ge_u32_e32 vcc, v8, v42
	s_and_b64 s[50:51], vcc, exec
	s_or_b64 s[50:51], s[42:43], s[50:51]
	s_branch .LBB5_107
.LBB5_110:                              ; %loop.exit.guard732
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[42:43], s[48:49]
	s_mov_b32 s52, s11
	s_mov_b32 s53, s33
	s_xor_b64 s[42:43], exec, s[42:43]
	s_cbranch_execz .LBB5_112
; %bb.111:                              ; %_ZN17hk_gemm_rs_mi300x17wait_reuse_creditEPKjjmPii.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s48, v88, 2
	v_readlane_b32 s50, v88, 4
	v_readlane_b32 s51, v88, 5
	v_readlane_b32 s49, v88, 3
	s_nop 0
	v_mov_b64_e32 v[6:7], s[50:51]
	flat_atomic_or v[6:7], v57
.LBB5_112:                              ; %.critedge427
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[76:77]
	s_mov_b64 s[44:45], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB5_116
; %bb.113:                              ;   in Loop: Header=BB5_85 Depth=1
	v_mov_b32_e32 v6, 0
	s_mov_b64 s[44:45], exec
	v_readlane_b32 s42, v88, 34
	v_readlane_b32 s43, v88, 35
	s_and_b64 s[42:43], s[44:45], s[42:43]
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_115
; %bb.114:                              ;   in Loop: Header=BB5_85 Depth=1
	v_readlane_b32 s48, v88, 2
	v_readlane_b32 s50, v88, 4
	v_readlane_b32 s51, v88, 5
	v_readlane_b32 s49, v88, 3
	s_nop 0
	v_mov_b64_e32 v[6:7], s[50:51]
	flat_load_dword v6, v[6:7] sc1
.LBB5_115:                              ; %_ZN17hk_gemm_rs_mi300x13error_bit_setEPKii.exit270
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v6, v58, v6
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v6, 0x2000000, v6
	v_cmp_eq_u32_e64 s[44:45], 0, v6
.LBB5_116:                              ; %Flow791
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_mov_b64 s[42:43], exec
	v_writelane_b32 v88, s42, 55
	s_nop 1
	v_writelane_b32 v88, s43, 56
	s_and_b64 s[42:43], s[42:43], s[44:45]
	v_readlane_b32 s44, v88, 51
	v_readlane_b32 s45, v88, 52
	s_mov_b64 exec, s[42:43]
	s_cbranch_execz .LBB5_84
; %bb.117:                              ; %.critedge435
                                        ;   in Loop: Header=BB5_85 Depth=1
	v_writelane_b32 v88, s53, 57
	v_writelane_b32 v88, s52, 58
	v_writelane_b32 v88, s97, 59
	s_nop 0
	v_readlane_b32 s0, v88, 38
	v_readlane_b32 s1, v88, 39
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB5_161
; %bb.118:                              ; %.lr.ph462
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_sub_i32 s11, s58, s79
	s_min_i32 s72, s11, 64
	s_abs_i32 s33, s72
	v_cvt_f32_u32_e32 v8, s33
	s_lshl_b32 s43, s10, 5
	s_ashr_i32 s97, s72, 31
	s_sub_i32 s10, 0, s33
	v_rcp_iflag_f32_e32 v8, v8
	s_lshl_b32 s9, s9, 6
	s_lshl_b32 s8, s8, 6
	s_sub_i32 s91, s9, s8
	v_mul_f32_e32 v8, 0x4f7ffffe, v8
	v_cvt_u32_f32_e32 v8, v8
	v_readlane_b32 s8, v88, 58
	v_cmp_gt_i32_e64 s[52:53], s11, v10
	s_lshl_b32 s8, s8, 2
	v_mul_lo_u32 v9, s10, v8
	s_lshl_b32 s10, s97, 7
	v_subrev_u32_e32 v60, s10, v55
	v_readlane_b32 s10, v88, 9
	s_add_i32 s8, s10, s8
	v_readlane_b32 s9, v88, 57
	s_sub_i32 s8, s8, s9
	v_readlane_b32 s9, v88, 53
	v_or_b32_e32 v6, s79, v43
	v_readlane_b32 s0, v88, 20
	s_sub_i32 s8, s8, s9
	v_readlane_b32 s9, v88, 59
	v_mov_b32_e32 v7, s0
	v_cmp_gt_i32_e32 vcc, s58, v6
	s_lshl_b32 s9, s9, 2
	v_readlane_b32 s48, v88, 2
	v_cndmask_b32_e32 v6, v7, v6, vcc
	s_sub_i32 s8, s8, s9
	v_ashrrev_i32_e32 v7, 31, v6
	v_readlane_b32 s49, v88, 3
	v_readlane_b32 s50, v88, 4
	v_readlane_b32 s51, v88, 5
	s_mul_i32 s42, s72, s73
	v_mul_hi_u32 v9, v8, v9
	s_lshl_b32 s8, s8, 5
	v_readlane_b32 s9, v88, 16
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[48:49]
	v_cmp_gt_i32_e64 s[0:1], s42, v0
	v_cmp_lt_i32_e64 s[48:49], s72, v53
	v_cmp_ge_i32_e64 s[50:51], s72, v53
	s_mov_b32 s76, 0
	v_add_u32_e32 v59, v8, v9
	s_lshl_b32 s77, s72, 1
	s_sub_i32 s90, 0, s72
	s_add_i32 s8, s9, s8
	v_readlane_b32 s11, v88, 10
	s_branch .LBB5_120
.LBB5_119:                              ; %._crit_edge460
                                        ;   in Loop: Header=BB5_120 Depth=2
	s_add_i32 s76, s76, 1
	s_add_i32 s8, s8, s62
	s_cmp_eq_u32 s76, s75
	s_cbranch_scc1 .LBB5_161
.LBB5_120:                              ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_124 Depth 3
                                        ;         Child Loop BB5_160 Depth 4
                                        ;         Child Loop BB5_145 Depth 4
                                        ;         Child Loop BB5_154 Depth 4
	v_readlane_b32 s10, v88, 18
	v_readlane_b32 s11, v88, 19
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB5_119
; %bb.121:                              ; %.lr.ph459.preheader
                                        ;   in Loop: Header=BB5_120 Depth=2
	v_mov_b64_e32 v[8:9], s[64:65]
	flat_load_dwordx4 v[26:29], v[8:9]
	flat_load_dwordx4 v[62:65], v[8:9] offset:16
	flat_load_dwordx4 v[66:69], v[8:9] offset:32
	flat_load_dwordx4 v[70:73], v[8:9] offset:48
	s_nop 0
	flat_load_dwordx2 v[8:9], v[8:9] offset:64
	s_mul_i32 s9, s76, s62
	s_add_i32 s11, s9, s43
	s_abs_i32 s45, s11
	s_mul_hi_u32 s54, s45, s96
	s_mul_i32 s55, s54, s47
	s_ashr_i32 s44, s11, 31
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
	v_readlane_b32 s45, v88, 8
	s_mul_i32 s45, s44, s45
	s_sub_i32 s11, s11, s45
	v_readlane_b32 s84, v88, 0
	s_cmp_eq_u32 s44, 0
	v_readlane_b32 s85, v88, 1
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 1
	v_mov_b32_e32 v12, s85
	s_mov_b32 s10, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v28, 0, v28, vcc
	v_cndmask_b32_e32 v29, 0, v29, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 2
	v_sub_co_u32_e64 v26, s[54:55], s84, v26
	v_cndmask_b32_e32 v28, v28, v62, vcc
	s_nop 0
	v_subb_co_u32_e64 v27, s[54:55], v12, v27, s[54:55]
	v_cndmask_b32_e32 v12, v29, v63, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 3
	v_cndmask_b32_e32 v28, v28, v64, vcc
	v_cndmask_b32_e32 v12, v12, v65, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 4
	v_cndmask_b32_e32 v12, v12, v67, vcc
	v_cndmask_b32_e32 v28, v28, v66, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 5
	v_cndmask_b32_e32 v28, v28, v68, vcc
	v_cndmask_b32_e32 v12, v12, v69, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 6
	v_cndmask_b32_e32 v12, v12, v71, vcc
	v_cndmask_b32_e32 v28, v28, v70, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s44, 7
	v_cndmask_b32_e32 v28, v28, v72, vcc
	v_cndmask_b32_e32 v12, v12, v73, vcc
	s_cselect_b64 vcc, -1, 0
	s_abs_i32 s54, s11
	s_mul_hi_u32 s55, s54, s46
	s_mul_i32 s55, s55, s59
	s_sub_i32 s54, s54, s55
	s_ashr_i32 s44, s11, 31
	s_sub_i32 s55, s54, s59
	s_cmp_ge_u32 s54, s59
	s_cselect_b32 s54, s55, s54
	s_sub_i32 s55, s54, s59
	s_cmp_ge_u32 s54, s59
	s_cselect_b32 s54, s55, s54
	v_readlane_b32 s55, v88, 16
	s_add_i32 s11, s11, s55
	s_add_i32 s55, s44, s8
	s_xor_b32 s54, s54, s44
	s_sub_i32 s45, s55, s45
	s_sub_i32 s44, s44, s54
	v_cndmask_b32_e32 v9, v12, v9, vcc
	v_cndmask_b32_e32 v8, v28, v8, vcc
	s_sub_i32 s45, s45, s54
	s_add_i32 s11, s11, s44
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[8:9]
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_mul_i32 s44, s80, s45
	s_mul_i32 s11, s11, s80
	v_cndmask_b32_e32 v9, 0, v27, vcc
	v_cndmask_b32_e32 v8, 0, v26, vcc
	s_add_i32 s44, s91, s44
	s_add_i32 s54, s11, s79
	v_lshl_add_u64 v[26:27], v[8:9], 0, v[24:25]
	s_ashr_i32 s45, s44, 31
	s_ashr_i32 s55, s54, 31
	v_lshl_add_u64 v[8:9], s[54:55], 1, v[8:9]
	v_lshl_add_u64 v[26:27], s[44:45], 1, v[26:27]
	s_branch .LBB5_124
.LBB5_122:                              ; %Flow774
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[54:55]
.LBB5_123:                              ; %_ZN17hk_gemm_rs_mi300x16emit_band_scalarEP14__hip_bfloat16lPKS0_iiiijj.exit
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_add_i32 s10, s10, s73
	s_cmp_ge_i32 s10, s62
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[86:87]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB5_119
.LBB5_124:                              ; %.lr.ph459
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_120 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB5_160 Depth 4
                                        ;         Child Loop BB5_145 Depth 4
                                        ;         Child Loop BB5_154 Depth 4
	s_andn2_b64 vcc, exec, s[92:93]
	s_cbranch_vccnz .LBB5_126
; %bb.125:                              ;   in Loop: Header=BB5_124 Depth=3
	flat_load_ushort v12, v[6:7]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	s_branch .LBB5_127
.LBB5_126:                              ;   in Loop: Header=BB5_124 Depth=3
	v_mov_b32_e32 v12, 0
.LBB5_127:                              ;   in Loop: Header=BB5_124 Depth=3
	s_add_i32 s11, s10, s9
	s_add_i32 s82, s11, s73
	v_cmp_le_i32_e32 vcc, s11, v44
	v_cmp_gt_i32_e64 s[54:55], s82, v44
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_129
; %bb.128:                              ;   in Loop: Header=BB5_124 Depth=3
	v_add_f32_e32 v28, v12, v2
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s74
	v_subrev_u32_e32 v29, s11, v44
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB5_129:                              ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s11, v46
	v_cmp_gt_i32_e64 s[54:55], s82, v46
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_131
; %bb.130:                              ;   in Loop: Header=BB5_124 Depth=3
	v_add_f32_e32 v28, v12, v3
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s74
	v_subrev_u32_e32 v29, s11, v46
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB5_131:                              ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s11, v47
	v_cmp_gt_i32_e64 s[54:55], s82, v47
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_133
; %bb.132:                              ;   in Loop: Header=BB5_124 Depth=3
	v_add_f32_e32 v28, v12, v4
	v_bfe_u32 v29, v28, 16, 1
	v_add3_u32 v28, v28, v29, s74
	v_subrev_u32_e32 v29, s11, v47
	v_lshl_add_u32 v29, v29, 7, v45
	ds_write_b16_d16_hi v29, v28
.LBB5_133:                              ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cmp_le_i32_e32 vcc, s11, v48
	v_cmp_gt_i32_e64 s[54:55], s82, v48
	s_and_b64 s[54:55], vcc, s[54:55]
	s_and_saveexec_b64 s[44:45], s[54:55]
	s_cbranch_execz .LBB5_135
; %bb.134:                              ;   in Loop: Header=BB5_124 Depth=3
	v_add_f32_e32 v12, v12, v5
	v_bfe_u32 v28, v12, 16, 1
	v_add3_u32 v12, v12, v28, s74
	v_subrev_u32_e32 v28, s11, v48
	v_lshl_add_u32 v28, v28, 7, v45
	ds_write_b16_d16_hi v28, v12
.LBB5_135:                              ; %_ZN17hk_gemm_rs_mi300x19stage_fragment_bf16IN7kittens2rtIfLi16ELi16ENS1_5ducks9rt_layout3colEEEEEvP14__hip_bfloat16iRKT_PKS7_iiiiii.exit
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[44:45]
	s_mul_i32 s11, s89, s10
	s_mul_hi_u32 s44, s88, s10
	s_add_i32 s45, s44, s11
	s_mul_i32 s44, s88, s10
	s_andn2_b64 vcc, exec, s[94:95]
	v_lshl_add_u64 v[28:29], s[44:45], 1, v[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB5_157
; %bb.136:                              ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b64 s[54:55], -1
	s_mov_b64 s[82:83], -1
	s_and_saveexec_b64 s[44:45], s[6:7]
	s_cbranch_execz .LBB5_142
; %bb.137:                              ; %.lr.ph.i
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b64 vcc, s[48:49]
	s_and_saveexec_b64 s[82:83], s[50:51]
; %bb.138:                              ;   in Loop: Header=BB5_124 Depth=3
	v_lshlrev_b32_e32 v12, 1, v18
	v_lshlrev_b32_e32 v30, 1, v10
	v_add3_u32 v12, v28, v12, v30
	v_or_b32_e32 v12, v20, v12
	v_and_b32_e32 v12, 15, v12
	v_cmp_eq_u32_e32 vcc, 0, v12
	s_andn2_b64 s[84:85], s[48:49], exec
	s_and_b64 vcc, vcc, exec
	s_or_b64 vcc, s[84:85], vcc
; %bb.139:                              ; %Flow770
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[82:83]
	s_mov_b64 s[82:83], 0
	s_and_saveexec_b64 s[84:85], vcc
; %bb.140:                              ; %.critedge.i
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b64 s[82:83], exec
; %bb.141:                              ; %Flow771
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[84:85]
	s_orn2_b64 s[82:83], s[82:83], exec
.LBB5_142:                              ; %_ZN17hk_gemm_rs_mi300x19emit_band_preflightEPK14__hip_bfloat16lS2_iiiijj.exit
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_cndmask_b32_e64 v12, 0, 1, s[82:83]
	s_nop 0
	v_readfirstlane_b32 s11, v12
	s_bitcmp1_b32 s11, 0
	s_cselect_b64 s[44:45], -1, 0
	s_and_b64 vcc, exec, s[44:45]
	s_cbranch_vccnz .LBB5_147
; %bb.143:                              ;   in Loop: Header=BB5_124 Depth=3
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB5_146
; %bb.144:                              ; %.lr.ph.i273.preheader
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v30, v60
	v_mov_b32_e32 v12, v0
.LBB5_145:                              ; %.lr.ph.i273
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_120 Depth=2
                                        ;       Parent Loop BB5_124 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v31, v12, v59
	v_mul_lo_u32 v61, v31, s33
	v_sub_u32_e32 v61, v12, v61
	v_add_u32_e32 v62, 1, v31
	v_subrev_u32_e32 v63, s33, v61
	v_cmp_le_u32_e32 vcc, s33, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_add_u32_e32 v62, 1, v31
	v_cmp_le_u32_e32 vcc, s33, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_xor_b32_e32 v31, s97, v31
	v_subrev_u32_e32 v61, s97, v31
	v_mad_u64_u32 v[62:63], s[82:83], s90, v61, v[12:13]
	v_lshlrev_b32_e32 v31, 7, v31
	v_mul_lo_u32 v63, s77, v61
	v_sub_u32_e32 v31, v31, v63
	v_add_u32_e32 v31, v30, v31
	ds_read_u16 v31, v31
	v_mad_i64_i32 v[64:65], s[82:83], s88, v61, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v63, v13
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]
	v_cmp_le_i32_e32 vcc, s42, v12
	v_lshl_add_u64 v[62:63], v[62:63], 1, v[64:65]
	v_add_u32_e32 v30, 0x400, v30
	s_or_b64 s[44:45], vcc, s[44:45]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[62:63], v31
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_145
.LBB5_146:                              ; %Flow762
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[54:55]
	s_mov_b64 s[54:55], 0
.LBB5_147:                              ; %Flow768
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_andn2_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB5_156
; %bb.148:                              ;   in Loop: Header=BB5_124 Depth=3
	s_and_saveexec_b64 s[54:55], s[6:7]
	s_cbranch_execz .LBB5_155
; %bb.149:                              ; %.lr.ph4.i
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_and_saveexec_b64 s[44:45], s[50:51]
	s_xor_b64 s[44:45], exec, s[44:45]
	s_cbranch_execz .LBB5_151
; %bb.150:                              ;   in Loop: Header=BB5_124 Depth=3
	ds_read_b128 v[62:65], v21
	v_lshl_add_u64 v[30:31], v[18:19], 1, v[28:29]
	v_lshlrev_b32_e32 v12, 1, v10
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[12:13]
	s_waitcnt lgkmcnt(0)
	flat_store_dwordx4 v[30:31], v[62:65]
.LBB5_151:                              ; %Flow765
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_andn2_saveexec_b64 s[44:45], s[44:45]
	s_cbranch_execz .LBB5_155
; %bb.152:                              ; %.preheader.i
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_and_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB5_155
; %bb.153:                              ; %.lr.ph.i275
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b32 s11, 0
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v12, v54
	v_mov_b64_e32 v[30:31], v[26:27]
.LBB5_154:                              ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_120 Depth=2
                                        ;       Parent Loop BB5_124 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ds_read_u16 v61, v12
	s_add_i32 s82, s11, 1
	v_add_u32_e32 v62, s11, v56
	s_cmp_gt_u32 s11, 6
	v_cmp_le_u32_e32 vcc, s72, v62
	s_mov_b32 s11, s82
	s_cselect_b64 s[82:83], -1, 0
	s_or_b64 s[82:83], s[82:83], vcc
	s_and_b64 s[82:83], exec, s[82:83]
	v_add_u32_e32 v12, 2, v12
	s_waitcnt lgkmcnt(0)
	flat_store_short v[30:31], v61
	s_or_b64 s[44:45], s[82:83], s[44:45]
	v_lshl_add_u64 v[30:31], v[30:31], 0, 2
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_154
.LBB5_155:                              ; %Flow767
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_or_b64 exec, exec, s[54:55]
.LBB5_156:                              ; %Flow769
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_branch .LBB5_123
.LBB5_157:                              ;   in Loop: Header=BB5_124 Depth=3
	s_cbranch_execz .LBB5_123
; %bb.158:                              ;   in Loop: Header=BB5_124 Depth=3
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB5_122
; %bb.159:                              ; %.lr.ph.i279.preheader
                                        ;   in Loop: Header=BB5_124 Depth=3
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v30, v60
	v_mov_b32_e32 v12, v0
.LBB5_160:                              ; %.lr.ph.i279
                                        ;   Parent Loop BB5_85 Depth=1
                                        ;     Parent Loop BB5_120 Depth=2
                                        ;       Parent Loop BB5_124 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_mul_hi_u32 v31, v12, v59
	v_mul_lo_u32 v61, v31, s33
	v_sub_u32_e32 v61, v12, v61
	v_add_u32_e32 v62, 1, v31
	v_subrev_u32_e32 v63, s33, v61
	v_cmp_le_u32_e32 vcc, s33, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_add_u32_e32 v62, 1, v31
	v_cmp_le_u32_e32 vcc, s33, v61
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v62, vcc
	v_xor_b32_e32 v31, s97, v31
	v_subrev_u32_e32 v61, s97, v31
	v_mad_u64_u32 v[62:63], s[82:83], s90, v61, v[12:13]
	v_lshlrev_b32_e32 v31, 7, v31
	v_mul_lo_u32 v63, s77, v61
	v_sub_u32_e32 v31, v31, v63
	v_add_u32_e32 v31, v30, v31
	ds_read_u16 v31, v31
	v_mad_i64_i32 v[64:65], s[82:83], s88, v61, 0
	v_add_u32_e32 v12, 0x200, v12
	v_mov_b32_e32 v63, v13
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]
	v_cmp_le_i32_e32 vcc, s42, v12
	v_lshl_add_u64 v[62:63], v[62:63], 1, v[64:65]
	v_add_u32_e32 v30, 0x400, v30
	s_or_b64 s[44:45], vcc, s[44:45]
	s_waitcnt lgkmcnt(0)
	flat_store_short v[62:63], v31
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execnz .LBB5_160
	s_branch .LBB5_122
.LBB5_161:                              ; %._crit_edge463
                                        ;   in Loop: Header=BB5_85 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	s_mov_b64 s[0:1], exec
	v_readlane_b32 s8, v88, 11
	v_readlane_b32 s9, v88, 12
	s_and_b64 s[8:9], s[0:1], s[8:9]
	s_mov_b64 exec, s[8:9]
	s_cbranch_execz .LBB5_163
; %bb.162:                              ;   in Loop: Header=BB5_85 Depth=1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
.LBB5_163:                              ; %_ZN17hk_gemm_rs_mi300x22release_payload_systemEv.exit
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	s_mov_b64 s[48:49], exec
	v_readlane_b32 s0, v88, 40
	v_readlane_b32 s84, v88, 45
	v_readlane_b32 s1, v88, 41
	v_readlane_b32 s76, v88, 42
	v_readlane_b32 s85, v88, 46
	v_readlane_b32 s90, v88, 49
	s_and_b64 s[0:1], s[48:49], s[0:1]
	v_readlane_b32 s72, v88, 8
	v_readlane_b32 s77, v88, 43
	v_readlane_b32 s82, v88, 44
	v_readlane_b32 s83, v88, 47
	v_readlane_b32 s85, v88, 48
	v_readlane_b32 s91, v88, 50
	v_readlane_b32 s10, v88, 59
	v_readlane_b32 s8, v88, 58
	v_readlane_b32 s11, v88, 57
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB5_83
; %bb.164:                              ; %.lr.ph465
                                        ;   in Loop: Header=BB5_85 Depth=1
	s_lshl_b32 s0, s8, 2
	v_readlane_b32 s8, v88, 9
	s_add_i32 s0, s8, s0
	s_sub_i32 s0, s0, s11
	v_readlane_b32 s1, v88, 53
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s1, s10, 2
	v_readlane_b32 s9, v88, 10
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s8, s0, 5
	s_mov_b32 s9, s75
	s_branch .LBB5_166
.LBB5_165:                              ; %_ZN17hk_gemm_rs_mi300x18publish_band_epochEPjPKN10hk_gemm_rs20symmetric_descriptorEiiij.exit
                                        ;   in Loop: Header=BB5_166 Depth=2
	s_add_i32 s9, s9, -1
	s_add_i32 s8, s8, s62
	s_cmp_lg_u32 s9, 0
	s_cbranch_scc0 .LBB5_83
.LBB5_166:                              ;   Parent Loop BB5_85 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b64_e32 v[30:31], s[68:69]
	flat_load_dwordx4 v[2:5], v[30:31]
	flat_load_dwordx4 v[6:9], v[30:31] offset:16
	flat_load_dwordx4 v[26:29], v[30:31] offset:32
	flat_load_dwordx4 v[60:63], v[30:31] offset:48
	s_abs_i32 s1, s8
	flat_load_dwordx2 v[30:31], v[30:31] offset:64
	s_mul_hi_u32 s11, s1, s96
	s_mul_i32 s33, s11, s47
	s_ashr_i32 s0, s8, 31
	s_sub_i32 s1, s1, s33
	s_xor_b32 s0, s0, s2
	s_add_i32 s42, s11, 1
	s_sub_i32 s33, s1, s47
	s_cmp_ge_u32 s1, s47
	s_cselect_b32 s11, s42, s11
	s_cselect_b32 s1, s33, s1
	s_add_i32 s33, s11, 1
	s_cmp_ge_u32 s1, s47
	s_cselect_b32 s1, s33, s11
	s_xor_b32 s1, s1, s0
	s_sub_i32 s11, s1, s0
	v_readlane_b32 s1, v88, 30
	s_mul_i32 s1, s1, s11
	s_add_i32 s1, s8, s1
	s_mul_i32 s0, s11, s72
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s0, s1, s0
	s_add_i32 s0, s8, s0
	v_readlane_b32 s33, v88, 28
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
	s_mul_i32 s10, s60, s56
	s_sub_i32 s0, s0, s33
	s_add_i32 s0, s0, s10
	s_mul_i32 s0, s0, s61
	s_add_i32 s0, s0, s78
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s66, s0
	s_addc_u32 s1, s67, s1
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmp_eq_u32 s11, 0
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 1
	v_mov_b32_e32 v12, s1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e32 v4, 0, v4, vcc
	v_cndmask_b32_e32 v5, 0, v5, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 2
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 3
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_cndmask_b32_e32 v5, v5, v9, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 4
	v_cndmask_b32_e32 v5, v5, v27, vcc
	v_cndmask_b32_e32 v4, v4, v26, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 5
	v_cndmask_b32_e32 v4, v4, v28, vcc
	v_cndmask_b32_e32 v5, v5, v29, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 6
	v_cndmask_b32_e32 v5, v5, v61, vcc
	v_cndmask_b32_e32 v4, v4, v60, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_eq_u32 s11, 7
	v_sub_co_u32_e64 v2, s[0:1], s0, v2
	v_cndmask_b32_e32 v4, v4, v62, vcc
	v_cndmask_b32_e32 v5, v5, v63, vcc
	s_cselect_b64 vcc, -1, 0
	v_subb_co_u32_e64 v3, s[0:1], v12, v3, s[0:1]
	v_cndmask_b32_e32 v5, v5, v31, vcc
	v_cndmask_b32_e32 v4, v4, v30, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]
	s_cmp_lg_u32 s11, s56
	s_mov_b64 s[0:1], -1
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_cndmask_b32_e32 v2, 0, v2, vcc
	s_cbranch_scc1 .LBB5_168
; %bb.167:                              ; %Flow758
                                        ;   in Loop: Header=BB5_166 Depth=2
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB5_165
	s_branch .LBB5_169
.LBB5_168:                              ;   in Loop: Header=BB5_166 Depth=2
	flat_store_dword v[2:3], v1 sc0 sc1
	s_cbranch_execnz .LBB5_165
.LBB5_169:                              ;   in Loop: Header=BB5_166 Depth=2
	flat_store_dword v[2:3], v1 sc1
	s_branch .LBB5_165
.LBB5_170:                              ; %.critedge267
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
		.amdhsa_next_free_vgpr 89
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
	.set _Z21gemm_rs_mi300x_kernelILi32ELi64ELi64ELb1EEv14mi300x_globals.num_vgpr, 89
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
; codeLenInByte = 11592
; TotalNumSgprs: 106
; NumVgprs: 89
; NumAgprs: 0
; TotalNumVgprs: 89
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 89
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
    .vgpr_count:     89
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
    .sgpr_spill_count: 49
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
    .sgpr_spill_count: 72
    .symbol:         _Z21gemm_rs_mi300x_kernelILi128ELi256ELi32ELb1EEv14mi300x_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     153
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
    .vgpr_count:     233
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
    .vgpr_count:     233
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
    .vgpr_count:     89
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
