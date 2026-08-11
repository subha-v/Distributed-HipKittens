; <256,256,32,true>  symbol _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
; k-loop blocks:
;   .LBB4_87 15787..15857
;   %bb.88 15858..15917
;   .LBB4_89 15918..16012
;   %bb.90 16013..16072
;   .LBB4_91 16073..16112
;   %bb.92 16113..16118
     15747| 	v_and_b32_e32 v152, 56, v152
     15748| 	;;#ASMSTART
     15749| 	global_load_dwordx4 v[146:149], v[146:147], off
     15750| 
     15751| 	;;#ASMEND
     15752| 	;;#ASMSTART
     15753| 	s_waitcnt vmcnt(0)
     15754| 	;;#ASMEND
     15755| 	v_xor_b32_e32 v151, v152, v151
     15756| 	;;#ASMSTART
     15757| 	ds_write_b64 v151, v[142:143]
     15758| 
     15759| 	;;#ASMEND
     15760| 	v_add_u32_e32 v142, v150, v168
     15761| 	v_lshrrev_b32_e32 v143, 4, v142
     15762| 	v_and_b32_e32 v143, 56, v143
     15763| 	v_xor_b32_e32 v142, v143, v142
     15764| 	v_add_u32_e32 v138, v138, v171
     15765| 	;;#ASMSTART
     15766| 	ds_write_b64 v142, v[144:145]
     15767| 
     15768| 	;;#ASMEND
     15769| 	v_lshrrev_b32_e32 v142, 4, v138
     15770| 	v_and_b32_e32 v142, 56, v142
     15771| 	v_xor_b32_e32 v138, v142, v138
     15772| 	;;#ASMSTART
     15773| 	ds_write_b64 v138, v[146:147]
     15774| 
     15775| 	;;#ASMEND
     15776| 	v_add_u32_e32 v138, v150, v171
     15777| 	v_lshrrev_b32_e32 v142, 4, v138
     15778| 	v_and_b32_e32 v142, 56, v142
     15779| 	v_xor_b32_e32 v138, v142, v138
     15780| 	;;#ASMSTART
     15781| 	ds_write_b64 v138, v[148:149]
     15782| 
     15783| 	;;#ASMEND
     15784| 	;;#ASMSTART
     15785| 	s_waitcnt lgkmcnt(0)
     15786| 	;;#ASMEND
>>   15787| .LBB4_87:                               ;   Parent Loop BB4_84 Depth=1
>>   15788|                                         ; =>  This Inner Loop Header: Depth=2
>>   15789| 	s_lshl_b32 s1, s0, 14
>>   15790| 	s_and_b32 s1, s1, 0x4000
>>   15791| 	s_add_i32 s52, s65, s1
>>   15792| 	v_add_u32_e32 v228, s52, v178
>>   15793| 	v_add_u32_e32 v142, v228, v216
>>   15794| 	v_lshrrev_b32_e32 v143, 4, v142
>>   15795| 	v_and_b32_e32 v143, 56, v143
>>   15796| 	v_xor_b32_e32 v142, v143, v142
>>   15797| 	;;#ASMSTART
>>   15798| 	ds_read_b64 v[164:165], v142 offset:0
>>   15799| 
>>   15800| 	;;#ASMEND
>>   15801| 	;;#ASMSTART
>>   15802| 	ds_read_b64 v[162:163], v142 offset:0x400
>>   15803| 
>>   15804| 	;;#ASMEND
>>   15805| 	;;#ASMSTART
>>   15806| 	ds_read_b64 v[160:161], v142 offset:0x800
>>   15807| 
>>   15808| 	;;#ASMEND
>>   15809| 	s_add_i32 s1, s69, s1
>>   15810| 	;;#ASMSTART
>>   15811| 	ds_read_b64 v[158:159], v142 offset:0xc00
>>   15812| 
>>   15813| 	;;#ASMEND
>>   15814| 	v_add_u32_e32 v138, s1, v179
>>   15815| 	;;#ASMSTART
>>   15816| 	ds_read_b64 v[156:157], v142 offset:0x1000
>>   15817| 
>>   15818| 	;;#ASMEND
>>   15819| 	;;#ASMSTART
>>   15820| 	ds_read_b64 v[146:147], v142 offset:0x1400
>>   15821| 
>>   15822| 	;;#ASMEND
>>   15823| 	v_add_u32_e32 v148, v138, v216
>>   15824| 	;;#ASMSTART
>>   15825| 	ds_read_b64 v[144:145], v142 offset:0x1800
>>   15826| 
>>   15827| 	;;#ASMEND
>>   15828| 	v_lshrrev_b32_e32 v149, 4, v148
>>   15829| 	;;#ASMSTART
>>   15830| 	ds_read_b64 v[142:143], v142 offset:0x1c00
>>   15831| 
>>   15832| 	;;#ASMEND
>>   15833| 	v_and_b32_e32 v149, 56, v149
>>   15834| 	v_xor_b32_e32 v154, v149, v148
>>   15835| 	;;#ASMSTART
>>   15836| 	ds_read_b64 v[148:149], v154 offset:0
>>   15837| 
>>   15838| 	;;#ASMEND
>>   15839| 	;;#ASMSTART
>>   15840| 	ds_read_b64 v[150:151], v154 offset:0x400
>>   15841| 
>>   15842| 	;;#ASMEND
>>   15843| 	;;#ASMSTART
>>   15844| 	ds_read_b64 v[152:153], v154 offset:0x800
>>   15845| 
>>   15846| 	;;#ASMEND
>>   15847| 	;;#ASMSTART
>>   15848| 	ds_read_b64 v[154:155], v154 offset:0xc00
>>   15849| 
>>   15850| 	;;#ASMEND
>>   15851| 	s_cmp_eq_u32 s37, s0
>>   15852| 	s_cselect_b64 s[92:93], -1, 0
>>   15853| 	s_cmp_lg_u32 s37, s0
>>   15854| 	;;#ASMSTART
>>   15855| 	s_waitcnt lgkmcnt(0)
>>   15856| 	;;#ASMEND
>>   15857| 	s_cbranch_scc1 .LBB4_89
>>   15858| ; %bb.88:                               ; %.loopexit.i
>>   15859|                                         ;   in Loop: Header=BB4_87 Depth=2
>>   15860| 	s_or_b64 s[0:1], s[10:11], s[6:7]
>>   15861| 	s_or_b64 s[0:1], s[0:1], s[4:5]
>>   15862| 	v_cndmask_b32_e64 v229, 0, v164, s[8:9]
>>   15863| 	v_and_b32_e32 v230, 0xffff0000, v165
>>   15864| 	s_mov_b64 vcc, s[0:1]
>>   15865| 	v_cndmask_b32_e64 v230, v230, v165, s[6:7]
>>   15866| 	v_cndmask_b32_sdwa v164, v229, v164, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15867| 	s_mov_b64 vcc, s[10:11]
>>   15868| 	v_cndmask_b32_sdwa v165, v230, v165, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15869| 	v_cndmask_b32_e64 v229, 0, v162, s[8:9]
>>   15870| 	v_and_b32_e32 v230, 0xffff0000, v163
>>   15871| 	s_mov_b64 vcc, s[0:1]
>>   15872| 	v_cndmask_b32_e64 v230, v230, v163, s[6:7]
>>   15873| 	v_cndmask_b32_sdwa v162, v229, v162, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15874| 	s_mov_b64 vcc, s[10:11]
>>   15875| 	v_cndmask_b32_sdwa v163, v230, v163, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15876| 	v_cndmask_b32_e64 v229, 0, v160, s[8:9]
>>   15877| 	v_and_b32_e32 v230, 0xffff0000, v161
>>   15878| 	s_mov_b64 vcc, s[0:1]
>>   15879| 	v_cndmask_b32_e64 v230, v230, v161, s[6:7]
>>   15880| 	v_cndmask_b32_sdwa v160, v229, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15881| 	s_mov_b64 vcc, s[10:11]
>>   15882| 	v_cndmask_b32_sdwa v161, v230, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15883| 	v_cndmask_b32_e64 v229, 0, v158, s[8:9]
>>   15884| 	v_and_b32_e32 v230, 0xffff0000, v159
>>   15885| 	s_mov_b64 vcc, s[0:1]
>>   15886| 	v_cndmask_b32_e64 v230, v230, v159, s[6:7]
>>   15887| 	v_cndmask_b32_sdwa v158, v229, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15888| 	s_mov_b64 vcc, s[10:11]
>>   15889| 	v_cndmask_b32_sdwa v159, v230, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15890| 	v_cndmask_b32_e64 v229, 0, v156, s[8:9]
>>   15891| 	v_and_b32_e32 v230, 0xffff0000, v157
>>   15892| 	s_mov_b64 vcc, s[0:1]
>>   15893| 	v_cndmask_b32_e64 v230, v230, v157, s[6:7]
>>   15894| 	v_cndmask_b32_sdwa v156, v229, v156, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15895| 	s_mov_b64 vcc, s[10:11]
>>   15896| 	v_cndmask_b32_sdwa v157, v230, v157, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15897| 	v_cndmask_b32_e64 v229, 0, v146, s[8:9]
>>   15898| 	v_and_b32_e32 v230, 0xffff0000, v147
>>   15899| 	s_mov_b64 vcc, s[0:1]
>>   15900| 	v_cndmask_b32_e64 v230, v230, v147, s[6:7]
>>   15901| 	v_cndmask_b32_sdwa v146, v229, v146, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15902| 	s_mov_b64 vcc, s[10:11]
>>   15903| 	v_cndmask_b32_sdwa v147, v230, v147, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15904| 	v_cndmask_b32_e64 v229, 0, v144, s[8:9]
>>   15905| 	v_and_b32_e32 v230, 0xffff0000, v145
>>   15906| 	s_mov_b64 vcc, s[0:1]
>>   15907| 	v_cndmask_b32_e64 v230, v230, v145, s[6:7]
>>   15908| 	v_cndmask_b32_sdwa v144, v229, v144, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15909| 	s_mov_b64 vcc, s[10:11]
>>   15910| 	v_cndmask_b32_sdwa v145, v230, v145, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15911| 	v_cndmask_b32_e64 v229, 0, v142, s[8:9]
>>   15912| 	v_and_b32_e32 v230, 0xffff0000, v143
>>   15913| 	s_mov_b64 vcc, s[0:1]
>>   15914| 	v_cndmask_b32_e64 v230, v230, v143, s[6:7]
>>   15915| 	v_cndmask_b32_sdwa v142, v229, v142, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15916| 	s_mov_b64 vcc, s[10:11]
>>   15917| 	v_cndmask_b32_sdwa v143, v230, v143, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   15918| .LBB4_89:                               ;   in Loop: Header=BB4_87 Depth=2
>>   15919| 	s_nop 1
>>   15920| 	v_mfma_f32_16x16x16_bf16 v[6:9], v[142:143], v[148:149], v[6:9]
>>   15921| 	v_add_u32_e32 v138, v138, v221
>>   15922| 	s_andn2_b64 vcc, exec, s[92:93]
>>   15923| 	v_mfma_f32_16x16x16_bf16 v[2:5], v[142:143], v[150:151], v[2:5]
>>   15924| 	v_mfma_f32_16x16x16_bf16 v[70:73], v[142:143], v[152:153], v[70:73]
>>   15925| 	v_mfma_f32_16x16x16_bf16 v[66:69], v[142:143], v[154:155], v[66:69]
>>   15926| 	v_add_u32_e32 v142, v228, v221
>>   15927| 	v_lshrrev_b32_e32 v143, 4, v142
>>   15928| 	v_and_b32_e32 v143, 56, v143
>>   15929| 	v_mfma_f32_16x16x16_bf16 v[78:81], v[158:159], v[148:149], v[78:81]
>>   15930| 	v_xor_b32_e32 v142, v143, v142
>>   15931| 	v_mfma_f32_16x16x16_bf16 v[74:77], v[158:159], v[150:151], v[74:77]
>>   15932| 	v_mfma_f32_16x16x16_bf16 v[62:65], v[158:159], v[152:153], v[62:65]
>>   15933| 	v_mfma_f32_16x16x16_bf16 v[58:61], v[158:159], v[154:155], v[58:61]
>>   15934| 	;;#ASMSTART
>>   15935| 	ds_read_b64 v[158:159], v142 offset:0
>>   15936| 
>>   15937| 	;;#ASMEND
>>   15938| 	v_mfma_f32_16x16x16_bf16 v[94:97], v[160:161], v[148:149], v[94:97]
>>   15939| 	v_mfma_f32_16x16x16_bf16 v[90:93], v[160:161], v[150:151], v[90:93]
>>   15940| 	v_mfma_f32_16x16x16_bf16 v[86:89], v[160:161], v[152:153], v[86:89]
>>   15941| 	v_mfma_f32_16x16x16_bf16 v[82:85], v[160:161], v[154:155], v[82:85]
>>   15942| 	;;#ASMSTART
>>   15943| 	ds_read_b64 v[160:161], v142 offset:0x400
>>   15944| 
>>   15945| 	;;#ASMEND
>>   15946| 	v_mfma_f32_16x16x16_bf16 v[54:57], v[156:157], v[148:149], v[54:57]
>>   15947| 	v_mfma_f32_16x16x16_bf16 v[50:53], v[156:157], v[150:151], v[50:53]
>>   15948| 	v_mfma_f32_16x16x16_bf16 v[46:49], v[156:157], v[152:153], v[46:49]
>>   15949| 	v_mfma_f32_16x16x16_bf16 v[42:45], v[156:157], v[154:155], v[42:45]
>>   15950| 	;;#ASMSTART
>>   15951| 	ds_read_b64 v[156:157], v142 offset:0x800
>>   15952| 
>>   15953| 	;;#ASMEND
>>   15954| 	v_mfma_f32_16x16x16_bf16 v[118:121], v[164:165], v[152:153], v[118:121]
>>   15955| 	v_mfma_f32_16x16x16_bf16 v[102:105], v[162:163], v[152:153], v[102:105]
>>   15956| 	v_mfma_f32_16x16x16_bf16 v[30:33], v[146:147], v[152:153], v[30:33]
>>   15957| 	v_mfma_f32_16x16x16_bf16 v[14:17], v[144:145], v[152:153], v[14:17]
>>   15958| 	;;#ASMSTART
>>   15959| 	ds_read_b64 v[152:153], v142 offset:0xc00
>>   15960| 
>>   15961| 	;;#ASMEND
>>   15962| 	v_mfma_f32_16x16x16_bf16 v[114:117], v[164:165], v[154:155], v[114:117]
>>   15963| 	v_mfma_f32_16x16x16_bf16 v[98:101], v[162:163], v[154:155], v[98:101]
>>   15964| 	v_mfma_f32_16x16x16_bf16 v[26:29], v[146:147], v[154:155], v[26:29]
>>   15965| 	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[154:155], v[10:13]
>>   15966| 	;;#ASMSTART
>>   15967| 	ds_read_b64 v[154:155], v142 offset:0x1000
>>   15968| 
>>   15969| 	;;#ASMEND
>>   15970| 	v_mfma_f32_16x16x16_bf16 v[122:125], v[164:165], v[150:151], v[122:125]
>>   15971| 	v_mfma_f32_16x16x16_bf16 v[106:109], v[162:163], v[150:151], v[106:109]
>>   15972| 	v_mfma_f32_16x16x16_bf16 v[34:37], v[146:147], v[150:151], v[34:37]
>>   15973| 	v_mfma_f32_16x16x16_bf16 v[18:21], v[144:145], v[150:151], v[18:21]
>>   15974| 	;;#ASMSTART
>>   15975| 	ds_read_b64 v[150:151], v142 offset:0x1400
>>   15976| 
>>   15977| 	;;#ASMEND
>>   15978| 	v_mfma_f32_16x16x16_bf16 v[126:129], v[164:165], v[148:149], v[126:129]
>>   15979| 	v_mfma_f32_16x16x16_bf16 v[110:113], v[162:163], v[148:149], v[110:113]
>>   15980| 	v_mfma_f32_16x16x16_bf16 v[38:41], v[146:147], v[148:149], v[38:41]
>>   15981| 	v_mfma_f32_16x16x16_bf16 v[22:25], v[144:145], v[148:149], v[22:25]
>>   15982| 	;;#ASMSTART
>>   15983| 	ds_read_b64 v[148:149], v142 offset:0x1800
>>   15984| 
>>   15985| 	;;#ASMEND
>>   15986| 	v_lshrrev_b32_e32 v144, 4, v138
>>   15987| 	;;#ASMSTART
>>   15988| 	ds_read_b64 v[142:143], v142 offset:0x1c00
>>   15989| 
>>   15990| 	;;#ASMEND
>>   15991| 	v_and_b32_e32 v144, 56, v144
>>   15992| 	v_xor_b32_e32 v138, v144, v138
>>   15993| 	;;#ASMSTART
>>   15994| 	ds_read_b64 v[162:163], v138 offset:0
>>   15995| 
>>   15996| 	;;#ASMEND
>>   15997| 	;;#ASMSTART
>>   15998| 	ds_read_b64 v[164:165], v138 offset:0x400
>>   15999| 
>>   16000| 	;;#ASMEND
>>   16001| 	;;#ASMSTART
>>   16002| 	ds_read_b64 v[146:147], v138 offset:0x800
>>   16003| 
>>   16004| 	;;#ASMEND
>>   16005| 	;;#ASMSTART
>>   16006| 	ds_read_b64 v[144:145], v138 offset:0xc00
>>   16007| 
>>   16008| 	;;#ASMEND
>>   16009| 	;;#ASMSTART
>>   16010| 	s_waitcnt lgkmcnt(0)
>>   16011| 	;;#ASMEND
>>   16012| 	s_cbranch_vccnz .LBB4_91
>>   16013| ; %bb.90:                               ; %.loopexit.i.1
>>   16014|                                         ;   in Loop: Header=BB4_87 Depth=2
>>   16015| 	s_or_b64 s[0:1], s[18:19], s[14:15]
>>   16016| 	s_or_b64 s[0:1], s[0:1], s[12:13]
>>   16017| 	v_cndmask_b32_e64 v138, 0, v158, s[16:17]
>>   16018| 	v_and_b32_e32 v228, 0xffff0000, v159
>>   16019| 	s_mov_b64 vcc, s[0:1]
>>   16020| 	v_cndmask_b32_e64 v228, v228, v159, s[14:15]
>>   16021| 	v_cndmask_b32_sdwa v158, v138, v158, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16022| 	s_mov_b64 vcc, s[18:19]
>>   16023| 	v_cndmask_b32_sdwa v159, v228, v159, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16024| 	v_cndmask_b32_e64 v138, 0, v160, s[16:17]
>>   16025| 	v_and_b32_e32 v228, 0xffff0000, v161
>>   16026| 	s_mov_b64 vcc, s[0:1]
>>   16027| 	v_cndmask_b32_e64 v228, v228, v161, s[14:15]
>>   16028| 	v_cndmask_b32_sdwa v160, v138, v160, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16029| 	s_mov_b64 vcc, s[18:19]
>>   16030| 	v_cndmask_b32_sdwa v161, v228, v161, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16031| 	v_cndmask_b32_e64 v138, 0, v156, s[16:17]
>>   16032| 	v_and_b32_e32 v228, 0xffff0000, v157
>>   16033| 	s_mov_b64 vcc, s[0:1]
>>   16034| 	v_cndmask_b32_e64 v228, v228, v157, s[14:15]
>>   16035| 	v_cndmask_b32_sdwa v156, v138, v156, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16036| 	s_mov_b64 vcc, s[18:19]
>>   16037| 	v_cndmask_b32_sdwa v157, v228, v157, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16038| 	v_cndmask_b32_e64 v138, 0, v152, s[16:17]
>>   16039| 	v_and_b32_e32 v228, 0xffff0000, v153
>>   16040| 	s_mov_b64 vcc, s[0:1]
>>   16041| 	v_cndmask_b32_e64 v228, v228, v153, s[14:15]
>>   16042| 	v_cndmask_b32_sdwa v152, v138, v152, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16043| 	s_mov_b64 vcc, s[18:19]
>>   16044| 	v_cndmask_b32_sdwa v153, v228, v153, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16045| 	v_cndmask_b32_e64 v138, 0, v154, s[16:17]
>>   16046| 	v_and_b32_e32 v228, 0xffff0000, v155
>>   16047| 	s_mov_b64 vcc, s[0:1]
>>   16048| 	v_cndmask_b32_e64 v228, v228, v155, s[14:15]
>>   16049| 	v_cndmask_b32_sdwa v154, v138, v154, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16050| 	s_mov_b64 vcc, s[18:19]
>>   16051| 	v_cndmask_b32_sdwa v155, v228, v155, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16052| 	v_cndmask_b32_e64 v138, 0, v150, s[16:17]
>>   16053| 	v_and_b32_e32 v228, 0xffff0000, v151
>>   16054| 	s_mov_b64 vcc, s[0:1]
>>   16055| 	v_cndmask_b32_e64 v228, v228, v151, s[14:15]
>>   16056| 	v_cndmask_b32_sdwa v150, v138, v150, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16057| 	s_mov_b64 vcc, s[18:19]
>>   16058| 	v_cndmask_b32_sdwa v151, v228, v151, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16059| 	v_cndmask_b32_e64 v138, 0, v148, s[16:17]
>>   16060| 	v_and_b32_e32 v228, 0xffff0000, v149
>>   16061| 	s_mov_b64 vcc, s[0:1]
>>   16062| 	v_cndmask_b32_e64 v228, v228, v149, s[14:15]
>>   16063| 	v_cndmask_b32_sdwa v148, v138, v148, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16064| 	s_mov_b64 vcc, s[18:19]
>>   16065| 	v_cndmask_b32_sdwa v149, v228, v149, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16066| 	v_cndmask_b32_e64 v138, 0, v142, s[16:17]
>>   16067| 	v_and_b32_e32 v228, 0xffff0000, v143
>>   16068| 	s_mov_b64 vcc, s[0:1]
>>   16069| 	v_cndmask_b32_e64 v228, v228, v143, s[14:15]
>>   16070| 	v_cndmask_b32_sdwa v142, v138, v142, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16071| 	s_mov_b64 vcc, s[18:19]
>>   16072| 	v_cndmask_b32_sdwa v143, v228, v143, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16073| .LBB4_91:                               ;   in Loop: Header=BB4_87 Depth=2
>>   16074| 	s_add_u32 s28, s28, 64
>>   16075| 	v_mfma_f32_16x16x16_bf16 v[126:129], v[158:159], v[162:163], v[126:129]
>>   16076| 	s_addc_u32 s29, s29, 0
>>   16077| 	s_add_u32 s30, s30, 64
>>   16078| 	s_addc_u32 s31, s31, 0
>>   16079| 	v_mfma_f32_16x16x16_bf16 v[122:125], v[158:159], v[164:165], v[122:125]
>>   16080| 	s_cmp_eq_u32 s83, s50
>>   16081| 	s_barrier
>>   16082| 	v_mfma_f32_16x16x16_bf16 v[118:121], v[158:159], v[146:147], v[118:121]
>>   16083| 	v_mfma_f32_16x16x16_bf16 v[114:117], v[158:159], v[144:145], v[114:117]
>>   16084| 	v_mfma_f32_16x16x16_bf16 v[110:113], v[160:161], v[162:163], v[110:113]
>>   16085| 	v_mfma_f32_16x16x16_bf16 v[106:109], v[160:161], v[164:165], v[106:109]
>>   16086| 	v_mfma_f32_16x16x16_bf16 v[102:105], v[160:161], v[146:147], v[102:105]
>>   16087| 	v_mfma_f32_16x16x16_bf16 v[98:101], v[160:161], v[144:145], v[98:101]
>>   16088| 	v_mfma_f32_16x16x16_bf16 v[94:97], v[156:157], v[162:163], v[94:97]
>>   16089| 	v_mfma_f32_16x16x16_bf16 v[90:93], v[156:157], v[164:165], v[90:93]
>>   16090| 	v_mfma_f32_16x16x16_bf16 v[86:89], v[156:157], v[146:147], v[86:89]
>>   16091| 	v_mfma_f32_16x16x16_bf16 v[82:85], v[156:157], v[144:145], v[82:85]
>>   16092| 	v_mfma_f32_16x16x16_bf16 v[78:81], v[152:153], v[162:163], v[78:81]
>>   16093| 	v_mfma_f32_16x16x16_bf16 v[74:77], v[152:153], v[164:165], v[74:77]
>>   16094| 	v_mfma_f32_16x16x16_bf16 v[62:65], v[152:153], v[146:147], v[62:65]
>>   16095| 	v_mfma_f32_16x16x16_bf16 v[58:61], v[152:153], v[144:145], v[58:61]
>>   16096| 	v_mfma_f32_16x16x16_bf16 v[54:57], v[154:155], v[162:163], v[54:57]
>>   16097| 	v_mfma_f32_16x16x16_bf16 v[50:53], v[154:155], v[164:165], v[50:53]
>>   16098| 	v_mfma_f32_16x16x16_bf16 v[46:49], v[154:155], v[146:147], v[46:49]
>>   16099| 	v_mfma_f32_16x16x16_bf16 v[42:45], v[154:155], v[144:145], v[42:45]
>>   16100| 	v_mfma_f32_16x16x16_bf16 v[38:41], v[150:151], v[162:163], v[38:41]
>>   16101| 	v_mfma_f32_16x16x16_bf16 v[34:37], v[150:151], v[164:165], v[34:37]
>>   16102| 	v_mfma_f32_16x16x16_bf16 v[30:33], v[150:151], v[146:147], v[30:33]
>>   16103| 	v_mfma_f32_16x16x16_bf16 v[26:29], v[150:151], v[144:145], v[26:29]
>>   16104| 	v_mfma_f32_16x16x16_bf16 v[22:25], v[148:149], v[162:163], v[22:25]
>>   16105| 	v_mfma_f32_16x16x16_bf16 v[18:21], v[148:149], v[164:165], v[18:21]
>>   16106| 	v_mfma_f32_16x16x16_bf16 v[14:17], v[148:149], v[146:147], v[14:17]
>>   16107| 	v_mfma_f32_16x16x16_bf16 v[10:13], v[148:149], v[144:145], v[10:13]
>>   16108| 	v_mfma_f32_16x16x16_bf16 v[6:9], v[142:143], v[162:163], v[6:9]
>>   16109| 	v_mfma_f32_16x16x16_bf16 v[2:5], v[142:143], v[164:165], v[2:5]
>>   16110| 	v_mfma_f32_16x16x16_bf16 v[70:73], v[142:143], v[146:147], v[70:73]
>>   16111| 	v_mfma_f32_16x16x16_bf16 v[66:69], v[142:143], v[144:145], v[66:69]
>>   16112| 	s_cbranch_scc1 .LBB4_93
>>   16113| ; %bb.92:                               ;   in Loop: Header=BB4_87 Depth=2
>>   16114| 	s_mov_b32 s0, s50
>>   16115| 	s_add_i32 s50, s0, 1
>>   16116| 	s_cmp_ge_i32 s50, s84
>>   16117| 	s_cbranch_scc0 .LBB4_86
>>   16118| 	s_branch .LBB4_87
     16119| .LBB4_93:                               ; %Flow2149
     16120|                                         ;   in Loop: Header=BB4_84 Depth=1
     16121| 	s_mov_b64 s[0:1], exec
     16122| 	v_readlane_b32 s28, v232, 39
     16123| 	v_readlane_b32 s29, v232, 40
     16124| 	s_and_b64 s[28:29], s[0:1], s[28:29]
     16125| 	s_mov_b64 exec, s[28:29]
     16126| 	s_cbranch_execz .LBB4_102
     16127| ; %bb.94:                               ;   in Loop: Header=BB4_84 Depth=1
     16128| 	v_readlane_b32 s28, v232, 35
     16129| 	v_readlane_b32 s29, v232, 36
     16130| 	s_and_b64 exec, exec, s[28:29]
     16131| 	s_cbranch_execz .LBB4_102
     16132| ; %bb.95:                               ;   in Loop: Header=BB4_84 Depth=1
     16133| 	v_add_u32_e32 v138, s64, v180
     16134| 	v_sub_u32_e32 v143, 0, v138
     16135| 	v_max_i32_e32 v143, v138, v143
     16136| 	v_mul_hi_u32 v144, v143, s71
     16137| 	v_mul_lo_u32 v145, v144, s70
     16138| 	v_sub_u32_e32 v143, v143, v145
     16139| 	v_add_u32_e32 v145, 1, v144
     16140| 	v_cmp_le_u32_e32 vcc, s70, v143
     16141| 	v_ashrrev_i32_e32 v142, 31, v138
     16142| 	v_xor_b32_e32 v142, s23, v142
     16143| 	v_cndmask_b32_e32 v144, v144, v145, vcc
     16144| 	v_subrev_u32_e32 v145, s70, v143
     16145| 	v_cndmask_b32_e32 v143, v143, v145, vcc
     16146| 	v_add_u32_e32 v145, 1, v144
     16147| 	v_cmp_le_u32_e32 vcc, s70, v143
     16148| 	s_nop 1
     16149| 	v_cndmask_b32_e32 v143, v144, v145, vcc
     16150| 	v_xor_b32_e32 v143, v143, v142
     16151| 	v_sub_u32_e32 v142, v143, v142
     16152| 	v_mul_lo_u32 v143, v142, s33
     16153| 	v_sub_u32_e32 v138, v138, v143
     16154| 	v_sub_u32_e32 v144, 0, v138
     16155| 	v_ashrrev_i32_e32 v143, 31, v138
     16156| 	v_max_i32_e32 v138, v138, v144
     16157| 	v_mul_hi_u32 v144, v138, s35
     16158| 	v_mul_lo_u32 v145, v144, s39
