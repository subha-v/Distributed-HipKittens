; <256,256,32,true>  symbol _Z21gemm_rs_mi300x_kernelILi256ELi256ELi32ELb1EEv14mi300x_globals
; k-loop blocks:
;   .LBB4_86 16012..16135
;   %bb.87 16136..16242
;   .LBB4_88 16243..16245
;   %bb.89 16246..16363
;   .LBB4_90 16364..16435
;   %bb.91 16436..16438
     15972| 	v_mov_b32_e32 v90, v2
     15973| 	v_mov_b32_e32 v91, v2
     15974| 	v_mov_b32_e32 v92, v2
     15975| 	v_mov_b32_e32 v93, v2
     15976| 	v_mov_b32_e32 v94, v2
     15977| 	v_mov_b32_e32 v95, v2
     15978| 	v_mov_b32_e32 v96, v2
     15979| 	v_mov_b32_e32 v97, v2
     15980| 	v_mov_b32_e32 v98, v2
     15981| 	v_mov_b32_e32 v99, v2
     15982| 	v_mov_b32_e32 v100, v2
     15983| 	v_mov_b32_e32 v101, v2
     15984| 	v_mov_b32_e32 v102, v2
     15985| 	v_mov_b32_e32 v103, v2
     15986| 	v_mov_b32_e32 v104, v2
     15987| 	v_mov_b32_e32 v105, v2
     15988| 	v_mov_b32_e32 v106, v2
     15989| 	v_mov_b32_e32 v107, v2
     15990| 	v_mov_b32_e32 v108, v2
     15991| 	v_mov_b32_e32 v109, v2
     15992| 	v_mov_b32_e32 v110, v2
     15993| 	v_mov_b32_e32 v111, v2
     15994| 	v_mov_b32_e32 v112, v2
     15995| 	v_mov_b32_e32 v113, v2
     15996| 	v_mov_b32_e32 v114, v2
     15997| 	v_mov_b32_e32 v115, v2
     15998| 	v_mov_b32_e32 v116, v2
     15999| 	v_mov_b32_e32 v117, v2
     16000| 	v_mov_b32_e32 v118, v2
     16001| 	v_mov_b32_e32 v119, v2
     16002| 	v_mov_b32_e32 v120, v2
     16003| 	v_mov_b32_e32 v121, v2
     16004| 	v_mov_b32_e32 v122, v2
     16005| 	v_mov_b32_e32 v123, v2
     16006| 	v_mov_b32_e32 v124, v2
     16007| 	v_mov_b32_e32 v125, v2
     16008| 	v_mov_b32_e32 v126, v2
     16009| 	v_mov_b32_e32 v127, v2
     16010| 	v_mov_b32_e32 v128, v2
     16011| 	v_mov_b32_e32 v129, v2
>>   16012| .LBB4_86:                               ; %.lr.ph834
>>   16013|                                         ;   Parent Loop BB4_84 Depth=1
>>   16014|                                         ; =>  This Inner Loop Header: Depth=2
>>   16015| 	s_lshl_b32 s1, s0, 14
>>   16016| 	s_and_b32 s1, s1, 0x4000
>>   16017| 	s_add_i32 s28, s65, s1
>>   16018| 	v_add_u32_e32 v138, s28, v202
>>   16019| 	v_add_u32_e32 v142, v138, v203
>>   16020| 	v_lshrrev_b32_e32 v143, 4, v142
>>   16021| 	v_and_b32_e32 v143, 56, v143
>>   16022| 	v_xor_b32_e32 v142, v143, v142
>>   16023| 	;;#ASMSTART
>>   16024| 	ds_read_b64 v[184:185], v142 offset:0
>>   16025| 
>>   16026| 	;;#ASMEND
>>   16027| 	;;#ASMSTART
>>   16028| 	ds_read_b64 v[180:181], v142 offset:0x400
>>   16029| 
>>   16030| 	;;#ASMEND
>>   16031| 	;;#ASMSTART
>>   16032| 	ds_read_b64 v[176:177], v142 offset:0x800
>>   16033| 
>>   16034| 	;;#ASMEND
>>   16035| 	;;#ASMSTART
>>   16036| 	ds_read_b64 v[168:169], v142 offset:0xc00
>>   16037| 
>>   16038| 	;;#ASMEND
>>   16039| 	;;#ASMSTART
>>   16040| 	ds_read_b64 v[156:157], v142 offset:0x1000
>>   16041| 
>>   16042| 	;;#ASMEND
>>   16043| 	;;#ASMSTART
>>   16044| 	ds_read_b64 v[152:153], v142 offset:0x1400
>>   16045| 
>>   16046| 	;;#ASMEND
>>   16047| 	v_add_u32_e32 v138, v138, v234
>>   16048| 	;;#ASMSTART
>>   16049| 	ds_read_b64 v[150:151], v142 offset:0x1800
>>   16050| 
>>   16051| 	;;#ASMEND
>>   16052| 	v_lshrrev_b32_e32 v144, 4, v138
>>   16053| 	;;#ASMSTART
>>   16054| 	ds_read_b64 v[142:143], v142 offset:0x1c00
>>   16055| 
>>   16056| 	;;#ASMEND
>>   16057| 	v_and_b32_e32 v144, 56, v144
>>   16058| 	v_xor_b32_e32 v138, v144, v138
>>   16059| 	;;#ASMSTART
>>   16060| 	ds_read_b64 v[188:189], v138 offset:0
>>   16061| 
>>   16062| 	;;#ASMEND
>>   16063| 	;;#ASMSTART
>>   16064| 	ds_read_b64 v[186:187], v138 offset:0x400
>>   16065| 
>>   16066| 	;;#ASMEND
>>   16067| 	;;#ASMSTART
>>   16068| 	ds_read_b64 v[182:183], v138 offset:0x800
>>   16069| 
>>   16070| 	;;#ASMEND
>>   16071| 	;;#ASMSTART
>>   16072| 	ds_read_b64 v[178:179], v138 offset:0xc00
>>   16073| 
>>   16074| 	;;#ASMEND
>>   16075| 	;;#ASMSTART
>>   16076| 	ds_read_b64 v[172:173], v138 offset:0x1000
>>   16077| 
>>   16078| 	;;#ASMEND
>>   16079| 	;;#ASMSTART
>>   16080| 	ds_read_b64 v[162:163], v138 offset:0x1400
>>   16081| 
>>   16082| 	;;#ASMEND
>>   16083| 	;;#ASMSTART
>>   16084| 	ds_read_b64 v[154:155], v138 offset:0x1800
>>   16085| 
>>   16086| 	;;#ASMEND
>>   16087| 	s_add_i32 s1, s69, s1
>>   16088| 	;;#ASMSTART
>>   16089| 	ds_read_b64 v[144:145], v138 offset:0x1c00
>>   16090| 
>>   16091| 	;;#ASMEND
>>   16092| 	v_add_u32_e32 v138, s1, v204
>>   16093| 	v_add_u32_e32 v146, v138, v203
>>   16094| 	v_lshrrev_b32_e32 v147, 4, v146
>>   16095| 	v_and_b32_e32 v147, 56, v147
>>   16096| 	v_xor_b32_e32 v146, v147, v146
>>   16097| 	;;#ASMSTART
>>   16098| 	ds_read_b64 v[164:165], v146 offset:0
>>   16099| 
>>   16100| 	;;#ASMEND
>>   16101| 	;;#ASMSTART
>>   16102| 	ds_read_b64 v[160:161], v146 offset:0x400
>>   16103| 
>>   16104| 	;;#ASMEND
>>   16105| 	v_add_u32_e32 v138, v138, v234
>>   16106| 	;;#ASMSTART
>>   16107| 	ds_read_b64 v[158:159], v146 offset:0x800
>>   16108| 
>>   16109| 	;;#ASMEND
>>   16110| 	v_lshrrev_b32_e32 v148, 4, v138
>>   16111| 	;;#ASMSTART
>>   16112| 	ds_read_b64 v[146:147], v146 offset:0xc00
>>   16113| 
>>   16114| 	;;#ASMEND
>>   16115| 	v_and_b32_e32 v148, 56, v148
>>   16116| 	v_xor_b32_e32 v138, v148, v138
>>   16117| 	;;#ASMSTART
>>   16118| 	ds_read_b64 v[174:175], v138 offset:0
>>   16119| 
>>   16120| 	;;#ASMEND
>>   16121| 	;;#ASMSTART
>>   16122| 	ds_read_b64 v[170:171], v138 offset:0x400
>>   16123| 
>>   16124| 	;;#ASMEND
>>   16125| 	;;#ASMSTART
>>   16126| 	ds_read_b64 v[166:167], v138 offset:0x800
>>   16127| 
>>   16128| 	;;#ASMEND
>>   16129| 	;;#ASMSTART
>>   16130| 	ds_read_b64 v[148:149], v138 offset:0xc00
>>   16131| 
>>   16132| 	;;#ASMEND
>>   16133| 	s_add_i32 s50, s0, 1
>>   16134| 	s_cmp_ge_i32 s50, s37
>>   16135| 	s_cbranch_scc1 .LBB4_88
>>   16136| ; %bb.87:                               ;   in Loop: Header=BB4_86 Depth=2
>>   16137| 	s_lshl_b32 s1, s50, 14
>>   16138| 	s_and_b32 s1, s1, 0x4000
>>   16139| 	s_add_i32 s28, s65, s1
>>   16140| 	v_add_u32_e32 v138, s28, v191
>>   16141| 	v_add_u32_e32 v245, v138, v192
>>   16142| 	v_lshrrev_b32_e32 v193, 4, v245
>>   16143| 	v_lshl_add_u64 v[246:247], v[130:131], 1, s[92:93]
>>   16144| 	s_or_b32 s29, s28, 8
>>   16145| 	v_and_b32_e32 v193, 56, v193
>>   16146| 	;;#ASMSTART
>>   16147| 	global_load_dwordx4 v[246:249], v[246:247], off
>>   16148| 
>>   16149| 	;;#ASMEND
>>   16150| 	v_lshl_add_u64 v[250:251], v[132:133], 1, s[92:93]
>>   16151| 	v_add_u32_e32 v254, s29, v191
>>   16152| 	v_xor_b32_e32 v193, v193, v245
>>   16153| 	;;#ASMSTART
>>   16154| 	global_load_dwordx4 v[250:253], v[250:251], off
>>   16155| 
>>   16156| 	;;#ASMEND
>>   16157| 	;;#ASMSTART
>>   16158| 	s_waitcnt vmcnt(0)
>>   16159| 	;;#ASMEND
>>   16160| 	;;#ASMSTART
>>   16161| 	ds_write_b64 v193, v[246:247]
>>   16162| 
>>   16163| 	;;#ASMEND
>>   16164| 	v_add_u32_e32 v193, v254, v192
>>   16165| 	v_lshrrev_b32_e32 v245, 4, v193
>>   16166| 	v_and_b32_e32 v245, 56, v245
>>   16167| 	v_xor_b32_e32 v193, v245, v193
>>   16168| 	v_add_u32_e32 v138, v138, v195
>>   16169| 	;;#ASMSTART
>>   16170| 	ds_write_b64 v193, v[248:249]
>>   16171| 
>>   16172| 	;;#ASMEND
>>   16173| 	v_lshrrev_b32_e32 v193, 4, v138
>>   16174| 	v_and_b32_e32 v193, 56, v193
>>   16175| 	v_xor_b32_e32 v138, v193, v138
>>   16176| 	;;#ASMSTART
>>   16177| 	ds_write_b64 v138, v[250:251]
>>   16178| 
>>   16179| 	;;#ASMEND
>>   16180| 	v_add_u32_e32 v138, v254, v195
>>   16181| 	v_lshrrev_b32_e32 v193, 4, v138
>>   16182| 	v_and_b32_e32 v193, 56, v193
>>   16183| 	v_xor_b32_e32 v138, v193, v138
>>   16184| 	s_add_i32 s1, s69, s1
>>   16185| 	;;#ASMSTART
>>   16186| 	ds_write_b64 v138, v[252:253]
>>   16187| 
>>   16188| 	;;#ASMEND
>>   16189| 	v_add_u32_e32 v138, s1, v191
>>   16190| 	v_add_u32_e32 v245, v138, v192
>>   16191| 	v_lshrrev_b32_e32 v254, 4, v245
>>   16192| 	v_lshl_add_u64 v[246:247], v[134:135], 1, s[30:31]
>>   16193| 	s_or_b32 s28, s1, 8
>>   16194| 	v_and_b32_e32 v254, 56, v254
>>   16195| 	;;#ASMSTART
>>   16196| 	s_waitcnt lgkmcnt(0)
>>   16197| 	;;#ASMEND
>>   16198| 	;;#ASMSTART
>>   16199| 	global_load_dwordx4 v[246:249], v[246:247], off
>>   16200| 
>>   16201| 	;;#ASMEND
>>   16202| 	v_lshl_add_u64 v[250:251], v[136:137], 1, s[30:31]
>>   16203| 	v_add_u32_e32 v193, s28, v191
>>   16204| 	v_xor_b32_e32 v245, v254, v245
>>   16205| 	;;#ASMSTART
>>   16206| 	global_load_dwordx4 v[250:253], v[250:251], off
>>   16207| 
>>   16208| 	;;#ASMEND
>>   16209| 	;;#ASMSTART
>>   16210| 	s_waitcnt vmcnt(0)
>>   16211| 	;;#ASMEND
>>   16212| 	;;#ASMSTART
>>   16213| 	ds_write_b64 v245, v[246:247]
>>   16214| 
>>   16215| 	;;#ASMEND
>>   16216| 	v_add_u32_e32 v245, v193, v192
>>   16217| 	v_lshrrev_b32_e32 v246, 4, v245
>>   16218| 	v_and_b32_e32 v246, 56, v246
>>   16219| 	v_xor_b32_e32 v245, v246, v245
>>   16220| 	v_add_u32_e32 v138, v138, v195
>>   16221| 	;;#ASMSTART
>>   16222| 	ds_write_b64 v245, v[248:249]
>>   16223| 
>>   16224| 	;;#ASMEND
>>   16225| 	v_lshrrev_b32_e32 v245, 4, v138
>>   16226| 	v_and_b32_e32 v245, 56, v245
>>   16227| 	v_xor_b32_e32 v138, v245, v138
>>   16228| 	;;#ASMSTART
>>   16229| 	ds_write_b64 v138, v[250:251]
>>   16230| 
>>   16231| 	;;#ASMEND
>>   16232| 	v_add_u32_e32 v138, v193, v195
>>   16233| 	v_lshrrev_b32_e32 v193, 4, v138
>>   16234| 	v_and_b32_e32 v193, 56, v193
>>   16235| 	v_xor_b32_e32 v138, v193, v138
>>   16236| 	;;#ASMSTART
>>   16237| 	ds_write_b64 v138, v[252:253]
>>   16238| 
>>   16239| 	;;#ASMEND
>>   16240| 	;;#ASMSTART
>>   16241| 	s_waitcnt lgkmcnt(0)
>>   16242| 	;;#ASMEND
>>   16243| .LBB4_88:                               ;   in Loop: Header=BB4_86 Depth=2
>>   16244| 	s_cmp_lg_u32 s84, s0
>>   16245| 	s_cbranch_scc1 .LBB4_90
>>   16246| ; %bb.89:                               ; %.preheader.1.i
>>   16247|                                         ;   in Loop: Header=BB4_86 Depth=2
>>   16248| 	v_and_b32_e32 v193, 0xffff0000, v185
>>   16249| 	s_or_b64 s[0:1], s[8:9], s[6:7]
>>   16250| 	v_cndmask_b32_e64 v193, v193, v185, s[6:7]
>>   16251| 	s_or_b64 s[0:1], s[0:1], s[4:5]
>>   16252| 	s_or_b64 s[28:29], s[18:19], s[12:13]
>>   16253| 	v_cndmask_b32_e64 v138, 0, v184, s[16:17]
>>   16254| 	v_and_b32_e32 v193, 0xffff, v193
>>   16255| 	s_mov_b64 vcc, s[0:1]
>>   16256| 	s_or_b64 s[28:29], s[28:29], s[10:11]
>>   16257| 	v_cndmask_b32_sdwa v184, v138, v184, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16258| 	v_cndmask_b32_e64 v185, v193, v185, s[8:9]
>>   16259| 	v_cndmask_b32_e64 v138, 0, v188, s[14:15]
>>   16260| 	v_and_b32_e32 v193, 0xffff0000, v189
>>   16261| 	s_mov_b64 vcc, s[28:29]
>>   16262| 	v_cndmask_b32_e64 v193, v193, v189, s[12:13]
>>   16263| 	v_cndmask_b32_sdwa v188, v138, v188, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16264| 	s_mov_b64 vcc, s[18:19]
>>   16265| 	v_cndmask_b32_sdwa v189, v193, v189, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16266| 	v_cndmask_b32_e64 v138, 0, v180, s[16:17]
>>   16267| 	v_and_b32_e32 v193, 0xffff0000, v181
>>   16268| 	s_mov_b64 vcc, s[0:1]
>>   16269| 	v_cndmask_b32_e64 v193, v193, v181, s[6:7]
>>   16270| 	v_cndmask_b32_sdwa v180, v138, v180, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16271| 	s_mov_b64 vcc, s[8:9]
>>   16272| 	v_cndmask_b32_sdwa v181, v193, v181, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16273| 	v_cndmask_b32_e64 v138, 0, v186, s[14:15]
>>   16274| 	v_and_b32_e32 v193, 0xffff0000, v187
>>   16275| 	s_mov_b64 vcc, s[28:29]
>>   16276| 	v_cndmask_b32_e64 v193, v193, v187, s[12:13]
>>   16277| 	v_cndmask_b32_sdwa v186, v138, v186, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16278| 	s_mov_b64 vcc, s[18:19]
>>   16279| 	v_cndmask_b32_sdwa v187, v193, v187, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16280| 	v_cndmask_b32_e64 v138, 0, v176, s[16:17]
>>   16281| 	v_and_b32_e32 v193, 0xffff0000, v177
>>   16282| 	s_mov_b64 vcc, s[0:1]
>>   16283| 	v_cndmask_b32_e64 v193, v193, v177, s[6:7]
>>   16284| 	v_cndmask_b32_sdwa v176, v138, v176, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16285| 	s_mov_b64 vcc, s[8:9]
>>   16286| 	v_cndmask_b32_sdwa v177, v193, v177, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16287| 	v_cndmask_b32_e64 v138, 0, v182, s[14:15]
>>   16288| 	v_and_b32_e32 v193, 0xffff0000, v183
>>   16289| 	s_mov_b64 vcc, s[28:29]
>>   16290| 	v_cndmask_b32_e64 v193, v193, v183, s[12:13]
>>   16291| 	v_cndmask_b32_sdwa v182, v138, v182, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16292| 	s_mov_b64 vcc, s[18:19]
>>   16293| 	v_cndmask_b32_sdwa v183, v193, v183, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16294| 	v_cndmask_b32_e64 v138, 0, v168, s[16:17]
>>   16295| 	v_and_b32_e32 v193, 0xffff0000, v169
>>   16296| 	s_mov_b64 vcc, s[0:1]
>>   16297| 	v_cndmask_b32_e64 v193, v193, v169, s[6:7]
>>   16298| 	v_cndmask_b32_sdwa v168, v138, v168, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16299| 	s_mov_b64 vcc, s[8:9]
>>   16300| 	v_cndmask_b32_sdwa v169, v193, v169, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16301| 	v_cndmask_b32_e64 v138, 0, v178, s[14:15]
>>   16302| 	v_and_b32_e32 v193, 0xffff0000, v179
>>   16303| 	s_mov_b64 vcc, s[28:29]
>>   16304| 	v_cndmask_b32_e64 v193, v193, v179, s[12:13]
>>   16305| 	v_cndmask_b32_sdwa v178, v138, v178, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16306| 	s_mov_b64 vcc, s[18:19]
>>   16307| 	v_cndmask_b32_sdwa v179, v193, v179, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16308| 	v_cndmask_b32_e64 v138, 0, v156, s[16:17]
>>   16309| 	v_and_b32_e32 v193, 0xffff0000, v157
>>   16310| 	s_mov_b64 vcc, s[0:1]
>>   16311| 	v_cndmask_b32_e64 v193, v193, v157, s[6:7]
>>   16312| 	v_cndmask_b32_sdwa v156, v138, v156, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16313| 	s_mov_b64 vcc, s[8:9]
>>   16314| 	v_cndmask_b32_sdwa v157, v193, v157, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16315| 	v_cndmask_b32_e64 v138, 0, v172, s[14:15]
>>   16316| 	v_and_b32_e32 v193, 0xffff0000, v173
>>   16317| 	s_mov_b64 vcc, s[28:29]
>>   16318| 	v_cndmask_b32_e64 v193, v193, v173, s[12:13]
>>   16319| 	v_cndmask_b32_sdwa v172, v138, v172, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16320| 	s_mov_b64 vcc, s[18:19]
>>   16321| 	v_cndmask_b32_sdwa v173, v193, v173, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16322| 	v_cndmask_b32_e64 v138, 0, v152, s[16:17]
>>   16323| 	v_and_b32_e32 v193, 0xffff0000, v153
>>   16324| 	s_mov_b64 vcc, s[0:1]
>>   16325| 	v_cndmask_b32_e64 v193, v193, v153, s[6:7]
>>   16326| 	v_cndmask_b32_sdwa v152, v138, v152, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16327| 	s_mov_b64 vcc, s[8:9]
>>   16328| 	v_cndmask_b32_sdwa v153, v193, v153, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16329| 	v_cndmask_b32_e64 v138, 0, v162, s[14:15]
>>   16330| 	v_and_b32_e32 v193, 0xffff0000, v163
>>   16331| 	s_mov_b64 vcc, s[28:29]
>>   16332| 	v_cndmask_b32_e64 v193, v193, v163, s[12:13]
>>   16333| 	v_cndmask_b32_sdwa v162, v138, v162, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16334| 	s_mov_b64 vcc, s[18:19]
>>   16335| 	v_cndmask_b32_sdwa v163, v193, v163, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16336| 	v_cndmask_b32_e64 v138, 0, v150, s[16:17]
>>   16337| 	v_and_b32_e32 v193, 0xffff0000, v151
>>   16338| 	s_mov_b64 vcc, s[0:1]
>>   16339| 	v_cndmask_b32_e64 v193, v193, v151, s[6:7]
>>   16340| 	v_cndmask_b32_sdwa v150, v138, v150, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16341| 	s_mov_b64 vcc, s[8:9]
>>   16342| 	v_cndmask_b32_sdwa v151, v193, v151, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16343| 	v_cndmask_b32_e64 v138, 0, v154, s[14:15]
>>   16344| 	v_and_b32_e32 v193, 0xffff0000, v155
>>   16345| 	s_mov_b64 vcc, s[28:29]
>>   16346| 	v_cndmask_b32_e64 v193, v193, v155, s[12:13]
>>   16347| 	v_cndmask_b32_sdwa v154, v138, v154, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16348| 	s_mov_b64 vcc, s[18:19]
>>   16349| 	v_cndmask_b32_sdwa v155, v193, v155, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16350| 	v_cndmask_b32_e64 v138, 0, v142, s[16:17]
>>   16351| 	v_and_b32_e32 v193, 0xffff0000, v143
>>   16352| 	s_mov_b64 vcc, s[0:1]
>>   16353| 	v_cndmask_b32_e64 v193, v193, v143, s[6:7]
>>   16354| 	v_cndmask_b32_sdwa v142, v138, v142, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16355| 	s_mov_b64 vcc, s[8:9]
>>   16356| 	v_cndmask_b32_sdwa v143, v193, v143, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16357| 	v_cndmask_b32_e64 v138, 0, v144, s[14:15]
>>   16358| 	v_and_b32_e32 v193, 0xffff0000, v145
>>   16359| 	s_mov_b64 vcc, s[28:29]
>>   16360| 	v_cndmask_b32_e64 v193, v193, v145, s[12:13]
>>   16361| 	v_cndmask_b32_sdwa v144, v138, v144, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16362| 	s_mov_b64 vcc, s[18:19]
>>   16363| 	v_cndmask_b32_sdwa v145, v193, v145, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
>>   16364| .LBB4_90:                               ;   in Loop: Header=BB4_86 Depth=2
>>   16365| 	v_mfma_f32_16x16x16_bf16 v[78:81], v[184:185], v[164:165], v[78:81]
>>   16366| 	s_add_u32 s30, s30, 64
>>   16367| 	s_addc_u32 s31, s31, 0
>>   16368| 	s_add_u32 s92, s92, 64
>>   16369| 	v_mfma_f32_16x16x16_bf16 v[82:85], v[184:185], v[160:161], v[82:85]
>>   16370| 	s_addc_u32 s93, s93, 0
>>   16371| 	s_cmp_eq_u32 s21, s50
>>   16372| 	v_mfma_f32_16x16x16_bf16 v[86:89], v[184:185], v[158:159], v[86:89]
>>   16373| 	s_barrier
>>   16374| 	v_mfma_f32_16x16x16_bf16 v[90:93], v[184:185], v[146:147], v[90:93]
>>   16375| 	v_mfma_f32_16x16x16_bf16 v[94:97], v[180:181], v[164:165], v[94:97]
>>   16376| 	v_mfma_f32_16x16x16_bf16 v[98:101], v[180:181], v[160:161], v[98:101]
>>   16377| 	v_mfma_f32_16x16x16_bf16 v[102:105], v[180:181], v[158:159], v[102:105]
>>   16378| 	v_mfma_f32_16x16x16_bf16 v[106:109], v[180:181], v[146:147], v[106:109]
>>   16379| 	v_mfma_f32_16x16x16_bf16 v[110:113], v[176:177], v[164:165], v[110:113]
>>   16380| 	v_mfma_f32_16x16x16_bf16 v[114:117], v[176:177], v[160:161], v[114:117]
>>   16381| 	v_mfma_f32_16x16x16_bf16 v[118:121], v[176:177], v[158:159], v[118:121]
>>   16382| 	v_mfma_f32_16x16x16_bf16 v[122:125], v[176:177], v[146:147], v[122:125]
>>   16383| 	v_mfma_f32_16x16x16_bf16 v[126:129], v[168:169], v[164:165], v[126:129]
>>   16384| 	v_mfma_f32_16x16x16_bf16 v[74:77], v[168:169], v[160:161], v[74:77]
>>   16385| 	v_mfma_f32_16x16x16_bf16 v[70:73], v[168:169], v[158:159], v[70:73]
>>   16386| 	v_mfma_f32_16x16x16_bf16 v[66:69], v[168:169], v[146:147], v[66:69]
>>   16387| 	v_mfma_f32_16x16x16_bf16 v[62:65], v[156:157], v[164:165], v[62:65]
>>   16388| 	v_mfma_f32_16x16x16_bf16 v[58:61], v[156:157], v[160:161], v[58:61]
>>   16389| 	v_mfma_f32_16x16x16_bf16 v[54:57], v[156:157], v[158:159], v[54:57]
>>   16390| 	v_mfma_f32_16x16x16_bf16 v[50:53], v[156:157], v[146:147], v[50:53]
>>   16391| 	v_mfma_f32_16x16x16_bf16 v[46:49], v[152:153], v[164:165], v[46:49]
>>   16392| 	v_mfma_f32_16x16x16_bf16 v[42:45], v[152:153], v[160:161], v[42:45]
>>   16393| 	v_mfma_f32_16x16x16_bf16 v[38:41], v[152:153], v[158:159], v[38:41]
>>   16394| 	v_mfma_f32_16x16x16_bf16 v[34:37], v[152:153], v[146:147], v[34:37]
>>   16395| 	v_mfma_f32_16x16x16_bf16 v[30:33], v[150:151], v[164:165], v[30:33]
>>   16396| 	v_mfma_f32_16x16x16_bf16 v[26:29], v[150:151], v[160:161], v[26:29]
>>   16397| 	v_mfma_f32_16x16x16_bf16 v[22:25], v[150:151], v[158:159], v[22:25]
>>   16398| 	v_mfma_f32_16x16x16_bf16 v[18:21], v[150:151], v[146:147], v[18:21]
>>   16399| 	v_mfma_f32_16x16x16_bf16 v[14:17], v[142:143], v[164:165], v[14:17]
>>   16400| 	v_mfma_f32_16x16x16_bf16 v[10:13], v[142:143], v[160:161], v[10:13]
>>   16401| 	v_mfma_f32_16x16x16_bf16 v[6:9], v[142:143], v[158:159], v[6:9]
>>   16402| 	v_mfma_f32_16x16x16_bf16 v[2:5], v[142:143], v[146:147], v[2:5]
>>   16403| 	v_mfma_f32_16x16x16_bf16 v[78:81], v[188:189], v[174:175], v[78:81]
>>   16404| 	v_mfma_f32_16x16x16_bf16 v[82:85], v[188:189], v[170:171], v[82:85]
>>   16405| 	v_mfma_f32_16x16x16_bf16 v[86:89], v[188:189], v[166:167], v[86:89]
>>   16406| 	v_mfma_f32_16x16x16_bf16 v[90:93], v[188:189], v[148:149], v[90:93]
>>   16407| 	v_mfma_f32_16x16x16_bf16 v[94:97], v[186:187], v[174:175], v[94:97]
>>   16408| 	v_mfma_f32_16x16x16_bf16 v[98:101], v[186:187], v[170:171], v[98:101]
>>   16409| 	v_mfma_f32_16x16x16_bf16 v[102:105], v[186:187], v[166:167], v[102:105]
>>   16410| 	v_mfma_f32_16x16x16_bf16 v[106:109], v[186:187], v[148:149], v[106:109]
>>   16411| 	v_mfma_f32_16x16x16_bf16 v[110:113], v[182:183], v[174:175], v[110:113]
>>   16412| 	v_mfma_f32_16x16x16_bf16 v[114:117], v[182:183], v[170:171], v[114:117]
>>   16413| 	v_mfma_f32_16x16x16_bf16 v[118:121], v[182:183], v[166:167], v[118:121]
>>   16414| 	v_mfma_f32_16x16x16_bf16 v[122:125], v[182:183], v[148:149], v[122:125]
>>   16415| 	v_mfma_f32_16x16x16_bf16 v[126:129], v[178:179], v[174:175], v[126:129]
>>   16416| 	v_mfma_f32_16x16x16_bf16 v[74:77], v[178:179], v[170:171], v[74:77]
>>   16417| 	v_mfma_f32_16x16x16_bf16 v[70:73], v[178:179], v[166:167], v[70:73]
>>   16418| 	v_mfma_f32_16x16x16_bf16 v[66:69], v[178:179], v[148:149], v[66:69]
>>   16419| 	v_mfma_f32_16x16x16_bf16 v[62:65], v[172:173], v[174:175], v[62:65]
>>   16420| 	v_mfma_f32_16x16x16_bf16 v[58:61], v[172:173], v[170:171], v[58:61]
>>   16421| 	v_mfma_f32_16x16x16_bf16 v[54:57], v[172:173], v[166:167], v[54:57]
>>   16422| 	v_mfma_f32_16x16x16_bf16 v[50:53], v[172:173], v[148:149], v[50:53]
>>   16423| 	v_mfma_f32_16x16x16_bf16 v[46:49], v[162:163], v[174:175], v[46:49]
>>   16424| 	v_mfma_f32_16x16x16_bf16 v[42:45], v[162:163], v[170:171], v[42:45]
>>   16425| 	v_mfma_f32_16x16x16_bf16 v[38:41], v[162:163], v[166:167], v[38:41]
>>   16426| 	v_mfma_f32_16x16x16_bf16 v[34:37], v[162:163], v[148:149], v[34:37]
>>   16427| 	v_mfma_f32_16x16x16_bf16 v[30:33], v[154:155], v[174:175], v[30:33]
>>   16428| 	v_mfma_f32_16x16x16_bf16 v[26:29], v[154:155], v[170:171], v[26:29]
>>   16429| 	v_mfma_f32_16x16x16_bf16 v[22:25], v[154:155], v[166:167], v[22:25]
>>   16430| 	v_mfma_f32_16x16x16_bf16 v[18:21], v[154:155], v[148:149], v[18:21]
>>   16431| 	v_mfma_f32_16x16x16_bf16 v[14:17], v[144:145], v[174:175], v[14:17]
>>   16432| 	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[170:171], v[10:13]
>>   16433| 	v_mfma_f32_16x16x16_bf16 v[6:9], v[144:145], v[166:167], v[6:9]
>>   16434| 	v_mfma_f32_16x16x16_bf16 v[2:5], v[144:145], v[148:149], v[2:5]
>>   16435| 	s_cbranch_scc1 .LBB4_92
>>   16436| ; %bb.91:                               ;   in Loop: Header=BB4_86 Depth=2
>>   16437| 	s_mov_b32 s0, s50
>>   16438| 	s_branch .LBB4_86
     16439| .LBB4_92:                               ; %Flow2205
     16440|                                         ;   in Loop: Header=BB4_84 Depth=1
     16441| 	s_mov_b64 s[0:1], exec
     16442| 	v_readlane_b32 s28, v255, 39
     16443| 	v_readlane_b32 s29, v255, 40
     16444| 	s_and_b64 s[28:29], s[0:1], s[28:29]
     16445| 	s_mov_b64 exec, s[28:29]
     16446| 	s_cbranch_execz .LBB4_101
     16447| ; %bb.93:                               ;   in Loop: Header=BB4_84 Depth=1
     16448| 	v_readlane_b32 s28, v255, 35
     16449| 	v_readlane_b32 s29, v255, 36
     16450| 	s_and_b64 exec, exec, s[28:29]
     16451| 	s_cbranch_execz .LBB4_101
     16452| ; %bb.94:                               ;   in Loop: Header=BB4_84 Depth=1
     16453| 	v_mul_lo_u32 v138, s42, v0
     16454| 	v_add_u32_e32 v138, s64, v138
     16455| 	v_sub_u32_e32 v143, 0, v138
     16456| 	v_max_i32_e32 v143, v138, v143
     16457| 	v_mul_hi_u32 v144, v143, s70
     16458| 	v_mul_lo_u32 v145, v144, s35
     16459| 	v_sub_u32_e32 v143, v143, v145
     16460| 	v_add_u32_e32 v145, 1, v144
     16461| 	v_cmp_le_u32_e32 vcc, s35, v143
     16462| 	v_ashrrev_i32_e32 v142, 31, v138
     16463| 	v_xor_b32_e32 v142, s22, v142
     16464| 	v_cndmask_b32_e32 v144, v144, v145, vcc
     16465| 	v_subrev_u32_e32 v145, s35, v143
     16466| 	v_cndmask_b32_e32 v143, v143, v145, vcc
     16467| 	v_add_u32_e32 v145, 1, v144
     16468| 	v_cmp_le_u32_e32 vcc, s35, v143
     16469| 	s_nop 1
     16470| 	v_cndmask_b32_e32 v143, v144, v145, vcc
     16471| 	v_xor_b32_e32 v143, v143, v142
     16472| 	v_sub_u32_e32 v142, v143, v142
     16473| 	v_mul_lo_u32 v143, v142, s33
     16474| 	v_sub_u32_e32 v138, v138, v143
     16475| 	v_sub_u32_e32 v144, 0, v138
     16476| 	v_ashrrev_i32_e32 v143, 31, v138
     16477| 	v_max_i32_e32 v138, v138, v144
     16478| 	v_mul_hi_u32 v144, v138, s34
