#!/usr/bin/env python3
from torch.utils import cpp_extension
import os

sources = [
    "module.cpp",
    "pingpong.cpp",
    "fenced_write.cpp",
    "test_kernel.cpp",
    "peer_manager.cpp",
    "torch.cpp",
    "all2all.cpp",
]

def compile_module(sources=None, cpp_sources=None, verbose=False):
    if verbose:
        print("compile_module()")

    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    include_dirs = [
        f"{rocm_path}/include",
    ]
    library_dirs = [
        f"{rocm_path}/lib",
    ]
    libraries = [
        "amdhip64",
        "hsa-runtime64",
    ]
    extra_cflags = [
        "-O2",
        "-x", "hip",
        "-march=native",
        # "-fgpu-rdc",
        "--offload-arch=native",
        # "--offload-arch=gfx942:xnack-",
        "-fdiagnostics-color=always",
        "--save-temps",
        "-Wall",
        # "-Wextra",
    ]
    extra_ldflags = [
        # "-fgpu-rdc", 
        "--hip-link",
    ]

    for lib_dir in library_dirs:
        extra_ldflags.append(f"-L{lib_dir}")

    for lib in libraries:
        extra_ldflags.append(f"-l{lib}")

    os.environ["CXX"] = "amdclang++"
    os.environ["CC"] = "amdclang"
    # os.environ["CLICOLOR_FORCE"] = "1"
    # cpp_extension._check_and_build_extension_h_precompiler_headers(
    #     extra_cflags=extra_cflags,
    #     extra_include_paths=include_dirs,
    # )
    kwargs = dict(
        name="all2all_module",
        extra_include_paths=include_dirs,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=verbose,
    )
    
    if sources is not None:
        return cpp_extension.load(**kwargs,
            sources=[f"{current_dir}/{x}" for x in sources],
            build_directory=os.environ.get("BUILD_DIR", None),
        )
    elif cpp_sources is not None:
        return cpp_extension.load_inline(**kwargs,
            cpp_sources=cpp_sources,
        )
    else:
        raise ValueError("Either sources or cpp_sources must be provided.")


import base64, gzip
cpp_sources = """ABzY8YKiS=0{`tj?Q$DAj{o%(Ty5FYays%ya;cPTtIA_L6W1k^RN~p0t8A*PEm2laB(=7?l}O2Y>K@{r@SfxV@C)n@*-n(Xx~rYqjU|!*0T2K|5~NzYyW-(D{?`H=<7D0)w>E(ck^D6o&4zL$qUia>>*)21lgnqX-n@)XUY<n})k9@6leP6?pIXS-Fd1=Nzq~z<-n=?}$>B{|6AWPg@x{s8=x+xcs#IEmx_tHK^m%mh`Xajc`SisvXXnxRZ<pskzrA?%bM%}ey1k*o{nunRm@I};9D$g3;r~&#m=(!XcE^2^*C<P;B1;FqzI=JIo5$1nMCQALbUICE-SK?hV*Es>&(BZ)y~TJ@C4pSdGnwZ}I_rqq|Nh>>f0I!VFN##;MO-X$aV$9Cp(*~$F>1uy#b5w69S!<Le8qnX6#pB|vt(9`g1i{YEDP@gYW`~jQ38O|tS5fh%6|y9h8+<XVk_@$4gWrChs~e>`I9Wp5%z5Xgj@%T)lPV8ud3l!5gbL&E?%5RQTR2WFD`xtP#BcEg34u51n0k9Tt?4ME?)fd=KMBT1$xucdfF-^s9s#iqhd9evv>+B-f!hawipy5CS6FgsMr@@0Pb=}4C5jO4Vo<`lX;OHLLm&WSVn^s)Eg@Hun27Y)!WPH?EG&Rr@#)!6KEVD9tPOyQDHzeFg&_AFxBGAO%2-1ZIRAp78hw2bOJ<|bcjwNGFdFL86ka_yhoaDTJTSve3nt6{#>mJnFk#)NN0KBgJFF-3NAZh$B<%C0R6ytJsfst8&(B-8t26&$Or^1#K$xl3NnuX^<_tlLNOS}nZgU5WMJ(}zigCd)3`7^!Z)F~Cb`3(BTKDF9y~DZoEbtF?XCRX*6_VJ2E#F3B%Q-*L)BL(0;0OZ1>LD@-v(RM<u@dPo5Q;**TKD%3yW5F)whFseG563+mtUf4TIHv)lv2Hd#zT2ZW%Tnh~J%#a=D}*!Sf|U85x=o-AsOptYOq>;TOao@@AHUn+;fkqx=7tO@zIwRp_?KNSLt~N5|r+!@_#}UG?}o^l0QGe&;>j-M~Un8Ea!&bo4>UNiJ{c{PE_xtpRU~ZnxWJtvnKpkRB2h-$N25iEPFJJ^EG$ghik4>XRPb!6$_r<^3I0Xh4I{U+J>cpAU7OD=4`x^bMcw*ALE<L6pm7Y@OMv<jhu8X9l2Ue`dwRKrEl+*Z^o_zRmIUf7{F5!#h<yVvSQogS388dB)ETo{{zbLG}ItdN)!PAIcu#9zG!i&17zYIfNP~Sx?8~jZ;vIrQ$D3)O_Y1^LxstMpE_8(==92REYHo9SU3a?Aw$(6wkJ|K{<_^$nUT1%7GdLG#^=PL2Z=V{;Ln5mz1u|0o7WPY4$H;`j;`?ZcG@FY-satDyM0-+SLA?+pUVTEME0l0dCza2XbEU!X$+lO2$)OHppp@rQd1|CUKsNGx;$Y$d^PXh<?7fnk0iB$bnbDYzo0m1oLVX=%@>V7n-?)JqlXXygo~34+rUd6<<wc8HJjTdC_(Ulwl|*vXF<R)-K>Vz8$cI`zv!GJMo(ZEW$tr6us)3QnGO}r+p`A1P(&b*4oQ>GD!!)wqjRvsr3X+9~+i7;f%%><1GCoU?_S$y^R>rmRQiXc*Q7ACUV#VBYB`&s__V_yI=s?>9A0lBupS+E#P{NMEe9$g2x{MI`~ecQiY&^3OC?}BoHxALG5MMQw<ew1cLbFY&fxD(I2IO|6(~;P>$U?={?J&Gz|~gx@?Wqgk}nxAnh6plN6NJi>Jattuw5M4bcycWR!`bsO}`Ava>Lg>0lybj5@#QPg-qm{CZC#*XW>}=CTNk3hpT_+=LPm;%`tw9cnB<Qb=2&R!CmJZG!Dgk~m=+EusN<>>~@*w;Cdhd`dyP4cMujYQ|(&ZAg`>T?<_NM=*+>MXj-v)*OfA=#>||m=8cS&tn*5>3KoW9qU_oBkh$yV7xRcw#Aq}!D3K-hyq=pmUi{Xm^hKtb*ceyFhmY?TXz#y<xR|MS$FQHJQYhUb@$aqxmGLdJMp&()L6WgSrSiR^hA+rABUmE>s2vMXCPdNS7+Ba^-xbVOCEt#o;;3<@M#}_UL<)@S{v3k&1Alq6o)Ny%#aKsP!yRJfnJbSh8`xRW3yUS&_?LsVg~b-F6`1K0ZgB6<Qg$*cxzCGByEL?E?;@^z>1huEY>o&E#k~7nFr0e)>Ln6vMsd$YkBwz%Hf|z+qss7R_ls^>+aFSxq*9Tq{4zxtoy2TV39D(e=TKJ6aBd2GS=Eihc#DpyMj`%!YV2;P6H5k8<mbYht~P5_aCodCCD6znPn7gmyJQ_MED~9ygHvDw@AT>avR@1AC;-uY<PsIVWPiyeadF}`v*jxF5E910{%`XX)c|(zd1Ilp!|-35_R+~0<4La{`}CPDInlm$QxYq%|TtFX*`RsWw!a1pBe|gqWdTZ;6v<!?(I0v<WTABkKn5l3G7wZdWRMN2NKX*?y8UII!hPxNKKPg(C?nU2f2N@srXREaMg*b@_;n{k4g!}#xOldwa=Wzvkz)d`%{`thLH_>L5i+vG_9`hEV3)bbD*|t6y2I3)Dy%_BZOUXbGC&zjq^%7iv@(NqYXe4@vu+Y$Nf;LxzY!0oXGs0?dCm$)bn{s?)<CuJZst<HtxeYVIGPH4-&^22k+wOCOEoWTy>n4J0-imO4A95=4x?WV~bc^u|@I}i&^rIg~Yy&xO(<|DIeA}uTk{A8U4rtDC;;X13R_v_x%wv2SrY1{P;*Ii2Ck0MprV5!I^H<k!W5v??duczB>gE)gAY(<A?9_P)rvy_{*A(2D4&f+vz!CJdGf9nT6t2bo%D}<Ptx;Is5eug9A~6$QdB!Qh|4w;dVs`t6`W}So7u6Cm>yzt_%b;kD_c?<O9CzRix8oaO<|$03wcA+PCgZE5{k08lvLE0^)iW=fRW59UUJNUJQ`?f_5S$9V)JEcMYKh1aC<?<DRfaC($E1ooBS{SY<WGQm9}k+|jVAglzRX*B%6>#(4}u;fmF{BIbu{s0#SbKAsDdy>8gpTc=09z%>n+uMVlM0uxlRsFt>YUP&<dbRYy|Mz~W$382txIFcGzq0!Ai@uU@0{8MWveK`N^^hGH4o;-QNB3O+9;;@JN&Od@0p5RF);6)LdC@8)99oz~?EGb|)I$1&NoQuygOT|Rau8Z-1|L1=dW9e`Z1L+82VQF9o@l0Gvfx9t_9Ktkb9l=4TE5s#-2MVB**zWeJl`kNS>j{G*2aHLDQ(Pz#rJ80SXozd*y)Fwx9SKSrp&cGt+j7CIi|e0JJc`(nJHxw2DBpN*)x@IB<B6l_4W2iAUqvP2#|*u^z;S4#lTphJ7l~B+0~D!p1MdfBngypC>^X6U;`N(Xml3Gu8E7PZ{`Jkp<#{Mxp1h9UK0kSLeijNd2zwnu!<5zU|CpdKvGDxtF6%DGIgnM)ujUi92lP;kpgl}<`5;T?Fqw)ZhxKnZC4E}p0wI}+d~uaam|bV6+LK8R=v!sK51I{11_)xU(SuBqJ}b5D2w2Z3S&<2xjKC3#*q}kQM6_x^##E+k8n4i*25Q~%DJjOVlrae6(Fl6MF-A~rF@S6iMam2aPV166^7G?KD-*SL2luaZdm1fflVtV*VjqwWM7WdmI$fX?gswQ~^h9E5$)NX-xI<rWz)IUe9HB^Ou-HP@=UIx2LjW4bAEiM=6ZRa3)B2VWBw~C;PBH}v6<NB<sS!B+0){Eq4#`R<&(lFd!=Y)(<Tdc;1_H*ImZzh|#N#-$C<;7+E{JPuxHzhYEgTS{A;*(wI*!K0s|WxUAU1f6*Zu@vHJ{5tGD_qSCn)i0oD9aG3E(P7hk#WwTfl96vc;m6tJAeJgeDmK<ze?9b@QW*w@JXB#WA;d=$!#vm0)p3#bKC45G{b8;7A_!T?Emcr18+KArQ!<DER<7H0h3HF&Ibja2V|49JzmJIM!#IZ4Cwk7!SrUo?F9*`o!E)6RJLRpVqPHj*t2Bi%*6xz{MFw6c5#E%!Ev&X)qL^+UNxwdm6l+nH<bk27A`eHp+-|eONnb0DZs4NkJ<h2Z+jOw}si1h+aD@$a~|K2gQuN-Z;4)N5ED(Ss-u(9%1mI*PErE0!!jT@1zQdP61xWsjeQB6ueS!%V?vN?QrlZ3&+9Y1R-NDeX+j>5zxU8`+Ivo?5Y1RCV1Ke+(1<>!K#OJ;CIm7|LOT>b`IV=5ts<|6G{y$z*)9y^Z0zp$=UF0<Yu>*G|X_J1adS?Ayk!p7|nQ&e*2huc&J8*K^y(%bfbDt7Gi5e`dyM+Xi#=lDqE*~q@cD3u4SPD%c})U5t-eID5cdk9kAG4g62&tt_x_h_0cXZE^FdkH~nukIRWN~1Ni^u*Ovnr<|OUc+^!kT_n*+!8uY<Aa3>8FAZoK-OJ?&$q4%B2YwqTi`D`(b<Pv08<h)oHMJUuF{Xl-$R*sY5P|l)ZGUcVyco`u!YMz(-sFd!&Kf?q(BHK?yZ4F%Ouu2BTbR+Nt0RBmW@f7z5SIWCw2j>{<wEVO${|wAn7KwUmxVts%F=mgcgi9Dosx)Dt>+o1jIbJf4+TP=fOirvX_mKBjBLR;&I}fk9W1!aUdaR<4iTqTbB>F`+@rkSSe`6wubzRa3_ss~e>r^+{M5j?SjwhqOSgz|YP_1@4OTSAeRC5afJ&sCYNT*<!GSU_su+7|UZ`V82J!(VVRv$vGurSHMH=M(7Xl+~e1e;5~N{5Sy><;Gh4PKvIb6?O^&R18UMEm=@<^xuH;I>6h+iQL$ghYe!u3SQxj+g%$>gROm-B5|?p1Dty%LZfKcfJDmVTdsU4Ln7-D4M6U>*#~bW^zJv;*N?})GVnsjCOD#13JOWU{Y2Sd#=9IG#-77Cotv}{4Dpu$L`AS(1n|=eL8`;%Cw9wyC3k*I>x^<Zki!IatO3eUKW4I=np-@iRi-?n3&#bLp&o1>%C$0DT9^<7YVH_X$^CQfmUw6La-;}q5JOogPt0?2^vLx?>_8V(QTsCp;L`4b*$AO3NA}SmZisI_J%1ltb+@#$O+d(tmM}cD*zlBEQ#}iyfIi=U<{|%Q@$n=+c+1sUjN6B7eAlv??*3Ro&ECSJW$t>-4*1YPhOtGe24w1vn_{+_K_q1EF_qQqRmU%9dr4jJCq|v2JY@1sbL|*=tIcsHEo+iR|uB3IZ6n}bxjEXu~i%EGq(onc~XL+o|oK(`YPc?8cjg<ND4>R>#3w3uXUt$<bpP4!!&@IM44tT*clP=;J&Ztpe8ZH4hD5r-;<g<Sy!r{t#<$FI?_Ti5~9^OGRmM~&4;BX408zL>{^9;?M_3b3)9-V8{nArq>tw6JJgkSYp8;q<5fuY8mWR$m)migY8sf+MHv1)G{aNrmb-n$d3`(2F|BLpe9l^QB<MO8xRWz%@Ujh%sQl!*7TSZ}l~2V7V*=f!As#=22IC*0JZM|BFZs@?a;uw4y3ixPxoEEyMc0$`3bZBC>!#8SPcz*FWHOPW8XZ+?{i#(494s6#OH2s+C{}8jXhscAl;FOI`g}25c9%F=y0d=Ve^bZ(`00otRz>b&h2EVq{osk$P9+!;ekAbD(tzm!7L@NjMv-bmx@-1)T+3r|p!YAncZ^AF!t&*jxXu!NK)Fx!P$3xuFD(&qbaZ6lLe{$9_hZJe5|%QGc*HB%OCYddP}At!)r@w*D3^JohQNKt28{u*;F1ELI<abvVdJ9%2mpc61<cupt=cBERcf0sL{S8_`v*_GO1%f9i(oLx9a%if!&1SD3UG~Dw%5Aw5v@{VzRe(=0VclTf!dQ=s4qL}%ULp|FU;=lWuIgrZoiS+9;M_Y`fvc-xegD%rg;?Qpnclq8%y=f7aD-0p1QgUz>S(GaI2b2<``J)*v+J8GysAIKu-l?n9iiRzE`IEPh1PtISB<_>nSuQs;ePtozLJ2<Z9)k^sln`DJ@^_<TmVbNzo4arz!O`5x(gCX5IHj9S&P6ZO-3R_mV5-U0u`Bpc>`l)my_3DO_^vvhco>2g9&V36ej59`LeIzr(mx^H6JZvG9Oid6Rx!c33GX&F)&YycK|#HA+K7=SU8BuQ!j2F^o|bv}dyTn9gNHiFxWY%jW1+z>+)yAW9Kj1@K7?f!FyT)!s3Jk3UF}FEUBV(JFO5$GjVpdvGPe`Ac^ak7)<qjWg6$o_E;QH^?Zn(B&K$eXTCBniFQ#wgNimznFt3@y~yPZ%FcoeRg{Cm%4M_Ip}a9=D^CVfCvqv&TCoB;ggvU=m@b8a-I&xMyOb3uZF5-lwpQDjqeWr?a6z6lV+yvQzbZPw?qf>ux8k;If(CW4e5WAVcS1Ba1@4~f1N-q*AIv|bKGZzpozu7i&QdbZ5FAfuAB@9l=4M=hvyI|+3J%j{Bvjvz2hpL(Dx^6lJHY%yGn`F+RSrJZ)KspQ$RbE9ewMCWm^LyFs3lC?C%{s)_>^m2})o7hzFSBr`Nxfd#n?`zt3eexl1Y5mZ6qTK#e_dmF@xQ(x8OIT{oaEcDT}T@Evzti^ul<(Xf&TUhbf_B@wsom41x|npCsR{jr=c@^N$(4?YB6ZaO4mOtD!}Ke?QY>JlMpgaO2S9mDJyu_nSQgUUC~#>`gi1^YUm<|ohaEH@(>N2I6K9`m;+JIvC8CqcG5c3rxufC#rYzQ>e%|5K$u@koi<^m><oA2buAD|1XZU%~XSnA1K8o(HNFs>Z`ZH~y}9S6}*t3G4A)G8``uQm9U+3%qL>f0RNl<3TZ5iN~NYl%tflxeG@+O>*w&8W&Nv|95*df3X&1B0Bw(wcCGRx+uWv4&HaoDN;Y^)Fv~q){jc9aXRrt?2-;FE#0^P_tw@m?T*6Wf$5%oR4xIbj7rzs9o4c1$+8*TgejWjF2rM)o|s%O(giFvJ8Q>S@n5KpX_&86cB2AUgG=Z&8}~u~tCq=K&@b<U__y^a>D~bu4sP<RL>T(3)xOJO47I!7Z3r-Qt)CYInyp8Lm7DQVx~Adk5@#HArM4D_mtW(6U2!c`CI7}j`c<{i9%NY+Z-I3-Wfal`4SqXkFh;X%JC*TaTcSNIU2R<Sde1RA1DqMo<-fqhMG*T>L|`gAG`TO)Gg%9<^dwW|2M;i!H>Knr_o$q7g$tqFY@ao&rWM^fjW(Fw!I?_#h0ts+Z_8bBZWmdzBJ@}@rMIx=HTy=^8chBhSX-8uTHeOavdRd{a=Mj+-|j&>!|P3hrHPzw?&AiiD#lALdGMu7dsrd%CKRnZK18ii4>7<9^p_9MlvQ-PDvC!@st+A~{6v|kDwl{74D8zqWkd6QYB`C?&s)-oN-jmI@`S-nIn<%RSBbs+jz<)(5dnLjLEll-TiX3?HC$^2sA+gRh$nG&34#+OJ~vQGfKMK;BLU+Ny-1>U{$7XO^);eglfA+2va5yJDW%*SB~0gP`w_hvEHHQ9ygHqlm8;T9mD@isg%@XbTs7#iua2)&abS*(Nn_|=dyq^zqJhr5Zy!#7IXej*!ySBs8{ZTmDprOFaF%je<_MpANI%W!?ud?=kbwc1Svu7ZHDq9zcs)Lqcs&Nz{<CYtj%jc#-n}>1g1?B}9dVYLrs5&^GTa{jp%Bm$?H!nl7&~;pK_qEy)GSeY!__3c3e0GQMt5|Vp%LXhlUPU*+#xShay;i{UmKDKNHrQ$3y8+XS{l%Hr&*WF?ahc1%wFYmTAR8#pw?d-khu`Nkv{G++G^K!!x`D%V;^0;UUUDX?Ao_^D=*ZFEM3fo(PAE4ie(7jr{FD<Ic+@bALus{&Bgct&(LgxhXSU%UHcv1c{QJvyS*C$!{p;ex;LBfxpoWrB>jZB40MNt_TFH^rA4$DC0Smy=wu+1A7z$HocSKn_Rq4zuD@g9lAf$+XsllqF2k%saThTC3~?Fyb0{o=($fx@7;H1_)gpuDBb};y=)KdS-&i*32sq)Z2OFbwG|FXRma-<b+AlS^&wi=G;+M_i40mOsy1F7QpvLZ-mWSVCE2v{B)M<${WA*fnC#+Dd&BbL{B)D0;Lyc8Um2C*)LXC#=9aLN5)ed7GdLuc(hm?18)WkO7HA-J-f!~9>Khl837L`^~JB_h1=TZ0Rli>QK<;o^|tsC}$e{p@Qob}FaJ+aW%FlMwTsOzn{gS&IbZ(5jJ>FQP`z8O}D^^cBg(RJ?o@;|qWTdzRut1A#pL4!{DS^(xaquzr|s)`&|#BzrV36H#s7Y#iIs!LzdJJtTeTCs&qDt0IG{7V{D0@XDgL!xgNe+r52N{7N#CDB~w2>8gAD6FTaa^GXUtPCNZ8Qo4$oz>l2bQMzTvVMjML@;<$RG;g6H&N;?qJ-<7)G<e?IJ;@$sazbh9H7Hei^Mf@$uXbpEGKCp;A&P5UMWX8&(BAXn^PVsJK`x?PN{RVIccE!)eVP^9xfFxdw~IkxO96+ORVxx5Vm;lc5p3m5P*yAZN`gTRtkr;>$xE99n~pVl}zi{9Juu*3^X(%2yN#Mn}6#osJ1-x@LN}KSN#43uUh>2wAT}}N+?__XGg$$^AMJI#K<*;4jaq1a#+p+>$sn8sB<yeRLl;lLT9d$9j=2^+*nA*T-SIB3(lJ|X<32RRN0@7P~kbPkLL?~5@Ia1U!>oz5X+xlUR2z|G?kn42$T{>c>{o+W(P4D=I<UIytl?WiD$B!uj`w!^&qQ~Z>YK!K5OukC2x0sx8hF^zrnD@x9h6*4N4Zw57;-@m>Z~380#Oftv*0j(pKhX?%?_!=%|9)ofy()jOxa1^|NlRUpcG-xIKRcU>!x}_XpVVf5XIIdWww?wC%(Yge-9f#4hR63z7<Iw?<ElqWC!mZjFJy!DDb;UpJK}9R+DVj_`YX8Uiy?uus99?U)6pm0Z%&kRral_jY1Y<%V(^4>!Iz8l_3aj)iOL$0{C3IgF`!LJxVnmuFQev*;3qv2;dpP$O8=44|fqNucYah$M4>n2Z?wT#OR+^>dZlM?-z{Gcd{?a*$51l9~KoaZ|0oOQL*&d#-gZui1m@5isv}7$iHTOlQ_qJy<bKW<hxjLm%4%KAJUg(N{RZx|vc*P~owmvJva;XDjRgq@vr*IaQUroS#(U-Kn~hTceVfRD7?yH!4|=p3T4u(x;>9^aLJ|e7j5O(Ad=92jyi(gKJWbhzqqr{axzOX793_h)OZ2A6RgM#C61Y1CyHTKk|9V-vRwvozvGDc`vp3bWBzFHiL<@?v#IC?#NGubJM*}P(L=?v{fa4st1#{I!zk;vA^m0w0X%`-CZ*1y!GJG(NXEzFy&|SV?STM8jUB6pJag#A<-kt=1ym+$xHQkdQA-6r4+>GV>v(IzO8k`zSY_ds%C7Ka5dbQsruW<BJR|!aX6ZIvDe(=I%rQR>&Br;DQ|2>Y*j0bQo4Npik7Y0gD5>q%x~!BHoq0-8O(^Sb=>r(SSr7(I&M@@bh~b%V`a~J`7Db3asyr`SuRzD`5R>Ehb{l>@@%okq@DAnN0B@B_a^998@$_QZYi_Io2pms()r4q3&YehPIL99pqh`AIjK|ox@ktefbE6lw@)_~sfn!Bo_53Z0rPw%H(dYe9J(Le&^ZryNuW<$n1thXvr@g2WzhDADJ;z)N3Qco!>Xfu4vp?H)b4@<-X>|f<Y4OT^#N0WmmR!&D!W^IkB8CL-odb2Eb%I})r?S7#!LTVAHF$FFD$xl+Mn9&mCzBc{>+_>$}j4gAA4_;p3@_!^b9Rt+uX|OzLX``tUBmt%=i5_LR%#_?k(MQEQG(d#>(6H8*y9q2VUwH;0%z0N0Za4A;ZQVt2pf%y^PZ&w-V0olhM7f+F7dR$Dd2vtg2s=8E9o;I({^2^CheILx=CS-`ksMW^=u=KdO>L*3YFC68sj9l+v#%+BaO2<y7|2^cPc96J}KS;OQjJuBDK(ba6dyIbK*u4$Rc7guZh1L$L}o>r9eiv+lmdcNkKd(0#l6C@oj2qoXQ41yO4&^+N5)Sh?{M4<4{vxuwV4dN!K_FW;(*kldx*G(N=jKYWX~e8YqQ00"""
cpp_sources = gzip.decompress(base64.b85decode(cpp_sources)).decode("utf-8")
module = compile_module(cpp_sources=cpp_sources)

import torch
import torch.distributed as dist
from task import input_t, output_t


# def local_kernel(data):
#     cfg, rank_data, rank, world_size = data
#     experts_per_rank = cfg.num_experts // world_size
#     weights = ((1 + rank_data.indices // experts_per_rank) * rank_data.weights).sum(dim=1, keepdim=True)
#     return rank_data.x * weights.to(cfg.out_dtype)

def init_process_group(*args, **kwargs):
    rank = kwargs.get('rank', None)
    world_size = kwargs.get('world_size', None)
    runner.reset(rank, world_size)
    return dist_init_process_group(*args, **kwargs)
dist_init_process_group = dist.init_process_group
dist.init_process_group = init_process_group

# def destroy_process_group(*args, **kwargs):
#     return dist_destroy_process_group(*args, **kwargs)
# dist_destroy_process_group = dist.destroy_process_group
# dist.destroy_process_group = destroy_process_group

# this increments every time a new process group is created
# dist.distributed_c10d._get_process_group_uid(dist.distributed_c10d._get_default_group())

def auto_barrier(**kwargs):
    if kwargs:
        return python_barrier(**kwargs)
    else:
        return runner.barrier()

python_barrier = dist.barrier
dist.barrier = auto_barrier

runner = module.PeerManager()

def custom_kernel(data: input_t) -> output_t:
    # return None
    return module.run_all2all(data, runner)
    # return local_kernel(data)
