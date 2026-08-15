// T2B shared-memory overlay for the P1-family phases (z-regen + phase-1b).
// Function-scope __shared__ arrays in two inlined bodies do NOT overlay —
// they sum, and the t2b TU sits within ~400 B of the 160 KB LDS ceiling.
// Namespace-scope __shared__ gives both phases the SAME storage; they run
// strictly sequentially (grid barrier between), so aliasing is sound.
#pragma once

namespace production_fused_moe::n2 {

#ifndef N2GM_G
#define N2GM_G 1
#endif
inline constexpr int kT2bShMrows = 32 * N2GM_G;

__shared__ int t2bsh_tok_lds[kT2bShMrows];
__shared__ int t2bsh_srow_lds[kT2bShMrows];
__shared__ float t2bsh_ascale_lds[56][kT2bShMrows];
__shared__ float t2bsh_b1s_lds[2][2][56];   // 1z: [gu][j][k]; 1b: plane [0]
__shared__ float t2bsh_amax_lds[2][kT2bShMrows];
__shared__ __align__(16) unsigned char t2bsh_a2_lds[kT2bShMrows][272];
__shared__ __align__(16)
    unsigned char t2bsh_a_lds[2][kT2bShMrows][8][16];

}  // namespace production_fused_moe::n2
