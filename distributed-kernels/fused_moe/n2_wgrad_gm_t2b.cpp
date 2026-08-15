// T2B WGRAD phase — dW13 = (swt·dZ)^T · X and dW2 = (swt·dY)^T · act(z),
// in-kernel, all CTAs, post-M8 (v2 of the wgrad ladder: v1 was host torch
// segment GEMMs ~1.7 s/iter fp32, v1.5 host bf16 slices ~0.3 s/iter; this
// phase deletes the host dequant traffic and the per-layer D2H syncs).
// Design: overnight/aug14/T2_BACKWARD_MEGA_DESIGN.md (wgrad filler section);
// v3 moves these tiles into the M7 slab-wait bubbles.
//
// GEOMETRY. Sorted-space rows are expert-contiguous at 32-row block
// granularity (sei per block, ascending), so per-expert K ranges are exact
// multiples of 32 — no ragged K. Per expert e with rows [s, t):
//   dW13[e][m][n] = sum_r wl(r) · dz(r, m) · x(tok(r), n)     m<4096, n<7168
//   dW2 [e][m][n] = sum_r wl(r) · dy(tok(r), m) · act(z)(r, n) m<7168, n<2048
// where wl(r) = swt[r] · (recv(r) < nvi[1]), recv(r) = sti[r] & 0xFFFFFF,
// tok(r) = recv(r) (receive-order row in a_dst/sc, the phase-1b addressing),
// dz/z are sorted-space phase outputs, act(z) = silu(g)·u recomputed from Zq.
//
// SCALES. Every 128-wide output tile spans exactly ONE per-128 scale group
// on each side, so the fp8 operands are pre-scaled into bf16 at LDS fill
// with one scalar per row per side (wl folded into A), and the MFMA runs
// v_mfma_f32_16x16x32_bf16. Per-K-row scales are why this phase cannot ride
// the fp8-MFMA-then-scale-per-group trick of phases 1/2.
//
// TILING. CTA tile 128(M) x 128(N); 4 waves split N into 4 x 32; per wave
// 8 x 2 mfma tiles x f32[4] acc = 64 VGPRs. K staged 32 rows at a time in
// LDS (single buffer, [128][40] bf16 per side, +8 pad destaggers banks),
// overlaid on the t2bsh_* arrays (strictly after their last use; per-CTA
// __syncthreads separates). Tiles enumerate expert-major and grid-stride so
// concurrent CTAs share the expert's rows in the MALL.
//
// OUTPUT. bf16 arenas [E,4096,7168] / [E,7168,2048], OVERWRITE semantics:
// every tile is fully accumulated by exactly one CTA per launch, zero-row
// experts write zeros — the host never zeroes between microbatches.
#pragma once

#include <hip/hip_bf16.h>
#include <cstdint>

namespace production_fused_moe::n2 {

// LDS overlay (k0pf6gm_t2b_shared.hpp storage, reinterpreted):
//   A tiles [128][40] bf16 = 10,240 B  -> t2bsh_a2_lds   (26,112 B)
//   B tiles [128][40] bf16 = 10,240 B  -> t2bsh_a_lds    (24,576 B)
//   row meta (sA/sB0/sB1/tok) [32] x4  -> t2bsh_amax_lds + t2bsh_b1s_lds
//   expert ranges [32][2]              -> t2bsh_tok_lds
inline constexpr int kWgKT = 32;      // K rows per stage
inline constexpr int kWgMT = 128;     // CTA tile M
inline constexpr int kWgNT = 128;     // CTA tile N
inline constexpr int kWgPad = 8;      // LDS row pad (bank destagger)
inline constexpr int kWgLdsW = kWgKT + kWgPad;   // 40

__device__ __forceinline__ float k0p6wg_fp8_to_f32(std::uint8_t b) {
  const __half_raw h = __hip_cvt_fp8_to_halfraw(b, __HIP_E4M3);
  return __half2float(__half(h));
}

__device__ __forceinline__ void k0p6wg_mfma161632(
    float (&d)[4], const __hip_bfloat16 (&a)[8], const __hip_bfloat16 (&b)[8]) {
  typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
  typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
  *(floatx4_t*)d = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
      *(const bf16x8_t*)a, *(const bf16x8_t*)b, *(const floatx4_t*)d, 0, 0, 0);
}

// One CTA tile: dW[m0..m0+128][n0..n0+128] over K rows [s, t).
// A source: fA(r) gives (byte row pointer, scale) for the M side.
// Templated on the B-side fill (x rows vs act(z) recompute).
template <int kMode>  // 0 = dW13, 1 = dW2
__device__ inline void k0p6wg_tile(
    const std::uint8_t* __restrict__ dzq, const float* __restrict__ dqdz,
    const std::uint8_t* __restrict__ zq, const float* __restrict__ dqz,
    const std::uint8_t* __restrict__ xrows, const float* __restrict__ xsc,
    const std::uint8_t* __restrict__ dyq, const float* __restrict__ dysc,
    const int* __restrict__ sti, const float* __restrict__ swt,
    int nrecv, int rowcap, int s, int t, int m0, int n0,
    __hip_bfloat16* __restrict__ out, int out_ld,
    __hip_bfloat16* ldsA, __hip_bfloat16* ldsB, float* ldsMeta) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int lr = lane & 15;        // mfma row/col index
  const int lq = lane >> 4;        // mfma K-block / acc quad
  const int n_wave = wv * 32;      // this wave's N strip within the tile

  float acc[8][2][4];
#pragma unroll
  for (int mi = 0; mi < 8; ++mi)
#pragma unroll
    for (int ni = 0; ni < 2; ++ni)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mi][ni][i] = 0.0f;

  // meta slices: [0..32) = sA (dz/dy scale * wl), [32..64) = sB group-0,
  // [64..96) = sB group-1 (dW2's u half), [96..128) = tok (as float bits? no
  // — ints stored via reinterpret)
  float* sA = ldsMeta;
  float* sB0 = ldsMeta + kWgKT;
  float* sB1 = ldsMeta + 2 * kWgKT;
  int* tokm = reinterpret_cast<int*>(ldsMeta + 3 * kWgKT);

  for (int k0 = s; k0 < t; k0 += kWgKT) {
    __syncthreads();
    // ---- row meta: one thread per K row --------------------------------
    if (tid < kWgKT) {
      const int r = k0 + tid;
      const int recv = sti[r] & 0x00FFFFFF;
      const bool live = recv < nrecv;
      const int tok = live ? recv : 0;
      const float wl = live ? swt[r] : 0.0f;
      tokm[tid] = tok;
      if (kMode == 0) {
        sA[tid] = dqdz[(std::size_t)r * 32 + (m0 >> 7)] * wl;
        sB0[tid] = xsc[(std::size_t)tok * 56 + (n0 >> 7)];
      } else {
        sA[tid] = dysc[(std::size_t)tok * 56 + (m0 >> 7)] * wl;
        sB0[tid] = dqz[(std::size_t)r * 32 + (n0 >> 7)];          // g half
        sB1[tid] = dqz[(std::size_t)r * 32 + 16 + (n0 >> 7)];     // u half
      }
    }
    __syncthreads();
    // ---- A fill: 128 (M) x 32 (K); element (m, kk): idx = tid*16 + j,
    // m = idx & 127 (j-contiguous -> 16 CONTIGUOUS source bytes per thread,
    // 8 threads cover one 128 B row segment coalesced), kk = idx >> 7.
    {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int idx = tid * 16 + j;
        const int m = idx & 127;
        const int kk = idx >> 7;
        const int r = k0 + kk;
        float v;
        if (kMode == 0) {
          v = k0p6wg_fp8_to_f32(dzq[(std::size_t)r * 4096 + m0 + m]);
        } else {
          v = k0p6wg_fp8_to_f32(
              dyq[(std::size_t)tokm[kk] * 7168 + m0 + m]);
        }
        ldsA[m * kWgLdsW + kk] = __float2bfloat16(v * sA[kk]);
      }
    }
    // ---- B fill (same mapping) -----------------------------------------
    {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int idx = tid * 16 + j;
        const int n = idx & 127;
        const int kk = idx >> 7;
        const int r = k0 + kk;
        float v;
        if (kMode == 0) {
          v = k0p6wg_fp8_to_f32(
                  xrows[(std::size_t)tokm[kk] * 7168 + n0 + n]) *
              sB0[kk];
        } else {
          const float g =
              k0p6wg_fp8_to_f32(zq[(std::size_t)r * 4096 + n0 + n]) * sB0[kk];
          const float u =
              k0p6wg_fp8_to_f32(zq[(std::size_t)r * 4096 + 2048 + n0 + n]) *
              sB1[kk];
          const float sig = 1.0f / (1.0f + __expf(-g));
          v = g * sig * u;
        }
        ldsB[n * kWgLdsW + kk] = __float2bfloat16(v);
      }
    }
    __syncthreads();
    // ---- MFMA: 8 M tiles x 2 N tiles, K = 32 in one instruction --------
    __hip_bfloat16 afrag[8];
    __hip_bfloat16 bfrag[2][8];
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      const int n = n_wave + 16 * ni + lr;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        bfrag[ni][j] = ldsB[n * kWgLdsW + 8 * lq + j];
    }
#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
      const int m = 16 * mi + lr;
#pragma unroll
      for (int j = 0; j < 8; ++j) afrag[j] = ldsA[m * kWgLdsW + 8 * lq + j];
#pragma unroll
      for (int ni = 0; ni < 2; ++ni)
        k0p6wg_mfma161632(acc[mi][ni], afrag, bfrag[ni]);
    }
  }

  // ---- store: lane holds D[m = 16*mi + 4*lq + i][n = n_wave + 16*ni + lr]
#pragma unroll
  for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = m0 + 16 * mi + 4 * lq + i;
        const int n = n0 + n_wave + 16 * ni + lr;
        out[(std::size_t)m * out_ld + n] = __float2bfloat16(acc[mi][ni][i]);
      }
    }
  }
  __syncthreads();
}

// The phase driver: expert ranges from sei, expert-major grid-stride tiles.
//   tiles per expert: dW13 32x56 = 1792, dW2 56x16 = 896 -> 2688.
__device__ inline void n2p6t2b_wgrad_body(
    const std::uint8_t* __restrict__ dzq, const float* __restrict__ dqdz,
    const std::uint8_t* __restrict__ zq, const float* __restrict__ dqz,
    const std::uint8_t* __restrict__ xrows, const float* __restrict__ xsc,
    const std::uint8_t* __restrict__ dyq, const float* __restrict__ dysc,
    const int* __restrict__ sti, const float* __restrict__ swt,
    const int* __restrict__ sei, const int* __restrict__ nvi,
    __hip_bfloat16* __restrict__ dw13, __hip_bfloat16* __restrict__ dw2,
    int n_experts, int bid, int nct,
    __hip_bfloat16* ldsA, __hip_bfloat16* ldsB, float* ldsMeta,
    int* ldsRange /* [n_experts][2] */) {
  const int tid = static_cast<int>(threadIdx.x);
  const int padded = nvi[0];
  const int nrecv = nvi[1];
  const int nblk = (padded + 31) >> 5;

  // per-expert block ranges (sei ascending): parallel scan, boundaries only.
  __syncthreads();
  for (int e = tid; e < 2 * n_experts; e += (int)blockDim.x) ldsRange[e] = 0;
  __syncthreads();
  for (int b = tid; b < nblk; b += (int)blockDim.x) {
    const int e = sei[b];
    if (e < 0 || e >= n_experts) continue;
    if (b == 0 || sei[b - 1] != e) ldsRange[2 * e + 0] = b;
    if (b == nblk - 1 || sei[b + 1] != e) ldsRange[2 * e + 1] = b + 1;
  }
  __syncthreads();

  constexpr int kT13 = (4096 / kWgMT) * (7168 / kWgNT);   // 32*56 = 1792
  constexpr int kT2 = (7168 / kWgMT) * (2048 / kWgNT);    // 56*16 = 896
  const int per_e = kT13 + kT2;
  const int total = n_experts * per_e;
  for (int tidx = bid; tidx < total; tidx += nct) {
    const int e = tidx / per_e;
    const int u = tidx - e * per_e;
    const int s = ldsRange[2 * e + 0] * 32;
    const int t = ldsRange[2 * e + 1] * 32;
    if (u < kT13) {
      const int m0 = (u / 56) * kWgMT;
      const int n0 = (u % 56) * kWgNT;
      k0p6wg_tile<0>(dzq, dqdz, zq, dqz, xrows, xsc, dyq, dysc, sti, swt,
                     nrecv, padded, s, t, m0, n0,
                     dw13 + (std::size_t)e * 4096 * 7168, 7168,
                     ldsA, ldsB, ldsMeta);
    } else {
      const int v = u - kT13;
      const int m0 = (v / 16) * kWgMT;
      const int n0 = (v % 16) * kWgNT;
      k0p6wg_tile<1>(dzq, dqdz, zq, dqz, xrows, xsc, dyq, dysc, sti, swt,
                     nrecv, padded, s, t, m0, n0,
                     dw2 + (std::size_t)e * 7168 * 2048, 2048,
                     ldsA, ldsB, ldsMeta);
    }
  }
}

}  // namespace production_fused_moe::n2
