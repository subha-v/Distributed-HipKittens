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
// v2.2 (AGPR tiles): the 128x128 shape measured 13,440 K-stages/CTA with
// fixed costs dominating (structure 7.9 ms + exposed loads 13.4 of 21.3
// total; MFMA nearly free).  Bigger tiles cut tile count, halve HBM
// re-reads, and amortize every fixed cost over more MFMA per stage.  The
// 256x256 shape needs 256 f32 accs/lane — the AGPR file — so build this
// WITHOUT -amdgpu-mfma-vgpr-form=1; WG_MT/WG_NT stay knobs and 128-class
// shapes remain buildable for vgpr-form TUs.
#ifndef WG_MT
#define WG_MT 256
#endif
#ifndef WG_NT
#define WG_NT 256
#endif
inline constexpr int kWgKT = 32;      // K rows per stage
inline constexpr int kWgMT = WG_MT;   // CTA tile M
inline constexpr int kWgNT = WG_NT;   // CTA tile N
inline constexpr int kWgNW = kWgNT / 4;   // per-wave N strip
inline constexpr int kWgMi = kWgMT / 16;  // MFMA M tiles per wave
inline constexpr int kWgNi = kWgNW / 16;  // MFMA N tiles per wave
static_assert(4096 % kWgMT == 0 && 7168 % kWgMT == 0, "M divides");
static_assert(4096 % kWgNT == 0 && 7168 % kWgNT == 0 && 2048 % kWgNT == 0,
              "N divides");
// LDS layout [m][k]: an MFMA fragment (8 k-consecutive bf16 at fixed m) is
// ONE aligned wg_bf16x8 load.  Row stride 40 elems = 80 B keeps every row
// 16 B-aligned (80 = 5*16); fills pay with strided 2 B writes.
inline constexpr int kWgLdsRow = kWgKT + 8;      // 40

typedef __attribute__((__vector_size__(2 * sizeof(float)))) float wg_f32x2;

// HARDWARE fp8 conversion (v_cvt_pk_f32_fp8).  The __hip_cvt_* helpers can
// lower to branchy byte-checking emulation (measured as the dominant cost of
// the whole phase); the builtin converts a packed byte pair in one VALU op.
// word=false converts bytes [0:1] of the dword, word=true bytes [2:3].
// The word selector must be an immediate — quad converts one dword to 4
// floats with two hardware ops.
__device__ __forceinline__ void k0p6wg_cvt_quad(unsigned int dw, float (&f)[4]) {
  const wg_f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(dw, false);
  const wg_f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8(dw, true);
  f[0] = lo[0];
  f[1] = lo[1];
  f[2] = hi[0];
  f[3] = hi[1];
}

// 16 fp8 bytes (one uint4) -> 16 scaled bf16 into contiguous LDS.
__device__ __forceinline__ void k0p6wg_cvt16(
    const uint4 raw, const float s, __hip_bfloat162* dst) {
  const std::uint16_t* pairs = reinterpret_cast<const std::uint16_t*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __half2_raw h2 = __hip_cvt_fp8x2_to_halfraw2(pairs[i], __HIP_E4M3);
    const float2 f = __half22float2(__half2(h2));
    dst[i] = __float22bfloat162_rn(make_float2(f.x * s, f.y * s));
  }
}

// Vector types BY VALUE: reference-and-cast forms take the accumulator's
// address, which defeats SROA and demotes the acc array to scratch — every
// MFMA then round-trips 4 floats through memory (measured 86 of 93 ms).
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float wg_f32x4;
typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 wg_bf16x8;

__device__ __forceinline__ wg_f32x4 k0p6wg_mfma161632(
    wg_f32x4 acc, wg_bf16x8 a, wg_bf16x8 b) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
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
    __hip_bfloat16* __restrict__ out, int out_ld, bool accumulate,
    __hip_bfloat16* ldsA, __hip_bfloat16* ldsB, float* ldsMeta) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int lr = lane & 15;        // mfma row/col index
  const int lq = lane >> 4;        // mfma K-block / acc quad
  const int n_wave = wv * kWgNW;   // this wave's N strip within the tile

  wg_f32x4 acc[kWgMi][kWgNi];
#pragma unroll
  for (int mi = 0; mi < kWgMi; ++mi)
#pragma unroll
    for (int ni = 0; ni < kWgNi; ++ni)
      acc[mi][ni] = (wg_f32x4){0.f, 0.f, 0.f, 0.f};
  (void)ldsMeta;

  // Per-thread stage state (software pipeline: the NEXT stage's meta chain
  // and raw bytes load during the CURRENT stage's MFMA).  Lane map
  // kk = tid&31 spans 32 DISTINCT k rows per wave-store (the m-segment
  // dimension is bank-degenerate at the 80 B row stride).  Each thread owns
  // kWgSegA/kWgSegB 16-byte segments of its row; a 256-wide tile spans TWO
  // 128-column scale groups, so scales are per-segment.
  constexpr int kWgSegA = kWgMT / 128;   // 16 B segments per thread, A side
  constexpr int kWgSegB = kWgNT / 128;
  const int kk = tid & 31;
  const int msegA = (tid >> 5) * (kWgMT / 8);
  const int msegB = (tid >> 5) * (kWgNT / 8);
  uint4 rawA[kWgSegA], rawB[kWgSegB], rawB2[kWgSegB];
  float p_sa[kWgSegA], p_sb0[kWgSegB], p_sb1[kWgSegB], p_wl;
  int m_sti;
  float m_swt;

  auto prefetch_meta = [&](int k0) __attribute__((always_inline)) {
    const int r = k0 + kk;
    m_sti = sti[r];
    m_swt = swt[r];
  };

  auto issue_raws = [&](int k0) __attribute__((always_inline)) {
    const int r = k0 + kk;
    const int recv = m_sti & 0x00FFFFFF;
    const bool live = recv < nrecv;
    const int tok = live ? recv : 0;
    p_wl = live ? m_swt : 0.0f;
    if (kMode == 0) {
#pragma unroll
      for (int sg = 0; sg < kWgSegA; ++sg) {
        p_sa[sg] = dqdz[(std::size_t)r * 32 + ((m0 + msegA + 16 * sg) >> 7)];
        rawA[sg] = *reinterpret_cast<const uint4*>(
            &dzq[(std::size_t)r * 4096 + m0 + msegA + 16 * sg]);
      }
#pragma unroll
      for (int sg = 0; sg < kWgSegB; ++sg) {
        p_sb0[sg] = xsc[(std::size_t)tok * 56 + ((n0 + msegB + 16 * sg) >> 7)];
        rawB[sg] = *reinterpret_cast<const uint4*>(
            &xrows[(std::size_t)tok * 7168 + n0 + msegB + 16 * sg]);
      }
    } else {
#pragma unroll
      for (int sg = 0; sg < kWgSegA; ++sg) {
        p_sa[sg] = dysc[(std::size_t)tok * 56 + ((m0 + msegA + 16 * sg) >> 7)];
        rawA[sg] = *reinterpret_cast<const uint4*>(
            &dyq[(std::size_t)tok * 7168 + m0 + msegA + 16 * sg]);
      }
#pragma unroll
      for (int sg = 0; sg < kWgSegB; ++sg) {
        const int nseg = n0 + msegB + 16 * sg;
        p_sb0[sg] = dqz[(std::size_t)r * 32 + (nseg >> 7)];        // g half
        p_sb1[sg] = dqz[(std::size_t)r * 32 + 16 + (nseg >> 7)];   // u half
        rawB[sg] = *reinterpret_cast<const uint4*>(
            &zq[(std::size_t)r * 4096 + nseg]);
        rawB2[sg] = *reinterpret_cast<const uint4*>(
            &zq[(std::size_t)r * 4096 + 2048 + nseg]);
      }
    }
  };

  // LDS is [m][k] (row stride 40 elems = 80 B, 16 B-aligned per row): an
  // MFMA fragment is ONE aligned wg_bf16x8 load; the fill pays with
  // strided 2 B writes (fire-and-forget).
  auto fill = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int sg = 0; sg < kWgSegA; ++sg) {
      const float sa = p_sa[sg] * p_wl;
      const unsigned int* pa =
          reinterpret_cast<const unsigned int*>(&rawA[sg]);
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        float f[4];
        k0p6wg_cvt_quad(pa[d], f);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          ldsA[(msegA + 16 * sg + 4 * d + i) * kWgLdsRow + kk] =
              __float2bfloat16(f[i] * sa);
      }
    }
    if (kMode == 0) {
#pragma unroll
      for (int sg = 0; sg < kWgSegB; ++sg) {
        const unsigned int* pb =
            reinterpret_cast<const unsigned int*>(&rawB[sg]);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          float f[4];
          k0p6wg_cvt_quad(pb[d], f);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            ldsB[(msegB + 16 * sg + 4 * d + i) * kWgLdsRow + kk] =
                __float2bfloat16(f[i] * p_sb0[sg]);
        }
      }
    } else {
#pragma unroll
      for (int sg = 0; sg < kWgSegB; ++sg) {
        const unsigned int* gp =
            reinterpret_cast<const unsigned int*>(&rawB[sg]);
        const unsigned int* up =
            reinterpret_cast<const unsigned int*>(&rawB2[sg]);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          float fg[4], fu[4];
          k0p6wg_cvt_quad(gp[d], fg);
          k0p6wg_cvt_quad(up[d], fu);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float g = fg[i] * p_sb0[sg];
            const float v =
                g * (1.0f / (1.0f + __expf(-g))) * (fu[i] * p_sb1[sg]);
            ldsB[(msegB + 16 * sg + 4 * d + i) * kWgLdsRow + kk] =
                __float2bfloat16(v);
          }
        }
      }
    }
  };

  const __bf16* ldsAb = reinterpret_cast<const __bf16*>(ldsA);
  const __bf16* ldsBb = reinterpret_cast<const __bf16*>(ldsB);

  if (s < t) {
    prefetch_meta(s);
    issue_raws(s);
  }
  for (int k0 = s; k0 < t; k0 += kWgKT) {
    __syncthreads();  // previous stage's MFMA reads are done
    fill();
    if (k0 + kWgKT < t) {
      prefetch_meta(k0 + kWgKT);
      issue_raws(k0 + kWgKT);
    }
    __syncthreads();
    // ---- MFMA: kWgMi x kWgNi tiles; one b128 frag load each ------------
    wg_bf16x8 bv[kWgNi];
#pragma unroll
    for (int ni = 0; ni < kWgNi; ++ni) {
      const int n = n_wave + 16 * ni + lr;
      bv[ni] = *reinterpret_cast<const wg_bf16x8*>(
          &ldsBb[n * kWgLdsRow + 8 * lq]);
    }
#pragma unroll
    for (int mi = 0; mi < kWgMi; ++mi) {
      const int m = 16 * mi + lr;
      const wg_bf16x8 av = *reinterpret_cast<const wg_bf16x8*>(
          &ldsAb[m * kWgLdsRow + 8 * lq]);
#pragma unroll
      for (int ni = 0; ni < kWgNi; ++ni)
        acc[mi][ni] = k0p6wg_mfma161632(acc[mi][ni], av, bv[ni]);
    }
  }

  // ---- store: lane holds D[m = 16*mi + 4*lq + i][n = n_wave + 16*ni + lr]
  // accumulate=true sums across microbatches (bf16 RMW; the host adds into
  // main_grad once per layer per iteration).
#pragma unroll
  for (int mi = 0; mi < kWgMi; ++mi) {
#pragma unroll
    for (int ni = 0; ni < kWgNi; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = m0 + 16 * mi + 4 * lq + i;
        const int n = n0 + n_wave + 16 * ni + lr;
        const std::size_t o = (std::size_t)m * out_ld + n;
        float v = acc[mi][ni][i];
        if (accumulate) v += __bfloat162float(out[o]);
        out[o] = __float2bfloat16(v);
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
    int n_experts, int bid, int nct, int tile_lo, int tile_hi,
    bool accumulate,
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

  constexpr int kN13 = 7168 / kWgNT;
  constexpr int kN2 = 2048 / kWgNT;
  constexpr int kT13 = (4096 / kWgMT) * kN13;
  constexpr int kT2 = (7168 / kWgMT) * kN2;
  const int per_e = kT13 + kT2;
  const int total = n_experts * per_e;
  const int hi = (tile_hi > 0 && tile_hi < total) ? tile_hi : total;
  for (int tidx = tile_lo + bid; tidx < hi; tidx += nct) {
    const int e = tidx / per_e;
    const int u = tidx - e * per_e;
    const int s = ldsRange[2 * e + 0] * 32;
    const int t = ldsRange[2 * e + 1] * 32;
    if (u < kT13) {
      const int m0 = (u / kN13) * kWgMT;
      const int n0 = (u % kN13) * kWgNT;
      k0p6wg_tile<0>(dzq, dqdz, zq, dqz, xrows, xsc, dyq, dysc, sti, swt,
                     nrecv, padded, s, t, m0, n0,
                     dw13 + (std::size_t)e * 4096 * 7168, 7168, accumulate,
                     ldsA, ldsB, ldsMeta);
    } else {
      const int v = u - kT13;
      const int m0 = (v / kN2) * kWgMT;
      const int n0 = (v % kN2) * kWgNT;
      k0p6wg_tile<1>(dzq, dqdz, zq, dqz, xrows, xsc, dyq, dysc, sti, swt,
                     nrecv, padded, s, t, m0, n0,
                     dw2 + (std::size_t)e * 7168 * 2048, 2048, accumulate,
                     ldsA, ldsB, ldsMeta);
    }
  }
}

// Slice-count helper for hosts: total tiles at the compiled shape.
__device__ inline int n2p6t2b_wgrad_total_tiles(int n_experts) {
  return n_experts *
         ((4096 / kWgMT) * (7168 / kWgNT) + (7168 / kWgMT) * (2048 / kWgNT));
}

}  // namespace production_fused_moe::n2
