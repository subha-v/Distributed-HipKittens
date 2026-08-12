#!/usr/bin/env bash
# exp_24: HOST-side verification that every g in the run plan encodes and decodes
# to the intended fields and passes config_is_valid. Also proves the negative
# cases are REJECTED (fail-closed guards), and that g <= 0xFF words are
# bit-identical to the pre-widening encoder.
set -uo pipefail
WORK=$HOME/exp24-stage/cfgtest
mkdir -p "$WORK"
cat > "$WORK/t.cpp" <<'CPP'
#include <cstddef>
#include <cstdint>
#include <cstdio>
// mps_host_bridge.cpp declares this the same way before including the adapter.
__device__ void k0p6_put_row_payload(unsigned char*, const std::uint64_t*,
                                     std::size_t, int, int);
#include "moe_mps_adapter.cuh"
using namespace hk_moe::mps;

static std::uint64_t old_encode(std::uint32_t C, std::uint32_t g,
                                std::uint32_t mode, std::uint32_t fr,
                                bool pf, bool ts) {
  return (std::uint64_t)C | ((std::uint64_t)g << 8) | ((std::uint64_t)mode << 16)
       | ((std::uint64_t)fr << 24) | ((std::uint64_t)(pf?1u:0u) << 32)
       | ((std::uint64_t)(ts?1u:0u) << 33);
}

static void show(const char* tag, std::uint32_t C, std::uint32_t g,
                 std::uint32_t mode, std::uint32_t fr, bool pf, bool ts) {
  const std::uint64_t w = encode_config(C, g, mode, fr, pf, ts);
  const config c = decode_config(w);
  printf("%-26s g=%-5u(0x%03x) word=0x%011llx | C=%-3u mode=%-3u fr=%-3u pf=%d ts=%d "
         "| g_dec=%-5u phys=%u detect=%d thr=%d depthsel=%u skipzero=%d | VALID=%d\n",
         tag, g, g, (unsigned long long)w, c.reserved_comm_ctas, c.mode,
         c.flush_rows, (int)c.pull_fallback, (int)c.timestamps,
         c.group_slices, physical_g(c), (int)detect_dual(c),
         (int)throttle_enabled(c), throttle_depth_sel(c),
         (int)skip_dead_part_zero(c), (int)config_is_valid(c));
}

int main() {
  printf("== run plan ==\n");
  show("1 control",            16, 33,  12, 16, false, false);
  show("2 +A skip part zero",  16, 97,  12, 16, false, false);
  show("3 depth 4",            16, 289, 12, 16, false, false);
  show("4 depth 16",           16, 545, 12, 16, false, false);
  show("5 depth 32",           16, 801, 12, 16, false, false);
  show("7 control +stamps",    16, 33,  12, 16, false, true);
  show("8 +A +stamps",         16, 97,  12, 16, false, true);
  printf("\n== composed candidates (A + each depth) ==\n");
  show("A+depth4",             16, 353, 12, 16, false, false);
  show("A+depth16",            16, 609, 12, 16, false, false);
  show("A+depth32",            16, 865, 12, 16, false, false);
  printf("\n== fail-closed cases (VALID must be 0) ==\n");
  show("A + dual detector",    16, 113, 12, 16, false, false);   // 0x71
  show("A on mode 13",         16, 97,  13, 16, false, false);
  show("A on mode 2",          16, 97,   2, 16, false, false);
  show("depth sel, no throttle",16, 257, 12, 16, false, false);  // 0x101
  show("reserved bit 0x80",    16, 161, 12, 16, false, false);   // 0xA1
  show("reserved bit 0x400",   16, 1057,12, 16, false, false);   // 0x421
  printf("\n== widening is bit-identical for g <= 0xFF ==\n");
  int bad = 0;
  for (std::uint32_t g = 0; g <= 0xFFu; ++g)
    for (std::uint32_t m = 0; m <= 13u; ++m) {
      const std::uint64_t a = encode_config(16, g, m, 16, false, false);
      const std::uint64_t b = old_encode(16, g, m, 16, false, false);
      if (a != b) ++bad;
      if (decode_config(a).group_slices != g) ++bad;
    }
  printf("mismatches over all g in [0,255] x mode in [0,13]: %d (must be 0)\n", bad);
  printf("\n== every legal historical config still validates identically ==\n");
  const std::uint32_t legacy_g[] = {1u, 2u, 4u, 16u, 17u, 33u, 49u};
  for (std::uint32_t g : legacy_g)
    for (std::uint32_t m = 0; m <= 13u; ++m) {
      const config c = decode_config(encode_config(16, g, m, 16, false, false));
      const config o = decode_config(old_encode(16, g, m, 16, false, false));
      if (config_is_valid(c) != config_is_valid(o))
        printf("  DIVERGENCE g=%u mode=%u new=%d old=%d\n", g, m,
               (int)config_is_valid(c), (int)config_is_valid(o));
    }
  printf("  (no DIVERGENCE lines above => validator unchanged for g <= 0xFF)\n");
  return 0;
}
CPP
docker exec subha_k1 bash -lc '
set -uo pipefail
DHK=/home/subvadla/Distributed-HipKittens
W=/home/subvadla/exp24-stage/cfgtest
hipcc -std=c++20 -O1 -DKITTENS_CDNA4 \
  -I$DHK/distributed-kernels/fused_moe -I$DHK/include \
  $W/t.cpp -o $W/t 2> $W/err.log
rc=$?
if [ $rc -ne 0 ]; then echo "hipcc rc=$rc"; grep -E "error" $W/err.log | head -20; exit 0; fi
$W/t
'
exit 0
