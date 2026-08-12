#!/usr/bin/env bash
# exp_21 ceilings probe. Run on the HOST (rocm-smi's setters need the host; its readers usually
# work in-container too, and run_saturation.py re-collects the same commands into the JSON).
#
# Every number that becomes a dashed horizontal in Fig 2 must come from THIS node. The sibling's
# 76.8 / 537.6 GB/s per-link and aggregate xGMI, 8 TB/s HBM3E and 148 GB/s service rate are
# gfx950 facts (FIGURE_SPECS.md section 5.1) and must never appear in a GEMM-RS artifact.
set -uo pipefail
OUT=${1:-/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation/ceilings_host.txt}

{
echo "########## identity ##########"
date -Is; hostname
echo
echo "########## clocks (must be pinned at 1900 before any timing) ##########"
rocm-smi --showclocks 2>&1
echo
rocm-smi --showmclk 2>&1
echo
echo "########## xGMI / interconnect, every spelling this stack might answer ##########"
for c in "--shownodesbw" "--showtopo" "--showtopoweight" "--showtopotype" "--showtoponuma" \
         "--showbw" "--showpids"; do
  echo "===== rocm-smi $c ====="
  rocm-smi $c 2>&1
  echo
done
echo "########## amd-smi, if present ##########"
if command -v amd-smi >/dev/null 2>&1; then
  echo "===== amd-smi static ====="; amd-smi static 2>&1 | head -200
  echo "===== amd-smi metric ====="; amd-smi metric 2>&1 | head -200
  echo "===== amd-smi list ====="; amd-smi list 2>&1 | head -60
else
  echo "amd-smi not present on this node"
fi
echo
echo "########## rocminfo agent summary ##########"
rocminfo 2>&1 | grep -E 'Name:|Marketing|Compute Unit|Max Clock|Chip 1st|Uuid|BDFID' | head -80
echo
echo "########## hipDeviceProp via hipinfo, if present ##########"
if command -v hipinfo >/dev/null 2>&1; then hipinfo 2>&1 | head -60; else echo "hipinfo absent"; fi
echo
echo "########## NOTES FOR result.md ##########"
cat <<'TXT'
- If no per-link xGMI bandwidth figure appears above, say so in result.md and use the measured
  plateau of the single-fanout curve as the EMPIRICAL per-link ceiling. An honest empirical
  ceiling beats a borrowed spec number.
- The bf16 MFMA peak for the panel-a horizontal is derived, with the arithmetic written out:
  sclk (from the pinned clock reading above) x 304 CU x per-CU bf16 FLOP/cycle implied by
  v_mfma_f32_16x16x16_bf16. Do not quote a datasheet TFLOPS number.
- The HBM horizontal comes from hipGetDeviceProperties on this node
  (2 x memoryClockRate x memoryBusWidth / 8), recorded in saturation.json with its source string.
TXT
} 2>&1 | tee "$OUT"
echo "wrote $OUT"
