#!/usr/bin/env bash
# Pull the node's own ceiling evidence out of ceilings.txt into something quotable.
# Read-only; no GPU. Every number printed here must come from THIS node -- no gfx950
# figure may appear in any exp_21 artifact.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation
C=$EXP/ceilings.txt
[ -f "$C" ] || { echo "no ceilings.txt yet"; exit 1; }

echo "########## sections captured ##########"
grep -n '^##########' "$C" | sed 's/^/  /'

echo
echo "########## rocm-smi --shownodesbw (xGMI per-link matrix) ##########"
sed -n '/########## shownodesbw/,/^########## showtopo /p' "$C" | head -40

echo
echo "########## link type / hops (showtopo) ##########"
sed -n '/########## showtopo /,/########## showtopoweight/p' "$C" \
  | grep -iE 'LINK TYPE|XGMI|hops|^GPU[0-9]' | head -30

echo
echo "########## amd-smi: xgmi / bandwidth / speed lines ##########"
grep -iE 'xgmi|max_bandwidth|bit_rate|link_speed|lanes|width' "$C" | head -40

echo
echo "########## memory clock (showmclk) ##########"
sed -n '/########## showmclk/,/##########/p' "$C" | grep -i mclk | head -10

echo
echo "########## hipDeviceProp_t bits that set the analytic HBM ceiling ##########"
grep -E 'memory_clock_rate_khz|memory_bus_width_bits|hbm_peak_gbps_from_props|gcn_arch|multi_processor_count|clock_rate_khz|l2_cache' "$C" | sed 's/^/  /'
