#!/usr/bin/env bash
# Pin/unpin GPU clocks for reproducible timing. Run on the HOST, not inside the
# container. Idle sclk on this node is ~132 MHz against ~1900 MHz under load, so
# unpinned clocks make short measurements unrepeatable.
#   set_clocks.sh pin [MHz]
#   set_clocks.sh unpin
#   set_clocks.sh show
set -uo pipefail

action=${1:-show}
freq=${2:-1900}

case "$action" in
  pin)
    rocm-smi --setperfdeterminism "$freq" 2>&1 | grep -E 'GPU|Success|Error' | head -12
    echo "-- resulting sclk --"
    rocm-smi --showclocks 2>&1 | grep sclk
    ;;
  unpin)
    rocm-smi --resetperfdeterminism 2>&1 | grep -E 'GPU|Success|Error' | head -12
    ;;
  show)
    rocm-smi --showclocks 2>&1 | grep -E 'sclk|mclk|fclk'
    echo "-- power / temp / utilization --"
    rocm-smi 2>&1 | tail -14
    ;;
  *)
    echo "usage: set_clocks.sh {pin [MHz]|unpin|show}"; exit 2
    ;;
esac
