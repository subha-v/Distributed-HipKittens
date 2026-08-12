#!/usr/bin/env bash
# exp_23 counter sampler for the amd-smi cross-check. Read-only queries; this is
# NOT a GPU job and does not take the lease. Run it alongside the instrumented
# screen and feed the CSV to xcheck.py.
#
# MEASURED on gbt350-odcdh2-c05-1, amd-smi 26.2.2 / ROCm 7.2.4, 2026-08-12:
#   * `amd-smi metric -g N --xgmi --csv` -> "gpu,xgmi_err" / "0,N/A".
#     THERE IS NO LIVE xGMI THROUGHPUT COUNTER ON THIS NODE.
#   * `rocm-smi --shownodesbw` -> "0-0 mps" for every pair (a topology
#     capability field, not a counter).
#   * one `amd-smi metric --csv` invocation costs 0.14-0.29 s, so 3 Hz is the
#     honest ceiling and 2 Hz is the safe rate. A 6.5 ms epoch is 1/300 of a
#     sample: this validates the epoch AVERAGE only, never the intra-epoch shape.
#
# Fields taken: gfx_activity (%) and umc_activity (%), the only live utilisation
# percentages available. See xcheck.py for what each one can and cannot prove.
#
# USAGE
#   bash e23_smi_sample.sh out.csv 0 600     # csv, gpu id, seconds
#   (start it ~5 s before the soak begins and let it run ~5 s past the end, so
#    xcheck.py's --window has idle samples on both sides to key off.)
set -uo pipefail
OUT=${1:?usage: e23_smi_sample.sh <out.csv> [gpu] [seconds] [rate_hz]}
GPU=${2:-0}
SECS=${3:-600}
RATE=${4:-2}
SLEEP=$(awk -v r="$RATE" 'BEGIN{printf "%.3f", 1.0/r}')

echo "utc_epoch_s,gpu,gfx_activity,umc_activity" > "$OUT"
END=$(( $(date +%s) + SECS ))
n=0
while [ "$(date +%s)" -lt "$END" ]; do
  t=$(date +%s.%N)
  # One process per sample keeps every field derived from that sample's own read;
  # `amd-smi metric --csv` prints a header line then one row per GPU.
  line=$(timeout 5 amd-smi metric -g "$GPU" --csv 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    # Column order comes from the header of THIS build: gpu, gfx_activity,
    # umc_activity are fields 1..3. Re-derive if amd-smi is upgraded.
    gfx=$(echo "$line" | cut -d, -f2)
    umc=$(echo "$line" | cut -d, -f3)
    case "$gfx$umc" in
      *N/A*) : ;;
      *) echo "$t,$GPU,$gfx,$umc" >> "$OUT"; n=$((n+1)) ;;
    esac
  fi
  sleep "$SLEEP"
done
echo "wrote $n samples to $OUT at a requested ${RATE} Hz" >&2
exit 0
