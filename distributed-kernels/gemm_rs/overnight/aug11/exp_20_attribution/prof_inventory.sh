#!/usr/bin/env bash
# exp_20: what is actually in prof/, and WHICH counters each existing cell holds.
#
# Necessary because run_counters.sh skips a cell whose CSV exists, keyed on the
# TAG alone. A prior session collected under the same tag names with a DIFFERENT
# counter set, so a tag named _g2 may hold g2 of the old definition. Trusting
# the tag would mislabel the data.
#
# Pure awk on the host: no heredoc, and nothing piped into `docker exec`.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
PROF=$OUT/prof

echo "===== prof/ inventory ====="
ls -la --time-style=full-iso "$PROF" 2>&1 | head -40

echo
echo "===== distinct Counter_Name per cell, read from the CSV header + rows ====="
for d in "$PROF"/*/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d")
  f="$d/p_counter_collection.csv"
  if [ ! -f "$f" ]; then
    printf '  %-24s NO CSV\n' "$tag"
    continue
  fi
  # rocprofv3 quotes CSV fields, so the header is compared with quotes stripped.
  names=$(awk -F, '
    NR==1 { for (i=1; i<=NF; i++) { h=$i; gsub(/"/, "", h);
                                    if (h=="Counter_Name") col=i } next }
    col && $col != "" { seen[$col]=1 }
    END { n=0; out=""; for (k in seen) { out = (n++ ? out "," k : k) }
          print (n ? out : "NO Counter_Name COLUMN") }' "$f" | tr ',' '\n' \
    | sort | tr '\n' ' ')
  rows=$(wc -l < "$f")
  printf '  %-24s rows=%-8s %s\n' "$tag" "$rows" "$names"
done

echo
echo "===== PROF lines (correctness under instrumentation) ====="
grep -h '^PROF' "$PROF"/*.log 2>/dev/null || echo "  none"
echo done
