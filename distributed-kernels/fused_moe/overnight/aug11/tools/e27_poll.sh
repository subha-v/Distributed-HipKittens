#!/usr/bin/env bash
# exp_27: poll a batch. usage: e27_poll.sh TAG
#
# THE BIT-IDENTITY GATE, and why it is a SAME-RUN comparison.
# The [MOK GATE] `relative` digits are not reproducible across runs: the control
# batch e26_m5b, five repeats of ONE config on ONE build, printed
# 0.008294/0.008293/0.008294/0.008293/0.008293 for mps_mega, and production
# wandered over 0.005733-0.005802. So "candidate run i == control run j" is not a
# gate, it is a coin flip on the last digit.
#
# But in every one of those runs mps_mega and pf6gm_mega printed the SAME digits,
# because they compute the same math against the same per-run reference. exp_27
# does not touch pf6gm_mega at all. So the drift-free test is:
#
#     within each run, mps_mega's (max_abs, relative) == pf6gm_mega's
#
# which is a same-run paired denominator -- the only kind this project accepts --
# and it is available in every single run rather than once per batch.
set -uo pipefail
TAG="${1:?usage: e27_poll.sh TAG}"
S=$HOME/overnight-scratch
echo "== now: $(date -u) =="
pgrep -af 'screen.sh|torchrun' | head -3 || echo "(nothing running)"
echo
echo "== driver tail =="
tail -6 "$S/${TAG}.driver.log" 2>/dev/null || echo "(no driver log)"

echo
echo "== BIT-IDENTITY: mps_mega vs pf6gm_mega, per run, same run =="
printf '%-56s %-10s %-10s %-10s %-10s %s\n' run mps_abs pf6_abs mps_rel pf6_rel identical
for f in $(ls -1 "$S/${TAG}"_*.log 2>/dev/null); do
  ma=$(grep -oE '\[MOK GATE\] mps_mega max_abs=[0-9.eE+-]+' "$f" | head -1 | sed 's/.*=//')
  mr=$(grep -oE '\[MOK GATE\] mps_mega max_abs=[0-9.eE+-]+ relative=[0-9.eE+-]+' "$f" | head -1 | sed 's/.*relative=//')
  pa=$(grep -oE '\[MOK GATE\] pf6gm_mega max_abs=[0-9.eE+-]+' "$f" | head -1 | sed 's/.*=//')
  pr=$(grep -oE '\[MOK GATE\] pf6gm_mega max_abs=[0-9.eE+-]+ relative=[0-9.eE+-]+' "$f" | head -1 | sed 's/.*relative=//')
  ok=NO; [ -n "$ma" ] && [ "$ma" = "$pa" ] && [ "$mr" = "$pr" ] && ok=YES
  printf '%-56s %-10s %-10s %-10s %-10s %s\n' "$(basename "$f" | cut -c1-56)" "$ma" "$pa" "$mr" "$pr" "$ok"
done

echo
echo "== other gates (must be uniform) =="
grep -hoE '\[MPS SOAK\] completed=[0-9]+/[0-9]+ pperr=[0-9]+ poison=[0-9]+' "$S/${TAG}"_*.log 2>/dev/null | sort | uniq -c
grep -hoE '\[MARK\] control_fails=\w+' "$S/${TAG}"_*.log 2>/dev/null | sort | uniq -c
grep -hoE '\[POISON SELFTEST\] arm=\S+ one_row_poisoned_fails=\w+ nonfinite=[0-9]+' "$S/${TAG}"_*.log 2>/dev/null | sort | uniq -c
grep -hoE '\[POISON\] \S+ arm=\S+ survivors=[0-9]+' "$S/${TAG}"_*.log 2>/dev/null | sed 's/rank=[0-9]*//' | sort | uniq -c | head

echo
echo "== CSV =="
if [ -f "$S/screen_${TAG}.csv" ]; then
  python3 - "$S/screen_${TAG}.csv" <<'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
cols=["idx","status","head","src_rev","prod_us","pf6gm_us","mps_us","ratio_vs_prod","ts_M6_us","ts_planM3toM5_us","ts_M7_us","ts_combine_us","mps_hsaco"]
w=[3,12,9,4,8,8,8,7,8,8,8,8,13]
print("  ".join(c[:x].ljust(x) for c,x in zip(cols,w)))
for r in rows:
    print("  ".join(str(r.get(c,""))[:x].ljust(x) for c,x in zip(cols,w)))
print()
for k in ("ts_M6_us","ts_planM3toM5_us","ts_M7_us","ts_combine_us","mps_us","prod_us","ratio_vs_prod"):
    v=[float(r[k]) for r in rows if r.get(k)]
    if len(v)>=2:
        m=st.mean(v); s=st.stdev(v)
        print(f"{k:20s} n={len(v)}  mean={m:9.2f}  sd={s:7.2f}  sem={s/len(v)**0.5:6.2f}  vals={['%.1f'%x for x in v]}")
PY
else echo "(no CSV yet)"; fi
