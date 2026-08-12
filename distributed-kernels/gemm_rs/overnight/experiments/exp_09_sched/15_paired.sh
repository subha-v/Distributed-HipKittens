#!/usr/bin/env bash
# exp_09_sched: alternating paired A/B of the candidate against the base state.
#
# Why this exists. Every arm tonight landed within 0.3% of the 230.84 us
# denominator, and two arms with a BYTE-IDENTICAL instruction stream
# (a1b_branchless and a2_fence_mfma differ only in register numbering) reported
# shape-6 means 1.2% apart. So the harness's run-to-run spread on the big
# shapes is comparable to every delta being claimed, and 230.84 was measured in
# an earlier session anyway. Rebuild both states now, alternate them, and let
# the four runs speak.
#
#   15_paired.sh <candidate-arm> [reps]
#
# Builds from arms/<candidate>/*.{cpp,cuh} and base/*.{cpp,cuh}, both of which
# were archived by isa_arm.sh, and restores the candidate at the end.
set -uo pipefail
CAND=${1:?usage: 15_paired.sh <candidate-arm> [reps]}
REPS=${2:-2}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
SRC=$ON/..
OUT=$EXP/paired_$CAND
mkdir -p "$OUT"

put() {   # put <dir-with-sources>
  cp -f "$1/gemm_rs_mi300x.cpp"            "$SRC/gemm_rs_mi300x.cpp"
  cp -f "$1/gemm_rs_mi300x_hk_adapter.cuh" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"
}
build() {
  docker exec dhk-gemmrs bash $ON/harness/build.sh > "$OUT/build_$1.log" 2>&1
  grep -q "ALL MODULES BUILT" "$OUT/build_$1.log" || { echo "BUILD FAILED: $1"; return 1; }
  echo "  built $1"
}
drain() {
  for i in $(seq 1 12); do
    n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
    [ "$n" = "0" ] && return 0
    [ "$i" = "6" ] && bash "$ON/tools/reap_stale.sh" >/dev/null 2>&1
    sleep 10
  done
  return 1
}
bench() {   # bench <tag>
  drain || { echo "  node dirty, skipping $1"; return 1; }
  docker exec -w $ON/harness dhk-gemmrs timeout 1800 \
    python3 -u m7_bench.py 3 50 > "$OUT/m7_$1.log" 2>&1
  cp -f "$ON/harness/m7_results.json" "$OUT/m7_$1.json" 2>/dev/null || true
  grep -E 'geometric mean \(pipelined' "$OUT/m7_$1.log" | sed "s/^/  $1: /"
}

for r in $(seq 1 "$REPS"); do
  echo "########## rep $r: candidate ($CAND) ##########"
  put "$EXP/arms/$CAND"; build "cand_r$r" && bench "cand_r$r"
  echo "########## rep $r: base ##########"
  put "$EXP/base";       build "base_r$r" && bench "base_r$r"
done

echo
echo "########## restoring the candidate and rebuilding ##########"
put "$EXP/arms/$CAND"
build "restore" || echo "!! RESTORE BUILD FAILED -- node .so does not match $CAND"
sha256sum "$SRC/gemm_rs_mi300x.cpp" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"

echo
echo "########## paired result ##########"
python3 - "$OUT" "$CAND" "$REPS" <<'PY'
import json, os, sys, math
out, cand, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
def means(tag):
    p = os.path.join(out, "m7_%s.json" % tag)
    if not os.path.exists(p): return None
    d = json.load(open(p))
    if not d.get("all_correct", True):
        print("  !! %s reported all_correct=False" % tag)
    return [float(x) for x in d["means_us"]]
series = {}
for r in range(1, reps + 1):
    for arm in ("cand", "base"):
        m = means("%s_r%d" % (arm, r))
        if m: series.setdefault(arm, []).append(m)
if not series.get("cand") or not series.get("base"):
    print("  (could not parse m7 json; read the logs)"); raise SystemExit
def gm(v): return math.exp(sum(math.log(x) for x in v) / len(v))
hdr = "  %-6s %-4s" % ("arm", "rep") + "".join("%10s" % ("s%d" % i) for i in range(1, 7)) + "%12s" % "geomean"
print(hdr); print("  " + "-" * (len(hdr) - 2))
agg = {}
for arm in ("cand", "base"):
    for i, m in enumerate(series[arm], 1):
        print("  %-6s %-4d" % (arm, i) + "".join("%10.2f" % x for x in m) + "%12.2f" % gm(m))
    per = [sum(s[j] for s in series[arm]) / len(series[arm]) for j in range(6)]
    agg[arm] = per
    print("  %-6s %-4s" % (arm, "avg") + "".join("%10.2f" % x for x in per) + "%12.2f" % gm(per))
print()
print("  per-shape candidate/base : " +
      "  ".join("%.4f" % (agg["cand"][j] / agg["base"][j]) for j in range(6)))
print("  geomean candidate        : %.2f us" % gm(agg["cand"]))
print("  geomean base             : %.2f us" % gm(agg["base"]))
print("  ratio                    : %.4f  (<1 is faster)" %
      (gm(agg["cand"]) / gm(agg["base"])))
sp = [max(s[j] for s in series[arm]) / min(s[j] for s in series[arm])
      for arm in ("cand", "base") for j in range(6)]
print("  worst within-arm spread  : %.2f%%  <- the noise floor these deltas live in"
      % ((max(sp) - 1) * 100))
PY
