#!/usr/bin/env bash
# One complete arm of the RELEASE_GROUP sweep: the full gate ladder, then M9,
# then the logs archived under arms/<tag>/ so the next arm cannot overwrite them
# (gate_ladder.sh writes fixed log names into the experiment's logs/ dir).
#
#   run_arm.sh <tag> [m9_epoch_scale]
#
# Another tenant of this node cycles 8-GPU evaluator jobs, and a Windows-side
# push takes long enough to lose the race between a clean check and a launch.
# Every GPU step therefore waits for a clean node from inside this shell, and a
# ladder that dies on a preflight (rather than on a gate) is retried rather than
# reported as a failure of the candidate.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity
TAG=${1:?usage: run_arm.sh <tag> [m9_scale]}
SCALE=${2:-1}
ATTEMPTS=${3:-4}
mkdir -p "$D/logs" "$D/arms/$TAG"

kfd_pids() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l; }

wait_clean() {   # wait_clean <label>
  local label=$1
  for i in $(seq 0 179); do
    if [ "$(kfd_pids)" = "0" ]; then
      [ "$i" -gt 0 ] && echo "  [$label] node clean after $((i * 20))s"
      return 0
    fi
    [ $((i % 6)) = 0 ] && \
      echo "  [$label] waiting for the node: $(kfd_pids) KFD pids ($((i * 20))s)"
    sleep 20
  done
  echo "  [$label] node never went clean"
  return 1
}

echo "########## arm $TAG ##########"
echo -n "RELEASE_GROUP in the pushed source: "
grep -oP '(?<=define HK_GEMM_RS_MI300X_RELEASE_GROUP )\d+' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp

ladder=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  wait_clean "ladder try $attempt" || break
  echo "########## gate ladder, attempt $attempt ##########"
  bash $ON/tools/gate_ladder.sh exp_05_release_granularity 2>&1 \
    | tee "$D/logs/ladder.log"
  ladder=${PIPESTATUS[0]}
  [ "$ladder" = "0" ] && break
  # A preflight failure says the node was taken, not that the candidate is bad.
  if grep -q 'FAILED AT: M[07] preflight\|FAILED AT: M0 node dirty' \
       "$D/logs/ladder.log"; then
    echo "### attempt $attempt lost the node at a preflight; retrying"
    continue
  fi
  echo "### attempt $attempt failed at a real gate; not retrying"
  break
done
echo "### ladder exit: $ladder"

# The golden is frozen and build.sh does not know about it.
[ -f $ON/harness/build/gemm_rs_mi300x_e3base.so ] || \
  docker exec dhk-gemmrs bash $D/build_golden.sh

m9=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  wait_clean "m9 try $attempt" || break
  bash $D/run_m9.sh "$SCALE" "$TAG"
  m9=$?
  [ "$m9" = "0" ] && break
  grep -q 'M9 REFUSED' "$D/logs/m9_${TAG}.log" 2>/dev/null || break
done
echo "### m9 exit: $m9"

cp $D/logs/*.log "$D/arms/$TAG/" 2>/dev/null
cp $D/logs/*.json "$D/arms/$TAG/" 2>/dev/null
cp $ON/harness/m7_results.json "$D/arms/$TAG/m7_results.json" 2>/dev/null

echo
echo "########## arm $TAG summary: ladder=$ladder m9=$m9 ##########"
sed -n '/GATE M7 /,$p' "$D/arms/$TAG/m7_bench.log" 2>/dev/null
echo "artifacts: $D/arms/$TAG"
[ "$ladder" = "0" ] && [ "$m9" = "0" ] && echo "ARM $TAG PASSED" || echo "ARM $TAG HAS FAILURES"
