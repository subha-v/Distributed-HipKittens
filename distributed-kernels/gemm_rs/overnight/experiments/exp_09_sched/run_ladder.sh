#!/usr/bin/env bash
# exp_09_sched: gate ladder with the two preflights this session kept paying
# for. Edits nothing; tools/gate_ladder.sh is called unmodified.
#
#   run_ladder.sh <exp_dir_name> [skip_soak]
#
# 1. CRLF. nsh.ps1 launders the script IT transports, but the scripts that one
#    docker-execs from the node tree come straight from push.ps1's scp, and
#    push.ps1's normalizing sed does not always win the race: three arms in a
#    row died with "syntax error: unexpected end of file" (a \r on a heredoc's
#    terminator) or "$'\r': command not found", which reads exactly like a
#    compile failure. Normalize and then PROVE it by parsing every tool the
#    ladder will run, in the container that will run it.
# 2. Stale GPU workers. Six wedged multiprocessing children were holding GPU 1
#    at 0% occupancy when this arm first tried to gate. Wait for a real drain
#    rather than aborting; only reap after waiting.
set -uo pipefail
EXP=${1:?usage: run_ladder.sh <exp_dir_name> [skip_soak]}
SKIP=${2:-}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
REL=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "########## preflight 1: CRLF ##########"
n0=$(grep -rlU $'\r' "$REL" --include='*.sh' --include='*.py' 2>/dev/null \
      | grep -v '/build/' | grep -v '/compbench/' | wc -l)
echo "files with CR before: $n0"
find "$REL" -type f -writable \( -name '*.sh' -o -name '*.py' \) \
  -not -path '*/build/*' -not -path '*/compbench/*' -exec sed -i 's/\r$//' {} +
n1=$(grep -rlU $'\r' "$REL" --include='*.sh' --include='*.py' 2>/dev/null \
      | grep -v '/build/' | grep -v '/compbench/' | wc -l)
echo "files with CR after : $n1"

bad=0
for f in tools/gate_ladder.sh tools/m1_build.sh tools/m2_isa.sh \
         tools/m2_report.sh harness/build.sh; do
  if docker exec dhk-gemmrs bash -n "$ON/$f" 2>/dev/null; then
    printf '  parses OK (in container): %s\n' "$f"
  else
    printf '  !! DOES NOT PARSE: %s\n' "$f"; bad=1
  fi
done
[ "$bad" = "0" ] || { echo "ABORT: a ladder tool does not parse"; exit 1; }

echo
echo "########## preflight 2: node drain ##########"
for i in $(seq 1 18); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  echo "[$(date +%H:%M:%S)] KFD pids: $n"
  [ "$n" = "0" ] && break
  [ "$i" = "9" ] && { echo "  90s with no drain -- reaping (SIGTERM, ours only)";
                      bash "$ON/tools/reap_stale.sh" >/dev/null 2>&1; }
  sleep 10
done
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || { echo "ABORT: node still dirty ($n pids)"; exit 1; }

echo
echo "########## gate ladder ##########"
bash "$ON/tools/gate_ladder.sh" "$EXP" $SKIP
rc=$?
echo "gate_ladder rc=$rc"
exit $rc
