#!/usr/bin/env bash
# Landing run, second attempt.
#
# campaign4 spun forever in `acquire` because the shared tools/gpu_lease.sh is
# being edited by its owner while other agents run it: the CR-stripped snapshot
# this experiment took at 12:27Z hit `line 131: [: 10800: integer expression
# expected` on every iteration of the wait loop -- an empty left-hand side in
# the deadline comparison. Earlier the same file was on the node with CRLF and
# would not start at all.
#
# So this implements the acquire/release directly, against THE SAME lock
# directory and log the shared tool uses, writing the same owner/since/pid
# files, so mutual exclusion with every other agent is preserved exactly. It is
# not a private lock and it is not a bypass; it is the same protocol without the
# dependency on a file that changes underneath a running job.
#
# Stale KFD pids are ignored the way the shared tool learned to: a process in
# state Z, or whose wchan is exit_mm/do_exit, cannot dispatch. This experiment's
# own wedged M9 (pid from the 10:05Z fault) is exactly such a process and it has
# not exited.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOCK=$ON/aug11/.gpu_lease
LEASELOG=$ON/aug11/gpu_lease.log
OWNER=exp_26
WAIT=${1:-9000}
mkdir -p "$D/logs" "$ON/experiments/exp_26_ps2_land/logs"

stage() { printf '\n########## %s ##########\n' "$1"; date -u +%FT%TZ; }
now() { date -u +%s; }
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LEASELOG" 2>/dev/null; }

live_kfd() {
  local out="" p st wc
  for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
    st=$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null)
    wc=$(cat /proc/$p/wchan 2>/dev/null)
    if [ "$st" = "Z" ] || [ "$wc" = "exit_mm" ] || [ "$wc" = "do_exit" ]; then
      continue
    fi
    out="$out $p"
  done
  echo "$out" | wc -w
}

stage "the default that is about to be gated"
val=$(grep -A1 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp \
  | tail -1 | awk '{print $3}')
echo "PERSHAPE default = $val"
if [ "$val" != "2" ]; then
  echo "ABORTED: the source default is not 2; nothing to land"
  exit 1
fi

stage "acquire the lease (same lock dir as tools/gpu_lease.sh)"
t0=$(now)
got=0
while [ $(( $(now) - t0 )) -lt "$WAIT" ]; do
  if mkdir "$LOCK" 2>/dev/null; then
    echo "$OWNER" > "$LOCK/owner"
    now > "$LOCK/since"
    echo "$$" > "$LOCK/pid"
    got=1
    break
  fi
  held=$(cat "$LOCK/owner" 2>/dev/null || echo '?')
  since=$(cat "$LOCK/since" 2>/dev/null || echo 0)
  age=$(( $(now) - since ))
  if [ $(( (($(now) - t0)) % 300 )) -lt 20 ]; then
    echo "  waiting: lease held by '$held' for ${age}s"
  fi
  sleep 20
done
if [ "$got" != "1" ]; then
  echo "ABORTED: no lease within ${WAIT}s"
  exit 1
fi
note "ACQUIRE $OWNER (exp_26 inline acquire; shared tool was erroring)"
echo "lease acquired after $(( $(now) - t0 ))s"

# Having the lease is necessary but not sufficient: wait for the node to drain.
while [ "$(live_kfd)" != "0" ]; do
  if [ $(( $(now) - t0 )) -ge "$WAIT" ]; then
    echo "ABORTED: node never drained; releasing"
    rm -rf "$LOCK"; note "TIMEOUT $OWNER -- node dirty"
    exit 1
  fi
  echo "  draining: $(live_kfd) live KFD pid(s)"
  sleep 15
done
echo "node clean"

stage "full gate ladder on the shipped rule"
bash "$ON/tools/gate_ladder.sh" exp_26_ps2_land 2>&1 | tee "$D/logs/ladder_ps2.log"
lrc=${PIPESTATUS[0]}
echo "ladder exit: $lrc"

stage "M9 on the shipped rule, golden = the incumbent"
docker exec -w $ON/harness dhk-gemmrs timeout 5400 \
  python3 -u $D/m9_vs_incumbent.py 2>&1 | tee "$D/logs/m9_ps2.log"
echo "m9 exit: ${PIPESTATUS[0]}"

stage "release the lease"
rm -rf "$LOCK"
note "RELEASE $OWNER"
echo "released"

stage "done"
echo "CAMPAIGN5 COMPLETE (ladder $lrc)"
