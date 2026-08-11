#!/usr/bin/env bash
# Reap OUR stale GPU processes before any timing run.
#
# Learned 2026-08-11 03:15 PT, two traps in one:
#
#  1. The previous session's rank-1 attempt left three multiprocessing.spawn
#     workers alive inside dhk-eval, spinning at 99% CPU and holding GPUs
#     2/6/7 at 100% with 5.5 GB VRAM each for 7+ hours. A
#     `ps | grep 'eval\.py|mp_smoke'` did NOT see them: spawned children carry
#     only `from multiprocessing.spawn import spawn_main` on their cmdline.
#     Match on the KFD process list, never on a hopeful cmdline pattern.
#
#  2. Those processes run as ROOT inside dhk-eval while we are uid 15523 on the
#     host, so a host-side `kill` fails with EPERM and prints nothing. Signals
#     must be sent from inside the container, where the PID namespace also
#     differs -- so match by pattern with pkill rather than by host PID.
#
# SIGTERM only by default. This kernel uses HIP IPC and SIGKILL can leak IPC
# mappings and wedge the node. `reap_stale.sh force` escalates, but only after
# TERM has been given time to work.
set -u
mode=${1:-term}
PAT='spawn_main|resource_tracker|eval\.py|mp_smoke|torchrun|iris'

show() { rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'; }

echo "===== KFD processes before ====="
show

before=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
if [ "$before" = "0" ]; then echo "NOTHING TO REAP"; exit 0; fi

# Any KFD pid that is NOT in one of our two containers is another tenant's and
# is strictly off-limits.
for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
  mine=no
  for c in dhk-eval dhk-gemmrs; do
    docker top "$c" 2>/dev/null | awk '{print $2}' | grep -qx "$p" && mine=$c
  done
  [ "$mine" = "no" ] && echo "  !! pid $p is NOT ours -- leaving it alone" \
                     || echo "  pid $p is ours (container $mine)"
done

for c in dhk-eval dhk-gemmrs; do
  echo "===== SIGTERM inside $c ====="
  docker exec "$c" pkill -TERM -f "$PAT" 2>&1 && echo "  pkill issued" || echo "  (no match)"
done
sleep 12

left=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids remaining after TERM: $left"

if [ "$left" != "0" ] && [ "$mode" = "force" ]; then
  for c in dhk-eval dhk-gemmrs; do
    echo "===== SIGKILL inside $c ====="
    docker exec "$c" pkill -KILL -f "$PAT" 2>&1 && echo "  pkill -KILL issued" || echo "  (no match)"
  done
  sleep 10
  left=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  echo "KFD pids remaining after KILL: $left"
fi

echo
echo "===== KFD processes after ====="
show
echo "===== utilization after ====="
sleep 5
rocm-smi 2>&1 | tail -13
echo "===== DONE ====="
[ "$left" = "0" ] && echo "NODE CLEAN" || echo "NODE STILL DIRTY"
