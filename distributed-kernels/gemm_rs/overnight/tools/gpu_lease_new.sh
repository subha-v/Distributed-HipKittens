#!/usr/bin/env bash
# Mutual exclusion for the 8-GPU node, so "one GPU job of ours at a time" is
# enforced by the filesystem instead of by the orchestrator's timing.
#
# Why: several agents run figure experiments in parallel tonight. They do CPU
# builds concurrently, which is fine, but two overlapping GPU campaigns would
# corrupt both sets of numbers -- and the corruption is invisible, because each
# arm still produces plausible microseconds. A dirty-node check alone is not
# enough: two agents can both see a clean node in the same second and both
# launch.
#
# The lock is a DIRECTORY, because mkdir is atomic on the shared filesystem
# whereas `[ -f ] && touch` is not.
#
#   gpu_lease.sh acquire <owner> [wait_s]   # default wait 1800s
#   gpu_lease.sh release <owner>
#   gpu_lease.sh status
#   gpu_lease.sh steal <owner> <reason>     # last resort, logs the preemption
#
# Stale locks: a lease whose PID is gone AND whose node shows no KFD pids is
# reaped automatically after STALE_S. Nothing here ever SIGKILLs anything --
# leaked HIP IPC mappings wedge this node, so reclaiming is SIGTERM-only and is
# left to a human/agent decision via `steal`.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOCK=$ON/aug11/.gpu_lease
LOG=$ON/aug11/gpu_lease.log
STALE_S=${STALE_S:-5400}

now() { date -u +%s; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

proc_state() { sed -n 's/.*) \([A-Z]\) .*/\1/p' "/proc/$1/stat" 2>/dev/null; }

# A KFD pid that has already left user space cannot issue GPU work ever again, so
# counting it as "draining" is a false positive -- and an expensive one: on
# 2026-08-12 a harness control wedged in amdgpu/KFD address-space teardown (state
# D, wchan exit_mm, zero CU occupancy, VRAM still mapped, CPU time flat) aborted
# both exp_21's and exp_24's acquire after their full 300 s drain window. Such a
# task cannot even be reclaimed: D state ignores SIGTERM, and SIGKILL is forbidden
# here anyway. So the drain check counts only pids that are still ALIVE in the
# sense that matters -- able to launch a kernel.
zombie_kfd_pids() {
  local p st wc
  for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
    [ -d "/proc/$p" ] || continue
    st=$(proc_state "$p")
    wc=$(cat "/proc/$p/wchan" 2>/dev/null)
    [ "$st" = "Z" ] && { echo "$p"; continue; }
    [ "$st" = "D" ] && [ "$wc" = "exit_mm" ] && echo "$p"
  done
}
live_kfd_pids() {
  local dead p
  dead=" $(zombie_kfd_pids | tr '\n' ' ') "
  for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
    case "$dead" in *" $p "*) continue;; esac
    echo "$p"
  done
}
kfd_pids() { live_kfd_pids | wc -l; }

note() { echo "$(stamp) $*" >> "$LOG" 2>/dev/null; }

read_owner() { cat "$LOCK/owner" 2>/dev/null || echo "?"; }
read_since() { cat "$LOCK/since" 2>/dev/null || echo 0; }

reap_if_stale() {
  [ -d "$LOCK" ] || return 1
  local age owner
  age=$(( $(now) - $(read_since) ))
  owner=$(read_owner)
  if [ "$age" -gt "$STALE_S" ] && [ "$(kfd_pids)" = "0" ]; then
    note "REAP stale lease held by '$owner' for ${age}s with a clean node"
    echo "  reaping stale lease from '$owner' (${age}s, node clean)"
    rm -rf "$LOCK"
    return 0
  fi
  return 1
}

case "${1:-status}" in
  acquire)
    OWNER=${2:?owner required}
    WAIT=${3:-1800}
    mkdir -p "$ON/aug11"
    t0=$(now)
    while :; do
      if mkdir "$LOCK" 2>/dev/null; then
        echo "$OWNER" > "$LOCK/owner"
        now > "$LOCK/since"
        echo "$$" > "$LOCK/pid"
        z=$(zombie_kfd_pids | tr '\n' ' ')
        if [ -n "${z// /}" ]; then
          echo "  NOTE: ignoring KFD pid(s) [$z] wedged in address-space teardown"
          echo "        (state D/Z, wchan exit_mm): they hold VRAM but cannot launch."
          note "IGNORE-ZOMBIE $OWNER -- pids [$z] in teardown"
        fi
        # Having the lease is necessary but not sufficient: a foreign tenant may
        # still be on the GPUs. Wait for drain rather than aborting on the first
        # sample -- a previous run's last worker lingers for tens of seconds
        # after its parent returns, and aborting there wastes an invocation.
        for i in $(seq 1 30); do
          n=$(kfd_pids)
          [ "$n" = "0" ] && break
          [ "$i" = "1" ] && echo "  lease held by '$OWNER'; $n live KFD pid(s) still draining"
          if [ "$i" = "30" ]; then
            echo "  ABORT: node still dirty after 300s; releasing the lease"
            note "ABORT $OWNER -- node dirty after 300s (live pids: $(live_kfd_pids | tr '\n' ' '))"
            rm -rf "$LOCK"
            exit 1
          fi
          sleep 10
        done
        note "ACQUIRE $OWNER"
        echo "GPU lease ACQUIRED by '$OWNER' after $(( $(now) - t0 ))s; node clean"
        exit 0
      fi
      reap_if_stale && continue
      if [ $(( $(now) - t0 )) -ge "$WAIT" ]; then
        echo "TIMEOUT after ${WAIT}s; lease held by '$(read_owner)' for $(( $(now) - $(read_since) ))s"
        note "TIMEOUT $OWNER waiting on $(read_owner)"
        exit 2
      fi
      sleep 15
    done
    ;;
  release)
    OWNER=${2:?owner required}
    if [ ! -d "$LOCK" ]; then echo "no lease held"; exit 0; fi
    held=$(read_owner)
    if [ "$held" != "$OWNER" ]; then
      echo "REFUSING: lease is held by '$held', not '$OWNER'. Use 'steal' and say why."
      exit 1
    fi
    rm -rf "$LOCK"
    note "RELEASE $OWNER"
    echo "GPU lease released by '$OWNER'"
    ;;
  steal)
    OWNER=${2:?owner required}; REASON=${3:?reason required}
    held=$(read_owner)
    note "STEAL by $OWNER from $held -- $REASON"
    rm -rf "$LOCK"
    echo "lease taken from '$held' by '$OWNER': $REASON"
    echo "DISCLOSE this preemption in the experiment's result.md."
    ;;
  status)
    if [ -d "$LOCK" ]; then
      echo "lease HELD by '$(read_owner)' for $(( $(now) - $(read_since) ))s (pid $(cat "$LOCK/pid" 2>/dev/null))"
    else
      echo "lease FREE"
    fi
    echo "KFD pid count: $(kfd_pids)"
    z=$(zombie_kfd_pids | tr '\n' ' ')
    [ -n "${z// /}" ] && echo "  ignored (wedged in teardown, hold VRAM, cannot launch): $z"
    ;;
  *) echo "usage: gpu_lease.sh {acquire <owner> [wait_s]|release <owner>|steal <owner> <reason>|status}"; exit 64;;
esac
