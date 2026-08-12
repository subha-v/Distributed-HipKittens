#!/usr/bin/env bash
# Shared liveness-aware KFD accounting. SOURCE this; do not execute it.
#
#   . "$ON/tools/kfd_live.sh"
#   kfd_live_count   # pids that can actually dispatch GPU work
#   kfd_stale_list   # pids holding a KFD entry that are already dead
#   kfd_wait_clean <tries> <sleep_s>   # drain wait that cannot hang on a corpse
#
# WHY THIS EXISTS, and it cost the figure queue about 40 minutes.
#
# `rocm-smi --showpids` lists a process until the driver finishes reclaiming its
# VRAM. A process that has CRASHED and wedged in `exit_mm` therefore keeps
# appearing indefinitely: it has no address space, it cannot launch a kernel, and
# it cannot be signalled (SIGTERM and SIGKILL are both no-ops against a task in
# exit_mm, and attempting them is how a stale entry escalates into a genuinely
# wedged node).
#
# Every drain check in this tree counted such a corpse as a live tenant, so a
# campaign that had already finished one shape sat in "waiting for kfd drain
# (fds=1)" forever, while all 8 GPUs passed a live 4096^3 bf16 matmul with 190 of
# 192 GiB free at idle temperature and power.
#
# The three cheap signals that separate a corpse from a running job:
#   * /proc/<pid>/wchan == exit_mm (or do_exit), or State: Z
#   * utime+stime in /proc/<pid>/stat frozen across a sampling interval
#   * CU OCCUPANCY 0 in rocm-smi --showpids for minutes
# And the decisive one: just try a small matmul on all 8 devices.
#
# Callers must still refuse to run when a LIVE foreign job is present. This only
# stops them waiting on the dead.
set -uo pipefail

_kfd_all() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/{print $1}'; }

_kfd_is_dead() {
  local p=$1 st wc
  [ -d "/proc/$p" ] || return 0                      # gone entirely
  st=$(awk '/^State:/{print $2}' "/proc/$p/status" 2>/dev/null)
  wc=$(cat "/proc/$p/wchan" 2>/dev/null)
  [ "$st" = "Z" ] && return 0
  [ "$wc" = "exit_mm" ] && return 0
  [ "$wc" = "do_exit" ] && return 0
  return 1
}

kfd_stale_list() {
  local p out=""
  for p in $(_kfd_all); do _kfd_is_dead "$p" && out="$out $p"; done
  echo "${out# }"
}

kfd_live_list() {
  local p out=""
  for p in $(_kfd_all); do _kfd_is_dead "$p" || out="$out $p"; done
  echo "${out# }"
}

kfd_live_count() { local l; l=$(kfd_live_list); [ -z "$l" ] && echo 0 || echo "$l" | wc -w; }

# Drain wait that reports the stale set instead of blocking on it.
kfd_wait_clean() {
  local tries=${1:-30} nap=${2:-10} i n s
  s=$(kfd_stale_list)
  [ -n "$s" ] && echo "  NOTE: ignoring dead KFD pid(s) that cannot dispatch: $s"
  for ((i=0; i<tries; i++)); do
    n=$(kfd_live_count)
    if [ "$n" = "0" ]; then
      [ "$i" -gt 0 ] && echo "  node clean after $((i*nap))s"
      return 0
    fi
    [ "$i" = "0" ] && echo "  $n LIVE KFD pid(s) -- waiting for the node to drain"
    sleep "$nap"
  done
  echo "  ABORT: $(kfd_live_count) LIVE KFD pid(s) after $((tries*nap))s"
  rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
  return 1
}
