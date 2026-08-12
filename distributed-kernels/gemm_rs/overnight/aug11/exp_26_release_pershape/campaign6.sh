#!/usr/bin/env bash
# Landing run, third attempt. campaign5 got M9 through cleanly and then the
# ladder exited 1 in under a minute: `gate_ladder.sh`'s M0 and M7 preflights
# count EVERY pid `rocm-smi --showpids` reports, and this experiment's own M9
# process from the 10:05Z fault is still resident in `D`/`exit_mm`, holding VRAM
# it will never release and dispatching nothing. It cannot be removed, so the
# preflight can never read 0 again on this node until it clears.
#
# The ladder is not forked. It is transformed at run time: the pid-count
# expression is replaced by a `live_kfd` that skips processes in state Z or with
# wchan exit_mm/do_exit -- exactly the predicate the shared lease tool adopted
# for the same reason. Every other line, every gate, every assertion and the
# order are byte-identical to tools/gate_ladder.sh, and the transform is
# asserted to have applied before anything runs, so a silently unpatched copy
# cannot masquerade as a pass.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOCK=$ON/aug11/.gpu_lease
LEASELOG=$ON/aug11/gpu_lease.log
OWNER=exp_26
WAIT=${1:-9000}
LADDER=/tmp/gate_ladder_stalesafe.sh
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

stage "build the stale-safe ladder from the shared one"
{
  echo '#!/usr/bin/env bash'
  echo 'live_kfd() {'
  echo '  out=""'
  echo '  for p in $(rocm-smi --showpids 2>/dev/null | awk "/^[0-9]+/{print \$1}"); do'
  echo '    st=$(awk "/^State:/{print \$2}" /proc/$p/status 2>/dev/null)'
  echo '    wc=$(cat /proc/$p/wchan 2>/dev/null)'
  echo '    case "$st$wc" in Z*|*exit_mm|*do_exit) continue;; esac'
  echo '    out="$out $p"'
  echo '  done'
  echo '  echo "$out" | wc -w'
  echo '}'
  # Match the whole assignment line rather than the pid expression: the
  # expression is full of characters that need escaping in both bash and sed
  # (|, $, {}, /), and the first attempt at escaping it matched nothing. The
  # assertion below is what makes a botched pattern a refusal instead of an
  # unpatched gate reporting a pass.
  tr -d '\r' < "$ON/tools/gate_ladder.sh" \
    | sed 's|^n=.*rocm-smi.*wc -l)$|n=$(live_kfd)|'
} > "$LADDER"
patched=$(grep -c 'n=$(live_kfd)' "$LADDER")
echo "preflights patched: $patched (expect 2)"
if [ "$patched" != "2" ]; then
  echo "ABORTED: the ladder transform did not apply; refusing to run an unpatched or half-patched gate"
  grep -n 'showpids' "$LADDER"
  exit 1
fi
echo "stale pids currently reported by rocm-smi but not dispatching:"
rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print "  " $1 " " $2}'
echo "live_kfd says: $(live_kfd)"

stage "acquire the lease (same lock dir as tools/gpu_lease.sh)"
t0=$(now)
got=0
while [ $(( $(now) - t0 )) -lt "$WAIT" ]; do
  if mkdir "$LOCK" 2>/dev/null; then
    echo "$OWNER" > "$LOCK/owner"; now > "$LOCK/since"; echo "$$" > "$LOCK/pid"
    got=1; break
  fi
  echo "  waiting: lease held by '$(cat "$LOCK/owner" 2>/dev/null)'"
  sleep 20
done
if [ "$got" != "1" ]; then echo "ABORTED: no lease"; exit 1; fi
note "ACQUIRE $OWNER"
echo "lease acquired after $(( $(now) - t0 ))s"

while [ "$(live_kfd)" != "0" ]; do
  if [ $(( $(now) - t0 )) -ge "$WAIT" ]; then
    echo "ABORTED: node never drained"; rm -rf "$LOCK"; note "TIMEOUT $OWNER"; exit 1
  fi
  echo "  draining: $(live_kfd) live KFD pid(s)"; sleep 15
done

stage "full gate ladder on the shipped rule (PERSHAPE=2)"
bash "$LADDER" exp_26_ps2_land 2>&1 | tee "$D/logs/ladder_ps2.log"
lrc=${PIPESTATUS[0]}
echo "ladder exit: $lrc"

stage "release the lease"
rm -rf "$LOCK"; note "RELEASE $OWNER"; echo released

stage "done"
echo "CAMPAIGN6 COMPLETE (ladder $lrc)"
exit $lrc
