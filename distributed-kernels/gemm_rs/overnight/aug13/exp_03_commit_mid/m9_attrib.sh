#!/usr/bin/env bash
# exp_03 M9-fault attribution. Node HOST. Context: the ladder's M9 run on the
# cmid module hit "Memory access fault by GPU node-7" and wedged in the GPU
# coredump handler (TERM could not land; kill -9 recovered it). The run's
# stdout was lost to python buffering, so WHERE it faulted is unknown - the
# sweep interleaves the GOLDEN module (gemm_rs_mi300x_e3base), the candidate,
# and the CTRL_PUBLISH_EARLY control every epoch.
#
# This script discriminates arm vs environment:
#   canary  M3 all-17 on the production module (node health after the fault)
#   run 1   M9 on gemm_rs_mi300x_cmid   (candidate, full scale, python -u so
#           the faulting case/epoch is visible this time)
#   run 2   M9 on gemm_rs_mi300x_cmid0  (base twin, same fresh build family)
#
# Interpretation, pre-registered:
#   1 faults / 2 green  -> arm-specific; cmid is REJECTED on correctness
#                          grounds regardless of any speed story.
#   1 faults / 2 faults -> environment/instrument (golden or harness);
#                          cmid exonerated pending an M9 infrastructure fix.
#   1 green  / 2 green  -> nonreproducible; rerun 1 before concluding anything.
# Wedge-proofing: timeout -k so KILL follows TERM; post-run container pkill +
# kfd wait so one wedge cannot block the next run.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
T=$ON/tools
EXP=$ON/aug13/exp_03_commit_mid
LOGS=$EXP/logs
NAME=dhk-gemmrs

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
. "$T/kfd_live.sh"
mkdir -p "$LOGS"

bash "$T/gpu_lease.sh" acquire aug13_m9attr 7200 || { LOG "ABORT: no lease"; exit 1; }
trap 'bash "$T/gpu_lease.sh" release aug13_m9attr' EXIT
bash "$T/set_clocks.sh" pin 1900 | tail -2

cleanup_wedge() { # kill any leftover m9 in the container, then wait clean
    docker exec "$NAME" pkill -9 -f m9_stale_slot 2>/dev/null
    sleep 3
    kfd_wait_clean 30 10
}

LOG "canary: M3 all-17 on the production module"
kfd_wait_clean 30 10 || { LOG "ABORT: node not clean before canary"; exit 1; }
docker exec -w "$ON/harness" "$NAME" bash -c "timeout -k 60 900 python3 m3_correctness.py all" \
    > "$LOGS/attr_m3_prod.log" 2>&1
if grep -q "17/17 shapes PASSED" "$LOGS/attr_m3_prod.log"; then
    LOG "canary M3(prod) green - node healthy after the fault"
else
    LOG "ABORT: CANARY M3(prod) FAILED - node health suspect, no attribution possible"
    tail -5 "$LOGS/attr_m3_prod.log"
    exit 1
fi

run_m9() { # run_m9 <n> <module-suffix>
    local n=$1 arm=$2
    LOG "M9 attribution run $n: $arm (full scale, unbuffered)"
    kfd_wait_clean 30 10 || { LOG "ABORT: node not clean before run $n"; exit 1; }
    docker exec -w "$ON/harness" -e HK_KERNEL_MODULE=gemm_rs_mi300x_$arm "$NAME" \
        bash -c "timeout -k 60 1500 setsid python3 -u m9_stale_slot.py" \
        > "$LOGS/attr_m9_${n}_${arm}.log" 2>&1
    local rc=$?
    if grep -q "Memory access fault" "$LOGS/attr_m9_${n}_${arm}.log"; then
        LOG "M9 run $n ($arm): MEMORY ACCESS FAULT (exit=$rc); last sweep marker:"
        grep -E "^-- |epoch " "$LOGS/attr_m9_${n}_${arm}.log" | tail -3 | sed 's/^/    /'
        cleanup_wedge
    else
        LOG "M9 run $n ($arm): exit=$rc"
        tail -2 "$LOGS/attr_m9_${n}_${arm}.log" | sed "s/^/    [$arm] /"
    fi
}

run_m9 1 cmid
run_m9 2 cmid0
LOG "ATTRIBUTION DONE (runs 1-2; see logs/attr_m9_*.log)"
