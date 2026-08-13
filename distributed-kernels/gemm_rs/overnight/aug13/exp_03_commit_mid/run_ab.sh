#!/usr/bin/env bash
# exp_03: full gate ladder + paired M7 for the COMMIT_MID arm. Node HOST.
# Order per plan.md: M1 build -> M2 census/placement -> lds_race -> M3 -> M4
# -> M5 -> M9 -> M7 paired (shapes 5,6 x fwd,rev orders).
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
T=$ON/tools
EXP=$ON/aug13/exp_03_commit_mid
LOGS=$EXP/logs
NAME=dhk-gemmrs

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
run() { docker exec "$NAME" bash -c "$1"; }
die() { LOG "ABORT: $*"; exit 1; }

. "$T/kfd_live.sh"
mkdir -p "$LOGS"

LOG "acquiring GPU lease"
bash "$T/gpu_lease.sh" acquire aug13_exp03 7200 || die "no lease"
trap 'bash "$T/gpu_lease.sh" release aug13_exp03' EXIT
bash "$T/set_clocks.sh" pin 1900 | tail -2

# The documented trap: harness/submission.py is a node-only COPY.
run "cp $ON/harness/hk_submission.py $ON/harness/submission.py"

LOG "M1: build arms"
run "bash $EXP/build_arms.sh" > "$LOGS/m1_build.log" 2>&1 || die "build failed (see logs/m1_build.log)"
grep -E "OK|FAIL" "$LOGS/m1_build.log"
grep -q "FAIL" "$LOGS/m1_build.log" && die "an arm failed to build"

LOG "M2: census + placement assert"
run "bash $EXP/census.sh" > "$LOGS/m2_census.log" 2>&1
grep -E "RATCHET|PLACEMENT|FAIL" "$LOGS/m2_census.log"
grep -q "PLACEMENT GATE: PASS" "$LOGS/m2_census.log" || die "placement gate failed"

LOG "lds_race_check on cmid ISA"
bash $ON/experiments/exp_03_mainloop/lds_race_check.sh "$EXP/isa/cmid/kernel.s" \
    > "$LOGS/lds_race_cmid.log" 2>&1 || true
tail -5 "$LOGS/lds_race_cmid.log"

LOG "M3 on cmid module (17 shapes, both tolerances)"
kfd_wait_clean 30 10 || die "node not clean before M3"
docker exec -w "$ON/harness" -e HK_KERNEL_MODULE=gemm_rs_mi300x_cmid "$NAME" \
    bash -c "timeout 1200 python3 m3_correctness.py all" > "$LOGS/m3_cmid.log" 2>&1 \
    || die "M3(cmid) failed"
grep -q "17/17 shapes PASSED" "$LOGS/m3_cmid.log" || die "M3(cmid) not 17/17"
LOG "M3(cmid) green"

LOG "M4 negative controls (control module)"
kfd_wait_clean 30 10 || die "node not clean before M4"
docker exec -w "$ON/harness" "$NAME" bash -c "timeout 900 python3 m4_controls.py" \
    > "$LOGS/m4.log" 2>&1 || die "M4 failed"
tail -3 "$LOGS/m4.log"

LOG "M5 soak (600 epochs) on cmid"
kfd_wait_clean 30 10 || die "node not clean before M5"
docker exec -w "$ON/harness" -e HK_KERNEL_MODULE=gemm_rs_mi300x_cmid "$NAME" \
    bash -c "timeout 1500 python3 m5_soak.py 600 512 4096 12288 1 1" \
    > "$LOGS/m5_cmid.log" 2>&1 || die "M5(cmid) failed"
tail -3 "$LOGS/m5_cmid.log"

LOG "M9 stale-slot on cmid (expected: unchanged/null)"
kfd_wait_clean 30 10 || die "node not clean before M9"
docker exec -w "$ON/harness" -e HK_KERNEL_MODULE=gemm_rs_mi300x_cmid "$NAME" \
    bash -c "timeout 1800 python3 m9_stale_slot.py" > "$LOGS/m9_cmid.log" 2>&1 \
    || { tail -8 "$LOGS/m9_cmid.log"; die "M9(cmid) failed"; }
tail -3 "$LOGS/m9_cmid.log"

LOG "M7: paired campaign, shapes 5 and 6, both allocation orders"
ROUNDS=${ROUNDS:-42}
for idx in 4 5; do
    for order in fwd rev; do
        kfd_wait_clean 30 10 || die "node not clean before M7 idx=$idx $order"
        LOG "  ab_cmid shape_idx=$idx order=$order rounds=$ROUNDS"
        docker exec -w "$EXP" -e AB_ORDER=$order -e AB_OUT="$EXP/logs/ab_s${idx}_${order}" \
            "$NAME" bash -c "timeout 2400 setsid python3 ab_cmid_mp.py $idx $ROUNDS 25 15 5 134$((10 + idx))" \
            > "$LOGS/ab_s${idx}_${order}.out" 2>&1 \
            || { tail -8 "$LOGS/ab_s${idx}_${order}.out"; die "M7 idx=$idx $order failed"; }
        tail -4 "$LOGS/ab_s${idx}_${order}.out"
    done
done

LOG "report"
python3 "$EXP/report_ab.py" "$LOGS" | tee "$LOGS/report.txt"
LOG "DONE"
