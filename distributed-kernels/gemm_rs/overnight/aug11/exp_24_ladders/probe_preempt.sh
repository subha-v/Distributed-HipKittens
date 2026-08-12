#!/usr/bin/env bash
# I held the lease and was SIGTERM'd at 06:40:53 with 5/6 shapes done. Find out by
# whom and why before resuming, because the answer changes what I do: a `steal` by
# another agent means I re-queue politely, whereas an OOM-killer or an operator
# means something about my run needs fixing first.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "===== gpu_lease.log (who did what, in order) ====="
tail -40 "$ON/aug11/gpu_lease.log" 2>/dev/null || echo "(no lease log)"

echo
echo "===== was it a steal? ====="
grep -iE 'steal|reap|preempt' "$ON/aug11/gpu_lease.log" 2>/dev/null | tail -10 || echo "no steal/reap entries"

echo
echo "===== kernel OOM / oom-killer? ====="
dmesg 2>/dev/null | tail -25 | grep -iE 'oom|kill|memory' || echo "(dmesg unreadable or nothing relevant)"

echo
echo "===== who is on the node NOW ====="
if [ -f "$D/toolsnap/gpu_lease.sh" ]; then bash "$D/toolsnap/gpu_lease.sh" status; fi
ps -eo pid,ppid,stat,etime,user,cmd --sort=start_time 2>/dev/null \
  | grep -E 'eval\.py|ladder_mp|m[0-9]_|exp_2|python3 -u' | grep -v grep | tail -20 || echo "no python jobs"

echo
echo "===== what survived: instrument A per-shape rank files ====="
for i in 0 1 2 3 4 5; do
  n=$(ls "$D/raw/ladder"/lad_s${i}.rank*.json 2>/dev/null | wc -l)
  echo "  shape idx $i : $n/8 rank files"
done

echo
echo "===== are the 5 completed shapes internally complete? ====="
docker exec dhk-gemmrs python3 - <<'PY'
import glob, json, os
D="/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders/raw/ladder"
for i in range(6):
    fs=sorted(glob.glob(f"{D}/lad_s{i}.rank*.json"))
    if not fs: continue
    ok=[]
    for f in fs:
        try:
            d=json.load(open(f))
            arms=sorted(d.get("arms",{}))
            protos={a:sorted(d["arms"][a].keys()) for a in arms}
            ok.append((os.path.basename(f), len(arms), protos.get(arms[0]) if arms else None))
        except Exception as e:
            ok.append((os.path.basename(f), f"BAD {e}", None))
    arms_n={x[1] for x in ok}
    print(f"shape {i}: {len(fs)} files, arms per file {arms_n}, protos {ok[0][2]}")
PY

echo
echo "===== full_run.log: the last thing before Terminated ====="
grep -nE 'Terminated|SIGTERM|TRAP|shape index|wall=' "$D/logs/full_run.log" | tail -20
