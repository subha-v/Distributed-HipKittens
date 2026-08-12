#!/usr/bin/env bash
# READ-ONLY. rank3's quick-run stderr was 75 bytes against 20 for every other
# rank; a per-rank asymmetry is exactly the kind of thing that is nothing 90% of
# the time and a real bug the other 10%, so look rather than assume.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
for r in 0 3; do
  echo "########## rank$r stderr ($(wc -c < "$D/raw/ladder_quick/lad_s1.rank$r.stderr") bytes) ##########"
  cat -A "$D/raw/ladder_quick/lad_s1.rank$r.stderr" 2>&1 | sed -n '1,10p'
done
echo
echo "########## syntax after the warm/provenance fixes ##########"
bash -n "$D/run_ladders.sh" && echo "OK run_ladders.sh"
docker exec dhk-gemmrs bash -lc "cd $D && python3 -m py_compile ladder_mp.py ladders.py && echo 'OK python'" 2>&1 | tail -2
echo
echo "########## node state (must still be clean; NOT launching anything) ##########"
rocm-smi --showpids 2>&1 | sed -n '3,10p'
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
rocm-smi --showperflevel 2>&1 | grep -c perf_determinism
echo "(count above = GPUs still pinned in perf_determinism, expect 8)"
