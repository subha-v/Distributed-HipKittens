#!/usr/bin/env bash
# exp_10 probe 11 -- NO GPU (grep only; safe to run beside a GPU job).
#
# INTEGRITY CHECK. launch_triton_kernel line 1464-1465:
#     if (M, N, local_K) not in __conf:
#         return origin((a, b, bias))
# i.e. for any shape rank-1 has no tuned config for, it SILENTLY falls back to a
# torch implementation. A benchmark number from that path is torch's, not
# rank-1's, and must never be reported as rank-1's score. Establish, for each of
# the six graded shapes, whether rank-1 runs its own kernel or the fallback.
set -uo pipefail

RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== A. origin(): what is the fallback? ====="
awk 'NR>=1704 && NR<=1740 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== B. __conf / online_config / online_config_group definitions ====="
grep -nE '^__conf|^online_config|^online_config_group|__conf *=|online_config *=' "$RANK1" | head

echo
echo "===== C. enumerate the keys, and check the six graded shapes ====="
docker exec -w /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1 \
  -e PATH=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin \
  -e PYTHONPATH=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/compat:/usr/local/lib/python3.10/dist-packages \
  dhk-gemmrs bash -lc 'python3 -u -c "
import submission as s
SHAPES = [(64,7168,18432),(512,4096,12288),(2048,2880,2880),
          (4096,4096,4096),(8192,4096,14336),(8192,8192,29568)]
conf = getattr(s, \"__conf\", None)
if conf is None:
    conf = getattr(s, \"_submission__conf\", None)
print(\"__conf keys:\")
try:
    for k in sorted(conf): print(\"   \", k)
except Exception as e:
    print(\"   could not enumerate:\", e)
print()
print(\"online_config keys:\")
for k in sorted(s.online_config): print(\"   \", k)
print()
print(\"online_config_group keys:\", sorted(s.online_config_group))
print()
print(\"=== per graded shape: own kernel or SILENT torch fallback? ===\")
for (m,n,k) in SHAPES:
    lk = k//8
    key = (m,n,lk)
    in_conf = key in conf if conf is not None else None
    in_oc = key in s.online_config
    g = (m//8)*n
    in_g = g in s.online_config_group
    verdict = \"OWN KERNEL\" if in_conf else \"*** TORCH FALLBACK ***\"
    print(f\"  m={m:<5} n={n:<5} k={k:<6} local_k={lk:<5} \"
          f\"__conf={in_conf} online_config={in_oc} \"
          f\"group[{g}]={in_g}   -> {verdict}\")
"' 2>&1 | tail -60
echo "===== DONE p11 ====="
