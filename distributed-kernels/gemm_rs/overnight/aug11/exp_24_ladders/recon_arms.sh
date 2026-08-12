#!/usr/bin/env bash
# Read-only recon for the paired ours/ours_prev test. No GPU work.
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
echo "=== harness submission files ==="
ls -la "$ON/harness"/*submission* 2>/dev/null
echo
echo "=== does harness/submission.py select the kernel module? ==="
grep -n 'HK_KERNEL_MODULE\|import\|custom_kernel' "$ON/harness/submission.py" 2>/dev/null | head -20
echo
echo "=== exp_26 build_arms.sh spec_flags ==="
sed -n '1,45p' "$ON/aug11/exp_26_release_pershape/build_arms.sh"
echo
echo "=== prebuilt arm modules present ==="
ls -la "$ON/harness/build"/gemm_rs_mi300x_ps*.so 2>/dev/null
echo
echo "=== exported init symbols of the two arms ==="
for m in ps0 ps2; do
  f="$ON/harness/build/gemm_rs_mi300x_$m.so"
  [ -f "$f" ] && echo "  $m: $(nm -D --defined-only "$f" 2>/dev/null | grep -o 'PyInit_[A-Za-z0-9_]*' | head -2 | tr '\n' ' ')"
done
