#!/usr/bin/env bash
# Who is using the GPUs right now? We hold the exclusive lease but must not
# time against a contended node, and must not SIGKILL another tenant's work.
set -u

echo "===== rocm-smi pids ====="
rocm-smi --showpids 2>&1 | head -40

echo
echo "===== processes holding /dev/kfd ====="
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  if ls -l /proc/$p/fd 2>/dev/null | grep -q kfd; then
    printf '%-8s %-12s %s\n' "$p" "$(stat -c %U /proc/$p 2>/dev/null)" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-140)"
  fi
done

echo
echo "===== container -> what is inside ====="
for c in ddt-o0-4df4e85f dhk-gemmrs dhk-eval; do
  echo "-- $c --"
  docker top "$c" 2>/dev/null | head -12 || echo "   (cannot inspect)"
done

echo
echo "===== per-GPU utilization, second sample ====="
sleep 5
rocm-smi 2>&1 | tail -14

echo "===== DONE ====="
