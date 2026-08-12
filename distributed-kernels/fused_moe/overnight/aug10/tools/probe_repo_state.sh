set -u
cd "$HOME/Distributed-HipKittens" || exit 1
echo "=== node git ==="
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git config core.autocrlf || echo "autocrlf unset"
echo "=== node blob hashes (content-normalized) ==="
for f in k0pf6gm_device_tile_mps.hip moe_mps_adapter.cuh n2_phase2_gm_mps.cpp moe_host_abi.hpp k0pf6gm_device_tile.hip; do
  printf '%s %s\n' "$(git rev-parse "HEAD:distributed-kernels/fused_moe/$f")" "$f"
done
echo "=== node worktree dirty? ==="
git status --porcelain | head -20
echo "=== remote head ==="
git ls-remote origin refs/heads/codex/distributed-hipkittens-scaffold
echo "=== amd-master ==="
git -C "$HOME/amd-master" rev-parse --abbrev-ref HEAD
git -C "$HOME/amd-master" rev-parse HEAD
git -C "$HOME/amd-master" status --porcelain | head -20
echo "=== harness dir ==="
ls -la "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/"
echo "=== prior smoke log tail ==="
ls -la "$HOME/k0-mok-synthetic-mps-smoke/" 2>&1 | head
tail -60 "$HOME/k0-mok-synthetic-mps-smoke/run1.log" 2>&1 | tail -60
