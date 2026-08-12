#!/usr/bin/env bash
# exp_24 batch e24a: control FIRST and LAST, Mechanism A, the four throttle
# depths, plus two timestamps=1 points that measure plan_M3toM5 directly (the
# sensitive read for Mechanism A -- the end-to-end ratio's 1-sigma is 0.52%).
#
# SCREEN_SYNC=0 on purpose: the node checkout was fetched+reset and verified
# (HEAD cd7918a5, source-identical to exp_24's 520fe9c6, SRC_REV 24) moments
# before this launch. Another agent is pushing to the same branch tonight, so a
# mid-batch resync could swap the kernel under a half-finished sweep.
set -uo pipefail
SC=$HOME/.overnight-scripts
mkdir -p "$SC" "$HOME/overnight-scratch"
cp "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug11/tools/screen.sh" \
   "$SC/screen.sh"
CFG=$HOME/overnight-scratch/e24a.cfgs
cat > "$CFG" <<'EOF'
C=16,g=33,mode=12,flush_rows=16
C=16,g=97,mode=12,flush_rows=16
C=16,g=289,mode=12,flush_rows=16
C=16,g=545,mode=12,flush_rows=16
C=16,g=801,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16,timestamps=1
C=16,g=97,mode=12,flush_rows=16,timestamps=1
EOF
echo "=== config list"
cat -n "$CFG"
echo "=== node state at launch"
git -C "$HOME/Distributed-HipKittens" rev-parse --short HEAD
pgrep -af 'torchrun|mpirun' | head
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
echo "=== launching detached"
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SCREEN_SYNC=0 setsid nohup timeout 5400 \
  bash "$SC/screen.sh" e24a "$CFG" \
  > "$HOME/overnight-scratch/e24a_batch.out" 2>&1 < /dev/null &
echo "batch pid=$!"
sleep 20
echo "=== first 40 lines of batch output"
head -40 "$HOME/overnight-scratch/e24a_batch.out"
exit 0
