#!/usr/bin/env bash
echo "=== pre-flight ==="
echo "HEAD: $(git -C ~/Distributed-HipKittens rev-parse --short HEAD)  (expected 0b82cd19)"
pgrep -af 'torchrun|mpirun' || echo "pgrep torchrun/mpirun: EMPTY"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "no stale lock"
echo
cat > ~/overnight-scratch/a11base.cfgs <<'EOF'
# Deliverable 2: confirmed baseline.
# 1+2 are the SAME point twice -- that pair IS the run-to-run noise band.
C=16,g=33,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
# 3 is the previous ratchet, a cross-check that the tree is what we think.
C=64,g=1,mode=2,flush_rows=16
EOF
echo "=== cfg list ==="; cat ~/overnight-scratch/a11base.cfgs
rm -f ~/overnight-scratch/screen_a11base.csv ~/overnight-scratch/screen_a11base.out
cd ~
setsid nohup timeout 5400 bash ~/.overnight-scripts/screen.sh a11base \
  ~/overnight-scratch/a11base.cfgs \
  > ~/overnight-scratch/screen_a11base.out 2>&1 < /dev/null &
sleep 8
echo "=== first 30 lines of driver output ==="
head -30 ~/overnight-scratch/screen_a11base.out
echo "=== running procs ==="
pgrep -af 'screen.sh|run_campaign' | head
exit 0
