#!/usr/bin/env bash
J=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
echo "=== the two competing mps_mega binaries ==="
for h in 19c7f3873e77 e92e2fb2cce5; do
  f=$J/$h/k0pf6gm_mps_mega.hsaco
  echo "$h  size=$(stat -c %s $f)  mtime=$(stat -c %y $f | cut -c1-19)  md5=$(md5sum $f | cut -c1-12)"
done
echo "=== do the two dirs hold the same set of kernels? ==="
for h in 19c7f3873e77 e92e2fb2cce5; do echo "-- $h"; ls $J/$h | head -20; done
echo
echo "=== pre-flight ==="
pgrep -af 'torchrun|mpirun' || echo "pgrep: EMPTY"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
cat > ~/overnight-scratch/a11noise.cfgs <<'EOF'
C=16,g=33,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
C=16,g=33,mode=12,flush_rows=16
EOF
rm -f ~/overnight-scratch/screen_a11noise.csv ~/overnight-scratch/screen_a11noise.out
cd ~
setsid nohup timeout 5400 bash ~/.overnight-scripts/screen.sh a11noise \
  ~/overnight-scratch/a11noise.cfgs \
  > ~/overnight-scratch/screen_a11noise.out 2>&1 < /dev/null &
sleep 5
echo "=== launched ==="
head -12 ~/overnight-scratch/screen_a11noise.out
exit 0
