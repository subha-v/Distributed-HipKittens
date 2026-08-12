#!/usr/bin/env bash
echo "=== syntax ==="
bash -n ~/.overnight-scripts/screen.sh && echo "bash OK"
awk "/<<'PY'/{f=1;next} /^PY\$/{f=0} f" ~/.overnight-scripts/screen.sh > /tmp/sp.py
python3 -c "import ast; ast.parse(open('/tmp/sp.py').read()); print('python OK')"
echo "=== NCOL sanity ==="
HDR=$(grep -m1 "^HDR=" ~/.overnight-scripts/screen.sh | sed "s/^HDR='//; s/'$//")
echo "$HDR" | tr ',' '\n' | wc -l
echo "=== pre-flight ==="
pgrep -af 'torchrun|mpirun' || echo "pgrep: EMPTY"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "no stale lock"
echo "HEAD: $(git -C ~/Distributed-HipKittens rev-parse --short HEAD)"
echo
mv -f ~/overnight-scratch/screen_a11base.csv ~/overnight-scratch/screen_a11base_pilot.csv
mv -f ~/overnight-scratch/screen_a11base.out ~/overnight-scratch/screen_a11base_pilot.out
cd ~
setsid nohup timeout 5400 bash ~/.overnight-scripts/screen.sh a11base \
  ~/overnight-scratch/a11base.cfgs \
  > ~/overnight-scratch/screen_a11base.out 2>&1 < /dev/null &
sleep 10
head -20 ~/overnight-scratch/screen_a11base.out
exit 0
