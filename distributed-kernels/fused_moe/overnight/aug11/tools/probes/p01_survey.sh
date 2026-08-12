#!/usr/bin/env bash
echo "===HOST==="; hostname; date -u +%FT%TZ
echo "===GITHEAD==="
cd ~/Distributed-HipKittens && git log -1 --format='%H %h %ci %s' && git rev-parse --abbrev-ref HEAD
echo "===GPUJOBS==="
pgrep -af 'torchrun|mpirun' || echo "(no torchrun/mpirun)"
echo "===SCRATCH_LS==="
ls -la ~/overnight-scratch/ 2>&1 | tail -60
echo "===CSV_LIST==="
ls -la ~/overnight-scratch/screen_*.csv 2>&1 | tail -40
echo "===OUT_LIST==="
ls -la ~/overnight-scratch/screen_*.out 2>&1 | tail -40
echo "===HOMEDIRS_K0MOK==="
ls -d ~/k0-mok-* 2>&1 | tail -60
echo "===OVN==="
ls -la ~/ovn/ 2>&1 | tail -40
echo "===TOOLS_ANYWHERE==="
ls -la ~/*.sh 2>&1 | tail -20
