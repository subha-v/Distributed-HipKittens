#!/usr/bin/env bash
# t12: reuse exp_26's loop finder / census / scratch bucketing on the CURRENT
# source's five builds. Answers: does bit 2 still realign the barrier partition,
# does bit 0 still move the vmcnt(0) drain, and where does scratch actually live
# now that exp_21/24 have changed the phase-2 register picture.
set -uo pipefail
SC=$HOME/overnight-scratch/e26act
mkdir -p "$SC/py"
cp "$HOME/overnight-scratch/e26/py/mask.py" "$SC/py/mask.py" 2>/dev/null \
  || { echo "no exp_26 mask.py to reuse"; exit 3; }
sha256sum "$SC/py/mask.py" | cut -c1-16
docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26act/out && python3 ../py/mask.py D M0 M4 M1 M5"
exit 0
