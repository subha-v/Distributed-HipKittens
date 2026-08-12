#!/usr/bin/env bash
# exp_34 step 0: make a PRIVATE scratch clone for CPU-only compiles.
# Never touches $HOME/Distributed-HipKittens (a GPU campaign is pinned to it).
set -uo pipefail
H=$HOME
SC=$H/e34
mkdir -p "$SC/out"
if [ -d "$SC/DHK/.git" ]; then
  echo "EXISTS"
else
  git clone -q --no-hardlinks "$H/Distributed-HipKittens" "$SC/DHK" && echo "CLONED"
fi
cd "$SC/DHK"
echo "=== head ==="
git log --oneline -1
git rev-parse HEAD
echo "=== branch ==="
git rev-parse --abbrev-ref HEAD
echo "=== dirty (must be empty) ==="
git status --porcelain
echo "=== container ==="
docker ps --format '{{.Names}}' | grep -c subha_k1
echo "=== GPU lease holder (do not disturb) ==="
rocm-smi --showpids 2>/dev/null | sed -n '1,12p'
