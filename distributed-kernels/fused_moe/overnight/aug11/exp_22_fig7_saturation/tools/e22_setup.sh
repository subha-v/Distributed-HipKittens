#!/usr/bin/env bash
# exp_22 step 0 (CPU ONLY): private scratch clone + toolchain capability probe.
# Touches NOTHING under $HOME/Distributed-HipKittens (a pinned GPU campaign owns
# it); clones FROM it read-only into $HOME/e22/DHK.
set -uo pipefail
H=$HOME
E22=$H/e22
mkdir -p "$E22/out" "$E22/src" "$E22/probe"

echo "===CLONE==="
if [ ! -d "$E22/DHK/.git" ]; then
  git clone --no-hardlinks --quiet "$H/Distributed-HipKittens" "$E22/DHK" || exit 1
fi
git -C "$E22/DHK" log --oneline -1
echo "clone_head=$(git -C "$E22/DHK" rev-parse HEAD)"
echo "pinned_head=$(git -C "$H/Distributed-HipKittens" rev-parse HEAD 2>/dev/null)"
ls -l "$E22/DHK/include/cdna4/ops/group/distributed/packet.cuh" 2>&1

echo "===PROBE_COMPILE==="
docker exec subha_k1 bash -lc '
set -u
P=/home/subvadla/e22/probe   # the container runs as root; HOME is NOT the user home
for v in A B C D E; do
  hipcc --offload-arch=gfx950 -std=c++20 -O3 -DKITTENS_CDNA4 \
        -DPROBE_VARIANT_$v -c "$P/e22_probe.hip" -o /dev/null \
        > "$P/probe_$v.log" 2>&1
  echo "variant $v exit=$? errors=$(grep -c "error:" "$P/probe_$v.log")"
done
echo "--- first error lines per failing variant ---"
for v in A B C D E; do
  if grep -q "error:" "$P/probe_$v.log"; then
    echo "### $v"; grep -m4 -A2 "error:" "$P/probe_$v.log"
  fi
done
'
echo "===DONE==="
