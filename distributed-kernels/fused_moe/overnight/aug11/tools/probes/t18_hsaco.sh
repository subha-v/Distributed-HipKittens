#!/usr/bin/env bash
# t18: run 5 of batch A resolved to a DIFFERENT mori jit content-hash dir than
# runs 1-4 at the same HEAD. Decide whether the BINARY changed (serious) or only
# the cache key did (harmless). Compare the hsaco bytes directly.
set -uo pipefail
JIT="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
A="$JIT/9df63b03fb16"; B="$JIT/b43b280256c6"
echo "== build dirs =="
for d in "$A" "$B"; do
  echo "---- $(basename "$d") ----"
  ls -la "$d" 2>/dev/null | head -20
done
echo "== mps_mega hsaco bytes identical? =="
if [ -f "$A/k0pf6gm_mps_mega.hsaco" ] && [ -f "$B/k0pf6gm_mps_mega.hsaco" ]; then
  sha256sum "$A/k0pf6gm_mps_mega.hsaco" "$B/k0pf6gm_mps_mega.hsaco"
  cmp "$A/k0pf6gm_mps_mega.hsaco" "$B/k0pf6gm_mps_mega.hsaco" \
    && echo "IDENTICAL BINARY -- the cache key moved, the code did not" \
    || echo "DIFFERENT BINARY -- the arm changed mid-batch"
else
  echo "one of the hsaco files is missing"
fi
echo "== every hsaco in both dirs =="
for d in "$A" "$B"; do
  echo "---- $(basename "$d") ----"
  for f in "$d"/*.hsaco; do [ -f "$f" ] && printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$(basename "$f")"; done
done
echo "== what else differs between the dirs =="
diff <(ls -1 "$A") <(ls -1 "$B") || true
echo "== how many build dirs exist, newest first =="
ls -1dt "$JIT"/*/ | head -8
echo "== full list of build dirs with mps hsaco mtimes =="
for d in $(ls -1dt "$JIT"/*/ | head -8); do
  f="$d/k0pf6gm_mps_mega.hsaco"
  [ -f "$f" ] && printf '%-16s %s %s\n' "$(basename "$d")" "$(stat -L -c %y "$f" | cut -c1-19)" "$(sha256sum "$f" | cut -c1-12)"
done
exit 0
