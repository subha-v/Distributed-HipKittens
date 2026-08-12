#!/usr/bin/env bash
# Guard against push.ps1 clobbering node-side kernel edits: print the LF-form
# sha256 of every source push.ps1 overwrites, so it can be compared with the
# Windows copy's LF-normalized hash before anything is shipped.
set -u
cd /home/subvadla/dhk/distributed-kernels/gemm_rs || exit 1
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh \
         gemm_rs_mi300x_constants.cuh gemm_rs_mi300x_host_abi.hpp \
         overnight/harness/hk_submission.py; do
  if [ -f "$f" ]; then
    printf '%s  %s  mtime=%s\n' "$(sha256sum "$f" | cut -c1-64)" "$f" \
      "$(date -r "$f" +%Y-%m-%dT%H:%M:%S)"
  else
    printf 'MISSING  %s\n' "$f"
  fi
done
echo "----- built modules -----"
ls -la overnight/harness/build/*.so 2>/dev/null | head -20
echo "DONE"
