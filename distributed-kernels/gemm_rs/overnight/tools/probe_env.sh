#!/usr/bin/env bash
# Environment probe for the 8x MI300X node. Prints a compact report of every
# toolchain fact the GEMM-RS gfx942 bring-up depends on.
set -u

CTR=ddt-o0-4df4e85f

hdr() { echo; echo "===== $* ====="; }

hdr "HOST: rocm / hipcc"
/opt/rocm/bin/hipcc --version 2>&1 | head -4
echo "offload targets:"
ls /opt/rocm/amdgcn/bitcode >/dev/null 2>&1 && echo "  bitcode ok"
/opt/rocm/llvm/bin/amdgpu-arch 2>/dev/null | sort -u || rocminfo 2>/dev/null | grep -m4 'gfx'

hdr "HOST: python packages"
python3 -V
for m in numpy pybind11 torch; do
  python3 -c "import $m,sys;print('  $m', getattr($m,'__version__','?'))" 2>/dev/null || echo "  $m MISSING"
done
python3 -m pip --version 2>&1 | head -1

hdr "CONTAINER $CTR: identity"
docker exec "$CTR" bash -c 'cat /etc/os-release | head -2; echo "hipcc: $(hipcc --version 2>/dev/null | head -1)"' 2>&1 | head -5

hdr "CONTAINER: python + torch"
docker exec "$CTR" bash -c 'python3 -V; python3 -c "import torch;print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"ngpu\",torch.cuda.device_count())"' 2>&1 | tail -4
echo "-- other pythons --"
docker exec "$CTR" bash -c 'ls /usr/bin/python3.* 2>/dev/null; ls -d /opt/venv /opt/conda 2>/dev/null; for p in /usr/bin/python3.10 /usr/bin/python3.11 /usr/bin/python3.12; do [ -x $p ] && $p -c "import torch;print(\"$p torch\",torch.__version__)" 2>/dev/null; done' 2>&1 | tail -8

hdr "CONTAINER: pybind11 / numpy"
docker exec "$CTR" bash -c 'python3 -c "import pybind11;print(\"pybind11\",pybind11.__version__,pybind11.get_include())"; python3 -c "import numpy;print(\"numpy\",numpy.__version__)"' 2>&1 | tail -4

hdr "NETWORK: can we reach github"
timeout 25 git ls-remote --heads https://github.com/ROCm/iris.git 2>&1 | head -3
echo "pip index:"
timeout 25 python3 -m pip download --no-deps --dest /tmp/pipprobe pybind11 2>&1 | tail -2

hdr "GPU: topology + partition"
rocm-smi --showtopo 2>&1 | grep -iE 'XGMI|Link|hops' | head -12
rocm-smi --showcomputepartition 2>&1 | grep -c SPX

hdr "DONOR: amd-master gemm-rs assets"
AM=$HOME/amd-master
ls "$AM/auto-gpu-kernel/k2_mi300x_megakernel/inputs/specs/reference-kernels/gemm-rs/" 2>/dev/null
echo "-- k1 harness --"
ls "$AM/auto-gpu-kernel/k1_comm_overlap/harness/" 2>/dev/null | head -30
echo "-- exp_13 sota src --"
ls "$AM/auto-gpu-kernel/k1_comm_overlap/experiments/exp_13_sota_baseline/src/" 2>/dev/null | head
echo "-- rank1 submission --"
ls -l "$AM/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/" 2>/dev/null

hdr "DONE"
