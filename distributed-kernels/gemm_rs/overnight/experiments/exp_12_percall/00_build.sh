#!/usr/bin/env bash
# exp_12: build the launch-cost probe module. Independent of the kernel build.
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_12_percall
BUILD=$ON/harness/build

sed -i 's/\r$//' "$EXP"/*.sh "$EXP"/*.py "$EXP"/*.cpp 2>/dev/null
mkdir -p "$BUILD" "$EXP/logs"

echo "===== nullk.so ====="
docker exec -w "$EXP" dhk-gemmrs bash -lc '
set -e
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])")
PBINC=$(python3 -c "import pybind11;print(pybind11.get_include())")
hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC \
  -I"$PBINC" -I"$PYINC" -I/opt/rocm/include/hip \
  nullk.cpp -o '"$BUILD"'/nullk.so
'
rc=$?
echo "build rc=$rc"
ls -la "$BUILD/nullk.so" 2>&1

echo "===== import + one launch smoke (device 0 only) ====="
docker exec -w "$EXP" dhk-gemmrs python3 -c '
import importlib.util, torch, time
spec = importlib.util.spec_from_file_location("nullk", "'"$BUILD"'/nullk.so")
nk = importlib.util.module_from_spec(spec); spec.loader.exec_module(nk)
torch.cuda.set_device(0)
cells = torch.zeros(608, dtype=torch.int32, device="cuda:0")
err = torch.zeros(1, dtype=torch.int32, device="cuda:0")
sink = torch.zeros(4, dtype=torch.int32, device="cuda:0")
s = torch.cuda.current_stream(0).cuda_stream
for kind in (0, 1):
    nk.launch(kind, s, 304, 512, 65536, cells.data_ptr(), err.data_ptr(), sink.data_ptr())
torch.cuda.synchronize()
print("epoch cells after one k_epoch launch: min", int(cells[:304].min()),
      "max", int(cells[:304].max()), "(expect 1/1)")
# host-side issue cost of the probe itself, so the instrument is not the answer
for _ in range(50):
    nk.launch(0, s, 304, 512, 65536, cells.data_ptr(), err.data_ptr(), sink.data_ptr())
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(2000):
    nk.launch(0, s, 304, 512, 65536, cells.data_ptr(), err.data_ptr(), sink.data_ptr())
issue = (time.perf_counter() - t) * 1e6 / 2000
torch.cuda.synchronize()
print(f"pybind+hipLaunchKernel issue cost: {issue:.2f} us/launch")
'
echo "DONE-build"
exit $rc
