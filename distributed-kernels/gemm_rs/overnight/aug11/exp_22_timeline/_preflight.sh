#!/usr/bin/env bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
echo "--- lease tool ---"; ls -l "$ON/tools/gpu_lease.sh" 2>&1
echo "--- lease status ---"; bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -20
echo "--- exp_22 files ---"; ls -l "$E22" 2>&1 | head -40
echo "--- production .so ---"; ls -l "$ON/harness/build"/*.so 2>&1
echo "--- containers ---"; docker ps --format '{{.Names}}' 2>&1
echo "--- torch.distributed.run present? ---"
docker exec dhk-gemmrs python3 -c "import torch,torch.distributed.run as r; print(torch.__version__, r.__file__)" 2>&1 | tail -2
echo "--- rocprofv3 ---"; docker exec dhk-gemmrs bash -lc 'which rocprofv3 && rocprofv3 --version 2>&1 | head -3' 2>&1 | head -5
echo "--- trace flag default in kernel ---"
grep -n "HK_GEMM_RS_MI300X_TRACE" "$ON/../gemm_rs_mi300x.cpp" | head -8
