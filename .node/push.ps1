# Push the frequently-edited trees to the MI300X node and normalize line endings.
# Usage: powershell -ExecutionPolicy Bypass -File .node\push.ps1
$node = "subvadla@banff-sc-cs47-05.dh170.dcgpu"
$repo = "/home/subvadla/dhk"

scp -q -r distributed-kernels\gemm_rs "${node}:${repo}/distributed-kernels/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp gemm_rs failed"; exit 1 }
scp -q -r .node "${node}:${repo}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp .node failed"; exit 1 }

ssh $node "cd $repo; find .node -type f \( -name '*.sh' -o -name '*.py' \) -exec sed -i 's/\r`$//' {} + ; find distributed-kernels/gemm_rs -type f -exec sed -i 's/\r`$//' {} + ; echo pushed"
