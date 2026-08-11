# Sync the GEMM-RS overnight tree and kernel sources to the MI300X node.
#   powershell -ExecutionPolicy Bypass -File tools\push.ps1
#
# WARNING, learned the hard way on 2026-08-10: normalize line endings on TEXT
# FILES ONLY, selected by extension. An earlier version ran `sed -i 's/\r$//'`
# over every file under the harness directory, which silently stripped 0x0D
# bytes out of the compiled build/*.so modules and corrupted them. The symptom
# was a dlopen segfault in every process, which looked convincingly like a
# multiprocessing or HIP-init bug and cost about an hour.
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
$repo = '/home/subvadla/dhk'
$rel  = 'distributed-kernels/gemm_rs'

ssh -q $node "mkdir -p $repo/$rel/overnight"
if ($LASTEXITCODE -ne 0) { Write-Error "mkdir failed"; exit 1 }

scp -q -r distributed-kernels\gemm_rs\overnight "${node}:${repo}/${rel}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp overnight failed"; exit 1 }
scp -q distributed-kernels\gemm_rs\*.cpp distributed-kernels\gemm_rs\*.cuh distributed-kernels\gemm_rs\*.hpp "${node}:${repo}/${rel}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp kernel sources failed"; exit 1 }

$exts = "-name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.cpp' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.txt' -o -name '*.json'"
$cmd  = "cd $repo; find $rel -type f -writable \( $exts \) -not -path '*/build/*' -not -path '*/compbench/*' -exec sed -i 's/\r`$//' {} + ; echo pushed"
ssh -q $node $cmd
