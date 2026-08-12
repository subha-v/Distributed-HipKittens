# Push the M9 gate only. harness/build.sh does not compile python, so no rebuild
# is needed and the module under test is untouched.
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
$dest = '/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness'
$f = 'distributed-kernels\gemm_rs\overnight\harness\m9_stale_slot.py'
$text = [IO.File]::ReadAllText($f).Replace("`r`n", "`n")
[IO.File]::WriteAllText((Resolve-Path $f), $text)
scp -q $f "${node}:${dest}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp failed"; exit 1 }
Write-Host "pushed m9_stale_slot.py"
