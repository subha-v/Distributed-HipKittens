# Push only this experiment's scripts, LF-normalized. Same reason as
# push_src.ps1: tools/push.ps1 syncs the whole tree including compbench and
# takes longer than a gate-ladder arm.
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
$dest = '/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_05_release_granularity'
$src  = 'distributed-kernels\gemm_rs\overnight\experiments\exp_05_release_granularity'

$files = Get-ChildItem -Path $src -File | Where-Object {
  $_.Extension -in '.sh', '.py', '.md'
} | ForEach-Object { $_.FullName }

foreach ($f in $files) {
  $text = [IO.File]::ReadAllText($f).Replace("`r`n", "`n")
  [IO.File]::WriteAllText($f, $text)
}

scp -q $files "${node}:${dest}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp failed"; exit 1 }
Write-Host "pushed $($files.Count) files"
