# Push ONLY the two kernel sources this experiment owns, with LF endings.
#
# tools/push.ps1 syncs the whole overnight tree including compbench and takes
# ~4 minutes, which is longer than a whole gate-ladder arm. Flipping
# RELEASE_GROUP between arms changes one character in one file, so the sweep
# would otherwise spend most of its wall time in scp.
#
# The LF normalization is not optional: tools/push.ps1 strips CR on the node
# after every copy, and without it a `#define ... 2\r` reaches the preprocessor
# with a carriage return in the replacement list.
#
#   powershell -ExecutionPolicy Bypass -File experiments\exp_05_release_granularity\push_src.ps1
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
$dest = '/home/subvadla/dhk/distributed-kernels/gemm_rs'
$files = @(
  'distributed-kernels\gemm_rs\gemm_rs_mi300x.cpp',
  'distributed-kernels\gemm_rs\gemm_rs_mi300x_constants.cuh'
)

foreach ($f in $files) {
  if (-not (Test-Path $f)) { throw "no such file: $f" }
  $text = [IO.File]::ReadAllText($f).Replace("`r`n", "`n")
  [IO.File]::WriteAllText((Resolve-Path $f), $text)
}

scp -q $files "${node}:${dest}/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp failed"; exit 1 }
ssh -q $node "grep -n 'define HK_GEMM_RS_MI300X_RELEASE_GROUP' $dest/gemm_rs_mi300x.cpp; file $dest/gemm_rs_mi300x.cpp"
