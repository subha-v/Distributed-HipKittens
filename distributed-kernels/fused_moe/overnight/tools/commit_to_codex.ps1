# Commit specific files onto codex/distributed-hipkittens-scaffold WITHOUT
# touching the shared working tree or HEAD.
#
# Why this exists: a second agent shares this checkout and has its own branch
# checked out (GEMM-RS). A plain `git commit` would land our fused-MoE work on
# their branch and drag their commits into ours on push. So we build the commit
# with plumbing against the remote codex head and push just that.
param(
  [Parameter(Mandatory = $true)][string]$Message,
  [Parameter(Mandatory = $true)][string[]]$Paths
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\subvadla\repos\Distributed-HipKittens'
Set-Location $repo

git fetch origin codex/distributed-hipkittens-scaffold --quiet
$base = (git rev-parse FETCH_HEAD).Trim()
Write-Output "base (remote codex head) = $base"

$idx = Join-Path $repo '.git\tmp-index-ovn'
if (Test-Path $idx) { Remove-Item $idx -Force }
$env:GIT_INDEX_FILE = $idx
try {
  git read-tree $base
  foreach ($p in $Paths) {
    $rel = $p.Replace('\', '/')
    $blob = (git hash-object -w -- $rel).Trim()
    git update-index --add --cacheinfo "100644,$blob,$rel"
    Write-Output "staged $rel -> $blob"
  }
  $tree = (git write-tree).Trim()
  $msgFile = Join-Path $env:TEMP "ovn-commit-msg.txt"
  Set-Content -Path $msgFile -Value $Message -Encoding utf8 -NoNewline
  $commit = (git commit-tree $tree -p $base -F $msgFile).Trim()
  Remove-Item $msgFile -Force -ErrorAction SilentlyContinue
} finally {
  Remove-Item $idx -Force -ErrorAction SilentlyContinue
  Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue
}
Write-Output "created commit $commit"
git push origin "${commit}:codex/distributed-hipkittens-scaffold"
Write-Output '=== remote codex head now ==='
git ls-remote origin codex/distributed-hipkittens-scaffold
