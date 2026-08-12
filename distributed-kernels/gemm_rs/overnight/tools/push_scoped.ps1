# Sync ONLY the named paths to the MI300X node, then strip CRLF on those paths.
#
# Why this exists: `push.ps1` scps the whole overnight tree AND all four kernel
# sources. With several subagents editing different experiments concurrently,
# any one of them running push.ps1 ships every other agent's half-written file
# and can silently revert a validated kernel while leaving build/*.so intact
# (HANDOFF.md, "Traps that cost real time"). This tool pushes a caller-declared
# path set instead, so two agents can never clobber each other.
#
# Paths are repo-root-relative, semicolon-separated, files or directories:
#   powershell -File tools\push_scoped.ps1 -Paths "distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation"
param(
  [Parameter(Mandatory = $true)][string]$Paths
)
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
$repo = '/home/subvadla/dhk'

# Locate the repo root regardless of the caller's cwd.
$root = (git rev-parse --show-toplevel) 2>$null
if (-not $root) { Write-Error "not inside the git worktree"; exit 1 }
$root = $root.Replace('/', '\')

$list = $Paths.Split(';') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$pushed = @()

foreach ($rel in $list) {
  $relFwd = $rel.Replace('\', '/')
  $local  = Join-Path $root $relFwd.Replace('/', '\')
  if (-not (Test-Path $local)) { Write-Error "no such local path: $rel"; exit 1 }

  # scp -r of a directory copies the directory ITSELF into the destination, so
  # the destination must be the parent, and the parent must already exist.
  $parent = ($relFwd -split '/')[0..(($relFwd -split '/').Length - 2)] -join '/'
  ssh -q $node "mkdir -p '$repo/$parent'"
  if ($LASTEXITCODE -ne 0) { Write-Error "mkdir failed for $parent"; exit 1 }

  if (Test-Path $local -PathType Container) {
    scp -q -r $local "${node}:${repo}/${parent}/"
  } else {
    scp -q $local "${node}:${repo}/${relFwd}"
  }
  if ($LASTEXITCODE -ne 0) { Write-Error "scp failed for $rel"; exit 1 }
  $pushed += $relFwd
}

# Text files only, selected by extension: an earlier version of push.ps1 ran
# sed over every file and stripped 0x0D out of compiled build/*.so modules,
# producing a dlopen segfault that looked like a HIP-init bug.
$exts   = "-name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.cpp' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.txt' -o -name '*.json' -o -name '*.csv'"
$joined = ($pushed | ForEach-Object { "'$_'" }) -join ' '
ssh -q $node "cd $repo; for p in $joined; do if [ -d `"`$p`" ]; then find `"`$p`" -type f -writable \( $exts \) -not -path '*/build/*' -exec sed -i 's/\r`$//' {} + ; else case `"`$p`" in *.sh|*.py|*.md|*.cpp|*.cuh|*.hpp|*.txt|*.json|*.csv) sed -i 's/\r`$//' `"`$p`" ;; esac; fi; done; echo scoped-push-ok"
exit $LASTEXITCODE
