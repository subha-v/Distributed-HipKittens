# Run a local bash script on the 8xMI300X node.
# Two reasons this exists instead of inline `ssh host "cmd"`:
#   1. PowerShell steals `2>/dev/null`, `$(...)`, `(`, `<` and friends from
#      double-quoted remote strings. Hours were lost to this on 2026-08-10.
#   2. Windows checkouts are CRLF; bash chokes on any stray \r.
# Piping the script body over stdin with CRs stripped sidesteps both.
#
#   powershell -ExecutionPolicy Bypass -File tools\nsh.ps1 -Script tools\probe_env.sh
param(
  [Parameter(Mandatory = $true)][string]$Script
)
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
if (-not (Test-Path $Script)) { throw "no such script: $Script" }
$body = (Get-Content -Raw $Script).Replace("`r", "")
$body | ssh -q -o BatchMode=yes -o ServerAliveInterval=30 $node "bash -s"
exit $LASTEXITCODE
