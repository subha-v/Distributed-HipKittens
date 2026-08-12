# Run a local bash script on the 8xMI350X node.
# Two reasons this exists instead of inline `ssh host "cmd"`:
#   1. PowerShell steals `2>/dev/null` and friends from double-quoted remote strings.
#   2. Windows checkouts are CRLF; bash chokes on any stray \r, so strip them all.
param(
  [Parameter(Mandatory = $true)][string]$Script
)
$node = 'subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu'
if (-not (Test-Path $Script)) { throw "no such script: $Script" }
$body = (Get-Content -Raw $Script).Replace("`r", "")
$body | ssh -q -o BatchMode=yes -o ServerAliveInterval=30 $node "bash -s"
exit $LASTEXITCODE
