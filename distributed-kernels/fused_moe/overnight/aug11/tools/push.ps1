# Copy a local file to the 8xMI350X node, stripping CRLF on the way.
# Companion to nsh.ps1 (which RUNS a local script remotely); this one PLACES
# a file so it can be invoked repeatedly, e.g. under setsid in the background.
param(
  [Parameter(Mandatory = $true)][string]$Local,
  [Parameter(Mandatory = $true)][string]$Remote
)
$node = 'subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu'
if (-not (Test-Path $Local)) { throw "no such file: $Local" }
$body = (Get-Content -Raw $Local).Replace("`r", "")
# tr on the far side also eats the trailing CR that PowerShell appends when it
# pipes a string into a native command.
$body | ssh -q -o BatchMode=yes -o ServerAliveInterval=30 $node "mkdir -p `$(dirname '$Remote') && tr -d '\r' > '$Remote'"
exit $LASTEXITCODE
