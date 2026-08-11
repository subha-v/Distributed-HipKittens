# Run a local bash script on the 8xMI300X node.
# Two reasons this exists instead of inline `ssh host "cmd"`:
#   1. PowerShell steals `2>/dev/null`, `$(...)`, `(`, `<` and friends from
#      double-quoted remote strings. Hours were lost to this on 2026-08-10.
#   2. Windows checkouts are CRLF; bash chokes on any stray \r.
#
# The body is base64-encoded and decoded on the far side. Piping the raw body
# over stdin used to work but PowerShell appends a CRLF to native-command
# stdin, which produced a spurious `bash: line N: $'\r': command not found`
# and an exit code of 127 on EVERY run -- masking real failures. Base64 makes
# the transport byte-exact and the exit code trustworthy.
#
#   powershell -ExecutionPolicy Bypass -File tools\nsh.ps1 -Script tools\probe_env.sh
#   powershell -ExecutionPolicy Bypass -File tools\nsh.ps1 -Script tools\x.sh -Args "600 512"
param(
  [Parameter(Mandatory = $true)][string]$Script,
  [string]$ArgLine = ""
)
$node = 'subvadla@banff-sc-cs47-05.dh170.dcgpu'
if (-not (Test-Path $Script)) { throw "no such script: $Script" }
$body = (Get-Content -Raw $Script).Replace("`r", "")
$b64  = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($body))
$cmd  = "echo $b64 | base64 -d > /tmp/nsh_`$`$.sh; bash /tmp/nsh_`$`$.sh $ArgLine; rc=`$?; rm -f /tmp/nsh_`$`$.sh; exit `$rc"
ssh -q -o BatchMode=yes -o ServerAliveInterval=30 $node $cmd
exit $LASTEXITCODE
