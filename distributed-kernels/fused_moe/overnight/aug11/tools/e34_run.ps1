# exp_34 helper: send a local bash script to the MI350X node with LF endings.
# Usage: powershell -NoProfile -File e34_run.ps1 <script.sh>
param([Parameter(Mandatory=$true)][string]$Script)
$node = 'subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu'
$text = [IO.File]::ReadAllText($Script).Replace("`r`n", "`n")
$tmp  = Join-Path $env:TEMP 'e34_send.sh'
[IO.File]::WriteAllText($tmp, $text)
& cmd.exe /c "ssh -o BatchMode=yes -o LogLevel=ERROR $node bash -s < `"$tmp`" 2>&1"
exit $LASTEXITCODE
