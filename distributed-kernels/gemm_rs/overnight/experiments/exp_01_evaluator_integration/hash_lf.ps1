# LF-normalized sha256 of text files, so Windows CRLF copies can be compared
# against the node's LF copies (push.ps1 strips CRs on the far side, so a raw
# Get-FileHash always disagrees and tells you nothing).
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Paths)
$sha = [Security.Cryptography.SHA256]::Create()
foreach ($p in $Paths) {
  $text  = (Get-Content -Raw $p).Replace("`r", "")
  $bytes = [Text.Encoding]::UTF8.GetBytes($text)
  $hex   = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
  "$hex  $p"
}
