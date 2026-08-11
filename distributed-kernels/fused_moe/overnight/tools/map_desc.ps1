# Lay out the 63 mps descriptor words: index, decimal, hex, and cluster gaps.
$w = @(
136992140559360,136992140690432,136750521384960,136993061078016,136993391379712,
136993391379456,137026718379008,137026718379520,136751127461888,137026617745408,
137026626920448,137026628231168,137026605154304,137026718380032,137026629541888,
137026630594560,137026611249152,137026718396416,137026611282432,137026605416448,
137026582085632,136993391380480,137026718396928,137026718397952,137026611299328,
136993978583040,137026605678592,136745937010688,136748290015232,137026611364864,
137026611398144,136993979893760,136750687059968,137026718398976,137026718399488,
136736959102976,136737943126016,136745467248640,137026579644416,1,
8,2000000,137026611436032,8223,4096,
8,32,40960,263136,6,
136993391379968,4096,137026716303360,137026716303872,137026716337152,
137026717126656,137026716337664,137026613542912,137026716864000,137026716995072,
137026717126144,136993979894016,268567048
)

Write-Output ("count = {0}" -f $w.Count)
Write-Output ""
Write-Output "idx  decimal            hex"
for ($i = 0; $i -lt $w.Count; $i++) {
  $v = [uint64]$w[$i]
  $tag = ""
  if ($v -lt 100000000) { $tag = "  <-- SCALAR" }
  Write-Output ("{0,3}  {1,-18} 0x{2:X}{3}" -f $i, $v, $v, $tag)
}

Write-Output ""
Write-Output "=== pointer words sorted ascending, with gap to previous ==="
$ptrs = @()
for ($i = 0; $i -lt $w.Count; $i++) {
  if ([uint64]$w[$i] -ge 100000000) { $ptrs += [pscustomobject]@{ idx = $i; v = [uint64]$w[$i] } }
}
$sorted = $ptrs | Sort-Object v
$prev = [uint64]0
foreach ($p in $sorted) {
  $gap = if ($prev -eq 0) { 0 } else { $p.v - $prev }
  $gapmib = [math]::Round($gap / 1MB, 3)
  Write-Output ("w{0,-3} 0x{1:X}  gap={2,-14} ({3} MiB)" -f $p.idx, $p.v, $gap, $gapmib)
  $prev = $p.v
}

Write-Output ""
Write-Output "=== span ==="
$min = $sorted[0].v; $max = $sorted[-1].v
Write-Output ("min w{0} 0x{1:X}   max w{2} 0x{3:X}   span={4} MiB" -f $sorted[0].idx, $min, $sorted[-1].idx, $max, [math]::Round(($max - $min) / 1MB, 2))
