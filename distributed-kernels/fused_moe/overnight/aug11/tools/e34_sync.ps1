# exp_34: push the two owned source files into the node's PRIVATE scratch clone
# (~/e34/DHK), LF-normalized. Never touches ~/Distributed-HipKittens.
$node = 'subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu'
$fm   = 'C:\Users\subvadla\repos\Distributed-HipKittens\distributed-kernels\fused_moe'
$stage = Join-Path $env:TEMP 'e34stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($f in @('k0pf6gm_device_tile_mps.hip', 'moe_mps_adapter.cuh')) {
    $t = [IO.File]::ReadAllText((Join-Path $fm $f)).Replace("`r`n", "`n")
    [IO.File]::WriteAllText((Join-Path $stage $f), $t)
}
& cmd.exe /c "ssh -o BatchMode=yes -o LogLevel=ERROR $node mkdir -p /home/subvadla/e34/DHK/distributed-kernels/fused_moe"
& cmd.exe /c "scp -o BatchMode=yes -o LogLevel=ERROR `"$stage\k0pf6gm_device_tile_mps.hip`" `"$stage\moe_mps_adapter.cuh`" ${node}:/home/subvadla/e34/DHK/distributed-kernels/fused_moe/"
& cmd.exe /c "ssh -o BatchMode=yes -o LogLevel=ERROR $node ""cd /home/subvadla/e34/DHK && git diff --stat && md5sum distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip distributed-kernels/fused_moe/moe_mps_adapter.cuh"""
exit $LASTEXITCODE
