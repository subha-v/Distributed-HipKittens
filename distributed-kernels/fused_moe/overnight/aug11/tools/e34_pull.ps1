# exp_34: pull the two owned source files back from the node's PRIVATE scratch
# clone (~/e34/DHK) into the local repo, so the working tree matches exactly what
# was compiled and gated. Never touches the node's ~/Distributed-HipKittens.
$node = 'subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu'
$fm   = 'C:\Users\subvadla\repos\Distributed-HipKittens\distributed-kernels\fused_moe'
$rem  = '/home/subvadla/e34/DHK/distributed-kernels/fused_moe'
foreach ($f in @('k0pf6gm_device_tile_mps.hip', 'moe_mps_adapter.cuh')) {
    & cmd.exe /c "scp -o BatchMode=yes -o LogLevel=ERROR ${node}:$rem/$f `"$fm\$f`""
}
& cmd.exe /c "ssh -o BatchMode=yes -o LogLevel=ERROR $node ""sha256sum $rem/k0pf6gm_device_tile_mps.hip $rem/moe_mps_adapter.cuh"""
Get-FileHash -Algorithm SHA256 (Join-Path $fm 'k0pf6gm_device_tile_mps.hip'), (Join-Path $fm 'moe_mps_adapter.cuh') |
  ForEach-Object { $_.Hash.ToLower() + '  ' + $_.Path }
