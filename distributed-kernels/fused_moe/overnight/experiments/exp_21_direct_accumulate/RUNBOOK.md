# exp_21 runbook — EXACT build+test sequence for mode 12 (run when node is ours)

Node: `ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.5.95.87`. Container for
builds: `subha_k1`. Node checkout must be synced per experiment **only when no
campaign is running** (`pgrep -af 'torchrun|mpirun'` empty, `rocm-smi
--showpids` = gpuagent only).

## 0. Sync + genco build + ISA/resource gate (~3 min)

```bash
ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.5.95.87 \
  "cd ~/Distributed-HipKittens && git fetch origin -q && git reset --hard origin/codex/distributed-hipkittens-scaffold -q && git log --oneline -1"
```

Then in `docker exec subha_k1 bash -lc "..."` (MORI_JIT_ROOT is
/usr/local/lib/python3.12/dist-packages/mori/_jit-sources there):

```bash
DHK=/workspace  # ABSENT: the node repo is at ~/, and subha_k1 does NOT mount it.
```

Build on the NODE filesystem instead (subha_k1 mounts ~/amd-master read-only
and the caches live on ~/). The campaign JIT builds inside ephemeral
containers with `-v ~/Distributed-HipKittens:/workspace/Distributed-HipKittens:ro`
+ include roots per BUILDING.md; the representative genco for a fast check:

```bash
ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.5.95.87 "docker exec subha_k1 bash -lc '
set -e
DHK_ROOT=/workspace/Distributed-HipKittens
AMD=/workspace/amd-master/auto-gpu-kernel/k0_fused_moe
MORI=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I\"\$DHK_ROOT/include\" -I\"\$DHK_ROOT/distributed-kernels/fused_moe\" \
  -I\"\$AMD/solution/hip/hkp\" -I\"\$AMD/prefill_opt/kernels\" \
  -I\"\$AMD/solution/hip\" \
  -I\"\$MORI\" -I\"\$MORI/include\" -I\"\$MORI/src\" \
  -I\"\$MORI/3rdparty/spdlog/include\" -I\"\$MORI/3rdparty/msgpack-c/include\" \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  \$DHK_ROOT/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
  -o /tmp/k0pf6gm_mps_mega.hsaco 2>&1 | grep -A12 k0pf6gm_mps_mega || true
llvm-objdump -d --mcpu=gfx950 /tmp/k0pf6gm_mps_mega.hsaco > /tmp/mps_isa.txt
echo \"atomic_pk_add_bf16: \$(grep -c atomic_pk_add_bf16 /tmp/mps_isa.txt)  cmpswap: \$(grep -c atomic_cmpswap /tmp/mps_isa.txt)\"
'"
```

Gates on the output (hold the budget):
- `SGPR 104±few / VGPR 256 / AGPR 256 / scratch ≤ 60-128 B / LDS 155,428+64 B`
- MFMA census 96+84 = 180; ZERO scratch-load/store inside both K-loops.
- `atomic_pk_add_bf16` count ~present (was: nonzero), `atomic_cmpswap: 0`
- `ds_read2_b64`/`ds_read_b64` in the epilogue region (the peer table).

Note: subha_k1 does NOT mount ~/Distributed-HipKittens — check mounts first
(`docker exec subha_k1 ls /workspace/`). If the DHK mount name differs, use
whatever the campaign driver binds (`docker inspect subha_k1 | grep -A8 Mounts`).
The campaigns build their own inside containers with the DHK mount
`/workspace/Distributed-HipKittens:ro` — confirm by `docker inspect` on one of
the completed campaign containers' configs archived in the logs.

## 1. ubench (fabric rate, ~2 min)

```bash
scp -i ~/.ssh/muhammad-gpu -P 2425 \
  distributed-kernels/fused_moe/overnight/experiments/exp_21_direct_accumulate/ubench_fabric_rate.cpp \
  subvadla@10.5.95.87:/tmp/
ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.5.95.87 \
  "docker cp /tmp/ubench_fabric_rate.cpp subha_k1:/tmp/ && docker exec subha_k1 bash -lc 'hipcc -O2 --offload-arch=gfx950 -munsafe-fp-atomics /tmp/ubench_fabric_rate.cpp -o /tmp/ubench_fabric_rate && /tmp/ubench_fabric_rate'"
```

Reading it: if `atomic-PEER` tracks `stores-PEER` within ~2x at 128-256 CTAs,
the fabric merges same-line RMWs and mode 12's epilogue will not be op-rate
bound. If `atomic-PEER` collapses toward `atomSC-PEER`, mode 12 is expected to
LOSE and the fallback is the hybrid (local part pre-reduce + pool pushes with
plain vector stores = today's mode 2) — still publishable as a measured limit
of fabric RMWs. `atomic-LOCAL` calibrates the source-side L2 RMW rate.

## 2. Screens (mode 12 vs mode 2 anchor; 25-95 s each)

Per point, from `~/amd-master/auto-gpu-kernel/k0_fused_moe`:

```bash
setsid timeout 1800 env K0_MOK_ARMS=production,mps_mega \
  K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
  K0_MOK_OUTPUT_ROOT=$HOME/k0-mok-exp21/C<c>_m12 \
  K0_MOK_RUN_TIMEOUT=1500 \
  K0_MPS_CFG="C=<c>,g=1,mode=12,flush_rows=16" \
  bash benchmarks/mok_synthetic_prefill/run_campaign.sh e21C<c>m12 1
```

Sweep C ∈ {4,8,16,32,64}. Discriminator: every runN.log must contain `[MARK]`
lines and runN/ eight rank JSONs (else a config rejection is masquerading).

## 3. Detector certification (per winning C)

Same as 2 with `g=17` (physical g=1 + dual-write detector). The dual run must
pass gates with `pperr == 0` (any 1<<27 = lost remote updates = blocker 1b is
LIVE — stop and report) — note the run will be slower (dual write + pull).

## 4. Full ladder + decision campaigns

At the winning C: `K0_MOK_ARMS=production,pf6gm_mega,mps_mega`, warmup 500 /
timed 100, 5 processes. Two independent campaigns, per sub-5% rule.

## Watch-items during the very first mode-12 launch

- `[MPS SOAK]` pperr == 0 across all ranks every epoch (bit 27 = detector).
- The `address (nil)` signature at first launch = the MPS guard config
  rejection (mode 12 <= mode cap in the NODE's adapter copy — a stale checkout
  masquerades this way; always pair the launch with the hsaco mtime).
- If mode 12's screen is ABOVE mode 2's 6,866 µs, do not iterate config yet:
  first read `atomdr-PEER` vs `atomic-PEER` from the ubench — that splits
  "fabric ACK latency exposed per task" from "fabric op rate saturated".
