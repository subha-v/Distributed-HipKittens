# Source audit and donor selection

This snapshot records the inputs used to scaffold Distributed HipKittens. It
prevents branch names, historical measurements, and abstraction-port results
from being conflated later. Remote branch heads were fetched on 2026-08-10;
commit IDs, not mutable branch names, are the durable references.

## Design source

The referenced Codex task `019fde9c-576d-70c2-bf46-af0e505cb1ae`
established the boundary used here:

- PGL answers only where the corresponding peer view is located.
- Payload movement, release, publication, observation, acquire, counters, and
  storage lifetime are distinct device operations.
- The operator owns CTA roles, dependency keys, epochs, routes, progress, and
  timeout control flow.
- Caller-owned bounded-wait results and named rank bases are code-generation
  constraints, not cosmetic API choices.

## `amd-master` branches

| ref | audited head | conclusion |
| --- | --- | --- |
| `origin/vllm-integration` | `7cede33497d1158960fb62093b5c4272a2911993` | The two commits after `b5ec0da0` add and then revert the MI350X baseline document. The tree at the audited head is exactly the `b5ec0da0` tree, so there is no net kernel or integration delta to port. |
| `origin/debug/pf6-first-launch` | `2705774b2bd4948869182c2fb603fe9b4809d021` | Contains the device-tile rewrite and the subsequent gfx942 evidence campaign. Its latest commit records a transient O1 node/DNS blocker; it is not a completed operator-performance result. |

The local debug checkout was intentionally left at
`e0b1a315b304649a4502afa7c14cf443ef4fc225` because it contains unrelated
working-tree changes and was 35 commits behind the fetched remote. No pull,
reset, checkout, or cleanup was performed in that repository.

The important debug-line checkpoints are:

- `feea09b267602d78f1e51f5cd2ffd38aa0f4a76c`: first organized
  distributed-device-tile rewrite and both donor-derived operator ports;
- `e0b1a315b304649a4502afa7c14cf443ef4fc225`: start of the gfx942 campaign;
- `0de57443750f4101cefa50ac94d1eb74aeaeaf45`: completed world-eight A/A
  calibration;
- `1bb22272f1ad8268d9c976965a7684f100fa9ac6` and
  `cda82cf07db97cfa6af250bfa87cf47192fc5b15`: real-callsite API twin
  documentation and host artifacts;
- `f39b163a58f9355455b53093a4627f9bccaee61a` through the audited head:
  fail-closed O1 bridge work, stock AG correctness preservation, and an
  infrastructure stop before a completed abstraction performance baseline.

The usable evidence is therefore deliberately narrow. Named PGL bases and the
caller-owned wait-result spelling have synthetic/compiler parity evidence on
gfx942. Dynamic pointer arrays/by-reference controls introduced calls and 80 B
of scratch. One direct 16-byte packet source shape was pinned. None of this is
a completed world-eight GEMM-RS or MoE abstraction-parity measurement, so the
campaign verdict remains local/incubating rather than promoted.

Two similarly named donor trees served different purposes and were audited
separately:

- `auto-gpu-kernel/k0_fused_moe/distributed_device_tile/` at `feea09b2...`
  contains the first header-only peer/completion/transfer abstractions and the
  donor-derived GEMM-RS and MoE rewrites. It is the source-design predecessor
  for this repository, not a dependency copied wholesale.
- `auto-gpu-kernel/k2_mi300x_megakernel/distributed_device_tile/` on the latest
  debug branch is the controlled gfx942 evidence campaign: frozen runners,
  dependency maps, capability probes, manual/helper twins, A/A calibration,
  and later infrastructure records. It establishes narrow compiler facts; it
  does not provide a promoted operator implementation.

The HipKittens fork itself was also audited across its architecture-specific
`types`, `ops`, tile/layout concepts, group/warp synchronization, and existing
`distributed-kernels` examples. PGL is consequently implemented beside the
ordinary global layout, while peer movement and synchronization live under
group operations. No scheduler, dependency graph, or operator-specific
distributed compute tile was added to those core hierarchies.

## IRIS boundary

The requested C++ IRIS source is the `iris/irisx` tree imported into
`amd-master` at `be47a17ad95a0dea5700548399efdb9b3885ce89`. The authoritative
header for this integration is
`iris/irisx/development/include/iris/iris.hpp`, whose audited SHA-256 is
`c4563f041497a808c30c78d57bb64130e1ba7333f106428d4892f4787b9284f2`.

IRIS's device translation rule is:

```text
peer pointer = peer heap base + (local pointer - local heap base)
```

IRIS therefore owns MPI/IPC exchange, fine-grained symmetric allocation, heap
mapping, and host lifetime. The HipKittens adapter validates the runtime world,
materializes named bases, and keeps the full runtime-indexed device view out of
MFMA regions. For an allocation that begins after heap offset zero, it first
shifts every peer heap base to the corresponding allocation base; PGL then
preserves the independent tensor-view offset inside that allocation.

The generic IRIS system fence currently maps every order through
`__threadfence_system()`. Distributed HipKittens does not reuse it for the core
consumer edge because a pure acquire must not also perform a producer-side
release/writeback.

## Selected performance donors

| port | immutable donor | selected evidence | port claim |
| --- | --- | --- | --- |
| GEMM-RS | exp-23 source at `182f3e269068d3809ebf9074395217351837aec5`, SHA-256 `70ce26dacaa2f97ff57284132e2e2b42ba80bcd7fd1fac20cecb0471de666d82` | gfx950 world-eight EV=1/F6/AV1 rung D, 376.7 us in the paired table; exp-24 REDV=1 independently improved the reducer by 5.47% | The new composition and replay credits are unmeasured. EV=1 and REDV=1 were not measured together. |
| Fused MoE | exp-64 source at `c5a0aac71d92e229c9a1a5f1743f69e2d1279f56`, SHA-256 `57152a087c62aee1ffaae464cd2972e321d5e283e0681bc5a248039a6dc1ae2c` | same source built with `K0P6GM_G=3` at `d34e5510c4672e998eb2e94672e46e2f784ca07f`, 6,919.8 versus 7,698.0 us (0.89846x) | CDNA4 parity target only. The adapted source is unbuilt/untimed and its roughly 155 KiB LDS footprint is not a gfx942 port. |

Exact paths, helper hashes, build flags, resource records, and stronger claim
boundaries live beside each port in `distributed-kernels/gemm_rs/` and
`distributed-kernels/fused_moe/`.
