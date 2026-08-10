# GEMM-RS provenance

## Frozen measured inputs

The authoritative repository is `amd-master` at commit
`182f3e269068d3809ebf9074395217351837aec5`.

| donor | authoritative path | SHA-256 | measured fact |
|---|---|---|---|
| exp 23 | `auto-gpu-kernel/k1_comm_overlap/experiments/exp_23_emit_timed/src/ladder_emit.cpp` | `70ce26dacaa2f97ff57284132e2e2b42ba80bcd7fd1fac20cecb0471de666d82` | EV=1/F6/AV1 rung D: 376.7 µs in the same-run table; 375.4 vs 525.5 µs paired against stock emit (-28.40%, 8/8 sign-consistent) |
| exp 24 | `auto-gpu-kernel/k1_comm_overlap/experiments/exp_24_reduce/src/ladder_reduce.cpp` | `62d8d61358ea5a6b3d8ea98579f1a16cab8078d2659a5fabe3e573f9fd211eb4` | REDV=1: 499.5 vs 528.4 µs (-5.47%, 6/6 sign-consistent) |

Both measurements were world 8 on gfx950. The exp-23 communication-dominant
shape was `(m=8192,n=4096,k=14336,bias=true)`. Correctness was bit-identical in
the timed runs and the recorded 600-replay gates were clean. The exp-23 kernel
used 220 VGPR, 83 SGPR, 128 B pre-existing scratch, 160,000 B dynamic LDS, two
waves/SIMD occupancy, and a 256-MFMA mainloop.

## Rewrite ledger

The first donor-derived abstraction rewrite is `amd-master` commit
`feea09b267602d78f1e51f5cd2ffd38aa0f4a76c`. This port retains its exact GEMM
schedule, task/owner map, EV=1 staging and packet addresses, device epochs,
source-slot layout, and ordered REDV=1 arithmetic. The mechanism adapter now
uses `kittens::pgl` and `kittens::distributed`.

The rewrite deliberately adds a directed per-tile reuse credit. The producer
waits for credit `e-1` before overwriting epoch `e`; after all owner loads drain,
the reducer publishes credit `e` to that source. This closes a replay race in
the measured donor, whose cumulative readiness signal alone did not prevent a
fast producer from overwriting a live landing slot.

## Claim boundary

EV=1 and REDV=1 were never measured together. Neither the new two-launch
composition nor D-prime inherits either timing. The credit repair and the new
HipKittens adapter also require fresh generated-ISA and GPU gates. The donor
numbers above are historical baselines only.

The older MI300X six-shape rank-1 submission (`gemm_rs_rank1_58abcf.py`, score
413.139) remains a useful architecture baseline, but it is a multi-launch
direct-peer/barrier/reduce implementation and is not the selected mechanism
port.
