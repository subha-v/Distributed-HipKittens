# exp_05 — role split at the dispatch→M6 boundary (COMET layer-0). DESIGNED, NOT BUILT.

**Why this one is next: it is the only remaining boundary whose prize is large
enough to move the standing objective.** The combine boundary is closed by
measurement (exp_03/exp_04: its whole prize is ~300 µs and its ceiling is
~0.88x production). The pre-M6 region is ~960 µs of a 6,886 µs kernel, and
CLAUDE.md attributes 757 µs of it to M1+M2. Recovering even half of that is
worth more than everything the combine boundary can offer.

**Status: premise VERIFIED, design sketched, deliberately not built tonight.**
The reason is in `## Why not tonight`, and it is a schedule judgement, not a
technical objection.

## The premise is alive — verified from source

CLAUDE.md marked "our 757 µs being *exposed* is `INFERRED`, profile first". It
is not merely inferred. The megakernel puts **four grid-wide barriers** between
the dispatch and the first GEMM, so M6 cannot begin until M1–M5 have completed
across the entire grid:

| line | construct | publishes |
|---|---|---|
| `:937` | `hkp::grid_barrier(bar, tid, bar_err)` | end of M2 (chunk-acquire, histogram, `row_remaining`) |
| `:1059` | `hkp::grid_barrier(...)` | M3 |
| `:1114` | `hkp::grid_barrier(...)` | M4 — `scratch`, `nvi`, `sei`, `pull_ptr` |
| `:1124` | `hkp::grid_barrier(...)` | M5 — `sti`, `swt`, `pull_src`, `part`, `sc_dst` |

A `grid_barrier` costs the straggler, so the pre-M6 region is a **sum of
maxima**: four separate global waits, each paying the slowest CTA. There is
zero overlap between the dispatch and M6 today.

Note what is *already* well built and must not be undone: **M2's chunk acquire
is per-`(source, chunk)`, not grid-wide** (`:923` polls
`chunk_ready[s][c]` on a wave stripe). So M2 already overlaps with peers' M1
sends. The exposed cost is therefore **the tail of the all-to-all** — the
slowest peer's slowest chunk — plus three more barriers behind it, not the bulk
of the transfer.

## Phase map (mps kernel)

| phase | line | what it does | comm? |
|---|---|---|---|
| M0 | `:528` | retire wait, per-block counter zero | no |
| M1 | `:636` | qpush: contiguous LDS row + ProtoLL128 `putPackets` to peers; publishes `chunk_ready[s][c]` (`:834`) | **send** |
| M2 | `:852` | per-`(s,c)` chunk acquire, unpack, histogram, `row_remaining` | **receive** |
| M3–M5 | `:1037` | plan + scatter (`k0pf4_dsort` verbatim, inline) | no |
| M6 | `:1129` | N2 phase 1 (G-stacked) | no |
| M6.9 | `:1144` | role selection — service pool is the **static tail**, `bid >= nct − C` (`:1169`) | — |
| M7 | `:1164` | N2 phase 2 | no |

## What COMET layer-0 requires here

COMET decomposes along the token dimension M and computes local tokens first.
Mapped onto this kernel, that is **not** "reserve CTAs for M1/M2" — it is a
restructuring of the M2→M6 chain into two passes:

1. **Local pass.** Tokens this rank already owns need no `chunk_ready` wait at
   all. Plan, scatter, and run M6 on them immediately.
2. **Remote pass.** As each `(source, chunk)` lands, plan/scatter that source's
   tokens and add their tiles to M6's work list.

The service pool would own the second pass's bookkeeping while compute CTAs
chew through the first pass's tiles — a genuine CTA role split at a new
boundary, satisfying the mandate.

## Blast radius — the honest risk list

1. **M3–M5 is `k0pf4_dsort` inlined verbatim.** It is a global sort of tokens by
   expert. Splitting it into local-then-remote passes means the sort no longer
   sees all tokens at once, and every index it produces (`nvi`, `sei`,
   `pull_ptr`, `sti`, `swt`, `pull_src`, `sc_dst`) is currently built under the
   assumption of a single global pass.
2. **`pull_ptr` / `pull_src` are consumed by M8** (`:1420` onwards) and are the
   combine's addressing. Any change to token ordering changes them.
3. **`row_remaining` is set in M2** and consumed by the M7.6 service loop as
   `target` (`moe_mps_adapter.cuh:322`). A two-pass M2 changes when it is final.
4. **`row_ready` is indexed by `[producer][row]`** with row ids independent per
   producer. A local-first reordering must not renumber rows across the two
   passes.
5. **The three barriers at `:1059/:1114/:1124` are publication points.** Any of
   them removed or made partial needs a per-consumer readiness flag to replace
   it, which is exactly the class of change that produced the (measured, benign)
   ordering hole on the combine side.
6. **The register constraint from exp_04 applies unchanged.** A dispatch service
   CTA gets the same VGPR 256 / AGPR 256 as an MFMA CTA and has no spare
   registers; LDS headroom is 8,412 B of 163,840 and the 155,428 B parity gate
   is byte-exact. Any design needing per-lane staging will spill, exactly as the
   MLP fan-out did.

## Staged build plan — cheapest informative stage first

**Stage 0 (no device code, highest information per unit risk): measure the
exposure.** Nothing in the current harness reports it, and
`K0_MPS_DEBUG_STOP` aborts the ranks (exitcode 2, verified tonight) so the
bisect ladder is unavailable. Cheapest replacement: reuse the existing MPS
timestamp block, which the kernel already writes (`K0P6_MPS_TS_*`) and the host
already allocates but **never reads** (`e004pf_k0pf_ab.py:1504-1505`). Add two
timestamps — one at M2's barrier, one at M6's entry — and a host-side print.
That is a small, additive harness edit of the kind CLAUDE.md permits with a
logged note, and it converts the 757 µs from `INFERRED` to measured. **Do this
before writing any of stage 1.**

**Stage 1 (device, contained): local-tokens-first ordering only, no role
split.** Keep all four barriers. Only change the order in which M6 walks its
tile list, so local-origin tiles come first. If the exposure is real but the
barriers dominate, this yields nothing — and that is a cheap, decisive negative.
If M6's tile order turns out to be driven by an index buffer rather than derived
from `blockIdx`, this stage may be a plan-side permutation with no device change
at all, which would make it very cheap. **That question is the first thing to
settle in the source.**

**Stage 2 (the real experiment): split M2→M5 into local and remote passes and
reserve a service pool for the remote pass.** Only attempt with stage 0's
measurement in hand and stage 1 green.

## Why not tonight

Stage 2 is multi-hour surgery on `k0pf4_dsort`, the most delicate part of the
kernel, under a byte-exact LDS gate, an exact ArchVGPR/AGPR parity gate, and a
correctness ladder that must pass on all 8 ranks. Started at this hour it would
most likely end as a half-built tree at dawn, and the ratchet rule says never
regress the best candidate to chase a hypothesis. Stage 0 is the correct next
action and it is cheap.

## Primitives (anticipated)

The two-pass design needs "publish that source `s`'s tokens are planned, and let
any CTA consume tiles from any planned source" — a **partial, source-scoped
barrier**. The library has `publish_tile_release` / `wait_tile_acquire_into` in
`roles.cuh`, which is the right shape, and `bounded_observe_acquire_into` in
`completion.cuh`. What is missing is the same gap the combine side hit twice
tonight: a **group-scoped release whose drain covers writers outside the calling
wave**. If that primitive is added for the combine path, it is directly reusable
here, which is an argument for adding it properly rather than patching the
combine call site.
