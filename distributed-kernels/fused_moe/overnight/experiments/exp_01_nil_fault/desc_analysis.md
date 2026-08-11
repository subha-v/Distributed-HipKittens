# exp_01 — descriptor dump analysis (first hard evidence)

Source: `K0_MPS_DESC_DUMP=1 K0_MPS_TRACE=1 K0_MPS_DEBUG_STOP=6`, run at
2026-08-11T07:02Z, output `$HOME/k0-mok-mps-descdump/run1/mps_desc_rankx.txt`.
Raw words preserved in `../../tools/map_desc.ps1` (the analysis script itself
carries the values, so this result is reproducible without the node).

The run's `exitcode 2` is EXPECTED and not a fault: a debug-stop run produces
zero output, so `rel_L2=1.0` fails the correctness gate. `progress_rank0.log`
reads `pf6mps_mega_body enter` → `pf6mps_mega_body launch-returned`, i.e. the
kernel entered and returned cleanly. No `Memory access fault` line anywhere.

## Result 1 — the config path is PROVEN CORRECT (subsystem cleared)

`desc[62] = 268567048 = 0x10020208`, and it decodes byte-wise to exactly the
requested `K0_MPS_CFG="C=8,g=2,mode=2,flush_rows=16"`:

| byte | value | field | requested |
|---|---:|---|---:|
| `[7:0]`   | 0x08 | `C` | 8 |
| `[15:8]`  | 0x02 | `g` | 2 |
| `[23:16]` | 0x02 | `mode` | 2 |
| `[31:24]` | 0x10 | `flush_rows` | 16 |

So host→descriptor→device config plumbing is end-to-end correct. Config
mis-parse is eliminated as a fault cause, and the sweep in mission step 4 can
trust `K0_MPS_CFG` to mean what it says.

## Result 2 — `desc[61]` is in the WRONG ADDRESS FAMILY

Sorting all pointer words by address exposes two disjoint families:

| family | address range | alignment | count | reading |
|---|---|---|---:|---|
| **S** | `0x7C5C8DE00000` … `0x7C5FDA600000` | all 2 MiB-aligned | 8 | mori symmetric heap segments |
| **T** | `0x7C97F7E00400` … `0x7CA004DFFC00` | 256 B / 512 B packed | 43 | torch caching allocator |

Family **S** = words 35, 36, 37, 27, 28, 2, 32, 8 — every one ends in
`00000` (2 MiB aligned), the signature of a symmetric-heap segment. Family **T**
is packed at 256/512 B, the signature of torch's caching allocator.

**`desc[61] = 0x7C9865820900` is in family T**, sitting exactly 256 bytes after
`desc[31] = 0x7C9865820800`. That is a small packed allocation neighbouring
other torch tensors.

This matters because the push path translates that pointer for peers via
`peer_ptr(local) = local - local_heap_base + peer_base[r]`. That arithmetic is
only meaningful if `local` lies inside `[local_heap_base, local_heap_base +
heapSize)`. A family-T pointer does not, so **every translated peer address
derived from `desc[61]` is garbage**. Note also that the existing host
alignment check on word 61 cannot catch this: `0x…900` is 256 B aligned, and so
is a torch pointer, and so is `nullptr`.

Corroborating gap evidence: there is a **448 MiB gap (469,762,048 B exactly)**
inside family S, between `0x7C5E89000000` (w37) and `0x7C5EA5000000` (w27) —
i.e. a 448 MiB symmetric-heap region does exist and is described by the
descriptor. The slots arena the design calls for is present in the heap; word 61
just does not point into that family.

## Status of the pre-registered decision rule

The `plan.md` rule fires branch two: `desc[61] != 0` but **outside** the heap
family ⇒ the handoff's original suspect #1 (wrong pointer flavor / not
heap-relative) is LIVE. My sharper "allocation returned null" variant (H1) is
FALSIFIED — word 61 is non-null.

Still to nail down before the fix is written:
1. The exact 9 words behind `desc[55]` (`local_heap_base`, `peer_base[0..7]`),
   so range membership is checked against real numbers rather than inferred
   from alignment families. `K0_MPS_DESC_DUMP` currently dumps the descriptor
   but NOT the snapshot it points to — that is a one-line diagnostic gap worth
   closing permanently.
2. Whether `(nil)` is explained by this. A family-T pointer translated with
   family-S bases gives a wrong-but-large address, not 0. So either a second
   null pointer exists in the tail, or the fault address is being reported after
   a wrap/round. This must be explained, not hand-waved: an unexplained `(nil)`
   means an unexplained bug.
