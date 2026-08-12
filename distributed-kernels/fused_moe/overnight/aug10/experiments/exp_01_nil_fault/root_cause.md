# exp_01 — ROOT CAUSE of the `address (nil)` fault (found, not yet fixed)

**Status: root-caused with high confidence, fix NOT yet written or gated.**
Found by static diff of the MPS M8 body against the working reference. The
overnight loop was paused by the user before the fix landed; this file is the
handoff.

## The defect

In `k0pf6gm_device_tile_mps.hip` M8, `pbase[t]` is assigned **only inside** the
`j2 < fanout[t]` guard:

```363:383:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
  unsigned long long pbase[NT];
  for (int t = 0; t < NT; ++t) {
    pbase[t] = 0ull;
    if ((lane >> 3) == t) {
      const int j2 = lane & 7;
      if (j2 < fanout[t]) {                 // <-- assignment trapped in here
        const int p   = pull_src[(size_t)(lo2[t] + j2) * 2 + 0];
        const int row = pull_src[(size_t)(lo2[t] + j2) * 2 + 1];
        pbase[t] = base_for(p, row);
      }
    }
  }
```

The working reference assigns it **unconditionally**, using safe defaults
`p = cur; row = 0;` so that an out-of-fanout lane still holds a *valid*
pointer:

```1030:1051:distributed-kernels/fused_moe/k0pf6gm_device_tile.hip
      int p = cur;
      int row = 0;
      if (j2 < fanout[t]) { p = pull_src[...]; row = pull_src[...]; }
      pbase[t] = (unsigned long long)hk_moe::peer_ptr(
          part + (size_t)row * (size_t)K0P6_H, p, symmetric);
```

The consumer is byte-identical in both files and is **not** fanout-guarded —
only the accumulate downstream of it is:

```408:410:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
      const unsigned long long pb = __shfl(pbase[t], 8 * t + jb + jj);
      v2[t][jj] = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const unsigned char*>(pb) + off);
```

So whenever `jb + jj >= fanout[t]`, the shuffled `pb` is `0` and the lane
issues a speculative load from `0 + off`, where `off = (c<<10) + (lane<<4)`.
At `c == 0, lane == 0` that address is **exactly 0**.

## Why this explains every observation, and the rival hypotheses do not

| observation | explained? |
|---|---|
| fault address is literally `(nil)` | yes — `off == 0` on the first iteration of the first lane |
| all 8 GPUs, instantly, first launch | yes — `fanout` is a popcount of distinct destination ranks, so `fanout < 8` on essentially every token, and dead tokens (`tok >= T`) give `fanout == 0` on all eight lanes |
| `debug_stop` 1–6 CLEAN | yes — the defect is in M8, which those stops never reach |
| **`mode=1` with `C=0` still faults** | yes — the defect is in the shared M8 template, so `mode`, `C` and `pull_fallback` are all irrelevant |
| mode 0 (push OFF) still faults | yes, same reason |

That last row is what kills the alternatives. The handoff's suspect #3
(`run_service` internals) cannot be it because at `C=0` **no service CTA
exists**. Suspect #1 (slot-61 pointer flavour) is dead by measurement: word 61
is a valid mori heap pointer 1.71 GiB into a 32 GiB heap, and `base_slot` never
calls `peer_ptr` at all. My own sharper variant of #1 (a null from a failed
448 MiB `shmem_malloc`) is dead twice over — the request had 73× headroom, and
`moe_host_abi.hpp:197-201` throws on a null `binding.slots` **before** the word
is written, so a null could never have reached the device.

## The fix (designed, NOT yet applied)

Restore the reference's safe default so every lane holds a dereferenceable
pointer regardless of `fanout`: hoist the `pbase[t]` assignment out of the
`j2 < fanout[t]` guard and initialise `p = cur; row = 0;` as the reference
does. The out-of-fanout lanes then load real bytes that the accumulate
discards, which is exactly the reference's (already-validated) behaviour.

Guard rails for whoever applies it:
- This is a **correctness restoration to reference parity**, not an
  optimization. It must not change the in-fanout path at all.
- Re-check the resource tuple afterwards; the reference form keeps two extra
  live values (`p`, `row`) across the loop, which is a plausible source of the
  MPS build's +24 B/lane scratch. Budget: exact ArchVGPR/AGPR parity, scratch
  as near 36 B as possible, no scratch ops inside either MFMA span.
- Then run the full gate ladder from scratch — modes 0/1/2 smoke, negative
  control, 600-epoch soak — before any timing.

## Secondary items raised by the audits (not blocking, do not lose)

1. **`symmetric == nullptr` reaching `base_pull`** (`..._mps.hip:1409-1413`,
   deref at `:1431`). In mode 2 with `pull_fallback=0` both `symmetric` and
   `part` are `nullptr`; the runtime `if (m8_pull)` should prevent the load,
   but the two `k0p6_mps_m8_batch` instantiations differ only in the lambda, so
   an if-conversion would turn this into a nil load. **Verify in ISA** after
   the Rank-1 fix — if the fault persists, this is the next candidate.
2. **`translate_peer` fails closed to `nullptr`** for out-of-range ranks
   (`peer.cuh:40-49`, via `pgl.cuh:124-137`) and **no MPS call site checks the
   result**: `moe_mps_adapter.cuh:232`, `:262`, `..._mps.hip:1535`. Currently
   guarded by `live = r < env.t_ext` (`moe_mps_adapter.cuh:319-321`), so
   reachable in principle only. This is a **primitives finding**: a fail-closed
   null that no caller checks is a footgun the library should not hand out.
3. **`K0_MPS_DEBUG_STOP` writes descriptor slot 49**, which is the donor's
   `K0_PF6_DEBUG_PHASE` word (`e004pf_k0pf_ab.py:2694`); the dump confirms
   `[49] == 6`. The bisect's meaning therefore depends on the kernel's
   interpretation of an inherited donor slot. Worth confirming before quoting
   "M0–M7 clean" as settled.
4. Dead code at `e004pf_k0pf_ab.py:2770-2772` re-patches
   `_pf6mps_desc_list[49]` after the tensor was already built at `:2764`.
5. `production` reports `rel_L2=0.00647 pperr=0` — its known-good value — yet
   `pass=False`. A known-good arm marked failing means the `pass` predicate
   measures something other than its name; do not trust it as a gate until
   explained.
