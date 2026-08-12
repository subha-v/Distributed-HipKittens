# exp_01 — the fix, its build gate, and the gate ladder

Companion to `root_cause.md` (the diagnosis). This file is the applied change,
its resource accounting, and the correctness evidence.

## The change

Commit `5dc61fb0`, one file, `+15 −9`:
`distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip`
sha256 after `709f4d99e25a0c69a5007cab2fcb1689f012377d8a3e86c401f1362892170835`.

Restores the parity port's form (`k0pf6gm_device_tile.hip:1030-1051`) inside
`k0p6_mps_m8_batch`:

```370:389:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
    if ((lane >> 3) == t) {
      const int j2 = lane & 7;
      int p = cur;
      int row = 0;
      if (j2 < fanout[t]) {
        p   = pull_src[(size_t)(lo2[t] + j2) * 2 + 0];
        row = pull_src[(size_t)(lo2[t] + j2) * 2 + 1];
        batch_ready = hk_moe::poll_epoch_system(/* ...unchanged... */);
      }
      pbase[t] = base_for(p, row);
    }
```

Three properties of the change, in the order a reviewer should check them:

1. **The in-fanout path is bit-identical.** The two `pull_src` reads and the
   `row_ready` poll stay inside the guard; only their `const` qualifier moved.
   Lanes with `j2 < fanout[t]` compute exactly the base they computed before.
2. **The out-of-fanout default is dereferenceable under BOTH address forms.**
   `base_slot(cur, 0)` (`:1440-1446`) collapses to `slots + (cur*MAXTOK + 0 −
   cur*MAXTOK)*H` = `slots + 0`, the first row of this rank's own slot buffer.
   `base_pull(cur, 0)` (`:1435-1439`) is `peer_ptr(part + 0, cur, symmetric)`,
   this rank's own `part` row 0. Both are inside live allocations, so the
   speculative `uint4` load at `:414-416` reads real bytes, and the accumulate
   at `:423` discards them because it is still `jb + jj < fanout[t]` guarded.
   This is precisely the reference's already-validated behaviour.
3. **`cur` had to be threaded in.** The helper is a template shared by four
   call sites and did not receive the local rank. Added as a positional
   parameter after `lane`; all four call sites updated (`:1455`, `:1461`,
   `:1487`, `:1501`). Additive, no caller semantics changed.

### Why `return` is still correct

The reference uses `continue` inside its own `for (tok0 ...)` loop; the MPS
helper uses `return` because the loop lives at the call site. The four call
sites are two `for (tok0 = gw*NT; tok0 < T; tok0 += nw*NT)` loops and two
`while (true) { claim ticket; if (batch >= nbatches) break; call; }` loops, so
in all four shapes a `return` from the helper resumes the caller's next
iteration. Equivalent. (Independently re-verified in `protocol_review.md`.)

## Build gate — gfx950, ROCm 7.2.4 / LLVM 22

Direct `--genco` build (`overnight-scratch/exp_01/build_fix.log` on the node):

| metric | parity port | MPS before fix | **MPS after fix** |
|---|---:|---:|---:|
| TotalSGPRs | 106 | 104 | **104** |
| ArchVGPRs | 256 | 256 | **256** |
| AGPRs | 256 | 256 | **256** |
| ScratchSize [B/lane] | 36 | 60 | **60** |
| LDS [B/block] | 155,428 | 155,428 | **155,428** |
| Occupancy [waves/SIMD] | 1 | 1 | **1** |

**The fix is resource-neutral.** `root_cause.md` flagged the reference form's
two extra live values (`p`, `row`) as a plausible source of the MPS build's
+24 B scratch; that hypothesis is now falsified — restoring the reference form
changed the scratch figure by zero bytes. The +24 B has some other origin and
remains the open budget item.

## Gate ladder — PASSED, first world-8 run

`exp01fix_m2_C8g2fr16`, 8 ranks, `K0_MPS_CFG=C=8,g=2,mode=2,flush_rows=16`
(the exact configuration that faulted deterministically before the fix).

| gate | result |
|---|---|
| first-launch memory fault | **none** — the `address (nil)` abort is gone |
| plan equivalence | `pass=True`, all 10 diff counters 0, `perr=0/0` |
| quant / gather / combine bit-exactness | `pass=True`, `byte_diffs=0`, `bf16_bit_diffs=0` |
| MoK correctness, `mps_mega` | `max_abs=0.035156 relative=0.008293 pass=True` (gates: ≤ 0.1 / ≤ 0.1) |
| `pperr` | **0** |
| negative control | `control_fails=True` — the deliberately-broken arm still fails |
| soak | `[MPS SOAK] completed=600/600 pperr=0 pass=True` |

Timing from that run is **not** reportable: `WARMUP=1 TIMED=1` and the harness
itself labels it `status=valid_diagnostic`. The decision numbers come from the
5-process campaign, recorded in `result.md`.

## Note on the `[MARK] eager ... pass=False` line

Both arms print `pass=False` on the `[MARK] eager` line while the `[MOK GATE]`
line prints `pass=True` for the same arm and iteration — including for
`production`, whose `rel_L2=0.00651` is its known-good value. This reproduces
secondary item 5 of `root_cause.md`. Conclusion unchanged: **the `[MARK] eager`
`pass` field is not a correctness gate** and must not be read as one. The gates
are `[MOK GATE]` (`max_abs`, `relative`), `pperr`, `[MARK] control_fails`, and
`[MPS SOAK]`. Left as-is deliberately — a harness predicate is not something to
"fix" mid-campaign when the real gates are unambiguous and passing.
