# exp_38 — restore the mode-12 ratchet that mode 14 took away

## The defect

Commit `291dfa08` ("mode 14 (coarse readiness)") regressed the mode-12 ratchet
arm by **+726.9 µs**, measured same-session with the same config and the same
`production` denominator:

| tree | mode-12 ratchet `C=16,g=353,mode=12,flush_rows=16` |
|---|---|
| rev 26, `f113d73f` | 6,497.3 µs |
| pin `291dfa08` (rev 28) | 7,224.2 µs (n=5) |

`production` (7,700) and `pf6gm_mega` (6,905) were unchanged to <5 µs, so the
node and the session were fine. Mode 12's own source lines were never edited.

The diagnostic signature is that **exp_24's injection bound went inert**: the
`g=353` vs `g=65` contrast (throttle on vs off) was −613.5 µs at rev 26 and only
−1.8 µs at the pin, while the throttle's four `vmcnt` instantiations were still
present in the ISA at identical counts and **every field of the resource gate
passed** (SGPR 106 / VGPR 256 / AGPR 256 / scratch 128 B / LDS 155,496 /
MFMA 180 / pk_add_bf16 282, zero scratch ops in either MFMA span).

A later commit `979cd233` / working-tree state added the exp_23 per-CTA event
ring to the same two files (rev 29). Its parity gate also passed **on the
resource tuple**, which we now know is not sufficient evidence, so the ring is
treated as equally suspect until it clears the stronger gate.

## The gate

Make the **default** build of `codex/distributed-hipkittens-scaffold` produce a
`.text` section that is **byte-identical (sha256)** to the one from `f113d73f`
(rev 26), while keeping the mode-14 and exp_23 code in the tree behind
compile-time flags that default OFF. `.text` byte-identity is the only parity
gate we now trust; the resource tuple is demoted to a secondary check.

## Method

1. **Reference.** Build `f113d73f`'s two files in an isolated TU, unbundle with
   `clang-offload-bundler --type=o --unbundle
   --targets=hipv4-amdgcn-amd-amdhsa--gfx950`, `llvm-objcopy
   --dump-section=.text`, record sha256. Reuses `aug11/tools/e27_fp.sh`'s
   procedure. → `tools/e38_10_gate.sh`
2. **Guard.** `K0P6_MPS_ENABLE_MODE14` (default 0) around every mode-14
   addition; `K0P6_MPS_E23_RING` demoted 1 → 0. Iterate until DEF `.text` ==
   REF `.text`. Flag-on builds must DIFFER from REF, or the flag is not
   reaching the code and the arm would be a lie.
3. **Attribute.** Nine-site ablation (`tools/e38_40_ablate.sh`): rename the Nth
   guard to a per-site macro and build with exactly one site live, so the
   perturbation can be charged to specific source edits rather than to
   "mode 14" as a blob.
4. **Mechanism.** `tools/e38_30_isa.py` measures the thing the instruction
   census cannot see — not how many `vmcnt` waits exist, but **how many remote
   atomics can be in flight between them**. A throttle only binds if the code
   would otherwise exceed its cap.
5. **GPU confirm.** Default build, `C=16,g=353,mode=12,flush_rows=16`,
   **stamps off**, 5-rotation campaigns, alternated with `g=65` so session
   drift cannot masquerade as the throttle effect. Two required checks:
   the end-to-end ratchet (~6,482.7 µs / ~0.8408× production) and the revived
   `g=353` vs `g=65` contrast (back near −613 µs, not −1.8).

## Pre-registered expectations

- The guard that matters is the **mode-14** one, because the +726.9 µs was
  measured at rev 28, which does not contain the ring at all.
- The ring will not reach `.text` identity when compiled in (it adds a store at
  every stamped boundary), so it must default off regardless of blame.
- Falsifier for the mechanism claim: if the perturbed builds show the SAME
  epilogue issue-run distribution as rev 26, then register pressure is not the
  story and the inert throttle needs a different explanation.
