# M23 "Ragged Seal" — implementation notes

Implements `../M23_RAGGED_SEAL_DESIGN.md` §8 as a **post-apply, in-container
patcher**. No HIP, no re-capture, no ABI change, no descriptor-layout change.

| Artifact | Path |
|---|---|
| The patcher (deploy copy) | `<scratchpad>/m23_patch.py` |
| The patcher (repo copy, byte-identical) | `m23_patch.py` |
| CPU tests | `test_m23_patch.py` |
| L0 offline replay harness | `offline_dispatcher_replay.py` |
| This file | `M23_IMPL_NOTES.md` |

Marker: `PF4H_M23_RAGGED_SEAL_V1`. Idempotent, fatal on any anchor mismatch,
plain string replacement with every anchor counted `== 1` — the same discipline
as `coverage_patch.py`.

---

## 1. Chain position (must not be inverted)

```
apply.py                        # installs the shim + the V3_M15 vLLM patch
  → python /covpatch/coverage_patch.py   # M15_COVERAGE counters + EPLB relaxes
    → python /covpatch/m23_patch.py      # THIS
      → vllm serve
```

`m23_patch.py` **exits 1** if `PF4H_COVERAGE_PATCH_V1` is absent from a
PF4H-patched `gpu_model_runner.py`. The single exception is a plain/rccl image
where `PF4H_INTEGRATION_PATCH_V3_M15` is itself absent — then both patchers
no-op and exit 0, matching `coverage_patch.py`'s behaviour exactly.

Because the coverage patch rewrites the seal region first, **every m23 anchor
inside that region is written against the post-coverage text** (the
`_cov["fail_min_tok"] = min(...)` tail, the `_cov["b4096_steps"] += 1` line and
the `M15_COVERAGE` `logger.info` block are all part of m23's anchors). The tests
prove this by running the real `coverage_patch.py` against mirror copies first.

---

## 2. Files patched, and what lands in each

| File (in container) | Design edit | Change |
|---|---|---|
| `vllm/v1/worker/dp_utils.py` | **E1, E2** | staging tensor `(4,dp)`→`(5,dp)`; `pf4h_ready` on row 4; `_post_process_pf4h_ready`; `pf4h_ready` / `return_pf4h_ready` threaded through `_synchronize_dp_ranks` and `coordinate_batch_across_dp` as **additive** kwargs |
| `vllm/v1/worker/gpu_model_runner.py` | **E3, E4, E5** | `_pf4h_m23_ragged_enabled` + `_pf4h_local_readiness`; readiness computed pre-all-reduce; `b4096_unanimous` / `pf4h_ragged_seal`; uniform-decode rescue; `sealed ⇒ PIECEWISE` assertion; `RAGGED_SEAL_RECEIPT` counters |
| `vllm/forward_context.py` | **E7** | `BatchDescriptor.pf4h_exact_b4096` docstring (field name **kept** — graph-key identity + `apply.py` sentinel) |
| `…/pf4h_integration/vllm_full.py` | **E6, E7** | new blocker `PF4H-FULL-020` (refuse `VLLM_MOE_SKIP_PADDING`); module docstring + the `_attest_dp_batch` and `_eligible` comments |
| `…/pf4h_integration/m15_vllm.py` | **E7** | module docstring |
| `…/pf4h_integration/contracts.py` | **E7** | `PREFILL_B4096_BUCKET` comment |
| `…/pf4h_integration/runtime.py` | **E10** | `rows_compared` on `WeightLayoutReceipt` / `CorrectnessReceipt` + `_m23_row_window_blockers` |

Explicitly **not** touched (per §8.1): `cudagraph_dispatcher.py`,
`m15_graph_body.py`, `m15_kernargs.py`, `m15_runtime.py`, the bucket capacities
in `contracts.py`, and `k0pf6gm_device_tile_m15.hip`.

---

## 3. The new predicate as installed

```python
m23_serving = (
    pf4h_graph_target is None            # not the capture drive
    and force_uniform_decode is None     # not a _dummy_run
    and force_num_active_loras is None
    and self._pf4h_m23_ragged_enabled()  # env + mode + dp8
)
b4096_unanimous = bool(
    m23_serving
    and not should_ubatch                                    # all-reduced
    and synced_cudagraph_mode == CUDAGraphMode.PIECEWISE.value  # min over ranks
    and num_tokens_across_dp is not None
    and all(int(v) == 4096 for v in num_tokens_across_dp.tolist())  # [max]*8
)
pf4h_ragged_seal = bool(b4096_unanimous and pf4h_ready_all)  # row 4, all-reduced
```

Every term is DP-unanimous by construction (design §4.1/§4.2). No local term
survives: `uniform_decode` is gone, `has_lora` and the `os.path.isfile`
activation latch are folded into the readiness bit that rides the existing
all-reduce.

`original_num_tokens_across_dp` is kept but **demoted to telemetry**
(`sealed_exact` / `min_orig` / `max_orig` / `sum_orig`).

---

## 4. Deviations from the design (and why)

Four, all deliberate, none changing the design's semantics.

**D1 — E8 and E9 are out of scope for an in-container patcher.**
`apply.py`, `tests/`, `m15_pin/`, `README.md`, `m18_replication.py` are *not*
installed into site-packages (design §"path conventions": the installed shim
copy "differs from the source-of-truth `pf4h_integration/` only by the presence
of" those files), and `apply.py` runs from a read-only mount *before*
`m23_patch.py`. A `RAGGED_SEAL` sentinel added to `apply.py` therefore could not
be checked, and a source-string assertion in `tests/test_full_integration.py`
would never run in the container. **Replacement:** `test_m23_patch.py` asserts
the very same things — every `apply.py` `PATCH_SENTINELS` string survives the
patch (`pf4h_exact_b4096: bool = False`, `return_unpadded_counts: bool = False`,
`PF4H_GRAPH_REPLAY_RECEIPT`, `M15_POST_CAPTURE_COMMIT`), and the M23 marker plus
the `PF4H-FULL-020` refusal are present. If the shim source tree is ever
regenerated, land E8/E9 there as ordinary source edits.

**D2 — E5's counters live in `_determine_batch_execution_and_padding`, not in
`execute_model`/`__init__`.** Every input the receipt needs (`b4096_unanimous`,
`pf4h_ready_all`, `m23_orig_counts`, the final `cudagraph_mode`) is local to
that function; hoisting them to `execute_model` would mean returning four more
values through a 6-tuple that four call sites unpack. Same counters, same
`RAGGED_SEAL_RECEIPT` line, fewer anchors, no signature change. State lives in
`self._pf4h_m23`, lazily created exactly like coverage's `self._pf4h_cov`, so
`__init__` is untouched.

**D3 — E2's "free win" (hoisting one `tensor.cpu()`) is NOT taken.** It is
offered as an optional optimisation, and it is the riskiest line in the whole
edit set: with a single host copy, `_post_process_dp_padding`'s
`num_tokens_across_dp.cpu()` becomes a *view* of the staging tensor rather than
a fresh contiguous copy, and that tensor is handed to `DPMetadata` and indexed
downstream. Minimal-risk choice: keep the four existing syncs and add
`_post_process_pf4h_ready` as a fifth read of 8 int32. Take the hoist as a
separate, separately-validated change if it ever matters.

**D4 — the M23 path is additionally gated on "not a `_dummy_run`".** The design
only excludes the capture drive (`pf4h_graph_target is not None`). But
`_dummy_run` also calls `_determine_batch_execution_and_padding` with
`pf4h_graph_target=None` **and** asserts
`cudagraph_runtime_mode == _cudagraph_mode` immediately afterwards — so a mode
override during warmup would crash the server before it ever serves. `_dummy_run`
is the only caller that passes the `force_*` overrides, so
`force_uniform_decode is None and force_num_active_loras is None` identifies the
real `execute_model` path exactly. This *strengthens* §4.4 row 6 ("capture stays
exact-4096 on all ranks — unchanged").

### Choices made where the design offered options

* **Uniform-decode rescue: landed ON in both arms** (design §4.3 recommendation),
  behind `VLLM_PF4H_B4096_UNIFORM_RESCUE` (default `1`) for a one-run ablation.
  **The stock arm gets it too — re-baseline before the A/B (risk R6).**
* **The rescue is unconditional for a step that actually sealed**, even with
  `VLLM_PF4H_B4096_UNIFORM_RESCUE=0`. This makes `pf4h_ragged_seal ⇒ PIECEWISE`
  structurally true rather than merely asserted, which is what protects
  `M15-DESC-012` (risk R3). Ablating the rescue therefore only affects *unsealed*
  in-bucket steps — which is the only thing an ablation wants to measure anyway.
* **`assert`, not a silent demotion, for the R3 invariant** (design's wording).
  A local demotion would be a *non-unanimous* seal change — the exact class of
  bug (R2) that deadlocks the collective. Crashing is strictly safer than
  hanging, and the invariant is already structurally guaranteed above.
* **Dispatcher left alone** (§4.4 row 9: dropping `not uniform_decode` at
  DISP:338 is "not recommended"); the rescue does the job at the runner.
* **`PF4H-FULL-020` is not gated on `VLLM_PF4H_M23_RAGGED`.** It is inert unless
  someone sets `VLLM_MOE_SKIP_PADDING=1` (default `0`, `envs.py:191/1499`), which
  no campaign arm does, so it is a behaviour no-op for every configuration in
  flight — while remaining defence-in-depth if the flag is ever flipped.
* **E10 requires `rows_compared` only when the ragged seal is active**
  (`VLLM_PF4H_M23_RAGGED != 0` and `graph_target == pf4h`) and the field defaults
  to `None`. Nothing in the deployed tree constructs `RuntimeClosure`,
  `WeightLayoutReceipt` or `CorrectnessReceipt` (verified: zero non-test
  constructors under `pf4h_integration/`), so this cannot block a live
  activation; it constrains the offline evidence path, which is where §5.3's
  hazard lives.

---

## 5. Scope guarantees

**With `VLLM_PF4H_M23_RAGGED=0`** — `_pf4h_local_readiness` returns `False`
(row 4 of the all-reduce is all-zero), `m23_serving` is `False`, so the seal
falls through to `elif pf4h_exact_b4096: … _pf4h_graph_operator_enabled()`, the
rescue never fires (`b4096_unanimous` is `False`), the counters never
increment, and no `RAGGED_SEAL_RECEIPT` is emitted. Byte-for-byte the pre-M23
decision on every step. Proven by `test_predicate`.

**On a stock-target server** (`VLLM_PF4H_B4096_GRAPH_TARGET=stock`, which the
camp3 `stock` arm uses with `VLLM_PF4H_INTEGRATION_MODE=m15`) — readiness is
`False` at its source, so `pf4h_ragged_seal` is permanently `False` and nothing
is ever stamped. The **only** behavioural delta is the uniform-decode rescue,
which the design deliberately lands in both arms for A/B fairness (§4.3, R6):
a rank whose local batch looked like a uniform decode now replays the
already-registered `regular_b4096` PIECEWISE graph instead of running the whole
model eagerly at 4096 padded tokens.

**Rollback** — `VLLM_PF4H_M23_RAGGED=0` reverts everything;
`VLLM_PF4H_B4096_UNIFORM_RESCUE=0` additionally removes the shared rescue for
unsealed steps. Both are read per step, no restart semantics beyond the server's
own env.

---

## 6. New environment variables

| Variable | Default | Effect |
|---|---|---|
| `VLLM_PF4H_M23_RAGGED` | `1` | `0` disables every M23 path (pre-M23 behaviour) |
| `VLLM_PF4H_B4096_UNIFORM_RESCUE` | `1` | `0` ablates the uniform-decode rescue **for unsealed steps only** |

---

## 7. New receipt

Emitted every 100 serving steps per rank, at `logger.info`:

```
RAGGED_SEAL_RECEIPT rank=%d steps=%d in_bucket=%d sealed=%d sealed_ragged=%d
  sealed_exact=%d refused_not_unanimous=%d refused_not_ready=%d eager_b4096=%d
  uniform_rescued=%d min_orig=%d max_orig=%d sum_orig=%d in_bucket_sum_orig=%d
```

`uniform_rescued` (probe P6) and `in_bucket_sum_orig` (probe P1's numerator;
`sum_orig` is its denominator) are additions to the design's §8.2 field list;
everything else is verbatim. `min_orig` reads `1073741824` if no DP vector was
ever seen.  A `RAGGED_SEAL_RECEIPT_FINAL rank=N k=v ...` line is also registered
via `atexit` (guarded by a bare `try/except`, because logging handlers may be
closed at interpreter teardown) so the last partial 100-step window is never
lost.

**Success criterion (design §8.2): `sealed == in_bucket` and `eager_b4096 == 0`
on every rank.**

The pre-existing `M15_COVERAGE` line is untouched and still emits, so camp3's
validity gate (`b4096_steps ≈ in_bucket ≈ 230/600`) keeps working: `b4096_steps`
now counts M23 seals, while `in_bucket`/`unanimity_fail`/`fail_min_tok` keep
their old (local-descriptor) definitions and give a free before/after contrast
inside one run.

---

## 8. Operator deploy checklist

1. **Stage the patcher** (the campaign wrapper already mounts and chains it):
   ```bash
   scp m23_patch.py <node>:/home/subvadla/eplb_campaign/m23_patch.py
   # run_m15_campaign_eplb_v2.sh:67,300-301,333,340 do the rest:
   #   -v $M23_PATCH:/covpatch/m23_patch.py:ro
   #   ... && python /covpatch/coverage_patch.py && python /covpatch/m23_patch.py && exec vllm serve
   ```
   `camp3_stock_vs_m15_ragged.sh` already exports
   `M23_PATCH=/home/subvadla/eplb_campaign/m23_patch.py` and runs
   `--arms stock,m15 --cells c32p --pairs 2`.

2. **Run the CPU gate before touching the node:**
   ```bash
   python3 test_m23_patch.py            # expect: M23_PATCH_TESTS_OK
   ```
   Set `M23_SCRATCH=<dir>` if the mirror lives elsewhere. If
   `coverage_patch.py` changes, re-run this **first** — it is what proves the
   post-coverage anchors still match exactly once.

3. **L0 (below) must pass on a real trace** before releasing the campaign gate:
   ```bash
   touch /home/subvadla/eplb_campaign/M23_READY   # camp3 blocks on this
   ```
   camp3 also waits for `CAMP1_RC=` in `camp1_20260818.log`.

4. **Both arms must use the same binary and the same
   `VLLM_PF4H_INTEGRATION_MODE=m15`**, flipping only
   `VLLM_PF4H_B4096_GRAPH_TARGET` (§7.3: an unpatched-vLLM baseline is invalid
   because the PF4H patch removes the ≤256 mixed PIECEWISE keys in *both* arms).

5. **Re-baseline.** The stock arm now also gets the uniform-decode rescue. A
   post-M23 candidate compared against a pre-M23 baseline over-credits M23 (R6).

---

## 9. Validation ladder (design §8.3), condensed to runnable steps

### L0 — offline dispatcher replay (no GPU) — **blocking**

```bash
# 0a. sanity: the harness itself, plus a synthetic c32p-shaped trace
python3 offline_dispatcher_replay.py --synthetic --steps 600 --per-rank

# 0b. write / inspect a trace file
python3 offline_dispatcher_replay.py --emit-synthetic trace.jsonl --steps 600
python3 offline_dispatcher_replay.py --trace trace.jsonl

# 0c. THE REAL GATE: replay a logged c32p trace
python3 offline_dispatcher_replay.py --from-m23-trace server_rank0.log
```

To produce the real trace, add one line next to the receipt emission (one rank
is enough — `m23_orig_counts` is already the all-reduced 8-vector):

```python
logger.info("M23_TRACE rank=%d step=%d n_orig=%s",
            self.parallel_config.data_parallel_rank,
            _m23["steps"], list(m23_orig_counts))
```

**Gates:** `sealed == in_bucket`, `refused_not_unanimous == 0`,
`eager_b4096 == 0` on every rank. Exit code 1 if any fails.

> On the *synthetic* trace `refused_not_unanimous` is deliberately non-zero and
> the harness prints an H2 diagnostic explaining why: a rank with a small
> **mixed** batch has no PIECEWISE key under the PF4H patch (DISP:204-212, §7.3),
> dispatches `NONE`, and collapses the synced mode for the whole group so DP
> padding does not happen. That is pre-existing and identical in both arms —
> M23 correctly refuses to seal it. On a real trace this class should be ~0
> (probe P6); if it is not, that is a *scheduler* finding, not an M23 bug.

Reference output on the built-in synthetic trace (600 steps, seed 0):

| | OLD predicate | NEW predicate |
|---|---|---|
| in_bucket / rank | 211 | 211 |
| sealed / rank | 3 | **211** |
| sealed_ragged | 0 | 208 |
| eager_b4096 / rank | **208** | **0** |
| uniform_rescued | 0 | 14 |

— i.e. it reproduces both design findings at once: the ~50× coverage gap **and**
the "unsealed in-bucket steps run eagerly, not on the stock graph" finding.

### L1 — coverage wiring, activation withheld (no hang risk)

One c32p run, `graph_target=pf4h`, **do not create the activation file**.
```bash
grep -o 'RAGGED_SEAL_RECEIPT.*' server*.log | tail -8
```
**Gate:** `in_bucket ≈ 230` per rank, `refused_not_ready == in_bucket`,
`sealed == 0`, `eager_b4096 == 0` (the rescue still runs), no `pperr`.
Also confirms probe **P7**: `refused_not_ready` must stop increasing once the
sentinel appears.

### L2 — mega live, the coverage claim

Same run with `/tmp/vllm-pf4h-enable` present.
**Gates:** `sealed == in_bucket` on all 8 ranks, `sealed_ragged ≈ 0.98 ×
in_bucket`, `eager_b4096 == 0`, zero `pperr`, no `M15-DESC-012` /
`M15-STATE-*` / `PF4H-*` blockers in the log. Diff the
`attest_descriptor_words` receipt against a pre-M23 run — it must be
byte-identical (risk R8).

### L3 — spin headroom (risk R1) — **blocking for L5**

Read `DescriptorSlot.SPIN_DBG` (slot 52, `m15_contracts.py:136`, written at
KERNEL:1238) after the run; report max and p99 spin fill per layer.
**Gate: peak < 25% of `M15_SPIN_LIMIT = 20_000_000`.**
If it fails, raise `spin_limit` — it is a descriptor scalar written at
`m15_runtime.py:939`, no recompile — *and* investigate the cross-rank attention
skew that caused it before claiming any throughput number.

### L4 — exact-token accuracy A/B

Two runs, same binary, same seed, `temperature=0`, flipping only
`VLLM_PF4H_B4096_GRAPH_TARGET`. **Gate:** identical generated token ids, or a
documented first-divergence-index distribution plus the row-restricted tensor
check (§5.2) inside `max_abs ≤ 0.02`, `rel_L2 ≤ 0.01`. Any tensor evidence must
carry `rows_compared = n_orig` (E10) and compare **only** rows `[0, n_orig)` —
padded rows are stale embeddings plus attention residue and disagree
arbitrarily in both arms by construction (§5.3, probe P4).

### L5 — throughput A/B — only after L2–L4

Re-baseline first (§8 step 5). Report tok/s and p50/p99 TTFT/ITL for both arms
**plus both arms' `RAGGED_SEAL_RECEIPT` and `M15_COVERAGE` lines**, so a reader
can verify the candidate arm was actually exercised.

### Runtime probes to collect while up there

| Probe | Where it comes from |
|---|---|
| **P1** token-weighted coverage | `in_bucket_sum_orig / sum_orig` in the receipt |
| **P2** peak spin fill | `SPIN_DBG` slot 52 — L3 |
| **P3** `max_cudagraph_capture_size == 512`, `max_num_batched_tokens == 4096` | log `compilation_config` once at startup; feed the real value to `offline_dispatcher_replay.py --max-capture-size` |
| **P4** non-finite padded rows | `torch.isfinite(hidden_states[n_orig:]).all()` on a debug step, **both arms** |
| **P5** stock dispatches padded rows | log `dispatch_recv_token_num.sum().item()` next to `a1.shape[0]` at MORI:97 — must not scale with `n_orig` |
| **P6** H1/H2 frequency | `uniform_rescued` and `refused_not_unanimous` in the receipt |
| **P7** activation-latch unanimity | `refused_not_ready` reaches a fixed value early and never increases |

---

## 10. Test coverage (`test_m23_patch.py`, 140 checks, CPU only)

1. `coverage_patch.main()` then `m23_patch.main()` on byte-copies of the
   deployed mirror; every anchor matches exactly once.
2. AST-parse of all nine patched/adjacent files.
3. Markers present; every `apply.py` sentinel and every coverage-patch marker
   (`PF4H_EPLB_CONTRACT_RELAX_V1`, `PF4H_EPLB_FULL018_RELAX_V1`,
   `PF4H_EPLB_WEIGHT005_RELAX_V1`) survives; `m23_patch` provably does **not**
   touch `weights.py`.
4. Idempotency: re-running both patchers changes zero bytes.
5. Chain order: m23-before-coverage exits non-zero **and writes nothing**;
   a plain (non-PF4H) tree exits 0 as a no-op; a mangled anchor is fatal.
6. The predicate over the design's decision table: exact-4096, ragged in-bucket,
   `VLLM_PF4H_M23_RAGGED=0`, H1 on both arms, rescue ablation (and its override
   for sealed steps), H2, sub-4096 buckets, ubatching, LoRA (flag and config),
   `VLLM_MOE_SKIP_PADDING`, one cold peer, unlatched local activation, capture
   drive (stamp + refusal + stock target), `_dummy_run`, `dp_size != 8`.
7. The predicate replica is pinned to the installed source: 17 GMR fragments and
   4 dp_utils fragments must appear verbatim.
8. Scope: every new GMR branch is gated on `m23_serving`.
9. The patched `dp_utils.py` **executes** under CPU torch: `_run_ar` stages a
   `(5, 8)` tensor and carries the bit on row 4 (and defaults it to 0 for
   unpatched callers); the reducer tolerates a legacy 4-row tensor; DP padding /
   min-mode still behave; and all three caller arities (3-, 4-, 5-tuple, plus
   both `dp_size == 1` early exits) are exercised — `forward_context.py:304` and
   the two spec-decode callers are unchanged.
10. The L0 harness agrees with the predicate replica on the same table, and its
    CLI runs end-to-end.
