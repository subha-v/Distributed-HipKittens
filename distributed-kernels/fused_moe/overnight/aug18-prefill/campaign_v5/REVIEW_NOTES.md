# campaign_v5 — adversarial review disposition (v5.0 → v5.1)

Two reviewers (code-correctness, fairness-protocol) returned 13 blocking, 15
major and 5 minor findings against the v5.0 assets. This file records what
happened to each one. **Every blocking and major finding is either fixed or
rebutted with file:line evidence below.** Nothing was deferred.

Files touched: `run_m15_campaign_eplb_v5.sh`, `bench_exact_token_ids_v3.py`
(client v3.1), `analyze_campaign_v5.py`, `ARMS.md`, `NIGHT_SCHEDULE.md`.

Re-verified: `bash -n run_m15_campaign_eplb_v5.sh`,
`python3 -m py_compile` on both Python files, the embedded `PYGATE` heredoc
compiled and exercised on four fixtures (pass / missing-arm+inert /
within-class-mismatch / no-peer), the analyzer run end-to-end on a synthetic
6-pair 3-arm campaign plus an unbalanced/void/two-root variant, and the
rotation balance checked exhaustively for (arms, pairs) ∈ {(3,6), (2,6), (3,3),
(4,8)}.

---

## Blocking

| # | finding | disposition |
|---|---|---|
| B1 | c1det gate demands byte-identical SHA across numerically incomparable arms → can never pass | **FIXED.** `numerics_class()` groups arms by KV dtype / parallelism / attention backend / all2all. Exact SHA equality is asserted only *within* a class (m15 vs stock — the comparison the kernel claim rests on); across classes the gate measures a per-prompt agreement fraction and the first divergent index and asserts nothing. ARMS.md §6.7. |
| B2 | gate silently accepts arm subsets; a missing c1det.json is skipped; accuracy-pass failure downgraded to WARNING | **FIXED.** The gate takes the campaign's arm list as `expected` and a missing result for any expected arm is `FAILURE: missing_result`. Both determinism cells are required per arm. The driver's message on a failed accuracy pass now says the arm will be reported as missing, not skipped. |
| B3 | native_default's determinism launch omits `--max-num-batched-tokens` → refuses to start | **FIXED**, and the whole approach changed: the det launch now *lowers* `--max-model-len` to `DET_MAX_MODEL_LEN=4608` and sets the batching window to the same value, on **every** arm including `native_default`. Fixes B3 and the separate major finding that the old 32768 window moved the mega off B4096. |
| B4 / F6 | order balancing broken: list reversal never moves the middle arm; odd PAIRS gives arm 1 an extra position-1 slot | **FIXED.** Rotation by `(pair-1) mod n_arms` with a reversal every `n_arms` pairs, plus a hard preflight that `PAIRS % n_arms == 0` (`ALLOW_UNBALANCED_PAIRS=1` overrides and stamps the receipt). Default `PAIRS` 5 → 6. The analyzer prints a §1.3 position-balance table and turns policy rows for an unbalanced cell into "no claim". |
| B5 | native_mirror lost MORI_GPU_ARCHS/MORI_SHMEM_MODE/AITER while still serving mori_high_throughput | **FIXED.** New `arm_is_vendor_config()` predicate: `arm_is_native()` still means untouched image (no patches, no PF4H mounts); only the vendor-config arms are denied our runtime env. native_mirror gets AITER/MoRI/HSA/HIP and no `VLLM_PF4H_*`. ARMS.md §6.3. |
| F1 | no invocation sets `PROMPT_SOURCE=qsl`; the default is synthetic | **FIXED.** The driver refuses `PROMPT_SOURCE != qsl` without `ALLOW_SYNTHETIC_PROMPTS=1`, and refuses `qsl` without `QSL_PKL`. Every NIGHT_SCHEDULE command exports both. The analyzer emits a "NOT A HEADLINE" banner if any manifest disagrees. |
| F2 | prefix-caching asymmetry + ~11% exact duplicate QSL prompts, both in the candidate's favour | **FIXED twice over.** (a) Prefix caching is off for every arm (`PREFIX_CACHING=1` re-enables it symmetrically); (b) the QSL cursor is now a coprime-stride permutation, so a cell up to pool size has zero exact duplicates, and `duplicate_prompts_in_cell` is recorded. ARMS.md §6.6. |
| F3 | c1det at conc 1 under DP8 leaves the mega inert; no seal receipt from accuracy servers | **FIXED.** `c8det` (conc 8, 64 prompts) runs on the same det server so all 8 DP ranks carry real tokens; the accuracy pass extracts `RAGGED_SEAL_RECEIPT` into `coverage_tail.txt`; the gate FAILs with `candidate_inert` if the candidate shows no sealed steps. |
| F4 | gate is exact-equality with no degradation path and no pairwise breakdown | **FIXED** — see B1. The gate now prints one line per comparison (within-class and cross-class) with agreement counts and the first divergent index, so a FAIL names which pair diverged and by how much. |
| F5 / B14 | B0/B1/B2 share RUN_TAG and ROOT → name collision or silent merge | **FIXED.** `reserve_container_name()` removes a *stopped* leftover (its evidence is already in `server.log`) and refuses a *running* one; `guard_output_dir()` refuses a pair directory that already holds cell manifests; `CAMPAIGN_INVOCATION.txt` refuses a ROOT whose recorded spec differs. NIGHT_SCHEDULE sets a distinct `RUN_TAG` per step. |
| F7 | OPEN_LOOP_BASE_RATE taken from the candidate's own position-1 run and applied to every arm | **FIXED.** Documented and enforced as the **minimum** ceiling across the invocation's arms, with a required `OPEN_LOOP_BASE_RATE_SOURCE` string recorded in the campaign receipt. B0 now calibrates both arm families. The analyzer additionally *suppresses* (not just warns about) any open cell whose arms saw different rates. |

## Major

| # | finding | disposition |
|---|---|---|
| M1 | analyzer index keyed without campaign → cross-invocation ratios labelled "within-pair" | **FIXED.** Key is `(campaign, cell, arm, pair)`; `pair_ratios` iterates `campaign_pairs()` and returns `((campaign, pair), ratio)`. Duplicates are reported in §1.0b, not only on stderr. |
| M2 | rate-mismatch check is advisory; ratios emitted anyway in §3/§4 and in the JSON | **FIXED.** `_open_loop_comparable()` result is consumed: §3 prints "Suppressed", §4 prints "invalid: offered rates differ", and the JSON carries `comparable` and `offered_rates` per cell. |
| M3 / M8 | open-loop cells scored by throughput, which the arrival schedule pins | **FIXED.** `OPEN_LOOP_METRIC = ttft_p99`; `LOWER_IS_BETTER` inverts the displayed ratio so >1 still means better. Per-cell tables now also carry ITL p50 and the input-tok/s column separately. |
| M4 | `achieved_request_rate` counts failed requests | **FIXED** in the client: achieved counts only `result.ok`; `returned_request_rate_including_errors`, `completed_ok`, `returned_total`, `failed` sit beside it; the analyzer's validity table gained a `failed` column and says which figure it shows. |
| M5 | ITL mis-attribution: k−1 zero-ms samples from the first SSE chunk | **FIXED.** Extra tokens in the first chunk are dropped from the series (they have no observable spacing) and counted in `itl_dropped_first_chunk` / `result.itl_dropped_first_chunk_total`, so a per-arm asymmetry in coalescing is visible instead of silently deflating p50. |
| M6 | prompt-SHA table reports OK for a single-arm pair | **FIXED.** §1.1 distinguishes `OK`, `**INCOMPLETE (n of m arms)**` and `**NO SHA**`, and lists the arms actually present. |
| M7 | receipt gate never gates (`check_receipts \| tee` discards status) | **REBUTTED IN MECHANISM, FIXED IN SUBSTANCE.** The claim's mechanism is wrong: `run_m15_campaign_eplb_v5.sh:2` sets `-o pipefail`, so the pipeline *did* take `check_receipts`' status — the real behaviour was the opposite failure, which the other reviewer's minor finding names correctly (a shortfall killed the driver mid-matrix instead of discarding the cell). Both are now moot: `record_receipts()` captures the status, writes `RECEIPT_GATE=PASS/FAIL` into `receipts.txt`, and continues; the analyzer excludes `FAIL` arms from every ratio and lists them in §4.1. |
| M9 | m15 det launch at 32768 batched tokens leaves the B4096 operating point | **FIXED** — see B3. |
| M10 | analyzer's receipt check only flags `" = 0"`; seal coverage never thresholded | **FIXED.** `_receipt_verdict()` parses `receipt NAME = N` and fails below 8, and parses `sealed`/`in_bucket` from the seal line and fails below `MIN_SEAL_PCT = 90`. The driver applies the same bar in `check_receipts`. |
| M11 | wall-clock model optimistic in three places, zero contingency | **FIXED** in NIGHT_SCHEDULE: `stop_server` budgeted at its 180 s bound, per-arm `c32p` runtimes with the native arm's range stated as unmeasured, o50p at 300 s (its arrival span alone is 217 s), `EPLB_PREWARM=0` on every command with the cost of forgetting it priced, ≥1 h of named contingency, and a decision rule that re-prices B2 from B0's measurement. |
| M12 | a rejected startup flag kills the whole driver | **FIXED.** Launch and readiness are guarded; a failure writes `PAIR_VOID.txt` + `ARM_STATUS.txt` and the campaign continues. NIGHT_SCHEDULE gains an explicit "the tuned baseline will not start" decision rule. |
| M13 | native_default cannot participate in the accuracy gate; gate has no pairwise breakdown | **FIXED** — see B3 and B1/F4. |

## Minor

| # | finding | disposition |
|---|---|---|
| m1 | `client_lag_ms` measured at coroutine entry, cannot detect client saturation | **FIXED as far as the design allows.** `_one` now stamps `sent_s` from the measurement epoch as its first statement, i.e. when the event loop actually ran the request, and `client_lag_ms` is computed from that rather than from the spawning wrapper. Residual, documented in the dataclass comment: connection setup inside aiohttp is still folded into TTFT (the connector is unbounded precisely so it is not a queue). |
| m2 | preflight does not validate cell names | **FIXED.** Preflight resolves every cell through `cell_params`, rejects determinism cells, and resolves `cell_rate` for open cells. `cell_is_open` no longer glob-matches `o75`. |
| m3 | schema string and `prompt_generator` changed vs v2 | **ACKNOWLEDGED AND DOCUMENTED, not reverted.** The schema move to the v2 pair is deliberate (the manifest gained `itl_ms` and `arrival`) and `analyze_campaign_v5.py` accepts all three strings while refusing unknown ones. `prompt_generator` changes again in v3.1 *because the generator changed*; a stability receipt that kept its name across a behaviour change would be worse than useless. The node's `summarize_m15_campaign.py` reads vllm-bench field names this client has never emitted — it reported zeros for v1 manifests too — so `write_bundle` now tolerates its failure and drops a `READ_THIS_FIRST.txt` naming the right analyzer. |
| m4 | o50p budgeted at "near two minutes" when its span is 218 s | **FIXED** in both the script comment and the schedule table (300 / 220 / 190 s for o50p / o75p / o90p). |
| m5 | `startup_seconds.txt` is a shared dotfile, not per-arm | **FIXED.** `wait_ready` takes the arm's output dir and writes `startup_seconds.txt` there, so the accuracy/profile/skew passes leave a record too. |
| m6 | receipt shortfall fatal under pipefail rather than discarding the cell | **FIXED** — see M7. |

---

## Residual risks the fixes do not remove

1. **`native_tuned_tp` has never been started on 0.25.1.** AMD's command targets
   0.14.0rc2. B0 is the canary and NIGHT_SCHEDULE names the fallback, but a
   systematically rejected flag still costs an arm-run to discover.
2. **`c8det` is not deterministic.** It exists to make the megakernel run during
   the accuracy pass, and its cross-arm comparison is an agreement fraction. A
   divergence there is expected and is reported with its magnitude; it is not
   evidence of a wrong kernel on its own.
3. **Prefix caching off changes our absolute numbers** relative to the banked
   camp3 receipts. Ratios remain comparable; absolutes do not.
4. **TTFT p99 as the open-loop metric is a choice**, not a derivation. It is the
   metric that responds to saturation, but an arm could win TTFT p99 and lose
   ITL; both are in the per-cell table and neither is hidden.
5. **The offered rate is bounded by the slowest arm.** The o-cells therefore say
   nothing about what the candidate could deliver above that rate — stated in
   the analyzer's §6 caveats.
6. **`m15` vs `stock` bit-exactness has never actually been observed.** The gate
   now *can* pass, which is the point of the fix; whether it *does* is tonight's
   measurement.
