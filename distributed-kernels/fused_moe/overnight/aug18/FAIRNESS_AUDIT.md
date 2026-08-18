# Training fairness audit — mega arm vs production turbo_gg-FP8D (2026-08-18)

Three-auditor (Opus) audit of the training A/B, prompted by the serving-side
fairness discovery.  Verdict: NO serving-style pathology (the mega kernel
runs 100%% of MoE layer-microbatches), but the comparison is unfair in both
directions: (1) the production FP8D bar runs its MoE in BF16 (delayed recipe
silently disables turbo fp8; zero weight-quant cost on their side); (2) the
mega arm issues 30.5%% more MoE FLOPs (z-regen, 20%%-dead wgrad padding,
producer, staging) — ~176 ms/iter self-inflicted; (3) measurement bias ~23 ms
against the mega arm (10-iteration avg tail + production-only end-of-run dip
+ cross-day drift); honest steady-state ratio 0.845x (1,589.5 vs 1,343.0).
Backbone handicap real but small (~15-35 ms).  Missing controls: vanilla-FP8D
with stock MoE, and the truly fair arm: turbo backbone + mega MoE.

============================== BACKBONE AUDITOR ==============================

--- findings ---

HEADLINE: the training comparison does NOT have a serving-style pathology. The mega kernel executes on 100% of MoE layer-microbatches (32/32 per iteration; the hook's shape guard `s*b != _T` at mega_sitecustomize.py:50 cannot fire at seq 4096 / mbs 1). The backbone/chassis handicap is real but SMALL — ~7-15 ms/iter, i.e. ~3-5% of the 293.7 ms BF16 gap and ~4% of the 259-309 ms mega-vs-production gap. The chassis is not the story.

The two biggest fairness defects point in OPPOSITE directions and neither is about the chassis:

F1 (FOR the mega arm — the largest finding). The banked production bar `t0_turbo_gg-FP8D` (fp8: hybrid, fp8_recipe: delayed, 1,328.4 ms / 24,667 tok/s) is NOT matched precision. Under the `delayed` recipe Primus returns a null turbo quant config ("Primus-Turbo not support delayed scaling"), so `enabled_turbo=False`, `PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled()` is False, and `PrimusTurboGroupedLinear.forward_internal` takes the BF16 branch. Production's MoE expert GEMMs run BF16 in the banked bar, and production pays NO expert-weight fp8 quantization (`_maybe_create_quantized_weight_buffers` sits inside that gate). Two banked claims break: (a) "the matched-precision race is ours 19.4k vs their delayed-recipe 24.5k" — it is fp8-MoE vs bf16-MoE; (b) "EVERY fp8 trainer pays [the weight-quant cost], including production's" — production's bar pays zero, so the mega arm's ~70 ms/iter producer (17.5 ms/layer x 4) is an unmatched cost, not a shared one. The corrected reading actually STRENGTHENS the capability-gap story: the only production FP8 recipe that trains on this build is the one that silently drops fp8 from the MoE entirely.

F2 (FOR the mega arm). The mega arm returns no probs gradient (`None,  # probs grad: v2` in moe_swap.py's backward). Because `probs` is the only differentiable edge out of `route()`, this zeroes BOTH the router weight gradient and the aux-loss gradient (MoEAuxLossAutoScaler never fires). It skips ~5-9 ms/iter of unfused router-backward work AND means the two arms optimize different parameter sets — so the loss comparison (mega 9.279e-3 vs prod 1.05e-2) is not a quality claim. Under `force_load_balancing` the router is trained on RANDOM logits (RandomSTE), so dropping its gradient removes a noise source and plausibly HELPS the mega loss.

F3 (AGAINST the mega arm, small). The chassis delta is exactly 6 flags out of 816 effective args. Only two touch anything outside the mega seam: `use_turbo_rms_norm` (patches `te.pytorch.RMSNorm` globally -> every norm in the model) and `moe_use_fused_router_with_aux_score` (router runs OUTSIDE the swap seam — mega_sitecustomize.py:61 calls stock `self.route()`). Everything else the mega arm loses (DeepEP dispatch/combine, turbo grouped GEMM, sync-free stage) is INSIDE the seam and is replaced by the mega kernel, so it is not a handicap.

F4 (missing control). `t0_vanilla-FP8D.yaml` was never run WITHOUT the hook. The only same-chassis A/B that exists is BF16: vanilla+legacy 1,677.0 vs vanilla+mega 1,660.5 = the mega swap is worth 16.5 ms/iter (+1.0%) on its own chassis, while the turbo chassis+MoE together are worth 293.7 ms (+17.5%). That single number is the most important one in the whole comparison and it is currently unstated.

F5 (reproducibility). The banked mega bar ran `K0_MEGA_FWD_SYNC=0`, a setting the team's own V6 write-up calls unstable on long runs ("barrier timeout at iter 6 ... long runs need sync1 until the retire-latch port"), n=1, on 2026-08-18. Production is n=2, stable config, 2026-08-15.

--- asymmetries ---

=== EXACT CHASSIS DIFF (816 effective args each; 6 real deltas + 1 inert) ===
| flag | production | mega | scope |
| enable_primus_turbo | True | False | master gate |
| use_turbo_rms_norm | True | False | BACKBONE — global RMSNorm swap, outside seam |
| moe_use_fused_router_with_aux_score | True | False | ROUTER — outside the mega seam |
| turbo_sync_free_moe_stage | 1 | 0 | inside seam (also auto-enables the fused router) |
| use_turbo_deepep | True | False | inside seam (dispatch/combine) |
| use_turbo_grouped_gemm / moe_use_legacy_grouped_gemm | True/False | False/True | inside seam (expert GEMM) |
| turbo_deepep_num_cu | 80 | 32 | INERT (deepep off) |
Derived patch deltas (from the applied-patch logs): `megatron.turbo.rms_norm` + `megatron.turbo.moe_dispatcher` + `megatron.moe_alltoall_dtoh_turbo_grouped_gemm` + `megatron.fp8.context` + `transformer_engine.pytorch.fp8` applied in production, SKIPPED in mega. The last two are gated on `fp8 AND enable_primus_turbo`, so the mega arm's non-MoE fp8 runs through stock Megatron's `get_fp8_context`.

=== AGAINST THE MEGA ARM (production advantaged) ===
A1. Backbone fusions absent (turbo RMSNorm + fused router forward). The only genuine chassis handicap.
A2. Primus fp8-context patch skipped — an unaudited config delta on the non-MoE fp8 path. Measured net effect: NONE (mega's fp8 gain on its chassis, 72.7 ms, EXCEEDS production's on turbo, 53.8 ms).
A3. Temporal confound: production banked 2026-08-15 (2 reps x 260); mega banked 2026-08-18 (1 rep). A same-day production smoke on 08-17 ran 1,348.4 ms inst vs the 08-15 full run's 1,269.6 inst. No interleaved re-bank.
A4. No same-chassis FP8 control (see F4).
A5. Mega re-derives topk from dense probs (`torch.topk(probs, k=8)` in the swap forward) where production's dispatcher consumes routing_map. Measured 0.029 ms/call.
A6. Mega pays a per-iteration fp8 weight-refresh producer with no production counterpart (see F1) — but this is a genuine cost of its design, not a rigging artifact; it just must not be excused as "everyone pays it".

=== FOR THE MEGA ARM (production disadvantaged / mega credited) ===
B1. Production's MoE runs BF16 in the banked bar (F1). Costs production ~0 ms in time but invalidates the "matched precision" framing and the producer excuse.
B2. No router/probs gradient (F2): skipped compute + non-comparable optimization problem + a confounded loss claim.
B3. `use_turbo_attention: false` in EVERY arm. Symmetric in absolute ms, but it inflates the shared non-MoE denominator (~750 ms/iter per the V6 back-solve), which flatters whichever arm is behind — currently the mega arm.
B4. `moe_router_force_load_balancing: true` — perfectly uniform routing. The mega wgrad is a uniform-capacity baddbmm over [32, 1280] padded rows (40,960 slots for 32,768 real token-copies = 25% padding even at perfect balance); turbo GG consumes ragged groups natively. Skew (the README's T3 regime) will hurt mega more. Unmeasured; direction known.
B5. `moe_shared_expert_overlap: false` in all arms, explicitly because "that overlap is one of OUR kernel's levers". Symmetric TODAY; it becomes an asymmetry the moment the mega arm folds shared-expert GEMMs into its service windows (the V6 next-session item) unless production is re-run with the flag on.
B6. Banked mega config is the acknowledged-flaky FWD_SYNC=0, n=1 (F5). Best-of-N risk on an unstable setting vs n=2 on a stable one.
B7. Memory: mega 206.1 GB vs production 151.5 GB (+54.6 GB, 82% vs 60% of HBM). Not priced in tok/s at mbs=1; it removes the mega arm's headroom for the standard production lever.

=== VERIFIED EQUAL (state these affirmatively in the writeup) ===
All 816 effective args except the 6 above; identical optimizer (Adam + distributed optimizer, main_grads fp32, use_precision_aware_optimizer=False, store_param_remainders=True); identical measurement window (skip 50, reset 200, 260 iters, rank-7 harmonic mean); identical model-FLOP accounting (709.8 TFLOP/iter in both); identical `optimizer` timer (49.3 vs 49.4 ms) and `batch-generator` (7.5 vs 7.1 ms); same seed/mock data; `moe_permute_fusion: true`, `moe_router_dtype: fp32`, `use_turbo_gemm: false`, `use_turbo_autotune: false` in both; TunableOp off in both; PrimusTopKRouter is the router CLASS in both arms (only the fused branch differs); the mega kernel runs on 100% of MoE layer-microbatches.

--- magnitude_estimates ---

=== MEASURED LEDGER (Megatron timers, rank-7, min-of-(min,max), final iteration) ===
| arm | fwd-compute | bwd-compute | optimizer | ms/iter (avg) | tok/s/GPU | peak HBM |
| vanilla-BF16 (legacy MoE) | 560.0 | 1005.5 | 51.4 | 1,677.0 | 19,540 | 152.6 GB |
| turbo_deepep-BF16 | 454.1 | 894.7 | 51.6 | 1,473.7 | 22,235 | 151.8 GB |
| turbo_gg-BF16 | 404.1 | 868.1 | 51.2 | 1,383.3 | 23,687 | 149.0 GB |
| turbo_gg-FP8D (PRODUCTION BAR) | 390.7 | 812.3 | 49.4 | 1,329.5 / 1,328.4 (rep1/rep2) | 24,667 | 151.5 GB |
| mega + vanilla-BF16 chassis (t1v6ns) | 500.3 | 1084.0 | — | 1,660.5 | 19,734 | — |
| mega + vanilla-FP8D chassis (t1v6stack, MEGA BAR) | 479.4 | 1027.2 | 49.3 | 1,587.8 | 20,637-20,739 | 206.1 GB |

=== TRIANGULATION OF THE 295 ms (arithmetic + explicit assumptions) ===
Total BF16 chassis+MoE gap: 1,677.0 - 1,383.3 = 293.7 ms elapsed (timer buckets agree: fwd 155.9 + bwd 137.4 = 293.3).

Step 1 — MEASURED sub-split (this part needs no assumptions):
  turbo_deepep -> turbo_gg = 1,473.7 - 1,383.3 = 90.4 ms (fwd 50.0 + bwd 26.6 = 76.6 bucket). This is the EXPERT GEMM ONLY (legacy gmm -> turbo GG + the D2H-skip patch). 100% inside the mega seam.
  vanilla -> turbo_deepep = 1,677.0 - 1,473.7 = 203.3 ms (fwd 105.9 + bwd 110.8 = 216.7 bucket). This BUNDLES {DeepEP dispatch/combine replacing Megatron alltoall+permute, sync-free stage} with {turbo RMSNorm, fused router}. No arm isolates them — that ablation was never run.

Step 2 — ANALYTIC bound on the backbone half of the 203.3:
 (a) turbo RMSNorm. Norm inventory per microbatch forward: 4 layers x (input_layernorm + pre_mlp_layernorm) + final_layernorm = 9 h-wide RMSNorms on [4096, 7168] bf16 = 117 MB read+write each; plus MLA q_layernorm(1536)/kv_layernorm(512) at ~1/5 and ~1/14 the traffic. At ~5 TB/s achievable HBM: ~23 us fwd, ~45 us bwd per h-wide norm -> 9 x 68 us = 0.61 ms/microbatch -> 4.9 ms/iter for a PERFECT (bandwidth-optimal) norm. Turbo replaces TE's already-fused norm, so the recoverable delta is a fraction of that: 2-6 ms/iter, hard ceiling ~10.
 (b) fused router. The fused op replaces only `topk_routing_with_score_function`; the aux-loss application stays eager in BOTH arms. That chain is ~12-15 launch-bound kernels on [4096,256] fp32 forward and ~15-20 backward; at ~10 us/launch: ~0.15 ms fwd + ~0.20 ms bwd per layer-microbatch -> 32 x 0.35 = ~11 ms/iter unfused, ~1-2 ms fused -> ~9-10 ms/iter recoverable. BUT the mega arm pays only the FORWARD half (it returns no probs grad, F2), so its exposure is ~4-5 ms/iter.
 => BACKBONE/CHASSIS HANDICAP ON THE MEGA ARM = 7-15 ms/iter, point estimate ~10 ms. That is 2.4-5.1% of the 293.7 ms BF16 gap and ~4% of the mega-vs-production gap. The remaining ~95% of the 293.7 is the MoE layer itself, which the mega arm replaces.
 Assumptions: (i) no isolating run exists, so (a)/(b) are analytic not measured; (ii) ~5 TB/s achievable HBM and ~10 us/kernel launch on this stack; (iii) use_turbo_attention: false everywhere so no attention fusion is in play.

Step 3 — cross-check against the mega-vs-production budget:
  1,587.8 - 1,328.4 = 259.4 ms (banked avgs); 1,580.0 - 1,271.5 = 308.5 ms (final-iteration inst; buckets agree at fwd +88.7, bwd +214.9 = +303.6).
  Attribution: chassis ~10; producer +70 (4 x 17.5, with NO production counterpart per F1); router-backward credit -5 to -9; residual ~185-235 ms = the mega stack's own non-kernel time. This matches the team's own V6 back-solve ("~326 ms of non-kernel time around our megas vs their ~30-50": in-mega drift stalls ~160 + host glue ~100-170) — except that back-solve assumed "shared non-MoE+reduce ~750 = ~750", which this audit corrects to 750 vs 750+~10.

=== SIZING F1 (does production's bf16-MoE bar under-report production?) ===
Iteration-ALIGNED comparison (iters 9-15, same warm-up state) across recipes on the same turbo_gg chassis:
  e4m3 + tensorwise (turbo fp8 MoE ON): 1,335-1,356 ms, fwd 444-465, bwd 807-812
  hybrid + tensorwise (turbo fp8 MoE ON): 1,344-1,358 ms, fwd 455-470, bwd 805-809
  hybrid + delayed (turbo fp8 MoE OFF/BF16): 1,334-1,343 ms, fwd 447-455, bwd 803-806
=> Turbo GG's fp8 path is worth ~0 ms on this workload. So F1 costs production essentially nothing in TIME (0-10 ms) — it is a claims/precision defect, not a hidden speed handicap. Corollary: fp8 in the MoE buys production nothing here, which independently supports the T4 BF16-megakernel direction.
Also: turbo GG OFF with TE grouped experts (t0_te-FP8) = 1,511 ms — turbo GG itself is worth ~180 ms regardless of precision.

=== OTHER MAGNITUDES ===
- fp8 on the non-MoE path: mega/vanilla chassis 1,660.5 -> 1,587.8 = 72.7 ms; production/turbo chassis 1,383.3 -> 1,329.5 = 53.8 ms. No evidence the skipped Primus fp8-context patch penalizes the mega arm.
- mega swap on its OWN chassis (BF16): 1,677.0 -> 1,660.5 = 16.5 ms, +1.0%.
- topk re-derivation: 0.029 ms/call x 32 = ~0.9 ms/iter.
- FWD_SYNC=0 vs =1: ~21 ms/iter (team's own A/B).
- Machine drift 08-15 vs 08-17: 0-40 ms, sign uncertain (same-day smoke 1,348.4 inst vs 1,269.6 inst, but the smoke has only 10 timed iters).
- use_turbo_attention: if enabling it saved ~80 ms for both arms, the headline ratio moves 1,587.8/1,328.4 = 1.195x -> 1.208x against the mega arm.

--- evidence ---

All node paths under subvadla@10.5.95.87; bench root = /home/subvadla/amd-master-m15pkt/auto-gpu-kernel/k0_fused_moe/training_bench/

CONFIG DIFF (exhaustive, from the two runs' own arg dumps):
- Extracted 816 `module_utils.py:241` lines from /home/subvadla/k0-training-bench/t0fp8dfull/turbo_gg-FP8D_rep1.log and /home/subvadla/k0-training-bench/t1v6stack/t1mega.log; diff = exactly the 7 lines listed in "asymmetries". fp8/fp8_recipe/optimizer/model/data args are byte-identical, confirming t1v6stack ran t0_vanilla-FP8D.yaml.
- Yamls: .../t0_vanilla-FP8D.yaml:66-70 (`enable_primus_turbo: false`, `moe_use_legacy_grouped_gemm: true`) vs .../t0_turbo_gg-FP8D.yaml:64-79.
- Applied-patch diff (same two logs): production applies `megatron.turbo.rms_norm`, `megatron.turbo.moe_dispatcher`, `megatron.moe_alltoall_dtoh_turbo_grouped_gemm`, `megatron.fp8.context`, `transformer_engine.pytorch.fp8`; the mega arm logs all five as "Skipped: (condition not met)".

F1 — PRODUCTION'S MoE RUNS BF16 IN THE BANKED BAR:
- Container source rocm/primus:v26.5, /workspace/Primus/primus/backends/megatron/core/fp8_utils.py:159-162 —
    `if config.fp8_recipe == Fp8Recipe.delayed:` ... `fp8_quant_config_none_reason = "Primus-Turbo not support delayed scaling."`  (fp8_quant_config stays None)
  then :234-240 — `primus_turbo_fp8_autocast(enabled=True if fp8_recipe is not None else False, ..., enabled_turbo=True if fp8_quant_config is not None else False, ...)` => enabled_turbo=False.
- /workspace/Primus/primus/backends/megatron/core/extensions/primus_turbo.py:519 — `cls.PRIMUS_TURBO_FP8_ENABLED = enabled_turbo and turbo_quant_config.is_fp8()`.
- primus_turbo.py:~1798-1810 (`PrimusTurboGroupedLinear.forward_internal`) — the fp8 path AND `_maybe_create_quantized_weight_buffers(...)` are both inside `if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():`.
- RUNTIME PROOF in the banked production log: `/home/subvadla/k0-training-bench/t0fp8dfull/turbo_gg-FP8D_rep1.log` -> `[fp8_utils.py:215] : Primus-Turbo FP8 delayed not work since Primus-Turbo not support delayed scaling.` (same line in the 08-17 smoke t0fp8dsmk).
- Timing A/B for the magnitude: iteration-aligned buckets from the rank-7 debug.logs of t0fp8full (e4m3 tensorwise), t0fp8hfull (hybrid tensorwise), t0fp8dfull (hybrid delayed), iters 9-15.

F2 — NO ROUTER GRADIENT:
- .../k0_train/moe_swap.py, `K0MegaMoEFunction.backward` return tuple: `dX, None,  # probs grad: v2 (dw_sorted mapping)`, ...
- .../k0_train/mega_sitecustomize.py:60-70 — `shared = self.shared_experts_compute(...)`, `probs, _routing_map = self.route(...)`, then `mega_moe(flat, probs, w13, w2, w1p, w2p, ...)`; the router runs stock and outside the seam.
- Loss curves (rank-7 debug.logs): mega iters 1/10/50/100/200 = 1.323e1 / 1.146e1 / 1.667e-1 / 2.973e-2 / 1.026e-2; production = 1.324e1 / 1.072e1 / 1.109e-1 / 3.348e-2 / 1.308e-2. Mega is BEHIND to ~iter 50, crosses over by ~iter 100.

F3 — SCOPE OF THE TWO BACKBONE FLAGS:
- /workspace/Primus/primus/backends/megatron/patches/turbo/rms_norm_patches.py:55 — `te.pytorch.RMSNorm = PrimusTurboRMSNorm` (global class swap; condition at :19-32 requires enable_primus_turbo + TP==1).
- /workspace/Primus/primus/backends/megatron/core/transformer/moe/router.py:86-89 — `if args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score: ... fused_router_and_auxiliary_loss(logits) else: super().routing(...)`; the fused op (:38-42) covers only sigmoid+group-topk+scaling — aux-loss application stays eager in both arms.
- /workspace/Primus/primus/backends/megatron/patches/args/rocm_arg_validation.py:52 — sync_free stage 1 auto-enables `moe_use_fused_router_with_aux_score: True`; :163-167 — turbo GG and legacy GG are mutually exclusive.
- /workspace/Primus/primus/backends/megatron/patches/turbo/fp8_patches.py:27-33 — `megatron.fp8.context` condition = `fp8 AND is_primus_turbo_can_patch`.

TIMERS (all from `.../logs/pre_trainer/rank-7/debug.log` inside each run's `*_workspace`):
- t0full_0814T2314/{vanilla,turbo_deepep,turbo_gg}_rep1: iteration 260 elapsed 1637.3/1677.0, 1419.8/1473.7, 1342.0/1383.3; fwd 560.0/454.1/404.1; bwd 1005.5/894.7/868.1.
- t0fp8dfull/turbo_gg-FP8D_rep1: iter 260 elapsed 1271.5/1329.5, fwd 390.7, bwd 812.3, optimizer 49.4, TFLOP/s 558.2; rep2 1269.6/1328.4, 24,667 tok/s.
- t1v6stack: iter 260 elapsed 1580.0/1587.8, fwd 479.4, bwd 1027.2, optimizer 49.3, TFLOP/s 449.2, tok/s 20,739/20,637, loss 9.278746E-03, grad norm 0.107, hip mem 206.12GB (82%). Production at iter 260: loss 1.203642E-02, grad norm 0.080, hip mem 151.54GB (60%).
- t1v6ns (identical env, BF16 chassis): iter 260 elapsed 1658.6/1660.5, fwd 500.3, bwd 1084.0, 19,734 tok/s.
- Same-day production smoke t0fp8dsmk (2026-08-17 23:14): iter 12 elapsed 1348.4/1362.8, fwd 458.8, bwd 805.8.
- t0fp8mx_te-FP8 (turbo GG off, TE experts): 1,511 ms.

RUN PROVENANCE:
- /home/subvadla/v6_ladder11.sh — `env SMOKE=0 K0_MEGA_FWD_SYNC=0 K0_MEGA_WGRAD_MODE=bmm K0_MEGA_BWD_CO=k0pf6gm_t2v6_fast.gfx950.co K0_MEGA_CFG=t0_vanilla-FP8D.yaml ./run_t1_mega.sh t1v6stack` (n=1).
- .../run_t0_baselines.sh:19,29 — REPEATS default 3, ARMS list.
- Host-time instrument (K0HT, t1v6ht2 rank-0 debug.log): f_topk 11.2 ms / 384 calls; f_producer 49.9 ms / 384 calls (host launch only); b_wgrad_host 3239.6 ms / 384 calls.
- Local: distributed-kernels/fused_moe/overnight/aug18/V6_EVIDENCE_AND_DESIGN.md sect.3 (the "shared non-MoE+reduce ~750 = ~750" assumption this audit corrects; "wgrad visible ~197 (bmm) vs ~214"; "~326 ms of non-kernel time"); sect.5 ("all 12-iter smokes are sync0-flattered; long runs need sync1").
- Local: distributed-kernels/fused_moe/overnight/aug14/T1_TRAINING_SWAP_DESIGN.md:47-54 (legacy weight layout is BF16-only; turbo layout claim — see FEASIBILITY, this is the one design assertion the audit falsifies).

FEASIBILITY (turbo backbone + mega MoE):
- Patch point survives: mega_sitecustomize.py:75 rebinds `MoELayer.forward` at import; Primus's topk_router_patches.py:195 rebinds `MoELayer.__init__` at `before_train`; the turbo dispatcher/rms_norm patches touch neither. No conflict.
- Turbo experts parameter: primus_turbo.py:1714-1728 — one consolidated `self.weights` of shape `[num_gemms, out_features, in_features]` = fc1 [32,4096,7168], fc2 [32,7168,2048]. mega_sitecustomize.py:32-34 already reads `fc1.weights`.
- BLOCKER 1: `w1p = getattr(self.experts, "weight1", None)` -> None on turbo; moe_swap.py:817-820 raises for stream/fill, and bmm mode dies at moe_swap.py:997 `w1p.main_grad.view(32, 7168, 4096)`.
- BLOCKER 2: arena orientation. moe_swap.py:~668-674 allocates bmm arenas as [32,7168,4096]/[32,2048,7168] (the LEGACY/backward orientation, matched to that main_grad view). Turbo's weights are the FORWARD orientation -> arenas and baddbmm operand order must flip.
- BLOCKER 3 (falsifies T1_TRAINING_SWAP_DESIGN.md:50-52): the producer gets SLOWER on turbo, not faster. weight_producer.py:155-170 `refresh_native_t` exists precisely BECAUSE legacy `[E,H,2I]`/`[E,I,H]` storage IS the backward orientation ("quantize ONCE per weight, transpose fp8 bytes"). Turbo's `[E,2I,H]` makes `w13.transpose(1,2)` non-contiguous (moe_swap.py:640-651), dropping to `refresh_t`, which does `.to(bf16).contiguous()` on 1.87 GB + 0.94 GB per layer plus a transpose-contiguous — roughly doubling the 17.5 ms/layer producer unless a forward-orientation native path is added.
- BLOCKER 4: `use_turbo_deepep: true` allocates symmetric dispatch buffers the mega path never uses — dead memory on an arm already at 82% HBM.
- Validation constraints checked: rocm_arg_validation.py:163-167 (turbo GG vs legacy GG mutually exclusive), :171-176 (sync_free stage>0 requires turbo; stage>1 requires turbo GG), :191-203 (DeepEP asserts). A turbo-chassis-only arm trips none of them.

--- fixes ---

Ordered by (value / cost). Items 1-3 are ~50 min GPU each and settle the entire chassis question.

1. RUN THE MISSING SAME-CHASSIS CONTROL (highest value). `t0_vanilla-FP8D.yaml`, no hook, 260 iters x 2 reps. This is a pure chassis+dispatcher+GEMM A/B at genuinely matched MoE precision, because BOTH sides run BF16 expert GEMMs (vanilla legacy gmm is bf16-only; turbo GG is bf16 under `delayed` per F1). It directly measures the "295 ms" question in the FP8 configuration and gives the denominator for "what the mega swap is worth on its own chassis" in FP8 (the BF16 answer is +16.5 ms / +1.0%).
   `ARMS=vanilla PROBE=0 REPEATS=2 ./run_t0_baselines.sh t0vanfp8d` after adding a `vanilla-FP8D` arm mapping to the runner (run_t0_baselines.sh:33 builds `t0_${arm}-BF16.yaml`; add the FP8D variant).

2. BUILD THE TRULY FAIR ARM — TIER 1, ZERO CODE CHANGE. New `t1_turbochassis-FP8D.yaml` = `t0_vanilla-FP8D.yaml` plus: `enable_primus_turbo: true`, `use_turbo_rms_norm: true`, `moe_use_fused_router_with_aux_score: true`, and explicitly `use_turbo_deepep: false`, `use_turbo_grouped_gemm: false`, `moe_use_legacy_grouped_gemm: true`, `turbo_sync_free_moe_stage: 0`, `use_turbo_attention: false`. Validation clears it (see evidence). The experts module stays legacy GroupedMLP, so `_expert_weights`'s legacy branch, `refresh_native_t`, the `[32,7168,4096]` arenas, and the `w1p/w2p` fused-wgrad contract are all untouched — nothing in moe_swap.py or the kernels changes. Run it with the same env as t1v6stack. Expected recovery 7-15 ms; the value is converting an assumption into a measurement and removing the "you gave them the whole chassis" objection permanently. DO NOT pursue Tier 2 (turbo experts module) for fairness — the mega kernel supplies its own dispatch/GEMM/combine, so turbo's expert module buys the comparison nothing and costs 4 concrete blockers (see evidence).

3. KILL THE TEMPORAL CONFOUND. Re-bank `t0_turbo_gg-FP8D` in the same session as the mega bar, ideally interleaved rep-by-rep (production rep1, mega rep1, production rep2, mega rep2). Quote both arms from that session.

4. RESTATE THE PRECISION CLAIM (do this before any writeup ships).
   - Say plainly: the production bar is FP8 non-MoE + BF16 MoE experts. Cite fp8_utils.py:159-162 and the runtime warning line.
   - DELETE "EVERY fp8 trainer pays it, including production's" from T1_RESULTS.md — the banked bar pays zero expert-weight quantization. Report the ~70 ms/iter producer as an unmatched cost of the mega design.
   - REPLACE the "matched-precision race" framing with the iteration-aligned recipe table (iters 9-15: tensorwise-with-fp8-MoE 1,335-1,358 vs delayed-with-bf16-MoE 1,334-1,343). It shows turbo GG's fp8 path is worth ~0 ms here — a STRONGER version of the capability-gap finding, and independent support for the T4 BF16-megakernel variant (aug15/T4_BF16_VARIANT_DESIGN.md).

5. REPORT THE STABLE MEGA CONFIG. Bank FP8D chassis + `K0_MEGA_FWD_SYNC=1`, n>=2, and quote it as the headline with the sync0 number as an upper bound; or land the retire-latch port first. Never quote an n=1 result on a setting your own notes call unstable.

6. PRICE THE ROUTER GRADIENT. State in the results table that the mega arm computes no router or aux-loss gradient; quantify the skipped work (~5-9 ms/iter); and WITHDRAW the loss comparison as a quality claim until a probs-grad arm exists. Add the confound explicitly: under `force_load_balancing` the router trains on random logits, so dropping its gradient plausibly improves the mega loss.

7. SYMMETRY GUARDS FOR THE NEXT ROUND.
   - Flip `use_turbo_attention: true` for BOTH arms (or document why not) — leaving it off inflates the shared denominator and flatters the trailing arm.
   - If/when the mega arm folds shared-expert GEMMs into its service windows (V6 next-session item 2), re-run production with `moe_shared_expert_overlap: true` in the same commit.
   - Run the T3 skewed-routing regime before any general "we beat production" claim; the mega wgrad's uniform-capacity padding (40,960 slots for 32,768 token-copies) is a known skew liability.

8. REPORT PEAK HBM NEXT TO tok/s (206.1 vs 151.5 GB). It is a real cost of the mega design and it is currently invisible in the headline.

9. CORRECT THE V6 BACK-SOLVE. distributed-kernels/fused_moe/overnight/aug18/V6_EVIDENCE_AND_DESIGN.md sect.3 assumes "shared non-MoE+reduce ~750 = ~750". Change to 750 (production) vs ~760 (mega) and note the ~10 ms chassis term, so the ~326 ms non-kernel deficit is not silently absorbing it.

10. CORRECT THE DESIGN DOC. aug14/T1_TRAINING_SWAP_DESIGN.md:50-52 states turbo's weight layout is "the SAME per-expert orientation as our kernel's w13/w2 contract" and implies mounting on turbo is favourable. It is the FORWARD orientation; the producer's fast path (`refresh_native_t`) needs the BACKWARD orientation, which only the LEGACY layout provides. Flag it so nobody plans a turbo-experts port on that premise.

--- surprises ---

1. The chassis story is nearly empty. I expected the focus question ("the mega arm runs on the vanilla backbone; production gets the whole Primus-Turbo chassis") to be a large number. It is ~10 ms/iter. The 6-flag diff is surgically small, and 4 of the 6 flags act entirely inside the swap seam that the mega kernel replaces. The serving audit's finding does not generalize: in training, the kernel runs on 32/32 layer-microbatches every iteration and sits squarely in the critical path.

2. The biggest fairness defect runs AGAINST production, not for it, and nobody noticed it. `fp8_recipe: delayed` silently disables Primus-Turbo's fp8 in the grouped GEMM. The warning is right there in the banked log (`fp8_utils.py:215`). The team's own recipe-matrix conclusion — "production FP8 trains ONLY with delayed scaling" — has a mechanism they never identified: delayed is the recipe that doesn't do fp8 in the MoE, which is exactly why it doesn't NaN.

3. ...and it costs production ~0 ms. Enabling turbo GG's real fp8 path (tensorwise) is indistinguishable from the bf16 path at iteration-aligned comparison. The MoE expert GEMMs on this proxy are not compute-bound. This simultaneously (a) makes the capability-gap claim stronger, (b) removes the "producer cost is universal" excuse, and (c) is a quiet argument that the whole fp8 megakernel bet buys less on this shape than assumed — the T4 BF16 variant looks better than its own design doc projects.

4. The banked design doc has a load-bearing assumption backwards. T1_TRAINING_SWAP_DESIGN.md treats turbo's `[E,2I,H]` weight layout as the favourable one for the mega kernel. It is the forward orientation; the producer's fast path exists precisely because the LEGACY layout is already the backward orientation. Mounting on turbo experts would roughly DOUBLE the producer, not trim it.

5. The one A/B that answers the core question was already run and never reported. t1v6ns (mega on the vanilla BF16 chassis, 1,660.5) vs t0 vanilla-BF16 (1,677.0) is a clean same-chassis measurement of what the mega swap is worth: 16.5 ms, +1.0%. It is in the logs, not in T1_RESULTS.md.

6. The mega arm's lower loss is probably an artifact of the missing router gradient, not better numerics — because `force_load_balancing` trains the router on random logits, so the mega arm is the only arm NOT injecting that noise into its weights. The loss curve shape (behind at iters 10/50, ahead from ~100) is exactly what a "same model minus a noise-driven parameter" trajectory looks like.

7. Two things I checked expecting trouble and found clean: TunableOp is off in both arms (the mega runner pins it explicitly, and the T0 runner inherits the same default — no accidental tuned-GEMM advantage), and the model-FLOP accounting is identical (709.8 TFLOP/iter in both), so Megatron is not crediting the arms differently.

============================== WORK AUDITOR ==============================

--- findings ---

TRAINING FAIRNESS AUDIT — mega arm (t1v6stack) vs production arm (t0_turbo_gg-FP8D). Verdict: unlike the serving audit, the kernel DOES run on 100% of heavy steps here (4/4 MoE layers x 8 microbatches x 260 iters, zero fallbacks, K0DBG confirms probs(4096,256) shape-match every call). But the comparison is unfair in BOTH directions, and the largest single item is that WE DO 30.5% MORE MoE MATH THAN PRODUCTION FOR THE SAME MODEL.

HEADLINE 1 — THE BANKED PRODUCTION NUMBER IS AN END-OF-RUN ARTIFACT.
Production's per-iteration time is flat at 1,342-1,347 ms for iterations 205-258, then drops to 1,270.3 / 1,269.6 ms on iterations 259-260 ONLY. Because `log_avg_reset_interval: 200` makes the reported "avg" field cover iters 201-260, those two outliers drag the reported average from 1,343.0 (iter 258) to exactly 1,328.4 ms / 24,667.5 tok/s — the banked headline. The mega arm shows no such dip (1,589.5 -> 1,587.8, -1.7 ms). The dip is entirely in the FORWARD bucket: production forward-compute is 455.8-457.8 ms at iters 256-258 and 390.7/389.8 at 259-260. Most likely mechanism: the delayed-fp8 async amax allreduce wait (`[Patch:delayed_scaling_update] ... subsequent steps wait on async handle launched by post-fwd_bwd hook`) not being paid on the final steps.
Honest steady-state restatement (iteration 258, both arms): production 1,343.0 ms / 24,399 tok/s; mega 1,589.5 ms / 20,615 tok/s. Ratio 0.845 instead of the claimed 0.841 — the reported comparison is biased AGAINST us by 0.4%, and the forward-gap it implies is 3x too large.

HEADLINE 2 — THE REAL DECOMPOSITION IS BACKWARD-ONLY.
Using un-dipped iteration-258 Megatron timers: forward 488.3 (mega) vs 455.8 (prod) = +32.5 ms. Backward 1,028.3 vs 816.8 = +211.5 ms. Sum +244 ms ~= the +246.5 ms iteration gap. The banked comparison mis-attributes ~57 ms of the gap to the forward. Our forward is essentially at parity once the unfused router and the exposed weight-producer are accounted for; 87% of the deficit is in the backward, and most of that is work production never does.

HEADLINE 3 — WE ISSUE 30.5% MORE MoE FLOPs.
Per layer-microbatch at balanced routing (32,768 sorted rows/rank, 1,024 rows/expert):
  production: fwd 2.886 + dgrad 2.886 + wgrad 2.886 = 8.66 TFLOP
  mega:       fwd 2.886 + Z-REGEN 1.924 + dgrad 2.886 + wgrad-as-issued 3.608 = 11.30 TFLOP
Per iteration (x32): 277.1 vs 361.7 TFLOP = +84.6 TFLOP (+30.5%). Two causes, both self-inflicted: (a) the backward regenerates z instead of the forward saving it (n2_phase1z_gm_t2b.cpp exists precisely because the forward epilogue's register ceiling could not hold acc for a second consumer); (b) the bmm wgrad runs a UNIFORM capacity of cap=1280 rows/expert against 1,024 actually-live rows — 20% of every wgrad GEMM is provably dead under the benchmark's own force_load_balancing regime.

HEADLINE 4 — DDP/GRAD-REDUCE IS A NON-ISSUE, NOT A CHEAT.
At EP8/DP8/TP1 the expert-data-parallel group size is 1, so expert grads need no reduction in EITHER arm. Measured: all-grads-sync 0.22-0.33 ms (mega) vs 0.21-0.31 ms (prod); optimizer-copy-to-main-grad 0.13-0.27 vs 0.13-0.25. Both arms already bypass grad->main_grad copies (ours via grad_added_to_main_grad, theirs via gradient_accumulation_fusion). The dummy-grad trick buys nothing production doesn't already have. Clean.

HEADLINE 5 — QUALITY PARITY IS NOT ESTABLISHED; THE LOSS COMPARISON IS NOT LIKE-FOR-LIKE.
16-iteration window means (iters 245-260): mega 8.62e-3, prod rep1 10.98e-3, rep2 10.01e-3. The gap is real (~6 SE, not noise) — but the two arms are not minimizing the same objective. Because probs receives None, MoEAuxLossAutoScaler's backward never fires: the mega arm optimizes pure lm loss while production optimizes lm + 0.001 x seq_aux_loss, and the mega arm's router weights receive EXACTLY ZERO gradient (they are frozen parameters). Grad norm 0.107 (mega) vs 0.080 (prod) at identical lr 1e-5 — different effective step size. A lower lm loss under a strictly easier objective and a larger step size is not a quality certificate. Separately, "matched precision" overstates the match in BOTH directions: our recipe (128x128 blockscale weights, just-in-time per-row/per-128 activation scales) is strictly higher fidelity than theirs (tensorwise, DELAYED, amax history 1024 = one step stale) — which favors us on quality and costs us on time — while our fp8-on-wire dispatch and our certified 6.35e-2 (y) / 7.22e-2 (dX) kernel error are materially coarser than production's bf16 dispatch/combine path. The robustness findings (their documented e4m3+tensorwise recipe NaNs 4/4 through turbo GG; 128x128 blockwise unsupported on ROCm) stand and favor us, but they are orthogonal to the speed claim and must not be blended into it.

HEADLINE 6 — THE BACKBONES DIFFER, AND THE CONTROL THAT WOULD SEPARATE THEM WAS NEVER RUN.
The mega arm runs on t0_vanilla-FP8D.yaml with enable_primus_turbo: False, use_turbo_rms_norm: False, moe_use_fused_router_with_aux_score: False. Production runs t0_turbo_gg-FP8D.yaml with all three True. Every one of the ~60 t1* runs on the node has enable_primus_turbo: False. So the 246 ms gap conflates (a) the MoE implementation with (b) the whole-model turbo backbone. There is no `t0_vanilla-FP8D` run with the STOCK MoE (no K0_TRAINING_MOE=mega) anywhere in ~/k0-training-bench — the one control that would isolate the MoE delta does not exist.

HEADLINE 7 — THE PER-LAYER TABLE COMPARES DIFFERENT WORK.
"fwd 4.75 ms + backward-dgrad 7.65 ms vs production turbo_gg ~20-24 ms" puts our fwd+dgrad against their fwd+bwd-INCLUDING-wgrad. All-in per layer-microbatch ours is 4.75 (fwd) + 7.65 (dgrad, z-regen already inside) + 6.16 (wgrad, 197/32) + 2.19 (producer, 17.5/8) = 20.75 ms. That is parity with 20-24, not 1.9x. Secondary: our 4.75 ms uses all 256 CUs; their number is measured with 80 of 256 CUs reserved for DeepEP comm (turbo_deepep_num_cu: 80), so it is not a GEMM-vs-GEMM comparison either.

--- asymmetries ---

SIGNED LEDGER. Sign convention: (+) inflates the mega arm's ms/iter or deflates production's = HELPS THEM / hurts us. (-) = HELPS US. Magnitudes are ms/iter on the 1,589.5-vs-1,343.0 steady-state baseline.

=== A. WORK THE MEGA ARM DOES THAT PRODUCTION DOES NOT (+) ===
A1 (+86 to +101)  Z-REGEN GEMM. The backward recomputes z = W13.x as a full GEMM1-shaped pass because the forward epilogue cannot hold acc for a second consumer. Production saves z from the forward (bias_swiglu saves its input). 1.924 TFLOP/layer-mb x 32 = 61.6 TFLOP/iter at the mega's own measured 608 TFLOP/s.
A2 (+39)  WGRAD UNIFORM-CAPACITY PADDING. bmm_wgrad hardcodes cap=1280 rows/expert; force_load_balancing gives exactly 1,024 live rows/expert. 0.722 TFLOP/layer-mb dead x 32 = 23.1 TFLOP/iter, plus 20% of the gather/dequant staging.
A3 (+45 to +75, partly inside the 197 ms wgrad total)  WGRAD GATHER/DEQUANT STAGING. a_dst/sc_stage are in RECEIVE order, not sorted order, so wgrad must gather 40,960 fp8 rows, materialize them as bf16, and apply scale x liveness — ~6.96 GB/layer-mb x 32 = 223 GB/iter of pure staging traffic. Production's wgrad reads its already-permuted, already-expert-contiguous activations in place. Pure protocol artifact.
A4 (+45 to +60 issued; +25 to +45 exposed)  FP8 WEIGHT PRODUCER EXCESS. refresh_native_t moves ~21.2 GB/layer (amax pass + quant pass + 4 AITER shuffle+copy round trips + 2 fp8 transposes) = ~17.5 ms/layer x 4 = ~70 ms/iter. Production's delayed-scaling cast needs NO amax pass (history-based), no AITER shuffle, and one transpose: ~7 GB/layer, ~6-10 ms/iter. Layer 0's refresh is fully exposed (the fwd launch waits refresh_done); layers 1-3 overlap but still steal CUs.
A5 (+11 to +15)  ONCE-PER-ITER ARENA -> main_grad ADD. 4 layers x (1.879 GB bf16 arena read + 3.759 fp32 read + 3.759 fp32 write for w13; 0.940+1.879+1.879 for w2) = 56.4 GB/iter. Production's wgrad epilogue lands in main_grad directly.
A6 (+15 to +35)  NON-TURBO BACKBONE. No turbo RMSNorm (~2.5 ms) and, more importantly, the UNFUSED Megatron router with device-limited routing + seq_aux_loss over [4096,256] fp32 (~15-30 small kernels/layer-mb where production runs 1-2 fused ones).
A7 (+3 to +12)  DEBUG/HOST GLUE PRODUCTION HAS NO ANALOG FOR. maybe_check_pperr does state.pperr.item() every 50 calls = ~1.3 full device drains/iter; every one of the 64 mega launches/iter builds a CPU int64 tensor, copies to pinned staging, does an H2D, and runs an index_copy_ kernel to rewrite 4 descriptor words.
A8 (+1.3 to +3.2 GPU, +0.9 host)  REDUNDANT TOPK. self.route() already ranks the experts and returns routing_map; moe_swap then runs torch.topk(probs, k=8) over [4096,256] fp32 again, plus an i64->i32 cast and two .contiguous() calls, per layer-microbatch.
A9 (+0.3 to +1.0)  dw_topk ROW-DOTS THAT ARE NEVER CONSUMED. The phase-1b epilogue computes dwacc += dH2*act(z) over all 67.1M (row,col) elements per layer-mb, does a 4-step shfl_xor reduce, and issues ~1.05M atomicAdds into dw_sorted — and the backward returns None for probs, so every byte is discarded.
A10 (+24, NOT in the banked number)  K0_MEGA_FWD_SYNC=1 costs 32 full torch.cuda.synchronize()/iter (t1v6fastfull2 1,684.9 vs t1v6ns 1,660.5). The banked stack correctly runs sync-free; noted so it does not creep back in.

=== B. WORK PRODUCTION DOES THAT THE MEGA ARM SKIPS (-) ===
B1 (-12 to -24)  ROUTER LINEAR BACKWARD. probs gets None, so the entire router branch has zero backward. Production pays dgrad + wgrad on a [4096,7168] x [7168,256] FP32 GEMM (moe_router_dtype: fp32) = 30.1 GFLOP/layer-mb = 962 GFLOP/iter, at a skinny-N fp32 efficiency of ~40-80 TFLOP/s.
B2 (-2 to -5)  ROUTER SCORE/TOPK/AUX BACKWARD. Softmax/sigmoid + device-limited group-topk scatter backward over [4096,256] fp32, plus MoEAuxLossAutoScaler's backward (moe_aux_loss_coeff 0.001, moe_router_load_balancing_type seq_aux_loss). None of it runs in the mega arm.
     NOTE: B1+B2 (-14 to -29) is almost exactly cancelled by A6+A8 (+16 to +38). The "we skip the router backward" advantage is a wash once you charge us for running the router forward unfused and then re-deriving topk.
B3 (-20 to -30)  FP32 ACCUMULATOR TRAFFIC. Production's wgrad read-modify-writes fp32 main_grad every microbatch: 5.64 GB x 2 x 32 = 361 GB/iter. Ours accumulates in bf16 arenas: 2.82 GB x 2 x 32 = 180 GB/iter, + the 56 GB final add = 237 GB/iter. We are ~124 GB/iter cheaper — an unclaimed win, bought with 8-microbatch bf16 accumulation precision.
B4 (-17)  WGRAD TOTAL. Measured/back-solved: our bmm wgrad ~197 ms visible vs their serial grouped GEMM ~214 ms.
B5 (-20 to -40, UNVERIFIED)  WIRE PRECISION. Our dispatch is fp8-on-wire (8,064 B/row, per-K128 in-flight amax) both forward and backward; turbo DeepEP under fp8:hybrid most likely dispatches bf16 (14,400 B/row equivalent, +90% per T2_PHASE_MAP.md:28). If so we move roughly half the xGMI bytes. This is a legitimate kernel design win AND an unpriced precision reduction. Must be verified before it is claimed either way.
B6 (unquantified)  PERMUTE/UNPERMUTE/SORT_CHUNKS staging that our megakernel fuses away. Legitimate; already inside their 214 ms.

=== C. MEASUREMENT AND PROTOCOL ASYMMETRIES ===
C1 (+14.6)  END-OF-RUN DIP contaminates production's banked average only (see findings). Removing it: production 1,343.0 / 24,399, and the honest ratio improves for us from 0.841 to 0.845.
C2 (+8 to +13)  CROSS-SESSION BANKING. Production banked 2026-08-15 06:37; mega banked 2026-08-18 02:56. The same-day (2026-08-17 23:14) production smoke ran 1,343.8/1,349.6/1,348.4 at iterations 10/11/12 against the Aug-15 full run's 1,335.7/1,336.2/1,341.5 at the same indices — the node was ~0.7% slower in the mega session.
C3 (risk, unsigned)  n=1 for the mega arm (t1v6stack, single rep) vs n=2-3 with <0.4% spread for production.
C4 (-, helps us)  PER-LAYER TABLE compares our fwd+dgrad against their fwd+bwd-incl-wgrad (see Headline 7). All-in it is 20.75 vs 20-24 ms = parity.
C5 (-, helps us)  force_load_balancing IS REQUIRED BY OUR IMPLEMENTATION, OPTIONAL FOR THEIRS. cap=1280 and pad_max/T=4096 bucketing assume near-uniform occupancy; production's grouped GEMM handles arbitrary segment sizes natively. The benchmark's routing regime is a precondition for one arm only. No T3 arm exists.
C6 (-, helps us)  SHAPE PINNING. mega_sitecustomize falls back to the stock forward unless s*b == 4096 exactly, so micro_batch_size=1 / seq 4096 is a hard constraint on us and a denied knob for them — production has 100 GB free HBM it cannot spend.
C7 (-, helps us)  CU BUDGET IN THE PER-LAYER CLAIM. Our 4.75 ms uses 256 CUs; their per-layer figure is measured with 80 CUs reserved for DeepEP comm.

=== D. GRADIENT COMPLETENESS AND QUALITY ===
D1  Router weights receive EXACTLY ZERO gradient in the mega arm (probs -> None kills both the STE path and the aux-loss autoscaler). We train a strictly smaller parameter set on a strictly easier objective (no 0.001 x aux-loss pull on lm loss).
D2  Grad norm 0.107 (mega) vs 0.080 (prod) at identical lr — different effective step size confounds any loss-curve claim.
D3  Precision is NOT matched in either direction: ours is finer on weights/activations (128x128 blockscale + JIT per-row scales vs tensorwise DELAYED with a 1024-entry amax history), coarser on the wire and in accumulation (fp8 dispatch, bf16 8-microbatch wgrad accumulation, certified 6.35e-2 y / 7.22e-2 dX error).
D4  Robustness/capability findings (their e4m3+tensorwise NaNs 4/4 through turbo GG; blockwise unsupported on ROCm) are real and favor us, but they are a separate claim from throughput and must not be folded into the ratio.

=== E. MEMORY ===
E1  206.65 GB (mega) vs 152.03 GB (prod) = +54.6 GB. Identified ~38 GB: dw13/dw2 bf16 arenas +11.3; zq/dzq/dqz/dqdz allocated at pad_max=263,136 rows when only 32,768 are ever live under balanced routing = +8.9 GB of which ~7.8 GB is DEAD capacity; fp8 weight buffers +5.6 net (we hold both orientations, they may not cache the transpose: fp8_weight_transpose_cache: False); mega symmetric heap + doubled parity ring ~+9; bmm transients ~+3.
E2  Does the delta buy speed? Partly yes and honestly so: the parity ring (~4.5 GB) is what makes wgrad async, and the bf16 arenas (11.3 GB) are what make the fp32 main_grad traffic 124 GB/iter cheaper (B3). The 7.8 GB of dead z-buffer capacity buys nothing. Production's 100 GB of headroom buys them nothing either — but only because C6 pins the microbatch on their behalf.

--- magnitude_estimates ---

NET LEDGER (ms/iter, on the honest 1,589.5 vs 1,343.0 steady-state baseline; observed gap = +246.5 ms):

  MEGA-ONLY WORK (+):
    A1 z-regen GEMM                              +86 .. +101
    A3 wgrad gather/dequant staging              +45 .. +75   (overlaps the 197 ms wgrad total)
    A4 producer excess over their cast           +45 .. +60 issued (+25 .. +45 exposed)
    A2 wgrad capacity padding (20% dead)         +39
    A6 non-turbo backbone (rmsnorm + router)     +15 .. +35
    A5 arena -> main_grad add                    +11 .. +15
    A7 pperr .item() + per-launch descriptor H2D  +3 .. +12
    A8 redundant torch.topk + casts               +1.3 .. +3.2
    A9 dw_topk row-dots (never consumed)          +0.3 .. +1.0
                                       SUBTOTAL +246 .. +341

  PRODUCTION-ONLY WORK (-):
    B3 fp32 main_grad rmw vs our bf16 arenas     -20 .. -30
    B1 router linear backward (fp32)             -12 .. -24
    B4 wgrad total (197 vs 214)                  -17
    B2 router score/topk/aux backward             -2 .. -5
    B5 bf16-vs-fp8 wire (UNVERIFIED)              -20 .. -40
                                       SUBTOTAL  -51 .. -116

  MEASUREMENT (+ = biased against us):
    C1 production end-of-run 2-iteration dip     +14.6
    C2 cross-session node drift                   +8 .. +13

  RESIDUAL (already measured, not an asymmetry): ~160 ms of in-mega drift stalls from the rank-2 straggler + ~100-170 ms host glue vs their ~30-50 (V6_EVIDENCE_AND_DESIGN.md section 3). This is our kernel's exposure to cross-rank skew, not a rules violation.

  Ledger closes: +246..+341 minus 51..116 = +130..+290 of work-content asymmetry against an observed +246.5, with the rest absorbed by the drift/glue residual. The ledger is coherent.

SIGNED NET, CONSERVATIVE MIDPOINT: +195 ms/iter of the +246.5 ms gap is work-content asymmetry that a fair comparison would remove or charge to the other side. Of that, roughly +176 ms is self-inflicted work our arm does and production does not (A1+A2+A5 alone = +136..+155, all of it fixable without touching the kernel's GEMM throughput), and roughly +23 ms is measurement/config bias against us (C1+C2).

TIMER DECOMPOSITION (iteration 258, un-dipped, both arms):
  forward-compute   488.3 (mega) vs 455.8 (prod)  = +32.5 ms
  backward-compute 1028.3 (mega) vs 816.8 (prod)  = +211.5 ms
  optimizer          49.3 vs 49.4                  =   0.0 ms
  all-grads-sync      0.22-0.33 vs 0.21-0.31       =   0.0 ms
  batch-generator     7.5-9.3 vs 7.1-8.7           =   0.0 ms
The banked (dipped) numbers claim +89 fwd / +216 bwd. The forward gap is over-stated by 2.7x.

MoE FLOP LEDGER (per iteration, 32 layer-microbatches, 32,768 rows/rank):
  production 277.1 TFLOP  |  mega 361.7 TFLOP  |  excess +84.6 TFLOP (+30.5%)
  of which z-regen 61.6 TFLOP and wgrad padding 23.1 TFLOP.

PROJECTED FAIR-FIGHT NUMBER: fixing A1 (save z), A2 (right-size cap), A6 (turbo backbone), A7 (drop the debug drains), A8 (drop the redundant topk) and correcting C1 gives roughly 1,589.5 - 86 - 39 - 15 - 5 - 2 = ~1,442 ms vs an honest production 1,343.0 -> 0.93x, i.e. within 7% at matched precision, with none of it requiring a faster GEMM. Do not publish this as a result; publish it as the target.

QUALITY NUMBERS:
  16-iteration window-mean lm loss (iters 245-260): mega 8.62e-3, prod rep1 10.98e-3, rep2 10.01e-3. Iteration-to-iteration sigma ~1.0-1.3e-3, SE of the 16-sample mean ~0.3e-3 -> the ~1.9e-3 gap is ~6 SE, real. Single-iteration comparison (9.279e-3 vs 1.05e-2) is NOT a valid statistic and lands close to the window means only by luck.
  Grad norm: 0.107 (mega) vs 0.080 (prod).
  Memory: 206.65 vs 152.03 GB rocm; peak rank-3 210.09 vs 155.57 GB.

--- evidence ---

All node paths are on 10.5.95.87 (ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@); read-only, no GPU runs launched. Local scratch copies of every k0_train/*.py, *.yaml, *.md are under /private/tmp/claude-501/-Users-subha-repos-Distributed-HipKittens/ecb427b0-bd4a-4e20-b530-a4cb49485078/scratchpad/node/.

END-OF-RUN DIP (C1) — the single most load-bearing measurement:
  /home/subvadla/k0-training-bench/t0fp8dfull/turbo_gg-FP8D_rep2_workspace/k0-training-bench/subvadla/t0_turbo_gg-FP8D/logs/pre_trainer/rank-7/debug.log
    iter 255: 1346.1/1343.0 ms, 24342.9/24399.5 tok/s
    iter 256: 1343.3/1343.0,   24393.7/24398.5
    iter 257: 1343.6/1343.1,   24388.2/24397.0
    iter 258: 1342.2/1343.0,   24413.6/24399.1     <- honest steady state
    iter 259: 1270.3/1334.9,   25795.5/24546.7     <- artifact
    iter 260: 1269.6/1328.4,   25809.7/24667.5     <- the banked headline
  rep1 same file pattern: iters 205-258 all 1341.9-1347.4 inst; 259 -> 1273.8; 260 -> 1271.5.
  Dip is forward-side only. rep1 `forward-compute` tail (iters 256-260): (456.92,459.20) (457.78,459.24) (455.81,457.33) (390.74,392.25) (389.79,391.63). `backward-compute` same window: 816.41 / 816.15 / 816.81 / 813.49 / 812.33 — flat.
  Mega has no dip: /home/subvadla/k0-training-bench/t1v6stack/t1mega_workspace/k0-training-bench/subvadla/t0_vanilla-BF16/logs/pre_trainer/rank-7/debug.log iters 255-260: 1588.6/1590.0, 1589.7/1589.5, 1582.2/1588.7, 1580.0/1587.8. forward-compute tail: (486.59) (488.28) (480.76) (479.39); backward-compute tail: (1028.88) (1028.33) (1028.22) (1027.16).
  Suspected mechanism, same log: "[Patch:delayed_scaling_update] Wrapped train_step with async amax allreduce support. First step uses synchronous fallback; subsequent steps wait on async handle launched by post-fwd_bwd hook."

BACKBONE MISMATCH (A6, Headline 6):
  Config dumps grepped from the two run logs.
    mega  (~/k0-training-bench/t1v6stack/t1mega.log): enable_primus_turbo: False, use_turbo_rms_norm: False, use_turbo_deepep: False, use_turbo_grouped_gemm: False, moe_use_fused_router_with_aux_score: False, moe_use_legacy_grouped_gemm: True, moe_token_dispatcher_type: alltoall, fp8: hybrid, fp8_recipe: delayed, moe_aux_loss_coeff: 0.001, moe_router_load_balancing_type: seq_aux_loss, recompute_*: None/False.
    prod  (~/k0-training-bench/t0fp8dfull/turbo_gg-FP8D_rep1.log): enable_primus_turbo: True, use_turbo_rms_norm: True, use_turbo_deepep: True, use_turbo_grouped_gemm: True, moe_use_fused_router_with_aux_score: True, moe_use_legacy_grouped_gemm: False, transformer_impl: transformer_engine, fp8: hybrid, fp8_recipe: delayed, same aux/recompute settings.
  All ~60 t1* runs in ~/k0-training-bench have enable_primus_turbo: False (swept). No stock-MoE run of t0_vanilla-FP8D.yaml exists anywhere under ~/k0-training-bench.
  Yamls: ~/amd-master-m15pkt/auto-gpu-kernel/k0_fused_moe/training_bench/t0_vanilla-FP8D.yaml (mega arm, `enable_primus_turbo: false`) and .../t0_turbo_gg-FP8D.yaml.
  The mega arm is FORCED onto legacy GroupedMLP because moe_swap.py:817-821 raises unless the raw weight1/weight2 params exist; but only turbo GROUPED GEMM is mutually exclusive with legacy GG — turbo RMSNorm and the fused router are not.

ROUTER GRADIENT ABSENT (B1, B2, D1):
  .../training_bench/k0_train/moe_swap.py:16-20 — "probs receives no gradient (returns None) ... the dw_sorted -> probs mapping lands in v2."
  moe_swap.py:1122-1130 and :1028-1036 and :1090-1098 — every backward return path emits `None` in the probs slot.
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug14/T1_TRAINING_SWAP_DESIGN.md:38-40 — "force_load_balancing replaces router logits with random draws via RandomSTE ... aux losses are still computed on the random logits and their grads ride on `probs` via `MoEAuxLossAutoScaler`." Since probs gets None, that autoscaler backward never fires.
  Router shape confirmed live: rank-0 debug.log "K0DBG r0 L0 probs(4096, 256) torch.float32 ids[0,255] wgt[0.2755,0.3543]".
  The router forward is still paid: .../k0_train/mega_sitecustomize.py:60-61 calls `self.shared_experts_compute(...)` then `self.route(hidden_states, padding_mask)`.

REDUNDANT TOPK (A8):
  moe_swap.py:831-834 — `topk_w, topk_idx = torch.topk(probs, k=8, dim=-1)` immediately after route() already ranked them; plus `.to(torch.int32).contiguous()` and `.to(torch.float32).contiguous()`.
  Host cost measured: ~/k0-training-bench/t1v6ht2/.../rank-7/debug.log "K0HT r7 call768 ... f_topk=11.0ms/384" = 0.92 ms/iter of host time alone.

dw_topk WASTED ROW-DOTS (A9):
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_t2b_abi.hpp:44 — "K0P6_D_T2B_DWTOPK 71 // dw_topk out [T, 8] f32 (rowdot(dH2, act_z))".
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/n2_phase1b_gm_t2b.cpp:415-434 — `dwacc[m][t] += dh2 * az;` then a 4-step `__shfl_xor` reduce and `atomicAdd(dw_sorted + srow[m][t], dwacc[m][t])`. Descriptor slot 71 is bound unconditionally at moe_swap.py:725.
  Geometry: kChunkColsB=256, kWaveColsB=64, kColTilesB=4 (n2_phase1b_gm_t2b.cpp:87-89) -> 8 chunks x 4 waves = 32 atomicAdds per sorted row per layer-mb = ~1.05M atomics/layer-mb.

Z-REGEN (A1):
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/n2_phase1z_gm_t2b.cpp:1-12 — "saving z from the FORWARD's epilogue costs 784 B/lane of scratch — the forward epilogue's register ceiling cannot hold acc for a second consumer (13 ms pathology, phase-map risk 9) ... Regenerating z in the BACKWARD costs one GEMM1-shaped pass (~2.7 ms class)."

WGRAD CAPACITY PADDING (A2):
  moe_swap.py:140 — `e, cap, H, inter = 32, 1280, 7168, 2048`; :146 `idx = starts.long().unsqueeze(1) * 32 + mgr._cap_arange` with `_cap_arange = torch.arange(1280)` (:328-330). Live rows/expert under force_load_balancing = 8 ranks x 4096 tokens x 8 topk / 256 experts = 1,024. 1280/1024 = 1.25x issued.
  Baddbmm calls at moe_swap.py:186-192 contract over the full K=cap=1280.

WGRAD STAGING (A3) and TRANSIENTS:
  moe_swap.py:155-182 — `deq_g` gathers `q.view(fp8)[rows_idx].to(bfloat16)` then an in-place `mul_`, for xg, dyg, dzg, zg, then `act = silu(zg[...])*zg[...]`.
  moe_swap.py:386-390 — "K0HT measured 8.4 ms/call of HOST time in the eager chain (~2 GB of temporaries per call -> caching-allocator churn)".
  Receive-order indexing is the root cause: moe_swap.py:70-73 and :150 `tok = (sti & 0x00FFFFFF)[idx]  # [E, cap] receive rows`.

WEIGHT PRODUCER (A4):
  .../k0_train/weight_producer.py:155-181 (`refresh_native_t`): two `weight_per_128x128_quant` calls (each an amax pass + a divide/cast pass), four `shuffle_weight(...)` + `.copy_()` round trips, and two fp8 `.transpose(1,2).contiguous()` byte transposes. Weights per layer: w13 [32,4096,7168] bf16 = 1.879 GB, w2 [32,7168,2048] = 0.940 GB; traffic model ~21.2 GB/layer, consistent with the ~17.5 ms/layer figure at ~1.2 TB/s effective (the strided amax and the AITER shuffles are far off peak).
  Refresh cadence and the exposure point: moe_swap.py:615-656 (`mb % self._num_microbatches() == 0`, queued on the side stream) and moe_swap.py:840-847 (the forward waits `refresh_done[layer_no]`).
  Production's counterpart: T1_TRAINING_SWAP_DESIGN.md:51-54 — "turbo quantizes weight/weightT once per microbatch-1 into cached fp8 buffers, 128x128 weight blocks + 1x128 activation scales" (note: that line describes the blockwise recipe, which T1_RESULTS.md later shows is UNSUPPORTED on ROCm; the shipped delayed recipe uses amax history and skips the amax pass entirely).

ARENA -> main_grad ADD (A5) and the DDP contract (Headline 4):
  moe_swap.py:995-1009 — `if mb % nmb == nmb - 1: w1p.main_grad.view(32,7168,4096).add_(mgr.dw13_arenas[layer_no])` (bmm mode, native orientation, contiguous), and moe_swap.py:1018-1019 `w1p.grad_added_to_main_grad = True`.
  Arena allocation: moe_swap.py:668-676 — [32,7168,4096] bf16 + [32,2048,7168] bf16 per layer.
  DDP symmetry measured: both arms' timer blocks show all-grads-sync 0.2x ms and optimizer-copy-to-main-grad 0.1x ms (quoted above).

HOST GLUE / SYNCS (A7):
  moe_swap.py:805-810 (`state.pperr.item()` every K0_MEGA_PPERR_EVERY=50 calls; _calls increments on both fwd and bwd = 64/iter).
  .../k0_train/runtime.py:302-355 (`launch`): builds a CPU `torch.tensor([my_ids,my_wgt,hidden,out])`, `st.copy_(...)`, `st.to(self.device, non_blocking=True)`, `state.desc.index_copy_(0, self._dyn_index, ...)` — per launch, 64 launches/iter.
  moe_swap.py:898-902 (`elif mgr.fwd_sync: torch.cuda.synchronize()`), and the A/B that retired it: t1v6fastfull2 1,684.9 avg vs t1v6ns 1,660.5 avg, both 260/260, both loss 9.16e-3.

WIRE PRECISION (B5, D3):
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug14/T2_PHASE_MAP.md:27 — 8,064-byte fp8 wire row with per-K128 in-flight amax (8-lane shfl_xor, scale = max(amax,1e-6)/448); :28 — "bf16 dY variant: payload becomes 14,336 B + 64 meta = 14,400 -> 15,360 B stride (+90%)"; :38 — the combine epilogue uses bf16 remote RMWs (`accumulate_peer_bf162`), so our combine is bf16 and only the dispatch is fp8.
  Certified kernel error: ~/amd-master-m15pkt/.../training_bench/T1_RESULTS.md — "y 6.35e-2, dX 7.22e-2 ... all at the fp8 activation-quant floor (weight-quant-aware reference isolates 5.4e-2)".

MEASURED BASELINE / DRIFT / GLUE RESIDUAL:
  /Users/subha/repos/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug18/V6_EVIDENCE_AND_DESIGN.md:54-68 — "shared non-MoE+reduce ~750 = ~750; fwd kernels ~152 ~= theirs; dgrad ~245 vs ~200; wgrad visible ~197 (bmm) vs ~214 (their serial grouped GEMM) — we already win wgrad; the deficit is ~326 ms of non-kernel time around our megas (in-mega drift stalls ~160, host glue ~100-170) vs their ~30-50" and "M2 rows_done wait ... 92 us (rank 2) to 4,353 us (rank 0) per CTA per launch — rank 2 is the straggler".
  V6 doc:99-112 — banked t1v6fastfull2 and the K0HT itemization (b_wgrad_host 270 ms/iter, f_launch_sync 287 ms/iter) with the finding that these are backpressure, not deletable glue (graph capture was a wash: 1,685.7 vs 1,684).

CROSS-SESSION DRIFT (C2):
  Same-day production smoke ~/k0-training-bench/t0fp8dsmk/.../rank-7/debug.log (2026-08-17 23:14), iters 10/11/12: 1343.8 / 1349.6 / 1348.4 ms inst.
  Aug-15 full run, same iteration indices: 1335.7 / 1336.2 / 1341.5 ms inst.

MEMORY (E1):
  mega iter-260 line: "rocm mem usage/free/total/usage_ratio: 206.65GB/45.34GB/251.98GB/82.01% | rank-3 rocm max mem usage: 210.09GB".
  prod iter-260 line: "rocm mem usage ... 152.03GB/99.95GB/251.98GB/60.33% | rank-3 rocm max: 155.57GB".
  Over-allocation source: .../k0_train/m15_contracts.py:504-509 — `M15_PREFILL_B4096_BUCKET(tokens_per_rank=4096, maxtok=4096, t_loc_max=40_960, pad_max=263_136)`; moe_swap.py:302-314 allocates zq/dzq at [pad_max, 4096] uint8 = 1.078 GB EACH per layer, while only 32,768 rows are ever live under balanced routing.

SHAPE PINNING (C6):
  .../k0_train/mega_sitecustomize.py:47-55 — `if s * b != _T or intermediate_tensors is not None: return _orig_forward(...)` with `_T = 4096`.

LOSS / QUALITY (D1-D4):
  Last-16-iteration lm loss series extracted from all three rank-7 debug logs (values listed in magnitude_estimates). seq_load_balancing_loss identical across arms (1.000194 vs 1.000191-1.000195) — the aux loss VALUE matches; only its gradient path differs.
  T1_RESULTS.md matched-precision table: e4m3+tensorwise+turboGG NaN at iter 23 (2/2), hybrid+tensorwise NaN at 37/33 (2/2), e4m3+blockwise "FP8 block scaled gemm not yet supported for ROCm", e4m3+tensorwise with turbo GG OFF stable @100 — implicating turbo GG's fp8 path specifically.

--- fixes ---

ORDERED BY (bias removed) / (effort). Items 1-4 change the published number without touching a kernel.

=== TIER 0: MEASUREMENT HYGIENE (do before anything is published) ===
F1. Kill the end-of-run dip. Report the iteration-258 avg field (or set log_avg_reset_interval so the window closes before the final two iterations, or post-process excluding iters 259-260). Restate: production 1,343.0 ms / 24,399 tok/s; mega 1,589.5 ms / 20,615 tok/s; ratio 0.845. Removes +14.6 ms of bias against us and, more importantly, removes a number that will not survive a reviewer replotting the per-iteration series.
F2. Re-bank both arms in ONE session, same day, >=3 reps each, alternating arms (mega/prod/mega/prod/mega/prod) so node thermal drift cancels. The mega arm currently has n=1. Removes C2 (+8..13 ms) and C3.
F3. Turn off the debug guards for banked runs: K0_MEGA_PPERR_EVERY=0 (or > train_iters*64), K0_MEGA_TRACE=0, K0_MEGA_HOSTTIME=0, K0_MEGA_TIME=0, K0_MEGA_PROF=0. Production runs no per-50-launch device drain. [-3 .. -12 ms]
F4. Publish the per-layer table ALL-IN and say so explicitly: mega 4.75 (fwd) + 7.65 (dgrad, z-regen included) + 6.16 (wgrad, 197/32) + 2.19 (producer, 17.5/8) = 20.75 ms/layer-microbatch vs production 20-24 ms = PARITY. Add a footnote that our kernel uses 256 CUs while their figure is measured with 80 of 256 reserved for DeepEP comm. Never place "4.75 + 7.65" next to "20-24" again.

=== TIER 1: THE MISSING CONTROL AND THE BACKBONE ===
F5. Run the control that does not exist: t0_vanilla-FP8D.yaml with STOCK MoE (no K0_TRAINING_MOE=mega, 260 iters, 3 reps). Without it the whole 246 ms is attributed to the MoE implementation. This one run converts the claim from "our stack vs their stack" into "our MoE vs their MoE, backbone held fixed" — which is what the README says the benchmark is for.
F6. Give the mega arm the turbo backbone. Only turbo GROUPED GEMM is mutually exclusive with legacy GroupedMLP. Create t0_mega-FP8D.yaml with: enable_primus_turbo: true, use_turbo_rms_norm: true, moe_use_fused_router_with_aux_score: true, moe_permute_fusion: true, use_turbo_grouped_gemm: false, moe_use_legacy_grouped_gemm: true, use_turbo_deepep: false. Smoke it first (the fused router must still return a [T,256] probs tensor the swap can topk). [-15 .. -35 ms]

=== TIER 2: DELETE OUR OWN DEAD WORK (biggest levers, no GEMM work needed) ===
F7. STOP REGENERATING Z. This is the single largest item on the ledger (+86..101 ms/iter). The blocker is the forward epilogue's 784 B/lane register ceiling (n2_phase1z_gm_t2b.cpp:1-12), not memory — zq is ALREADY allocated per layer. Options, cheapest first: (a) split the forward epilogue into two passes over the same accumulators so acc has one live consumer at a time; (b) stage z through the existing t2bsh overlay LDS instead of registers; (c) spill acc to the already-allocated zq via a dedicated store loop after the quant epilogue retires. Even a 50%-effective fix is worth more than every scheduling experiment in the V6 campaign combined.
F8. RIGHT-SIZE THE WGRAD CAPACITY. moe_swap.py:140 `cap = 1280` against 1,024 live rows/expert. Make cap a bucket-derived constant = ceil(world*T*topk/E_global/32)*32 plus a small imbalance margin, or better: drive baddbmm from the per-expert segment lengths (they are already computed as `starts`/`ends`) with a two-bucket split rather than one uniform capacity. [-39 ms] Note this coupling explicitly in the writeup: cap=1280 is only safe under force_load_balancing.
F9. DELETE THE dw_topk ROW-DOTS while probs-grad is unimplemented — add a DWMODE bit that compiles out the `dwacc` accumulate, the shfl_xor reduce, and the ~1.05M atomicAdds/layer-mb (n2_phase1b_gm_t2b.cpp:415-434). Better: land v2 and USE them (see F13). Do not keep paying for a gradient you throw away. [-0.3 .. -1.0 ms, and it closes the honesty gap]
F10. DELETE THE REDUNDANT TOPK. self.route() already ranks the experts; take topk_ids/topk_w from the routing_map + probs the router returns (Megatron's own dispatch_preprocess conversion is the precedent cited in T1_TRAINING_SWAP_DESIGN.md:65) instead of torch.topk(probs, k=8) at moe_swap.py:831. [-1.3 .. -3.2 ms GPU, -0.9 ms host]
F11. FUSE THE PRODUCER. refresh_native_t currently does 2 amax passes + 2 quant passes + 4 shuffle+copy round trips + 2 fp8 byte-transposes = ~21.2 GB/layer. Fold the AITER shuffle into the quant kernel's store (emit shuffled bytes directly, no round trip) and emit BOTH orientations from the single amax pass. Target ~8 GB/layer. [-30 .. -45 ms]
F12. Fold the once-per-iteration arena -> main_grad add into the last microbatch's wgrad epilogue (accumulate the 8th microbatch directly into fp32 main_grad with beta=1 on the arena's contents). [-11 .. -15 ms]

=== TIER 3: GRADIENT AND QUALITY PARITY (required before any convergence claim) ===
F13. LAND THE PROBS GRADIENT (v2). dw_sorted is already computed and already scattered-addressable via sti/sei (n2_phase1b_gm_t2b.cpp:34-37 documents the host-side scatter to [T,8]). Until it exists, the two arms are not minimizing the same objective and no loss comparison is admissible. Cost when landed: +12..24 ms of router linear backward, i.e. we give back B1 — say so.
F14. If v2 will not land before the writeup, make the objectives match the other way: run a production control with moe_aux_loss_coeff: 0 and the router frozen (requires_grad=False on the router weight), so both arms optimize pure lm loss with a fixed router. Then the loss curves are comparable.
F15. Report loss as a curve with >=2 seeds and a window-mean statistic, never a single final iteration. State the window-mean numbers (mega 8.62e-3 vs prod 10.01-10.98e-3 over iters 245-260) alongside grad norm (0.107 vs 0.080) and note the step-size confound.
F16. Add a NUMERICS column next to the speed column: our certified y 6.35e-2 / dX 7.22e-2 at the fp8 quant floor, our fp8-on-wire dispatch and bf16 8-microbatch wgrad accumulation, against their bf16 dispatch/combine and fp32 main_grad accumulation — and, in our favor, our JIT 128x128 blockscale vs their DELAYED tensorwise (amax history 1024, one step stale). "Matched precision" is currently doing work it cannot support in either direction.
F17. VERIFY the DeepEP dispatch dtype under fp8:hybrid before claiming or conceding B5. If they dispatch bf16 while we dispatch fp8, we move ~half the xGMI bytes and that must be stated as a design choice with a precision cost, not booked as a free win.
F18. Keep the robustness/capability findings (their documented recipe NaNs 4/4 through turbo GG; 128x128 blockwise unsupported on ROCm) in a SEPARATE section from the throughput ratio. They are strong and they are ours; blending them into the speed claim will read as spin.

=== TIER 4: SCOPE HONESTY ===
F19. Run the T3 natural-routing arm. cap=1280, pad_max bucketing, and the fixed T=4096 shape are preconditions our implementation needs and theirs does not (their grouped GEMM takes arbitrary segments). Publish the balanced-routing number with an explicit caveat until T3 exists.
F20. State the shape pin. mega_sitecustomize.py:47-55 silently falls back to the stock forward for any s*b != 4096, so micro_batch_size=1 is a hard constraint on us and a denied knob for production (which is sitting on 100 GB of free HBM). Either run a production arm at micro_batch_size=2 as a sensitivity check, or state that the pin is our constraint and that the comparison holds it fixed on both arms for that reason.
F21. Right-size zq/dzq. They are allocated at pad_max=263,136 rows (1.078 GB each per layer) when 32,768 rows are live under the benchmark's routing — ~7.8 GB of the 54.6 GB memory delta is dead capacity that buys nothing. Size them from t_loc_max with an overflow guard.
F22. Fix the shared grad stub hazard (moe_swap.py:441-449): one uninitialized tensor is assigned as .grad for every layer's w1/w2 simultaneously. Harmless today because grad_added_to_main_grad short-circuits the read, but it is a silent-corruption trap if any future Megatron path touches param.grad. Use a per-param zero-size or per-shape-per-layer stub.

--- surprises ---

S1. THE BANKED PRODUCTION NUMBER IS AN ARTIFACT, AND IT IS EXACTLY THE HEADLINE. Production's iteration time is dead flat at 1,342-1,347 ms for 54 consecutive iterations and then drops to ~1,270 ms for the last TWO. Those two iterations are what turn the reported average into precisely 1,328.4 ms / 24,667.5 tok/s. The mega arm has no such dip. I did not expect the exact banked pair to fall out of a two-iteration end-of-run tail.

S2. THE DIP IS 100% FORWARD-SIDE, WHICH MEANS THE PUBLISHED FORWARD GAP IS 2.7x TOO BIG. Production forward-compute: 456.9 / 457.8 / 455.8 at iters 256-258, then 390.7 / 389.8. Using un-dipped timers the forward gap is +32.5 ms, not +89. Essentially ALL of our deficit (211.5 of 244 ms) is in the backward. That completely reframes the campaign: the forward kernel is already at parity once you charge us for the unfused router and the exposed producer, and every remaining ms is in the backward.

S3. CORRECTING THE MEASUREMENT MAKES US LOOK BETTER, NOT WORSE. Honest steady-state ratio is 0.845 vs the reported 0.841. The two errors point in opposite directions (production's dip is bigger than ours) and the net bias runs against us. Fixing a measurement artifact that favors the other side is the cheapest 0.4% on the board.

S4. WE ISSUE 30.5% MORE MoE MATH THAN PRODUCTION. 361.7 vs 277.1 TFLOP/iter for the identical model. The kernels are not slower — we are running an extra GEMM1 (z-regen, 61.6 TFLOP/iter) and 20% dead wgrad (23.1 TFLOP/iter). The entire V6 campaign has been trying to schedule work into bubbles when 85 TFLOP/iter of the work should not exist.

S5. 20% OF OUR WGRAD IS PROVABLY DEAD UNDER THE BENCHMARK'S OWN ROUTING REGIME. cap=1280 rows/expert against exactly 1,024 live rows under force_load_balancing. The `wl_f` mask zeroes the weights but the GEMM still contracts over K=1280. ~39 ms/iter for a one-constant change.

S6. THE ROUTER-BACKWARD ADVANTAGE IS A WASH. I expected "we skip the router backward" to be a meaningful unearned win (-14..-29 ms). It is almost exactly cancelled by running the router FORWARD unfused on the non-turbo backbone (+13..29) plus the redundant torch.topk (+1.3..3.2). We took the gradient-completeness hit and got no speed for it.

S7. DDP IS COMPLETELY CLEAN — AND FOR A REASON NOBODY WROTE DOWN. At EP8/DP8/TP1 on 8 GPUs the expert-data-parallel group size is 1, so expert grads are never reduced in either arm. all-grads-sync 0.22-0.33 vs 0.21-0.31 ms. The dummy-grad + grad_added_to_main_grad contract buys us nothing production doesn't already get from gradient_accumulation_fusion. This was the item I most expected to be a hidden cheat; it isn't.

S8. OUR ACCUMULATION SCHEME IS AN UNCLAIMED WIN. bf16 arenas accumulating across 8 microbatches move ~237 GB/iter versus production's per-microbatch fp32 main_grad read-modify-write at ~361 GB/iter. We are ~124 GB/iter (20-30 ms) CHEAPER, bought with 8-microbatch bf16 accumulation precision. Nobody has claimed this, and nobody has priced the precision it costs.

S9. 7.8 GB OF THE 54.6 GB MEMORY GAP IS PURE DEAD SPACE. zq/dzq are allocated at pad_max=263,136 rows — the worst-case all-tokens-to-one-rank capacity — while only 32,768 rows are ever live under the routing regime the benchmark actually runs. Meanwhile the parts of the memory delta that DO buy speed (parity ring, bf16 arenas) are legitimate and defensible.

S10. THE MATCHED-PRECISION STORY IS WRONG IN BOTH DIRECTIONS AT ONCE. Our weight/activation scaling is strictly finer than theirs (128x128 blockscale + JIT per-row amax vs tensorwise with a 1024-entry DELAYED history), while our wire and accumulation are strictly coarser (fp8 dispatch vs bf16; bf16 8-mb wgrad accumulation vs fp32; certified 6.35e-2 y error). "Both arms are fp8" hides two opposite-signed precision deltas. That is also the most plausible explanation for the loss gap being real but uninterpretable.

S11. THE LOSS GAP IS STATISTICALLY REAL BUT NOT A QUALITY RESULT. I expected the 9.279e-3 vs 1.05e-2 comparison to dissolve into iteration noise (single-iteration sigma is ~1.0-1.3e-3, i.e. comparable to the gap). It does not — 16-iteration window means are 8.62e-3 vs 10.01/10.98e-3, about 6 SE apart. But the mega arm is minimizing a strictly easier objective (no 0.001 x aux-loss pull, router frozen at zero gradient) and taking larger steps (grad norm 0.107 vs 0.080 at identical lr). The number survives; the interpretation does not.

S12. EVERY ONE OF ~60 MEGA RUNS ON THE NODE HAS enable_primus_turbo: False, AND THE CONTROL THAT WOULD SEPARATE BACKBONE FROM MoE WAS NEVER RUN. t0_vanilla-FP8D.yaml exists (created the same night as the banked stack run) but only ever as the mega arm's config — there is no stock-MoE run of it anywhere under ~/k0-training-bench. One 260-iteration run closes the largest interpretive hole in the campaign.

============================== MEAS AUDITOR ==============================

--- findings ---

MEASUREMENT-DISCIPLINE AUDIT — training arm (mega) vs production arm (Primus-Turbo + DeepEP + turbo GG)

Verdict up front: unlike the serving audit, the training comparison is NOT deeply broken. Token streams are provably identical, warmup is clean and symmetric, and day-to-day drift is small. But there are 6 real discipline gaps, the largest of which is a config asymmetry (not a timing artifact), and the net of all corrections moves the headline ~1 point in the mega arm's favor (0.841x -> 0.850x), or ~4-5 points if the structural asymmetry is fixed by rerun. No correction reaches parity — the 0.84x verdict is directionally robust.

=== F1. THE BANKED "avg" IS A 10-ITERATION TAIL, NOT 200 (the biggest measurement gap) ===
`log_avg_skip_iterations: 50` + `log_avg_reset_interval: 200` (t0_vanilla-FP8D.yaml:56-57, identical in t0_turbo_gg-FP8D.yaml) means the running-average accumulator RESETS at iteration 251. The "avg" field quoted at iteration 260 is therefore the mean of iterations 251-260 (N=10), not the intended 200-iteration timed window.

Verified arithmetically from the logs:
  production rep2: (1343.0*8 + 1270.3)/9 = 1334.9 (matches log); (1334.9*9 + 1269.6)/10 = 1328.4 (matches banked)
  mega t1v6stack: (1589.5*8 + 1582.2)/9 = 1588.7 (matches log); (1588.7*9 + 1580.0)/10 = 1587.8 (matches banked)

This interacts with a second artifact: EVERY production-stack run has a deterministic end-of-run speedup in its final 2 iterations, and the mega arm essentially does not.
  turbo_gg-FP8D:   steady 1343 -> 1270.3 / 1269.6  (delta -72 ms)   [both reps]
  turbo_gg-BF16:   steady 1393 -> 1349.3 / 1342.0  (delta -47 ms)
  turbo_deepep:    steady 1483 -> 1429.6 / 1419.8  (delta -59 ms)
  vanilla-BF16:    steady 1684 -> 1639.3 / 1637.3  (delta -46 ms)
  mega t1v6stack:  steady 1588 -> 1582.2 / 1580.0  (delta  -7 ms)
Wall-clock timestamps confirm these are real, not a timer bug (prod rep2 iters 257->260 span 3.882 s = 1294.0 ms/iter, matching the three inst values).

Because the window is exactly 10 iterations and exactly 2 of them are the anomalous ones, production banks a -11 to -14 ms discount that the mega arm does not get.

=== F2. STRUCTURAL: THE MEGA ARM RUNS ON THE WEAKER (NON-TURBO) BACKBONE ===
This is the single largest unfairness and it is not a timing artifact — it is in the yaml. `t0_vanilla-FP8D.yaml:66` sets `enable_primus_turbo: false`; `t0_turbo_gg-FP8D.yaml:64` sets `true` plus `use_turbo_rms_norm: true`. Production applies 31 patches, mega applies 26. The 5 production-only patches are:
  megatron.moe_alltoall_dtoh_turbo_grouped_gemm   <- MoE-only, correctly irrelevant to mega
  megatron.turbo.moe_dispatcher                   <- MoE-only, correctly irrelevant to mega
  megatron.turbo.rms_norm                         <- SHARED non-MoE backbone
  megatron.fp8.context                            <- SHARED non-MoE fp8 path
  transformer_engine.pytorch.fp8                  <- SHARED non-MoE fp8 path
The mega arm's own log records the denial explicitly:
  t1v6stack/t1mega.log:1215  "[Patch] (skipped): megatron.fp8.context (condition not met)"
  t1v6stack/t1mega.log:1216  "[Patch] (skipped): transformer_engine.pytorch.fp8 (condition not met)"
So the mega arm runs FP8 on the *unoptimized* TE path and with stock RMSNorm, while the bar it is measured against gets both. Both arms share ~650 ms/iter of non-MoE work (T1_RESULTS iteration budget), so this is a live handicap on ~40% of the mega arm's iteration.

=== F3. RANK-7-ONLY REPORTING IS FINE FOR THE AVERAGE (no fix needed) ===
Only rank-7 emits iteration lines (`grep -c "elapsed time per iteration"` on rank-0/debug.log = 0; print_rank_last). MoK's "median of per-iteration max-across-ranks" is NOT reproducible from these logs. However the rank-local timer already brackets the straggler at the average level: every iteration terminates in a DP grad reduce-scatter + optimizer step + param all-gather, so all ranks exit together and re-enter together. Confirmed empirically — t1v6stack iters 257->260 span 4.752 s of wall clock = 1584.0 ms/iter, exactly equal to the mean of the three logged inst values (1589.7, 1582.2, 1580.0). Over a 200-iteration window, rank-local mean == true throughput by construction. This flatters NEITHER arm. Caveat: per-iteration *inst* values and any p95 quoted from them are rank-local and must not be presented as cross-rank; and the Megatron timer min/max buckets cited in T0_RESULTS ("min/max midpoint") are not present in these rank-7 debug logs, so that fwd/bwd attribution is not re-derivable from the banked artifacts.

=== F4. WARMUP IS CLEAN AND SYMMETRIC (no gap) ===
The avg field only begins at iteration 51 — iterations 1-50 carry an inst value but no avg, so cold start cannot leak into any banked average. Cold start is enormous and correctly excluded: iter1 = 34.7 s (mega) / 40.5 s (production), iter2 = 18.2 s / 21.0 s. Both arms are within 0.5% of their 100-200 mean by iterations 5-10:
  mega:       it5-10 1584, it41-50 1587, it100-200 1589
  production: it5-10 1334, it41-50 1331, it100-200 1344
A mild, symmetric hump at iters 21-40 (~+0.6% in every arm) is fully inside the discarded window. 50 warmup iterations is generous. Nothing to fix.

=== F5. FP8 DID ACTIVATE — the -73 ms is real in magnitude but MIS-ATTRIBUTED ===
Three independent confirmations that t0_vanilla-FP8D.yaml's inserted keys took effect (this closes the "silently-ignored key" worry):
  1. config dump: t1v6stack/t1mega.log:498 `fp8 : hybrid`, :505 `fp8_recipe : delayed`
  2. patch count 26/26 vs 22/22 for the non-fp8 t1v6ns, incl. `megatron.fp8.delayed_scaling_update` (t1mega.log:1237-1239)
  3. DECISIVE — the numerics changed: iteration-51 loss is 1.773915e-1 in t1v6stack vs 1.683452e-1 in t1v6ns and 1.683146e-1 in t1v6fastfull2. A silently-ignored key cannot move the loss.
BUT the attribution is confounded. The 4 patches gained when fp8_recipe=delayed turns on are `fp8.delayed_scaling_update`, `te.delayed_scaling_reduce_amax`, `te.delayed_scaling_save_original_input`, and — critically — `megatron.grad_zero_and_data_prefetch`, described in the log as "DDP grad_data.zero_() on secondary stream; data HtoD prefetch on secondary stream" (t1mega.log:1242). That is a pure scheduling win with nothing to do with fp8 arithmetic, and it rides along for free. So "-73 ms from FP8 non-MoE" should read "-70 ms from switching the non-MoE stack to the fp8-delayed recipe, which bundles fp8 math AND a free critical-path scheduling patch". The comparison against production is unaffected (production is also fp8-delayed and also gets the patch); only the internal narrative is wrong.

=== F6. TOKEN / DATA PARITY: PROVEN IDENTICAL ===
`seq_load_balancing_loss` is a routing statistic computed from the actual input tokens. It matches to 7 significant figures across all 18 runs audited, both arms, both days:
  iter 51 = 1.000192E+00, iter 100 = 1.000193E+00, iter 200 = 1.000192E+00, iter 260 = 1.000194E+00
`consumed samples` = 3264 / 6400 / 12800 / 16640 at those iterations in every run. Same seed 1234, same mock_data, same GBS 64. This is a strong fingerprint — token-stream parity is not an open question.

=== F7. PROVENANCE GAP: THE MEGA ARM'S CODE IS NOT SNAPSHOTTED PER RUN ===
run_t1_mega.sh copies only `k0_train/mega_sitecustomize.py` into $OUT/hook; the rest of the package (including moe_swap.py) is mounted live read-only from /bench. moe_swap.py carries .bak_t3/.bak_v6/.bak_v62/.../.bak_v68 backups, i.e. it was edited in place between runs. The kernel object is selected by K0_MEGA_BWD_CO/K0_MEGA_WG_CO, neither of which is recorded in the run dir or echoed to the log. The sitecustomize hook md5 is identical (7fb8738d...) across t1v6ns/t1v6stack/t1v6fastfull2, but that proves nothing about moe_swap.py. Consequence: it cannot be *proven* from artifacts that t1v6ns and t1v6stack differ only in the yaml — which is exactly what the -73 ms claim asserts.

=== F8. ORDER BALANCE AND CLOCKS ===
run_t0_baselines.sh:78-79 is `for rep { for arm }` — arm order is fixed (vanilla, deepep, gg) in every repeat, never rotated. Arm position is therefore confounded with arm identity, and the measured position effects are not negligible (see F-variance below: +18.7 ms first-run penalty, +3.4 ms monotone thermal soak across three back-to-back runs). The mega arm never went through this runner at all — it has its own single-shot script.
`rocm-smi --showperflevel` returns `auto` on all 8 GPUs: clocks are not locked, DVFS is free to move. Idle junction temps already span 10 C (GPU0 68, GPU2 66, GPU5 66 vs GPU1/4/7 58). No rocm-smi telemetry is captured in any run directory, so the straggler's behavior during the banked runs is unrecoverable.

--- asymmetries ---

Direction is relative to the mega arm. "-" = handicaps mega, "+" = flatters mega.

--- AGAINST THE MEGA ARM ---

A1 (-11 to -14 ms). Aggregation window. The banked 10-iteration tail happens to contain both of production's anomalous fast final iterations (-72 ms each), and the mega arm has no comparable artifact. Banked vs steady-state median (iters 51-258): production rep2 1328.4 vs 1339.5 (-11.1 ms, -0.83%); production rep1 1329.5 vs 1343.2 (-13.7 ms, -1.02%); mega t1v6stack 1587.8 vs 1588.3 (+0.5 ms, +0.03%). This also silently discounts every T0 number: vanilla 1677.0/1684.5, turbo_deepep 1473.7/1483.5, turbo_gg-BF16 1383.3/1393.0 — all ~7-11 ms optimistic.

A2 (-9.2 ms, -0.69%). Day-to-day drift, measured not assumed. Identical config t0_turbo_gg-FP8D on a matched window (iters 5-12): Aug 15 06:30 = 1335.0 ms, Aug 15 06:47 = 1335.6 ms, Aug 17 23:14 (t0fp8dsmk) = 1344.6 ms. The node was ~0.7% SLOWER on the night the mega numbers were taken. The bar was measured in the fast regime; the mega arm in the slow regime. Note this is far smaller than feared — the days-apart pairing is a real but minor defect, and it is minor only because a same-day production smoke happens to exist.

A3 (-20 to -50 ms, estimated). Non-turbo backbone (F2). Mega denied megatron.turbo.rms_norm + megatron.fp8.context + transformer_engine.pytorch.fp8 on ~650 ms/iter of shared non-MoE work. Not correctable by arithmetic — requires a rerun on a turbo-enabled backbone.

A4 (-5 to -25 ms, estimated). Self-imposed synchronization the production arm does not pay:
  moe_swap.py:319 + :805-810 — `maybe_check_pperr` does `int(state.pperr.item())` every K0_MEGA_PPERR_EVERY=50 mega calls. At 4 layers x 8 microbatches x 2 = 64 calls/iter that is ~1.3 forced device->host syncs per iteration.
  moe_swap.py:321 — K0_MEGA_FWD_SYNC defaults to 1. T1_RESULTS itself states "the fwd-side sync alone caps ring runahead", and that post-launch syncs cost -200 ms/iter in t1smoke6.
Both are debug/safety instruments, not part of the algorithm.

A5 (unquantified, likely small). Fixed arm order in run_t0_baselines.sh means the production ladder's own internal deltas (+21.3% vanilla->turbo_gg) carry a position bias of up to ~18 ms.

--- FLATTERING THE MEGA ARM ---

B1 (+102 tok/s, +0.49%). Mixed-field banking. "1,587.8 ms / 20,739 tok/s" pairs the AVG ms field with the INST tok/s field. 20,739.2 is the inst value at iteration 260, which pairs with inst 1580.0 ms; the value that pairs with avg 1587.8 ms is the harmonic-mean field, 20,637.1 (check: 32,768 tok / 1.5878 s = 20,637). Production's banked pair (1328.4 / 24,667.5) IS self-consistent, so the error is one-sided.

B2 (+2 to 4 ms, +0.15-0.25%). No router/probs gradient (moe_swap.py:17-20, and line 1124 `None,  # probs grad: v2`). Skipped backward FLOPs: router weight [7168, 256], dgrad + wgrad = 30 GFLOP per layer-microbatch x 32 = 963 GFLOP/iter, ~2 ms at ~450 TFLOP/s, plus <1 ms of topk/aux-loss backward. IMPORTANT: the forward is NOT shortchanged — mega_sitecustomize.py:60 calls `self.shared_experts_compute(hidden_states)` and :61 calls `self.route(...)`, and :71-72 adds the shared expert back. I checked this specifically because dropping the shared expert would have been worth ~70 ms/iter; it is not dropped.

B3 (unpriced). HBM: mega 206.12 GB / 81.80% vs production 151.54 GB / 60.14% — +54.6 GB, +36%. At a fixed memory budget production could trade that headroom for a larger micro-batch; the comparison holds GBS fixed but not memory.

B4 (claim-level, not time). The loss-quality claim is not supportable in either direction (see magnitude_estimates F13).

--- FAVORING PRODUCTION ON THE QUALITY AXIS (a finding the writeup does not yet make) ---

C1. Rep selection. Production turbo_gg-FP8D rep1 suffered genuine loss excursions inside the timed window — iteration 232: 0.11446 and iteration 237: 0.15801, against a local median of 0.0106 (11x and 15x). Across all 18 full runs audited (three T0 BF16 arms x 3 reps, FP8/FP8H/FP8D x 2 reps, and every mega run) this is the ONLY run with any excursion. The banked bar is rep2 — the clean rep, which is also the faster rep on both the banked and the steady-state metric. T1_RESULTS calls this recipe "stable @100"; the excursions occur at 232/237, beyond that check.

--- magnitude_estimates ---

--- CORRECTED HEADLINE ---

As banked (mixed fields):              mega 20,739 / prod 24,667 = 0.8408x
Self-consistent banked:                mega 20,637 / prod 24,667 = 0.8366x   [B1 removed]
Steady-state medians (iters 51-258):   mega 1588.3 ms -> 20,631 tok/s
                                       prod 1339.5 ms -> 24,463 tok/s  = 0.8434x   [A1 removed]
+ drift-corrected to Aug-18 conditions (prod x 1.0069):
                                       prod ~1348.7 ms -> 24,295 tok/s  = 0.8492x  [A2 removed]
+ pooling both prod reps (1341.4 median) then drift:
                                       prod ~1350.6 ms -> 24,262 tok/s  = 0.8503x

Arithmetic-only corrections are worth +1.0 point to the mega arm (0.841x -> 0.850x).
With A3 + A4 fixed by rerun (25-75 ms off the mega arm): mega ~1513-1563 ms -> 20,965-21,658 tok/s = 0.864-0.892x.

Bottom line on risk: no combination of corrections reaches parity. The 0.84x claim is directionally safe; the point estimate is ~1 point too harsh as measured, and up to ~5 points too harsh if the config asymmetry is removed. The gap is real.

--- NOISE FLOOR (the number every sub-claim must clear) ---

Within-run scatter, steady window (iters 51-258, sd of inst):
  production turbo_gg-FP8D: 3.9-4.1 ms      mega t1v6stack: 3.1 ms      mega t1v6ns: 1.8 ms
  (the mega arm is measurably QUIETER than production — megakernel work is more deterministic)

Between-run, same day, same config:
  production FP8D rep1 vs rep2: banked 1329.5 vs 1328.4 (delta 1.1 ms); steady median 1343.2 vs 1339.5 (delta 3.7 ms, 0.28%)
  NOTE: the "<0.4% spread" cited in T0_RESULTS is computed on the fragile 10-iteration window. On steady-state medians the spread is comparable (~0.3%), so the claim survives — but it certifies a number nobody should be quoting.

Back-to-back mega smoke triples (it5-12 mean — the cleanest replicate data available):
  t1v62a/b/c: 1955.6 / 1954.4 / 1954.2   range 1.4 ms, sd 0.6
  t1v63a/b/c: 1985.1 / 1986.6 / 1988.5   range 3.4 ms, sd 1.4  (monotone increasing = thermal soak within the triple)
  t1v64a/b/c: 1942.5 / 1924.2 / 1923.8   range 18.7 ms, sd 8.7 (first-run-of-session penalty on rep a)

=> single back-to-back A/B pair resolves ~3 ms at best; ~19 ms if the "A" is the first run after a rebuild.
=> sigma for two runs launched separately in the same session: ~1.5-2 ms typical, ~9 ms worst case.

--- CLAIM-BY-CLAIM RISK ---

CLAIM "mega 1,587.8 ms / 20,739 tok/s, 0.84x of production"
  Risk: LOW on direction, MEDIUM on the point estimate. Fix B1 (tok/s should be 20,637), then correct A1+A2. Restate as ~0.85x with same-day paired runs, or ~0.86-0.89x on a turbo backbone.

CLAIM "-73 ms from the FP8 non-MoE stack (t1v6ns 1660.5 -> t1v6stack 1587.8)"
  Magnitude: SAFE. On steady medians 1658.0 -> 1588.3 = -69.7 ms, roughly 15-40 sigma. Both are full 260-iteration runs 34 minutes apart.
  Attribution: WRONG. Bundles megatron.grad_zero_and_data_prefetch, a free critical-path scheduling patch gated on fp8_recipe=delayed. Split it: run t1v6ns with fp8_recipe=delayed but fp8 disabled on the non-MoE layers, or measure the patch alone.
  Provenance: WEAK (F7) — the two runs' moe_swap.py is not snapshotted.

CLAIM "-7.5 ms from fastcvt"
  Risk: HIGH — NOT ESTABLISHED. 7.5 ms sits inside the demonstrated single-pair band (up to 18.7 ms via first-run penalty, and 3.4 ms of thermal drift accumulates across three consecutive runs). It is ~4-5 sigma of the *good* case and <0.5 sigma of the *bad* case. Needs an ABABAB triple in one session with the first run of the session discarded. Note the numerics confirm fastcvt is perf-only (t1v6ns iter-260 loss 9.163537e-3 vs t1v6fastfull2 9.164823e-3), so only the timing is in question.

CLAIM "mega loss 9.279e-3 beats production 1.05e-2"
  Risk: HIGH — NOT DEFENSIBLE, and the direction is not the interesting one.
  Both banked figures are single-iteration values. Within-run sd over iterations 211-260 is 1.5e-3 (mega) and 1.3e-3 (production) — the quoted gap is ~1 sigma of scatter.
  Proper statistic, mean over iters 211-260:
      mega t1v6stack             8.864e-3  (sd 1.51e-3)
      production FP8D rep2       1.086e-2  (sd 1.34e-3)
      production FP8D rep1       1.589e-2  (sd 2.50e-2 — contaminated by the 232/237 excursions)
      BF16 GOLD REFERENCE        9.837e-3 / 9.840e-3 / 9.841e-3  (turbo_deepep / turbo_gg / vanilla, 3 reps each, agreeing to 4 digits)
  Against the BF16 reference the mega arm is 9.9% BELOW and production is 10.4% ABOVE. A loss *below* the BF16 reference is not evidence of better numerics — most likely it reflects a different optimization trajectory from the frozen router (no probs gradient). Honest statement: both fp8 arms land within +-10% of the BF16 reference; neither deviation is evidence of numerical superiority.

CLAIM "production hybrid+delayed is the stable matched-precision bar"
  Risk: WEAKENED, in the mega arm's favor. 1 of 2 full reps shows 11x/15x loss excursions at iterations 232 and 237 — the only such event in 18 audited full runs. This is an unclaimed robustness result. It should be reported, and it raises the rep count needed for the production bar.

CLAIM "T0 repeat spread <0.4%"
  Risk: LOW but the number is measured on the wrong statistic (the 10-iteration tail). Recompute on steady-state medians; it holds (~0.3%).

CLAIM "kernels at parity with production GEMMs (fwd 4.75, dgrad 7.65 ms)"
  Not audited here (standalone microbenchmarks, different rig). Flagging only that these are cross-rank max / mid-half mean over n=20 per T1_RESULTS, which is a *different* discipline from the e2e numbers — do not present them in the same table without labelling the aggregation.

--- evidence ---

All paths on the node (ssh -i ~/.ssh/muhammad-gpu -p 2425 subvadla@10.5.95.87), read-only; no GPU runs launched.

CONFIGS
  ~/amd-master-m15pkt/auto-gpu-kernel/k0_fused_moe/training_bench/t0_vanilla-FP8D.yaml
     :9   exp_name: t0_vanilla-BF16   <- FP8D yaml carries the BF16 exp_name, so ALL mega runs write to a workspace directory named "t0_vanilla-BF16" regardless of precision. Labeling hazard: the run tree cannot be told apart by path.
     :32-33  fp8: hybrid / fp8_recipe: delayed   (the keys inserted after seq_length)
     :56-57  log_avg_skip_iterations: 50 / log_avg_reset_interval: 200   <- the 10-iteration window
     :66     enable_primus_turbo: false
  ~/.../t0_turbo_gg-FP8D.yaml
     :64-75  enable_primus_turbo: true, use_turbo_rms_norm: true, use_turbo_grouped_gemm: true, use_turbo_deepep: true, turbo_deepep_num_cu: 80, moe_use_fused_router_with_aux_score: true
     :77-79  fp8: hybrid / fp8_recipe: delayed
  Full diff between the two arms' yamls is exactly: exp_name, the fp8 key position, and the turbo block. Everything else (seq 4096, GBS 64, EP8/TP1/PP1, force_load_balancing, mock_data, seed 1234, 260 iters, overlap_grad_reduce/param_gather, moe_shared_expert_overlap: false) is identical.

BANKED LOG LINES (rank-7 debug.log, iteration 260)
  ~/k0-training-bench/t0fp8dfull/turbo_gg-FP8D_rep2_workspace/.../rank-7/debug.log
     "elapsed time per iteration (ms): 1269.6/1328.4 ... tokens/s/GPU inst/harmonic mean: 25809.7/24667.5 ... lm loss: 1.048341E-02"
  ~/k0-training-bench/t1v6stack/t1mega_workspace/.../rank-7/debug.log
     "elapsed time per iteration (ms): 1580.0/1587.8 ... tokens/s/GPU inst/harmonic mean: 20739.2/20637.1 ... lm loss: 9.278746E-03"
     -> banked "20,739" is the INST field; the field pairing with 1587.8 is 20,637.1.

WINDOW RESET (arithmetic proof, from consecutive log lines)
  prod rep2 iters 257-260 avg field: 1343.1, 1343.0, 1334.9, 1328.4 with inst 1343.6, 1342.2, 1270.3, 1269.6
     (1343.0*8+1270.3)/9 = 1334.9 exactly; (1334.9*9+1269.6)/10 = 1328.4 exactly  -> N=9 then N=10
  mega iters 257-260 avg field: 1589.5, 1589.5, 1588.7, 1587.8 with inst 1590.0, 1589.7, 1582.2, 1580.0
     (1589.5*8+1582.2)/9 = 1588.7 exactly; (1588.7*9+1580.0)/10 = 1587.8 exactly

STEADY-STATE vs BANKED (median of inst over iters 51-258, computed over all 210 logged lines)
  t0fp8dfull rep1   banked 1329.5  median 1343.2  sd 4.1   last2 = 1273.8, 1271.5
  t0fp8dfull rep2   banked 1328.4  median 1339.5  sd 3.9   last2 = 1270.3, 1269.6
  t1v6stack         banked 1587.8  median 1588.3  sd 3.1   last2 = 1582.2, 1580.0
  t1v6ns            banked 1660.5  median 1658.0  sd 1.8   last2 = 1658.0, 1658.6
  t1v6fastfull2     banked 1684.9  median 1690.2  sd 2.7   last2 = 1663.7, 1663.1
  t0full_0814T2314 (BF16, 3 reps each): deepep 1473.7/1483.5, 1476.7/1485.8, 1468.9/1479.6
                                        turbo_gg 1383.3/1393.0, 1386.2/1396.5, 1388.1/1397.5
                                        vanilla 1677.0/1684.5, 1678.6/1685.2, 1686.7/1698.2

DRIFT (matched window, iterations 5-12, identical config t0_turbo_gg-FP8D)
  t0fp8dfull rep1 (Aug 15 06:30): [1336,1329,1333,1334,1334,1336,1336,1342] mean 1335.0
  t0fp8dfull rep2 (Aug 15 06:47): [1327,1331,1332,1336,1339,1338,1340,1343] mean 1335.6
  t0fp8dsmk       (Aug 17 23:14): [1342,1343,1340,1344,1346,1344,1350,1348] mean 1344.6   -> +9.2 ms, +0.69%

REPLICATE TRIPLES (mega smokes, it5-12 mean)
  t1v62a/b/c 1955.6 / 1954.4 / 1954.2   |  t1v63a/b/c 1985.1 / 1986.6 / 1988.5  |  t1v64a/b/c 1942.5 / 1924.2 / 1923.8

WARMUP RAMP (inst, per arm)
  mega t1v6stack:  it1 34,676 ms  it2 18,229  it5-10 1584  it11-20 1591  it21-30 1596  it31-40 1599  it41-50 1587  it51-60 1586  it100-200 1589
  prod rep2:       it1 40,615 ms  it2 21,080  it5-10 1334  it11-20 1342  it21-30 1347  it31-40 1338  it41-50 1340  it51-60 1344  it100-200 1340

FP8 ACTIVATION (t1v6stack/t1mega.log)
  :498  fp8 : hybrid (str)          :505  fp8_recipe : delayed (str)          :506  fp8_wgrad : True (bool)
  :1215 [Patch] (skipped): megatron.fp8.context (condition not met)
  :1216 [Patch] (skipped): transformer_engine.pytorch.fp8 (condition not met)
  :1237-1239 [Patch] Applied: megatron.fp8.delayed_scaling_update (priority=40)
  :1242 [Patch:grad_zero_and_data_prefetch] DDP grad_data.zero_() on secondary stream; data HtoD prefetch on secondary stream
  :1355 Applied 26/26 patches   (vs t1v6ns 22/22, vs t0fp8dfull rep1 cli.log 31/31)
  Numerical proof: iter-51 loss 1.773915E-01 (t1v6stack, fp8) vs 1.683452E-01 (t1v6ns, no fp8) vs 1.683146E-01 (t1v6fastfull2, no fp8)

PATCH SET DIFF (production 31 minus mega 26)
  megatron.fp8.context | transformer_engine.pytorch.fp8 | megatron.turbo.rms_norm | megatron.moe_alltoall_dtoh_turbo_grouped_gemm | megatron.turbo.moe_dispatcher

TOKEN PARITY (seq_load_balancing_loss / consumed samples, all 18 runs, both days)
  it51 1.000192E+00 / 3264   it100 1.000193E+00 / 6400   it200 1.000192E+00 / 12800   it260 1.000194E+00 / 16640

LOSS STATISTICS (mean over iterations 211-260)
  t1v6stack 8.8637e-03 (sd 1.51e-3) | t1v6ns 8.7281e-03 | t1v6fastfull2 8.7299e-03
  t0fp8dfull rep2 1.0863e-02 (sd 1.34e-3) | rep1 1.5892e-02 (sd 2.50e-2, max 1.5801e-01)
  BF16 gold: turbo_deepep 9.8367/9.8366/9.8369e-03, turbo_gg 9.8400/9.8397/9.8403e-03, vanilla 9.8416/9.8412/9.8412e-03
LOSS EXCURSIONS (>5x local 31-iteration median, iterations >=120), across 18 full runs
  t0fp8dfull turbo_gg-FP8D rep1: iter 232 = 0.11446 (local median 0.01056), iter 237 = 0.15801 (local median 0.01058)
  every other run (incl. rep2, all BF16 arms, all FP8/FP8H arms, all mega runs): none

CODE
  k0_train/mega_sitecustomize.py:60  shared = self.shared_experts_compute(hidden_states)
                                :61  probs, _routing_map = self.route(hidden_states, padding_mask)
                                :71-72  if shared is not None: out = out + shared     <- shared expert and router forward NOT skipped
  k0_train/moe_swap.py:17-20   docstring: "probs receives no gradient (returns None)"
                      :319     self._pperr_check = int(os.environ.get("K0_MEGA_PPERR_EVERY", "50"))
                      :321     self.fwd_sync = os.environ.get("K0_MEGA_FWD_SYNC", "1") == "1"
                      :805-810 maybe_check_pperr -> err = int(state.pperr.item())   <- forced D2H sync ~1.3x/iter
                      :1124    None,  # probs grad: v2 (dw_sorted mapping)
  run_t1_mega.sh: copies only mega_sitecustomize.py into $OUT/hook; PYTHONPATH="/hook:/bench" imports k0_train live from the mutable bench dir
  run_t0_baselines.sh:78-79   for rep in $(seq 1 "$REPEATS"); do  for arm in "${arm_list[@]}"; do    <- fixed arm order, never rotated

NODE STATE
  rocm-smi --showperflevel: "auto" on all 8 GPUs (clocks unlocked)
  idle junction temps: GPU0 68, GPU1 58, GPU2 66, GPU3 62, GPU4 58, GPU5 66, GPU6 64, GPU7 58 C  (10 C spread at idle; GPU0 currently hottest)
  no rocm-smi telemetry captured in any run directory

MEMORY
  mega: hip 206.12 GB / 81.80%, rank-3 max 210.09 GB / 83.37%
  prod: hip 151.54 GB / 60.14%, rank-3 max 155.57 GB / 61.74%

DOCS
  T1_RESULTS.md last modified Aug 15 06:29 — it does NOT contain any Aug-17/18 number. The banked headline (1587.8 / 20,739 / -73 ms / -7.5 ms fastcvt) is un-journaled: it exists only in conversation, not in any results artifact on the node or in the repo.

--- fixes ---

=== IMMEDIATE, NO GPU TIME (do these before anything else) ===

FIX-1  Stop quoting the log's `avg` field. Set `log_avg_reset_interval: 1000000` in every yaml so the accumulator never resets, and independently compute the headline from the log as: median of per-iteration `inst` over iterations 51-258, explicitly dropping the final 2 iterations (documented end-of-run artifact worth -46 to -72 ms on production arms and -7 ms on the mega arm). Report p95 alongside the median. Re-derive every T0_RESULTS/T1_RESULTS number this way — all of them shift by +7 to +14 ms.

FIX-2  Fix the mixed-field pairing. Banked mega tok/s becomes 20,637 (the harmonic-mean field that pairs with 1587.8 ms), not 20,739. Rule: ms/iter and tok/s must come from the same field of the same line, and both arms must use the same field.

FIX-3  Restate the -73 ms attribution as "switching the non-MoE stack to the fp8-delayed recipe", and note that this bundles `megatron.grad_zero_and_data_prefetch` (a free critical-path scheduling patch) alongside fp8 arithmetic.

FIX-4  Withdraw the loss-quality claim as currently stated. Replace with: mean +- sd over iterations 211-260, against the BF16 gold reference (9.84e-3). Report that both fp8 arms sit within +-10% of it (mega 8.86e-3, production 1.086e-2) and that neither deviation is evidence of numerical superiority, given the mega arm's frozen router.

FIX-5  Report the production robustness finding: turbo_gg-FP8D rep1 shows 11x/15x loss excursions at iterations 232 and 237, unique across 18 audited full runs, and the banked bar is the clean rep. This is a legitimate result in the mega arm's favor that the current writeup does not make.

FIX-6  Provenance. Change run_t1_mega.sh to snapshot the ENTIRE k0_train/ tree into $OUT (not just mega_sitecustomize.py), plus md5sum of every .co under $K0_MEGA_DIR, the full K0_MEGA_* env, and `git rev-parse HEAD`. Also write the merged yaml (the --export_config path is already there but t1mega_merged.yaml is missing from t1v6stack). And give the FP8D yaml its own exp_name so the workspace path is not a lie.

=== THE MINIMAL DEFENSIBLE PROTOCOL (one session, ~2 hours GPU) ===

Prerequisites (5 min):
  P0  Lock clocks: `rocm-smi --setperfdeterminism <MHz>` at a level all 8 GPUs sustain, or at minimum start a background `rocm-smi --showtemp --showclocks --csv` sampler at 10 s into $OUT for every run. Currently perflevel=auto and nothing is logged.
  P1  Build `t0_turboback-FP8D.yaml`: the production yaml verbatim (enable_primus_turbo: true, use_turbo_rms_norm: true, use_turbo_attention: false, fp8 hybrid/delayed) with ONLY the MoE implementation swapped to the megakernel. Set use_turbo_deepep / use_turbo_grouped_gemm false since the swap bypasses them, but keep turbo enabled so the shared non-MoE backbone gets megatron.turbo.rms_norm, megatron.fp8.context and transformer_engine.pytorch.fp8. This is the apples-to-apples mega arm and it removes the single largest asymmetry.
  P2  Sub-claim controls, same session: K0_MEGA_PPERR_EVERY=100000000 and K0_MEGA_FWD_SYNC=0 as an A/B against the defaults, to price the self-imposed syncs.

Headline (8 runs, ~75-90 min at ~9 min/run for 260 iterations + startup):
  P3  ABBA ABBA. A = production t0_turbo_gg-FP8D, B = mega t0_turboback-FP8D. N=4 pairs, order-balanced so arm identity is not confounded with session position or thermal soak. DISCARD the first run of the session entirely (measured first-run penalty: +18.7 ms) — so launch a throwaway 12-iteration smoke first.
  P4  Report the PAIRED difference (B_i - A_i for each pair) with its mean and sd, not two independent means. With within-arm sd ~3-4 ms and a ~250 ms effect, N=3 pairs already gives >50:1 confidence on the headline; N=4 buys ~5 ms resolution for the sub-claims and costs one extra 18-minute cycle.
  P5  Report the mega arm's HBM alongside (currently +54.6 GB / +36%) so the resource asymmetry is disclosed rather than discovered.

Sub-10 ms claims (fastcvt, pperr sync, fwd sync) — 6 runs each, ~25 min each:
  P6  Back-to-back ABABAB in ONE session, N=3 each, 260 iterations (not smokes — the 12-iteration smokes have sd 60+ ms because they include the ramp). Report the paired mean difference +- sd. The -7.5 ms fastcvt claim does not survive its current single-pair evidence and must either clear this bar or be dropped from the writeup.

Quality:
  P7  >=3 reps per arm at 260 iterations; report loss as mean +- sd over iterations 211-260, plus an explicit excursion scan (>5x local 31-iteration median, iterations >=120) for every run. Publish excursion counts per arm — that is the honest way to state the FP8-stability finding in both directions.

=== WHAT THIS BUYS ===
The headline moves from "1,587.8 vs 1,328.4 = 0.84x, measured 3 days apart, N=1 vs N=2, on mismatched backbones, from a 10-iteration tail" to "paired same-day, order-balanced, N=4, matched backbones, steady-state median of 208 iterations". Expected landing zone: 0.86-0.89x. The verdict does not flip, which is exactly why running this is worth it — it converts a number an adversarial reviewer can dismantle into one they cannot.

--- surprises ---

1. The training comparison is far cleaner than the serving one. I went in expecting another 2%-of-heavy-steps catastrophe and did not find one. Token streams are provably identical to 7 significant figures across both arms and both days; cold start is fully excluded and both arms are within 0.5% of steady state by iteration 5; within-run sd is 2-4 ms on a 1.3-1.6 s iteration. The measurement fidelity here is genuinely good — the defects are in aggregation and config parity, not in whether the kernel ran.

2. The aggregation bug is invisible unless you do the arithmetic. `log_avg_reset_interval: 200` reads like "average 200 iterations" and actually means "reset the accumulator at iteration 251, then average 10". I only caught it because the avg field moved 8.1 ms in a single step, which is impossible with 200 samples. Every number in T0_RESULTS.md and T1_RESULTS.md is affected.

3. The end-of-run artifact is real wall-clock, not a timer bug, and it is arm-dependent. Production drops 46-72 ms in its final 2 iterations in EVERY run of EVERY production arm across both days; the mega arm drops 7 ms. I confirmed against log timestamps (prod iters 257->260 span 3.882 s = 1294.0 ms/iter, matching the inst values exactly). I could not determine the mechanism. It only matters because the 10-iteration window happens to contain exactly those 2 iterations — with FIX-1 it becomes harmless. Worth understanding anyway, since a stack that gets 5% faster when training is about to end is doing something on the critical path it does not need to do.

4. The two arms are running different framework stacks and nobody flagged it. `enable_primus_turbo: false` on the mega arm is not a neutral choice — it costs the shared non-MoE backbone three optimization patches, two of which (megatron.fp8.context, transformer_engine.pytorch.fp8) are fp8 patches that the mega arm qualifies for on the merits and is denied on a technicality. The mega arm's own log prints "(condition not met)" for both. This is the one finding that is worth GPU time to fix rather than arithmetic to correct.

5. A same-day production control already existed and nobody used it. t0fp8dsmk was run on Aug 17 at 23:14, five hours before the banked mega number, with the exact production config. It answers the thermal-drift question directly: +9.2 ms (+0.69%), against the mega arm. Without it the days-apart pairing would have been an open-ended worry; with it, it is a bounded 0.7% correction. This is a strong argument for always shooting a same-day control smoke of the opposing arm, at trivial cost.

6. The banked tok/s and the banked ms/iter for the mega arm come from different fields of the same log line, and the error runs in the mega arm's own favor. 20,739 is the instantaneous rate of the single fastest iteration of the run; the value pairing with 1587.8 ms is 20,637. Production's pair is self-consistent. A reviewer who divides 32,768 by 1.5878 finds this in ten seconds.

7. The production bar has a loss-stability problem that the writeup does not claim. One of two full reps of the "stable matched-precision bar" shows 11x and 15x loss excursions at iterations 232 and 237 — the only such event across 18 full runs I scanned, including three BF16 arms at 3 reps each. T1_RESULTS certifies this recipe as "stable @100"; the excursions are at 232/237. The banked bar is the clean rep, which is also the faster rep. This is the mirror image of the usual cherry-pick worry and it favors the incumbent.

8. The mega arm is measurably more deterministic than production (sd 1.8-3.1 ms vs 3.9-4.1 ms) and has a shorter cold start (34.7 s vs 40.5 s for iteration 1). Neither is in the writeup. Both are real, cheap, defensible claims sitting unused in the logs.

9. The mega arm handicaps itself with debug instrumentation that is on by default. K0_MEGA_PPERR_EVERY=50 forces a `.item()` device-to-host sync roughly 1.3 times per iteration, and K0_MEGA_FWD_SYNC defaults to 1 despite T1_RESULTS explicitly documenting that the forward sync caps ring runahead. Production has no equivalent. Pricing these is two runs.

10. The headline is un-journaled. T1_RESULTS.md was last touched Aug 15 06:29 and contains none of the Aug-17/18 numbers. The entire banked claim — 1,587.8 / 20,739 / -73 ms / -7.5 ms — exists only in conversation. That is how a mixed-field tok/s value survives to become a headline.
