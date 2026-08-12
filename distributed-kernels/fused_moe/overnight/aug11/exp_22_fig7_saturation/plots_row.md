# Paste-ready row(s) for aug11/PLOTS.md

`aug11/PLOTS.md` is outside this experiment's ownership, so the row lives here
for the orchestrator to merge rather than being written directly.

| figure | what it shows | data file | generating experiment | status |
|---|---|---|---|---|
| Fig 2 | per-resource throughput vs CTA count on MI350X (MFMA / HBM / xGMI), each curve isolated and concurrent, with the knee annotated — the NanoFlow-Fig-7 analog | `exp_22_fig7_saturation/saturation.json` (schema `exp22-saturation-1`, documented in that folder's `result.md` §1) | exp_22 | **built, CPU gate green, awaiting GPU lease** — ubench + one-command sweep + summarizer all gated; `saturation.json` not yet produced |

Plot script contract (per plan.md §"Plot spec"), for whoever draws it:

* three panels, x = CTA count, y = TFLOPS / GB/s / GB/s;
* series per panel: `concurrency == "isolated"` vs `"concurrent"`, with `mlp` as
  the line style in panel c;
* dashed horizontal ceilings at 76.8 (single link), 537.6 (aggregate egress),
  8,000 (HBM), 2,300 (bf16 TFLOPS), and 148 GB/s (the cost model's service-rate
  requirement) — all of them are in `saturation.json.peaks_used`;
* knee annotation = `derived.knees[<series>].knee_ctas` (smallest C reaching 90%
  of that series' plateau); do not re-derive it;
* drop, or mark, any concurrent point whose
  `derived.concurrent_over_isolated[...].concurrent_is_really_concurrent` is
  false — that flag means the two roles did not overlap enough for the point to
  mean what the axis label says.
