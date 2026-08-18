# M20 serving pair #1 — REMOVED (obsolete serving methodology, 2026-08-18)

This file held the stock-vs-m20 c32p serving pair of 2026-08-14 (a −19.6%
end-to-end result) and the reasoning built on it. It was removed on 2026-08-18:
per-step instrumentation showed the megakernel's activation seal fired on only
~2% of the padded-4096 heavy steps (so the candidate arm ran production kernels
for ~98% of the heavy MoE work), the candidate arm additionally ran with no
cudagraph at all on the unsealed steps, and both arms' baselines were depressed
by a uniform-decode rank running the whole model eagerly. Every delta here was
therefore measured against a broken control with a doubly-handicapped candidate.

Historical content: `git log --follow -- <this path>`.
Correction and the new protocol: `../../../../docs/distributed/SERVING_BENCHMARK_METHODOLOGY.md`,
`../aug18-prefill/M23_RAGGED_SEAL_DESIGN.md`, `../aug18-prefill/m23/M23_IMPL_NOTES.md`.

What survives (not a serving A/B, unaffected): the m18diag stock routing
histograms show **per-layer damage is uniform** — raw max rank load ≈5.5× on
every one of the 58 routed layers, per-layer replication benefit ≈4.45× each, so
there are no "most-damaged" layers and a budgeted 12/58-layer replica cache can
address only ~1/5 of the replication opportunity regardless of which layers are
picked. The memory dial itself was verified working (KV pool −7.88 GiB = the
configured 8,065 MB cache exactly).
