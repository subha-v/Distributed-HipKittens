# overnight/ — date-organized experiment log

One folder per overnight session. Ledgers inside a session folder are
append-only and archived verbatim (provenance): supersede, never rewrite.

| folder | session | contents |
|---|---|---|
| `aug10/` | 2026-08-10 → 08-11 MPS loop | CLAUDE.md (loop mandate), STATUS.md (morning read: **mps_mega ratchet 0.888× production**), experiments exp_01–exp_21, LESSONS.md, harness tools |
| `aug11/` | 2026-08-11 planning | exp_22 (NanoFlow-Fig-7-analog saturation ubench), exp_23 (NanoFlow-v2-Fig-10-analog resource timeline) — planned, unbuilt |
| `aug18-prefill/` | 2026-08-17 prefill line | ROUTE_REPLAY_PLAN.md + `route_capture/` — serving raw-route capture hook (`skewhook_v2/`) and MoK captured-route replay (`mok_patch/`, `K0_MOK_ROUTE_FILE` / `K0_MOK_ROUTE_ORDER={captured,shuffled}`): the never-run A/B that separates expert popularity from run correlation. Built, locally tested, unrun on GPU |

Numbering is continuous across sessions (aug10 ended at exp_21; aug11 starts
at exp_22).

**Path note:** aug10 files were written when they lived at `overnight/` root,
so their relative links one level up (`../DESIGN_MPS.md`,
`../k0pf6gm_device_tile_mps.hip`, `../MPS_OVERNIGHT_HANDOFF.md`) now resolve
one level short — read them as `../../` from inside `aug10/`. Contents were
deliberately left byte-identical rather than rewritten.
