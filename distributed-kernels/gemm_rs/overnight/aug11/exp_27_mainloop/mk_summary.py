#!/usr/bin/env python3
"""Emit kloop_census_summary.json — the plot-ready artifact for exp_27.

Every field carries a `basis` of measured / counted / derived / inferred.
Nothing here runs on a GPU; the measured inputs are quoted from
aug11/exp_20_attribution/{ablation,counters}.json.
"""
import json
import math
import os

CLK = 1.9e9  # pinned sclk, Hz

# ---- COUNTED: k-loop census, from isa/gemm_rs_mi300x-...-gfx942.s -----------
INST = {
    "<32,64,128,false>":  dict(bm=32, bn=64, bk=128, k_tail=False, vgpr=98,
                               sgpr=106, spill=0, waves=4, blocks=1,
                               mfma=8, ds_read=16, ds_write=6, gload=3,
                               valu=81, salu=22, barrier=1, waitcnt=6,
                               nop=2, branch=2, total=147),
    "<64,128,64,false>":  dict(bm=64, bn=128, bk=64, k_tail=False, vgpr=104,
                               sgpr=106, spill=0, waves=4, blocks=1,
                               mfma=16, ds_read=16, ds_write=6, gload=3,
                               valu=63, salu=23, barrier=1, waitcnt=6,
                               nop=2, branch=1, total=137),
    "<128,192,32,true>":  dict(bm=128, bn=192, bk=32, k_tail=True, vgpr=136,
                               sgpr=106, spill=0, waves=3, blocks=10,
                               mfma=24, ds_read=14, ds_write=6, gload=3,
                               valu=85, salu=52, barrier=1, waitcnt=6,
                               nop=0, branch=6, total=197),
    "<256,256,32,false>": dict(bm=256, bn=256, bk=32, k_tail=False, vgpr=246,
                               sgpr=106, spill=0, waves=2, blocks=1,
                               mfma=64, ds_read=24, ds_write=8, gload=4,
                               valu=58, salu=24, barrier=1, waitcnt=6,
                               nop=3, branch=2, total=194),
    "<256,256,32,true>":  dict(bm=256, bn=256, bk=32, k_tail=True, vgpr=248,
                               sgpr=106, spill=0, waves=2, blocks=4,
                               mfma=64, ds_read=24, ds_write=8, gload=4,
                               valu=98, salu=46, barrier=1, waitcnt=6,
                               nop=3, branch=3, total=257),
    "<32,64,64,false>":   dict(bm=32, bn=64, bk=64, k_tail=False, vgpr=91,
                               sgpr=106, spill=0, waves=5, blocks=6,
                               mfma=4, ds_read=8, ds_write=4, gload=2,
                               valu=52, salu=25, barrier=1, waitcnt=6,
                               nop=3, branch=4, total=163),
    "<32,64,64,true>":    dict(bm=32, bn=64, bk=64, k_tail=True, vgpr=92,
                               sgpr=106, spill=0, waves=5, blocks=10,
                               mfma=4, ds_read=8, ds_write=4, gload=2,
                               valu=76, salu=40, barrier=1, waitcnt=6,
                               nop=1, branch=6, total=148),
}

# ---- MEASURED: exp_20 ------------------------------------------------------
SHAPES = {
    5: dict(inst="<256,256,32,false>", full_us=641.9, gemm_pool_us=305.2,
            mfma_busy_us=101.7, trips=112, ng=272, waves=2, k_iters=56,
            mfma_instrs=14680064),
    6: dict(inst="<256,256,32,true>", full_us=1632.5, gemm_pool_us=992.2,
            mfma_busy_us=421.2, trips=464, ng=256, waves=4, k_iters=116,
            mfma_instrs=60817408),
}
FULL_ALL = [67.1, 67.5, 87.7, 203.5, 641.9, 1632.5]


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def cycle_budget(idx):
    s = SHAPES[idx]
    c = INST[s["inst"]]
    cyc_trip = s["gemm_pool_us"] * 1e-6 * CLK / s["trips"]
    mfma_simd = 2 * c["mfma"] * 16          # 2 waves/SIMD, 16 cyc/instr
    lds_bytes = 8 * (c["ds_read"] + c["ds_write"]) * 512
    l1_bytes = 8 * c["gload"] * 1024
    return {
        "shape_index": idx,
        "instantiation": s["inst"],
        "trips_per_producer_cta": s["trips"],
        "gemm_pool_us": {"v": s["gemm_pool_us"], "basis": "measured"},
        "mfma_busy_us_device": {"v": s["mfma_busy_us"], "basis": "measured"},
        "cycles_per_trip": {"v": round(cyc_trip, 1), "basis": "derived",
                            "from": "gemm_pool_us * 1.9GHz / trips"},
        "pipes": [
            {"name": "mfma", "scope": "per SIMD", "cycles": mfma_simd,
             "basis": "counted instrs x measured 16.0 cyc/instr"},
            {"name": "valu", "scope": "per SIMD", "cycles": 2 * c["valu"] * 4,
             "basis": "counted instrs", "rate_assumed": "4 cyc / wave64 op"},
            {"name": "lds", "scope": "per CU", "cycles": lds_bytes // 128,
             "bytes": lds_bytes, "basis": "counted bytes",
             "rate_assumed": "128 B/clk/CU"},
            {"name": "l1_global", "scope": "per CU", "cycles": l1_bytes // 64,
             "bytes": l1_bytes, "basis": "counted bytes",
             "rate_assumed": "64 B/clk/CU"},
            {"name": "scalar", "scope": "per CU", "cycles": 8 * c["salu"],
             "basis": "counted instrs", "rate_assumed": "1 op/clk"},
            {"name": "issue_slots", "scope": "per SIMD",
             "cycles": 2 * c["total"], "basis": "counted instrs",
             "rate_assumed": "1 instr/clk"},
        ],
        "mfma_floor_cycles_per_trip": {"v": mfma_simd, "basis": "measured"},
        "residual_cycles_per_trip": {"v": round(cyc_trip - mfma_simd, 1),
                                     "basis": "derived"},
        "residual_share": round(1 - mfma_simd / cyc_trip, 4),
        "residual_us": round(s["gemm_pool_us"] * (1 - mfma_simd / cyc_trip), 1),
        "accounted_cycles_lo_hi": [500, 800],
        "unaccounted_cycles_lo_hi": [round(cyc_trip - mfma_simd - 800),
                                     round(cyc_trip - mfma_simd - 500)],
    }


CLOSURE = [
    dict(arm="S=2, BK=32  (today)", lds_bytes=65536, barriers_per_tile=116,
         mfma_per_barrier=256 * 256 * 32, verdict="argmax — incumbent"),
    dict(arm="S=1, BK=64  (dispatch direction 1)", lds_bytes=65536,
         barriers_per_tile=116, mfma_per_barrier=256 * 256 * 32,
         verdict="EXACT WASH — closed"),
    dict(arm="S=2, BK=16  (dispatch direction 2)", lds_bytes=32768,
         barriers_per_tile=231, mfma_per_barrier=256 * 256 * 16,
         verdict="strictly worse — closed"),
    dict(arm="S=4, BK=16", lds_bytes=65536, barriers_per_tile=231,
         mfma_per_barrier=256 * 256 * 16, verdict="strictly worse — closed"),
]

CANDIDATES = [
    dict(id="A", name="counter pass to split the residual", dcyc=[0, 0],
         s6_us=[0, 0], s5_us=[0, 0], risk="none", gate="n/a",
         experiment="rocprofv3 SQ_WAIT_*/SQ_LDS_* on shapes 5,6; 20 min",
         basis="n/a", rank=1),
    dict(id="B", name="hoist load_commit above half-1 MFMAs",
         dcyc=[-300, -150], s6_us=[-73, -37], s5_us=[-23, -11], risk="low",
         gate="lds_race_check + M2 ISA position + M3 both tol + M9 null",
         experiment="one-line move; paired M7, 3 draws, half reversed",
         basis="inferred", rank=2, recommended=True),
    dict(id="C", name="prefill swizzled LDS offsets + SGPR LDS bases",
         dcyc=[-230, -100], s6_us=[-60, -25], s5_us=[-18, -8],
         risk="low-med", gate="M2 VGPR<=248 + M3 both tol + lds_race_check",
         experiment="port donor prefill_swizzled_offsets/readfirstlane bases",
         basis="inferred", rank=3),
    dict(id="D", name="retire K_TAIL=true on shape 6 (zero-pad in LDS)",
         dcyc=[-400, -250], s6_us=[-90, -40], s5_us=[0, 0], risk="med",
         gate="M3 @2e-3 all 17 + NEW fragment-layout fingerprint",
         experiment="adapter prototype; rows 3 and 6 are the only K_TAIL rows",
         basis="inferred", rank=4),
    dict(id="E", name="phase-offset warp-group ping-pong (donor)",
         dcyc=[-1400, -600], s6_us=[-350, -150], s5_us=[-105, -45],
         risk="high", gate="protocol-review MANDATORY + full ladder + M9",
         experiment="design now, build only after A reports", basis="inferred",
         rank=5),
    dict(id="F", name="wave-private LDS staging (barrier-free)",
         dcyc=[-1600, -800], s6_us=[-400, -250], s5_us=[None, None],
         risk="high", gate="full ladder; may relocate the bound into L2",
         experiment="data-path rewrite", basis="inferred", rank=6),
    dict(id="G", name="ds_read_b128 via a custom 16x32 shared layout",
         dcyc=[-150, -60], s6_us=[-50, -20], s5_us=[-15, -6], risk="med-high",
         gate="fragment-layout fingerprint + M3 both tol",
         experiment="large; not before A", basis="inferred", rank=7),
    dict(id="H", name="v_mfma_f32_32x32x8 atom", dcyc=[-150, -50],
         s6_us=[-60, -20], s5_us=[-18, -6], risk="med",
         gate="fragment-layout fingerprint + M3 both tol",
         experiment="new fragment types; identical peak, schedule-only",
         basis="inferred", rank=8),
    dict(id="X", name="S=1 BK=64 / rolling half-BK", dcyc=[0, 0],
         s6_us=[0, 0], s5_us=[0, 0], risk="n/a",
         gate="n/a", experiment="CLOSED on arithmetic (closure_table)",
         basis="derived", rank=99),
]


def ev(s5, s6, label):
    xs = FULL_ALL[:4] + [s5, s6]
    g = geomean(xs)
    return dict(scenario=label, shape5_us=round(s5, 1), shape6_us=round(s6, 1),
                geomean_us=round(g, 2),
                delta_pct=round(100 * (g / geomean(FULL_ALL) - 1), 2))


def main():
    out = {
        "experiment": "exp_27_mainloop",
        "what": "static cycle account of the GEMM k-loop + ranked mechanisms",
        "provenance": {
            "gemm_rs_mi300x.cpp_sha256_16": "5f9f47258842c64d",
            "gemm_rs_mi300x_hk_adapter.cuh_sha256_16": "e9b3d1c17d49ee0b",
            "gemm_rs_mi300x_constants.cuh_sha256_16": "c993b2a6004a6977",
            "hipcc": "HIP 7.2.53211-c2d9476115",
            "utc": "20260812T110649Z",
            "clock_hz": CLK,
            "gpu_used": False,
        },
        "method": ("instruction counts COUNTED from hipcc --save-temps ISA via "
                   "kloop_hist.py back-edge loop detection; pool and MFMA-busy "
                   "figures MEASURED by exp_20; cycle rates other than the MFMA "
                   "rate are ASSUMED and labelled per line; candidate deltas are "
                   "INFERRED"),
        "instantiations": [dict(name=k, **v) for k, v in INST.items()],
        "cycle_budget": [cycle_budget(5), cycle_budget(6)],
        "closure_table": CLOSURE,
        "candidates": CANDIDATES,
        "peak_calibration": {
            "shape6_gflop_per_rank": 496.1,
            "shape6_tflops_over_gemm_pool": 500.0,
            "producer_cu_peak_tflops_at_1900mhz": 996.0,
            "fraction_of_producer_peak": 0.502,
            "tuned_library_reference": "0.60-0.75 of peak on MI300X bf16",
            "basis": "derived from measured pool + counted geometry",
        },
        "expected_value": [
            ev(641.9, 1632.5, "today"),
            ev(626.9, 1582.5, "row B at its pre-registered -5% of the pool"),
            ev(551.9, 1382.5, "realistic full capture (~70% of producer peak)"),
            ev(437.9, 1061.5, "entire non-MFMA mainloop (roofline, unreachable)"),
        ],
        "graded_gap": {
            "today_geomean_ratio_vs_rank1": 1.098,
            "per_shape_ratio": [0.850, 1.072, 1.102, 1.120, 1.286, 1.208],
            "after_row_B": 1.092,
            "after_realistic_capture": 1.046,
            "note": ("exp_24 measured a ~90 us constant both arms pay per graded "
                     "call, which compresses the ratio toward 1; netting it out "
                     "puts the honest kernel-to-kernel ratio near 1.14x. Report "
                     "both protocols, never blend."),
        },
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "kloop_census_summary.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
    for e in out["expected_value"]:
        print("  %-58s geomean %8.2f us  %+6.2f%%"
              % (e["scenario"], e["geomean_us"], e["delta_pct"]))
    for b in out["cycle_budget"]:
        print("  shape %d: %s cyc/trip, mfma floor %s, residual %s (%.1f%%)"
              % (b["shape_index"], b["cycles_per_trip"]["v"],
                 b["mfma_floor_cycles_per_trip"]["v"],
                 b["residual_cycles_per_trip"]["v"],
                 100 * b["residual_share"]))


if __name__ == "__main__":
    main()
