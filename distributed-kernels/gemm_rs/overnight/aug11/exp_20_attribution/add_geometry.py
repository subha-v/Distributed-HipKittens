"""Additively enrich ablation.json with per-shape geometry, x SOL and M7 bound.

Additive on purpose: it loads the JSON, adds keys, writes it back. Nothing is
recomputed, so it cannot disagree with the measured stage table it is annotating.

Geometry is ARITHMETIC on the landed shape table (host_abi.hpp `scored_shapes`),
not a measurement -- it costs no GPU time. `k_local` is K/8 because the GEMM's K
range is partitioned across the 8 ranks, so a rank's mainloop runs over K/8, and
that is what sets k_iters.

`bound` and the M7 mean vector are merged from m7_results.json when present.
M7 labels a shape bound=HOST when issuing eight launches from one CPU thread
costs more than 0.9x the measured wall, i.e. the number describes the host, not
the kernel -- which is the caveat that makes shapes 1 and 2 unrankable here.

Usage: add_geometry.py <ablation.json> [m7_results.json] [clocks_before=...]
"""

import json
import sys

# (BM, BN, BK, NUM_REDUCER_CTAS) verbatim from gemm_rs_mi300x_host_abi.hpp:109-116
TILE_TABLE = {
    1: (32, 64, 128, 56),
    2: (64, 128, 64, 32),
    3: (128, 192, 32, 32),
    4: (256, 256, 32, 32),
    5: (256, 256, 32, 32),
    6: (256, 256, 32, 48),
}
# Published task.yml speed-of-light, microseconds: the bf16 MFMA roofline with
# ZERO budget for the reduce-scatter, so unreachable by construction.
SOL_US = {1: 6.46, 2: 8.19, 3: 23.04, 4: 65.54, 5: 131.07, 6: 379.43}
CU_COUNT = 304
WORLD = 8
RELEASE_GROUP = 4


def main():
    path = sys.argv[1]
    m7_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") \
        else None
    extra = dict(kv.split("=", 1) for kv in sys.argv[2:] if "=" in kv)

    doc = json.load(open(path))

    m7 = None
    if m7_path:
        try:
            m7 = json.load(open(m7_path))
        except (OSError, ValueError) as exc:
            print(f"  no M7 merge ({exc})")

    for record in doc["shapes"]:
        index = record["shape_index"]
        m, n, k = record["m"], record["n"], record["k"]
        bm, bn, bk, nr = TILE_TABLE[index]
        num_gemm_ctas = CU_COUNT - nr
        k_local = k // WORLD
        k_iters = -(-k_local // bk)
        tiles = (m // bm) * (-(-n // bn))
        waves = -(-tiles // num_gemm_ctas)
        tiles_per_cta = -(-tiles // num_gemm_ctas)
        # RELEASE_GROUP_FULL_ONLY=1 refuses a PARTIAL group, so a CTA owning
        # fewer than RELEASE_GROUP tiles falls back to a release per tile. This
        # is why release grouping is active on exactly one graded shape.
        rgroup = RELEASE_GROUP if tiles_per_cta >= RELEASE_GROUP else 1
        record["geometry"] = {
            "bm": bm, "bn": bn, "bk": bk,
            "num_reducer_ctas": nr,
            "num_gemm_ctas": num_gemm_ctas,
            "k_local": k_local,
            "k_iters": k_iters,
            "gemm_tiles": tiles,
            "producer_waves": waves,
            "waves_x_k_iters": waves * k_iters,
            "last_wave_fill": round(
                (tiles - (waves - 1) * num_gemm_ctas) / num_gemm_ctas, 4),
            "tiles_per_cta": tiles_per_cta,
            "effective_release_group": rgroup,
            "release_grouping_active": rgroup > 1,
        }
        record["sol_us"] = SOL_US[index]
        record["x_sol"] = round(record["full_us"] / SOL_US[index], 2)

        if m7:
            key = str(index - 1)
            means = m7.get("means_us") or []
            detail = (m7.get("detail") or {}).get(key, {})
            if len(means) >= index:
                record["m7"] = {
                    "mean_us": round(means[index - 1], 2),
                    "device_max_us": round(m7["device_max_us"][index - 1], 2),
                    "samples_us": m7.get("samples_us", {}).get(key),
                    "full_vs_m7_mean": round(
                        record["full_us"] / means[index - 1], 4),
                }
                host = detail.get("host") or {}
                issue = host.get("host_issue_us_per_op")
                if issue is not None:
                    record["m7"]["host_issue_us_per_op"] = round(issue, 2)
                    record["m7"]["bound"] = (
                        "HOST" if issue > 0.9 * means[index - 1] else "device")

    doc["schema"]["shapes[].geometry"] = (
        "arithmetic on the landed shape table, NOT a measurement. "
        "waves_x_k_iters is the overhead-bound cost column (producer waves x "
        "mainloop iterations per tile); effective_release_group is 1 wherever "
        "tiles_per_cta < RELEASE_GROUP, because RELEASE_GROUP_FULL_ONLY=1 "
        "refuses a partial group.")
    doc["schema"]["shapes[].x_sol"] = (
        "full_us / the published task.yml SOL for this shape. The SOL table is "
        "the bf16 MFMA roofline with no budget for the reduce-scatter, so it is "
        "a scale marker, not a target.")
    if m7:
        doc["schema"]["shapes[].m7"] = (
            "same-config M7 gate timing (3 rotations x 50 pipelined iters, "
            "order-rotated). full_vs_m7_mean is the freshness cross-check: the "
            "ablation's `full` arm is a separate build of the same source, so "
            "it should agree with M7 within the shape's allocation-noise floor. "
            "bound=HOST means issuing eight launches from one CPU thread costs "
            ">0.9x the wall, i.e. the number describes the host, not the kernel.")
    for key in ("before", "after"):
        if key in extra or f"clocks_{key}" in extra:
            doc.setdefault("clocks", {})[key] = extra.get(
                key, extra.get(f"clocks_{key}", ""))

    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2)

    header = (f"{'shape':<18}{'full':>9}{'xSOL':>7}{'tile':>13}{'NR':>4}"
              f"{'tiles':>7}{'wav':>5}{'k_it':>6}{'w*ki':>7}{'rgrp':>6}"
              f"{'M7 mean':>9}{'f/M7':>7}{'bound':>8}")
    print(f"wrote {path}")
    print(header)
    for record in doc["shapes"]:
        g = record["geometry"]
        m7r = record.get("m7", {})
        tile = "{}/{}/{}".format(g["bm"], g["bn"], g["bk"])
        print(f"{record['shape']:<18}{record['full_us']:>9.1f}"
              f"{record['x_sol']:>7.2f}"
              f"{tile:>13}"
              f"{g['num_reducer_ctas']:>4}{g['gemm_tiles']:>7}"
              f"{g['producer_waves']:>5}{g['k_iters']:>6}"
              f"{g['waves_x_k_iters']:>7}{g['effective_release_group']:>6}"
              f"{m7r.get('mean_us', float('nan')):>9.1f}"
              f"{m7r.get('full_vs_m7_mean', float('nan')):>7.3f}"
              f"{m7r.get('bound', '-'):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
