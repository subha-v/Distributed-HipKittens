"""Add the per-shape rollup that counters.json's deliverable schema requires.

Additive: the `cells` block stays exactly as collected and remains the
provenance for every number here. This only merges a shape's three counter
groups and derives the per-shape quantities the attribution is ranked with.

Why a merge is needed at all: only four TCC counters fit in one pass, so a
shape's fabric volume (g1), latency and writeback total (g2), and writeback
origin and mtype mix (g3) come from three separate dispatch windows of the same
config. Counts are clock- and schedule-independent, so merging them is sound --
but a RATIO between two counters is only meaningful when both came from the SAME
pass, which is why TCC_EA0_WRREQ_LEVEL and TCC_EA0_WRREQ were deliberately
collected together and the ratio is flagged with the pass it came from.

Usage: rollup_counters.py <counters.json> [ablation.json]
"""

import json
import re
import sys

WORLD = 8
BYTES_PER_ELEM = 2          # bf16 output
EA_LINE = 64                # full-width EA transaction

SHAPES = {
    1: (64, 7168, 18432),
    2: (512, 4096, 12288),
    3: (2048, 2880, 2880),
    4: (4096, 4096, 4096),
    5: (8192, 4096, 14336),
    6: (8192, 8192, 29568),
}


def main():
    path = sys.argv[1]
    ablation_path = sys.argv[2] if len(sys.argv) > 2 else None
    doc = json.load(open(path))
    cells = doc["cells"]

    full_us = {}
    if ablation_path:
        try:
            ablation = json.load(open(ablation_path))
            for record in ablation["shapes"]:
                full_us[record["shape_index"]] = record["full_us"]
        except (OSError, ValueError, KeyError) as exc:
            print(f"  no ablation merge ({exc})")

    # The protocol's flag traffic, measured rather than assumed: the emit-local
    # control routes every payload byte to the local rank, so whatever fabric
    # requests remain are publications and credits.
    control = cells.get("s6_emitlocal_g1", {})
    flag_requests = (control.get("derived", {}) or {}).get(
        "fabric_requests_not_dram")

    def group(shape_index, suffix, arm="full"):
        return cells.get(f"s{shape_index}_{arm}_g{suffix}", {}) \
                    .get("counters_per_rank_per_launch", {})

    shapes_out = []
    for index in sorted(SHAPES):
        m, n, k = SHAPES[index]
        g1, g2, g3 = (group(index, s) for s in ("1", "2", "3"))
        if not g1:
            continue
        wr = g1["TCC_EA0_WRREQ"]
        wr64 = g1["TCC_EA0_WRREQ_64B"]
        wrdram = g1["TCC_EA0_WRREQ_DRAM"]
        wr32 = wr - wr64
        fabric_req = wr - wrdram
        lo = 32.0 * min(fabric_req, wr32) + 64.0 * max(0.0, fabric_req - wr32)
        hi = 64.0 * min(fabric_req, wr64) + 32.0 * max(0.0, fabric_req - wr64)
        mid = (lo + hi) / 2.0
        useful = (WORLD - 1) / WORLD * m * n * BYTES_PER_ELEM
        payload_req = useful / EA_LINE

        record = {
            "shape_index": index,
            "shape": f"{m}x{n}x{k}",
            "m": m, "n": n, "k": k,
            "usable": all(cells.get(f"s{index}_full_g{s}", {}).get("usable")
                          for s in ("1", "2", "3")),
            "groups": [s for s in ("1", "2", "3")
                       if f"s{index}_full_g{s}" in cells],
            # --- fabric volume ---
            "fabric_bytes_mb": round(mid / 1e6, 3),
            "fabric_bytes_mb_bounds": [round(lo / 1e6, 3), round(hi / 1e6, 3)],
            "useful_bytes_mb": round(useful / 1e6, 3),
            "amplification": round(mid / useful, 4),
            # --- transaction size ---
            "ea_write_requests": round(wr),
            "ea_write_requests_64b": round(wr64),
            "wrreq_64b_share": round(wr64 / wr, 5),
            "ea_write_requests_dram": round(wrdram),
            "fabric_requests": round(fabric_req),
            "bytes_leaving_l2_mb": round((64.0 * wr64 + 32.0 * wr32) / 1e6, 3),
            "ea_read_requests": round(g1.get("TCC_EA0_RDREQ", float("nan"))),
        }

        # Payload vs protocol flags on the fabric.
        if flag_requests is not None:
            record["fabric_payload_requests_analytic"] = round(payload_req)
            record["fabric_flag_requests_measured"] = round(flag_requests)
            record["flag_share_of_fabric_requests"] = round(
                flag_requests / fabric_req, 5)

        # --- latency: LEVEL/WRREQ, and only from the pass that had both ---
        if "TCC_EA0_WRREQ_LEVEL" in g2 and "TCC_EA0_WRREQ" in g2:
            record["ea_write_latency_cycles"] = round(
                g2["TCC_EA0_WRREQ_LEVEL"] / g2["TCC_EA0_WRREQ"], 1)
            record["ea_write_latency_same_pass"] = True
        if "TCC_CYCLE" in g2:
            record["tcc_cycles"] = round(g2["TCC_CYCLE"])

        # --- L2 write / writeback counts ---
        if "TCC_WRITEBACK" in g2:
            record["l2_writeback_lines"] = round(g2["TCC_WRITEBACK"])
        for name, key in (("TCC_WRITE", "l2_write_requests"),
                          ("TCC_NORMAL_WRITEBACK", "l2_writeback_capacity"),
                          ("TCC_ALL_TC_OP_WB_WRITEBACK", "l2_writeback_wb_op"),
                          ("TCC_NC_REQ", "l2_noncoherent_requests")):
            if name in g3:
                record[key] = round(g3[name])
        total_wb = record.get("l2_writeback_lines")
        cap = record.get("l2_writeback_capacity")
        wbop = record.get("l2_writeback_wb_op")
        if total_wb:
            if cap is not None:
                record["writeback_capacity_share"] = round(cap / total_wb, 4)
            if wbop is not None:
                record["writeback_release_share"] = round(wbop / total_wb, 4)

        # --- achieved fabric bandwidth, using the ablation's own `full` ---
        if index in full_us:
            record["full_us_from_ablation"] = full_us[index]
            record["achieved_fabric_gbps"] = round(
                mid / (full_us[index] * 1e-6) / 1e9, 1)
        shapes_out.append(record)

    doc["what"] = ("one rocprofv3 1.1.0 counter pass at the current best "
                   "config, gfx942, ALL SIX graded shapes, plus two shape-6 "
                   "controls")
    doc["shapes"] = shapes_out
    doc["schema"].update({
        "shapes[]": "per-shape rollup merged from that shape's three counter "
                    "groups; `cells` remains the provenance",
        "shapes[].useful_bytes_mb": "7/8 * M*N*2 -- of the eight output slices "
                    "a rank produces it keeps one locally and sends seven "
                    "off-rank, so 7/8 of the bf16 output crosses the fabric",
        "shapes[].amplification": "fabric_bytes_mb / useful_bytes_mb. 1.0 means "
                    "the fabric carries exactly the payload and nothing else.",
        "shapes[].ea_write_latency_cycles": "TCC_EA0_WRREQ_LEVEL / "
                    "TCC_EA0_WRREQ, that counter's own documented use. A "
                    "QUEUEING signal, not a bandwidth one: it rises when a "
                    "fixed request count is concentrated on fewer links.",
        "shapes[].fabric_flag_requests_measured": "off-die requests that remain "
                    "when every payload byte is routed to the local rank "
                    "(the s6_emitlocal_g1 control), i.e. the protocol's "
                    "publications and credits. Constant in the world size, so "
                    "it is a negligible share on the large shapes and a "
                    "material one on the smallest.",
        "shapes[].writeback_capacity_share": "TCC_NORMAL_WRITEBACK / "
                    "TCC_WRITEBACK: lines that left L2 by ordinary capacity "
                    "eviction. The complement, writeback_release_share, is the "
                    "share pushed by the release's buffer_wbl2 sc0 sc1.",
        "shapes[].achieved_fabric_gbps": "fabric_bytes_mb over the ablation's "
                    "`full` wall time for the same shape and config. An "
                    "average over the whole kernel, not a peak.",
    })
    doc.setdefault("caveats", []).extend([
        "Counter collection instruments and may serialize dispatches, so no "
        "time measured under it is meaningful. Only transaction COUNTS are used.",
        "TCC_EA0_WRREQ_STALL is unusable on this ASIC (it reads ~81 M cycles "
        "while every *_CREDIT_STALL sub-counter reads ~0) and was deliberately "
        "not collected.",
    ])

    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2)

    print(f"wrote {path}: per-shape rollup for {len(shapes_out)} shapes")
    header = (f"{'shape':<18}{'fabricMB':>10}{'usefulMB':>10}{'amp':>8}"
              f"{'64B%':>7}{'EAlat':>7}{'GB/s':>7}{'wbCap%':>8}{'wbRel%':>8}"
              f"{'flag%':>7}  usable")
    print(header)
    for r in shapes_out:
        print(f"{r['shape']:<18}{r['fabric_bytes_mb']:>10.2f}"
              f"{r['useful_bytes_mb']:>10.2f}{r['amplification']:>8.4f}"
              f"{100 * r['wrreq_64b_share']:>7.1f}"
              f"{r.get('ea_write_latency_cycles', float('nan')):>7.0f}"
              f"{r.get('achieved_fabric_gbps', float('nan')):>7.1f}"
              f"{100 * r.get('writeback_capacity_share', float('nan')):>8.1f}"
              f"{100 * r.get('writeback_release_share', float('nan')):>8.1f}"
              f"{100 * r.get('flag_share_of_fabric_requests', float('nan')):>7.2f}"
              f"  {r['usable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
