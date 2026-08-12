"""Aggregate exp_20's rocprofv3 CSVs into counters.json.

Same aggregation as exp_08's prof_agg.py -- filter to the megakernel, keep the
dispatch ordinals [warm, warm+meas) PER AGENT, sum, divide by agents x
dispatches -- but emitting JSON instead of a table, because the deliverable is
plot-ready data.

Two properties this preserves from exp_08 and must not lose:
  * Rows are filtered to `gemm_rs_mi300x_kernel` first. The CSV also holds
    torch's buffer fills, the input generator and the eight `torch.matmul`
    calls of the correctness oracle.
  * The warmup dispatches are dropped per agent, not globally.

Every cell also records whether the kernel was CORRECT and error-free under
instrumentation, taken from the driver's PROF line: counter collection can
serialize dispatches until the bounded credit/ready spins time out, and a
kernel that returned early on the sticky error bit reports deflated traffic.
A cell without `correct=1 errors=none` is marked unusable.

Usage: mk_counters_json.py <prof_dir> <out.json> <warm> <meas>
"""

import collections
import csv
import glob
import json
import os
import re
import sys
import time

KERNEL = "gemm_rs_mi300x_kernel"

# MI300X SPX: 304 CU x 4 SIMD. Used to turn SIMD-summed MFMA busy cycles into
# an occupancy time at the pinned 1900 MHz.
SIMDS = 304 * 4


def aggregate(path, warm, meas):
    rows_all = list(csv.DictReader(open(path)))
    if not rows_all:
        return None, "empty csv"
    keys = rows_all[0].keys()

    def pick(*cands):
        for c in cands:
            if c in keys:
                return c
        return None

    k_agent = pick("Agent_Id", "Agent_Index", "Device_Id", "agent_id")
    k_disp = pick("Dispatch_Id", "Dispatch_Index", "dispatch_id")
    k_name = pick("Counter_Name", "counter_name")
    k_val = pick("Counter_Value", "counter_value")
    k_kern = pick("Kernel_Name", "kernel_name")
    if not all([k_agent, k_disp, k_name, k_val, k_kern]):
        return None, f"unexpected columns: {list(keys)}"

    rows = [r for r in rows_all if KERNEL in r[k_kern]]
    if not rows:
        return None, "no megakernel rows"

    per_agent_disp = collections.defaultdict(set)
    for r in rows:
        per_agent_disp[r[k_agent]].add(int(float(r[k_disp])))
    keep = {a: set(sorted(d)[warm:warm + meas])
            for a, d in per_agent_disp.items()}

    total = collections.defaultdict(float)
    for r in rows:
        if int(float(r[k_disp])) in keep[r[k_agent]]:
            total[r[k_name]] += float(r[k_val])

    agents = sorted(keep)
    ndisp = min(len(keep[a]) for a in agents)
    if not agents or ndisp == 0:
        return None, "nothing kept in the measured window"
    scale = 1.0 / (len(agents) * ndisp)
    return {
        "agents": len(agents),
        "dispatches_per_agent": ndisp,
        "counters_per_rank_per_launch": {n: round(v * scale, 1)
                                         for n, v in sorted(total.items())},
    }, None


def derive(counters):
    """Ratios exp_08 validated. Only computed when their inputs are present."""
    out = {}

    def g(name):
        return counters.get(name)

    wr, wr64, wrdram = g("TCC_EA0_WRREQ"), g("TCC_EA0_WRREQ_64B"), \
        g("TCC_EA0_WRREQ_DRAM")
    if wr and wr64 is not None and wrdram is not None:
        wr32 = wr - wr64
        nondram = wr - wrdram
        out["ea_write_requests"] = round(wr, 1)
        out["frac_64B"] = round(wr64 / wr, 4)
        out["bytes_leaving_l2_MB"] = round((64.0 * wr64 + 32.0 * wr32) / 1e6, 2)
        out["frac_requests_to_dram"] = round(wrdram / wr, 4)
        out["fabric_requests_not_dram"] = round(nondram, 1)
        # The 32/64 split is known only in aggregate, so bound rather than
        # assume a mix.
        lo = 32.0 * min(nondram, wr32) + 64.0 * max(0.0, nondram - wr32)
        hi = 64.0 * min(nondram, wr64) + 32.0 * max(0.0, nondram - wr64)
        out["fabric_bytes_MB_bounds"] = [round(lo / 1e6, 2), round(hi / 1e6, 2)]

    cyc = g("TCC_CYCLE")
    if cyc:
        for name, label in (("TCC_EA0_WRREQ_STALL", "wrreq_stall_frac_of_cycle"),
                            ("TCC_TAG_STALL", "tag_stall_frac_of_cycle")):
            if g(name) is not None:
                out[label] = round(g(name) / cyc, 4)
    if g("TCC_WRITEBACK") is not None:
        out["l2_writebacks"] = round(g("TCC_WRITEBACK"), 1)

    busy, mfma, wait = g("SQ_BUSY_CYCLES"), g("SQ_VALU_MFMA_BUSY_CYCLES"), \
        g("SQ_WAIT_ANY")
    insts = g("SQ_INSTS_MFMA")
    if insts is not None:
        out["mfma_instructions"] = round(insts, 1)
    if mfma is not None:
        # SQ_VALU_MFMA_BUSY_CYCLES is summed over SIMDs, so MFMA occupancy is
        # cycles / (CU x SIMD) and converts to time at the pinned clock. This
        # is used INSTEAD of a ratio against SQ_BUSY_CYCLES, which is not a
        # comparable denominator on this ASIC -- it reads an order of magnitude
        # BELOW MFMA busy, which is impossible under a shared normalization.
        out["mfma_busy_cycles_summed_over_simds"] = round(mfma, 1)
        out["mfma_busy_cycles_per_simd"] = round(mfma / SIMDS, 1)
        out["mfma_busy_us_at_1900MHz"] = round(mfma / SIMDS / 1900.0, 1)
        if insts:
            out["cycles_per_mfma_instruction"] = round(mfma / insts, 2)
    if busy is not None:
        out["sq_busy_cycles_raw"] = round(busy, 1)
        out["sq_busy_note"] = ("raw only: SQ_BUSY_CYCLES reads below "
                               "SQ_VALU_MFMA_BUSY_CYCLES here, so the two do "
                               "not share a normalization and no ratio of "
                               "them is reported")
    if wait is not None:
        out["sq_wait_any_raw"] = round(wait, 1)
    return out


def main():
    prof_dir, out_path = sys.argv[1], sys.argv[2]
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    meas = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    cells, unavailable = {}, []
    for csv_path in sorted(glob.glob(f"{prof_dir}/*/p_counter_collection.csv")):
        tag = os.path.basename(os.path.dirname(csv_path))
        cell, err = aggregate(csv_path, warm, meas)
        log = f"{prof_dir}/{tag}.log"
        prof_line = ""
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                if line.startswith("PROF"):
                    prof_line = line.strip()
        if cell is None:
            cells[tag] = {"error": err, "prof_line": prof_line}
            continue
        cell["prof_line"] = prof_line
        cell["usable"] = bool(re.search(r"correct=1", prof_line)
                              and "errors=none" in prof_line)
        cell["derived"] = derive(cell["counters_per_rank_per_launch"])
        cells[tag] = cell

    # A requested counter that produced no column anywhere did not resolve.
    seen = set()
    for cell in cells.values():
        seen |= set(cell.get("counters_per_rank_per_launch", {}))
    for log_path in sorted(glob.glob(f"{prof_dir}/*.log")):
        text = open(log_path, errors="replace").read()
        for m in re.finditer(r"(?:Invalid|not found|unable to find).{0,80}", text):
            unavailable.append(f"{os.path.basename(log_path)}: {m.group(0)}")

    doc = {
        "experiment": "aug11/exp_20_attribution",
        "what": "one rocprofv3 1.1.0 counter pass at the current best config, "
                "gfx942, shapes 6 and 5",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warm": warm, "meas": meas,
        "schema": {
            "cells[tag].counters_per_rank_per_launch":
                "each counter summed over the measured dispatch window and "
                "divided by agents x dispatches, i.e. one rank, one launch",
            "cells[tag].usable":
                "the driver reported correct=1 and errors=none UNDER "
                "instrumentation. Counter collection can serialize dispatches "
                "until the bounded credit/ready spins time out; a kernel that "
                "returned early on the sticky error bit reports deflated "
                "traffic, so an unusable cell must not be read.",
            "cells[tag].derived.fabric_bytes_MB_bounds":
                "the 32/64 B split is known only in aggregate, so off-die "
                "bytes are bounded rather than assumed",
            "cells[tag].derived.mfma_busy_us_at_1900MHz":
                "SQ_VALU_MFMA_BUSY_CYCLES / (304 CU x 4 SIMD) / 1900 MHz: the "
                "wall time the MFMA pipes are occupied, comparable directly "
                "against ablation.json's gemm pool for the same shape",
            "tags": "s<shape>_<arm>_g<group>; s6_emitlocal_g1 is the "
                    "known-answer control (off-die writes must collapse) and "
                    "s6_prod_g1 is the production-vs-scratch cross-check",
        },
        "counters_known_unusable_on_gfx942": [
            "TCC_EA0_WRREQ_GMI_CREDIT_STALL",
            "TCC_EA0_WRREQ_IO_CREDIT_STALL",
            "TCC_EA0_WRREQ_DRAM_CREDIT_STALL",
        ],
        "counters_seen": sorted(seen),
        "collection_errors": unavailable,
        "cells": cells,
    }
    with open(out_path, "w") as handle:
        json.dump(doc, handle, indent=2)
    print(f"wrote {out_path} with {len(cells)} cells")
    for tag, cell in cells.items():
        if "error" in cell:
            print(f"  {tag:<22} ERROR {cell['error']}")
            continue
        print(f"  {tag:<22} usable={cell['usable']} "
              f"agents={cell['agents']} disp={cell['dispatches_per_agent']} "
              f"{cell['derived']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
