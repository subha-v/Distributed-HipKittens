"""Aggregate one rocprofv3 counter_collection CSV into egress figures.

Sums each counter over ALL rows of the measured dispatch window, per agent,
because rocprofv3 may emit one row per hardware instance (16 TCC x 8 XCC on
gfx942) or one pre-reduced row -- summing handles both, and the per-agent split
is what lets a single rank's egress be compared with the analytic payload.

Definitions used, verbatim from `rocprofv3 --list-avail` on this node:
  TCC_EA0_WRREQ      "Number of transactions (either 32-byte or 64-byte) going
                      over the TC_EA_wrreq interface. Atomics may travel over
                      the same interface and are generally classified as write
                      requests. This does not include probe commands."
  TCC_EA0_WRREQ_64B  "Number of 64-byte transactions going (64-byte write or
                      CMPSWAP) over the TC_EA_wrreq interface."
  TCC_EA0_WRREQ_DRAM "Number of TCC/EA write requests (either 32-byte of
                      64-byte) destined for DRAM (MC)."
So bytes leaving L2 = 64*WRREQ_64B + 32*(WRREQ - WRREQ_64B), and the requests
NOT destined for DRAM are the ones the fabric carries off-die.

The CSV also contains every OTHER kernel the process ran -- torch's buffer
fills, the input generator, and the eight `torch.matmul` calls of the
correctness oracle -- so rows are filtered to the megakernel by name before
anything is summed.

Usage: prof_agg.py <csv> <warm> <meas> [label] [kernel_substr]
"""

import collections
import csv
import sys


def main():
    path = sys.argv[1]
    warm = int(sys.argv[2])
    meas = int(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else path
    want_kernel = sys.argv[5] if len(sys.argv) > 5 else "gemm_rs_mi300x_kernel"

    allrows = list(csv.DictReader(open(path)))
    if not allrows:
        print(f"{label}: EMPTY CSV")
        return 1
    keys = allrows[0].keys()

    def pick(*cands):
        for c in cands:
            if c in keys:
                return c
        raise SystemExit(f"no column among {cands}; have {list(keys)}")

    k_agent = pick("Agent_Id", "Agent_Index", "Device_Id", "agent_id")
    k_disp = pick("Dispatch_Id", "Dispatch_Index", "dispatch_id")
    k_name = pick("Counter_Name", "counter_name")
    k_val = pick("Counter_Value", "counter_value")
    k_kern = pick("Kernel_Name", "kernel_name")
    k_beg = "Start_Timestamp" if "Start_Timestamp" in keys else None
    k_end = "End_Timestamp" if "End_Timestamp" in keys else None

    rows = [r for r in allrows if want_kernel in r[k_kern]]
    if not rows:
        print(f"{label}: no rows match kernel '{want_kernel}'; saw "
              f"{sorted({r[k_kern].split('(')[0][:40] for r in allrows})}")
        return 1

    # Per agent, the dispatch ordinals in launch order. Keep only [warm, warm+meas).
    per_agent_disp = collections.defaultdict(set)
    for r in rows:
        per_agent_disp[r[k_agent]].add(int(float(r[k_disp])))
    keep = {}
    for agent, disps in per_agent_disp.items():
        ordered = sorted(disps)
        keep[agent] = set(ordered[warm:warm + meas])

    tot = collections.defaultdict(float)          # counter -> sum over agents
    per_agent = collections.defaultdict(lambda: collections.defaultdict(float))
    kernels = collections.Counter()
    ndisp = collections.defaultdict(int)
    durs = []
    for r in rows:
        a = r[k_agent]
        d = int(float(r[k_disp]))
        if d not in keep[a]:
            continue
        v = float(r[k_val])
        tot[r[k_name]] += v
        per_agent[a][r[k_name]] += v
        kernels[r[k_kern].split("(")[0][:52]] += 1
        if k_beg and k_end:
            durs.append((float(r[k_end]) - float(r[k_beg])) / 1e3)
    for a in keep:
        ndisp[a] = len(keep[a])

    agents = sorted(per_agent)
    nag = len(agents)
    ndis = min(ndisp[a] for a in agents) if agents else 0
    print(f"=== {label}")
    print(f"    agents={nag} kept_dispatches_per_agent={ndis} "
          f"kernels={sorted(kernels)}")
    if durs:
        durs.sort()
        print(f"    profiled dispatch us: min={min(durs):.1f} "
              f"med={durs[len(durs) // 2]:.1f} max={max(durs):.1f} "
              f"(PROFILER-PERTURBED, not a timing result)")
    if nag == 0 or ndis == 0:
        print("    NOTHING KEPT -- check warm/meas window")
        return 1

    # Per-rank per-launch means.
    scale = 1.0 / (nag * ndis)
    print(f"    --- per-rank per-launch counter means "
          f"(divided by {nag} agents x {ndis} dispatches) ---")
    for name in sorted(tot):
        print(f"    {name:<34} {tot[name] * scale:>18,.1f}")

    def g(name):
        return tot.get(name, float("nan")) * scale

    wr = g("TCC_EA0_WRREQ")
    wr64 = g("TCC_EA0_WRREQ_64B")
    wrdram = g("TCC_EA0_WRREQ_DRAM")
    if wr == wr:  # not NaN
        wr32 = wr - wr64
        bytes_ea = 64.0 * wr64 + 32.0 * wr32
        nondram = wr - wrdram
        print("    --- derived ---")
        print(f"    EA write requests / rank / launch   {wr:>18,.0f}")
        print(f"      of which 64 B                     {wr64:>18,.0f}"
              f"   ({100.0 * wr64 / wr:.1f}%)")
        print(f"      of which 32 B                     {wr32:>18,.0f}"
              f"   ({100.0 * wr32 / wr:.1f}%)")
        print(f"    bytes leaving L2 via EA (MB)        {bytes_ea / 1e6:>18,.2f}")
        print(f"    EA wr requests destined DRAM        {wrdram:>18,.0f}"
              f"   ({100.0 * wrdram / wr:.1f}%)")
        print(f"    EA wr requests NOT DRAM (fabric)    {nondram:>18,.0f}"
              f"   ({100.0 * nondram / wr:.1f}%)")
        # Upper and lower bounds on fabric bytes: we know the 32/64 split only
        # in aggregate, so bound it rather than assume a mix.
        lo = 32.0 * min(nondram, wr32) + 64.0 * max(0.0, nondram - wr32)
        hi = 64.0 * min(nondram, wr64) + 32.0 * max(0.0, nondram - wr64)
        print(f"    fabric bytes (MB), bounds           "
              f"{lo / 1e6:>10,.2f} .. {hi / 1e6:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
