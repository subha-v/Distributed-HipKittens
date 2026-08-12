#!/usr/bin/env python3
"""Score the paired ours/ours_prev campaign against the three pre-registered
outcomes.

Pre-registered BEFORE looking at any number (dispatch, verbatim in intent):

  CONFIRMS  shape 5 faster for `ours`, paired, outside the null -> the win is
            real, the cross-run ladder simply cannot see it, PERSHAPE=2 stays.
  REFUTES   shape 5 slower or flat, paired, with `ours_null` tight -> the win was
            an artifact of exp_26's instrument and PERSHAPE reverts to 0. Never
            to 1, which is the same rule with a worse schedule and is dominated
            by both.
  INDETERMINATE  inside the null -> reported as such, with what would settle it.
            The tie is not broken by preference.

Sign convention throughout: delta = (ours - ours_prev) / ours_prev * 100, so
NEGATIVE means the shipped rule is faster, matching exp_26's "-6.56%".

Shapes 1-4 and 6 are extra controls, not filler: `tiles_per_cta` is 1, 1, 1, 1
and 4 there, and the two arms' rules agree at those values, so the arms are the
same computation and any delta is instrument bias -- five more null arms.
"""
import glob
import json
import math
import os
import statistics
import sys

D = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(D, "raw", "ab_prev")
SHAPE5 = 4  # index of 8192x4096x14336, the only row whose rgroup changes


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.rank*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("error"):
            print(f"  ERROR in {os.path.basename(path)}:")
            print("    " + d["error"].strip().splitlines()[-1])
            continue
        base = os.path.basename(path).split(".rank")[0]
        runs.setdefault(base, []).append(d)
    return runs


def paired(rows, proto, a="ours", b="ours_prev"):
    """Pool the per-round deltas across ranks. Each rank timed the same rounds in
    the same order, so a round is comparable within a rank; ranks are pooled
    because all eight are measuring the same collective operation."""
    deltas, wins, total = [], 0, 0
    for d in rows:
        pa = d["per_round"][a][proto]
        pb = d["per_round"][b][proto]
        for x, y in zip(pa, pb):
            deltas.append((x - y) / y * 100.0)
            total += 1
            if x < y:
                wins += 1
    return deltas, wins, total


def sign_test_p(wins, total):
    """Two-sided exact binomial p under H0: p(win) = 0.5. No scipy on this box."""
    if total == 0:
        return None
    k = min(wins, total - wins)
    tail = sum(math.comb(total, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2.0 ** total))


def summarize(runs):
    out = {}
    print("################ integrity ################")
    for base in sorted(runs):
        rows = runs[base]
        r0 = rows[0]
        eq = all(r.get("all_ranks_bit_identical") for r in rows)
        print(f"  {base}: {len(rows)}/8 ranks  shape={r0['shape_label']}  "
              f"alloc={r0['alloc_order']}  rounds={r0['rounds']}")
        print(f"      geometry tiles={r0['geometry']['tiles']} "
              f"producers={r0['geometry']['producers']} "
              f"tiles/CTA={r0['geometry']['tiles_per_cta']}   "
              f"rgroup={r0['rgroup']}  distinct={r0['arms_distinct_on_this_shape']}")
        print(f"      torch.equal all ranks: {eq}   "
              f"blocks of {r0['perms_per_block']} permutations   "
              f"modules={r0['kernel_modules']}")
        if len(rows) != 8:
            print(f"      WARNING only {len(rows)} ranks present")
    return out


def main():
    runs = load()
    if not runs:
        print("no samples found")
        return 1
    summarize(runs)

    print("\n################ paired deltas: (ours - ours_prev) / ours_prev ################")
    print("negative == the shipped PERSHAPE=2 rule is FASTER (exp_26's sign)")
    print(f"{'run':>26} {'proto':>10} {'median%':>9} {'mean%':>8} "
          f"{'wins':>11} {'p':>10} {'null%':>8} {'verdict':>16}")
    table = {}
    for base in sorted(runs):
        rows = runs[base]
        for proto in ("pipelined", "graded"):
            dl, wins, total = paired(rows, proto)
            nl, nwins, ntotal = paired(rows, proto, "ours", "ours_null")
            if not dl:
                continue
            med = statistics.median(dl)
            nmed = statistics.median(nl)
            # The floor is the null contrast's own magnitude on this shape and
            # protocol: a treatment delta smaller than the bias between two
            # identical binaries is not a measurement of anything.
            resolved = abs(med) > abs(nmed)
            if not resolved:
                v = "inside null"
            elif med < 0:
                v = "ours FASTER"
            else:
                v = "ours SLOWER"
            p = sign_test_p(wins, total)
            print(f"{base:>26} {proto:>10} {med:>+9.2f} "
                  f"{statistics.mean(dl):>+8.2f} {wins:>5}/{total:<5} "
                  f"{p:>10.2e} {nmed:>+8.2f} {v:>16}")
            table[(base, proto)] = {
                "median_pct": med, "mean_pct": statistics.mean(dl),
                "wins": wins, "rounds": total, "sign_test_p": p,
                "null_median_pct": nmed, "resolved_vs_null": resolved,
                "verdict": v,
                "shape_label": rows[0]["shape_label"],
                "shape_index": rows[0]["shape_index"],
                "alloc_order": rows[0]["alloc_order"],
                "rgroup": rows[0]["rgroup"],
                "arms_distinct": rows[0]["arms_distinct_on_this_shape"],
            }

    print("\n################ the corrected null floors, both protocols ################")
    print("ours vs ours_null, per-round paired median magnitude. These supersede")
    print("the floors quoted all night, which were inflated by the cyclic-order")
    print("defect (a full rotation still pins every arm PAIR at a fixed offset).")
    print(f"{'shape':>18} {'alloc':>6} {'graded%':>9} {'pipelined%':>11}")
    floors = {}
    for base in sorted(runs):
        rows = runs[base]
        lbl = rows[0]["shape_label"]
        alloc = "fwd" if rows[0]["alloc_order"][0] == "ours" else "rev"
        g, _, _ = paired(rows, "graded", "ours", "ours_null")
        p, _, _ = paired(rows, "pipelined", "ours", "ours_null")
        gm = abs(statistics.median(g)) if g else None
        pm = abs(statistics.median(p)) if p else None
        print(f"{lbl:>18} {alloc:>6} "
              f"{(f'{gm:9.2f}' if gm is not None else '        -')} "
              f"{(f'{pm:11.2f}' if pm is not None else '          -')}")
        floors[f"{lbl}_{alloc}"] = {"graded_pct": gm, "pipelined_pct": pm}

    print("\n################ THE VERDICT ON SHAPE 5 ################")
    s5 = {k: v for k, v in table.items() if v["shape_index"] == SHAPE5}
    if not s5:
        print("  shape 5 did not run -- no verdict")
        outcome = "NOT RUN"
    else:
        for (base, proto), v in sorted(s5.items(), key=lambda x: x[0][1]):
            print(f"  {proto:>10} {v['alloc_order'][0]:>9}-first: "
                  f"median {v['median_pct']:+.2f}%  "
                  f"wins {v['wins']}/{v['rounds']}  "
                  f"p={v['sign_test_p']:.2e}  null {v['null_median_pct']:+.2f}%  "
                  f"-> {v['verdict']}")
        pipe = [v for (b, p), v in s5.items() if p == "pipelined"]
        # Primary protocol is pipelined: that is where the disagreement is loudest
        # (+2.87% cross-run vs exp_26's -6.56%) and where exp_26 measured.
        faster = [v for v in pipe if v["verdict"] == "ours FASTER"]
        slower = [v for v in pipe if v["verdict"] == "ours SLOWER"]
        inside = [v for v in pipe if v["verdict"] == "inside null"]
        if faster and not slower:
            outcome = "CONFIRMS exp_26"
        elif slower and not faster:
            outcome = "REFUTES exp_26"
        elif inside and not faster and not slower:
            outcome = "INDETERMINATE"
        else:
            outcome = ("INDETERMINATE (allocation orders disagree: "
                       f"{[v['verdict'] for v in pipe]})")
        print(f"\n  PRE-REGISTERED OUTCOME: {outcome}")
        if outcome.startswith("CONFIRMS"):
            print("  -> PERSHAPE=2 stays.")
        elif outcome.startswith("REFUTES"):
            print("  -> PERSHAPE reverts to 0. Not to 1: same rule, worse")
            print("     schedule, dominated by both.")
        else:
            print("  -> no change on this evidence; report as unresolved.")

    print("\n################ controls: the five shapes where the arms agree ################")
    print("rgroup identical there, so the arms are the same computation and any")
    print("delta is this instrument's bias, measured five more times.")
    for (base, proto), v in sorted(table.items()):
        if v["shape_index"] == SHAPE5 or proto != "pipelined":
            continue
        flag = "" if abs(v["median_pct"]) <= max(abs(v["null_median_pct"]), 1.0) \
            else "  <-- BIAS above the null on identical code"
        print(f"  {v['shape_label']:>18} {proto:>10} "
              f"median {v['median_pct']:+7.2f}%  null {v['null_median_pct']:+7.2f}%"
              f"{flag}")

    # ---- order-balanced estimate ------------------------------------------
    # The two allocation orders disagree in SIGN and each is individually
    # hyper-significant, which is the signature of a systematic artifact rather
    # than noise: perfectly consistent within a configuration, reversing between
    # them. Averaging the two orders cancels a position effect to first order.
    # Its residual is not assumed -- it is MEASURED on the five shapes where the
    # two arms compile to the same rgroup and the truth is therefore exactly 0.
    print("\n################ order-balanced estimate, and its measured residual ################")
    print("mean of the two allocation orders. On the five shapes where the arms")
    print("are the same computation the true value is 0, so their spread IS the")
    print("residual bias of this balancing.")
    print(f"{'#':>2} {'shape':>18} {'proto':>10} {'fwd%':>8} {'rev%':>8} "
          f"{'balanced%':>10} {'arms distinct':>14}")
    balanced = {}
    by_shape = {}
    for (base, proto), v in table.items():
        by_shape.setdefault((v["shape_index"], proto), {})[base[-3:]] = v
    for (si, proto) in sorted(by_shape):
        got = by_shape[(si, proto)]
        if "fwd" not in got or "rev" not in got:
            continue
        f, r = got["fwd"]["median_pct"], got["rev"]["median_pct"]
        bal = (f + r) / 2.0
        distinct = got["fwd"]["arms_distinct"]
        balanced[(si, proto)] = {"fwd_pct": f, "rev_pct": r,
                                 "balanced_pct": bal, "arms_distinct": distinct}
        print(f"{si+1:>2} {got['fwd']['shape_label']:>18} {proto:>10} "
              f"{f:>+8.2f} {r:>+8.2f} {bal:>+10.2f} {str(distinct):>14}")
    for proto in ("pipelined", "graded"):
        ctrl = [v["balanced_pct"] for (si, p), v in balanced.items()
                if p == proto and not v["arms_distinct"]]
        tgt = [v["balanced_pct"] for (si, p), v in balanced.items()
               if p == proto and v["arms_distinct"]]
        if ctrl and tgt:
            resid = max(abs(c) for c in ctrl)
            print(f"\n  {proto}: control residual (5 shapes, truth = 0) "
                  f"mean {sum(ctrl)/len(ctrl):+.2f}%, worst |{resid:.2f}|%")
            print(f"  {proto}: shape 5 order-balanced {tgt[0]:+.2f}%  "
                  f"-> {'ABOVE' if abs(tgt[0]) > resid else 'within'} the worst "
                  f"control residual")
            print(f"  {proto}: exp_26 claimed -6.56%; order-balanced estimate is "
                  f"{tgt[0]:+.2f}%")

    with open(os.path.join(D, "ab_prev.json"), "w") as fh:
        json.dump({"outcome": outcome,
                   "sign_convention": "(ours - ours_prev)/ours_prev*100; "
                                      "negative == PERSHAPE=2 faster",
                   "primary_protocol": "pipelined",
                   "paired": [{"run": k[0], "proto": k[1], **v}
                              for k, v in table.items()],
                   "order_balanced": [{"shape_index": k[0], "proto": k[1], **v}
                                      for k, v in balanced.items()],
                   "corrected_null_floors": floors}, fh, indent=2)
    print(f"\nwrote {os.path.join(D, 'ab_prev.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
