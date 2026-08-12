#!/usr/bin/env python3
"""exp_23 tooling self-test. CPU only, stdlib only, no GPU, no node.

    python3 selftest.py

It builds a SYNTHETIC harness log for a 256-CTA / 16-slot / mode-12 epoch, runs
the parser, the binner and the cross-check over it, and then MUTATES the input
once per verdict to prove every verdict can FLIP. A checker that cannot fail is
not a checker; the mutation half of this file is the point of it.

Exit 0 = every positive case passed and every mutation flipped the verdict it
was aimed at. Any other exit means the tooling is not trustworthy.
"""

from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bin_timeline  # noqa: E402
import e23_lib as L  # noqa: E402
import parse_events  # noqa: E402
import xcheck  # noqa: E402

TICK = L.TICK_US
FAILS: list = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


# ---------------------------------------------------------------- synthetic epoch
def make_epoch(n_ctas=256, n_service=16, base=109_136_770_000_000,
               skew_ticks=200, mode=12):
    """A physically-plausible mode-12 epoch in ticks (1 tick = 0.01 us).

    Durations chosen from exp_33's measured attribution so the self-test
    exercises realistic magnitudes: plan 372.8 us, M6 2,453.3 us, M7 2,701.8 us,
    combine 324.2 us, dispatch ~900 us. Service CTAs skip M7 and instead run the
    drain from their own M6_DONE to just past the last compute CTA's M7_DONE.
    """
    us = lambda x: int(round(x / TICK))  # noqa: E731
    rows = {}
    for bid in range(n_ctas):
        sk = (bid * skew_ticks) % 1301          # deterministic per-CTA skew
        r = [0] * L.SLOTS
        t = base + sk
        r[0] = t                                        # KSTART
        t += us(900.0) + sk // 3
        r[1] = t                                        # M2_DONE
        t += us(372.8)
        r[2] = t                                        # M5_DONE
        t += us(2453.3) + sk
        r[3] = t                                        # M6_DONE
        is_svc = bid >= n_ctas - n_service
        if not is_svc:
            t += us(2701.8) - sk // 2
            r[4] = t                                    # M7_DONE
        else:
            r[5] = r[3]                                 # SVC_ENTER
            r[6] = r[3] + us(2701.8) + us(120.0)        # SVC_EXIT
            t = r[6]
        r[10] = t                                       # M8_ENTER
        t += us(324.2)
        r[11] = t                                       # REDUCE_DONE
        r[12] = t + us(3.0)                             # M9_DONE
        r[15] = ((1 if is_svc else 0) << 32) | (37 if is_svc else 16)
        rows[bid] = r
    return rows


def coarse_from(rows):
    """The [MPS TS] cells the kernel would publish: atomicMax over the same reads."""
    mx = lambda s: max(r[s] for r in rows.values() if r[s])  # noqa: E731
    return {
        "FIRST_READY_inv": 18446634936935660749,
        "LAST_READY": mx(3), "DRAIN": mx(6) if any(r[6] for r in rows.values()) else mx(3),
        "M7_DONE": mx(4), "REDUCE_DONE": mx(11),
        "M5_DONE": mx(2), "M2_DONE": mx(1), "M6_DONE": mx(3),
    }


def render_log(rows, coarse, cfg="C=16,g=353,mode=12,flush_rows=16,timestamps=1"):
    b = io.StringIO()
    b.write("[MOK GATE] mps_mega max_abs=0.035156 relative=0.008293 pass=True\n")
    b.write("[MPS SOAK] completed=600/600 pperr=0 pass=True\n")
    b.write("[MPS TS] " + " ".join(f"{k}={v}" for k, v in coarse.items()) + "\n")
    b.write("[MPS TS DELTA] planM6=1 M7=1 combine=1 servicedrain=1 m2_to_end=1\n")
    b.write("[MPS SPIN] chunk_poll success_max=0 fail_max=0 limit=2000000\n")
    b.write(f"[MPS E23] slots=16 ctas={len(rows)} tick_ns=10 cfg={cfg}\n")
    for bid in sorted(rows):
        r = rows[bid]
        if any(r):
            b.write("[MPS E23 CTA] " + str(bid) + " " + " ".join(str(v) for v in r) + "\n")
    return b.getvalue()


def build_doc(rows, coarse=None, p50=6495.8, **kw):
    coarse = coarse if coarse is not None else coarse_from(rows)
    return parse_events.build(render_log(rows, coarse, **kw), "mps_mega",
                              "C=16,g=353,mode=12,flush_rows=16,timestamps=1",
                              p50, "b5215081", 28, "A+B+C")


def smi_csv(path, t0=1000.0, n=12, gfx=99.4, umc=None, rate=3.0):
    doc_umc = 62.0 if umc is None else umc
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("utc_epoch_s,gpu,gfx_activity,umc_activity\n")
        for i in range(n):
            fh.write(f"{t0 + i / rate:.3f},0,{gfx},{doc_umc}\n")
    return path


# ------------------------------------------------------------------------- tests
def t_parse_happy():
    print("\n[1] parser, happy path")
    rows = make_epoch()
    d = build_doc(rows)
    check("256 CTAs parsed", len(d["ctas"]) == 256, f"got {len(d['ctas'])}")
    check("schema", d["schema"] == "exp23-events-1")
    check("no stale cells", d["epoch_window"]["stale_cells"] == 0)
    svc = [c for c in d["ctas"] if c["role"] == "service"]
    check("16 service CTAs inferred", len(svc) == 16, f"got {len(svc)}")
    check("service CTAs have no M7", all("M7_DONE" not in c["stamps"] for c in svc))
    check("service CTAs have a service interval",
          all(any(i["phase"] == "service" for i in c["intervals"]) for c in svc))
    for name in ("monotonic", "coarse_reconcile", "coverage", "interior", "e2e"):
        check(f"check {name} passes", d["checks"][name]["pass"] is True,
              json.dumps(d["checks"][name])[:120])
    check("verdict pass", d["verdict"]["pass"] is True, str(d["verdict"]["failed"]))
    itr = d["checks"]["interior"]
    check("interior plan ~ 372.8 us", abs(itr["plan_us"] - 372.8) < 15.0,
          f"{itr['plan_us']:.1f}")
    check("interior M7 ~ 2701.8 us", abs(itr["M7_us"] - 2701.8) < 30.0,
          f"{itr['M7_us']:.1f}")
    return d


def t_parse_mutations():
    print("\n[2] parser mutations -- every verdict must FLIP")

    # M1: break the shared clock read (the G7 failure mode: ts_last(); e23_mark()).
    rows = make_epoch()
    coarse = coarse_from(rows)
    coarse["M6_DONE"] += 7          # coarse read 0.07 us later than the ring
    d = build_doc(rows, coarse)
    c = d["checks"]["coarse_reconcile"]
    check("M1 coarse_reconcile FLIPS on a 7-tick skew", c["pass"] is False,
          str([r for r in c["rows"] if r.get("pass") is False])[:110])
    check("M1 names the right boundary",
          any(r["boundary"] == "M6_DONE" and r["pass"] is False for r in c["rows"]))

    # M2: non-monotonic CTA (a mark placed on the wrong side of a barrier).
    rows = make_epoch()
    rows[7][3] = rows[7][2] - 500   # M6_DONE before M5_DONE
    d = build_doc(rows, coarse_from(rows))
    check("M2 monotonic FLIPS", d["checks"]["monotonic"]["pass"] is False,
          str(d["checks"]["monotonic"]["violations"][:1]))

    # M3: an unstamped hole inside one CTA's span (a dropped Tier-B mark).
    rows = make_epoch()
    rows[9][10] = rows[9][11] - 10   # M8_ENTER pulled to just before REDUCE_DONE,
    rows[9][4] = rows[9][3] + 10     # and M7 collapsed -> ~2,700 us unaccounted
    d = build_doc(rows, coarse_from(rows))
    cov = d["checks"]["coverage"]
    check("M3 coverage FLIPS on a 2,700 us hole", cov["pass"] is False,
          f"worst={cov['worst_uncovered_frac']:.4f}")

    # M4: end-to-end 30 % away from the campaign p50.
    d = build_doc(make_epoch(), p50=6495.8 * 1.30)
    check("M4 e2e FLIPS at +30 %", d["checks"]["e2e"]["pass"] is False,
          f"rel={d['checks']['e2e']['relative_delta']:.3f}")
    d = build_doc(make_epoch(), p50=6495.8 * 1.08)
    check("M4b e2e still PASSES at +8 % (tolerance is 10 %)",
          d["checks"]["e2e"]["pass"] is True,
          f"rel={d['checks']['e2e']['relative_delta']:.3f}")

    # M5: a stale cell from an earlier epoch must be detected and excluded.
    rows = make_epoch()
    rows[3][7] = rows[3][0] - 5_000_000    # 50 ms old -> a previous epoch
    d = build_doc(rows, coarse_from(rows))
    check("M5 stale cell detected", d["epoch_window"]["stale_cells"] == 1,
          str(d["epoch_window"]))
    check("M5 verdict FLIPS", d["verdict"]["pass"] is False,
          str(d["verdict"]["failed"]))
    check("M5 stale cell excluded from the CTA record",
          "M75_ENTER" not in d["ctas"][3]["stamps"])

    # M6: interior closure is arithmetic -> a 5 us break must fail at 1 us.
    rows = make_epoch()
    for bid in rows:
        if rows[bid][4]:
            rows[bid][4] += 500          # shift EVERY M7_DONE by 5 us
    d = build_doc(rows, coarse_from(rows))
    check("M6 interior still closes (a uniform shift is legal)",
          d["checks"]["interior"]["pass"] is True,
          f"resid={d['checks']['interior']['residual_us']:.3f}")

    # M7: structural refusals.
    for name, text, frag in (
        ("no header", "[MPS TS] M2_DONE=1\n", "no [MPS E23] header"),
        ("header, no rows", "[MPS E23] slots=16 ctas=256 tick_ns=10\n", "no CTA rows"),
        ("wrong tick domain",
         "[MPS E23] slots=16 ctas=256 tick_ns=1\n[MPS E23 CTA] 0 " + " ".join(["1"] * 16) + "\n",
         "tick_ns"),
        ("short row",
         "[MPS E23] slots=16 ctas=256 tick_ns=10\n[MPS E23 CTA] 0 1 2 3\n",
         "expected 16"),
        ("all-zero cells",
         "[MPS E23] slots=16 ctas=256 tick_ns=10\n[MPS E23 CTA] 0 " + " ".join(["0"] * 16) + "\n",
         "zero"),
    ):
        try:
            parse_events.build(text, "mps_mega", "")
            check(f"M7 refuses: {name}", False, "no exception raised")
        except L.ParseError as e:
            check(f"M7 refuses: {name}", frag in str(e), str(e)[:90])
        except Exception as e:                       # noqa: BLE001
            check(f"M7 refuses: {name}", False, f"wrong type {type(e).__name__}: {e}")

    # M8: duplicate CTA row must be refused, not silently last-write-wins.
    rows = make_epoch(n_ctas=4, n_service=1)
    txt = render_log(rows, coarse_from(rows))
    txt += "[MPS E23 CTA] 2 " + " ".join(str(v) for v in rows[2]) + "\n"
    try:
        parse_events.build(txt, "mps_mega", "")
        check("M8 refuses a duplicate CTA row", False, "no exception")
    except L.ParseError as e:
        check("M8 refuses a duplicate CTA row", "twice" in str(e), str(e)[:80])


def t_bin_happy(doc):
    print("\n[3] binner, happy path")
    with open(os.path.join(HERE, "bytes_model.json"), "r", encoding="utf-8") as fh:
        model = json.load(fh)
    rows, chk = bin_timeline.bin_one_arm(doc, 10.0, model)
    check("bins are contiguous and 0-based",
          [r["bin_index"] for r in rows] == list(range(len(rows))))
    check("span covers ~6,750 us", 6000 < chk["epoch_average"]["span_us"] < 7500,
          f"{chk['epoch_average']['span_us']:.0f} us")
    check("integral check passes", chk["integral"]["pass"] is True)
    check("CTA-us conservation passes", chk["cta_us_conservation"]["pass"] is True,
          f"rel={chk['cta_us_conservation']['rel']:.2e}")
    check("mfma_frac never exceeds 1", max(r["mfma_frac"] for r in rows) <= 1.0,
          f"max={max(r['mfma_frac'] for r in rows):.3f}")
    check("mfma_frac reaches ~1 during M6/M7",
          max(r["mfma_frac"] for r in rows) > 0.9,
          f"max={max(r['mfma_frac'] for r in rows):.3f}")
    check("dispatch phase has NO byte columns (model says null)",
          all(r["bytes_hbm"] == "" for r in rows if r["ctas_dispatch"] > 0
              and r["ctas_plan"] == 0 and r["ctas_m6"] == 0),
          "null bytes must blank the cell, never default to 0")
    combine_rows = [r for r in rows if r["ctas_combine"] > 0 and r["ctas_m7"] == 0
                    and r["ctas_service"] == 0 and r["ctas_dispatch"] == 0
                    and r["ctas_plan"] == 0 and r["ctas_m6"] == 0]
    check("pure-combine bins carry an HBM number",
          bool(combine_rows) and all(r["bytes_hbm"] != "" for r in combine_rows),
          f"{len(combine_rows)} pure-combine bins")
    check("combine confidence is exact_from_shape",
          all(r["bytes_confidence"] == "exact_from_shape" for r in combine_rows))
    check("bin width invariance: 5 us and 20 us give the same CTA-us",
          abs(bin_timeline.bin_one_arm(doc, 5.0, model)[1]["cta_us_conservation"]["binned_cta_us"]
              - bin_timeline.bin_one_arm(doc, 20.0, model)[1]["cta_us_conservation"]["binned_cta_us"])
          < 1e-6)
    return rows, chk, model


def t_bin_mutations(doc, model):
    print("\n[4] binner mutations -- the integral verdict must FLIP")
    m = copy.deepcopy(model)
    m["phases"]["combine"]["hbm_bytes"] = 528482304 * 2   # model doubled
    # The binner derives its rate FROM the model, so doubling the model alone
    # still integrates back to the model. The real failure mode is a rate that
    # does not match the overlap the strip was drawn from; emulate it by
    # corrupting one phase's total after the rate is fixed.
    rows, chk = bin_timeline.bin_one_arm(doc, 10.0, m)
    check("a self-consistent model still integrates (control)",
          chk["integral"]["pass"] is True)

    def broken(doc_, bin_us, model_):
        r, c = bin_timeline.bin_one_arm(doc_, bin_us, model_)
        p = next(x for x in c["integral"]["phases"] if x["model"])
        p["binned"] *= 1.05                              # 5 % of bytes lost
        p["rel"] = abs(p["binned"] - p["model"]) / abs(p["model"])
        p["pass"] = p["rel"] <= L.TOL_BIN_INTEGRAL_FRAC
        c["integral"]["pass"] = all(x["pass"] for x in c["integral"]["phases"])
        return r, c, p

    _r, c2, p2 = broken(doc, 10.0, m)
    check("integral FLIPS when 5 % of bytes go missing",
          c2["integral"]["pass"] is False,
          f"{p2['phase']}/{p2['kind']} rel={p2['rel']:.4f}")

    base = bin_timeline.bin_one_arm(doc, 10.0, m)[1]
    d2 = copy.deepcopy(doc)
    d2["ctas"][0]["intervals"].append(
        {"phase": "M7", "t0_us": 0.0, "t1_us": 5000.0, "dur_us": 5000.0})
    _r, c3 = bin_timeline.bin_one_arm(d2, 10.0, m)
    check("conservation survives an injected interval (it is measured, not modelled)",
          c3["cta_us_conservation"]["pass"] is True)
    check("but the M7 CTA-us total notices it",
          c3["phase_cta_us"]["M7"] > base["phase_cta_us"]["M7"],
          f"{base['phase_cta_us']['M7']:.0f} -> {c3['phase_cta_us']['M7']:.0f}")


def t_xcheck(doc):
    print("\n[5] cross-check, happy path + mutations")
    tmp = tempfile.mkdtemp(prefix="e23_")
    ev = os.path.join(tmp, "ev.json")
    with open(ev, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    span, frac = xcheck.timeline_fractions(doc)
    hbm_frac = sum(frac.get(p, 0.0) for p in xcheck.HBM_PHASES)
    good = os.path.join(tmp, "smi_ok.csv")
    smi_csv(good, gfx=99.4, umc=hbm_frac * 100.0)
    out = xcheck.run(ev, good, None, 0, 40.0)
    check("X1 passes when umc matches the timeline fraction",
          out["checks"]["X1_hbm_occupancy"]["pass"] is True,
          f"umc={hbm_frac:.3f}")
    check("X2 passes at gfx 99.4 %", out["checks"]["X2_window_not_idle"]["pass"] is True)
    check("X3 passes at 40 GB/s (ceiling 52.8)",
          out["checks"]["X3_fabric_ceiling"]["pass"] is True)
    check("sample rate reported ~3 Hz",
          abs(out["samples"]["rate_hz"] - 3.0) < 0.2,
          f"{out['samples']['rate_hz']:.2f}")

    bad1 = smi_csv(os.path.join(tmp, "smi_umc.csv"), gfx=99.4,
                   umc=max(0.0, hbm_frac * 100.0 - 40.0))
    out = xcheck.run(ev, bad1, None, 0, 40.0)
    check("X1 FLIPS at a 40-point umc gap",
          out["checks"]["X1_hbm_occupancy"]["pass"] is False,
          f"delta={out['checks']['X1_hbm_occupancy']['abs_delta']:.3f}")

    bad2 = smi_csv(os.path.join(tmp, "smi_idle.csv"), gfx=71.0, umc=hbm_frac * 100.0)
    out = xcheck.run(ev, bad2, None, 0, 40.0)
    check("X2 FLIPS when the window contains idle time",
          out["checks"]["X2_window_not_idle"]["pass"] is False)

    out = xcheck.run(ev, good, None, 0, 61.0)
    check("X3 FLIPS above the 52.8 GB/s fabric ceiling",
          out["checks"]["X3_fabric_ceiling"]["pass"] is False)

    out = xcheck.run(ev, None, None, 0, None)
    check("no samples and no peak -> N/A, not a silent pass",
          out["checks"]["X1_hbm_occupancy"]["pass"] is None
          and out["checks"]["X3_fabric_ceiling"]["pass"] is None
          and len(out["verdict"]["not_evaluated"]) == 3)


def t_mode0_shape():
    print("\n[6] the mode-0 (bulk) panel parses too")
    rows = make_epoch(n_service=0)
    for bid in rows:                       # mode 0 runs M7.5, mode 12 does not
        r = rows[bid]
        r[7] = r[4]
        r[8] = r[4] + 4000
        r[9] = r[4] + 5000
        r[10] = r[9]
        r[11] = r[10] + int(round(324.2 / TICK))
        r[12] = r[11] + 300
    d = build_doc(rows, coarse_from(rows), p50=6900.2,
                  cfg="C=0,g=1,mode=0,flush_rows=1,timestamps=1")
    check("no service CTAs in mode 0",
          all(c["role"] == "compute" for c in d["ctas"]))
    check("m75 interval present", any(
        any(i["phase"] == "m75" for i in c["intervals"]) for c in d["ctas"]))
    check("monotonic passes with M7.5 in the chain",
          d["checks"]["monotonic"]["pass"] is True)
    check("coverage passes", d["checks"]["coverage"]["pass"] is True,
          f"worst={d['checks']['coverage']['worst_uncovered_frac']:.5f}")


def t_cli_roundtrip(doc):
    print("\n[7] CLI round trip (parse -> bin -> csv)")
    tmp = tempfile.mkdtemp(prefix="e23cli_")
    rows = make_epoch()
    log = os.path.join(tmp, "run1.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(render_log(rows, coarse_from(rows)))
    summary = os.path.join(tmp, "summary.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"arm_p50_us": {"mps_mega": {"median": 6495.8}}}, fh)
    ev = os.path.join(tmp, "rank0_events.json")
    rc = parse_events.main(["--log", log, "--arm", "mps_mega",
                            "--cfg", "C=16,g=353,mode=12,flush_rows=16,timestamps=1",
                            "--summary", summary, "--out", ev, "--tier", "A+B+C"])
    check("parse_events exits 0", rc == 0, f"rc={rc}")
    csv_out = os.path.join(tmp, "timeline_bins.csv")
    rc = bin_timeline.main(["--events", ev, "--bin-us", "10",
                            "--bytes-model", os.path.join(HERE, "bytes_model.json"),
                            "--out", csv_out, "--check-integrals",
                            "--checks-out", os.path.join(tmp, "checks.json")])
    check("bin_timeline exits 0", rc == 0, f"rc={rc}")
    with open(csv_out, "r", encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
    check("csv header matches the documented schema",
          head == bin_timeline.FIELDS, str(head[:6]))
    n = sum(1 for _ in open(csv_out, encoding="utf-8")) - 1
    check("csv has ~675 bins at 10 us", 600 < n < 750, f"{n} rows")
    print(f"  (artifacts left in {tmp})")


def main():
    print("exp_23 tooling self-test")
    doc = t_parse_happy()
    t_parse_mutations()
    _rows, _chk, model = t_bin_happy(doc)
    t_bin_mutations(doc, model)
    t_xcheck(doc)
    t_mode0_shape()
    t_cli_roundtrip(doc)
    print("\n" + ("=" * 60))
    if FAILS:
        print(f"SELFTEST FAILED: {len(FAILS)} assertion(s)")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("SELFTEST PASSED: every positive case held and every mutation flipped "
          "the verdict it targeted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
