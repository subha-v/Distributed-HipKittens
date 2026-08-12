#!/usr/bin/env python3
"""Pull the specific series and checks the write-up needs out of saturation.json."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "saturation.json"))
pts = d["points"]
dv = d["derived"]


def ser(**sel):
    s = [p for p in pts if all(p.get(k) == v for k, v in sel.items())]
    return sorted(s, key=lambda p: p["ctas"])


def line(tag, s):
    print("%-34s %s" % (tag, "  ".join("C%d=%.1f" % (p["ctas"], p["value"])
                                       for p in s)))


print("=== xgmi isolated, payload only (p0) ===")
for fan in ("single", "rr7"):
    for m in (1, 4, 8):
        line("%s mlp%d" % (fan, m), ser(mode="xgmi", fanout=fan, mlp=m,
                                        concurrency="isolated", protocol=0))
print()
print("=== xgmi concurrent, payload only (p0) ===")
for fan in ("single", "rr7"):
    for m in (1, 4, 8):
        line("%s mlp%d" % (fan, m), ser(mode="xgmi", fanout=fan, mlp=m,
                                        concurrency="concurrent", protocol=0))
print()
print("=== xgmi isolated WITH protocol (mlp4) ===")
for fan in ("single", "rr7"):
    for g in (2, 16):
        line("%s g%d" % (fan, g), ser(mode="xgmi", fanout=fan, mlp=4,
                                      concurrency="isolated", protocol=1,
                                      proto_g=g))
print()
print("=== hbm / mfma ===")
line("hbm isolated", ser(mode="hbm", concurrency="isolated"))
line("hbm concurrent", ser(mode="hbm", concurrency="concurrent"))
for shape in (4, 16, 256):
    line("mfma shape%d" % shape, ser(mode="mfma", mlp=shape))
print()
print("=== peak fractions at the knee / plateau ===")
for name, kn in sorted(dv["knees"].items()):
    pk = 537.6 if "rr7" in name else 76.8 if "xgmi" in name else \
        8000.0 if name.startswith("hbm") else 2300.0
    print("  %-52s knee=%-5s plateau=%9.2f = %5.1f%% of %.1f  monotone=%s"
          % (name, kn.get("knee_ctas"), kn["plateau"],
             100 * kn["plateau"] / pk, pk, kn.get("monotone")))
print()
print("=== concurrency proof: overlap_pct over concurrent points ===")
con = [p for p in pts if p["concurrency"] == "concurrent"]
ov = sorted(p["overlap_pct_median"] for p in con)
print("  n=%d  min=%.1f  p25=%.1f  median=%.1f  max=%.1f  n_below_90=%d"
      % (len(con), ov[0], ov[len(ov) // 4], ov[len(ov) // 2], ov[-1],
         len([x for x in ov if x < 90.0])))
res = [p for p in pts if p["concurrency"] == "reserved_only"]
ov2 = sorted(p["overlap_pct_median"] for p in res)
print("  reserved_only: n=%d min=%.1f median=%.1f" % (len(ov2), ov2[0],
                                                      ov2[len(ov2) // 2]))
print()
print("=== concurrent / isolated (payload only), worst and best ===")
ci = [(v["ratio"], k) for k, v in dv["concurrent_over_isolated"].items()
      if "|p0|" in k]
ci.sort()
for r, k in ci[:4] + ci[-3:]:
    print("  %-50s ratio=%.4f" % (k, r))
print("  n=%d median=%.4f" % (len(ci), sorted(r for r, _ in ci)[len(ci) // 2]))
print()
print("=== protocol over payload ===")
po = sorted(dv["protocol_over_payload"].items())
for k, v in po:
    if "|isolated" in k:
        print("  %-40s ratio=%.4f  (%.1f -> %.1f)"
              % (k, v["ratio"], v["payload_only"], v["with_protocol"]))
print()
print("=== Q2: where interference exceeds reservation ===")
for k, v in sorted(dv["q2_dedicated_pool"].items()):
    if v["interference_exceeds_reservation"]:
        print("  %-48s C=%-4s interf=%6.2f%% reserve=%5.2f%% payload=%7.1f"
              % (k, v["ctas_reserved"], 100 * v["interference_cost_frac"],
                 100 * v["reservation_cost_frac_from_H4_linearity"],
                 v["payload_delivered_GBps"]))
print("  summary: %s" % json.dumps(dv["q2_summary"], indent=2))
print()
print("=== single-link interference vs C (payload only) ===")
for fan in ("single", "rr7"):
    row = []
    for k, v in sorted(dv["q2_dedicated_pool"].items()):
        if "|%s|p0|" % fan in k and "xgmi" in k:
            row.append((v["ctas_reserved"], 100 * v["interference_cost_frac"]))
    print("  %-7s %s" % (fan, "  ".join("C%d=%.2f%%" % t for t in sorted(row))))
print()
print("=== hypotheses ===")
for k, v in sorted(dv["hypotheses"].items()):
    print("  %s" % k)
    for kk, vv in v.items():
        print("      %-34s %s" % (kk, vv))
