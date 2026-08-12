"""Geomeans for exp_14, with the statistic named beside every vector.

LESSONS.md records a phantom 7.3% regression that came from comparing an M7 MEAN
to a recorded BEST, so nothing here is printed without its statistic attached,
and the paired ratio -- the only denominator that survives this node -- is
computed from the within-run contrasts rather than from any cross-session pair.
"""

import math

def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))

# Gate M7, 3 rotations x 50, MEANS. Same script, same protocol, both sessions.
M7_BEFORE = [66.63, 89.12, 92.26, 200.64, 658.26, 1662.12]   # exp_13 logs
M7_AFTER = [65.65, 66.95, 86.92, 201.65, 637.47, 1610.43]    # exp_14 logs

# Recorded BEST-of-arm denominator carried by the ledger.
BEST_BEFORE = [63.44, 85.68, 89.41, 198.71, 613.70, 1616.63]

# Paired within-run contrast, median over 15 independent allocation draws
# (pool.py, gain_b%). Rows 4-6 keep the same instantiation, so 0 by construction.
PAIRED_GAIN_PCT = [1.70, 32.80, 6.76, 0.0, 0.0, 0.0]

# Directly measured bests of the landed arms over those same 15 draws.
MEASURED_BEST_AFTER_123 = [61.71, 64.12, 82.71]
MEASURED_BEST_BEFORE_123 = [62.92, 85.32, 88.96]

print("GATE M7 (MEANS, 3 rotations x 50) -- same script both sessions")
print(f"  before (exp_13): {M7_BEFORE}  geomean {gm(M7_BEFORE):.2f} us")
print(f"  after  (exp_14): {M7_AFTER}  geomean {gm(M7_AFTER):.2f} us")
print(f"  per-shape ratio: "
      f"{[round(a / b, 4) for a, b in zip(M7_AFTER, M7_BEFORE)]}")
print(f"  geomean ratio  : {gm(M7_AFTER) / gm(M7_BEFORE):.4f} "
      f"({(gm(M7_AFTER) / gm(M7_BEFORE) - 1) * 100:+.2f}%)")
print("  CAVEAT: rows 4-6 have an unchanged tile and an unchanged instantiation,")
print("  yet moved +0.5 / -3.2 / -3.1 %. That is cross-session drift, not this")
print("  change, and it inflates the M7 delta by about 2 points.")

ratios = [1.0 / (1.0 + g / 100.0) for g in PAIRED_GAIN_PCT]
print("\nPAIRED within-run contrast (15 allocation draws, best-of-block)")
print(f"  per-shape ratio: {[round(r, 4) for r in ratios]}")
paired = gm(ratios)
print(f"  geomean ratio  : {paired:.4f} ({(paired - 1) * 100:+.2f}%)")
print("  This is the trustworthy figure: every ratio is a same-process")
print("  comparison against the tile it replaces, and rows 4-6 contribute")
print("  exactly 1.0 because their instantiation did not change.")

after_best = [b * r for b, r in zip(BEST_BEFORE, ratios)]
print("\nBEST-of-arm denominator vector for the ledger")
print(f"  before: {[round(x, 2) for x in BEST_BEFORE]}  "
      f"geomean {gm(BEST_BEFORE):.2f} us")
print(f"  after : {[round(x, 2) for x in after_best]}  "
      f"geomean {gm(after_best):.2f} us")

print("\nCross-check: directly measured bests of both tiles over the same draws")
for i, (a, b) in enumerate(zip(MEASURED_BEST_AFTER_123,
                               MEASURED_BEST_BEFORE_123)):
    print(f"  shape {i + 1}: {b:.2f} -> {a:.2f} us  ({(a / b - 1) * 100:+.2f}%)")
print("  (the T arm reproduces the ledger's recorded bests to within 0.5-0.7%,")
print("   which is what makes the scaled vector above meaningful)")
