"""Run every static-check gate to completion, collecting failures instead of
aborting on the first one.

`gemm_rs_mi300x_static_checks.py` raises on the FIRST failed requirement, and it
currently fails at `row 1: reducer 56 != RadeonFlow 32` -- an assertion that
exp_13 made deliberately false when it moved row 1's split to 56 and row 6's to
48. So the suite has been aborting before it reaches most of its gates since
exp_13 landed, and it cannot tell exp_14 whether the retile broke anything.

This probe patches `require` to record rather than raise, so every gate runs and
the failures can be separated into "stale expectation about a landed change" and
"actually broken by exp_14". It MODIFIES NOTHING on disk: the checks file is not
owned by this experiment and its stale assertion is reported, not edited.
"""

import importlib.util
import os
import sys

GEMM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
PATH = os.path.join(GEMM, "gemm_rs_mi300x_static_checks.py")

spec = importlib.util.spec_from_file_location("sc", PATH)
sc = importlib.util.module_from_spec(spec)
sys.modules["sc"] = sc
spec.loader.exec_module(sc)

failures = []


def collecting_require(condition, message):
    if not condition:
        failures.append(message)


sc.require = collecting_require
# The @check decorator and the gate functions captured `require` from the module
# global namespace, so rebinding the attribute is enough for anything that looks
# it up at call time.
sc.__dict__["require"] = collecting_require

os.chdir(GEMM)
for name in ("check_gates", "address_sim", "main"):
    fn = getattr(sc, name, None)
    if fn is None or name == "main":
        continue
    try:
        fn()
    except Exception as exc:                                  # noqa: BLE001
        failures.append(f"{name} raised {type(exc).__name__}: {exc}")

print(f"collected {len(failures)} failed requirement(s)\n")
for i, message in enumerate(failures, 1):
    print(f"{i:>3}. {message}")

TILE_WORDS = ("bm", "bn", "bk", "tile", "gate 16", "dispatch", "lds", "eb",
              "col_count", "lrow")
NR_WORDS = ("reducer", "RadeonFlow", "nred")
print("\n--- classification ---")
stale = [m for m in failures if any(w in m for w in NR_WORDS)]
tiley = [m for m in failures
         if any(w in m.lower() for w in TILE_WORDS) and m not in stale]
other = [m for m in failures if m not in stale and m not in tiley]
print(f"  stale NR expectations (exp_13's landed change): {len(stale)}")
for m in stale:
    print(f"    - {m}")
print(f"  tile/dispatch related (would be exp_14's): {len(tiley)}")
for m in tiley:
    print(f"    - {m}")
print(f"  other: {len(other)}")
for m in other:
    print(f"    - {m}")
