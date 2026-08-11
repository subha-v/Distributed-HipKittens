"""Pre-launch smoke test: peer access, shape resolution, descriptor arithmetic.

Runs no kernel. Everything here must pass before a single launch is trusted.
"""

import sys

import harness_lib as H
from harness_lib import rt, WORLD

SCORED = [
    (64, 7168, 18432, False),
    (512, 4096, 12288, True),
    (2048, 2880, 2880, True),
    (4096, 4096, 4096, False),
    (8192, 4096, 14336, True),
    (8192, 8192, 29568, False),
]

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print(f"  FAIL {message}")
    return condition


print(f"device_count = {rt.device_count()}")
check(rt.device_count() >= WORLD, "fewer than 8 devices visible")

print("\n== peer access ==")
report = rt.enable_peer_access(WORLD)
print(f"  enabled={report['enabled']} already={report['already_enabled']} "
      f"failures={list(report['failures'])[:4]}")
check(len(report["failures"]) == 0, "peer access could not be fully enabled")

print("\n== resolve_shape on the six scored shapes ==")
header = (f"{'m':>5}{'n':>6}{'k':>7}{'bias':>6}{'row':>4}{'BM':>5}{'BN':>5}"
          f"{'BK':>4}{'EB':>5}{'lrow':>5}{'col':>5}{'gemmT':>7}{'redT':>6}"
          f"{'NR':>4}{'LDS':>7}{'sigW':>6}")
print(header)
print("-" * len(header))
for m, n, k, bias in SCORED:
    p = rt.resolve_shape(m, n, k, bias)
    print(f"{m:>5}{n:>6}{k:>7}{str(bias):>6}{p['config_row']:>4}"
          f"{p['bm']:>5}{p['bn']:>5}{p['bk']:>4}{p['eb']:>5}"
          f"{p['lrow_count']:>5}{p['col_count']:>5}{p['gemm_tiles']:>7}"
          f"{p['red_tiles']:>6}{p['num_reducer_ctas']:>4}"
          f"{p['lds_bytes']:>7}{p['signal_words_total']:>6}")
    check(p["lds_bytes"] <= 65536, f"LDS over 64 KiB for {m}x{n}")
    check(p["config_row"] != 0, f"scored shape {m}x{n}x{k} fell to generic row")
    check(p["num_gemm_ctas"] + p["num_reducer_ctas"] == rt.CU_COUNT,
          "CTA split does not partition 304")

print("\n== generic path (official test shapes not in the table) ==")
for m, n, k, bias in [(64, 2880, 2880, True), (8192, 8192, 28672, False),
                      (4096, 2880, 2880, True), (512, 4608, 36864, False)]:
    p = rt.resolve_shape(m, n, k, bias)
    print(f"  {m}x{n}x{k} bias={bias}: row={p['config_row']} "
          f"BM/BN/BK={p['bm']}/{p['bn']}/{p['bk']} eb={p['eb']} "
          f"even_k={p['even_k']} lds={p['lds_bytes']} "
          f"packet={p['packet_fast_path']}")
    check(p["config_row"] == 0, f"{m}x{n}x{k} unexpectedly matched the table")

print("\n== descriptor rebasing arithmetic ==")
shape = (512, 4096, 12288, True)
plan = rt.resolve_shape(*shape)
c_bytes = int(plan["c_heap_bytes"])
sig_off = ((c_bytes + 4095) // 4096) * 4096
heap_bytes = sig_off + ((int(plan["signal_words_total"]) * 4 + 4095) // 4096) * 4096
heaps = [rt.fine_alloc(r, heap_bytes) for r in range(WORLD)]
try:
    for rank in range(WORLD):
        c_heap = heaps[rank]
        signals = heaps[rank] + sig_off
        described = rt.describe_descriptors(rank, heaps, c_heap, signals)
        c_bases = list(described["c_heap"]["peer_bases"])
        s_bases = list(described["signals"]["peer_bases"])
        # c_heap sits at heap offset 0, so its peer bases are the heap bases;
        # the signal region must be offset by exactly sig_off on every rank.
        check(c_bases == list(heaps),
              f"rank{rank} c_heap peer bases != heap bases")
        check(s_bases == [h + sig_off for h in heaps],
              f"rank{rank} signal peer bases mis-rebased")
        check(described["c_heap"]["local_allocation_base"] == c_heap,
              f"rank{rank} local c_heap anchor wrong")
    print("  peer-base rebasing exact on all 8 ranks")

    print("\n== descriptor upload ==")
    for rank in range(WORLD):
        c_ptr, s_ptr = rt.make_descriptors(rank, rank, heaps, heaps[rank],
                                           heaps[rank] + sig_off)
        check(s_ptr - c_ptr == 72, f"rank{rank} descriptor stride != 72 bytes")
        raw = rt.device_to_host(rank, c_ptr, 144)
        words = [int.from_bytes(raw[i * 8:(i + 1) * 8], "little")
                 for i in range(18)]
        check(words[0] == heaps[rank],
              f"rank{rank} uploaded local anchor mismatch")
        check(words[1:9] == list(heaps),
              f"rank{rank} uploaded c_heap peer bases mismatch")
        check(words[9] == heaps[rank] + sig_off,
              f"rank{rank} uploaded signal anchor mismatch")
    print("  72-byte PODs land on device byte-exact")
finally:
    for rank in range(WORLD):
        rt.free_device(rank, heaps[rank])

print()
if failures:
    print(f"SMOKE FAILED: {len(failures)} problem(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("SMOKE PASSED")
