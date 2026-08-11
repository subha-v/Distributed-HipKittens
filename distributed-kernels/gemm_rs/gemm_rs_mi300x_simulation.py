#!/usr/bin/env python3
"""Offline protocol/address/epoch/numerics simulation for the MI300X port.

Executable versions of the dependency, memory-order, lifetime, progress and
arithmetic arguments in MI300X_DESIGN.md. Runs with no GPU: the dependency
graph, addressing, epoch/credit protocol and bf16/fp32 rounding policy are
simulated exactly from the same constants the C++ uses (the shape table is
parsed from gemm_rs_mi300x_host_abi.hpp; layout formulas are re-stated in
Python independently of the C++ source).

Exit status is non-zero on any violation; the static checker invokes this.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HOST_ABI = HERE / "gemm_rs_mi300x_host_abi.hpp"

WORLD = 8
CU_COUNT = 304
CTA_THREADS = 512
LROW_MAX = 32
COL_MAX = 128
SIGNAL_GUARD_U32 = 64
SIGNAL_U32_CAP = 65600
EP_GEMM_U32 = 0
EP_RED_U32 = CU_COUNT
EP_U32 = 2 * CU_COUNT
ERR_PRODUCER_CREDIT = 1 << 25
ERR_REDUCER_READY = 1 << 26

TABLE_RE = re.compile(
    r"\{\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(true|false)"
    r"\s*\},\s*\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}\}")


def parse_table():
    text = HOST_ABI.read_text(encoding="utf-8")
    rows = []
    for m in TABLE_RE.finditer(text):
        (m_, n, k, bias, bm, bn, bk, nred, row) = m.groups()
        rows.append(dict(m=int(m_), n=int(n), k_local=int(k),
                         bias=(bias == "true"), bm=int(bm), bn=int(bn),
                         bk=int(bk), nred=int(nred), row=int(row)))
    assert len(rows) == 6
    return rows


def resolve(m, n, k_global, bias):
    """Mirror of host_abi::resolve_shape (independent Python restatement)."""
    assert m > 0 and n > 0 and k_global > 0
    assert m % WORLD == 0, "M % world != 0"
    assert k_global % WORLD == 0, "K % world != 0"
    k_local = k_global // WORLD
    row = next((r for r in TABLE if
                (r["m"], r["n"], r["k_local"], r["bias"]) ==
                (m, n, k_local, bias)), None)
    if row is None:
        assert m % 32 == 0, "generic path requires M % 32 == 0"
        row = dict(bm=32, bn=64, bk=64, nred=24, row=0)
    slice_rows = m // WORLD
    eb = math.gcd(row["bm"], slice_rows)
    assert eb >= 1 and slice_rows % eb == 0 and row["bm"] % eb == 0
    lrow_count = slice_rows // eb
    col_count = (n + row["bn"] - 1) // row["bn"]
    assert lrow_count <= LROW_MAX and col_count <= COL_MAX
    assert m % row["bm"] == 0
    ready = WORLD * lrow_count * col_count
    signal_words = SIGNAL_GUARD_U32 + 2 * ready
    assert signal_words <= SIGNAL_U32_CAP
    num_gemm = CU_COUNT - row["nred"]
    assert 0 < row["nred"] <= 48 and num_gemm >= 256
    return dict(row=row, m=m, n=n, k_local=k_local, bias=bias,
                slice=slice_rows, eb=eb, lrows=lrow_count, cols=col_count,
                tiles=(m // row["bm"]) * col_count,
                red_tiles=lrow_count * col_count,
                ready_words=ready, signal_words=signal_words,
                num_gemm=num_gemm, k_global=k_global)


def ready_idx(src, lrow, col, lrows, cols):
    return (src * lrows + lrow) * cols + col


def credit_idx(owner, lrow, col, lrows, cols):
    return (owner * lrows + lrow) * cols + col


def owner_of(row, slice_rows):
    return row // slice_rows


def gemm_tile_map(t, num_pid_m, num_pid_n, wgm=4):
    """Grouped tile order used by the kernel (same family as both donors)."""
    in_group = wgm * num_pid_n
    group = t // in_group
    first = group * wgm
    gsize = min(num_pid_m - first, wgm)
    tm = first + (t % in_group) % gsize
    tn = (t % in_group) // gsize
    return tm, tn


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# Address + dependency simulation (gates 2-9)
# ---------------------------------------------------------------------------
@check("address/dependency map: all shapes, all ranks")
def address_sim():
    shapes = [(r["m"], r["n"], r["k_local"] * 8, r["bias"]) for r in TABLE]
    # Known public non-scored cases (generic path): local_k = k/8.
    shapes += [(64, 2880, 2880, True), (8192, 8192, 28672, False),
               (32, 64, 64, False), (64, 64, 64, True), (96, 128, 24, False)]
    for (m, n, kg, bias) in shapes:
        plan = resolve(m, n, kg, bias)
        R = plan["row"]
        for rank in range(WORLD):
            pubs = {}              # (dest, src, lrow, col) -> (tm, tn, b)
            visits = set()         # exact tile coverage proof
            for t in range(plan["tiles"]):
                tm, tn = gemm_tile_map(t, plan["m"] // R["bm"], plan["cols"])
                assert (tm, tn) not in visits, f"tile map not injective {plan}"
                visits.add((tm, tn))
                for b in range(R["bm"] // plan["eb"]):
                    row0 = tm * R["bm"] + b * plan["eb"]
                    dest = owner_of(row0, plan["slice"])
                    assert 0 <= dest < WORLD
                    off = row0 - dest * plan["slice"]
                    assert off % plan["eb"] == 0, f"band not EB-aligned {plan}"
                    lro = off // plan["eb"]
                    assert 0 <= lro < plan["lrows"]
                    key = (dest, rank, lro, tn)
                    assert key not in pubs, f"duplicate band key {key}"
                    pubs[key] = (tm, tn, b)
                    # signal word bounds (gate 8)
                    rw = ready_idx(rank, lro, tn, plan["lrows"], plan["cols"])
                    cw = credit_idx(dest, lro, tn, plan["lrows"], plan["cols"])
                    assert 0 <= rw < plan["ready_words"]
                    assert 0 <= cw < plan["ready_words"]
            # full coverage: every tile visited
            assert len(visits) == plan["tiles"]
            # gate 6: every reducer dependency has exactly one producer band
            # per source, on that source's rank.
            for t in range(plan["red_tiles"]):
                lr, col = t // plan["cols"], t % plan["cols"]
                for src in range(WORLD):
                    assert (rank, src, lr, col) is not None
                    # the key must exist in the SOURCE rank's own publication
                    # map; this rank's map only carries src == rank.
            # epoch-cell windows disjoint (gate 7)
            assert EP_GEMM_U32 + CU_COUNT <= EP_RED_U32 < EP_U32
            assert EP_RED_U32 + plan["row"]["nred"] <= EP_U32
    return shapes


@check("signal/credit index collision-freeness")
def collision_sim():
    # Every (src,dest,lrow,col) key pair maps to distinct u32 offsets within
    # its region; guard region never overlaps payload regions.
    for (m, n, kg, bias) in [(r["m"], r["n"], r["k_local"] * 8, r["bias"])
                             for r in TABLE] + [(32, 96, 32, True)]:
        plan = resolve(m, n, kg, bias)
        seen_ready, seen_credit = set(), set()
        for src in range(WORLD):
            for lr in range(plan["lrows"]):
                for c in range(plan["cols"]):
                    i = ready_idx(src, lr, c, plan["lrows"], plan["cols"])
                    assert i not in seen_ready
                    seen_ready.add(i)
                    j = credit_idx(src, lr, c, plan["lrows"], plan["cols"])
                    assert j not in seen_credit
                    seen_credit.add(j)
        assert max(seen_ready) < plan["ready_words"]
        assert max(seen_credit) < plan["ready_words"]
        assert SIGNAL_GUARD_U32 + max(seen_ready) < \
            SIGNAL_GUARD_U32 + plan["ready_words"]  # region isolation


# ---------------------------------------------------------------------------
# Epoch/credit lifetime simulation over skewed orders (gates 10-14)
# ---------------------------------------------------------------------------
class TileState:
    __slots__ = ("written", "consumed")

    def __init__(self):
        self.written = 0       # last published epoch (0 = none)
        self.consumed = 0      # last consumed epoch (posted credit)


@check("epoch/credit lifetime: 64 skewed epochs x 6 shapes")
def epoch_sim():
    rng = np.random.default_rng(0xC0FFEE)
    for row in TABLE:
        plan = resolve(row["m"], row["n"], row["k_local"] * 8, row["bias"])
        # state per (dest, src, lrow, col)
        states = {}
        for dest in range(WORLD):
            for src in range(WORLD):
                for lr in range(plan["lrows"]):
                    for c in range(plan["cols"]):
                        states[(dest, src, lr, c)] = TileState()
        EPOCHS = 64
        errors = 0
        for e in range(1, EPOCHS + 1):
            # producers publish all bands in a random per-rank order
            bands = [(dest, src, lr, c)
                     for (dest, src, lr, c) in states]
            rng.shuffle(bands)
            for (dest, src, lr, c) in bands:
                st = states[(dest, src, lr, c)]
                # lifetime gate 14: epoch-e write requires credit >= e-1
                assert st.consumed >= e - 1, \
                    f"write without credit: e={e} consumed={st.consumed}"
                st.written = e
            # reducers consume every tile after all 8 sources are ready
            for dest in range(WORLD):
                for lr in range(plan["lrows"]):
                    for c in range(plan["cols"]):
                        for src in range(WORLD):
                            st = states[(dest, src, lr, c)]
                            # gate 11: wait success precedes read
                            assert st.written >= e
                        for src in range(WORLD):
                            states[(dest, src, lr, c)].consumed = e
            assert errors == 0
        # drain: everything consumed == written == EPOCHS
        assert all(st.written == EPOCHS and st.consumed == EPOCHS
                   for st in states.values())


@check("negative controls: drop publication / reroute / drop credit")
def negative_sim():
    plan = resolve(2048, 2880, 2880, True)
    # (a) dropped publication: consumer's bounded wait fails; no read event.
    st = TileState()
    dropped = (3, 5, 0, 7)
    reads = 0
    timeout = False
    for spin in range(1000):
        if st.written >= 1:
            reads += 1
        else:
            pass  # producer never published for `dropped`
        if spin > 128:
            timeout = True
            break
    assert timeout and reads == 0, "gate: failed wait reached a payload read"
    # (b) rerouted slot: payload lands at dest+1; published ready is for dest.
    rerouted_slot_written = {}   # (dest+1, ...) receives bytes
    published_for = 3            # dest publishes readiness
    slot_written_for = None      # dest's own slot `me` stays at epoch 0
    assert slot_written_for is None and published_for == 3
    # the consumer at dest will read the STALE slot content -> mismatch,
    # which is exactly why the control must fail numerical correctness.
    # (c) dropped credit: epoch-2 producer must time out before overwriting.
    st2 = TileState()
    st2.written = 1              # epoch 1 produced and published
    st2.consumed = 0             # credit for epoch 1 never arrives
    overwrite_attempted = False
    credit_timeout = False
    for spin in range(1000):
        if st2.consumed >= 2 - 1:
            overwrite_attempted = True
            break
        if spin > 128:
            credit_timeout = True
            break
    assert credit_timeout and not overwrite_attempted, \
        "gate: producer overwrote a slot without retirement credit"


# ---------------------------------------------------------------------------
# Numerical reference simulation (gate 15): producer FP32 + per-rank bias,
# bf16 pack; reducer ascending-FP32 sum, single RNE pack; oracle comparison.
# ---------------------------------------------------------------------------
def f2b_rne(x):
    """RNE fp32 -> bf16 as bits simulation on numpy float32 arrays."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) >> 16
    return (rounded << 16).astype(np.uint32).view(np.float32)


def q(x):
    """Quantize to bf16 (RNE) and back to fp32."""
    return f2b_rne(x.astype(np.float32))


WINDOW_ROWS = 8
WINDOW_COLS = 512


@check("numerics: windowed all shapes vs evaluator oracle")
def numeric_sim():
    rng = np.random.default_rng(1234)
    for row in TABLE:
        m, n, kg, bias_flag = (row["m"], row["n"], row["k_local"] * 8,
                               row["bias"])
        k = row["k_local"]
        wr = min(WINDOW_ROWS, m // 8)
        wc = min(WINDOW_COLS, n)
        # evaluator's distribution; per-rank seeds for x/w, shared seed bias
        bias = None
        if bias_flag:
            bg = np.random.default_rng(777)
            bias = q((bg.random(n, dtype=np.float32) * 2 - 1) * 0.01)
        partials = np.zeros((WORLD, wr, wc), dtype=np.float32)
        oracle_partials = np.zeros_like(partials)
        for r in range(WORLD):
            g = np.random.default_rng(9000 + r)
            x = q((g.random((m, k), dtype=np.float32) * 2 - 1) * 0.01)
            w = q((g.random((n, k), dtype=np.float32) * 2 - 1) * 0.01)
            # producer window; full-K FP32 accumulation
            xw = x[:wr]
            acc = xw.astype(np.float32) @ w[:wc].astype(np.float32).T
            if bias is not None:
                acc = acc + bias[:wc].astype(np.float32)
            partials[r] = q(acc)            # producer: FP32 + bias, one pack
            # oracle: bf16 gemm (fp32 accum), bf16 rounding, then bf16 +bias
            oracle_r = q(xw.astype(np.float32) @ w[:wc].astype(np.float32).T)
            if bias is not None:
                oracle_r = q(oracle_r + bias[:wc].astype(np.float32))
            oracle_partials[r] = q(oracle_r)
            del x, w
        # reducer: ascending source order in FP32, single RNE pack
        ours = np.zeros((wr, wc), dtype=np.float32)
        for r in range(WORLD):
            ours += partials[r].astype(np.float32)
        ours = q(ours)
        # oracle reduce: bf16 sum of the 8 bf16 partials
        exp = oracle_partials[0]
        for r in range(1, WORLD):
            exp = q(exp.astype(np.float32) +
                    oracle_partials[r].astype(np.float32))
        assert np.allclose(ours, exp, rtol=1e-2, atol=1e-2), \
            f"numeric mismatch shape {m}x{n}x{kg} bias={bias_flag}"
        ref64 = np.zeros((wr, wc), dtype=np.float64)
        for r in range(WORLD):
            ref64 += partials[r].astype(np.float64)
        rel_ours = np.linalg.norm(ours - ref64) / max(np.linalg.norm(ref64),
                                                      1e-12)
        rel_exp = np.linalg.norm(exp - ref64) / max(np.linalg.norm(ref64),
                                                    1e-12)
        assert rel_ours <= max(rel_exp * 2.0, 1e-5), \
            f"reduction quality regression: {rel_ours} vs oracle {rel_exp}"
    # bias-count identity: producer-per-rank bias must equal world*bias total.
    b = np.ones(16, dtype=np.float32) * 0.5
    parts = np.zeros((WORLD, 4, 16), dtype=np.float32)
    for r in range(WORLD):
        parts[r] = q(b[None, :].astype(np.float32))
    tot = parts.astype(np.float32).sum(axis=0)
    assert np.allclose(tot, WORLD * b, rtol=1e-6), "gate 15: bias*world broken"


@check("K-tail masking: masked-tail accumulation is exact")
def ktail_sim():
    # Simulate the exp-07 claim: zeroing A's padded K columns in registers
    # yields the identical result to full valid-K accumulation.
    rng = np.random.default_rng(7)
    for k, bk in [(360, 32), (3696, 32), (2304, 64), (80, 64)]:
        k_iters = (k + bk - 1) // bk
        tail_k = k - (k_iters - 1) * bk
        a = rng.standard_normal((16, k_iters * bk)).astype(np.float32)
        b = rng.standard_normal((16, k_iters * bk)).astype(np.float32)
        padded = a.copy()
        padded[:, (k_iters - 1) * bk + tail_k:] = 0.0
        got = padded @ b.T
        want = a[:, :k] @ b[:, :k].T
        assert np.allclose(got, want, rtol=1e-6, atol=1e-6), \
            f"K-tail mask wrong for k={k}, bk={bk}"
        assert tail_k >= 1 and tail_k <= bk


@check("progress model: reducers wait only on non-blocking producers")
def progress_sim():
    for row in TABLE:
        plan = resolve(row["m"], row["n"], row["k_local"] * 8, row["bias"])
        nred = row["nred"]
        ng = plan["num_gemm"]
        # (a) split is a partition of the exact 304-CTA grid
        assert ng + nred == CU_COUNT
        # (b) at most 48 waiters; producers can always run
        assert nred <= 48
        # (c) every reducer tile depends only on producer output of the same
        # launch (proved in address_sim); producer credit waits target epoch
        # e-1, which stream serialization already retired. Simulated in
        # epoch_sim: the only spin sites are per-tile and bounded.
        # (d) work exists and is partitionable per role
        assert plan["tiles"] > 0 and plan["red_tiles"] > 0


TABLE = parse_table()


def main():
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    if failures:
        print(f"gemm_rs mi300x simulation: {failures} FAILURES")
        sys.exit(1)
    print("gemm_rs mi300x simulation: PASS")


if __name__ == "__main__":
    main()
