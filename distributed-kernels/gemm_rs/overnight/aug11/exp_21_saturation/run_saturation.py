#!/usr/bin/env python3
"""exp_21 driver: resource-saturation curves vs CTA count for GEMM-RS on 8x MI300X.

Paper Fig 2 / Q4(a). Produces saturation.json. See plan.md for the controls and the
pre-registered predictions, design.md for the task graph and the rejected alternatives.

One process drives all eight devices (the harness model, NOT the evaluator's one-process-per-rank
topology -- that caveat is carried in the JSON and must appear in result.md).

Every throughput here comes from the ROLE's own s_memrealtime span, never from host wall time,
because in the concurrent arms the two roles finish at different times and wall time attributes
nothing. Host wall time is recorded per point as a cross-check only.
"""

import argparse
import importlib.util
import json
import os
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
WORLD = 8


def load_module():
    path = os.path.join(BUILD, "sat_ubench.so")
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run build.sh first")
    spec = importlib.util.spec_from_file_location("sat_ubench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Sweep grid. 304-wide because MI300X SPX has 304 CU; the sibling's 256 is a
# gfx950 fact (FIGURE_SPECS.md section 5.2).
# ---------------------------------------------------------------------------
CTAS_A = [32, 64, 96, 128, 160, 192, 224, 256, 288, 304]
CTAS_B = [8, 16, 32, 64, 128, 256, 304]
CTAS_C = [1, 2, 4, 8, 16, 32, 64]
# in-flight remote-op bound: 16 B peer stores issued per thread before the drain.
# 0 == unbounded == the production shape (production drains once per release group).
DEPTHS = [0, 1, 4, 8]
FANOUTS = {"single": 0, "rr7": 1}


class Geometry:
    """Sizes chosen in plan.md section 4; every one is recorded in the JSON."""

    def __init__(self, mod, args):
        self.N = args.n
        self.BM, self.BN, self.BK = mod.BM, mod.BN, mod.BK
        self.WIN_ROWS = mod.WIN_ROWS
        self.WIN_BYTES = mod.WIN_BYTES

        # ---- mode a: cache-resident operands, so panel a prices MFMA issue ----
        self.a_tiles_m = 4
        self.a_tiles_n = 4
        self.k_iters = args.k_iters
        self.a_rows = self.a_tiles_m * self.BM
        self.a_cols = self.k_iters * self.BK            # K, exactly divisible -> K_TAIL == false
        self.b_rows = self.a_tiles_n * self.BN
        self.a_work = args.a_work

        # ---- mode b: REDV=1 reduce over a working set above the 256 MB MALL ----
        self.red_slot_rows = args.red_slot_rows
        self.red_slot_elems = self.red_slot_rows * self.N
        self.b_strip_rows = self.BM
        self.b_cols_per_row = self.N // self.BN
        self.b_row_bands = self.red_slot_rows // self.b_strip_rows
        self.b_work = args.b_work

        # ---- mode c: 16 KB emit windows into slot `me` of the destination rank ----
        self.c_work = args.c_work
        self.c_cols_per_row = self.N // self.BN
        # one unique destination window per pushed window, so the verify is exact
        self.c_slot_rows = max(self.c_work, self.WIN_ROWS)
        self.c_slot_elems = self.c_slot_rows * self.N
        self.heap_bytes = WORLD * self.c_slot_elems * 2
        self.sig_bytes = 1 << 16

    def summary(self):
        return {
            "N": self.N, "BM": self.BM, "BN": self.BN, "BK": self.BK,
            "WIN_ROWS": self.WIN_ROWS, "WIN_BYTES": self.WIN_BYTES,
            "mode_a": {"tiles_m": self.a_tiles_m, "tiles_n": self.a_tiles_n,
                       "k_iters": self.k_iters, "K": self.a_cols, "work_tiles": self.a_work,
                       "flops_per_tile": 2 * self.BM * self.BN * self.a_cols,
                       "operand_bytes": (self.a_rows + self.b_rows) * self.a_cols * 2},
            "mode_b": {"strip_rows": self.b_strip_rows, "cols_per_row": self.b_cols_per_row,
                       "row_bands": self.b_row_bands, "work_strips": self.b_work,
                       "distinct_strips": self.b_row_bands * self.b_cols_per_row,
                       "bytes_per_strip": 9 * self.b_strip_rows * self.BN * 2,
                       "working_set_bytes": WORLD * self.red_slot_elems * 2
                                            + self.red_slot_elems * 2},
            "mode_c": {"work_windows": self.c_work, "cols_per_row": self.c_cols_per_row,
                       "slot_rows": self.c_slot_rows,
                       "bytes_per_launch_per_rank": self.c_work * self.WIN_BYTES,
                       "heap_bytes_per_rank": self.heap_bytes},
        }


def sh(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:                                  # noqa: BLE001
        return f"<failed: {exc}>"


def clocks_snapshot():
    """rocm-smi if reachable, sysfs otherwise, 'unavailable' if neither. Recorded per point."""
    out = sh("rocm-smi --showclocks")
    lines = [l.strip() for l in out.splitlines() if "clk" in l.lower()]
    if lines:
        return " | ".join(lines)
    out = sh("for f in /sys/class/drm/card*/device/pp_dpm_sclk; do "
             "echo -n \"$f:\"; grep '\\*' $f; done")
    return out if out and "<failed" not in out else "unavailable"


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
class Fixture:
    def __init__(self, mod, geo, payload_fine):
        self.mod, self.geo = mod, geo
        self.payload_fine = payload_fine
        alloc_payload = mod.fine_alloc if payload_fine else mod.plain_alloc

        print(f"[alloc] heap {geo.heap_bytes/2**30:.2f} GiB/rank "
              f"({'fine' if payload_fine else 'coarse'}-grained), "
              f"red_src {WORLD*geo.red_slot_elems*2/2**20:.0f} MiB/rank", flush=True)
        self.heap = [alloc_payload(r, geo.heap_bytes) for r in range(WORLD)]
        self.sig = [mod.fine_alloc(r, geo.sig_bytes) for r in range(WORLD)]
        self.red_src = [mod.plain_alloc(r, WORLD * geo.red_slot_elems * 2) for r in range(WORLD)]
        self.red_dst = [mod.plain_alloc(r, geo.red_slot_elems * 2) for r in range(WORLD)]
        self.a_mat = [mod.plain_alloc(r, geo.a_rows * geo.a_cols * 2) for r in range(WORLD)]
        self.b_mat = [mod.plain_alloc(r, geo.b_rows * geo.a_cols * 2) for r in range(WORLD)]
        self.stamps = [mod.plain_alloc(r, mod.GRID_MAX * 2 * 8) for r in range(WORLD)]
        self.work_done = [mod.plain_alloc(r, mod.GRID_MAX * 4) for r in range(WORLD)]
        self.res_done = [mod.plain_alloc(r, 4) for r in range(WORLD)]

        for r in range(WORLD):
            mod.fill_region(r, self.a_mat[r], geo.a_rows * geo.a_cols, 0)
            mod.fill_region(r, self.b_mat[r], geo.b_rows * geo.a_cols, 0)
            mod.fill_region(r, self.red_src[r], WORLD * geo.red_slot_elems, 2)
        print("[alloc] operands and reduce sources filled", flush=True)

    def close(self):
        for group in (self.heap, self.sig, self.red_src, self.red_dst, self.a_mat,
                      self.b_mat, self.stamps, self.work_done, self.res_done):
            for r in range(WORLD):
                try:
                    self.mod.free_device(r, group[r])
                except Exception:                             # noqa: BLE001
                    pass

    def cfg(self, rank, mode, conc, control, grid, res_ctas, depth, fanout,
            protocol, release_group, deadline_ticks):
        geo = self.geo
        return {
            "device": rank, "me": rank, "mode": mode,
            "conc": 1 if conc else 0, "control": 1 if control else 0,
            "grid": grid, "num_res_ctas": res_ctas,
            "num_gemm_ctas": (self.mod.GRID_MAX - res_ctas) if conc else 0,
            "a_ptr": self.a_mat[rank], "b_ptr": self.b_mat[rank],
            "a_rows": geo.a_rows, "a_cols": geo.a_cols, "b_rows": geo.b_rows,
            "k_iters": geo.k_iters, "a_tiles_m": geo.a_tiles_m, "a_tiles_n": geo.a_tiles_n,
            "a_work": geo.a_work,
            "red_src_base": self.red_src[rank], "red_slot_elems": geo.red_slot_elems,
            "red_dst": self.red_dst[rank], "red_row_stride": geo.N, "red_valid_cols": geo.N,
            "b_strip_rows": geo.b_strip_rows, "b_cols_per_row": geo.b_cols_per_row,
            "b_row_bands": geo.b_row_bands, "b_work": geo.b_work,
            "heap_local": self.heap[rank], "heap_bases": list(self.heap),
            "c_slot_elems": geo.c_slot_elems, "c_row_stride": geo.N,
            "c_cols_per_row": geo.c_cols_per_row, "c_work": geo.c_work,
            "depth": depth, "fanout": fanout,
            "protocol": protocol, "release_group": release_group,
            "sig_local": self.sig[rank], "sig_bases": list(self.sig),
            "deadline_ticks": deadline_ticks,
            "stamps": self.stamps[rank], "work_done": self.work_done[rank],
            "res_done": self.res_done[rank],
        }


# ---------------------------------------------------------------------------
# One rotation: reset instrumentation, launch on all 8 devices, sync, read back.
# ---------------------------------------------------------------------------
def one_rotation(mod, fx, cfgs):
    for r in range(WORLD):
        mod.zero_region(r, fx.stamps[r], mod.GRID_MAX * 2 * 8)
        mod.zero_region(r, fx.work_done[r], mod.GRID_MAX * 4)
        mod.zero_region(r, fx.res_done[r], 4)
    wall0 = time.perf_counter()
    for r in range(WORLD):
        mod.launch(cfgs[r])
    for r in range(WORLD):
        mod.device_synchronize(r)
    wall_s = time.perf_counter() - wall0
    stamps, done = [], []
    for r in range(WORLD):
        raw = mod.device_to_host(r, fx.stamps[r], mod.GRID_MAX * 2 * 8)
        stamps.append(struct.unpack(f"<{mod.GRID_MAX*2}Q", raw))
        raw = mod.device_to_host(r, fx.work_done[r], mod.GRID_MAX * 4)
        done.append(struct.unpack(f"<{mod.GRID_MAX}I", raw))
    return wall_s, stamps, done


def role_span(stamps, lo, hi):
    """max end - min start over CTAs [lo, hi). Only CTAs that actually stamped count."""
    starts = [stamps[2 * p] for p in range(lo, hi) if stamps[2 * p] != 0]
    ends = [stamps[2 * p + 1] for p in range(lo, hi) if stamps[2 * p + 1] != 0]
    if not starts or not ends:
        return 0
    return max(ends) - min(starts)


def measure(mod, fx, geo, mode, overlay, ctas, depth, fanout_name, protocol,
            rotations, warmup_ms, release_group, deadline_ticks=0):
    conc = overlay in ("concurrent", "reserve_control")
    control = overlay == "reserve_control"
    grid = mod.GRID_MAX if conc else ctas
    res_ctas = ctas
    fanout = FANOUTS.get(fanout_name, 0)
    cfgs = [fx.cfg(r, mode, conc, control, grid, res_ctas, depth, fanout,
                   protocol, release_group, deadline_ticks) for r in range(WORLD)]

    # Duration-based warmup. Idle sclk here is ~125-132 MHz against ~1900 pinned; a
    # fixed-iteration warmup already produced a 60%-wrong number once on this node.
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < warmup_ms:
        one_rotation(mod, fx, cfgs)

    # The reported metric of a point:
    #   mode a                -> the GEMM role's TFLOPS
    #   mode b/c, resource arm -> the resource role's GB/s
    #   reserve_control        -> the GEMM role's TFLOPS (the capacity control has no traffic)
    metric_is_gemm = (mode == 0) or control

    samples, per_rank_all, res_spans, gemm_spans, walls = [], [], [], [], []
    for _ in range(rotations):
        wall_s, stamps, done = one_rotation(mod, fx, cfgs)
        walls.append(wall_s)
        rank_res, rank_gemm, rspan, gspan = [], [], [], []
        for r in range(WORLD):
            st, dn = stamps[r], done[r]
            res_lo, res_hi = 0, (res_ctas if conc else grid)
            gm_lo, gm_hi = ((res_ctas, mod.GRID_MAX) if conc else (0, grid if mode == 0 else 0))

            sp_res = role_span(st, res_lo, res_hi)
            units = sum(dn[res_lo:res_hi])
            rspan.append(sp_res)
            if mode == 0 or control or sp_res == 0 or units == 0:
                rank_res.append(None)
            else:
                byts = (9 * geo.b_strip_rows * geo.BN * 2 * units) if mode == 1 \
                       else (geo.WIN_BYTES * units)
                rank_res.append(byts / (sp_res / TICK_HZ) / 1e9)

            sp_gm = role_span(st, gm_lo, gm_hi) if gm_hi > gm_lo else 0
            units_gm = sum(dn[gm_lo:gm_hi]) if gm_hi > gm_lo else 0
            gspan.append(sp_gm)
            if sp_gm == 0 or units_gm == 0:
                rank_gemm.append(None)
            else:
                flops = 2 * geo.BM * geo.BN * geo.a_cols * units_gm
                rank_gemm.append(flops / (sp_gm / TICK_HZ) / 1e12)

        res_spans.append(statistics.median(rspan) if rspan else 0)
        gemm_spans.append(statistics.median([s for s in gspan if s] or [0]))
        chosen = rank_gemm if metric_is_gemm else rank_res
        vals = [v for v in chosen if v is not None]
        samples.append(statistics.median(vals) if vals else 0.0)
        per_rank_all.append({"resource": rank_res, "gemm_tflops": rank_gemm})

    value = statistics.median(samples)
    mid = samples.index(sorted(samples)[len(samples) // 2])
    gemm_vals = [v for v in per_rank_all[mid]["gemm_tflops"] if v is not None]
    gemm_med = statistics.median(gemm_vals) if gemm_vals else None
    work = geo.a_work if mode == 0 else (geo.b_work if mode == 1 else geo.c_work)
    return {
        "value": value,
        "samples": samples,
        "per_rank": per_rank_all[mid],
        "concurrent_gemm_tflops": gemm_med if conc else None,
        "gemm_tflops": gemm_med,
        "res_span_ticks": statistics.median(res_spans),
        "gemm_span_ticks": statistics.median(gemm_spans),
        "host_wall_ms": statistics.median(walls) * 1e3,
        "rounds": (work + ctas - 1) // ctas,
    }


# ---------------------------------------------------------------------------
# Destination-side checksum for one mode-c configuration.
# ---------------------------------------------------------------------------
def checksum_mode_c(mod, fx, geo, ctas, depth, fanout_name, protocol, overlay, release_group,
                    deadline_ticks):
    fanout = FANOUTS[fanout_name]
    # Which windows land in slot s on rank d, exactly as sat_mode_c addresses them.
    plan = {}
    for d in range(WORLD):
        for s in range(WORLD):
            if s == d:
                continue
            if fanout == 0:
                count = geo.c_work if (s + 1) % WORLD == d else 0
            else:
                r0 = (d - s - 1) % WORLD
                count = 0 if r0 > 6 else max(0, (geo.c_work - r0 + 6) // 7)
            if count:
                plan.setdefault(d, []).append((s, count))

    # Poison exactly the extents that will be written, with a value the pattern never makes.
    for d, entries in plan.items():
        for s, count in entries:
            rows = ((count + geo.c_cols_per_row - 1) // geo.c_cols_per_row) * geo.WIN_ROWS
            rows = min(rows, geo.c_slot_rows)
            mod.fill_region(d, fx.heap[d] + (s * geo.c_slot_elems + 0) * 2, rows * geo.N, 1)

    conc = overlay in ("concurrent", "reserve_control")
    control = overlay == "reserve_control"
    grid = mod.GRID_MAX if conc else ctas
    cfgs = [fx.cfg(r, 2, conc, control, grid, ctas, depth, fanout, protocol,
                   release_group, deadline_ticks) for r in range(WORLD)]
    one_rotation(mod, fx, cfgs)

    bad_total, fold = 0, 0
    for d, entries in plan.items():
        for s, count in entries:
            mism, f = mod.verify_slot(d, fx.heap[d], geo.c_slot_elems, s, geo.N,
                                      geo.c_cols_per_row, count)
            bad_total += int(mism)
            fold ^= int(f)
    return bad_total == 0, bad_total, fold


# ---------------------------------------------------------------------------
# Tick-rate calibration. Two stages, because the rate is the unknown: the sibling's
# 100 MHz is a gfx950 statement (FIGURE_SPECS.md section 5.7), not an input here.
# ---------------------------------------------------------------------------
TICK_HZ = 0.0


def calibrate(mod, reps=5):
    global TICK_HZ
    ticks, wall_ns = mod.calibrate_ticks(0, 1_000_000)
    rough = ticks / (wall_ns / 1e9)
    target = int(rough * 0.2)
    rates = []
    for _ in range(reps):
        ticks, wall_ns = mod.calibrate_ticks(0, target)
        rates.append(ticks / (wall_ns / 1e9))
    TICK_HZ = statistics.median(rates)
    spread = (max(rates) - min(rates)) / min(rates) * 100.0
    print(f"[calib] s_memrealtime = {TICK_HZ/1e6:.4f} MHz "
          f"({1e9/TICK_HZ:.3f} ns/tick), spread {spread:.3f}% over {reps} reps "
          f"(rough first pass {rough/1e6:.3f} MHz)", flush=True)
    return TICK_HZ, spread, rates


# ---------------------------------------------------------------------------
def ceilings(mod, args):
    props = mod.device_props(0)
    out = {
        "hbm_peak_gbps": {
            "value": props["hbm_peak_gbps_from_props"],
            "source": ("hipGetDeviceProperties on this node: "
                       f"2 x memoryClockRate({props['memory_clock_rate_khz']} kHz) x "
                       f"memoryBusWidth({props['memory_bus_width_bits']} bit) / 8"),
        },
        "mfma_bf16_peak_tflops": {
            "value": None,
            "source": ("derived in result.md from THIS node's pinned sclk x 304 CU x the gfx942 "
                       "per-CU bf16 rate implied by v_mfma_f32_16x16x16_bf16; sclk taken from "
                       "the clocks_before field, not from a datasheet"),
        },
        "xgmi": {
            "value": None,
            "source": "see ceilings.txt: rocm-smi --shownodesbw / --showtopo / amd-smi",
            "raw": {},
        },
        "note": ("MI350X figures from the sibling spec (76.8 / 537.6 GB/s per-link and aggregate "
                 "xGMI, 8 TB/s HBM3E, 148 GB/s service rate) are gfx950 facts and are deliberately "
                 "absent here. If this node reports no per-link xGMI number, result.md uses the "
                 "measured plateau as the empirical ceiling and says so."),
    }
    for name, cmd in (("shownodesbw", "rocm-smi --shownodesbw"),
                      ("showtopo", "rocm-smi --showtopo"),
                      ("showtopoweight", "rocm-smi --showtopoweight"),
                      ("showbw", "rocm-smi --showbw"),
                      ("showmclk", "rocm-smi --showmclk"),
                      ("amdsmi_static", "amd-smi static"),
                      ("amdsmi_metric", "amd-smi metric")):
        if args.skip_smi:
            out["xgmi"]["raw"][name] = "<skipped>"
        else:
            out["xgmi"]["raw"][name] = sh(cmd)[:8000]
    with open(os.path.join(HERE, "ceilings.txt"), "w") as fh:
        for k, v in out["xgmi"]["raw"].items():
            fh.write(f"########## {k} ##########\n{v}\n\n")
        fh.write(f"########## hipDeviceProp_t (device 0) ##########\n{json.dumps(props, indent=2)}\n")
    return out, props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "saturation.json"))
    ap.add_argument("--modes", default="abc")
    ap.add_argument("--rotations", type=int, default=5)
    ap.add_argument("--warmup-ms", type=float, default=400.0)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k-iters", type=int, default=116)      # K = 3712, K_TAIL == false
    ap.add_argument("--a-work", type=int, default=12160)     # 304 x 40 tiles
    ap.add_argument("--b-work", type=int, default=50688)     # ~59.8 GB touched
    ap.add_argument("--c-work", type=int, default=32768)     # 512 MB pushed per rank per launch
    ap.add_argument("--red-slot-rows", type=int, default=4096)
    ap.add_argument("--release-group", type=int, default=4)  # the shipped constant
    ap.add_argument("--payload-coarse", action="store_true",
                    help="coarse-grained (cached) mode-c destination: the production-parity "
                         "cross-check. Default is fine-grained, so the no-protocol isolated arm "
                         "cannot report L2 bandwidth as fabric bandwidth (plan.md 8.8).")
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--skip-smi", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 1 rotation, 3 CTA points per mode, depth {0,1}, single fanout")
    args = ap.parse_args()

    mod = load_module()
    print(f"[node] {mod.device_count()} devices visible", flush=True)
    # Peer access FIRST, always: without it a kernel on device d cannot dereference a pointer
    # owned by device p even inside one process.
    peer = mod.enable_peer_access(WORLD)
    print(f"[peer] {peer}", flush=True)
    if peer["failures"]:
        sys.exit(f"peer access incomplete: {peer['failures']}")

    ceil, props = ceilings(mod, args)
    tick_hz, tick_spread, tick_reps = calibrate(mod)

    ctas_a, ctas_b, ctas_c, depths, fanouts = CTAS_A, CTAS_B, CTAS_C, DEPTHS, list(FANOUTS)
    if args.quick:
        args.rotations = 1
        args.warmup_ms = 50.0
        args.a_work = max(304, args.a_work // 16)
        args.b_work = max(512, args.b_work // 16)
        args.c_work = max(1024, args.c_work // 8)
        ctas_a, ctas_b, ctas_c = [32, 152, 304], [8, 64, 304], [1, 8, 64]
        depths, fanouts = [0, 1], ["single"]

    geo = Geometry(mod, args)
    print(f"[geom] {json.dumps(geo.summary())}", flush=True)
    fx = Fixture(mod, geo, payload_fine=not args.payload_coarse)

    points = []
    t_start = time.time()

    def record(mode, overlay, ctas, depth, fanout, protocol, res, extra=None):
        rec = {
            "mode": "abc"[mode], "ctas": ctas,
            "depth": depth if mode == 2 else None,
            "fanout": fanout if mode == 2 else None,
            "overlay": overlay,
            "protocol": bool(protocol),
            # the reserve control has no traffic, so its reported value is the GEMM role's TFLOPS
            "metric": "TFLOPS" if (mode == 0 or overlay == "reserve_control") else "GBps",
            "payload_granularity": "coarse" if args.payload_coarse else "fine",
            "value": res["value"], "samples": res["samples"], "per_rank": res["per_rank"],
            "concurrent_gemm_tflops": res["concurrent_gemm_tflops"],
            "gemm_tflops": res["gemm_tflops"],
            "res_span_us": res["res_span_ticks"] / tick_hz * 1e6,
            "gemm_span_us": res["gemm_span_ticks"] / tick_hz * 1e6,
            "host_wall_ms": res["host_wall_ms"],
            "rounds": res["rounds"],
            "checksum_ok": None, "checksum_mismatches": None, "checksum_fold": None,
        }
        if extra:
            rec.update(extra)
        points.append(rec)
        print(f"  [{rec['mode']}] C={ctas:<4} d={rec['depth']} f={rec['fanout']} "
              f"{overlay:<15} proto={rec['protocol']!s:<5} "
              f"{rec['value']:10.2f} {rec['metric']:<7} "
              f"span={rec['res_span_us']/1e3:8.2f} ms "
              f"gemm={('%.1f' % rec['gemm_tflops']) if rec['gemm_tflops'] else '-':>7} TF"
              f"{'  chk=' + str(rec['checksum_ok']) if rec['checksum_ok'] is not None else ''}",
              flush=True)
        return rec

    # ---- mode a: isolated only (the sibling's overlay covers b and c) ----
    if "a" in args.modes:
        print("\n=== mode a (MFMA) ===", flush=True)
        for c in ctas_a:
            cb = clocks_snapshot()
            res = measure(mod, fx, geo, 0, "isolated", c, 0, "single", 0,
                          args.rotations, args.warmup_ms, args.release_group)
            rec = record(0, "isolated", c, 0, "single", 0, res)
            rec["clocks_before"], rec["clocks_after"] = cb, clocks_snapshot()

    # ---- mode b: isolated, concurrent, C-matched reserve control ----
    if "b" in args.modes:
        print("\n=== mode b (HBM, REDV=1 reduce) ===", flush=True)
        for c in ctas_b:
            cb = clocks_snapshot()
            res = measure(mod, fx, geo, 1, "isolated", c, 0, "single", 0,
                          args.rotations, args.warmup_ms, args.release_group)
            rec = record(1, "isolated", c, 0, "single", 0, res)
            rec["clocks_before"], rec["clocks_after"] = cb, clocks_snapshot()

            if c < mod.GRID_MAX:
                cb = clocks_snapshot()
                live = measure(mod, fx, geo, 1, "concurrent", c, 0, "single", 0,
                               args.rotations, args.warmup_ms, args.release_group)
                rlive = record(1, "concurrent", c, 0, "single", 0, live)
                rlive["clocks_before"], rlive["clocks_after"] = cb, clocks_snapshot()

                cb = clocks_snapshot()
                ctl = measure(mod, fx, geo, 1, "reserve_control", c, 0, "single", 0,
                              args.rotations, args.warmup_ms, args.release_group,
                              deadline_ticks=int(live["res_span_ticks"]))
                rctl = record(1, "reserve_control", c, 0, "single", 0, ctl)
                rctl["clocks_before"], rctl["clocks_after"] = cb, clocks_snapshot()
                if live["concurrent_gemm_tflops"] and ctl["concurrent_gemm_tflops"]:
                    rlive["gemm_slowdown"] = (ctl["concurrent_gemm_tflops"]
                                              / live["concurrent_gemm_tflops"])

    # ---- mode c: the money panel ----
    if "c" in args.modes:
        print("\n=== mode c (xGMI 16 B peer-packet emit) ===", flush=True)
        for c in ctas_c:
            ref_span = None
            for fan in fanouts:
                for d in depths:
                    for proto in (0, 1):
                        for overlay in ("isolated", "concurrent"):
                            cb = clocks_snapshot()
                            res = measure(mod, fx, geo, 2, overlay, c, d, fan, proto,
                                          args.rotations, args.warmup_ms, args.release_group)
                            rec = record(2, overlay, c, d, fan, proto, res)
                            rec["clocks_before"] = cb
                            rec["clocks_after"] = clocks_snapshot()
                            if (overlay == "concurrent" and d == depths[0]
                                    and fan == fanouts[0] and proto == 0):
                                ref_span = int(res["res_span_ticks"])
                            if not args.no_checksum:
                                ok, bad, fold = checksum_mode_c(
                                    mod, fx, geo, c, d, fan, proto, overlay,
                                    args.release_group, ref_span or 0)
                                rec["checksum_ok"] = ok
                                rec["checksum_mismatches"] = bad
                                rec["checksum_fold"] = fold
                                if not ok:
                                    print(f"    !! CHECKSUM FAILED: {bad} mismatched elements",
                                          flush=True)
            if ref_span:
                cb = clocks_snapshot()
                ctl = measure(mod, fx, geo, 2, "reserve_control", c, 0, fanouts[0], 0,
                              args.rotations, args.warmup_ms, args.release_group,
                              deadline_ticks=ref_span)
                rctl = record(2, "reserve_control", c, 0, fanouts[0], 0, ctl)
                rctl["clocks_before"], rctl["clocks_after"] = cb, clocks_snapshot()
                for rec in points:
                    if (rec["mode"] == "c" and rec["ctas"] == c
                            and rec["overlay"] == "concurrent"
                            and rec["concurrent_gemm_tflops"]
                            and ctl["concurrent_gemm_tflops"]):
                        rec["gemm_slowdown"] = (ctl["concurrent_gemm_tflops"]
                                                / rec["concurrent_gemm_tflops"])

    out = {
        "experiment": "exp_21_saturation",
        "figure": "paper Fig 2 / Q4(a) -- NanoFlow v1 Fig 7 analog, GEMM-RS instance",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_minutes": (time.time() - t_start) / 60.0,
        "node": {
            "host": sh("hostname"),
            "arch": props["gcn_arch_name"],
            "cu_count": props["multi_processor_count"],
            "device_props": props,
            "world": WORLD,
            "topology_caveat": ("one process drives 8 devices with hipDeviceEnablePeerAccess, "
                                "NOT the evaluator's one-process-per-rank HIP-IPC topology"),
            "clocks_pinned": "tools/set_clocks.sh pin 1900 (standing rule on this node)",
        },
        "ceilings": ceil,
        "tick_rate_hz": tick_hz,
        "tick_rate_spread_pct": tick_spread,
        "tick_rate_samples_hz": tick_reps,
        "geometry": geo.summary(),
        "config": vars(args),
        "schema": {
            "points[]": ("mode a|b|c, ctas, depth (mode c; 0 = unbounded in-flight bound = the "
                         "production shape), fanout single|rr7, overlay "
                         "isolated|concurrent|reserve_control, protocol bool, metric "
                         "TFLOPS|GBps, value = median over rotations of the per-rank median, "
                         "samples = one per rotation, per_rank = {resource, gemm_tflops} from the "
                         "median rotation, concurrent_gemm_tflops, gemm_slowdown = "
                         "control/live TFLOPS on the same 304-C CTAs, res_span_us and gemm_span_us "
                         "from s_memrealtime, rounds = ceil(work/ctas), checksum_ok, "
                         "checksum_mismatches, checksum_fold, payload_granularity, "
                         "clocks_before, clocks_after"),
        },
        "points": points,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[done] {len(points)} points -> {args.out} "
          f"({(time.time()-t_start)/60.0:.1f} min)", flush=True)
    fx.close()


if __name__ == "__main__":
    main()
