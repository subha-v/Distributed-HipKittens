#!/usr/bin/env python3
"""exp_22 arm (a): the reference GEMM+RCCL baseline, one process per rank.

This replicates the reference submission the evaluator ships -- exp026's
`submission.py`, which `tools/run_reference_arm.sh` stages -- namely a per-rank
`torch.matmul(x, w.T)` (+ bias) in bf16 followed by
`torch.distributed.reduce_scatter_tensor`.  Inputs follow the evaluator's
`generate_input` semantics verbatim (the same code `harness/harness_lib.py`
copied), so the seed axis is shared with arm (b).

Why a standalone driver rather than a rocprofv3 wrap of eval.py: rank-0-only
profiling is trivial here, the evaluator's ~92 us/call of harness machinery
does not appear in the trace as gaps belonging to no kernel, and the figure
needs a clean repeated single-epoch window.  Fidelity is checked, not assumed:
the per-call device time printed below is compared in result.md against
run_reference_arm.sh's shape-5 benchmark number.

Run under torchrun with 8 ranks; see b0_capture.sh.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def generate_input(rank, world_size, m, n, k, has_bias, seed):
    """Verbatim from the evaluator's reference.py (via harness/harness_lib.py)."""
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert k % world_size == 0, "k must be divisible by world_size"
    local_k = k // world_size

    x = (torch.rand((m, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01
    w = (torch.rand((n, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01

    bias = None
    if has_bias:
        gen.manual_seed(seed)
        bias = (torch.rand((n,), dtype=torch.bfloat16, device=device,
                           generator=gen) * 2 - 1) * 0.01
    return x, w, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=14336)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7168)
    ap.add_argument("--warmup-ms", type=float, default=800.0,
                    help="duration-based, not iteration-based: idle sclk on "
                         "this node is ~132 MHz against ~1900 under load")
    ap.add_argument("--epochs", type=int, default=12,
                    help="measured epochs; the timeline uses one of them")
    ap.add_argument("--out", default=None, help="per-rank boundary json")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    x, w, bias = generate_input(rank, world, args.m, args.n, args.k,
                                bool(args.bias), args.seed)
    rows = args.m // world
    out = torch.empty((rows, args.n), dtype=torch.bfloat16,
                      device=f"cuda:{rank}")

    def epoch():
        partial = torch.matmul(x, w.T)
        if bias is not None:
            partial = partial + bias
        dist.reduce_scatter_tensor(out, partial.contiguous())

    # Warm up by duration, then re-sync the whole world so the measured epochs
    # start from a common point on every rank.
    start = time.perf_counter()
    while (time.perf_counter() - start) * 1e3 < args.warmup_ms:
        for _ in range(4):
            epoch()
        torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    # Measured region: each epoch is bracketed by a full device sync so the
    # trace segments unambiguously into epochs without any clock correlation.
    boundaries, device_us = [], []
    evt_a = torch.cuda.Event(enable_timing=True)
    evt_b = torch.cuda.Event(enable_timing=True)
    for _ in range(args.epochs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
        evt_a.record()
        epoch()
        evt_b.record()
        torch.cuda.synchronize()
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
        boundaries.append([t0, t1])
        device_us.append(evt_a.elapsed_time(evt_b) * 1e3)

    device_us.sort()
    median = device_us[len(device_us) // 2]
    print(f"REF rank={rank} m={args.m} n={args.n} k={args.k} "
          f"epochs={args.epochs} device_us_median={median:.2f} "
          f"min={device_us[0]:.2f} max={device_us[-1]:.2f}", flush=True)

    if args.out and rank == 0:
        with open(args.out, "w") as handle:
            json.dump({
                "schema": "exp22.b0_boundaries.v1",
                "shape": {"m": args.m, "n": args.n, "k": args.k,
                          "has_bias": bool(args.bias), "seed": args.seed},
                "epochs": args.epochs,
                "monotonic_boundaries_s": boundaries,
                "device_us_sorted": device_us,
                "device_us_median": median,
            }, handle, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
