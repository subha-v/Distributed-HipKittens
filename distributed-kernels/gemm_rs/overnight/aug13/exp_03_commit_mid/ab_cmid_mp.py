#!/usr/bin/env python3
"""exp_03 M7: `cmid` (COMMIT_MID=1) against `base` (COMMIT_MID=0), same-run
paired, inside instrument A's own 8-process pool.

This is exp_24's `ab_prev_mp.py` with the arms swapped — the strongest paired
design this project has: both binaries in one pool, within-round pairing,
complete-permutation (Latin-square) ordering so every arm-pair offset is
balanced by construction, bit-exactness asserted on every rank before any
timing, and the whole campaign run in BOTH allocation orders (AB_ORDER=fwd and
rev) because construction order alone fabricated a p~1e-101 result in exp_26.

Arms, three .so files from one source (build_arms.sh):

    cmid   gemm_rs_mi300x_cmid    COMMIT_MID=1  the candidate (commit between
                                                the MFMA halves)
    base   gemm_rs_mi300x_cmid0   COMMIT_MID=0  the incumbent tail commit
    null   gemm_rs_mi300x_cmid0b  COMMIT_MID=0  second build of `base`

`null` is what makes `cmid` interpretable: nothing an instruction can see
separates it from `base`, so whatever this instrument reports between them is
its own bias, and no `cmid` delta smaller than that contrast means anything.

The arms MUST be bit-identical in output: the edit moves stores and waits, not
arithmetic. A torch.equal failure on any rank voids the comparison outright.

STATISTIC: within-round paired median + wins-out-of-rounds, per protocol
(graded and pipelined), exactly as ab_prev_mp reports.

Usage: ab_cmid_mp.py <shape_index> <rounds> <graded_iters> <pipe_iters> <burst> <port>
Env:   AB_ORDER=fwd|rev  arm construction (allocation) order
       AB_OUT            output path prefix
"""
import itertools
import json
import os
import random
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

D24 = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders"
ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
HARNESS = f"{ON}/harness"
sys.path.insert(0, D24)
sys.path.insert(0, HARNESS)

import ladder_mp as L  # noqa: E402  -- the instrument itself

WORLD = L.WORLD
SCORED = L.SCORED

# arm -> (kernel module name, COMMIT_MID it was compiled with)
ARMS = {
    "cmid": ("gemm_rs_mi300x_cmid",   1),
    "base": ("gemm_rs_mi300x_cmid0",  0),
    "null": ("gemm_rs_mi300x_cmid0b", 0),
}
ARM_ORDER_FWD = ["cmid", "base", "null"]


def load_arm(key):
    """Load harness/submission-equivalent bound to this arm's kernel module.

    hk_submission resolves HK_KERNEL_MODULE at IMPORT time, so the env var is
    set around each import; that is what lets three differently-compiled
    kernels live in one process.
    """
    mod_name, _ = ARMS[key]
    prev = os.environ.get("HK_KERNEL_MODULE")
    os.environ["HK_KERNEL_MODULE"] = mod_name
    try:
        return L.load_module(f"{key}_submission", f"{HARNESS}/submission.py")
    finally:
        if prev is None:
            os.environ.pop("HK_KERNEL_MODULE", None)
        else:
            os.environ["HK_KERNEL_MODULE"] = prev


def block_orders(arms, rounds, seed):
    """Complete-permutation blocks: all n! orders per block, blocks shuffled.

    Identical on every rank -- seeded from the shape alone, never from rank or
    the wall clock, because the ranks must walk the same sequence.
    """
    perms = [list(p) for p in itertools.permutations(arms)]
    rng = random.Random(seed)
    out = []
    while len(out) < rounds:
        block = list(perms)
        rng.shuffle(block)
        out.extend(block)
    return out[:rounds], len(perms)


def _worker(rank, shape_index, rounds, giters, piters, burst, port, out_path,
            alloc_order):
    err = open(f"{out_path}.rank{rank}.stderr", "wb", 0)
    os.dup2(err.fileno(), 2)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("HK_DEBUG", "0")
    warm_ms = float(os.environ.get("AB_WARM_MS", "400"))

    def say(msg):
        if rank == 0:
            print(f"[rank0] {msg}", flush=True)

    torch.cuda.set_device(rank)
    keys = list(alloc_order)
    modules = {key: load_arm(key) for key in keys}   # allocation epoch order
    dist.init_process_group("nccl", init_method="env://", rank=rank,
                            world_size=WORLD,
                            device_id=torch.device(f"cuda:{rank}"))
    result = {}
    try:
        m, n, k, has_bias, seed = SCORED[shape_index]
        say(f"--- shape {shape_index + 1}: {m}x{n}x{k} alloc_order={keys} "
            f"rounds={rounds} graded={giters} pipe={piters}x{burst}")
        data = L.generate_input(rank, WORLD, m, n, k, has_bias, seed)
        impls = {key: modules[key].custom_kernel for key in keys}

        rt = modules[keys[0]].rt
        plan = rt.resolve_shape(m, n, k, has_bias)
        geometry = {"config_row": int(plan["config_row"]),
                    "tiles": int(plan["gemm_tiles"]),
                    "num_gemm_ctas": int(plan["num_gemm_ctas"]),
                    "num_reducer_ctas": int(plan["num_reducer_ctas"])}
        say(f"  geometry: {geometry}")

        # ---- warm every arm before any arm is timed ----
        outs = {}
        for key in keys:
            outs[key], calls = L.warm(impls[key], data, warm_ms, 20, rank)
            say(f"  warm {key}: {calls} calls")

        # ---- correctness, then bit-exactness between the arms ----
        want = L.oracle(data, rank, m)
        correctness = {key: L.check(outs[key], want) for key in keys}
        for key in keys:
            say(f"    {key:>6} {correctness[key]}")
        bad = [k2 for k2, c in correctness.items() if not c["allclose_1e-2"]]
        if bad:
            raise RuntimeError(f"refusing to time: wrong at 1e-2: {bad}")

        # The edit moves stores and waits, never arithmetic, so the arms must
        # be bit-identical on every rank; anything else means the edit changed
        # the computation and timing it is void. All-reduced: one bad rank
        # stops all eight.
        equal = {}
        base_out = outs["base"]
        for key in keys:
            if key == "base":
                continue
            equal[f"base_vs_{key}"] = bool(torch.equal(base_out, outs[key]))
        flag = torch.tensor([1 if all(equal.values()) else 0],
                            dtype=torch.int32, device=f"cuda:{rank}")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_ranks_equal = bool(int(flag.item()))
        say(f"  torch.equal {equal}  all_ranks={all_ranks_equal}")
        if not all_ranks_equal:
            raise RuntimeError(
                f"arms are NOT bit-identical (rank {rank}: {equal}); the edit "
                f"changed the computation, timing is void")

        # ---- the paired campaign ----
        orders, perms_per_block = block_orders(ARM_ORDER_FWD, rounds,
                                               0xC31D + shape_index)
        say(f"  ordering: complete blocks of {perms_per_block} permutations, "
            f"{rounds} rounds")
        per_round = {key: {"graded": [], "pipelined": []} for key in keys}
        for r in range(rounds):
            order = orders[r]
            protos = (("graded", "pipelined") if r % 2 == 0
                      else ("pipelined", "graded"))
            for proto in protos:
                for key in order:
                    if proto == "graded":
                        got, out = L.time_graded(impls[key], data, giters)
                    else:
                        got, out = L.time_pipelined(impls[key], data, piters,
                                                    burst)
                    per_round[key][proto].append(statistics.median(got))
                    post = L.check(out, want)
                    if not post["allclose_1e-2"]:
                        raise RuntimeError(f"{key} wrong after {proto}: {post}")
            if rank == 0 and (r + 1) % 6 == 0:
                a = statistics.median(per_round["cmid"]["pipelined"])
                b = statistics.median(per_round["base"]["pipelined"])
                say(f"  round {r+1}/{rounds}: pipelined median cmid={a:.2f} "
                    f"base={b:.2f} ({(a - b) / b * 100:+.2f}%)")

        result = {
            "shape_index": shape_index, "shape": [m, n, k, int(has_bias)],
            "shape_label": f"{m}x{n}x{k}",
            "rounds": rounds, "graded_iters": giters,
            "pipe_iters": piters, "burst": burst,
            "alloc_order": keys, "arm_orders": orders,
            "perms_per_block": perms_per_block,
            "ordering": "complete-permutation blocks (balanced by construction)",
            "geometry": geometry,
            "torch_equal": equal, "all_ranks_bit_identical": all_ranks_equal,
            "correctness": correctness,
            "kernel_modules": {k2: ARMS[k2][0] for k2 in keys},
            "commit_mid": {k2: ARMS[k2][1] for k2 in keys},
            "per_round": per_round,
            "is_rank0": rank == 0,
        }
    except Exception:
        result = {"shape_index": shape_index, "error": traceback.format_exc(),
                  "is_rank0": rank == 0}
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    with open(f"{out_path}.rank{rank}.json", "w") as h:
        json.dump(result, h, indent=2)
    # Exit NONZERO on failure (the ab_prev lesson: a swallowed exception must
    # not report exit 0).
    if result.get("error"):
        os._exit(1)


def main():
    shape_index = int(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    giters = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    piters = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    burst = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    port = int(sys.argv[6]) if len(sys.argv) > 6 else 13400

    order = os.environ.get("AB_ORDER", "fwd")
    alloc = ARM_ORDER_FWD if order == "fwd" else list(reversed(ARM_ORDER_FWD))
    out_path = os.environ.get("AB_OUT", f"ab_cmid_shape{shape_index}_{order}")
    print(f"=== paired cmid/base, shape[{shape_index}]="
          f"{SCORED[shape_index]} rounds={rounds} graded={giters} "
          f"pipe={piters}x{burst} alloc={alloc} port={port} ===", flush=True)

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(WORLD):
        p = ctx.Process(target=_worker,
                        args=(rank, shape_index, rounds, giters, piters, burst,
                              port, out_path, alloc))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 2


if __name__ == "__main__":
    sys.exit(main())
