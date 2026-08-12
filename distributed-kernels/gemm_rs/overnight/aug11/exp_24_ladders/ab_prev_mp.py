#!/usr/bin/env python3
"""The decisive test: `ours` (PERSHAPE=2) against `ours_prev` (PERSHAPE=0),
same-run paired, inside instrument A's own 8-process pool.

Why this exists. exp_26 measured shape 5 at -6.56% pipelined in 80 of 80 paired
rounds; exp_24's re-measure, comparing two ladder RUNS six hours apart, read
+2.87%. A ~9.5-point disagreement including a sign flip cannot be settled by
preferring one instrument, and the cross-run comparison is known to be
drift-contaminated (on the small shapes every arm inflated together, the no-op
floor kernel by +19-29%). So put both binaries in ONE pool and pair them
within-round, which is the design both measurements agree is the strongest.

What is reused verbatim, and why that matters: every timing region, the warmup,
the input generation and the oracle are imported from `ladder_mp` rather than
reimplemented, so this is literally instrument A's measurement with a sixth arm,
not a look-alike. The one structural difference from exp_26 is deliberate and is
the hypothesis under test: exp_26's `ab_pershape.py` drives 8 devices from ONE
process, while instrument A is 8 processes under torch.distributed -- closer to
the evaluator's topology, and a plausible place for a release-granularity effect
to change size or sign.

Three arms, three separate .so files built from one source (`build_ab_arms.sh`):

    ours       gemm_rs_mi300x_ab2   PERSHAPE=2   the shipped rule
    ours_prev  gemm_rs_mi300x_ab0   PERSHAPE=0   the pre-exp_26 rule
    ours_null  gemm_rs_mi300x_ab2b  PERSHAPE=2   a second build of `ours`

`ours_null` is what makes `ours_prev` interpretable: nothing an instruction can
see separates it from `ours`, so whatever this instrument reports between them is
its own bias, and no `ours_prev` delta smaller than that means anything.

ORDERING. The tree's inherited scheme -- `order[i] = arms[(rep + i) mod n]` --
is defective: every arm is first exactly once, but the RELATIVE offset of every
arm PAIR is the constant `(j - k) mod n` in every rep, so a neighbour effect
becomes a fixed offset on that pair rather than noise. That single defect
explains exp_05's false positive, exp_23's twin-pair result and exp_24's own
inflated null floors. With only three arms the fix can be exact rather than
statistical: enumerate ALL 3! = 6 permutations as one block and run whole blocks,
shuffling the block's internal sequence. Within every block each ordered pair
(j, k) has j before k in exactly 3 of the 6 permutations and each relative offset
appears equally often, so the offsets are balanced BY CONSTRUCTION -- the
Latin-square property, complete rather than in expectation.

STATISTIC. The within-round paired median, not best-of-pass. Best-of-pass
compares two arms' luckiest rounds from different draws, and it is the statistic
that misled both instruments. Reported as the median over rounds of the per-round
delta, plus wins-out-of-rounds, plus a sign test.

Usage: ab_prev_mp.py <shape_index> <rounds> <graded_iters> <pipe_iters> <burst> <port>
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

D = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders"
ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
HARNESS = f"{ON}/harness"
sys.path.insert(0, D)
sys.path.insert(0, HARNESS)

import ladder_mp as L  # noqa: E402  -- the instrument itself

WORLD = L.WORLD
SCORED = L.SCORED

# arm -> (kernel module name, PERSHAPE it was compiled with)
ARMS = {
    "ours":      ("gemm_rs_mi300x_ab2",  2),
    "ours_prev": ("gemm_rs_mi300x_ab0",  0),
    "ours_null": ("gemm_rs_mi300x_ab2b", 2),
}
ARM_ORDER_FWD = ["ours", "ours_prev", "ours_null"]
RELEASE_GROUP = 4


def expected_rgroup(pershape, ppc):
    """What each arm's own compiled rule selects at this tiles-per-CTA.

    PERSHAPE=0 keeps rgroup in the two-element set {RELEASE_GROUP, 1};
    PERSHAPE=2 makes it a member of the literal ladder {RELEASE_GROUP, 2, 1}.
    The arms are only distinct where these disagree, which is ppc == 2 or 3.
    """
    if ppc >= RELEASE_GROUP:
        return RELEASE_GROUP
    if pershape == 2 and ppc >= 2:
        return 2
    return 1


def load_arm(key):
    """Load harness/submission.py under its own name, bound to its own kernel.

    submission.py resolves HK_KERNEL_MODULE at IMPORT time, so the env var is set
    around each import; that is what lets three differently-compiled kernels live
    in one process.
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
    """Complete-permutation blocks: all n! orders per block, block order shuffled.

    Identical on every rank -- seeded from the shape alone, never from rank or the
    wall clock, because the ranks must walk the same sequence.
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

        # ---- geometry and each arm's own rgroup, before anything is timed ----
        # Reuse the runtime the arms themselves imported. Loading dhk_rt.so under
        # a fresh name fails outright -- an extension module only exports
        # PyInit_<its own name> -- which is also why each arm needed its own
        # separately-named .so rather than one file imported three times.
        rt = modules[keys[0]].rt
        plan = rt.resolve_shape(m, n, k,
                               True if os.environ.get("VS_FORCE_BIAS") == "1"
                               else has_bias)
        tiles = int(plan["gemm_tiles"])
        producers = int(plan["num_gemm_ctas"])
        ppc = -(-tiles // producers)
        geometry = {"config_row": int(plan["config_row"]), "tiles": tiles,
                    "producers": producers, "tiles_per_cta": ppc,
                    "num_reducer_ctas": int(plan["num_reducer_ctas"])}
        rgroups = {key: expected_rgroup(ARMS[key][1], ppc) for key in keys}
        arms_distinct = rgroups["ours"] != rgroups["ours_prev"]
        say(f"  geometry: {tiles} tiles / {producers} producers = {ppc} per CTA")
        say(f"  rgroup per arm: {rgroups}  distinct={arms_distinct}")

        # ---- warm every arm before any arm is timed ----
        outs = {}
        for key in keys:
            outs[key], calls = L.warm(impls[key], data, warm_ms, 20, rank)
            say(f"  warm {key}: {calls} calls")

        # ---- correctness, then bit-exactness between the arms ----
        want = L.oracle(data, rank, m)
        correctness = {key: L.check(outs[key], want) for key in keys}
        for key in keys:
            say(f"    {key:>10} {correctness[key]}")
        bad = [k for k, c in correctness.items() if not c["allclose_1e-2"]]
        if bad:
            raise RuntimeError(f"refusing to time: wrong at 1e-2: {bad}")

        # exp_26 established the arms are bit-identical on all 8 ranks. If that
        # ever fails they are not the same computation and no timing comparison
        # between them means anything -- so it is an assertion, not a note. The
        # verdict is all-reduced: one bad rank stops all eight.
        equal = {}
        base = outs["ours"]
        for key in keys:
            if key == "ours":
                continue
            same = bool(torch.equal(base, outs[key]))
            equal[f"ours_vs_{key}"] = same
        flag = torch.tensor([1 if all(equal.values()) else 0],
                            dtype=torch.int32, device=f"cuda:{rank}")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_ranks_equal = bool(int(flag.item()))
        say(f"  torch.equal {equal}  all_ranks={all_ranks_equal}")
        if not all_ranks_equal:
            raise RuntimeError(
                f"arms are NOT bit-identical (rank {rank}: {equal}); the two "
                f"arms are not the same computation, so timing them is void")

        # ---- the paired campaign ----
        orders, perms_per_block = block_orders(ARM_ORDER_FWD, rounds,
                                               0xAB27 + shape_index)
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
                    # One round contributes ONE number per arm per protocol: its
                    # median. Pairing is by round, so the round is the unit.
                    per_round[key][proto].append(statistics.median(got))
                    post = L.check(out, want)
                    if not post["allclose_1e-2"]:
                        raise RuntimeError(f"{key} wrong after {proto}: {post}")
            if rank == 0 and (r + 1) % 6 == 0:
                a = statistics.median(per_round["ours"]["pipelined"])
                b = statistics.median(per_round["ours_prev"]["pipelined"])
                say(f"  round {r+1}/{rounds}: pipelined median ours={a:.2f} "
                    f"ours_prev={b:.2f} ({(a - b) / b * 100:+.2f}%)")

        result = {
            "shape_index": shape_index, "shape": [m, n, k, int(has_bias)],
            "shape_label": f"{m}x{n}x{k}",
            "rounds": rounds, "graded_iters": giters,
            "pipe_iters": piters, "burst": burst,
            "alloc_order": keys, "arm_orders": orders,
            "perms_per_block": perms_per_block,
            "ordering": "complete-permutation blocks (balanced by construction)",
            "geometry": geometry, "rgroup": rgroups,
            "arms_distinct_on_this_shape": arms_distinct,
            "torch_equal": equal, "all_ranks_bit_identical": all_ranks_equal,
            "correctness": correctness,
            "kernel_modules": {k: ARMS[k][0] for k in keys},
            "pershape": {k: ARMS[k][1] for k in keys},
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
    # Exit NONZERO on failure. The smoke run died in every rank on an import
    # error and still reported `exit codes: [0]*8`, because catching the
    # exception to write a diagnostic JSON also swallowed the process status --
    # so the runner's all-zero check would have called a total failure a pass.
    if result.get("error"):
        os._exit(1)


def main():
    shape_index = int(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    giters = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    piters = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    burst = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    port = int(sys.argv[6]) if len(sys.argv) > 6 else 13300

    order = os.environ.get("AB_ORDER", "fwd")
    alloc = ARM_ORDER_FWD if order == "fwd" else list(reversed(ARM_ORDER_FWD))
    out_path = os.environ.get("AB_OUT", f"ab_shape{shape_index}_{order}")
    print(f"=== paired ours/ours_prev, shape[{shape_index}]="
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
