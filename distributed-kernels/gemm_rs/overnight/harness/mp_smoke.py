"""Minimal multi-process reproducer for hk_submission, outside the evaluator.

The evaluator hides failures behind opaque 60 s / 180 s multiprocessing
timeouts, which makes each debug cycle minutes long and mostly uninformative.
This mimics its structure -- spawn 8 ranks, init_process_group("nccl"), call
custom_kernel repeatedly, destroy the group, do it again for a second shape --
but with full stderr, per-step logging and its own timeouts.

Usage:
  python3 mp_smoke.py            # two shapes, a few calls each
  python3 mp_smoke.py 512 4096 12288 1 5
"""

import os
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORLD = 8


def generate_input(rank, world_size, m, n, k, has_bias, seed):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
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
    return (x, w, bias)


def _worker(rank, cases, calls, port):
    import faulthandler
    faulthandler.enable()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("HK_DEBUG", "1")

    def say(message):
        print(f"[rank {rank}] {message}", flush=True)

    # Import order matters for diagnosis: loading the HIP pybind modules before
    # vs after torch/RCCL initializes the device is the first thing to rule out.
    order = os.environ.get("HK_IMPORT_ORDER", "before")
    if order == "before":
        say("importing submission BEFORE set_device/init_process_group")
        import submission  # noqa: F401
        say("import ok")

    say(f"set_device({rank})")
    torch.cuda.set_device(rank)
    say("set_device ok")

    if order != "before":
        say("importing submission AFTER set_device")
        import submission  # noqa: F401
        say("import ok")

    try:
        for index, (m, n, k, has_bias, seed) in enumerate(cases):
            # One process group per case, exactly like eval.py does.
            dist.init_process_group("nccl", init_method="env://", rank=rank,
                                    world_size=WORLD,
                                    device_id=torch.device(f"cuda:{rank}"))
            try:
                say(f"case {index}: m={m} n={n} k={k} bias={has_bias}")
                from submission import custom_kernel

                data = generate_input(rank, WORLD, m, n, k, has_bias, seed)
                for call in range(calls):
                    t0 = time.perf_counter()
                    out = custom_kernel(tuple(
                        t.clone() if torch.is_tensor(t) else t for t in data))
                    torch.cuda.synchronize(rank)
                    dt = (time.perf_counter() - t0) * 1e3
                    say(f"  call {call} ok  {dt:8.2f} ms")

                # Correctness against the evaluator's oracle.
                x, w, bias = data
                partial = torch.matmul(x, w.T)
                if bias is not None:
                    partial = partial + bias
                full = torch.empty_like(partial)
                dist.all_reduce(partial)
                full = partial
                rows = m // WORLD
                want = full[rank * rows:(rank + 1) * rows, :]
                ok = torch.allclose(out, want, rtol=2e-2, atol=2e-2)
                diff = (out.float() - want.float()).abs().max().item()
                say(f"  correctness allclose={ok} max|diff|={diff:.3e}")
            finally:
                dist.destroy_process_group()
        say("DONE")
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        raise


def main():
    if len(sys.argv) > 5:
        m, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        has_bias = bool(int(sys.argv[4]))
        calls = int(sys.argv[5])
        cases = [(m, n, k, has_bias, 663)]
    else:
        calls = 3
        cases = [
            (512, 4096, 12288, True, 663),
            (2048, 2880, 2880, True, 166),
        ]

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(WORLD):
        p = mp.Process(target=_worker, args=(rank, cases, calls, 12399))
        p.start()
        procs.append(p)

    deadline = time.time() + 600
    for p in procs:
        p.join(timeout=max(5, deadline - time.time()))
    alive = [i for i, p in enumerate(procs) if p.is_alive()]
    if alive:
        print(f"TIMED OUT, ranks still alive: {alive}")
        for p in procs:
            if p.is_alive():
                p.terminate()
        return 1
    codes = [p.exitcode for p in procs]
    print(f"exit codes: {codes}")
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
