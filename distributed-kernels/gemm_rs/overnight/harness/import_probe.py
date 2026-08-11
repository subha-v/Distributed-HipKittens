"""Isolate the dlopen segfault: plain process vs spawned worker, per module."""

import importlib.util
import os
import sys
import traceback

BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
MODULES = ["dhk_rt", "gemm_rs_mi300x"]


def load(name):
    path = os.path.join(BUILD, f"{name}.so")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_all(tag, import_torch_first):
    import faulthandler
    faulthandler.enable()
    print(f"[{tag}] torch_first={import_torch_first}", flush=True)
    if import_torch_first:
        import torch
        print(f"[{tag}] torch {torch.__version__} imported", flush=True)
    for name in MODULES:
        print(f"[{tag}] loading {name} ...", flush=True)
        try:
            load(name)
            print(f"[{tag}] {name} OK", flush=True)
        except Exception:
            print(f"[{tag}] {name} EXC\n{traceback.format_exc()}", flush=True)
    print(f"[{tag}] end", flush=True)


def _child(tag, import_torch_first):
    try_all(tag, import_torch_first)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    torch_first = len(sys.argv) > 2 and sys.argv[2] == "torch"

    if mode == "plain":
        try_all("plain", torch_first)
        return 0

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=_child, args=("spawn", torch_first))
    p.start()
    p.join(120)
    print(f"[main] spawn child exitcode={p.exitcode}", flush=True)
    return 0 if p.exitcode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
