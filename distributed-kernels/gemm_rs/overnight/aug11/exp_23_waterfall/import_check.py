"""Assert one rung module imports and exposes the fixed entry point.

A link step can produce a file that is not an importable module (wrong
PyInit_ symbol, missing device fatbin), and that failure is silent until the
sweep tries to load it. Separate file rather than a heredoc: heredocs piped
through docker exec inside an nsh-transported script have silently produced no
output on this node.

usage: import_check.py <path/to/module.so> <module_name>
"""

import importlib.util
import sys


def main():
    path, name = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exports = [x for x in dir(module) if not x.startswith("_")]
    # bind_function's exported name is fixed by the source; only the module
    # name varies per rung, so every rung must still expose this symbol.
    if "gemm_rs_mi300x" not in exports:
        print(f"  {name}: FAIL entry point missing, exports={exports}")
        return 1
    print(f"  {name}: OK exports={exports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
