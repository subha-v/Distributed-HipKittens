"""Compatibility shim so the frozen rank-1 submission can run on this node.

The submission replaces Triton's HIPLauncher with its own, and that class's
__init__ does:

    from triton.backends.amd.driver import (compile_module_from_src,
        wrap_handle_tensor_descriptor, include_dirs)
    ...
    has_tensor_desc_arg = any(isinstance(sig, str)
                              and sig.startswith("tensordesc")
                              for sig in self.signature.values())
    self.launch = (wrap_handle_tensor_descriptor(mod.launch)
                   if has_tensor_desc_arg else mod.launch)

`wrap_handle_tensor_descriptor` does not exist in the Triton shipped here
(3.6.0 in every ROCm image on this node), so the *import* fails even though the
symbol is only *called* when a kernel argument is a tensor descriptor. The
GEMM-RS kernels pass plain pointers, so that branch is never taken.

This shim therefore supplies the name and nothing else. It deliberately raises
if it is ever actually called: that way the repair is behaviour-preserving when
the branch is dead, and fails loudly rather than silently diverging if it is
not. Verify after any run that SHIM_WAS_CALLED never appears in stderr.

Placed on PYTHONPATH as sitecustomize so it applies to the evaluator's spawned
per-rank workers too, without editing the frozen submission.
"""

import os
import sys


def _install_iris_aliases():
    """iris renamed its IPC handle type to be backend-agnostic.

    rank-1 calls `iris.hip.hipIpcMemHandle_t()`. Every iris revision available on
    this node (8 checkouts, 4df4e85f..31c6221f) spells it `gpuIpcMemHandle_t`;
    the old name exists in none of them. This is a pure rename, so aliasing it is
    exact. Without the alias the per-rank helper process that builds rank-1's
    symmetric heap dies, and the parent then waits for a heap_bases_*.pkl that
    never appears -- which is why the evaluator only ever reported a timeout.
    """
    try:
        import iris.hip as hip
    except Exception:
        return
    for old, new in (("hipIpcMemHandle_t", "gpuIpcMemHandle_t"),
                     ("cudaIpcMemHandle_t", "gpuIpcMemHandle_t")):
        if not hasattr(hip, old) and hasattr(hip, new):
            setattr(hip, old, getattr(hip, new))
            if os.environ.get("COMPAT_SHIM_VERBOSE"):
                print(f"[compat] aliased iris.hip.{old} -> {new}",
                      file=sys.stderr)


def _install():
    _install_iris_aliases()
    try:
        from triton.backends.amd import driver
    except Exception:
        return

    if not hasattr(driver, "wrap_handle_tensor_descriptor"):
        def wrap_handle_tensor_descriptor(launch):
            raise NotImplementedError(
                "SHIM_WAS_CALLED: wrap_handle_tensor_descriptor is a "
                "compatibility stub for Triton 3.6.0 and must never be "
                "invoked. A kernel argument was a tensordesc, so this run is "
                "NOT comparable to the submission's original environment."
            )

        driver.wrap_handle_tensor_descriptor = wrap_handle_tensor_descriptor
        if os.environ.get("COMPAT_SHIM_VERBOSE"):
            print("[compat] added triton wrap_handle_tensor_descriptor stub",
                  file=sys.stderr)


_install()
