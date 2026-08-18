# `m15_eplb0` — staged shim patch for the M15 + EPLB placement-only compose

Design and evidence: `../M15_EPLB0_COMPOSE.md`. Read it first — in particular
§1 (the router already emits **physical** ids, so no gather is needed) and §7
(placement-only EPLB still costs **≈1.31 GiB/rank** of transfer buffer).

Everything here is behind **`VLLM_PF4H_M15_EPLB`**, exact `0`/`1`, default
`0`. With it unset, `PF4H-FULL-018` refuses EPLB exactly as today and the
serving path is byte-identical. No vLLM file is patched; no HIP source
changes; no kernel rebuild.

## Contents

| file | status | note |
|---|---|---|
| `m15_eplb.py` | **new** | selection gates, placement proof, post-rearrangement re-attestation |
| `m15_contracts.py` | modified | `EPLB_COMPOSE_ENV` + `eplb_compose_enabled()` |
| `vllm_full.py` | modified | `PF4H-FULL-018` becomes conditional |
| `m15_vllm.py` | modified | hook install, `require_attested()`, `register_layer_weights()` |
| `m15_eplb0.diff` | — | the three diffs, unified, against the deployed sources |
| `eplb0_predict.py` | tool | zero-GPU falsifier; runs vLLM's own policy over the skew histograms |

The three modified files are **full copies of the currently deployed sources
with the changes applied**, taken from the live mirror at
`.../scratchpad/mirror/pf4h_integration/`. Verify the base matches your node
before overwriting (see step 1).

SHA-256 of what is staged here:

```
10d9aeb0531bb60d43a37d31fb20596d7bba626b53dc11ef305e9aec02ee57b2  eplb0_predict.py
d00e1d98a1a991b26c34a6159a316d718ae5c0558a9dd816ec3339c0200d658a  m15_contracts.py
21537136501fffa45276b050c529ebe7e0c355c35af28376bde6fa415f683f34  m15_eplb.py
92db744084854a0db1415bd3525f083329ab1850ad6a548eb753e6b055c61dbd  m15_vllm.py
e17dc9a4edf06cecfdd775240ad79c068cc49634311a74a8aa65b07c5f4cc322  vllm_full.py
```

## Unified diffs

Also in `m15_eplb0.diff`, applicable with `patch -p1` from the
`pf4h_integration` package directory.

### `m15_contracts.py`

```diff
@@ -40,6 +40,7 @@
 M15_MODE_VALUE = "m15"
 RING_SLOTS_ENV = "VLLM_PF4H_M15_RING_SLOTS"
 ALLOCATE_PART_ENV = "VLLM_PF4H_M15_ALLOCATE_PART"
+EPLB_COMPOSE_ENV = "VLLM_PF4H_M15_EPLB"
```

plus a new `eplb_compose_enabled(environ=None) -> bool` after
`allocate_part()`, matching the `ring_slots()`/`allocate_part()` shape: exact
`"0"`/`"1"` parse, `M15-ENV-004` blocker on anything else, `False` when unset.

### `vllm_full.py`

```diff
@@
+def _eplb_compose_gate(
+    moe_config: Any, blockers: list[ActivationBlocker]
+) -> bool:
+    """Whether the EPLB placement-only compose owns the EPLB decision.
+
+    Imported lazily: ``m15_contracts``/``m15_eplb`` ship only with
+    ``--integration-mode m15``, and a ``full``-mode install must keep this
+    module importable.  With the file set absent, or the variable unset, the
+    caller falls through to the verbatim PF4H-FULL-018 refusal.
+    """
+
+    try:
+        from .m15_contracts import eplb_compose_enabled
+        from .m15_eplb import validate_eplb_compose
+    except ImportError:
+        return False
+    if not eplb_compose_enabled():
+        return False
+    blockers.extend(validate_eplb_compose(moe_config))
+    return True
@@ in validate_full_selection
-    if bool(getattr(parallel, "enable_eplb", False)):
+    if not _eplb_compose_gate(moe_config, blockers) and bool(
+        getattr(parallel, "enable_eplb", False)
+    ):
         blockers.append(
             ActivationBlocker(
                 "PF4H-FULL-018",
                 "EPLB can remap experts after graph capture",
-                "disabled EPLB and fixed linear EP8 ownership",
+                "disabled EPLB and fixed linear EP8 ownership, or "
+                "VLLM_PF4H_M15_EPLB=1 for the placement-only compose",
             )
         )
```

The lazy import matters: `vllm_full.py` also serves `--integration-mode full`,
where `m15_*.py` are not installed (`apply.py` `M15_RUNTIME_FILES`).

### `m15_vllm.py`

```diff
@@ imports
+from .m15_eplb import (
+    eplb_compose_enabled,
+    install_eplb_hooks,
+    register_layer_weights,
+    require_attested,
+    validate_runtime_compose,
+)
@@ M15PrepareAndFinalize.post_init_setup, after fused_experts.bind_controller(self)
+        if eplb_compose_enabled():
+            validate_runtime_compose(runtime)
+            install_eplb_hooks()
@@ M15PrepareAndFinalize.prepare, on the active path before _require_bound()
+        require_attested()
@@ M15Experts.apply, after runtime.bind_layer_weights(...)
+        register_layer_weights(
+            controller.layer_id, w1, w1_scale, w2, w2_scale
+        )
```

`install_eplb_hooks()` wraps `EplbState.add_model` (to prove the initial
placement) and `EplbState.step` (to re-prove after each rearrangement). It is
in time because `process_weights_after_loading` completes inside
`model_loader.load_model`, which `GPUModelRunner.load_model` calls before
`EplbState.add_model` (`gpu_model_runner.py:5308-5391`).

## Deploy

All paths below are on the node.

**1. Verify the base.** The staged copies were made from the mirror of the
deployed tree. Confirm nothing drifted:

```bash
PKG=$(python3 -c 'import vllm.model_executor.layers.fused_moe.experts.pf4h_integration as p; print(p.__path__[0])')
for f in m15_contracts.py vllm_full.py m15_vllm.py; do
  echo "== $f"; diff -u "$PKG/$f" "<staged>/$f" | head -60
done
```

Every hunk must be one of the three above. If any *other* difference appears,
stop — the node has drifted from the mirror and the copies must be re-derived
rather than pasted over.

**2. Back up and install.**

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for f in m15_contracts.py vllm_full.py m15_vllm.py; do
  cp -p "$PKG/$f" "$PKG/$f.pre_eplb0.$STAMP"
done
cp <staged>/m15_contracts.py <staged>/vllm_full.py <staged>/m15_vllm.py "$PKG/"
cp <staged>/m15_eplb.py "$PKG/"
python3 -m py_compile "$PKG"/m15_eplb.py "$PKG"/m15_contracts.py \
                      "$PKG"/m15_vllm.py "$PKG"/vllm_full.py
```

`m15_eplb.py` is a new file; if `apply.py` is re-run later, add it to
`M15_RUNTIME_FILES` so a reinstall does not drop it.

**3. Null gate — prove the default is unchanged.** Start the server with the
existing m15 recipe and **`VLLM_PF4H_M15_EPLB` unset**, no `--enable-eplb`.
Expect the usual `M15_SELECTION_RECEIPT` / `M15_ACTIVATION_RECEIPT` /
`M15_GRAPH_COMMIT_RECEIPT` and the banked timing. Any deviation here means the
patch was not additive; roll back before going further.

**4. Zero-GPU falsifier — do this before step 5.**

```bash
python3 <staged>/eplb0_predict.py --self-test            # no vLLM needed
python3 <staged>/eplb0_predict.py $M15_SKEW_OUT/skew_rank*_pid*.json
```

The tool sums every rank's per-layer histogram, calls vLLM's own
`DefaultEplbPolicy.rebalance_experts(weight, 256, 8, 1, 8)` — the exact
placement-only configuration — and prints per-layer max-rank load before and
after, plus the projected critical-rank relief. **If the worst-layer relief is
under ~15 %, stop: the skew is not stationary and this arm cannot pay.**
Treat the number as an upper bound (§8 of the design).

**5. Live gate — eager first.** EPLB on, graph target off, so Python runs every
step and every pointer gate is live:

```bash
export VLLM_PF4H_INTEGRATION_MODE=m15
export VLLM_PF4H_M15_EPLB=1
unset VLLM_PF4H_B4096_GRAPH_TARGET
vllm serve ... \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts":0,"use_async":false,"window_size":128,"step_interval":300,"log_balancedness":false}'
```

`use_async` **defaults to true** and is refused (`M15-EPLB-006`); it must be
set false explicitly. Expect at load:

```
M15_EPLB_HOOK_RECEIPT state=installed env=VLLM_PF4H_M15_EPLB
M15_EPLB_PLACEMENT_RECEIPT models=1 layers=58 experts=256 state=permutation
```

and, after the first rearrangement (~300 steps here), no
`PF4HActivationError`. Sweep `poll_all_protocol_errors()` for `pperr`.

Budget note: add **≥1.4 GiB/rank** of headroom before this step (§7 of the
design). If the server OOMs at load with the usual
`--gpu-memory-utilization`, lower it rather than reaching for
`VLLM_PF4H_M15_RING_SLOTS`.

**6. Graph gate.** Same as step 5 plus `VLLM_PF4H_B4096_GRAPH_TARGET=pf4h`.
Confirm `M15_GRAPH_COMMIT_RECEIPT`, then let it run past two rearrangements.

**7. Timing arms.** Compare against the banked EPLB-off number on identical
prompts. Also record the post-EPLB skew histogram (`M15_SKEW_OUT` with the
existing sitecustomize) and check it against step 4's projection.

**8. Accuracy.** Protocols A/B/C in §9 of the design. Protocol A needs a
five-line sitecustomize to dump
`eplb_state.model_states[k].physical_to_logical_map.cpu()` and the
`setup_eplb_from_mapping` seeding path; it is the receipt that matters.

## Refused combinations (fail at load, by design)

| code | condition |
|---|---|
| `M15-EPLB-001` | `VLLM_PF4H_M15_EPLB=1` without `--enable-eplb` |
| `M15-EPLB-003/004` | `num_redundant_experts != 0` / physical count != 256 |
| `M15-EPLB-006` | `use_async` true (the vLLM default) |
| `M15-EPLB-007` | `enable_elastic_ep` |
| `M15-EPLB-008/009` | M18 replica sets / M20 replica-cache budget active |
| `M15-EPLB-016/017` | replica count != 1, or the map is not a permutation |
| `M15-EPLB-018/019` | a map or weight pointer moved across a rearrangement |

`M15-EPLB-008/009` is the one to watch operationally: **EPLB0 and M18/M20 are
mutually exclusive.** M18/M20 hold copies of expert weights; EPLB rewrites only
the live base slabs, so the replica pools would silently serve stale weights.

## Rollback

```bash
for f in m15_contracts.py vllm_full.py m15_vllm.py; do
  cp -p "$PKG/$f.pre_eplb0.$STAMP" "$PKG/$f"
done
rm -f "$PKG/m15_eplb.py" "$PKG"/__pycache__/m15_eplb*.pyc
```

Or simply `unset VLLM_PF4H_M15_EPLB` — with the variable unset the patched
files take the identical code path as the originals.
