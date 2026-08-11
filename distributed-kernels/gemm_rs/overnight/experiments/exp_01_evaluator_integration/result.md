# exp_01 — our kernel under the official evaluator: root cause found and fixed

**Session: 2026-08-11, 10:17Z–11:12Z (≈55 min, time-boxed at 60).**
**Verdict: the hang is localized, root-caused, fixed and the fix is verified —
but NOT yet run through the evaluator's test mode. No performance number is
reported, and none may be quoted from this session.**

## Headline

1. **Suspect #1 (`hipIpcOpenMemHandle` / `hipErrorAlreadyMapped`) is DEAD.**
   All 8 ranks complete the whole IPC exchange, both regions, every peer, with
   zero `FAILED peer` and zero `AlreadyMapped`.
2. **The "6 of 8 ranks reach `state ready`" signature is superseded.** With the
   current code **all 8** ranks reach `state ready` and go on to complete
   **101 consecutive `custom_kernel` calls each** (808 total) under `eval.py`,
   with **zero error bits**, in **1.38 s** wall.
3. **The real blocker was ours, and it was one line.** `hk_submission.py` held a
   module-global **strong reference** to the previous `ProcessGroup` in
   `_LAST_PG`. That reference outlives `destroy_process_group()` and pins the
   old NCCL communicator and its TCPStore, so the **next**
   `init_process_group` on the same fixed `MASTER_PORT` never completes. Every
   rank blocks in `_new_process_group_helper`.
4. **Fix: demote `_LAST_PG` to a `weakref`.** Verified: 8/8 ranks, two shapes,
   two process groups, `allclose=True`, `max|diff| = 9.766e-04`, exit codes all
   zero. A same-session control proves the hang was ours, not the environment.

## What was run, in order

| # | Step | Result |
|---|---|---|
| 0 | `rocm-smi --showpids`, stale-process scan, container check | clean: no KFD pids, no stragglers, `dhk-gemmrs` + `dhk-eval` up |
| 0b | LF-normalized hash of node vs Windows sources | **`gemm_rs_mi300x.cpp` differs**; `hk_submission.py`, `run_ours_evaluator.sh` identical. `push.ps1` deliberately NOT run (see hazards) |
| 1 | `mp_smoke.py`, 2 cases, `HK_DEBUG=1` | **case 0 PASS** on all 8 ranks (`allclose=True`, `9.766e-04`); **case 1 HUNG in the 2nd `init_process_group`**; `timeout` fired at 600 s (rc 124) |
| 2 | `HK_ONE=1 tools/run_ours_evaluator.sh` | evaluator raised `multiprocessing.TimeoutError`; run partly corrupted by a CRLF bug in the script (see hazards) |
| 2b | `eval.py benchmark cases_one.txt` staged and invoked directly, `HK_DEBUG=1` | **the decisive run** — see below |
| 3 | probe: is a `ProcessGroup` weakref-able? | yes — `weakref OK -> True`, `group_name = '0'` |
| 4 | CONTROL arm: plain `matmul` + `reduce_scatter_tensor` submission, `mp_smoke` 2 cases | **PASS both cases**, 16/16 `allclose=True`, all 8 `DONE`, exit codes `[0]*8` |
| 4 | CANDIDATE arm: `_LAST_PG` → weakref, `mp_smoke` 2 cases | **PASS both cases**, 16/16 `allclose=True` at `9.766e-04`, all 8 `DONE`, exit codes `[0]*8` |

## The decisive evidence (step 2b, single graded shape 64×7168×18432)

Step counts over all 8 ranks in one `eval.py benchmark` run:

```
enter custom_kernel                808     cache vote                       8
cache hit                          800     cache miss                       8
c_heap:  get_ipc_handle done         8      c_heap: all_gather_object done   8
c_heap:  opened all peers            8      signals: opened all peers        8
descriptors uploaded                 8      setup barrier returned           8
device synchronize returned          8      state ready                    808
launch issued                      808      launch synchronized            808
ERROR BITS                           0      FAILED peer                      0
```

- `cache vote [False, False, False, False, False, False, False, False]` ×8 —
  **unanimous, so setup was collective-symmetric.** The two-tier branch
  decision works exactly as designed.
- `808 = 8 ranks × 101 calls` = 1 obligatory correctness check + 100 timed
  repeats (`max_repeats=100`). Every one of them reached
  `launch synchronized`.
- **The evaluator's own correctness oracle passed.**
  `_run_distributed_benchmark` calls `wrap_check_implementation` after the
  first `custom_kernel` and does `if not good: return message`. The 100
  subsequent repeats prove `good` was True on all 8 ranks. This is eval.py's
  own checker, on a graded shape — not our harness's.
- Elapsed from `cache vote` to the last `launch synchronized`: **1.38 s**
  (`t = 20354231.827 → 20354233.209`).
- Then, silence, and the `faulthandler` watchdog dumped **every rank** at:

```
_new_process_group_helper   distributed_c10d.py:2253
init_process_group          distributed_c10d.py:1838
_run_distributed_benchmark  eval.py:322          <-- dist.init_process_group
worker                      multiprocessing/pool.py:125
```

- Parent side: `TimeoutError` from `run_multi_gpu_benchmark`, `eval.py:428`
  (`el.get(timeout=180)`), reached from `run_benchmarking`, `eval.py:485`.

So at least two `_run_distributed_benchmark` dispatches occurred: the first ran
101 calls to completion, the second could not create its process group.

**The falsifiable prediction in the task brief was tested and held.** No rank
was stranded at `state ready` or at `setup barrier returned`; all 8 got
`device synchronize returned` and all 808 calls completed. The device
synchronize is doing what we thought. Suspect #2 is also not the fault here.

## Root cause

`hk_submission.py` kept the previous process group alive:

```python
_LAST_PG = None          # was assigned the ProcessGroup object itself
...
settled = (state is not None and group is not None
           and group is _LAST_PG and key == _LAST_KEY)
...
    _LAST_PG = group     # strong reference, survives destroy_process_group
```

The in-code comment said the strong reference was deliberate — "so a recycled
object address cannot make a new group look like the old one". That reasoning is
sound about address reuse and wrong about lifetime: `destroy_process_group()`
relies on the last reference being dropped to finalize the NCCL communicator and
its TCPStore. Holding one in a module global that lives as long as the pool
worker means the old communicator is never finalized, and the next
`init_process_group` on eval.py's fixed `MASTER_PORT = "12356"` (or mp_smoke's
`12399`) blocks forever inside `_new_process_group_helper`.

This is why the failure always looked like it was "the second test case": the
first group is fine, and *every* group after it is impossible.

### Why the control settles the attribution

The reference submission (plain `matmul` + `reduce_scatter_tensor`, no globals,
no IPC) ran the identical two-case mp_smoke driver, i.e. the identical
init/destroy/init sequence on the identical port, and **passed**. So a second
`init_process_group` is perfectly healthy in this container. The fault was in
our retained state, not in torch, NCCL, or the node.

## The fix (applied, verified)

`_LAST_PG` becomes a weakref and the identity test compares the referent:

```python
import weakref
...
settled = (state is not None and group is not None
           and _LAST_PG is not None and _LAST_PG() is group
           and key == _LAST_KEY)
...
    _LAST_PG = weakref.ref(group) if group is not None else None
```

- **Still symmetric across ranks.** The predicate is "is the current default
  group the same object as the one I last settled on". While a group is current,
  torch holds its own reference in `_world`, so the weakref to *that* group
  cannot be dead. For any new group, `_LAST_PG() is group` is False on every
  rank whether the old referent was collected or not. All eight ranks therefore
  compute the same answer at the same call, which is the property the vote
  exists to protect. Both cases in the candidate arm show unanimous
  `cache vote [False × 8]`.
- **Address reuse is still handled.** The original worry does not return: the
  test compares the referent object, and a weakref to a collected object returns
  `None` rather than resurrecting a recycled address.
- `weakref.ref` on `torch._C._distributed_c10d.ProcessGroup` was probed on this
  node before relying on it: supported (`weakref OK -> True`).

`hk_submission.py` sha256 (LF-normalized), **identical on node and Windows**:

| | sha256 |
|---|---|
| before | `9bd8e6aee238b8a6d6d789de2e98b140c9041dc4971a86692e60c6b79b7cdd79` |
| after | `b2d71956cf6a578bb8e81183119cb4d98e26e7e96a6f0c2e6a6e91403dfc9d05` |

## Numbers

**None are reported. Every run this session used `HK_DEBUG=1`**, which puts a
`torch.cuda.synchronize()` and unbuffered stderr writes inside every timed call,
and **test mode has not passed 11/11 in this session**, so per the measurement
rule no benchmark figure from here is quotable. The evaluator never emitted a
`benchmark.0.mean` anyway — it timed out before logging one.

For the record, the only wall-clock observation is a *diagnostic* one: 101
`HK_DEBUG=1` calls on shape 1 across 8 ranks took 1.38 s end to end, i.e. ~13.7
ms per call including the debug synchronize, stderr flush, `clear_l2_cache()`
and two `dist.barrier()`s per iteration. That is a debug-mode figure and must
not be compared to anything.

## Hazards found this session (all cost time, all now documented)

1. **`push.ps1` would have reverted the kernel.** It scps
   `distributed-kernels/gemm_rs/*.cpp *.cuh *.hpp` from Windows to the node.
   The node's `gemm_rs_mi300x.cpp` is **not** the Windows copy — LF-normalized
   `9c98be82…` on the node vs `f3d74d60…` on Windows, while the other three
   sources and `hk_submission.py` matched byte-for-byte. Running `push.ps1`
   would have overwritten the validated node kernel with a stale Windows one,
   silently, while leaving `build/*.so` alone. **`push.ps1` was not run.**
   `nsh.ps1` needs no push (it base64s the script body), so nothing was lost.
   *Before any future `push.ps1`, diff the kernel sources both ways.*
   `experiments/exp_01_evaluator_integration/hash_lf.ps1` does the
   LF-normalized comparison that makes this checkable in one command.
2. **The node's `tools/run_ours_evaluator.sh` has CRLF line endings and is
   broken.** Its trailing `if [ "${HK_ONE:-0}" = "1" ] … else … fi` is corrupted
   (`line 66: est: command not found`, `timeout: invalid time interval '1500\r'`,
   `$'fi\r': command not found`), so **it ran the `HK_ONE` branch and then the
   full-suite branch as well.** It is CRLF precisely *because* `push.ps1`
   (which strips CRs on the far side) was not run. Either push it or
   `sed -i 's/\r$//'` it before trusting it.
3. **`run_ours_evaluator.sh` overwrites its own evidence.** Both `go` calls
   write `$DIR/$mode.stderr.txt`, so a second `benchmark` invocation truncates
   the first one's stderr. That is what erased the step counts on the first
   attempt. Step 2b stages and invokes `eval.py` directly with per-run log
   names instead.
4. **`timeout` does not reap mp_smoke's ranks.** It SIGTERMs only the direct
   child; the 8 spawned ranks can outlive it. Checked after every run this
   session; none leaked, but `03_reap.sh` exists for it (SIGTERM only — never
   SIGKILL a HIP IPC process).
5. **`benchmark` mode is not correctness-blind.** `recheck=False` only disables
   the *per-iteration* re-check; `_run_distributed_benchmark` always runs one
   obligatory `wrap_check_implementation` before the timed loop and bails out
   with a message if it fails. So a benchmark run that produces 100 repeats has
   passed the evaluator's oracle once on that shape. It is still not a
   substitute for 11/11 test mode, because it checks one shape and one input.
6. **`benchmark` mode has no warmup pass** — the `# warmup` line is in
   `leaderboard` mode only. But `run_benchmarking` (eval.py 469–490) does
   dispatch `run_single_benchmark` more than once per case list; the exact
   structure of lines 469–490 and 370–410 was not read and is the one loose
   end in the account above.

## Exactly where this stopped, and what to do next

Nothing is blocked. The next session starts at the top of the ladder with the
fixed submission already in place on both the node and Windows:

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
# 0. the runner script is CRLF-broken on the node; fix it first, do NOT push.ps1
#    until the kernel-source diff in hazard 1 is resolved:
docker exec dhk-gemmrs sed -i 's/\r$//' $ON/tools/run_ours_evaluator.sh
# 1. test mode, 11 shapes, the gate that licenses reporting any number:
HK_DEBUG=1 bash $ON/tools/run_ours_evaluator.sh   # expect check: pass 11/11
# 2. ONLY if 11/11 passed, in the SAME session, benchmark with debug OFF:
HK_DEBUG=0 bash $ON/tools/run_ours_evaluator.sh
```

Expected cost: test mode ran 11/11 in 93 s for the reference arm in exp026, and
our kernel now does 101 calls per shape in 1.4 s, so the suite should be minutes,
not the 25-minute timeout cycles this integration used to take.

Two open risks for that run, neither yet observed:

- **Shape count.** Only shape 1 (64×7168×18432) and mp_smoke's two shapes have
  been through the fixed path. 11 test shapes means 11 process groups and 11
  `_ShapeState` builds; the weakref path is exercised once per group, so this
  should scale, but it is unproven past two.
- **Worker↔rank permutation.** The pool reassigns workers between cases, which
  is what the vote exists for. Both mp_smoke cases voted unanimously `False`
  (fresh key each time), so the *mixed*-vote path — some workers holding the key
  and others not — has still never been exercised on hardware. Watch the
  `cache vote` lines in test mode: a mixed vote there is the first real test of
  that branch, and it should produce a rebuild on all 8 ranks, not a hang.

Also still not done, and untouched by this session: the reference GEMM+RCCL
calibration arm (`tools/run_competition_bench.sh`, `reference`), and rank-1
repair #5. The 20-minute calibration slot was consumed by the root-cause work.
