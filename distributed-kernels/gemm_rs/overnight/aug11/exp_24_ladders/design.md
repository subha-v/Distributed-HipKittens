# exp_24 — design: two instruments, five arms, two protocols

## 1. Why two instruments and not one

The dispatch asks for three arms, same-run interleaved, at 3 rotations × 50
iterations, with a null arm run as *the same submission under two module names*,
under both protocols, reported as best / median / mean / samples.

The three existing `tools/` drivers cannot deliver that, and the reason is in
`eval.py`, not in the drivers:

| requirement | can `eval.py` do it? |
|---|---|
| same-run interleaved | **no.** One evaluator run measures one `submission.py`. Three runs are three runs, and exp_10 recorded that two separately-staged evaluator runs already gave this project a misleading answer once — 1.37× where the interleaved instrument said 1.06×. |
| null arm as one file under two module names | **no.** `_run_distributed_benchmark` does `from submission import custom_kernel`. One module, one name. |
| 3 rotations × 50 iterations | **no.** The loop is adaptive: it breaks when `err/mean < 0.001`, or `mean·runs > max_time_ns`, or total wall > 120 s. The fixtures show `runs: 100` for every shape, and the count is not a parameter of the protocol. |
| median, raw samples | **no.** `calculate_stats` emits `runs/mean/std/err/best/worst`. No median, no sample vector. |
| pipelined protocol | **no.** The timed region always contains the trailing `synchronize + barrier`. |
| duration-based warmup | **no.** There is *no* warmup: one obligatory correctness call, then the timed loop. The first timed iteration carries RCCL channel setup and, for rank-1, JIT. |

So `eval.py` is the wrong instrument for the ladder and the right instrument for
the cross-check — it is the competition's literal harness, and the paper needs
that number to exist. Hence:

- **Instrument A, `ladder_mp.py`** — the ladder. Five arms, one pool, both
  protocols, full rotation, raw samples. Reimplements `eval.py`'s timed region
  verbatim rather than reimplementing `eval.py`.
- **Instrument B, the three existing evaluator drivers** — orchestrated, rotated,
  raw output captured under this experiment. Cross-check only.

`ladders.json` carries A under `protocols` and B under `evaluator_crosscheck`.
The headline ratios come from A. Both are labelled by instrument everywhere.

## 2. Instrument A — `ladder_mp.py`

### 2.1 Arms

| key | module name loaded | source file | in ratios? | in correctness gate? |
|---|---|---|---|---|
| `ours` | `ours_submission` | `harness/submission.py` | yes | yes |
| `ours_null` | `ours_null_submission` | **the same** `harness/submission.py` | no | yes |
| `reference` | `reference_submission` | `compbench/reference/submission.py` | yes | yes |
| `rank1` | `rank1_submission` | `compbench/rank1/submission.py` (patched copy) | yes | yes |
| `harness_floor` | — | in-process closure | no | **no, deliberately wrong** |

`ours` and `ours_null` are the identical file loaded twice under distinct names,
which is what makes the null arm a null: it shares the kernel, the module code and
the build, and differs only in allocation identity and position in the rotation —
exactly the two things exp_14 identified as this node's noise sources.

`harness_floor` returns a preallocated, correctly-shaped output tensor and does
nothing else. Under the graded protocol its time *is* the protocol's constant:
`clear_l2 → sync → barrier → clone → sync → barrier`. This turns the remembered
"57–79 µs empty timed region plus 15–20 µs of clone" into a number measured in the
same run as the ratios it is used to net out. Under the pipelined protocol it
measures the residual host issue cost.

### 2.2 One shape per invocation

Non-negotiable, and not for tidiness: rank-1 caches its compiled kernel in module
globals (`pre_compile_cache`, `pre_compile_cache2`) keyed by **nothing**, so the
cache is only valid for the shape it was built for. `eval.py` survives this by
destroying the process group between shapes, which runs rank-1's own
cache-clearing patch. A fresh pool per shape avoids reaching into rank-1's state
at all.

### 2.3 Timed regions

Graded, verbatim from `eval.py:349-366`:

```
clear_l2_cache()
torch.cuda.synchronize(); dist.barrier()
t0 = perf_counter_ns()
out = kernel(_clone_data(data))
torch.cuda.synchronize(); dist.barrier()
t1 = perf_counter_ns()
```

One documented deviation: `eval.py` times **on rank 0 only**. This instrument
times on all eight ranks and pools them, because the graded number is a per-call
wall time and every rank's samples are valid measurements of it — that is exp_10's
choice and it is what gives the medians their sample count. The rank-0-only
subset is retained separately in the JSON as `rank0_us`, so the evaluator-faithful
statistic is always recoverable without a re-run.

Pipelined:

```
clone = _clone_data(data)            # once, outside
clear_l2_cache()
torch.cuda.synchronize(); dist.barrier()
t0 = perf_counter_ns()
for _ in range(burst): out = kernel(clone)
torch.cuda.synchronize(); dist.barrier()
t1 = perf_counter_ns()
sample = (t1 - t0) / burst
```

The single clone is reused across the burst. All three submissions treat
`(x, w, bias)` as read-only; the instrument re-checks the burst's final output
against the oracle after timing, so a mutating arm fails rather than reporting a
fast number.

### 2.4 Rotation

```
for rep in range(reps):                       # reps defaults to 5 == len(ARMS)
    order = ARMS[rep % n:] + ARMS[:rep % n]   # every arm first exactly once
    protos = ("graded","pipelined") if rep % 2 == 0 else ("pipelined","graded")
```

Shape order is rotated by the driver across invocations, so no shape is
systematically first either.

### 2.5 Warmup

Duration-based: each arm is called until **both** `WARM_MS` (default 400 ms) has
elapsed **and** at least 20 calls have completed, then `synchronize` + `barrier`.
Fixed-iteration warmup on this node produced a 60% wrong number once, because
idle sclk is ~125 MHz against ~1900 pinned. Every arm is warmed before *any* arm
is timed, so no arm is timed while another is still cold.

### 2.6 Correctness before timing

Oracle is `matmul` (+ bias) → `all_reduce` → this rank's row slice, the same one
`mp_smoke` and exp_10 use. Every arm except `harness_floor` is checked at both
`1e-2` (graded gate) and `2e-3` (regression gate) before a single sample is taken,
and the instrument refuses to time a shape where any gated arm fails. A number for
a wrong kernel is worthless.

### 2.7 `VS_FORCE_BIAS`

Set to 1, reproducing what the official evaluator actually does. `eval.py`'s cases
parser does `int(val)` and keeps the raw string on `ValueError`, so
`has_bias: False` becomes the **string** `"False"`, which is truthy —
`reference.generate_input` therefore builds a bias for all six graded shapes,
including the three declaring `False`, and `ref_kernel` adds it, so the evaluator
stays self-consistent. rank-1's author noticed
(`#####?????? bench all have bias?` on `__conf`) and its cached fast path
dereferences `bias.data_ptr()` unconditionally, so a genuinely absent bias makes
rank-1 raise `AttributeError` on its **second** call. Running with bias on all six
shapes for all arms identically reproduces the graded reality and needs no patch to
rank-1. A genuine `has_bias=False` ladder is not obtainable for rank-1 without
repairing that line, and is out of scope.

## 3. Instrument B — the official evaluator, rotated

| arm | driver | container |
|---|---|---|
| `ours` | `tools/run_ours_evaluator.sh` | `dhk-gemmrs` |
| `reference` | `tools/run_reference_arm.sh` | `dhk-gemmrs` |
| `rank1` | `experiments/exp_10_rank1/r1_eval.sh rank1 <mode> <timeout>` | `dhk-gemmrs` |

Arm order is rotated per rotation index so no arm is systematically first.
rank-1 runs `benchmark` (warm) → `test` → `benchmark` (bench) in that order,
because `eval.py`'s test mode hardcodes a 60 s per-rank timeout that a cold
compile of rank-1's kernel exceeds. That pass order is bench3's, and it is kept.

`rocm-smi --showpids` preflight runs before **all three** arms, not just the
reference (which is the only driver that does it itself).

`eval.py` hardcodes `MASTER_PORT = "12356"`, so Instrument B's runs are strictly
serialized with a KFD-fd drain wait and a socket settle between them.

Each arm's `*.popcorn.txt`, `*.stdout.txt` and `*.stderr.txt` are copied out of
`compbench/<arm>/` into `raw/eval/rot<N>/<arm>/` under this experiment before the
next arm runs, because every driver begins with `rm -rf $DIR`.

### The popcorn format the parser must handle

```
benchmark-count: 6
benchmark.0.spec: world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
benchmark.0.runs: 100
benchmark.0.mean: 293261.57            <- NANOSECONDS
benchmark.0.std: 274024.16
benchmark.0.err: 27402.42
benchmark.0.best: 251421.0
benchmark.0.worst: 2989278.0
check: pass
```

Failure form is `benchmark.<i>.status: fail` followed by `benchmark.<i>.error:`.
A truncated run (rank-1 hit the 1500 s wall in exp_10 after three shapes) simply
stops emitting blocks, with `benchmark-count: 6` still at the top — which is
exactly the silent-null hazard the parser must fail on rather than paper over.

## 4. Why `tools/run_rank1_bench3.sh` is not the rank-1 driver

Read-only probing of the node found three defects, all consistent with bench3
predating exp_10's repair #6:

1. **No `AMDGCN_USE_BUFFER_OPS=0`.** exp_10 proved this is *the* blocker: Triton
   3.6.0 lowers rank-1's peer stores to `buffer_store_dwordx2`, whose voffset is
   32 bits, truncating element offsets of −6.6e8 … −4.4e9. Four ranks fault with
   `Write access to a read-only page` at addresses that match the int32-truncated
   heap bases exactly. bench3 builds its own `ENVS` string and passes it
   explicitly to `docker exec`, so the knob cannot be injected from outside.
2. **`COMPAT=$NODE/compat` does not exist.** `ls`: no such directory. The compat
   tree is at `$NODE/tools/compat`, so bench3's `PYTHONPATH` does not carry
   `sitecustomize.py` and repair #3 (the `wrap_handle_tensor_descriptor` stub) is
   absent.
3. **`$NODE/patch_rank1.py` does not exist.** The tool is at
   `$NODE/tools/patch_rank1.py`. bench3's staging step exits non-zero.

`experiments/exp_10_rank1/r1_eval.sh` has all three right, plus it clears
`TRITON_CACHE_DIR` (the cached `hsaco` has `buffer_store` baked in, so leaving the
cache makes the knob look ineffective) and runs in `dhk-gemmrs` rather than
`dhk-eval` — which matters, because the two containers have separate filesystems
and `dhk-gemmrs` is where exp_10's iris staging actually landed. It is the driver
that produced exp_10 §4. Nothing under `tools/` was edited; bench3's pass ordering
is preserved.

## 5. Staging

`run_ladders.sh stage` reproduces the drivers' staging without running them,
because Instrument A needs the arms present without an evaluator pass:

- `ours` ← `cp` the evaluator files from the exp026 tree + `harness/submission.py`
- `reference` ← the same evaluator files + the exp026 tree's own `submission.py`
- `rank1` ← the same evaluator files + `python3 tools/patch_rank1.py <frozen> <dest>`

`tools/patch_rank1.py` refuses to emit anything unless the frozen source hashes to
`7940fcb8…f0dc5`, so the hash gate is enforced by the tool on every run and not by
this experiment's discipline. The frozen file is never opened for writing; it is
`-rw-rw-r-- subvadla` and everything runs out of `compbench/`.

`compbench/rank1/` retains `ipc_handles_rank*.bin` from prior runs; staging
removes them, along with `.triton`, before every rank-1 pass.

## 6. Failure modes and what each one means

| symptom | reading | action |
|---|---|---|
| `Memory access fault` in a rank-1 arm | the buffer-ops knob did not take, or `.triton` was stale | clear `.triton`, confirm the env reached the container; **not** a rank-1 result |
| `SHIM_WAS_CALLED` in any stderr | repair #3 stopped being dead code | the run is void; the repair is no longer behaviour-preserving and must be re-argued |
| `AttributeError: 'NoneType' … data_ptr` from rank-1 on call 2 | `VS_FORCE_BIAS` did not reach the workers | fix the env, not the submission |
| an arm's `custom_kernel` returns in ~10 µs on shape 6 | rank-1 fell back to torch (`(M,N,local_K) not in __conf` → `return origin(...)`) | verify all six shapes are in `__conf`, `online_config` and `online_config_group` before quoting any number |
| `benchmark-count: 6` but fewer than six blocks | evaluator hit its wall | the parser **fails loudly**; partial ladders are not published |
| `ours` vs `ours_null` differ by more than the exp_14 4.28% floor | the instrument is noisier than the effects | more reps, not a narrative |
| `harness_floor` scales with shape size | the netting argument in `plan.md` §7 is wrong | report the raw ladder only and say why |

## 7. Node discipline

One 8-GPU job of ours at a time. Preflight requires `rocm-smi --showpids` to be
empty *and* the KFD fd count to be zero, with a bounded drain wait between arms.
`timeout --signal=TERM` and `setsid` on every long run; **never SIGKILL** — this
kernel uses HIP IPC and leaked mappings wedge the node. Clocks are pinned before
anything is timed and the pinned state is recorded in the JSON so an unpinned run
is identifiable after the fact rather than mysterious.

## 8. Runtime budget

| phase | expected wall |
|---|---|
| preflight + pin + provenance + staging | ~2 min |
| Instrument A, 6 shapes × (pool setup + rank-1 JIT + 5 arms × 2 protocols) | 35–60 min, dominated by rank-1's per-shape JIT and shapes 5/6 |
| Instrument B, 3 arms × (test + benchmark), rank-1 also a warm pass | 45–90 min, capped by `eval.py`'s own 120 s-per-shape convergence limit and the drivers' `timeout` values |
| aggregation | seconds |
| **total** | **~1.5–2.5 h** |

`LAD_QUICK=1` runs one shape with reduced reps for a ~6 min end-to-end validation
of the whole pipeline before the real ladder is committed to.
