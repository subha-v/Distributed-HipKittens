# exp_10 — the frozen rank-1 submission, running and benchmarked on this node

**Verdict: rank-1 RUNS.** Correct on all six graded shapes at both `1e-2` and
`2e-3`, on all 8 ranks, and it is **1.78× faster than our kernel** under the
graded protocol measured same-run interleaved on this node.

The blocker was never the symmetric-heap bootstrap. It was a Triton 3.6.0 AMD
codegen optimization (`buffer_*` ops) silently truncating rank-1's 64-bit
peer-heap offsets to 32 bits. The fix is one environment variable and **zero
additional edits to the submission**.

---

## 1. Provenance — the frozen submission is unmodified

```
frozen source : /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/
                references/submissions/gemm_rs_rank1_58abcf.py
sha256        : 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5
expected      : 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5   MATCH
staged copy   : abd5aa73f7b6167bacb3d5319181fa65e82c4a08cff5af4eb76ebc32a0b84257
diff frozen→staged : 8 lines, all inside the one disclosed packed_metadata repair
```

The frozen file was never opened for writing. `tools/patch_rank1.py` verified the
hash before emitting the copy, and everything ran out of
`$ON/compbench/rank1/`. iris revision `4df4e85f`.

---

## 2. Repairs — what was needed, what was not

### 2.1 Repair #1 (iris staging) — needed, and it had silently regressed

rank-1 needs iris at its hardcoded `python3.10` path for **two independent**
reasons, both at module scope:

```python
25: os.system("sudo sed -i '66,82 s/^/#/' /usr/local/lib/python3.10/.../iris/__init__.py")
26: with open("/usr/local/lib/python3.10/dist-packages/iris/__init__.py", "r") as f:   # literal read
32: import iris                                                                        # importable
```

Staged `~/amd-master/iris/iris` (rev `4df4e85f`) to exactly that path.

**Why previous sessions' staging never worked.** It was done in `dhk-eval`.
`dhk-eval` and `dhk-gemmrs` are **separate containers with separate
filesystems** — writing `/usr/local/lib` in one does nothing for the other. The
charter's description of `dhk-eval` as existing "only to write `/usr/local/lib`
for rank-1's iris staging" is therefore wrong in effect. The working move is
`docker exec -u 0 dhk-gemmrs`, which has root in *our* container. Additionally
the container had been recreated since the last attempt, so the path was empty
at the start of this session.

Behaviour-preserving: supplies the declared dependency, byte-unmodified.

### 2.2 Repair #2 (no-op `sudo`) — needed, verified

rank-1 runs `sudo sed -i '66,82 s/^/#/'` on iris's `__init__.py`. In revision
`4df4e85f` lines 66–82 are exactly:

```
68: from . import hip
71: from . import experimental
74: from .logging import (...)
```

so the real edit would comment out the very `iris.hip` the submission then
references. A no-op `sudo` shim sits early on PATH
(`tools/compat/bin/sudo`). Confirmed `import iris` and `import iris.hip` both
succeed **unpatched**.

Behaviour-preserving: that line exists to work around a *different* iris
revision; on this one it is destructive, and skipping it leaves iris as shipped.

### 2.3 Repair #3 (Triton `wrap_handle_tensor_descriptor` stub) — needed, verified dead

Stub installed via `tools/compat/sitecustomize.py`; it raises `SHIM_WAS_CALLED`
if ever invoked. **`SHIM_WAS_CALLED` appears in no run's stderr**, on any rank,
across every run in this experiment. Behaviour-preserving and now positively
verified rather than merely argued.

### 2.4 Repair #4 (`packed_metadata` `"iiiiii"` → `"iii"`) — needed, verified

Applied to a copy by `tools/patch_rank1.py`. Triton 3.6.0 packs three ints;
rank-1's own C launcher parses six. The dropped `clusterDim{X,Y,Z}` are
forwarded to `_launch` and never read — only `num_warps` and `shared_memory`
reach `hipModuleLaunchKernel`. Full diff is 8 lines and contains nothing else.

Behaviour-preserving, and it matters that rank-1 keeps its own lean launcher
rather than falling back to Triton's, which would have understated it on the
small shapes.

### 2.5 Repair #5 (`iris.hip.hipIpcMemHandle_t` alias) — **MOOT. Not needed.**

This is the repair three sessions have been blocked on, and it patches dead code.

`iris.hip.hipIpcMemHandle_t()` is referenced in exactly one place: inside the
string literal `CREATE_SHEMEM_CODE`. And `CREATE_SHEMEM_CODE` occurs **exactly
once in the whole 1810-line file — at its own definition on line 38.** There is
no `subprocess`, no `Popen`, no `multiprocessing`, no `os.fork` / `os.exec` /
`os.spawn` anywhere in the submission; the only `os.system` calls are the `sudo
sed`, `rm -rf a.txt`, `rm -rf *.bin` and `touch now.txt`.

**Consequences:**

- **`heap_bases_*.pkl` can never be written by this submission.** Nothing ever
  executes the code that writes it. Its count was 0 because the file is
  unreachable, not because a helper was dying. The "cheap progress probe" that
  three sessions gated on was measuring dead scaffolding, and
  `tools/check_rank1d.sh` should be retired or rewritten.
- The real bootstrap artifact is **`ipc_handles_rank*.bin`**, written by rank-1's
  C++ `init_shmem` via `hipIpcGetMemHandle`. Those were *already* being produced
  for all 8 ranks in the previous session's logs — the bootstrap had been
  working the whole time.

The alias is harmless and is left installed, but it is not load-bearing and
never was.

### 2.6 Repair #6 — **the actual blocker.** `AMDGCN_USE_BUFFER_OPS=0`

An environment variable. **No source change.**

**Root cause.** rank-1 addresses peer heaps with a pointer trick: it passes a
100-element (200-byte) bf16 tensor `A_ptr_index_hack` as the kernel's base
pointer, and eight large *negative* element offsets as `tl.constexpr`:

```python
1573: A_ptr_index_hack = torch.empty(100, dtype=torch.bfloat16, device=a.device)
1576: base_addrs.append((base_ptr[i].item() - A_ptr_index_hack.data_ptr()) // 2)
1339: ptr_diff = tl.cast(heap_base_0, tl.int64)          # ... one per peer
1364: c_ptrs = a_ptr + ptr_diff + stride_cm*offs_cm[:,None] + stride_cn*offs_cn[None,:]
```

Measured magnitudes on this node are −6.6e8 … −4.4e9 elements, i.e. five of the
eight exceed int32 range.

Triton 3.6.0's AMD backend decides **per pointer argument** whether to lower
memory ops to `buffer_*` form, and the test is the declared size of the tensor:

```python
triton/backends/amd/compiler.py:184
    if knobs.amd.use_buffer_ops and HIPBackend.is_within_2gb(arg):
```

A 200-byte tensor trivially "is within 2 GB", so the epilogue store becomes
`buffer_store_dwordx2`, whose voffset field is **32 bits**. The offset is
correct in the TTIR —

```
%c_ptrs_40 = tt.addptr %a_ptr, %10 : !tt.ptr<bf16>, i64        # correct i64
tt.store %c_ptrs_48, %c cacheModifier = cg
```

— and is then silently narrowed during lowering. rank-1's pointer hack violates
exactly the invariant that test is trying to establish.

**Evidence that this is the mechanism, not a guess.** Four ranks faulted with
`Write access to a read-only page`. For every one of them the fault address is
`A_ptr_index_hack + 2 × int32_truncate(base_addr[i])` plus a small offset, and
never the true heap base:

| rank | GPU | fault address | matches | offset into that heap |
|---|---|---|---|---|
| 1 | dev 1 | `0x7ef8eae1c000` | `heap[6]` int32-truncated | +114 688 B |
| 2 | dev 2 | `0x7f7de2642000` | `heap[6]` int32-truncated | +270 336 B |
| 5 | dev 5 | `0x7f3706699000` | `heap[6]` int32-truncated | +626 688 B |
| 7 | dev 7 | `0x7f1ce32c6000` | `heap[5]` int32-truncated | +811 008 B |

Every offset is inside that shape's 896 KiB per-peer data region, so the *tile*
arithmetic was right and only the base was wrong. The exact (untruncated) base
would place each fault 2036 MiB into a 1 GiB heap — impossible. The truncated
addresses land in 32 GiB `---p` no-permission reserved VA regions, which is why
a write there is reported as a read-only page.

Everything else was ruled out first, with evidence: all eight IPC heaps map
`rw-s`, 1 GiB each; host-side writes to every peer heap succeed (`WRITE_OK` ×8);
and a stage bisect showed `init_shmem` **and** the C++ `dist_barrier` kernel —
which performs system-scope peer atomic *writes* — both succeed on all 8 ranks.
Only the Triton GEMM epilogue faulted.

**Effect of the fix.** The epilogue changes from `buffer_store_dwordx2` to
`global_store_dwordx2` (full 64-bit address), and correctness passes on all six
shapes on all eight ranks.

**Behaviour-preserving argument.** The kernel *cannot be correct at all* without
64-bit peer addressing, so 64-bit addressing is necessarily what its original
environment produced; buffer-op lowering here makes it incorrect. Disabling the
pass restores the addressing width the submission requires rather than changing
its algorithm, tiling, configs, launcher or data movement.

**Disclosed residual bias, and its direction.** The knob is global to rank-1's
Triton kernels, so the A/B operand loads — which legitimately *are* within 2 GB
and would benefit from buffer ops — also lose that lowering. Buffer ops save
address VGPRs, so this can only **penalize** rank-1 slightly. The bias runs
*against* the arm that already wins by 1.78×, so it cannot change the verdict;
if anything rank-1's true margin is larger. Our arm is hand-written HIP compiled
with `hipcc` and contains no Triton, so the knob cannot affect it at all.

### 2.7 `has_bias` — no repair, but a protocol finding that changes the setup

`eval.py`'s own cases parser types values with `int(val)` and keeps the raw
string on `ValueError`:

```python
eval.py:69   match = r"\s*([a-zA-Z_]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
eval.py:80-83  try: val = int(val)
               except ValueError: pass
```

So `has_bias: False` becomes the **string** `"False"`, which is truthy, and
`reference.generate_input`'s `if has_bias:` therefore builds a bias tensor for
**all six** graded shapes — including the three that declare `has_bias: False`.
`ref_kernel` adds it too, so the evaluator stays self-consistent and correctness
still passes.

rank-1's author evidently suspected this; the comment sits on `__conf` itself:

```python
1734: __conf = [
1735:     (64, 7168, 18432//8), #####?????? bench all have bias?
```

It matters because rank-1's cached fast path dereferences bias unconditionally:

```python
1457:             bias.data_ptr(),          # AttributeError if bias is None
```

With a genuinely absent bias, rank-1 raises `AttributeError: 'NoneType' object
has no attribute 'data_ptr'` on its **second** call — which is what happened
when the comparison harness first ran shapes 1, 4 and 6 with real booleans.

**Resolution taken:** the comparison below was run with a bias present on all
six shapes, i.e. reproducing what the official evaluator actually does, for both
arms identically. No patch to rank-1 was needed. A *genuine* `has_bias=False`
comparison is not possible for rank-1 without repairing that line.

### 2.8 Integrity check — no silent torch fallback

`launch_triton_kernel` silently returns a torch implementation for any shape it
has no tuned config for:

```python
1468:         if (M, N, local_K) not in __conf:
1469:             return origin((a, b, bias))       # F.linear + reduce_scatter_tensor
```

Verified all six graded shapes are present in `__conf`, `online_config`
**and** `online_config_group`, so every number below is rank-1's own kernel and
not torch wearing its name.

---

## 3. The comparison — same run, interleaved, paired

`experiments/exp_10_rank1/mp_vs_rank1.py`. Both arms in the **same** 8-process
pool, on the **same** inputs, alternating, with the arm order reversed every rep
so neither is systematically first. One fresh pool per shape (rank-1's compiled
kernel lives in module globals keyed by nothing, so it is only valid for the
shape it was built for; the evaluator gets away with it by destroying the
process group between shapes).

Per timed iteration, the evaluator's `full` protocol verbatim: clone the input,
`clear_l2_cache()`, `synchronize` + `barrier`, time, `synchronize` + `barrier`.
15 iterations × 2 reps × 8 ranks = **240 samples per arm per shape**.

| # | shape | arm | best | median | mean | sd% | worst |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | ours | 181.78 | 195.05 | 205.06 | 17.5 | 367.77 |
| 1 | | **rank-1** | **176.05** | **181.72** | **183.06** | 2.3 | 192.84 |
| 2 | 512×4096×12288 | ours | 187.17 | 201.62 | 211.99 | 16.7 | 372.41 |
| 2 | | **rank-1** | **135.46** | **140.38** | **142.67** | 5.7 | 182.58 |
| 3 | 2048×2880×2880 | ours | 194.00 | 205.09 | 213.50 | 16.1 | 394.63 |
| 3 | | **rank-1** | **153.25** | **164.42** | **165.39** | 3.2 | 185.50 |
| 4 | 4096×4096×4096 | ours | 810.13 | 829.96 | 840.01 | 4.5 | 1036.33 |
| 4 | | **rank-1** | **262.22** | **285.72** | **284.49** | 4.2 | 312.98 |
| 5 | 8192×4096×14336 | ours | 745.94 | 765.99 | 782.56 | 6.0 | 945.72 |
| 5 | | **rank-1** | **571.97** | **592.87** | **596.10** | 2.2 | 622.03 |
| 6 | 8192×8192×29568 | ours | 6469.97 | 6644.83 | 6672.38 | 2.3 | 7301.74 |
| 6 | | **rank-1** | **1474.85** | **1525.06** | **1549.55** | 5.0 | 2075.27 |

All times µs.

### Geometric means — the competition's ranking statistic

| arm | geo best | geo median | geo mean |
|---|---:|---:|---:|
| ours | 543.61 | 569.38 | 586.52 |
| **rank-1** | **305.22** | **320.49** | **322.98** |
| ours / rank-1 | **1.781×** | **1.777×** | **1.816×** |

**rank-1 is faster on every one of the six shapes**, by 1.03× on shape 1 and by
1.27–4.39× on the other five.

### Per-shape ratio (ours / rank-1, >1 means rank-1 faster)

| shape | best | mean |
|---|---:|---:|
| 1 | 1.033× | 1.120× |
| 2 | 1.382× | 1.486× |
| 3 | 1.266× | 1.291× |
| 4 | **3.090×** | 2.953× |
| 5 | 1.304× | 1.313× |
| 6 | **4.387×** | 4.306× |

### Correctness, both arms, both tolerances

| # | shape | arm | 1e-2 | 2e-3 | max\|diff\| |
|---:|---|---|---|---|---:|
| 1 | 64×7168×18432 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.47e-3 |
| 2 | 512×4096×12288 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.47e-3 |
| 3 | 2048×2880×2880 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.47e-3 |
| 4 | 4096×4096×4096 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.47e-3 |
| 5 | 8192×4096×14336 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.47e-3 |
| 6 | 8192×8192×29568 | ours / rank-1 | pass / pass | pass / pass | 9.77e-4 / 1.95e-3 |

Oracle is `matmul` + `all_reduce` + the rank's row slice, the same one `mp_smoke`
uses. Nothing here is a number for an unverified kernel.

---

## 4. Cross-check against the official evaluator

rank-1 was also run through the official `eval.py benchmark` on the same six
shapes. It completed three before hitting its 1500 s wall (`exit=143`); shapes 4–6
were not reached. `eval.py` reports **nanoseconds** (calibrated against the
reference run recorded in `LESSONS.md`, which was written up in µs):

| # | shape | eval.py best | eval.py mean | interleaved best | interleaved mean |
|---:|---|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 217.25 | 232.77 | 176.05 | 183.06 |
| 2 | 512×4096×12288 | 186.26 | 219.22 | 135.46 | 142.67 |
| 3 | 2048×2880×2880 | 195.62 | 255.48 | 153.25 | 165.39 |

Same magnitude, evaluator slightly higher — expected, since `eval.py` runs up to
100 iterations and its `worst` samples reach 628–2852 µs, dragging the mean. The
interleaved figures are the ones to quote, because they are paired.

No memory access faults, no `SHIM_WAS_CALLED`, and `check` did not fail on any
completed shape.

### For context only, not comparable

rank-1's published score is **413.139 µs**, from GPU MODE's machine under their
protocol. Measured here it is **305.22 µs** (geo best) / **322.98 µs** (geo
mean). Same order of magnitude, which is a sanity check on the whole setup, but
the two numbers are not comparable and the published one should not be used as
the denominator now that a local measurement exists.

---

## 5. Where this leaves the ratchet

The charter's ratchet level 1 is now available: **rank-1 measured on this node,
geo best 305.22 µs / geo mean 322.98 µs**, and we are **1.78× behind it**.

The margin is not uniform, and that is the actionable part:

- **Shape 6 alone is the story: 4.39×** (6470 µs vs 1475 µs). Our own pipelined
  harness reports 1828.89 µs for shape 6, so the graded protocol costs us ~3.5×
  on that shape while rank-1 loses almost nothing. This is the same collapse
  already recorded for the evaluator (ours 7155 µs vs reference 1719 µs) and it
  is worth more than every mainloop experiment combined.
- **Shape 4 is second at 3.09×** (810 µs vs 262 µs), and our 810 µs against our
  own pipelined 200.33 µs is a 4× protocol penalty — the largest ratio of the
  six.
- **Our variance is the other signal.** On the three small shapes our relative
  sd is 16–18% against rank-1's 2–6%, and our worst samples are ~2× our median
  while rank-1's are ~1.1×. The graded score is a **mean**, so that tail is
  charged to us directly.
- Shape 1 is nearly a tie (1.03× best), so the small-shape floor is not the gap.

The gap is concentrated exactly where the existing evidence already pointed: the
cold-L2 / per-call egress path on the large shapes, not the GEMM mainloop.

---

## 6. Reproducing this

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1

# one-time, needs root IN OUR CONTAINER (not dhk-eval):
bash $E/p5_iris.sh                       # stage iris + verify all repairs

# stage + patch a copy of the frozen submission (refuses on hash mismatch):
bash $E/p4_stage.sh

# the comparison (shapes with a native bias / shapes needing VS_FORCE_BIAS=1):
bash $E/vs_sweep.sh 1,2,4 15 2 12500 0
bash $E/vs_sweep.sh 0,3,5 15 2 12600 1
bash $E/p16_final.sh                     # aggregate + provenance
```

Every rank-1 run needs `AMDGCN_USE_BUFFER_OPS=0` **and** a cleared
`TRITON_CACHE_DIR` — the cached `hsaco` has `buffer_store` baked in, so leaving
the cache in place makes the knob look ineffective.

## 7. Artifacts

| file | what |
|---|---|
| `p1_survey.sh` … `p16_final.sh` | the no-GPU probe ladder, in order |
| `r1_min.py` / `r1_min.sh` | minimal 8-rank driver, per-rank fd-level stderr |
| `r1_bisect.py` / `r1_bisect.sh` | stage bisect: init / dist_barrier / GEMM |
| `r1_ptrs.py` / `r1_ptrs.sh` | IPC heap pointer table + page permissions |
| `r1_run.sh` | generic runner with the buffer-ops knob |
| `r1_eval.sh` | the official evaluator, either arm |
| `mp_vs_rank1.py` / `vs_sweep.sh` / `vs_report.py` | the interleaved comparison |
| `vs_logs/vs_s*.rank*.json` | every raw sample, per rank per arm |

## 8. Methodology notes worth keeping

- **Drive rank-1 with a minimal 8-rank driver, never `eval.py`, while
  debugging.** The fault reproduced in ~90 s instead of ~25 min, and per-rank
  `os.dup2` on fd 2 is what preserved the HSA fault text — a GPU memory fault
  aborts the process from the runtime, so anything buffered in Python's
  `sys.stderr` object is lost.
- **`AMD_SERIALIZE_KERNEL=3` is what turned an opaque abort into a traceback**
  pinned to `submission.py:1038 → jit.py:744 → launch_triton_kernel:1503`.
- **Dump `/proc/self/maps` before the fault and match the address afterwards.**
  That is what separated "IPC mapping is read-only" (false — all `rw-s`) from
  "the computed address is wrong" (true).
- **A progress probe you have not proven is reachable is worse than none.**
  `heap_bases_*.pkl` cost three sessions.
