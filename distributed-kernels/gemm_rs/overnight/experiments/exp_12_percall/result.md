# exp_12 — the ~100 µs per-call fixed cost: where it goes, and the ~20 µs of it that was ours

**Verdict in one line: about 70 µs of the ~100 µs is the evaluator's own
`torch.cuda.synchronize() + dist.barrier()`, another ~15–20 µs is its
`_clone_data`, the device pays nothing per call, and the only part that was ours
was a ~20–35 µs python/pybind host path. Removing that host path is worth
7–10% of the graded geometric mean and moves us from 1.238× behind the frozen
rank-1 to 1.167×, with shape 1 flipping to a win.**

Measured on `banff-sc-cs47-05.dh170.dcgpu`, 8× MI300X gfx942 SPX, container
`dhk-gemmrs`, clocks at `perf_determinism`, node verified clean before every
run. Exclusive GPU access; one 8-GPU job at a time.

---

## Step 1 — the decomposition

### The instrument

`mp_percall.py` reproduces the graded timed region verbatim in eight processes
and inserts three extra timestamps inside it, which costs a few hundred
nanoseconds and splits it four ways:

```
t0    -> t_pay   the clone            (the evaluator's _clone_data)
t_pay -> t_iss   the HOST path        (python + pybind + hipLaunchKernel)
t_iss -> t_syn   the DEVICE           (execution + drain)
t_syn -> t1      the trailing barrier (NCCL; waits for the slowest rank)
```

`time.perf_counter_ns()` is `CLOCK_MONOTONIC`, so all eight ranks' stamps are
comparable on one host. That also gives barrier-exit skew and issue skew for
free, which matter here because no rank's grid can finish before every other
rank's grid is up.

### Where the ~100 µs actually goes

`oldfull` (the graded arm exactly as it was), medians over 40 iterations, µs.
`resid` is what the four components do not explain — the wait for the slowest
rank to issue, which is real and charged to the score but belongs to no single
component.

| # | shape | total | clone | **host** | device | **barrier** | resid |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 247.9 | 21.9 | **38.6** | 82.3 | **86.3** | 18.7 |
| 2 | 512×4096×12288 | 230.9 | 21.0 | **26.0** | 87.9 | **74.7** | 21.3 |
| 3 | 2048×2880×2880 | 239.7 | 21.2 | **26.4** | 94.2 | **75.1** | 22.8 |
| 4 | 4096×4096×4096 | 335.2 | 15.7 | **25.5** | 196.7 | **75.8** | 21.6 |
| 5 | 8192×4096×14336 | 776.5 | 21.5 | **25.8** | 629.7 | **80.0** | 19.6 |
| 6 | 8192×8192×29568 | 1927.0 | 13.9 | **21.5** | 1823.9 | **73.9** | −6.2 |

So the ~100 µs is, in order of size:

1. **The trailing `synchronize` + `barrier`: 74–86 µs.** This is the evaluator's,
   not ours, and every submission pays it. Proven by the `floor` arm — the
   identical structure with **nothing at all** inside the timed region — which
   costs **57–79 µs**, essentially all of it in the barrier. It also varies
   between runs (54–86 µs) with host state, which is why the small shapes'
   graded numbers are noisy at the 10% level no matter what the kernel does.
2. **Our host path: 21–39 µs.** The only reducible term. See below.
3. **The evaluator's `_clone_data`: 14–22 µs**, and it scales with the input
   bytes exactly as a device-to-device memcpy should. Also not ours.
4. **The device: nothing.** Graded device time equals our own pipelined device
   time on every shape. Against M7's `dev_max` (75.9 / 87.0 / 90.0 / 199.2 /
   654.2 / 1830.7) the graded arm measures 83.1 / 92.1 / 94.8 / 202.1 / 637.8 /
   1783.1 — within ±3%, and shapes 5 and 6 come out *below* it. **There is no
   per-call device penalty at all.**

### All three device-side candidate mechanisms are falsified

`nullk.cpp` launches probe kernels that match the real kernel's launch geometry
(512 threads, `__launch_bounds__(512, 1)`, dynamic LDS reserved through
`hipFuncSetAttribute`) but do nothing. Medians over 50 iterations under the
identical barrier-bracketed structure, µs:

| arm | grid | LDS | shape 1 block | shape 4 block |
|---|---:|---:|---:|---:|
| `floor` (no kernel at all) | — | — | 78.7 | 57.5 |
| `null1` | 1 | 0 | 94.6 | 70.6 |
| `null8` | 8 | 64 KB | 90.8 | 70.1 |
| `null76` | 76 | 64 KB | 90.0 | 70.5 |
| `null152` | 152 | 64 KB | 90.2 | 71.0 |
| `null304` | 304 | 64 KB | 89.9 | 71.1 |
| `null304l0` | 304 | 0 | 90.1 | 70.3 |
| `null608` | 608 | 64 KB | 90.4 | 70.8 |
| `epoch304` | 304 | 64 KB | 90.4 | 70.6 |

- **"Launching and draining a 304-CTA persistent grid" costs nothing extra.**
  1 CTA and 608 CTAs are the same number to within 1%. A launch costs 12–16 µs
  over the floor and that cost is entirely per-launch, not per-CTA. **Candidate
  1 falsified** — and with it any intervention that shrinks the grid.
- **The 64 KB dynamic-LDS request (1 workgroup/CU) costs nothing**: `null304l0`
  = `null304`.
- **The start-up handshake costs nothing.** `epoch304` runs `epoch32` verbatim
  on the same per-CTA cell layout plus the sticky error-bit load in all 304
  CTAs, and is indistinguishable from the empty kernel. 304 atomics at kernel
  start are not a serialization point. **Candidate 2 falsified**, so the
  epoch/credit protocol was never opened and
  `exp_05_release_granularity/protocol_review.md` §2 did not need to be
  re-litigated.
- **Kernel drain costs nothing** (candidate 4): graded device time = pipelined
  device time, above.
- **`hk_submission`'s per-call path is a cache hit in steady state** (candidate
  5) — but the cache hit itself was expensive, which is candidate 3, and it is
  the answer.

### What the host path was made of

`03_hostprofile.py`, in the real eight-rank pool (eight python interpreters on
one host is itself part of the cost). Rank 0, µs per call:

| component | µs |
|---|---:|
| whole `custom_kernel` | 23.5 |
| `state.launch` alone | 19.2 |
| the pybind entry with every argument prebuilt | 7.5–12.7 |
| `torch.cuda.current_stream(rank).cuda_stream` | 2.3 |
| two `Tensor` duck-type constructions | 2.1 |
| three eagerly-built log f-strings | 0.7 |
| `dist.get_rank()` | 0.7 |
| eight `int(plan[...])` lookups | 0.4 |
| `torch.cuda.set_device` | 0.4 |

The dominant term is inside pybind. `pyutils`' `from_object<GL>` converter
re-derives every tensor argument from a python object on **every call**:
`__class__.__name__`, `is_contiguous()`, `device.type`, `shape` and
`data_ptr()` — about eight attribute lookups, two python calls and two
`std::string` comparisons per tensor, four tensors deep. All of it recomputes
values that cannot change on a cached (rank, shape) state.

Note the absolute numbers here are a *lower* bound: the tight loop keeps
everything hot, while in the graded structure the host has just returned from a
device sync and a barrier, and the same path measures 21–39 µs.

---

## Step 2 — the intervention

A **prebound launch path**, additive, in two parts.

1. `gemm_rs_mi300x.cpp`, `namespace prebound`: `configure(...)` binds everything
   invariant for one (rank, shape) into a `mi300x_globals` held in a per-process
   table and returns an opaque handle; `run(handle, a, b, bias, stream)` copies
   that POD, writes the three pointers and the stream, and calls the same
   `dispatch_gemm_rs_mi300x`. Same POD, same dispatch table, still exactly one
   kernel launch per call (gate 16). The original `gemm_rs_mi300x` binding is
   untouched, because the single-process validation harness drives it.
   The handle is deliberately **not** a cache keyed on shape: a module-global
   keyed on nothing is exactly the hazard that makes the competitor's cached
   launch path raise on its second call with a new shape.
2. `harness/hk_submission.py`: `_ShapeState.__init__` calls `configure` from the
   same place it builds its descriptors, and `custom_kernel` gains a fast path
   that makes the **same** decision the old one did — same group-identity test,
   same key, same state — with only the redundant work removed: the rank comes
   out of the cached key instead of a c10d round trip, log f-strings are not
   built when nothing will read them, the stream comes from
   `torch._C._cuda_getCurrentRawStream` instead of a python `Stream` object, and
   the launch goes through the handle. Anything unexpected falls through to the
   original path, which is unchanged. `HK_DEBUG=1` always takes the full path,
   since its whole purpose is the per-call synchronize and error-bit read.

The stream is read per call rather than captured at configure time: it is the
one thing that can legitimately change without the state changing.

### Effect on the host path

Same run, same pool, interleaved, arm order flipped every rep, `_FAST` toggled
between arms — `oldfull` and `oursfull` are the same code path apart from the
intervention. Median host µs:

| # | old | new |
|---|---:|---:|
| 1 | 39.1 | **7.4** |
| 2 | 22.3 | **4.8** |
| 3 | 21.5 | **4.9** |
| 4 | 21.5 | **4.8** |
| 5 | 21.7 | **5.0** |
| 6 | 21.5 | **4.8** |

4.8 µs against a measured floor of 4.86 µs for a bare pybind call plus
`hipLaunchKernel` on this node. **The host path is now at its floor.**

Issue skew across ranks collapsed with it (shape 1: 29.2 → 12.9 µs; shape 2:
6.6 → 4.6), and `resid` — the wait for the slowest rank to issue — fell with
it, so the total gain exceeds the host-mean reduction. The host path's
*variance* across ranks was costing as much as ~10 µs on top of its mean,
because the protocol runs at the pace of the last rank to launch.

### Effect on the graded per-call figure

Two independent paired runs, best and median of 40 iterations per shape:

| # | run ab1 old → new (best) | ratio | run ab2 old → new (best) | ratio |
|---|---|---:|---|---:|
| 1 | 236.5 → 199.0 | 0.842 | 217.6 → 191.5 | 0.880 |
| 2 | 225.9 → 191.8 | 0.849 | 185.2 → 166.9 | 0.901 |
| 3 | 230.1 → 201.2 | 0.874 | 191.9 → 171.1 | 0.891 |
| 4 | 328.2 → 294.8 | 0.898 | 292.5 → 274.8 | 0.940 |
| 5 | 759.6 → 739.4 | 0.973 | 730.9 → 722.9 | 0.989 |
| 6 | 1898.1 → 1885.4 | 0.993 | 1889.1 → 1869.3 | 0.990 |
| | **geomean 424.1 → 383.0** | **0.903** | **geomean 382.3 → 355.9** | **0.931** |

Medians: 436.2 → 392.0 (0.899) and 395.1 → 365.1 (0.924). The absolute level
differs between runs because the evaluator's barrier does (see the `floor` arm);
the paired delta is what is comparable, and it is **−7% to −10% on the graded
geometric mean**, concentrated on the four smaller shapes exactly as a constant
additive saving must be.

### Effect against the frozen rank-1 submission

`06_vs_rank1.sh` re-runs exp_10's `mp_vs_rank1.py` under exp_10's rematch
protocol unchanged — both arms in one 8-process pool, alternating, order
reversed every rep, one fresh pool per shape, `VS_FORCE_BIAS=1`, 192 pooled
samples per arm per shape. Only `VS_OUT` differs, so exp_10's logs stand.

| # | shape | ours before | **ours now** | rank-1 now | ratio before | **ratio now** |
|---|---|---:|---:|---:|---:|---:|
| 1 | 64×7168×18432 | 181.19 | **170.12** | 177.34 | 1.011 | **0.959 WIN** |
| 2 | 512×4096×12288 | 186.69 | **165.46** | 134.18 | 1.375 | **1.233** |
| 3 | 2048×2880×2880 | 194.31 | **172.50** | 155.64 | 1.233 | **1.108** |
| 4 | 4096×4096×4096 | 312.94 | **294.09** | 266.68 | 1.178 | **1.103** |
| 5 | 8192×4096×14336 | 762.76 | **741.61** | 572.47 | 1.323 | **1.295** |
| 6 | 8192×8192×29568 | 1970.98 | **1969.49** | 1458.45 | 1.350 | **1.350** |
| | **geomean (best)** | 381.69 | **357.45** | 306.23 | **1.238** | **1.167** |
| | **geomean (median)** | 404.03 | **369.82** | 318.26 | 1.266 | **1.162** |
| | **geomean (mean)** | 446.13 | **382.06** | 320.98 | 1.364 | **1.190** |

rank-1 reproduced its own numbers to within 0.7% (306.23 against 308.25 best),
which is what licenses the before/after comparison. All twelve arm-shape pairs
correct at both `1e-2` and `2e-3`.

**Shape 1 is now a win.** The remaining deficit is concentrated on shapes 5 and
6 (1.295×, 1.350×), the two large-egress shapes — that is E2's territory, not
this axis'.

### Effect on the pipelined figure: none, as predicted

M7, 3 rotations × 50: `77.72 / 88.51 / 91.56 / 201.20 / 655.97 / 1835.31`,
geomean **231.16 µs** against the 230.65 denominator — unchanged to 0.2%. The
single-process harness drives the original entry point, so it cannot see this
change, and the brief's warning that "the pipelined geomean will barely move
even on a win" is exactly what happened.

---

## Gate verdicts

Full ladder, `gate_ladder.sh exp_12_percall`, in order, all green:

| gate | verdict |
|---|---|
| M0 node clean | 0 KFD pids, all 8 GPUs `perf_determinism` |
| M1 build | all three modules; entry points `gemm_rs_mi300x`, `..._configure`, `..._run` |
| M2 resources / ISA | 6 instantiations, VGPR 91/91/163/246/248/91, **AGPR 0, scratch 0, no VGPR spills**, no scratch sites |
| M3 correctness | **17/17 shapes PASSED at both 1e-2 and 2e-3** |
| M4 negative controls | 3/3 fail exactly as designed (drop-publication → bit 26 on rank 5 and zero payload reads; drop-credit → bit 25 at epoch 2; reroute-slot → detectable bitwise corruption) |
| M5 soak | 600 skewed changing-input epochs, epoch and signal cells exact, worst max\|diff\| 4.883e-04 |
| M7 timing | geomean 231.16 µs pipelined, all shapes correct |

The ladder drives `harness_lib`, i.e. the **original** entry point, so it proves
the kernel binary did not regress but does not by itself exercise the new path.
That is covered separately, and more strictly:

- **`04_equiv.py`: the two launch paths are bit-identical.** On all six graded
  shapes on all eight ranks, `state.launch` and `state.launch_fast` on the same
  inputs give `torch.equal(ref, got) == True`, `max|diff| = 0.000e+00`, output
  non-zero, error bits 0. Both paths launch the same kernel, so the only thing
  that can differ is the POD they build, and a wrong field could still pass an
  allclose on some shapes — bit equality cannot.
- Every `mp_percall.py` run verifies the last arm's output against the
  all-reduce oracle on all eight ranks at both `1e-2` and `2e-3`; all passed.
- No tolerance was widened, no iteration count or shape changed, and no error
  bit was ever set.

---

## Is what is left reducible? No, and that closes this axis

Remaining non-device cost per call, after the fix (shape 1, `oursfull`):

| term | µs | ours? |
|---|---:|---|
| trailing `synchronize` + `barrier` | 72.6 | **no** — the evaluator's, 57–79 µs with an empty timed region |
| `_clone_data` | 19.6 | **no** — the evaluator's, scales with input bytes |
| host path | 7.4 | ours, and at the 4.9 µs pybind+launch floor |
| wait for the slowest issuer | 3.4 | ours only indirectly; it is OS scheduling of 8 processes after a barrier |

Roughly 92 of the remaining ~103 µs is harness machinery that every submission
pays, including rank-1's. The ~11 µs that is ours is within ~6 µs of the floor
for issuing any HIP kernel at all from python. **There is no second act on this
axis**; the next per-call microsecond would have to come from removing python
from the call entirely (a C++ `custom_kernel`), for a ceiling of ~5 µs.

The corollary matters for the standing ratchet: because ~92 µs of every graded
call is a constant charged to both arms, the *kernel-to-kernel* ratio against
rank-1 is worse than the 1.167× headline. Subtracting it, shapes 5 and 6 are
1.36× and 1.38× on device work alone. **E2 (XGMI egress on the two large
shapes) is where the rest of the gap lives.**

---

## Files

| file | what |
|---|---|
| `plan.md` | pre-registered arms, deltas and expectations |
| `nullk.cpp` | probe kernels: empty, and `epoch32` + error-bit check, at our launch geometry |
| `mp_percall.py` | the 8-process graded harness with the five timestamps and all twelve arms |
| `03_hostprofile.py` | itemization of the host path in the real pool |
| `04_equiv.py` | bit-exact equivalence of the two launch paths |
| `00_build.sh` | builds `nullk.so`, smoke-tests it, measures the bare pybind+launch floor |
| `00_presync_check.sh` | LF-form hashes of everything `push.ps1` overwrites, run before pushing |
| `01_run.sh` | driver for `mp_percall.py` |
| `02_rebuild.sh` | rebuild + M1/M2, so no number is attributed to a stale `.so` |
| `03_run_hostprofile.sh`, `05_rebuild_sync_equiv.sh`, `06_vs_rank1.sh` | drivers |
| `logs/`, `vs_logs/` | every raw sample, per rank, per arm |

Sources changed: `gemm_rs_mi300x.cpp` (additive, after `dispatch_gemm_rs_mi300x`
and in the binding block; the kernel template is untouched) and
`harness/hk_submission.py`. Nothing under `include/**`, `tools/**` or
`gemm_rs_mi300x_host_abi.hpp` was modified.

**Operational note that cost a step to learn:** `harness/submission.py` is a
*copy* of `hk_submission.py` that exists only on the node — the evaluator and
every `mp_*` harness import `submission`. Editing `hk_submission.py` without
recopying measures the old code and looks exactly like a null result.
`05_rebuild_sync_equiv.sh` does the resync and asserts `cmp` afterwards.
