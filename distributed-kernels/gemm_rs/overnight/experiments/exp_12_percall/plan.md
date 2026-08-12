# exp_12 — the ~100 µs per-call fixed cost: localize it first

## The finding being attacked

Graded best minus our own pipelined device time, per shape (µs):

| # | shape | graded best | pipelined device | difference |
|---|---|---|---|---|
| 1 | 64×7168×18432 | 181.19 | 77.85 | 103 |
| 2 | 512×4096×12288 | 186.69 | 88.47 | 98 |
| 3 | 2048×2880×2880 | 194.31 | 91.06 | 103 |
| 4 | 4096×4096×4096 | 312.94 | 202.56 | 110 |
| 5 | 8192×4096×14336 | 762.76 | 651.61 | 111 |
| 6 | 8192×8192×29568 | 1970.98 | 1818.67 | 152 |

Nearly constant. On shape 1 it is 57% of the runtime. exp_07 step 2 already
proved it is **not** the clone and **not** the L2 flush (`bare` arm was
186.9 µs against 201.1 µs pipelined on shape 4, i.e. the residual survives
removing both), but it never went further than that.

## Step 1 — decompose, before writing any kernel change

The graded timed region is

```
clear_l2_cache(); torch.cuda.synchronize(); dist.barrier()
t0 = perf_counter_ns()
custom_kernel(_clone_data(data))
torch.cuda.synchronize(); dist.barrier()
t1 = perf_counter_ns()
```

`mp_percall.py` reproduces it in 8 processes and inserts **three extra
timestamps** inside the region, which costs nothing and splits it four ways:

```
t0   -> t_pay   the clone            (evaluator's _clone_data)
t_pay-> t_iss   the HOST path        (python + pybind + hipLaunchKernel)
t_iss-> t_syn   the DEVICE           (execution + drain)
t_syn-> t1      the trailing barrier (NCCL, waits for the slowest rank)
```

`time.perf_counter_ns()` is `CLOCK_MONOTONIC`, so the eight ranks' stamps are
directly comparable on one host. That gives a fifth quantity for free:
**barrier-exit skew**, `max_r(t0) − min_r(t0)`, which matters because our grid
cannot make progress until every rank's grid is up (rank A's producers write
into rank B's heap and rank B's reducers wait on it).

Arms, all under the identical structure, interleaved, order flipped every rep:

| arm | work inside the timed region | isolates |
|---|---|---|
| `floor` | nothing at all | sync + barrier + timer floor |
| `null1` | empty kernel, grid 1, 512 thr, 0 LDS | one bare `hipLaunchKernel` |
| `null76` / `null152` / `null304` | empty kernel, 512 thr, 64 KB dyn LDS | **does launch cost scale with CTA count?** |
| `null304l0` | empty kernel, grid 304, 0 LDS | does the 64 KB request (1 CTA/CU) cost |
| `epoch304` | 304 CTAs each running `epoch32` verbatim + the sticky error-bit load | the start-up handshake |
| `ours` | the real kernel, no clone, no flush | protocol + GEMM + drain |
| `oursclone` | real kernel + clone | the clone |
| `oursfull` | real kernel + clone + flush | the graded number, verbatim |

Deltas: `null304 − null1` = grid-residency cost; `epoch304 − null304` = the
start-up handshake; `ours − epoch304 − pipelined device` = drain + the rest;
`oursfull − ours` = clone + flush.

`nullk.cpp` is a standalone probe module. It matches our kernel's launch
geometry exactly (`__launch_bounds__(512, 1)`, dynamic LDS via
`hipFuncSetAttribute`, per-CTA epoch cell) and touches no kernel source.

## Step 2 — only after step 1

Directed by the decomposition, not chosen in advance. If it is irreducible
launch/barrier latency that is a finding and this axis closes.

## Files owned

`experiments/exp_12_percall/**`, plus one additive change to
`harness/hk_submission.py`: an `HK_KERNEL_MODULE` env override so an ablation
arm (`gemm_rs_abl_noproto`) can be driven through the same multi-process path.
Nothing else is touched.
