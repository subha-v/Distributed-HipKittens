# exp_01 — our kernel under the official evaluator

**Status at handoff: blocked, one suspect ranked #1 with a concrete fix.**
This is mission step 1 and nothing downstream is worth doing first: until it
lands, no number we produce is comparable to the competitor's.

## Goal

`eval.py test cases.txt` passes 11/11 and `eval.py benchmark cases_bench.txt`
returns six per-shape means for our kernel, one process per rank, under
`torch.distributed`, with a HIP IPC symmetric heap.

## What already works — do not rebuild it

- `harness/hk_submission.py` implements the evaluator's `custom_kernel(data)`
  contract over our kernel: per-rank heap allocation, `hipIpcGetMemHandle` /
  `all_gather_object` / `hipIpcOpenMemHandle` exchange, then the **production**
  host ABI (`snapshot_allocation_descriptors`) unmodified.
- `harness/mp_smoke.py` runs that path outside the evaluator, 8 ranks, real
  NCCL process group, and gets `allclose=True, max|diff|=9.766e-04` on
  512×4096×12288. **The cross-process protocol is proven.**
- Two integration bugs are already fixed: state keyed by rank + process-group
  identity (the evaluator's pool reassigns workers between test cases), and the
  removal of all teardown between cases.

## The failure

Under `eval.py benchmark`, the warmup case completes and the first *timed* case
hangs. Instrumented stderr shows **6 of 8 ranks** printing `state ready`; two
are stuck inside `_ShapeState.__init__`. Per-step logging
(`get_ipc_handle` → `all_gather_object` → `opened all peers` → `setup barrier`)
was added but the run that would localize it was never completed. **Run that
first** — it is one `HK_ONE=1 bash tools/run_ours_evaluator.sh` away and prints
the full stderr.

## Suspect #1 and the fix to try

`hipIpcOpenMemHandle` returning `hipErrorAlreadyMapped`. We never free, but we
*do* allocate a fresh heap per process group; if the allocator returns an
address whose IPC handle this process already mapped, the open throws and that
rank dies inside setup — which would hit some ranks and not others, matching
the 6-of-8 signature exactly.

Fix: **cache state by `(rank, m, n, k, has_bias)` persistently across process
groups** and skip the exchange entirely for a shape already set up. All ranks
then skip the collectives together, so setup stays symmetric; the retained peer
pointers are still valid because nothing was freed and the processes are the
same; and epoch counters stay consistent because every rank reuses the same
state the same number of times.

Suspects #2 (NCCL barrier is enqueued, not blocking — a `torch.cuda.synchronize`
was added but never validated) and #3 (pool worker respawn breaking collective
symmetry) are in `../../HANDOFF.md`.

## Validation

```bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
# fast loop, full stderr, no evaluator timeouts in the way:
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'cp hk_submission.py submission.py; HK_DEBUG=1 python3 -u mp_smoke.py'
# then the real thing, one shape:
HK_ONE=1 bash $ON/tools/run_ours_evaluator.sh
# then the full suite:
bash $ON/tools/run_ours_evaluator.sh
```

Done when: test mode reports `check: pass` for 11/11, and benchmark mode
reports six `benchmark.N.mean` values with plausible spreads. Record all six
means, the geomean, and the ratio to our own harness's 285.7 µs — the gap
between the two protocols is itself a result worth writing down.

## Deliverables

`result.md` with: the localized root cause, the fix, 11/11 test output, the six
benchmark means + geomean, and the protocol-gap observation. Then append to
`../LESSONS.md`.
