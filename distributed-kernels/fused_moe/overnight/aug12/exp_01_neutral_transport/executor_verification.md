# Executor verification plan

An API name is not executor evidence. `host_copy_path`, `mori_put_path`, and
their timing points keep executor `unknown` until the exact size/configuration
has a trace artifact.

## Available profiler

Node/container probe:

- `rocprofv3` 1.1.0, git `c2d9476…`, ROCm 7.2.3
- supports HIP runtime, HSA, kernel, memory-copy, and memory-allocation traces
- supports queue grouping with `--group-by-queue`
- MPI placement is inside the launcher:
  `mpirun -n 2 rocprofv3 ... -- ./rank_transport ...`

## Host copy trace

For every selected record-size point:

```bash
mpirun -n 2 --bind-to numa \
  rocprofv3 \
    --hip-runtime-trace \
    --hsa-amd-trace \
    --memory-copy-trace \
    --kernel-trace \
    --group-by-queue \
    --output-format json \
    --output-directory <point-trace-dir> \
    -- ./rank_transport --method host_copy_path <exact-point-args>
```

Acceptance:

1. the timed transfer appears in the memory-copy domain
2. bytes and direction match the point schema
3. no CU payload kernel appears for the transfer
4. the copy's agent/queue metadata identifies the executor path
5. every measured size/configuration has its own trace

If the trace cannot distinguish an SDMA executor from another copy path, keep
the method labeled `host_copy_path`; it may enter the complete-path crossover
but not the hardware-engine map.

### 64 KiB anchor result

The first exact anchor trace resolves the path as **CU**, not SDMA:

- 8,192 `hipMemcpyPeerAsync(..., sizeBytes=65536, ...)` calls were traced.
- The memory-copy domain emitted zero records.
- Each inspected peer-copy API correlation directly names a
  `__amd_rocclr_copyBuffer` kernel dispatch on the same stream/queue. For
  example, correlation 3354 is a 64 KiB device 1→0 API call and kernel-id 8
  dispatch on queue 2.
- Kernel ids 8 and 16 are the two devices' `__amd_rocclr_copyBuffer`
  specializations; the trace contains 8,250 such dispatches total. The 58
  dispatches beyond the 8,192 peer-copy calls are auxiliary runtime copies,
  so the API count—not the raw symbol count—is the payload count.
- No `hsa_amd_memory_async_copy_on_engine` call was observed.

This falsifies acceptance item 3's assumption that no CU payload kernel would
appear. For CU-lowered host APIs, a direct API→kernel correlation is positive
executor evidence even though the memory-copy domain is empty. The adjudication
record is `raw/host_copy_anchor_executor_v1.json`; the full trace is
`raw/host_copy_anchor_trace_v1.json`.

## MORI same-API pair

The installed MORI JIT sources contain both P2P and SDMA specializations for
`ShmemPutMemNbi*` and dispatch through `GpuStates::transportTypes[pe]`.
That establishes build capability, not the runtime selector. Before timing:

1. pin the installed JIT-source tree hash/version
2. identify and record how `transportTypes[pe]` is populated
3. force P2P and SDMA without changing API, grouping, heap type, QP count,
   quiet/completion, or publication
4. collect executor traces for both methods at every size
5. run the unordered payload/signal negative control for SDMA

Without a verified force mechanism, report complete paths only and do not
claim an SDMA-versus-CU hardware-engine main effect.
