"""World-8 validation harness for the MI300X/gfx942 GEMM-RS megakernel.

One process drives all eight devices. Every buffer the protocol touches is
allocated once per rank inside a fine-grained symmetric heap at identical
offsets, so the eight heap bases are a genuine symmetric allocation and the
production host ABI (snapshot_allocation_descriptors / resolve_shape) can be
used unmodified.

Setup contract from MI300X_DESIGN.md section 3: sig and ep_cell are zeroed
exactly once, never per call, because epochs are device-derived.
"""

import importlib.util
import os
import statistics
import time

import torch

WORLD = 8
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")


def _load(name):
    path = os.path.join(BUILD_DIR, f"{name}.so")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load("dhk_rt")


def load_kernel(control=False, module_name=None):
    if module_name is not None:
        return _load(module_name)
    if control:
        return _load("gemm_rs_mi300x_control")
    # aug13 exp_03: let the gate ladder (M3/M5/M9) run against a candidate
    # module without touching the production .so. Explicit module_name and the
    # control path are never overridden; default behaviour is unchanged when
    # the variable is unset.
    return _load(os.environ.get("HK_KERNEL_MODULE", "gemm_rs_mi300x"))


# ---------------------------------------------------------------------------
# pyutils duck-types its gl<> arguments on __class__.__name__ == "Tensor" and
# reads only shape / data_ptr / is_contiguous / device.type. The symmetric heap
# is raw fine-grained memory with no torch storage, so it is presented through
# this shim. Shapes are always spelled 4-D: pyutils left-pads shorter shapes,
# which would silently move the batch extent into the depth slot.
# ---------------------------------------------------------------------------
class _Device:
    def __init__(self, kind):
        self.type = kind


class Tensor:
    def __init__(self, pointer, shape):
        self._pointer = int(pointer)
        self.shape = tuple(int(x) for x in shape)
        self.device = _Device("cuda")

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self._pointer


def _align(value, alignment=4096):
    return ((value + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Input generation, copied verbatim from the official evaluator's reference.py
# so the harness cannot drift from the graded distribution or seeding rule.
# ---------------------------------------------------------------------------
def generate_input(rank, world_size, m, n, k, has_bias, seed):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert k % world_size == 0, "k must be divisible by world_size"
    local_k = k // world_size

    x = (torch.rand((m, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01
    w = (torch.rand((n, local_k), dtype=torch.bfloat16, device=device,
                    generator=gen) * 2 - 1) * 0.01

    bias = None
    if has_bias:
        gen.manual_seed(seed)
        bias = (torch.rand((n,), dtype=torch.bfloat16, device=device,
                           generator=gen) * 2 - 1) * 0.01
    return (x, w, bias)


def reference_reduce_scatter(inputs, m, n):
    """The evaluator's oracle with reduce_scatter_tensor replaced by a local sum.

    ref_kernel computes bf16(x @ w.T) + bias per rank and then reduce-scatters.
    The per-rank partial is materialized in bf16 exactly as the reference does;
    only the cross-rank reduction is performed in fp32 here, which is strictly
    more accurate than RCCL's bf16 tree and far inside the 1e-2 tolerance.
    """
    accumulator = None
    for rank in range(WORLD):
        x, w, bias = inputs[rank]
        partial = torch.matmul(x, w.T)
        if bias is not None:
            partial = partial + bias
        moved = partial.to("cuda:0", non_blocking=False).float()
        accumulator = moved if accumulator is None else accumulator + moved
    full = accumulator.to(torch.bfloat16)
    rows = m // WORLD
    return [full[rank * rows:(rank + 1) * rows, :] for rank in range(WORLD)]


class GemmRS:
    """Allocated-once, launched-many harness instance for one shape."""

    def __init__(self, m, n, k, has_bias, num_reducer_ctas=None,
                 spin_limit=2_000_000, control=False,
                 payload_fine=False, signal_fine=True, module_name=None):
        """payload_fine / signal_fine select allocation granularity.

        Signals must be fine-grained: they are relaxed system-scope atomics
        with no surrounding fence. The payload defaults to coarse-grained
        (cached) because the protocol publishes it with a directional release
        (buffer_wbl2 sc0 sc1, i.e. an L2 writeback) and consumes it after a
        pure acquire (buffer_inv) -- that handshake exists precisely so the
        payload heap can be cached. Fine-grained payload memory is uncached,
        which also defeats L2 on the reducer's *local* reads of all 8 slots.
        """
        self.m, self.n, self.k, self.has_bias = m, n, k, has_bias
        self.spin_limit = int(spin_limit)
        self.control = control
        self.payload_fine = payload_fine
        self.signal_fine = signal_fine
        self.module = load_kernel(control=control, module_name=module_name)
        # bind_function's exported name is fixed by the source; only the module
        # name varies for ablation builds.
        self.entry = (self.module.gemm_rs_mi300x_control if control
                      else self.module.gemm_rs_mi300x)

        if num_reducer_ctas is None:
            self.plan = rt.resolve_shape(m, n, k, has_bias)
        else:
            self.plan = rt.resolve_shape_with_split(m, n, k, has_bias,
                                                    num_reducer_ctas)
        plan = self.plan
        self.k_local = plan["k_local"]
        self.slice_rows = plan["slice_rows"]

        c_bytes = _align(int(plan["c_heap_bytes"]))
        sig_bytes = _align(int(plan["signal_words_total"]) * 4)

        # Two independent symmetric allocations, identical on every rank.
        payload_alloc = rt.fine_alloc if payload_fine else rt.plain_alloc
        signal_alloc = rt.fine_alloc if signal_fine else rt.plain_alloc
        self.c_heap = [payload_alloc(r, c_bytes) for r in range(WORLD)]
        self.sig = [signal_alloc(r, sig_bytes) for r in range(WORLD)]

        # Local, non-symmetric state.
        self.ep_cell = [rt.plain_alloc(r, rt.EP_U32 * 4) for r in range(WORLD)]
        self.err = [rt.plain_alloc(r, 4) for r in range(WORLD)]

        self.desc_c, self.desc_sig = [], []
        for r in range(WORLD):
            c_ptr, s_ptr = rt.make_descriptors_split(
                r, r, self.c_heap, self.c_heap[r], self.sig, self.sig[r])
            self.desc_c.append(c_ptr)
            self.desc_sig.append(s_ptr)

        self.c_heap_view = [
            Tensor(self.c_heap[r], (WORLD, 1, self.slice_rows, self.n))
            for r in range(WORLD)
        ]
        self.out = [torch.zeros((self.slice_rows, self.n),
                                dtype=torch.bfloat16, device=f"cuda:{r}")
                    for r in range(WORLD)]
        self.n_calls = 0
        self.inputs = None

    # -- lifetime ---------------------------------------------------------
    def close(self):
        for r in range(WORLD):
            rt.free_device(r, self.c_heap[r])
            rt.free_device(r, self.sig[r])
            rt.free_device(r, self.ep_cell[r])
            rt.free_device(r, self.err[r])
        self.c_heap = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- inputs -----------------------------------------------------------
    def set_inputs(self, seed):
        self.inputs = [generate_input(r, WORLD, self.m, self.n, self.k,
                                      self.has_bias, seed)
                       for r in range(WORLD)]
        self._rebuild_args()
        return self.inputs

    def _rebuild_args(self, ctrl_flags=0, ctrl_rank=0, ctrl_arg0=0,
                      streams=None):
        """Precompute the full 27-argument tuple once per input change.

        pyutils builds each gl<> argument by introspecting the Python object
        (__class__.__name__, is_contiguous(), device.type, shape, data_ptr()).
        Doing that for four torch tensors on every launch cost ~11 us of host
        time per launch, i.e. ~90 us per world-8 operation -- more than the
        device work for the three smallest graded shapes, which made those
        measurements report the host rather than the kernel. Passing cheap
        duck-typed views and a prebuilt tuple moves that work off the hot path.
        """
        plan = self.plan
        self._args = []
        for rank in range(WORLD):
            x, w, bias = self.inputs[rank]
            torch.cuda.set_device(rank)
            # During graph capture the launch must go to the capturing stream,
            # not to the stream that happened to be current when the argument
            # tuple was first built.
            stream_handle = (streams[rank].cuda_stream if streams is not None
                             else torch.cuda.current_stream(rank).cuda_stream)
            self._args.append((
                Tensor(x.data_ptr(), (1, 1, self.m, self.k_local)),
                Tensor(w.data_ptr(), (1, 1, self.n, self.k_local)),
                self.c_heap_view[rank],
                Tensor(self.out[rank].data_ptr(),
                       (1, 1, self.slice_rows, self.n)),
                self.desc_c[rank], self.sig[rank], self.desc_sig[rank],
                self.ep_cell[rank],
                0 if bias is None else bias.data_ptr(),
                self.err[rank],
                stream_handle, self.spin_limit,
                rank, self.m, self.n, self.k_local,
                int(plan["lrow_count"]), int(plan["col_count"]),
                int(plan["eb"]), int(plan["num_gemm_ctas"]),
                int(plan["ready_words"]),
                1 if plan["packet_fast_path"] else 0,
                int(ctrl_flags), int(ctrl_rank), int(ctrl_arg0),
                int(plan["config_row"]), 1 if plan["even_k"] else 0,
            ))

    # -- launching --------------------------------------------------------
    def _launch_one(self, rank, ctrl_flags=0, ctrl_rank=0, ctrl_arg0=0):
        if ctrl_flags or ctrl_rank or ctrl_arg0:
            if getattr(self, "_ctrl_key", None) != (ctrl_flags, ctrl_rank,
                                                    ctrl_arg0):
                self._rebuild_args(ctrl_flags, ctrl_rank, ctrl_arg0)
                self._ctrl_key = (ctrl_flags, ctrl_rank, ctrl_arg0)
        elif getattr(self, "_ctrl_key", None) not in (None, (0, 0, 0)):
            self._rebuild_args()
            self._ctrl_key = (0, 0, 0)
        torch.cuda.set_device(rank)
        self.entry(*self._args[rank])

    def launch(self, ctrl_flags=0, ctrl_rank=0, ctrl_arg0=0, skew=None,
               sync=True):
        """Launch one epoch on all eight ranks.

        `skew` is an optional rank order, used by the soak to start ranks in a
        deliberately adverse sequence. Reducers spin on peer producers, so a
        skewed launch order exercises the bounded-wait path rather than a
        lock-step best case.
        """
        order = list(range(WORLD)) if skew is None else list(skew)
        for rank in order:
            self._launch_one(rank, ctrl_flags, ctrl_rank, ctrl_arg0)
        self.n_calls += 1
        if sync:
            self.sync()

    def sync(self):
        for rank in range(WORLD):
            rt.device_synchronize(rank)

    # -- inspection -------------------------------------------------------
    def error_bits(self):
        out = []
        for rank in range(WORLD):
            raw = rt.device_to_host(rank, self.err[rank], 4)
            out.append(int.from_bytes(raw, "little", signed=True))
        return out

    def error_report(self):
        names = []
        for rank, bits in enumerate(self.error_bits()):
            flags = []
            if bits & rt.ERR_PRODUCER_CREDIT:
                flags.append("PRODUCER_CREDIT_TIMEOUT(bit25)")
            if bits & rt.ERR_REDUCER_READY:
                flags.append("REDUCER_READY_TIMEOUT(bit26)")
            if bits and not flags:
                flags.append(f"other=0x{bits:x}")
            if flags:
                names.append(f"rank{rank}: " + ",".join(flags))
        return names

    def epoch_cells(self, rank):
        raw = rt.device_to_host(rank, self.ep_cell[rank], rt.EP_U32 * 4)
        return [int.from_bytes(raw[i * 4:(i + 1) * 4], "little")
                for i in range(rt.EP_U32)]

    def check_epochs(self, expected=None):
        """Every scheduled CTA's device cell must equal the launch count.

        Producer window is [0, num_gemm_ctas); reducer window is
        [EP_RED_U32, EP_RED_U32 + num_reducer_ctas). Unused cells must stay 0.
        """
        expected = self.n_calls if expected is None else expected
        gemm = int(self.plan["num_gemm_ctas"])
        red = int(self.plan["num_reducer_ctas"])
        problems = []
        for rank in range(WORLD):
            cells = self.epoch_cells(rank)
            prod = cells[0:gemm]
            unused_prod = cells[gemm:rt.EP_RED_U32]
            reducers = cells[rt.EP_RED_U32:rt.EP_RED_U32 + red]
            unused_red = cells[rt.EP_RED_U32 + red:rt.EP_U32]
            if any(v != expected for v in prod):
                bad = [(i, v) for i, v in enumerate(prod) if v != expected]
                problems.append(f"rank{rank} producer cells != {expected}: "
                                f"{bad[:6]} ({len(bad)} total)")
            if any(v != expected for v in reducers):
                bad = [(i, v) for i, v in enumerate(reducers) if v != expected]
                problems.append(f"rank{rank} reducer cells != {expected}: "
                                f"{bad[:6]} ({len(bad)} total)")
            if any(v != 0 for v in unused_prod) or any(v != 0 for v in unused_red):
                problems.append(f"rank{rank} unscheduled cells nonzero")
        return problems

    def signal_words(self, rank):
        total = int(self.plan["signal_words_total"])
        raw = rt.device_to_host(rank, self.sig[rank], total * 4)
        return [int.from_bytes(raw[i * 4:(i + 1) * 4], "little")
                for i in range(total)]

    def check_signals(self, expected=None):
        """Every touched ready/credit cell must read exactly the final epoch."""
        expected = self.n_calls if expected is None else expected
        guard = rt.SIGNAL_GUARD_U32
        ready_words = int(self.plan["ready_words"])
        problems = []
        for rank in range(WORLD):
            words = self.signal_words(rank)
            if any(v != 0 for v in words[:guard]):
                problems.append(f"rank{rank} guard region written")
            ready = words[guard:guard + ready_words]
            credit = words[guard + ready_words:guard + 2 * ready_words]
            for label, region in (("ready", ready), ("credit", credit)):
                bad = [(i, v) for i, v in enumerate(region) if v != expected]
                if bad:
                    problems.append(
                        f"rank{rank} {label} != {expected}: {bad[:6]} "
                        f"({len(bad)}/{len(region)})")
        return problems

    # -- verification -----------------------------------------------------
    def verify(self, rtol=1e-2, atol=1e-2):
        expected = reference_reduce_scatter(self.inputs, self.m, self.n)
        results = []
        for rank in range(WORLD):
            want = expected[rank].to(f"cuda:{rank}")
            got = self.out[rank]
            ok = torch.allclose(got, want, rtol=rtol, atol=atol)
            diff = (got.float() - want.float()).abs()
            results.append({
                "rank": rank,
                "allclose": bool(ok),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "ref_absmax": float(want.float().abs().max()),
                "got_absmax": float(got.float().abs().max()),
                "mismatches": int((~torch.isclose(got, want, rtol=rtol,
                                                  atol=atol)).sum()),
                "numel": int(got.numel()),
            })
        return results

    # -- timing -----------------------------------------------------------
    #
    # Two measurements, because they answer different questions and a single
    # number here would be misleading.
    #
    # time_pipelined: `iters` operations are queued back to back on each rank's
    #   stream with no host synchronization in between, then divided out. This
    #   is the throughput-relevant figure and the one comparable to a benchmark
    #   loop. Queuing across ranks is safe by construction rather than by luck:
    #   per-rank stream order preserves the cross-launch invariant, and the
    #   directed retirement credits are exactly what let rank A start epoch e
    #   while rank B is still finishing e-1.
    #
    # time_single_shot: one operation at a time with a full host sync around
    #   it. This is latency-relevant but, in a single process, it also charges
    #   the operation for eight sequential kernel launches from one CPU thread.
    #   The launch skew is reported alongside it so a reader can see when that
    #   overhead dominates (it does for the smallest shapes).
    def _events(self):
        starts, ends = [], []
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            starts.append(torch.cuda.Event(enable_timing=True))
            ends.append(torch.cuda.Event(enable_timing=True))
        return starts, ends

    def _timed_block(self, iters):
        starts, ends = self._events()
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            starts[rank].record(torch.cuda.current_stream(rank))
        wall_start = time.perf_counter()
        for _ in range(iters):
            self.launch(sync=False)
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            ends[rank].record(torch.cuda.current_stream(rank))
        self.sync()
        wall = (time.perf_counter() - wall_start) * 1e6 / iters
        per_rank = []
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            per_rank.append(starts[rank].elapsed_time(ends[rank]) * 1e3 / iters)
        return wall, per_rank

    def time_pipelined(self, iters=50, warmup_ms=400, blocks=3):
        """Warmup is duration-based, not iteration-based, and that matters.

        Idle sclk on this node is ~132 MHz against ~1900 MHz under load, so a
        fixed iteration count warms a 100 us kernel for only a few milliseconds
        and measures a GPU that is still ramping. An early version of this
        harness reported 95 us and 153 us for the same shape for exactly that
        reason. Clocks should additionally be pinned with
        `rocm-smi --setperfdeterminism`; see set_clocks.sh.
        """
        warmup_start = time.perf_counter()
        while (time.perf_counter() - warmup_start) * 1e3 < warmup_ms:
            for _ in range(8):
                self.launch(sync=False)
            self.sync()

        walls, rank_sets = [], []
        for _ in range(blocks):
            wall, per_rank = self._timed_block(iters)
            walls.append(wall)
            rank_sets.append(per_rank)
        best = walls.index(min(walls))
        return {
            "mode": "pipelined",
            "iters": iters,
            "blocks": blocks,
            "wall_us": statistics.median(walls),
            "wall_min_us": min(walls),
            "wall_spread_pct": (max(walls) - min(walls)) / min(walls) * 100.0,
            "block_walls_us": walls,
            "device_us_max": max(rank_sets[best]),
            "device_us_mean": statistics.mean(rank_sets[best]),
            "device_us_min": min(rank_sets[best]),
            "per_rank_us": rank_sets[best],
            "errors": self.error_report(),
        }

    def build_graphs(self, reps=20):
        """Capture `reps` operations per rank into one graph per rank.

        Graph capture is the only way to measure the smaller shapes here: the
        host needs ~11 us to issue a launch, so eight launches per operation put
        a ~90 us floor under any eagerly-issued measurement, well above the
        device work for shapes 1-3. Capturing `reps` operations amortizes the
        host cost by that factor.

        This is also Gate M8's subject. The epochs are device-derived cells, so
        a replay is not a stale re-run of a captured host epoch value: every
        replay advances every touched cell. That is asserted, not assumed.
        """
        capture_streams = []
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            capture_streams.append(torch.cuda.Stream(device=rank))
        self._rebuild_args(streams=capture_streams)
        self._ctrl_key = ("graph",)

        graphs = []
        try:
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_streams[rank]):
                    for _ in range(reps):
                        self.entry(*self._args[rank])
                graphs.append(graph)
        except Exception:
            # Leaving a stream mid-capture poisons every later HIP call,
            # including the hipFree in close(), which then masks the real error.
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                try:
                    torch.cuda.current_stream(rank).synchronize()
                except Exception:
                    pass
            self._rebuild_args()
            self._ctrl_key = (0, 0, 0)
            raise
        self.graphs = graphs
        self.graph_reps = reps
        self.graph_streams = capture_streams
        return graphs

    def replay_graphs(self, times=1, sync=True):
        for _ in range(times):
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                self.graphs[rank].replay()
            self.n_calls += self.graph_reps
        if sync:
            self.sync()

    def time_graph(self, reps=20, replays=20, warmup_ms=400, blocks=3):
        self.build_graphs(reps=reps)
        warmup_start = time.perf_counter()
        while (time.perf_counter() - warmup_start) * 1e3 < warmup_ms:
            self.replay_graphs(times=2)

        walls, per_op_device = [], []
        for _ in range(blocks):
            starts, ends = self._events()
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                starts[rank].record(torch.cuda.current_stream(rank))
            wall_start = time.perf_counter()
            self.replay_graphs(times=replays, sync=False)
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                ends[rank].record(torch.cuda.current_stream(rank))
            self.sync()
            total_ops = replays * reps
            walls.append((time.perf_counter() - wall_start) * 1e6 / total_ops)
            per_op_device.append(max(
                starts[r].elapsed_time(ends[r]) * 1e3 / total_ops
                for r in range(WORLD)))
        return {
            "mode": "graph",
            "reps_per_graph": reps,
            "replays": replays,
            "wall_us": statistics.median(walls),
            "wall_min_us": min(walls),
            "device_us_max": statistics.median(per_op_device),
            "block_walls_us": walls,
            "errors": self.error_report(),
        }

    def time_host_issue(self, iters=50):
        """Host-side cost of issuing one operation, with the device excluded.

        This is the floor of any measurement that issues eight launches per
        operation from one CPU thread: if it exceeds the device time, the
        reported number is a property of the host, not of the kernel. It is
        measured by enqueuing without ever synchronizing, so the device runs
        behind and the loop times pure submission.
        """
        for _ in range(8):
            self.launch(sync=False)
        self.sync()
        start = time.perf_counter()
        for _ in range(iters):
            self.launch(sync=False)
        issue = (time.perf_counter() - start) * 1e6 / iters
        self.sync()
        return {"host_issue_us_per_op": issue,
                "host_issue_us_per_launch": issue / WORLD}

    def time_single_shot(self, iters=20, warmup=5):
        for _ in range(warmup):
            self.launch()
        wall_samples, crit_samples, skew_samples = [], [], []
        for _ in range(iters):
            self.sync()
            starts, ends = self._events()
            launch_times = []
            wall_start = time.perf_counter()
            for rank in range(WORLD):
                torch.cuda.set_device(rank)
                starts[rank].record(torch.cuda.current_stream(rank))
                launch_times.append(time.perf_counter())
                self._launch_one(rank)
                ends[rank].record(torch.cuda.current_stream(rank))
            self.n_calls += 1
            self.sync()
            wall_samples.append((time.perf_counter() - wall_start) * 1e6)
            crit_samples.append(max(
                starts[r].elapsed_time(ends[r]) * 1e3 for r in range(WORLD)))
            skew_samples.append((launch_times[-1] - launch_times[0]) * 1e6)
        return {
            "mode": "single_shot",
            "iters": iters,
            "wall_mean_us": statistics.mean(wall_samples),
            "wall_median_us": statistics.median(wall_samples),
            "device_max_mean_us": statistics.mean(crit_samples),
            "launch_skew_us": statistics.mean(skew_samples),
            "errors": self.error_report(),
        }
