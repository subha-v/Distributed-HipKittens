"""Our MI300X GEMM-RS megakernel, packaged for the OFFICIAL evaluator.

The evaluator runs one process per rank under torch.distributed, so the
symmetric heap that the single-process harness got for free has to be built
with HIP IPC here: each rank allocates its own payload and signal regions,
publishes an IPC handle, and opens the other seven. The resulting eight base
addresses are handed to the unmodified production host ABI
(snapshot_allocation_descriptors) exactly as before.

Why the epoch protocol is safe under this harness: eval.py's distributed
benchmark loop barriers the ranks and has rank 0 broadcast the stop decision,
so every rank calls custom_kernel the same number of times. The device-derived
per-CTA epochs therefore stay in lockstep across ranks, which is the one thing
this protocol requires of its caller.

State is cached per (rank, shape) and survives destroy_process_group, so the
allocation, the IPC exchange and the descriptor construction happen once per
rank per shape for the life of the process, never inside the timed region.
"""

import faulthandler
import importlib.util
import os
import sys
import time
import weakref

import torch
import torch.distributed as dist

BUILD_DIR = os.environ.get(
    "HK_BUILD_DIR", "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/build")


def _load(name):
    path = os.path.join(BUILD_DIR, f"{name}.so")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load("dhk_rt")
_kernel = _load(os.environ.get("HK_KERNEL_MODULE", "gemm_rs_mi300x"))
_entry = _kernel.gemm_rs_mi300x
# The prebound pair (exp_12). An ablation build selected through
# HK_KERNEL_MODULE does not carry it, so the duck-typed path stays available.
_configure = getattr(_kernel, "gemm_rs_mi300x_configure", None)
_run = getattr(_kernel, "gemm_rs_mi300x_run", None)

# torch.cuda.current_stream(rank).cuda_stream builds a python Stream object and
# costs 2.3 us of the graded critical path, measured in an 8-rank pool. The
# private accessor returns the same handle for ~0.2 us. Resolved once, at
# import, so the call site has no getattr and no fallback branch.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if _raw_stream is None:
    def _raw_stream(index):
        return torch.cuda.current_stream(index).cuda_stream

WORLD = 8
SPIN_LIMIT = int(os.environ.get("HK_SPIN_LIMIT", 20_000_000))
DEBUG = bool(int(os.environ.get("HK_DEBUG", "0")))
DUMP_AFTER = float(os.environ.get("HK_DUMP_AFTER", 120))

# Set on entry to custom_kernel so every line carries the rank even outside a
# process group. pid is printed with it because the pool's worker-to-rank map
# is what has to be read out of these logs.
_RANK = -1


def _say(message):
    """time.monotonic() is CLOCK_MONOTONIC, so the eight logs are comparable."""
    print(f"[hk {time.monotonic():.6f} pid {os.getpid()} rank {_RANK}] "
          f"{message}", file=sys.stderr, flush=True)


def _log(message):
    if DEBUG:
        _say(message)


_ARMED_PID = None


def _arm_watchdog():
    """Turn a wedged rank into eight Python stacks instead of a pool timeout.

    Armed from custom_kernel rather than at import: dump_traceback_later's
    timer thread does not survive a fork, so a worker that inherited this
    module from the parent would carry no watchdog at all.
    """
    global _ARMED_PID
    if not DEBUG or _ARMED_PID == os.getpid():
        return
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(DUMP_AFTER, exit=False, repeat=True)
    _ARMED_PID = os.getpid()
    _say(f"watchdog armed, dumping every {DUMP_AFTER:g}s")


class _Device:
    def __init__(self, kind):
        self.type = kind


class Tensor:
    """Duck-typed view for pyutils' gl<> arguments.

    pyutils checks __class__.__name__ == "Tensor" and reads only shape,
    data_ptr, is_contiguous and device.type. Shapes are always spelled 4-D
    because pyutils left-pads shorter ones, which would silently move the batch
    extent into the depth slot.
    """

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


# State is keyed by (rank, shape) and lives as long as the process does.
#
# eval.py drives the ranks with multiprocessing.Pool(8) and calls
# init_process_group / destroy_process_group once per test case. The pool
# reuses worker processes, and the worker-to-rank assignment is NOT stable
# across test cases: a worker that served rank 6 can serve rank 1 next. A cache
# keyed only by shape therefore hands a later rank an allocation belonging to
# another device, which the evaluator catches as
# "Output device mismatch: cuda:6 != cuda:1". Hence the rank in the key, and
# hence torch.cuda.set_device(rank) before anything is allocated or launched.
#
# Nothing is bound to the process group any more. Discarding state per group
# meant a fresh heap and a fresh IPC exchange for a shape this process had
# already mapped, and hipIpcOpenMemHandle answers hipErrorAlreadyMapped when a
# handle resolves to an allocation this process already imported -- which kills
# the ranks it hits and leaves the rest wedged in the next collective.
#
# The group is still tracked, but only to decide when the collective setup path
# has to be re-entered: the first call of a new group, and any call whose shape
# differs from the last one. _LAST_PG is a WEAKREF, not a strong reference:
# a strong reference here outlives destroy_process_group and pins the old
# NCCL communicator and its TCPStore, and the next init_process_group on the
# same MASTER_PORT then never completes -- measured under eval.py, where all
# eight ranks finished 101 calls and then blocked in
# _new_process_group_helper. A weakref cannot make a recycled address look
# like the old group either, because the test below compares the referent.
# HK_DEBUG takes the full path unconditionally: its whole purpose is the
# per-call synchronize and error-bit read, and a fast path that skipped them
# would turn a debugging session into a false clean bill of health.
#
# Both halves of the prebound pair are required: the handle only exists if
# `configure` ran. They are exported by the same binding block so a module
# carrying one but not the other cannot be built, but the predicate says so
# rather than relying on it.
_FAST = _run is not None and _configure is not None and not DEBUG

_STATES = {}      # (rank, m, n, k, has_bias) -> _ShapeState
_LAST_PG = None
_LAST_KEY = None
_RETAINED = []   # keeps every allocation alive; see custom_kernel's docstring


def _current_group():
    try:
        return dist.distributed_c10d._get_default_group()
    except Exception:
        return None


class _ShapeState:
    def __init__(self, m, n, k, has_bias, rank):
        self.rank = rank
        torch.cuda.set_device(rank)
        self.plan = rt.resolve_shape(m, n, k, has_bias)
        plan = self.plan
        self.m, self.n = m, n
        self.k_local = plan["k_local"]
        self.slice_rows = plan["slice_rows"]

        c_bytes = _align(int(plan["c_heap_bytes"]))
        sig_bytes = _align(int(plan["signal_words_total"]) * 4)

        # Payload coarse-grained, signals fine-grained. Measured on this node:
        # granularity is worth <1% either way, but the signals are relaxed
        # system-scope atomics with no surrounding fence, so they get the
        # fine-grained allocation.
        self.c_local = rt.plain_alloc(rank, c_bytes)
        try:
            self.sig_local = rt.fine_alloc(rank, sig_bytes)
            self.sig_fine = True
        except Exception:
            self.sig_local = rt.plain_alloc(rank, sig_bytes)
            self.sig_fine = False

        _log(f"allocated c={hex(self.c_local)} sig={hex(self.sig_local)} "
             f"fine_sig={self.sig_fine}")

        self.c_bases = self._exchange(self.c_local, rank, "c_heap")
        self.sig_bases = self._exchange(self.sig_local, rank, "signals")

        self.ep_cell = rt.plain_alloc(rank, rt.EP_U32 * 4)
        self.err = rt.plain_alloc(rank, 4)

        self.desc_c, self.desc_sig = rt.make_descriptors_split(
            rank, rank, self.c_bases, self.c_local,
            self.sig_bases, self.sig_local)
        _log("descriptors uploaded")

        self.c_view = Tensor(self.c_local, (WORLD, 1, self.slice_rows, n))
        self.out = torch.zeros((self.slice_rows, n), dtype=torch.bfloat16,
                               device=f"cuda:{rank}")
        self.out_view = Tensor(self.out.data_ptr(),
                               (1, 1, self.slice_rows, n))

        # Everything invariant for this (rank, shape) is bound into the kernel
        # module once, here, so the per-call path carries only the three input
        # pointers and the stream. Argument order must match
        # prebound::configure in gemm_rs_mi300x.cpp.
        self.handle = -1
        if _configure is not None:
            self.handle = _configure(
                self.m, self.k_local, n, self.k_local,
                self.c_local, WORLD, self.slice_rows, n,
                self.out.data_ptr(), self.slice_rows, n,
                self.desc_c, self.sig_local, self.desc_sig, self.ep_cell,
                self.err, _raw_stream(rank), SPIN_LIMIT,
                rank, self.m, n, self.k_local,
                int(plan["lrow_count"]), int(plan["col_count"]),
                int(plan["eb"]), int(plan["num_gemm_ctas"]),
                int(plan["ready_words"]),
                1 if plan["packet_fast_path"] else 0,
                int(plan["config_row"]), 1 if plan["even_k"] else 0)

    def _exchange(self, local_pointer, rank, label):
        """Publish this rank's allocation and open every peer's."""
        handle = rt.ipc_get_handle(rank, local_pointer)
        _log(f"{label}: get_ipc_handle done {handle[:6].hex()}")
        handles = [None] * WORLD
        dist.all_gather_object(handles, handle)
        # The handle prefixes identify the exported allocations: two exchanges
        # that gather the same prefix for the same peer are re-importing memory
        # this process already holds, which is what hipErrorAlreadyMapped is
        # complaining about.
        digests = [h[:6].hex() if h else None for h in handles]
        _log(f"{label}: all_gather_object done "
             f"{sum(h is not None for h in handles)}/{WORLD} {digests}")
        bases = []
        for peer in range(WORLD):
            if peer == rank:
                bases.append(local_pointer)
                continue
            try:
                bases.append(rt.ipc_open_handle(rank, handles[peer]))
            except Exception as error:
                # dhk_rt raises "hipIpcOpenMemHandle: <hipGetErrorString>";
                # printed here because the pool holds the traceback until every
                # other rank has already timed out.
                _say(f"{label}: hipIpcOpenMemHandle FAILED peer {peer} "
                     f"handle {digests[peer]}: "
                     f"{type(error).__name__}: {error}")
                raise
            _log(f"{label}: opened peer {peer} at {hex(bases[peer])}")
        _log(f"{label}: opened all peers")
        return bases

    def launch_fast(self, x, w, bias):
        """The prebound launch: three pointers, the stream, and a handle.

        The stream is read per call rather than captured at configure time so a
        caller that enters a stream context between calls still gets its work
        on the right stream; it is the one thing here that can legitimately
        change without the state changing.
        """
        _run(self.handle, x.data_ptr(), w.data_ptr(),
             0 if bias is None else bias.data_ptr(), _raw_stream(self.rank))
        return self.out

    def launch(self, x, w, bias):
        plan = self.plan
        rank = self.rank
        _entry(
            Tensor(x.data_ptr(), (1, 1, self.m, self.k_local)),
            Tensor(w.data_ptr(), (1, 1, self.n, self.k_local)),
            self.c_view, self.out_view,
            self.desc_c, self.sig_local, self.desc_sig, self.ep_cell,
            0 if bias is None else bias.data_ptr(),
            self.err,
            torch.cuda.current_stream(rank).cuda_stream, SPIN_LIMIT,
            rank, self.m, self.n, self.k_local,
            int(plan["lrow_count"]), int(plan["col_count"]), int(plan["eb"]),
            int(plan["num_gemm_ctas"]), int(plan["ready_words"]),
            1 if plan["packet_fast_path"] else 0,
            0, 0, 0,
            int(plan["config_row"]), 1 if plan["even_k"] else 0,
        )
        return self.out

    def error_bits(self):
        raw = rt.device_to_host(self.rank, self.err, 4)
        return int.from_bytes(raw, "little", signed=True)


def _setup(key, m, n, k, has_bias, rank, state):
    """Bring this rank to a usable state for `key`, in step with the other seven.

    The vote is the point of this function. Whether a worker already holds
    state for a key depends on which rank the pool handed it last time, so the
    local answer is not the group's answer: after a permutation some workers
    hold the key and others do not, and a rank that skipped the exchange while
    its peers entered it would hang all seven of them. Every rank that is not
    already settled on (this group, this key) reaches this vote, the vote is
    unanimous-or-rebuild, and so all eight then run the same collectives.

    Rebuilding on ranks that already had state is deliberate. A rebuild zeroes
    that rank's signal region and epoch cells; a partial rebuild would leave
    rank A publishing epoch e+1 into a peer whose reducers are waiting for 1.
    """
    votes = [None] * WORLD
    dist.all_gather_object(votes, state is not None)
    _log(f"cache vote {votes}")

    if not all(votes):
        state = _ShapeState(m, n, k, has_bias, rank)
        state.key = key
        _STATES[key] = state
        _RETAINED.append(state)

    # Every rank must finish setup before any rank launches: a producer writes
    # into a peer's heap on its very first tile. A NCCL barrier is only
    # enqueued, so it is followed by a device sync -- on the reuse path too,
    # where the heaps are old but the group and its streams are new.
    _log("setup barrier")
    dist.barrier()
    _log("setup barrier returned")
    torch.cuda.synchronize(rank)
    _log("device synchronize returned")
    return state


def custom_kernel(data):
    """Nothing is ever freed, deliberately.

    Releasing the previous shape's heap between test cases means closing peer
    IPC mappings while other ranks may still hold them, and it makes the setup
    path asymmetric if the evaluator's process pool respawns a worker (a fresh
    worker has no state to release and so would skip a collective the others
    take, deadlocking the group). Retaining every allocation costs at most a
    few hundred MB per device across the whole case list and removes both
    hazards.
    """
    global _LAST_PG, _LAST_KEY, _RANK

    x, w, bias = data

    # ---- fast path -------------------------------------------------------
    # The graded protocol is barrier -> call -> synchronize -> barrier, so
    # nothing overlaps this function and every microsecond in it is charged to
    # the score. Measured in an 8-rank pool this path was 20-36 us per call,
    # against ~4 us for the launch itself.
    #
    # It is the SAME decision the slow path below makes -- same group identity
    # test, same key, same state -- with only the redundant work removed: the
    # rank comes out of the cached key instead of a c10d round trip (within a
    # group the rank is fixed, and any new group fails the identity test), the
    # log f-strings are not built when nothing will read them, and the launch
    # goes through the prebound handle. Anything unexpected falls through to
    # the full path, which is unchanged.
    if _FAST and _LAST_KEY is not None and _LAST_PG is not None:
        group = _current_group()
        if group is not None and _LAST_PG() is group:
            rank = _LAST_KEY[0]
            state = _STATES.get(_LAST_KEY)
            if (state is not None and state.handle >= 0
                    and x.shape[0] == state.m and w.shape[0] == state.n
                    and x.shape[1] == state.k_local
                    and (bias is not None) == _LAST_KEY[4]):
                torch.cuda.set_device(rank)
                return state.launch_fast(x, w, bias)
    # ---- end fast path ---------------------------------------------------

    rank = dist.get_rank()
    _RANK = rank
    _arm_watchdog()
    torch.cuda.set_device(rank)

    m, k_local = x.shape
    n = w.shape[0]
    k_global = k_local * WORLD
    key = (rank, m, n, k_global, bias is not None)
    _log(f"enter custom_kernel key={key}")

    # Settled means this rank used this very state on its previous call in this
    # very group, which every other rank did too: within a group the rank is
    # fixed and the shape is the group's, so the three tests below give the same
    # answer on all eight. An unavailable group identity settles nothing and
    # falls back to voting on every call -- slow, but still symmetric.
    group = _current_group()
    state = _STATES.get(key)
    settled = (state is not None and group is not None
               and _LAST_PG is not None and _LAST_PG() is group
               and key == _LAST_KEY)
    _log(f"cache {'hit' if state is not None else 'miss'} "
         f"settled={settled} keys={len(_STATES)}")
    if not settled:
        state = _setup(key, m, n, k_global, bias is not None, rank, state)
        _LAST_PG = weakref.ref(group) if group is not None else None
        _LAST_KEY = key
    _log("state ready")

    out = state.launch(x, w, bias)
    _log("launch issued")
    if DEBUG:
        torch.cuda.synchronize(rank)
        _log("launch synchronized")
        bits = state.error_bits()
        if bits:
            _say(f"ERROR BITS 0x{bits:x}")
    return out
