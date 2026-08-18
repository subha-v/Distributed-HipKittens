#!/usr/bin/env python3
"""M23 L0 -- offline dispatcher replay (no GPU, no node, no network).

Replays a logged per-step, per-rank batch trace through a pure-Python model of
the deployed decision chain

    _pad_for_sequence_parallelism
      -> CudagraphDispatcher.dispatch          (local, pre-DP)
      -> coordinate_batch_across_dp            (min mode, max padding)
      -> CudagraphDispatcher.dispatch          (re-dispatch, post-DP)
      -> the seal predicate                    (OLD exact vs NEW M23 ragged)
      -> the uniform-decode rescue / eager fallback

and reports, for both predicates, how many steps the megakernel would actually
have been sealed for.  This is the ladder rung the design (section 8.3, L0)
puts BEFORE any node time, because it is what proves hazards H1 (the
uniform-decode rank) and H2 (the step where no DP padding happened) are handled
-- a non-unanimous seal is a collective deadlock, not a wrong answer.

GATES (exit non-zero if either fails, per design 8.3 L0 / 8.2):

    sealed == in_bucket                on every rank
    refused_not_unanimous == 0         on every rank

TRACE FORMAT -- one JSON object per line, one line per model step:

    {"step": 17,
     "ranks": [
       {"num_tokens": 4096, "num_reqs": 1, "max_num_scheduled_tokens": 4096},
       {"num_tokens": 1,    "num_reqs": 1, "max_num_scheduled_tokens": 1},
       ... exactly dp_size entries, in rank order ...
     ]}

  Per-rank keys:
    num_tokens                 REQUIRED, the unpadded live token count
                               (GMR's ``num_tokens_unpadded``)
    num_reqs                   optional (default 1)
    max_num_scheduled_tokens   optional (default num_tokens)
    uniform_decode             optional bool; overrides the derivation
    has_lora                   optional bool  (default false)
    num_active_loras           optional int   (default 0)
    ready                      optional bool  (default true) -- whether this
                               rank's PF4H activation file has latched yet

  Convenience: a line may instead be a bare list of ints, taken as the
  per-rank ``num_tokens`` with everything else defaulted:

    [4096, 1, 3300, 4096, 512, 4096, 77, 2048]

USAGE

    # gate a real trace
    python3 offline_dispatcher_replay.py --trace c32p_rank_trace.jsonl

    # write a synthetic c32p-shaped trace, then replay it
    python3 offline_dispatcher_replay.py --emit-synthetic trace.jsonl --steps 600
    python3 offline_dispatcher_replay.py --trace trace.jsonl --per-rank

HOW TO PRODUCE A REAL TRACE ON THE NODE
    Add one debug line per step next to the RAGGED_SEAL_RECEIPT emission:

        logger.info("M23_TRACE rank=%d step=%d n_orig=%s",
                    self.parallel_config.data_parallel_rank,
                    _m23["steps"], list(m23_orig_counts))

    ``m23_orig_counts`` is already the all-reduced 8-vector of *original*
    counts, so ONE rank's log is enough: grep the lines, take the vector as
    the ``ranks`` list.  ``--from-m23-trace`` parses exactly that grep output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------
# model of the deployed configuration (P3 confirms these on the live server)
# --------------------------------------------------------------------------
DEFAULT_MAX_CAPTURE_SIZE = 512          # compilation_config.max_cudagraph_capture_size
DEFAULT_MAX_BATCHED_TOKENS = 4096       # asserted at cudagraph_dispatcher.py:236-241
DEFAULT_CAPTURE_SIZES = (
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
    128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416,
    448, 480, 512,
)

NONE, PIECEWISE, FULL = 0, 1, 2
MODE_NAME = {NONE: "NONE", PIECEWISE: "PIECEWISE", FULL: "FULL"}


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
    pf4h_exact_b4096: bool = False


@dataclass
class RankStep:
    num_tokens: int
    num_reqs: int = 1
    max_num_scheduled_tokens: int | None = None
    uniform_decode: bool | None = None
    has_lora: bool = False
    num_active_loras: int = 0
    ready: bool = True

    def is_uniform_decode(self, uniform_decode_query_len: int = 1) -> bool:
        """GMR._is_uniform_decode (GMR:3820-3839)."""
        if self.uniform_decode is not None:
            return self.uniform_decode
        mx = (
            self.max_num_scheduled_tokens
            if self.max_num_scheduled_tokens is not None
            else self.num_tokens
        )
        return (
            mx == uniform_decode_query_len
            and self.num_tokens == mx * self.num_reqs
        )


@dataclass
class Config:
    dp_size: int = 8
    graph_target: str = "pf4h"              # "pf4h" | "stock"
    integration_mode: str = "m15"           # "m15" | "full" | ""
    max_capture_size: int = DEFAULT_MAX_CAPTURE_SIZE
    capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES
    # PF4H removes the <=256 mixed PIECEWISE keys (DISP:204-212, design 7.3).
    drop_mixed_keys_le_256: bool = True
    full_decode_graphs: bool = True
    m23: bool = True
    uniform_rescue: bool = True


# --------------------------------------------------------------------------
# CudagraphDispatcher.dispatch  (vllm/v1/cudagraph_dispatcher.py)
# --------------------------------------------------------------------------
class Dispatcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        pf4h = cfg.integration_mode in ("full", "m15")
        self.piecewise_keys: set[BatchDescriptor] = set()
        for bs in cfg.capture_sizes:
            if pf4h and cfg.drop_mixed_keys_le_256 and bs <= 256:
                continue                                    # DISP:204-212
            self.piecewise_keys.add(BatchDescriptor(num_tokens=bs))
        self.full_keys: set[BatchDescriptor] = set()
        if cfg.full_decode_graphs:
            for bs in cfg.capture_sizes:
                self.full_keys.add(
                    BatchDescriptor(num_tokens=bs, num_reqs=bs, uniform=True)
                )
        self.regular_b4096 = BatchDescriptor(num_tokens=4096)
        self.pf4h_b4096 = replace(self.regular_b4096, pf4h_exact_b4096=True)
        if pf4h:                                             # DISP:243-255
            self.piecewise_keys.add(
                self.regular_b4096
                if cfg.graph_target == "stock"
                else self.pf4h_b4096
            )

    def _round_up(self, n: int) -> int | None:
        for bs in self.cfg.capture_sizes:
            if bs >= n:
                return bs
        return None

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int,
        valid_modes: set[int] | None = None,
    ) -> tuple[int, BatchDescriptor]:
        cfg = self.cfg
        allowed = set(valid_modes) if valid_modes else {NONE, PIECEWISE, FULL}

        # DISP:331-359 -- the (max_capture_size, 4096] B4096 bucket rule.
        if (
            cfg.integration_mode in ("full", "m15")
            and PIECEWISE in allowed
            and cfg.max_capture_size < num_tokens <= 4096
            and not uniform_decode
            and not has_lora
            and num_active_loras == 0
            and (
                self.regular_b4096 in self.piecewise_keys
                or self.pf4h_b4096 in self.piecewise_keys
            )
        ):
            return PIECEWISE, self.regular_b4096

        # DISP:361-368 -- fall through to NONE.
        if num_tokens > cfg.max_capture_size or allowed <= {NONE}:
            return NONE, BatchDescriptor(num_tokens)

        bs = self._round_up(num_tokens)
        if bs is None:
            return NONE, BatchDescriptor(num_tokens)
        if uniform_decode and FULL in allowed:
            key = BatchDescriptor(num_tokens=bs, num_reqs=bs, uniform=True)
            if key in self.full_keys:
                return FULL, key
        if PIECEWISE in allowed:
            key = BatchDescriptor(num_tokens=bs)
            if key in self.piecewise_keys:
                return PIECEWISE, key
        return NONE, BatchDescriptor(num_tokens)


# --------------------------------------------------------------------------
# the two seal predicates
# --------------------------------------------------------------------------
@dataclass
class Counters:
    steps: int = 0
    in_bucket: int = 0
    sealed: int = 0
    sealed_exact: int = 0
    sealed_ragged: int = 0
    refused_not_unanimous: int = 0
    refused_not_ready: int = 0
    eager_b4096: int = 0
    uniform_rescued: int = 0
    piecewise_steps: int = 0
    tok: int = 0
    in_bucket_tok: int = 0
    h2_min_local_tok: int = 1 << 30

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StepResult:
    modes: list[int] = field(default_factory=list)
    sealed: list[bool] = field(default_factory=list)
    unanimous: bool = False
    # H2: at least one rank sat at a padded 4096 while the group did not.
    h2: bool = False
    h2_blocker_tokens: tuple[int, ...] = ()


def replay_step(
    ranks: list[RankStep],
    cfg: Config,
    disp: Dispatcher,
    *,
    m23: bool,
    counters: list[Counters],
) -> StepResult:
    n = len(ranks)

    # ---- local, pre-DP dispatch on every rank ----------------------------
    local_modes, local_descs, uds = [], [], []
    for r in ranks:
        ud = r.is_uniform_decode()
        uds.append(ud)
        mode, desc = disp.dispatch(
            r.num_tokens, ud, r.has_lora, r.num_active_loras
        )
        local_modes.append(mode)
        local_descs.append(desc)

    # ---- coordinate_batch_across_dp (dp_utils) ---------------------------
    synced_mode = min(local_modes)                      # DPU:92-98
    should_dp_pad = synced_mode != 0                    # DPU:152
    padded_local = [d.num_tokens for d in local_descs]
    if should_dp_pad:                                   # DPU:77-89
        across = [max(padded_local)] * n
    else:
        across = list(padded_local)
    orig_counts = tuple(r.num_tokens for r in ranks)    # DPU:161

    # ---- readiness bit (row 4 of the all-reduce; design 4.3) -------------
    if m23 and cfg.m23 and cfg.graph_target == "pf4h" and cfg.dp_size == 8:
        ready_all = all(
            r.ready and not r.has_lora and r.num_active_loras == 0
            for r in ranks
        )
    else:
        ready_all = False

    # ---- re-dispatch with the DP-agreed count ----------------------------
    modes, descs = [], []
    for i, r in enumerate(ranks):
        mode, desc = disp.dispatch(
            across[i],
            uds[i],
            r.has_lora,
            r.num_active_loras,
            valid_modes={synced_mode},
        )
        modes.append(mode)
        descs.append(desc)

    m23_active = (
        m23 and cfg.m23 and cfg.dp_size == 8
        and cfg.integration_mode in ("full", "m15")
    )
    # A property of the STEP, not of the predicate: is the whole group padded
    # into the B4096 bucket under a synced PIECEWISE mode?  Both predicates
    # are reported against this same denominator so the arms are comparable.
    group_b4096 = bool(
        synced_mode == PIECEWISE and all(v == 4096 for v in across)
    )
    b4096_unanimous = bool(m23_active and group_b4096)
    ragged_seal = bool(b4096_unanimous and ready_all)
    exact = orig_counts == (4096,) * n

    out = StepResult(unanimous=group_b4096)
    if not group_b4096 and any(d.num_tokens == 4096 for d in descs):
        out.h2 = True
        # the ranks whose LOCAL dispatch returned NONE are what collapsed the
        # synced mode and cancelled DP padding (design 4.1, hazard H2)
        out.h2_blocker_tokens = tuple(
            ranks[i].num_tokens
            for i in range(n)
            if local_modes[i] == NONE
        )
    for i in range(n):
        c = counters[i]
        mode = modes[i]
        desc = descs[i]
        c.steps += 1
        c.tok += ranks[i].num_tokens

        if m23_active:
            sealed = ragged_seal
        else:
            # the pre-M23 exact gate (GMR:3947-3971)
            sealed = (
                cfg.integration_mode in ("full", "m15")
                and cfg.dp_size == 8
                and not uds[i]
                and not ranks[i].has_lora
                and desc.num_tokens == 4096
                and exact
                and cfg.graph_target == "pf4h"
                and ranks[i].ready
            )

        rescued = False
        if (
            b4096_unanimous
            and mode != PIECEWISE
            and (ragged_seal or cfg.uniform_rescue)
        ):
            mode = PIECEWISE
            desc = BatchDescriptor(num_tokens=4096)
            rescued = True

        if (
            desc.num_tokens == 4096
            and cfg.graph_target == "pf4h"
            and not sealed
        ):
            mode = NONE                                  # GMR:3973-3981

        if sealed and mode != PIECEWISE:
            raise AssertionError(
                f"rank {i}: sealed step is {MODE_NAME[mode]}, not PIECEWISE "
                "(risk R3 -- this would trip M15-DESC-012 mid-serve)"
            )

        if group_b4096:
            c.in_bucket += 1
            c.in_bucket_tok += ranks[i].num_tokens
            if m23_active and not ready_all:
                c.refused_not_ready += 1
            if mode == NONE:
                c.eager_b4096 += 1
        elif desc.num_tokens == 4096:
            c.refused_not_unanimous += 1
            c.h2_min_local_tok = min(c.h2_min_local_tok, min(orig_counts))
        if rescued and mode == PIECEWISE:
            c.uniform_rescued += 1
        if sealed:
            c.sealed += 1
            if exact:
                c.sealed_exact += 1
            else:
                c.sealed_ragged += 1
        if mode == PIECEWISE:
            c.piecewise_steps += 1

        out.modes.append(mode)
        out.sealed.append(sealed)
    return out


def replay(
    trace: list[list[RankStep]], cfg: Config, *, m23: bool
) -> tuple[list[Counters], dict]:
    """Returns (per-rank counters, H2 diagnostics)."""
    disp = Dispatcher(cfg)
    counters = [Counters() for _ in range(cfg.dp_size)]
    diag = {"h2_steps": 0, "h2_blockers": {}}
    for ranks in trace:
        res = replay_step(ranks, cfg, disp, m23=m23, counters=counters)
        if res.h2:
            diag["h2_steps"] += 1
            for tok in res.h2_blocker_tokens:
                diag["h2_blockers"][tok] = diag["h2_blockers"].get(tok, 0) + 1
    return counters, diag


# --------------------------------------------------------------------------
# trace IO
# --------------------------------------------------------------------------
def parse_trace(path: str, dp_size: int) -> list[list[RankStep]]:
    trace: list[list[RankStep]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                rows = [{"num_tokens": int(v)} for v in obj]
            else:
                rows = obj["ranks"]
            if len(rows) != dp_size:
                raise SystemExit(
                    f"{path}:{lineno}: {len(rows)} ranks, expected {dp_size}"
                )
            trace.append(
                [
                    RankStep(
                        num_tokens=int(r["num_tokens"]),
                        num_reqs=int(r.get("num_reqs", 1)),
                        max_num_scheduled_tokens=(
                            int(r["max_num_scheduled_tokens"])
                            if r.get("max_num_scheduled_tokens") is not None
                            else None
                        ),
                        uniform_decode=r.get("uniform_decode"),
                        has_lora=bool(r.get("has_lora", False)),
                        num_active_loras=int(r.get("num_active_loras", 0)),
                        ready=bool(r.get("ready", True)),
                    )
                    for r in rows
                ]
            )
    return trace


def parse_m23_trace(path: str, dp_size: int) -> list[list[RankStep]]:
    """Parse ``M23_TRACE ... n_orig=[a, b, ...]`` lines out of a server log."""
    trace = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "M23_TRACE" not in line or "n_orig=" not in line:
                continue
            vec = line.split("n_orig=", 1)[1].strip()
            vec = vec[vec.index("[") : vec.index("]") + 1]
            vals = [int(v) for v in json.loads(vec)]
            if len(vals) != dp_size:
                continue
            trace.append([RankStep(num_tokens=v) for v in vals])
    return trace


def synthesize(steps: int, dp_size: int, seed: int = 0) -> list[list[RankStep]]:
    """A c32p-shaped trace: ~38% in-bucket, mostly ragged, some decode ranks.

    Chunked prefill at max_num_batched_tokens=4096 with prefix caching gives a
    handful of full 4096 chunks, many partial chunks in (512, 4096), and a
    long tail of small decode batches; ``fail_min_tok=1`` says the modal
    in-bucket failure is one rank running a single-token decode.
    """
    rng = random.Random(seed)
    trace = []
    for _ in range(steps):
        heavy = rng.random() < 0.383
        ranks = []
        for _r in range(dp_size):
            if heavy:
                roll = rng.random()
                if roll < 0.55:
                    n = 4096
                elif roll < 0.90:
                    n = rng.randrange(520, 4096)
                elif roll < 0.97:
                    n = 1                       # the modal H1 rank
                else:
                    n = rng.randrange(2, 512)
                nreq = 1 if n == 1 else max(1, n // rng.randrange(512, 4097))
                ranks.append(
                    RankStep(
                        num_tokens=n,
                        num_reqs=nreq if n > 1 else 1,
                        max_num_scheduled_tokens=1 if n == 1 else n,
                    )
                )
            else:
                nreq = rng.randrange(1, 33)
                ranks.append(
                    RankStep(
                        num_tokens=nreq,
                        num_reqs=nreq,
                        max_num_scheduled_tokens=1,
                    )
                )
        trace.append(ranks)
    return trace


def dump_trace(trace: list[list[RankStep]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# M23 offline dispatcher replay trace (one step per line)\n")
        for i, ranks in enumerate(trace):
            f.write(
                json.dumps(
                    {
                        "step": i,
                        "ranks": [
                            {
                                "num_tokens": r.num_tokens,
                                "num_reqs": r.num_reqs,
                                "max_num_scheduled_tokens": (
                                    r.max_num_scheduled_tokens
                                    if r.max_num_scheduled_tokens is not None
                                    else r.num_tokens
                                ),
                            }
                            for r in ranks
                        ],
                    }
                )
                + "\n"
            )


# --------------------------------------------------------------------------
def total(counters: list[Counters]) -> Counters:
    t = Counters()
    for c in counters:
        for k, v in c.as_dict().items():
            if k == "h2_min_local_tok":
                t.h2_min_local_tok = min(t.h2_min_local_tok, v)
            else:
                setattr(t, k, getattr(t, k) + v)
    return t


def report(
    name: str, result: tuple[list[Counters], dict], per_rank: bool
) -> Counters:
    counters, diag = result
    t = total(counters)
    n = len(counters)
    print(f"\n=== {name} ===")
    print("  (mean per rank; [lo..hi] shown when the ranks disagree)")
    for label, key in (
        ("steps", "steps"),
        ("in_bucket", "in_bucket"),
        ("sealed", "sealed"),
        ("  sealed_exact", "sealed_exact"),
        ("  sealed_ragged", "sealed_ragged"),
        ("refused_not_unanimous", "refused_not_unanimous"),
        ("refused_not_ready", "refused_not_ready"),
        ("eager_b4096", "eager_b4096"),
        ("uniform_rescued", "uniform_rescued"),
        ("piecewise_steps", "piecewise_steps"),
    ):
        vals = [getattr(c, key) for c in counters]
        spread = "" if min(vals) == max(vals) else f"   [{min(vals)}..{max(vals)}]"
        print(f"  {label:<23}{sum(vals) // n:>6}{spread}")
    if t.tok:
        print(
            f"  token-weighted in-bucket share  "
            f"{100.0 * t.in_bucket_tok / t.tok:.1f}%   (probe P1)"
        )
    if diag["h2_steps"]:
        blockers = sorted(diag["h2_blockers"].items(), key=lambda kv: -kv[1])[:6]
        print(
            f"  H2 diagnostic: {diag['h2_steps']} step(s) had >=1 rank padded "
            f"to 4096 while the group was NOT padded.\n"
            "    Cause: those ranks' peers dispatched to NONE, collapsing the\n"
            "    synced mode (dp_utils:92-98,152).  Under the PF4H patch a\n"
            "    small MIXED batch has no PIECEWISE key at all "
            "(cudagraph_dispatcher:204-212,\n"
            "    design 7.3) -- pre-existing, identical in both arms.\n"
            f"    blocking rank token counts (count): {blockers}"
        )
    if per_rank:
        for i, c in enumerate(counters):
            print(
                f"    rank {i}: in_bucket={c.in_bucket} sealed={c.sealed} "
                f"ragged={c.sealed_ragged} not_unanimous={c.refused_not_unanimous} "
                f"eager_b4096={c.eager_b4096} rescued={c.uniform_rescued}"
            )
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--trace", help="JSONL trace (see the module docstring)")
    src.add_argument("--from-m23-trace", help="server log with M23_TRACE lines")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="replay a synthetic c32p-shaped trace instead of a file",
    )
    ap.add_argument("--emit-synthetic", metavar="PATH", help="write a synthetic trace and exit")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dp-size", type=int, default=8)
    ap.add_argument("--graph-target", choices=("pf4h", "stock"), default="pf4h")
    ap.add_argument("--integration-mode", default="m15")
    ap.add_argument("--max-capture-size", type=int, default=DEFAULT_MAX_CAPTURE_SIZE)
    ap.add_argument("--no-uniform-rescue", action="store_true")
    ap.add_argument("--per-rank", action="store_true")
    ap.add_argument(
        "--no-gate",
        action="store_true",
        help="report only; do not exit non-zero when a gate fails",
    )
    args = ap.parse_args(argv)

    if args.emit_synthetic:
        dump_trace(synthesize(args.steps, args.dp_size, args.seed), args.emit_synthetic)
        print(f"wrote {args.steps} steps to {args.emit_synthetic}")
        return 0

    if not (args.trace or args.from_m23_trace or args.synthetic):
        ap.error("one of --trace / --from-m23-trace / --synthetic is required")

    if args.trace:
        trace = parse_trace(args.trace, args.dp_size)
        source = args.trace
    elif args.from_m23_trace:
        trace = parse_m23_trace(args.from_m23_trace, args.dp_size)
        source = args.from_m23_trace
    else:
        trace = synthesize(args.steps, args.dp_size, args.seed)
        source = f"synthetic(steps={args.steps}, seed={args.seed})"

    if not trace:
        print(f"no steps parsed from {source}", file=sys.stderr)
        return 2

    cfg = Config(
        dp_size=args.dp_size,
        graph_target=args.graph_target,
        integration_mode=args.integration_mode,
        max_capture_size=args.max_capture_size,
        uniform_rescue=not args.no_uniform_rescue,
    )

    print(f"trace: {source}   steps={len(trace)}   dp_size={cfg.dp_size}")
    print(
        f"config: graph_target={cfg.graph_target} "
        f"integration_mode={cfg.integration_mode} "
        f"max_capture_size={cfg.max_capture_size} "
        f"uniform_rescue={cfg.uniform_rescue}"
    )

    old_result = replay(trace, cfg, m23=False)
    new_result = replay(trace, cfg, m23=True)
    old = report("OLD predicate (all eight ranks exactly 4096)",
                 old_result, args.per_rank)
    new = report("NEW predicate (M23 ragged seal)", new_result, args.per_rank)

    n = cfg.dp_size
    print("\n=== delta ===")
    if old.sealed:
        print(f"  coverage multiplier   {new.sealed / old.sealed:.1f}x")
    else:
        print(f"  coverage multiplier   inf ({old.sealed // n} -> {new.sealed // n})")
    print(f"  sealed/rank           {old.sealed // n} -> {new.sealed // n}")
    print(f"  eager B4096/rank      {old.eager_b4096 // n} -> {new.eager_b4096 // n}")

    print("\n=== L0 gates (design 8.3) ===")
    ok = True
    new_counters, _ = new_result
    for i, c in enumerate(new_counters):
        if c.sealed != c.in_bucket:
            ok = False
            print(
                f"  FAIL rank {i}: sealed={c.sealed} != in_bucket={c.in_bucket}"
            )
        if c.refused_not_unanimous:
            ok = False
            print(
                f"  FAIL rank {i}: refused_not_unanimous="
                f"{c.refused_not_unanimous} (expected 0; H2 class -- see the "
                "H2 diagnostic above)"
            )
        if c.eager_b4096:
            ok = False
            print(f"  FAIL rank {i}: eager_b4096={c.eager_b4096} (expected 0)")
    if ok:
        print("  PASS  sealed == in_bucket, refused_not_unanimous == 0, "
              "eager_b4096 == 0 on every rank")
    return 0 if (ok or args.no_gate) else 1


if __name__ == "__main__":
    sys.exit(main())
