#!/usr/bin/env python3
"""In-process expert-skew observer, v2: histograms (v1, verbatim) + RAW ROUTES.

v1 (m15_router_skew.py) is preserved byte-for-byte in behaviour: same hook
site, same per-layer histograms, same per-call coverage/max-rank histograms,
same incremental JSON flush, same SIGTERM/atexit dump.  v2 ADDS one thing:

  RAW per-chunk route capture.  The first M15_ROUTE_CAPTURE_MAX router calls
  per rank whose token count equals M15_ROUTE_CAPTURE_TOKENS (default 4096 =
  the prefill chunk the MoK harness replays) are stored VERBATIM --
  topk_ids as [tokens,8] uint8 (256 experts fit a byte) and topk_weights as
  [tokens,8] float16 -- and written to $M15_SKEW_OUT/routes/ as one .npz per
  rank, flushed every M15_ROUTE_CAPTURE_FLUSH captures and on teardown.

Why: every kernel-level MoK result to date replays an AGGREGATE histogram
(i.i.d. Gumbel-top-k, K0_MOK_ROUTE_HIST) which fixes only the marginal expert
popularity.  Real chunks are RUN-CORRELATED -- adjacent tokens of a prompt
route alike, so a contiguous chunk concentrates destinations far beyond the
aggregate.  That correlation is the standing explanation for the serving
regressions the aggregate replay never predicted, and it has never been
measured on our kernel.  These files are the input that makes it measurable
(loader: benchmarks/mok_synthetic_prefill/route_replay.py).

Enabled by M15_SKEW_OUT=<dir> exactly as v1.  Raw capture is additionally
gated on M15_ROUTE_CAPTURE_MAX > 0 (default 64 => ~6 MB/rank resident).

Serving-safety rules obeyed here:
  * no torch.cuda.synchronize() anywhere;
  * the only device->host traffic added is two ~128 KB .cpu() copies per
    CAPTURED call, and the capture count is hard-bounded;
  * captures are SKIPPED while the stream is graph-capturing (a D2H copy
    inside a capture is illegal); the 4096-token prefill chunk we want runs
    eager, and the skip count is recorded so the provenance is explicit;
  * every added path is wrapped so a diagnostic can never break serving.

Dump layout under M15_SKEW_OUT:
  skew_rank<N>_pid<PID>.json      v1 histograms (unchanged)
  captures/route_<layer>_r<N>_p<PID>_<nnnnn>.npy   v1 opt-in per-layer .npy
  routes/routes_rank<N>_pid<PID>.npz               v2 raw capture (NEW)
"""
from __future__ import annotations
import atexit, json, os, sys, threading, time

GLOBAL_EXPERTS = 256

# v2 raw-capture .npz contract.  route_replay.py validates exactly these keys;
# keep the two in sync (route_replay.CAPTURE_KEYS / CAPTURE_FORMAT_VERSION).
ROUTE_CAPTURE_FORMAT_VERSION = 1


def install() -> None:
    out_dir = os.environ.get("M15_SKEW_OUT", "").strip()
    if not out_dir:
        return
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.router import fused_moe_router as R
    except Exception as exc:  # pragma: no cover
        print(f"M15_SKEW_OBSERVER import failed: {exc}", file=sys.stderr, flush=True)
        return

    cls = R.FusedMoERouter
    if getattr(cls, "_m15_skew_installed", False):
        return
    original = cls.select_experts
    lock = threading.Lock()
    hist: dict[str, "torch.Tensor"] = {}
    steps: dict[str, int] = {}
    tokens: dict[str, int] = {}
    step_ms: list[float] = []
    capsplit = {"captured": 0, "eager": 0}
    # PER-CALL statistics for the M18 static-replication question: the
    # aggregate histogram fixes the MEAN replica-set coverage, but the kernel
    # pays per-chunk — a chunk whose hot experts are outside the set gets the
    # full skew penalty plus the replication carry cost.  These histograms
    # record the distribution the aggregate hides.
    _rep_env = os.environ.get(
        "M15_SKEW_REP_SET", "6,5,4,2,0,3,1,7,8,9,20,10,11,19,17,16"
    )
    rep_ids = [int(x) for x in _rep_env.split(",") if x.strip()]
    cov_hist = [0] * 50          # per-call rep-set coverage, 2% bins
    before_hist = [0] * 81       # per-call max rank load (x fair share), 0.1 bins
    after_hist = [0] * 81        # same, after ideal rep-set redistribution
    # RAW ROUTE CAPTURE for MoK-Updated replay: for the router keys listed in
    # M15_SKEW_CAPTURE_LAYERS (csv, e.g. "router041,router001"), save each
    # call's topk_ids verbatim as int16 .npy until the per-layer cap.  These
    # are the real chunks the harness rotates through its timed iterations.
    cap_layers = set(
        x.strip()
        for x in os.environ.get("M15_SKEW_CAPTURE_LAYERS", "").split(",")
        if x.strip()
    )
    cap_max = int(os.environ.get("M15_SKEW_CAPTURE_MAX", "64"))
    cap_counts: dict[str, int] = {}
    FLUSH_EVERY = int(os.environ.get("M15_SKEW_FLUSH_EVERY", "50"))
    _flush = {"fn": (lambda _r="": None)}
    last = {"t": None}
    seq = {"n": 0}
    # ---- v2 raw per-chunk route capture (the K0_MOK_ROUTE_FILE producer) ----
    # Bounded, layer-agnostic, token-count-gated: the first ROUTE_MAX calls of
    # exactly ROUTE_TOKENS rows, whichever layers they belong to.  In a DP
    # worker that is the first ROUTE_MAX MoE layers of the first full-size
    # prefill chunk — one real forward pass through the layer stack.
    ROUTE_MAX = int(os.environ.get("M15_ROUTE_CAPTURE_MAX", "64"))
    ROUTE_TOKENS = int(os.environ.get("M15_ROUTE_CAPTURE_TOKENS", "4096"))
    ROUTE_FLUSH = int(os.environ.get("M15_ROUTE_CAPTURE_FLUSH", "16"))
    route_ids: list = []          # [tokens,8] uint8 per captured call
    route_wgt: list = []          # [tokens,8] float16 per captured call
    route_layer: list = []        # layer key per captured call
    route_call: list = []         # that layer's own call index at capture time
    route_skips = {"capturing": 0, "shape": 0, "range": 0, "error": 0}
    route_state = {"written": 0}
    _route_flush = {"fn": (lambda _r="": None)}

    def observing_select_experts(self, hidden_states, router_logits,
                                 topk_indices_dtype=None, *, input_ids=None):
        topk_weights, topk_ids = original(
            self, hidden_states, router_logits,
            topk_indices_dtype=topk_indices_dtype, input_ids=input_ids,
        )
        try:
            # DO NOT filter on is_current_stream_capturing(). The first version
            # counted only out-of-capture calls and recorded NOTHING on either
            # arm: with cudagraphs on, the router's Python body runs during
            # CAPTURE (flag True, so it was skipped) and never again during
            # replay. Count every call and record the capture split instead, so
            # the data is there and the provenance is explicit.
            try:
                cap = bool(torch.cuda.is_current_stream_capturing())
            except Exception:
                cap = False
            due = False
            route_due = False
            if True:
                # Stable per-layer key; routers are per-MoE-layer instances.
                key = getattr(self, "layer_name", None) or getattr(
                    self, "_m15_key", None)
                if key is None:
                    with lock:
                        key = f"router{seq['n']:03d}"
                        seq["n"] += 1
                    try:
                        self._m15_key = key
                    except Exception:
                        pass
                flat = topk_ids.detach().reshape(-1).to(torch.int64)
                counts = torch.bincount(flat, minlength=GLOBAL_EXPERTS).cpu()
                if cap_layers and key in cap_layers:
                    with lock:
                        cap_n = cap_counts.get(key, 0)
                        cap_counts[key] = cap_n + 1
                    if cap_n < cap_max:
                        try:
                            import numpy as _np

                            cap_dir = os.path.join(out_dir, "captures")
                            os.makedirs(cap_dir, exist_ok=True)
                            rank_tag = os.environ.get("VLLM_DP_RANK") or "0"
                            _np.save(
                                os.path.join(
                                    cap_dir,
                                    f"route_{key}_r{rank_tag}_p{os.getpid()}"
                                    f"_{cap_n:05d}.npy",
                                ),
                                topk_ids.detach().cpu().numpy().astype(
                                    _np.int16
                                ),
                            )
                        except Exception:
                            pass
                # ---- v2: bounded RAW capture of full-size prefill chunks ----
                # Checked under the lock first so a race can never overrun the
                # bound (that bound is the memory contract: ROUTE_MAX * 96 KB).
                take = False
                if ROUTE_MAX > 0:
                    with lock:
                        take = len(route_ids) < ROUTE_MAX
                if take:
                    try:
                        if cap:
                            # A D2H copy inside a graph capture is illegal, and
                            # 4096-token prefill chunks run eager anyway.
                            with lock:
                                route_skips["capturing"] += 1
                        elif (
                            topk_ids.dim() != 2
                            or int(topk_ids.shape[0]) != ROUTE_TOKENS
                        ):
                            with lock:
                                route_skips["shape"] += 1
                        else:
                            import numpy as _np

                            _ids = (
                                topk_ids.detach()
                                .to(torch.int32)
                                .cpu()
                                .numpy()
                            )
                            _wgt = (
                                topk_weights.detach()
                                .to(torch.float16)
                                .cpu()
                                .numpy()
                            )
                            if _ids.min() < 0 or _ids.max() >= GLOBAL_EXPERTS:
                                with lock:
                                    route_skips["range"] += 1
                            else:
                                _ids8 = _ids.astype(_np.uint8)
                                with lock:
                                    if len(route_ids) < ROUTE_MAX:
                                        route_ids.append(_ids8)
                                        route_wgt.append(
                                            _np.ascontiguousarray(_wgt)
                                        )
                                        route_layer.append(str(key))
                                        route_call.append(
                                            int(steps.get(key, 0))
                                        )
                                        route_due = (
                                            len(route_ids) % ROUTE_FLUSH == 0
                                            or len(route_ids) == ROUTE_MAX
                                        )
                    except Exception:
                        with lock:
                            route_skips["error"] += 1
                total_slots = int(counts.sum())
                cov_bin = before_bin = after_bin = None
                if total_slots:
                    covered = int(counts[rep_ids].sum())
                    cov = covered / total_slots
                    rank_before = counts.reshape(8, 32).sum(1)
                    residual = counts.clone()
                    residual[rep_ids] = 0
                    rank_after = residual.reshape(8, 32).sum(1)
                    max_before = float(rank_before.max()) / total_slots * 8.0
                    max_after = (
                        float(rank_after.max()) / total_slots + cov / 8.0
                    ) * 8.0
                    cov_bin = min(int(cov * 50), 49)
                    before_bin = min(int(max_before / 0.1), 80)
                    after_bin = min(int(max_after / 0.1), 80)
                now = time.perf_counter()
                with lock:
                    if key not in hist:
                        hist[key] = torch.zeros(GLOBAL_EXPERTS, dtype=torch.int64)
                        steps[key] = 0
                        tokens[key] = 0
                    hist[key] += counts
                    steps[key] += 1
                    tokens[key] += int(topk_ids.shape[0])
                    capsplit["captured" if cap else "eager"] += 1
                    if cov_bin is not None:
                        cov_hist[cov_bin] += 1
                        before_hist[before_bin] += 1
                        after_hist[after_bin] += 1
                    if last["t"] is not None:
                        step_ms.append((now - last["t"]) * 1000.0)
                    last["t"] = now
                    total = sum(steps.values())
                    due = total % FLUSH_EVERY == 0
                if due:
                    # INCREMENTAL FLUSH. The teardown-only dump did not survive
                    # the driver's stop path -- workers are SIGKILLed after
                    # docker stop's grace period, so atexit/SIGTERM never ran and
                    # skew/stock/skew stayed empty 8 minutes after teardown.
                    # Writing every FLUSH_EVERY calls to a temp + atomic rename
                    # makes the evidence survive ANY teardown behaviour; the
                    # exit flush is now a bonus, not the only path.
                    _flush["fn"]("incremental")
                if route_due:
                    # Same crash-safety argument, same atomic-rename discipline:
                    # the raw capture is complete after ROUTE_MAX calls, long
                    # before teardown, so it must be on disk before teardown.
                    _route_flush["fn"]("incremental")
        except Exception:
            pass  # never break serving for a diagnostic
        return topk_weights, topk_ids

    cls.select_experts = observing_select_experts
    cls._m15_skew_installed = True

    def _rank_tag() -> str:
        return (
            os.environ.get("VLLM_DP_RANK")
            or os.environ.get("RANK")
            or os.environ.get("LOCAL_RANK")
            or "0"
        )

    def dump(_reason: str = "exit") -> None:
        try:
            # MANY processes in a vLLM server import sitecustomize -- API
            # servers, helpers, short-lived children -- and most never route a
            # token.  They would write layers=0 files and, since anything
            # without a rank env falls back to "0", clobber the real rank-0
            # worker's histogram.  Only a process that actually observed
            # routing may write.
            if not hist:
                return
            rank = _rank_tag()
            os.makedirs(out_dir, exist_ok=True)
            with lock:
                payload = {
                    "rank": int(rank),
                    "source": "select_experts-inprocess",
                    "global_experts": GLOBAL_EXPERTS,
                    "per_layer_steps": dict(steps),
                    "per_layer_tokens": dict(tokens),
                    "per_layer_histogram": {
                        k: [int(x) for x in v.tolist()] for k, v in hist.items()
                    },
                    "step_ms": [round(x, 4) for x in step_ms],
                    "capture_split": dict(capsplit),
                    "percall": {
                        "rep_set": list(rep_ids),
                        "cov_hist_2pct_bins": list(cov_hist),
                        "max_rank_before_hist_0p1_bins": list(before_hist),
                        "max_rank_after_hist_0p1_bins": list(after_hist),
                    },
                    "route_capture": {
                        "format_version": ROUTE_CAPTURE_FORMAT_VERSION,
                        "max": ROUTE_MAX,
                        "tokens": ROUTE_TOKENS,
                        "held": len(route_ids),
                        "written": route_state["written"],
                        "skips": dict(route_skips),
                        "layers": list(route_layer),
                    },
                    "pid": os.getpid(),
                    "dp_rank": os.environ.get("VLLM_DP_RANK"),
                }
            # TP=1/DP=8: more than one process can report rank 0, so the
            # PID disambiguates and nothing can clobber anything else.
            path = os.path.join(
                out_dir,
                "skew_rank%s_pid%d.json" % (payload["rank"], os.getpid()),
            )
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
            if _reason != "incremental":
                print(f"M15_SKEW_OBSERVER wrote {path} layers={len(hist)}",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"M15_SKEW_OBSERVER dump failed: {exc}",
                  file=sys.stderr, flush=True)

    def dump_routes(_reason: str = "exit") -> None:
        """Write the raw capture as ONE .npz per rank, atomically.

        Keys (route_replay.CAPTURE_KEYS): format_version, topk_ids
        [N,tokens,8] uint8, topk_weights [N,tokens,8] float16, layers [N] str,
        call_index [N] int32, seq [N] int32, tokens, topk, num_experts, rank,
        pid, captured_utc.  Rewritten in full on every flush -- N is at most 64
        and the payload at most ~6 MB, so a full rewrite is cheaper than any
        append protocol and is trivially crash-safe under atomic rename.
        """
        try:
            with lock:
                n = len(route_ids)
                if n == 0:
                    return
                ids = list(route_ids)
                wgt = list(route_wgt)
                lay = list(route_layer)
                cidx = list(route_call)
            import numpy as _np

            rank = _rank_tag()
            rdir = os.path.join(out_dir, "routes")
            os.makedirs(rdir, exist_ok=True)
            path = os.path.join(
                rdir, "routes_rank%s_pid%d.npz" % (rank, os.getpid())
            )
            tmp = path + ".tmp.npz"
            payload = dict(
                format_version=_np.int32(ROUTE_CAPTURE_FORMAT_VERSION),
                topk_ids=_np.stack(ids).astype(_np.uint8),
                topk_weights=_np.stack(wgt).astype(_np.float16),
                layers=_np.array(lay, dtype="U32"),
                call_index=_np.asarray(cidx, dtype=_np.int32),
                seq=_np.arange(n, dtype=_np.int32),
                tokens=_np.int32(ROUTE_TOKENS),
                topk=_np.int32(ids[0].shape[1]),
                num_experts=_np.int32(GLOBAL_EXPERTS),
                rank=_np.int32(int(rank)),
                pid=_np.int32(os.getpid()),
                captured_utc=_np.array(
                    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), dtype="U32"
                ),
            )
            with open(tmp, "wb") as fh:
                _np.savez(fh, **payload)
            os.replace(tmp, path)
            route_state["written"] = n
            if _reason != "incremental":
                print(
                    f"M15_ROUTE_CAPTURE wrote {path} calls={n} "
                    f"skips={route_skips}",
                    file=sys.stderr, flush=True,
                )
        except Exception as exc:  # pragma: no cover
            print(f"M15_ROUTE_CAPTURE dump failed: {exc}",
                  file=sys.stderr, flush=True)

    _flush["fn"] = dump
    _route_flush["fn"] = dump_routes

    def dump_all(_reason: str = "exit") -> None:
        dump(_reason)
        dump_routes(_reason)

    atexit.register(dump_all)
    # Also dump on SIGTERM, which is how the driver stops servers.
    try:
        import signal
        prev = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):
            dump_all("sigterm")
            if callable(prev):
                prev(signum, frame)
            else:
                sys.exit(0)
        signal.signal(signal.SIGTERM, on_term)
    except Exception:
        pass
    print(
        "M15_SKEW_OBSERVER v2 installed on FusedMoERouter.select_experts "
        f"(route_capture max={ROUTE_MAX} tokens={ROUTE_TOKENS})",
        file=sys.stderr, flush=True,
    )


install()
