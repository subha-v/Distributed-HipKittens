"""M24 gate: are DP-dummy calls RANK-SYNCHRONOUS across the 8 workers?

Decision this feeds: the tier-1 "skip the whole dummy step" optimization needs
ALL 8 ranks dummy on the SAME serving step, because the megakernel's slab
rendezvous couples the ranks. A lone dummy rank can only shed its own
send-side work. G0B_LOCAL_FILL_HISTOGRAM.md caveat 5.3 flags this as
"the first thing to verify"; this script does the local half of that check.

Corpus: the same 8 worker .npz files x 64 B4096 calls used by
g0b_local_fill.py. The dummy classification is REUSED VERBATIM from that
script (modal sorted-top-8-set coverage > 0.95 => dummy call), so counts are
directly comparable (160/512 = 31.2%).

ASSUMPTION under test: call index i is the SAME serving step in every worker
file. The corpus carries no step key, so this can only be validated by
internal consistency -- if the assumption holds AND dummy steps are
rank-synchronous, the per-index dummy count c[i] = sum_w d_w[i] must be
bimodal on {0, 8}. Scattered dummies, or a broken index alignment, both
produce a binomial-looking middle. Lag robustness (-3..+3 per worker) is run
to rule out a capture that started a call or two apart.
"""
import glob
import itertools
import numpy as np

ROUTES = ("/private/tmp/claude-501/-Users-subha-repos-Distributed-HipKittens/"
          "4077c4da-1965-47cb-aa86-44e5f55b46c4/scratchpad/routes/*.npz")
T_MAX, TILE = 4096, 256
DUMMY_PF = 0.95      # identical to g0b_local_fill.py
PAD_MIN_ROWS = 8
PAD_WSTD = 0.02
LAGS = range(-3, 4)


def classify(f):
    """Per-call (is_dummy, n_real) for one worker file -- logic copied from
    g0b_local_fill.py::analyze so the two scripts agree exactly."""
    z = np.load(f)
    ids, w = z["topk_ids"], z["topk_weights"].astype(np.float32)
    dummy, n_real = [], []
    for c in range(ids.shape[0]):
        rows = ids[c]
        view = np.ascontiguousarray(np.sort(rows, axis=1)).view(
            [("", rows.dtype)] * 8).ravel()
        _, inv, cnt = np.unique(view, return_inverse=True, return_counts=True)
        m = int(np.argmax(cnt))
        mask = inv == m
        pf = cnt[m] / rows.shape[0]
        wstd = float(w[c][mask].std(axis=0).mean())
        pad = int(cnt[m]) if (cnt[m] >= PAD_MIN_ROWS and wstd < PAD_WSTD) else 0
        n_real.append(rows.shape[0] - pad)
        dummy.append(bool(pf > DUMMY_PF))
    return np.array(dummy), np.array(n_real, float)


def load():
    files = sorted(glob.glob(ROUTES))
    if not files:
        raise SystemExit(f"corpus not found: {ROUTES}")
    D, R = [], []
    for f in files:
        d, r = classify(f)
        D.append(d)
        R.append(r)
    n = min(len(d) for d in D)
    return files, np.array([d[:n] for d in D]), np.array([r[:n] for r in R])


def kappa(a, b):
    """Cohen's kappa for two boolean vectors."""
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def agreement(D):
    W = D.shape[0]
    obs = np.eye(W)
    kap = np.eye(W)
    for i, j in itertools.combinations(range(W), 2):
        o = float((D[i] == D[j]).mean())
        k = kappa(D[i], D[j])
        obs[i, j] = obs[j, i] = o
        kap[i, j] = kap[j, i] = k
    off = ~np.eye(W, dtype=bool)
    return obs, kap, float(obs[off].mean()), float(np.nanmean(kap[off]))


def hist_c(D):
    c = D.sum(axis=0)
    h = np.bincount(c, minlength=D.shape[0] + 1)
    return c, h


def expected_binomial(D):
    """Poisson-binomial expectation of the c[i] histogram under the null that
    each worker is dummy independently at its own observed rate."""
    W, n = D.shape
    p = D.mean(axis=1)
    dist = np.array([1.0])
    for pw in p:
        nxt = np.zeros(len(dist) + 1)
        nxt[:-1] += dist * (1 - pw)
        nxt[1:] += dist * pw
        dist = nxt
    return dist * n


def best_lags(D):
    """Per-worker lag in LAGS maximizing agreement with worker 0."""
    W, n = D.shape
    lags = [0]
    for w in range(1, W):
        best, bl = -1.0, 0
        for L in LAGS:
            if L >= 0:
                a, b = D[0][L:], D[w][:n - L] if L else D[w]
            else:
                a, b = D[0][:n + L], D[w][-L:]
            s = float((a == b).mean())
            if s > best:
                best, bl = s, L
        lags.append(bl)
    return lags


def apply_lags(M, lags):
    """Return the matrix restricted to indices where every lagged worker has
    a sample. Worker w contributes M[w][i + lag_w] at common index i."""
    n = M.shape[1]
    lo = max(0, -min(lags))
    hi = min(n, n - max(lags))
    return np.array([M[w][lo + lags[w]:hi + lags[w]] for w in range(M.shape[0])])


def block(tag, D, R, out):
    P = out.append
    c, h = hist_c(D)
    W, n = D.shape
    exp = expected_binomial(D)
    P(f"\n== {tag}  (workers={W}, aligned indices={n}) ==")
    P("per-worker dummy rate: " +
      " ".join(f"w{w}={D[w].mean()*100:.1f}%" for w in range(W)))
    P(f"pooled dummy rate = {D.mean()*100:.2f}%  "
      f"(g0b: 160/512 = 31.2% over all 64 calls)")
    P("c[i] histogram (c = # workers dummy at index i):")
    P("  c   :  " + " ".join(f"{k:5d}" for k in range(W + 1)))
    P("  obs :  " + " ".join(f"{int(v):5d}" for v in h))
    P("  indep: " + " ".join(f"{v:5.1f}" for v in exp))
    ext = (h[0] + h[W]) / n
    ext_e = (exp[0] + exp[W]) / n
    P(f"fraction of indices with c in {{0,{W}}}: {ext*100:.1f}%  "
      f"(independent-null expectation {ext_e*100:.1f}%)")
    obs, kap, mo, mk = agreement(D)
    P(f"pairwise observed agreement: mean={mo:.3f} "
      f"min={obs[~np.eye(W,dtype=bool)].min():.3f} "
      f"max={obs[~np.eye(W,dtype=bool)].max():.3f}")
    P(f"pairwise Cohen kappa      : mean={mk:.3f} "
      f"min={np.nanmin(kap[~np.eye(W,dtype=bool)]):.3f} "
      f"max={np.nanmax(kap[~np.eye(W,dtype=bool)]):.3f}")
    P("agreement matrix (observed / kappa):")
    for i in range(W):
        P("  w%d  " % i + " ".join(
            "  --- " if i == j else f"{obs[i,j]:.2f}/{kap[i,j]:+.2f}"
            for j in range(W)))
    # n_real correlation on indices where NO worker is dummy
    both = c == 0
    if both.sum() >= 4:
        sub = R[:, both]
        cc = np.corrcoef(sub)
        off = cc[~np.eye(W, dtype=bool)]
        P(f"n_real cross-worker correlation on the {int(both.sum())} "
          f"all-non-dummy indices: mean r={off.mean():+.3f} "
          f"min={off.min():+.3f} max={off.max():+.3f}")
        P("  n_real spread on those indices: " +
          f"mean={sub.mean():.0f} std_within_index={sub.std(axis=0).mean():.0f} "
          f"std_across_index={sub.mean(axis=0).std():.0f}")
    else:
        P(f"n_real correlation: only {int(both.sum())} all-non-dummy indices "
          "-- not computed")
    # partial-fill band: the alignment evidence that block structure cannot fake
    band = both & (R.min(axis=0) < T_MAX) & (R.max(axis=0) > 0)
    if band.sum() >= 4:
        sub = R[:, band]
        cc = np.corrcoef(sub)
        off = cc[~np.eye(W, dtype=bool)]
        P(f"n_real correlation on the {int(band.sum())} PARTIAL-FILL "
          f"non-dummy indices (0 < n_real < 4096 for some rank): "
          f"mean r={off.mean():+.3f} min={off.min():+.3f}")
        P("  index:  " + " ".join(f"{i:5d}" for i in np.flatnonzero(band)))
        P("  w0    : " + " ".join(f"{v:5.0f}" for v in R[0][band]))
        P("  spread: " + " ".join(f"{v:5.0f}" for v in sub.std(axis=0)))
    # dummy run structure -- are dummies contiguous blocks or interleaved?
    P("dummy pattern per worker (D=dummy, .=real), index 0 -> n-1:")
    for w in range(W):
        P("  w%d  %s" % (w, "".join("D" if x else "." for x in D[w])))
    return h, ext, mo, mk


def main():
    files, D, R = load()
    out = ["M24 dummy-step rank-synchrony probe",
           "corpus files (sorted):"]
    for i, f in enumerate(files):
        out.append(f"  w{i}: {f.split('/')[-1]}  calls={D.shape[1]} "
                   f"dummy={int(D[i].sum())}")
    block("RAW INDEX ALIGNMENT (lag 0)", D, R, out)
    out.append("\nlag-agreement profile vs w0 (rows = workers, cols = lag "
               "-3..+3); a flat profile means the lag test is weakly "
               "discriminating (block-structured dummies):")
    out.append("        " + " ".join(f"{L:+5d}" for L in LAGS))
    n = D.shape[1]
    for w in range(D.shape[0]):
        row = []
        for L in LAGS:
            a, b = (D[0][L:], D[w][:n - L]) if L >= 0 else (D[0][:n + L], D[w][-L:])
            row.append(float((a == b).mean()))
        out.append(f"  w{w}    " + " ".join(f"{v:5.3f}" for v in row))
    # n_real is the sharp alignment probe: unlike the block-structured dummy
    # vector, it is a continuously varying signal, so a wrong lag destroys it.
    out.append("\nn_real correlation r(w0, w_shifted) vs lag, mean over "
               "w1..w7 (sharp alignment probe -- block structure cannot fake "
               "this):")
    prof = []
    for L in LAGS:
        rs = []
        for w in range(1, D.shape[0]):
            a, b = (R[0][L:], R[w][:n - L]) if L >= 0 else (R[0][:n + L], R[w][-L:])
            rs.append(np.corrcoef(a, b)[0, 1])
        prof.append(np.mean(rs))
    out.append("        " + " ".join(f"{L:+6d}" for L in LAGS))
    out.append("  r     " + " ".join(f"{v:+6.3f}" for v in prof))
    lags = best_lags(D)
    out.append("\nbest per-worker lag vs w0 (search -3..+3): " +
               " ".join(f"w{w}={l:+d}" for w, l in enumerate(lags)))
    if any(lags):
        block("BEST-LAG ALIGNMENT", apply_lags(D, lags),
              apply_lags(R, lags), out)
    else:
        out.append("all best lags are 0 -- best-lag block identical to raw.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
