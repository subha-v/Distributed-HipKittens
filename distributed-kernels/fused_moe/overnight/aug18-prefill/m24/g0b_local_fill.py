"""G0b (local surrogate): per-call real-row (n_orig) distribution from the
captured route corpus, and the M24 fill-aware cost model it feeds.

Corpus: 8 worker .npz files x 64 B4096 calls = 512 rank-calls of raw
topk_ids [4096,8] uint8 + fp16 topk_weights, captured on the m15+M23 server
at cell c32p (routecap1).

Degeneracy discrimination (identical to CORPUS_FINDINGS.md):
  * group rows by their sorted top-8 expert set; take the modal group.
  * DUMMY CALL  <=> modal group covers > 95% of the 4096 rows.
    This reproduces the file's 160/512 dummy count EXACTLY.
  * PAD GROUP within any call <=> modal group has >= 8 rows AND the
    across-row STD of its top-8 weights is < 0.02 (near-zero => identical
    hidden states => DP-dummy / pad rows).  The weight test is what
    separates pads from the genuine hot-set modal pattern {0..7}, which
    covers ~16% of rows in real calls but carries VARYING weights.
  n_real = 4096 - |pad group| (or 4096 when no group qualifies).

Outputs: n_real histogram, fill f = n_real/4096, T_eff = round_up(n_real,256),
padding multipliers before/after M24, and the distribution-weighted MoE cost
ratio rho = (1-phi) + phi*(T_eff/4096).
"""
import glob
import numpy as np

ROUTES = ("/private/tmp/claude-501/-Users-subha-repos-Distributed-HipKittens/"
          "4077c4da-1965-47cb-aa86-44e5f55b46c4/scratchpad/routes/*.npz")
T_MAX, TILE = 4096, 256
DUMMY_PF = 0.95      # modal-group coverage above which the whole call is dummy
PAD_MIN_ROWS = 8     # a modal group smaller than this is not treated as pad
PAD_WSTD = 0.02      # near-zero across-row weight STD => identical hidden states


def analyze():
    files = sorted(glob.glob(ROUTES))
    if not files:
        raise SystemExit(f"corpus not found: {ROUTES}")
    n_real, is_dummy, pad_wstd = [], [], []
    for f in files:
        z = np.load(f)
        ids, w = z["topk_ids"], z["topk_weights"].astype(np.float32)
        for c in range(ids.shape[0]):
            rows = ids[c]
            view = np.ascontiguousarray(np.sort(rows, axis=1)).view(
                [("", rows.dtype)] * 8).ravel()
            _, inv, cnt = np.unique(view, return_inverse=True,
                                    return_counts=True)
            m = int(np.argmax(cnt))
            mask = inv == m
            pf = cnt[m] / rows.shape[0]
            wstd = float(w[c][mask].std(axis=0).mean())
            pad = int(cnt[m]) if (cnt[m] >= PAD_MIN_ROWS and wstd < PAD_WSTD) else 0
            n_real.append(rows.shape[0] - pad)
            is_dummy.append(bool(pf > DUMMY_PF))
            pad_wstd.append(wstd)
    return (np.array(n_real, float), np.array(is_dummy), np.array(pad_wstd),
            len(files))


def teff(n):
    return np.minimum(np.ceil(n / TILE) * TILE, T_MAX)


def q(x, p):
    return float(np.percentile(x, p))


def rho(t, phi):
    return float(np.mean((1 - phi) + phi * (t / T_MAX)))


def report(n_real, is_dummy, wstd, nfiles):
    L = []
    P = L.append
    n_all, n_nd = n_real, n_real[~is_dummy]
    P(f"files={nfiles} calls={len(n_real)} dummy_calls={int(is_dummy.sum())} "
      f"({is_dummy.mean()*100:.1f}%)  [CORPUS_FINDINGS: 160/512 = 31.2%]")
    for name, x in (("ALL CALLS", n_all), ("NON-DUMMY", n_nd)):
        f = x / T_MAX
        t = teff(x)
        P(f"\n== {name} (n={len(x)}) ==")
        P(f"n_real   mean={x.mean():8.1f} p10={q(x,10):7.0f} p50={q(x,50):7.0f} p90={q(x,90):7.0f}")
        P(f"fill f   mean={f.mean():8.4f} p10={q(f,10):7.4f} p50={q(f,50):7.4f} p90={q(f,90):7.4f}")
        P(f"T_eff    mean={t.mean():8.1f} p10={q(t,10):7.0f} p50={q(t,50):7.0f} p90={q(t,90):7.0f}")
        P("deciles n_real: " + " ".join(f"{q(x,d):.0f}" for d in range(0, 101, 10)))
        P("deciles T_eff : " + " ".join(f"{q(t,d):.0f}" for d in range(0, 101, 10)))
        nz = x > 0
        P(f"pad multiplier PRE-M24  (4096/n_real): mean={np.mean(T_MAX/x[nz]):.3f} "
          f"(harmonic/agg = {T_MAX*nz.sum()/x[nz].sum():.3f})")
        P(f"pad multiplier POST-M24 (T_eff/n_real): mean={np.mean(t[nz]/x[nz]):.3f} "
          f"(agg = {t[nz].sum()/x[nz].sum():.3f})")
        P(f"mean T_eff/4096 (work retained) = {np.mean(t)/T_MAX:.4f}")
        for phi in (0.80, 0.90):
            P(f"rho(phi={phi:.2f}) = {rho(t, phi):.4f}  "
              f"=> step speedup {1/rho(t, phi):.3f}x")
    # tier-1 only
    t1 = np.where(is_dummy, 0.0, float(T_MAX))
    P("\n== TIER-1 ONLY (skip dummy calls entirely, full 4096 otherwise) ==")
    P(f"mean T_eff/4096 = {t1.mean()/T_MAX:.4f}")
    for phi in (0.80, 0.90):
        P(f"rho(phi={phi:.2f}) = {rho(t1, phi):.4f}  => step speedup {1/rho(t1, phi):.3f}x")
    return "\n".join(L)


if __name__ == "__main__":
    print(report(*analyze()))
