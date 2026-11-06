"""A direct port of kmatch 1.1.6's cross-validation without an outcome
(Mata: Kmatch_cv's M0, _Kmatch_cvs, Kmatch_i_PS_cvs, Kmatch_Ridge), loop by
loop and without weights, to check match_stream's window kernel
(kernels/kernel.py score_cv_pass) and summed criterion
(_kernel.score_criterion) on small data."""

import numpy as np


def kmatch_ridge(w, pi, pj, h, r):
    if pj.size == 1:
        return np.array([1.0])
    if np.all(pj == pj[0]):                     # invsym(variance()) has a zero
        return w / w.sum()
    W = w.sum()
    P = (pj * w).sum() / W
    D = pj - P
    S = (D ** 2 * w).sum()
    d = pi - P
    return w / W + w * D * (d / (S + r * h * abs(d)))


def kmatch_cvs(fs, fg, os_, og, h, K, r=None, sd=0.0):
    """MISE for focal scores fs (strata fg) against other-arm scores os_
    (strata og); K the kernel on u = (s_j - s_i) / h; r the ridge parameter
    (None: kernel matching); sd > 0: the limit factor."""
    n_o = os_.size
    C1 = np.full(fs.size, np.nan)
    cc1, cc2 = np.zeros(n_o), np.zeros(n_o)
    for i in range(fs.size):
        c = fs[i]
        win = np.flatnonzero((og == fg[i]) & (np.abs(os_ - c) < h))
        win = win[np.argsort(os_[win], kind="stable")]
        if win.size == 0:
            continue
        P = os_[win]
        w = K((P - c) / h)
        w1 = kmatch_ridge(w, c, P, h, r) if r is not None else w / w.sum()
        C1[i] = (P * w1).sum() / w1.sum()
        if win.size == 1:
            cc2[win[0]] += 1
            d = P.copy()
        elif np.any(np.abs(1 - w1) < 1e-8):
            d = np.array([C1[i] - (np.delete(P * w1, j).sum() / np.delete(w1, j).sum())
                          for j in range(win.size)])
        else:
            d = w1 * (P - C1[i]) / (1 - w1)
        cc1[win] += d
    M0 = fs.mean()
    ok = np.isfinite(C1)
    with np.errstate(divide="ignore", invalid="ignore"):
        M1 = (C1[ok].sum() - cc1) / (ok.sum() - cc2)
    v = (M0 - M1) ** 2
    mise = np.mean(v[np.isfinite(v)])
    if sd > 0:
        mise *= max(1.0, (h / sd) ** 0.015)
    return mise
