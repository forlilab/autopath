"""Frames from sMD pulls, state labels from r, and feature relevance tests."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

BIN_NM, MIN_PULLS, RHO_PRUNE, FISHER_FLOOR, BULK_NM = 0.05, 20, 0.8, 0.5, 0.3
MAX_KEPT = 30


def find_pulls(traj_dir, direction="forward"):
    out = []
    for dat in sorted(glob.glob(os.path.join(traj_dir, f"sMD_*_{direction}.dat"))):
        dcd = dat.replace(".dat", "_aligned.dcd")
        if os.path.exists(dcd):
            out.append((dat, dcd))
    return out


def pull_frames(dat, n_frames):
    d = pd.read_csv(dat, comment="#")
    t = d["time"].to_numpy(float)
    tk = t[-1] * np.arange(n_frames) / max(n_frames - 1, 1)
    return pd.DataFrame({"t_ps": tk,
                         "r": np.interp(tk, t, d["r_after"].to_numpy(float)),
                         "W": np.interp(tk, t, np.cumsum(d["dW_protocol"].to_numpy(float)))})


def labels_from_r(r, first_r, r_max):
    lab = np.full(len(r), "vestibule", dtype=object)
    lab[r <= np.percentile(first_r, 95)] = "bound"
    lab[r >= r_max - BULK_NM] = "bulk"
    return lab


def prune(V, names, score, rho=RHO_PRUNE):
    sd = V.std(0)
    ok = np.flatnonzero(sd > 1e-3 * (V.max(0) - V.min(0) + 1e-12))
    if len(ok) < 2:
        return list(ok)
    C = np.abs(np.corrcoef(V[:, ok].T))
    np.fill_diagonal(C, 1.0)
    cl = fcluster(linkage(squareform(1 - C, checks=False), "average"), 1 - rho, "distance")
    return [int(ok[np.flatnonzero(cl == c)[np.argmax(score[ok][cl == c])]]) for c in np.unique(cl)]


def fisher(x, y):
    a, b = x[y], x[~y]
    return float((a.mean() - b.mean()) ** 2 / (a.var() + b.var() + 1e-12))


def fisher_multi(x, labels, order):
    """var(class means, ddof=1) / mean(class variances), classes unweighted; = fisher() for 2 classes."""
    g = [x[labels == c] for c in order]
    return float(np.var([a.mean() for a in g], ddof=1) / (np.mean([a.var() for a in g]) + 1e-12))


def _resid_ranks(v, rr):
    """Ranks of v with the OLS fit on centred ranks of r removed, scaled to unit sd (0 if degenerate)."""
    rv = rankdata(v)
    rv = rv - rv.mean()
    if rr @ rr > 0:
        rv = rv - (rv @ rr) / (rr @ rr) * rr
    sd = rv.std()
    return rv / sd if sd > 1e-9 * len(rv) else np.zeros_like(rv)


def _usable_bins(x, W, r, pull):
    """{bin: (x, W, r) per-pull means} for bins with enough pulls and non-constant x, W means."""
    df = pd.DataFrame({"x": x, "W": W, "r": r, "bin": np.round(r / BIN_NM).astype(int), "pull": pull})
    out = {}
    for b, g in df.groupby(["bin", "pull"])[["x", "W", "r"]].mean().groupby(level=0):
        if len(g) >= MIN_PULLS and np.ptp(g.x.values) > 0 and np.ptp(g.W.values) > 0:
            out[b] = (g.x.values, g.W.values, g.r.values)
    return out


def hidden_variable_p(x, W, r, pull, speed, n_perm=1000, rng=None):
    """Family-wise permutation p of max_bin |rho_bar|, rho_bar = tanh(mean_g arctanh rho_g),
    rho_g the Spearman correlation of per-pull means of x and W partialling out r."""
    rng = rng or np.random.default_rng(0)
    speeds = np.unique(speed)
    if len(speeds) > 1:
        groups = [speed == s for s in speeds]
    else:
        pulls = np.unique(pull)
        half = set(rng.permutation(pulls)[: len(pulls) // 2])
        m = np.array([p in half for p in pull])
        groups = [m, ~m]
    tabs = [t for t in (_usable_bins(x[g], W[g], r[g], pull[g]) for g in groups) if t]
    if not tabs:
        return 1.0, 0.0
    bins = sorted(set.intersection(*(set(t) for t in tabs)))
    if not bins:
        return 1.0, 0.0
    z = np.zeros((n_perm + 1, len(bins)))
    for t in tabs:
        for k, b in enumerate(bins):
            xb, wb, rb = t[b]
            rr = rankdata(rb) - (len(rb) + 1) / 2
            rx, rw = _resid_ranks(xb, rr), _resid_ranks(wb, rr)
            n = len(rx)
            perm = rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
            rho = np.r_[rw @ rx, rw[perm] @ rx] / n
            z[:, k] += np.arctanh(np.clip(rho, -0.999, 0.999))
    rbar = np.tanh(z / len(tabs))
    k = int(np.argmax(np.abs(rbar[0])))
    null = np.abs(rbar[1:]).max(1)
    return float((1 + np.sum(null >= abs(rbar[0, k]))) / (1 + n_perm)), float(rbar[0, k])


def select(V, names, families, frames, labels, n_perm=1000, rng=None, hidden=True, order=None):
    """Keep Fisher-relevant (or F3) features, prune redundancy, cap at MAX_KEPT by Fisher.
    The hidden-variable p/rho are report-only (NaN when skipped): positions at fixed r are distorted by spring lag.
    order=None: bound vs vestibule (r classes); else multi-class Fisher over the ordered classes."""
    if order is None:
        y_bv = labels != "bulk"
        fis = np.array([fisher(V[y_bv, k], labels[y_bv] == "bound") for k in range(V.shape[1])])
    else:
        fis = np.array([fisher_multi(V[:, k], labels, order) for k in range(V.shape[1])])
    p, rho = np.full(len(names), np.nan), np.full(len(names), np.nan)
    if hidden:
        p[:], rho[:] = 1.0, 0.0
        for k, fam in enumerate(families):
            if fam != "F3":
                p[k], rho[k] = hidden_variable_p(V[:, k], frames.W.values, frames.r.values,
                                                 frames.pull.values, frames.speed.values, n_perm, rng)
    relevant = (fis >= FISHER_FLOOR) | (np.array(families) == "F3")
    idx = np.flatnonzero(relevant)
    kept = [int(idx[i]) for i in prune(V[:, idx], [names[i] for i in idx], fis[idx])]
    f3 = [k for k, f in enumerate(families) if f == "F3"]
    rest = sorted((k for k in kept if k not in f3), key=lambda k: -fis[k])
    kept = sorted(set(f3) | set(rest[:max(MAX_KEPT - len(f3), 0)]))
    return {"kept": kept, "fisher": fis, "p_hidden": p, "rho_hidden": rho}
