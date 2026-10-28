"""Hummer-Szabo reweighting of steered-MD frames to the unbiased density."""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

logger = logging.getLogger("autopath.pulling.hummer_szabo")

_KEYS = ["speed", "step"]
F_SOURCES = ("jarzynski", "cumulant", "force", "self_consistent")


def check_full_support(rows: pd.DataFrame) -> None:
    """Raise if any (speed, step) lacks a row from every pull of its speed.

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``trajname``, ``speed``, ``step``.

    Raises
    ------
    ValueError
        "not supported" when support is incomplete.
    """
    for speed, g in rows.groupby("speed"):
        n = g["trajname"].nunique()
        counts = g.groupby("step")["trajname"].nunique()
        short = counts[counts != n]
        if len(short):
            raise ValueError(
                f"not supported: speed {speed} has {len(short)} steps (first {short.index[0]}) "
                f"without rows from all {n} pulls; truncate to the common support first"
            )


def slice_table(rows: pd.DataFrame, grids: dict, beta: float) -> pd.DataFrame:
    """One row per (speed, step) with lam, n_pulls, lnZ, ess, Wmean, Wvar.

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``speed``, ``step``, ``work``, ``trajname``.
    grids : dict
        ``grids[speed]`` DataFrame with ``step`` and ``r_target_protocol``.
    beta : float
        Inverse temperature, 1/(kJ/mol).

    Returns
    -------
    pandas.DataFrame
        Columns ``speed, step, lam, n_pulls, lnZ, ess, Wmean, Wvar``.
    """
    check_full_support(rows)
    g = rows.groupby(_KEYS, sort=True)["work"]
    codes = g.ngroup().to_numpy()
    a = -beta * rows["work"].to_numpy(float)
    amax = pd.Series(a).groupby(codes).max().to_numpy()
    e = np.exp(a - amax[codes])
    n = np.bincount(codes)
    s1 = np.bincount(codes, e)
    s2 = np.bincount(codes, e * e)
    agg = g.agg(["mean", "var"])
    t = agg.index.to_frame(index=False)
    t["n_pulls"] = n
    t["lnZ"] = amax + np.log(s1) - np.log(n)
    t["ess"] = s1 * s1 / s2
    t["Wmean"] = agg["mean"].to_numpy()
    t["Wvar"] = agg["var"].to_numpy()
    lam = pd.concat([g.assign(speed=s) for s, g in grids.items()], ignore_index=True)
    t = t.merge(lam.rename(columns={"r_target_protocol": "lam"}), on=_KEYS, how="left")
    if t["lam"].isna().any():
        raise ValueError("not supported: protocol grid does not cover every (speed, step)")
    return t[["speed", "step", "lam", "n_pulls", "lnZ", "ess", "Wmean", "Wvar"]]


def f_jarzynski(slices: pd.DataFrame, beta: float) -> np.ndarray:
    """Jarzynski free energy per slice, -lnZ/beta.

    Parameters
    ----------
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    beta : float
        Inverse temperature, 1/(kJ/mol).

    Returns
    -------
    numpy.ndarray
        F per slice, kJ/mol.
    """
    return -slices["lnZ"].to_numpy() / beta


def f_cumulant(slices: pd.DataFrame, beta: float) -> np.ndarray:
    """Second-order cumulant free energy per slice, Wmean - beta Wvar / 2.

    Parameters
    ----------
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    beta : float
        Inverse temperature, 1/(kJ/mol).

    Returns
    -------
    numpy.ndarray
        F per slice, kJ/mol.
    """
    if (slices["n_pulls"] < 2).any():
        raise ValueError("not supported: cumulant F needs at least 2 pulls in every slice")
    return (slices["Wmean"] - 0.5 * beta * slices["Wvar"]).to_numpy(float)


def f_force_v0(slices: pd.DataFrame, realized_speed_map: dict) -> np.ndarray:
    """Mean work extrapolated to v -> 0 per step, mapped to every slice.

    Parameters
    ----------
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    realized_speed_map : dict
        Nominal speed -> realized speed used as the regressor.

    Returns
    -------
    numpy.ndarray
        F per slice, kJ/mol; NaN for steps trimmed by the extrapolation.
    """
    # Lazy: Estimators is the heavier module and may import this one.
    from .Estimators import extrapolate_to_v0

    if slices["speed"].nunique() < 2:
        raise ValueError("not supported: force F needs at least 2 speeds")
    missing = sorted(set(slices["speed"].astype(float)) - set(map(float, realized_speed_map)))
    if missing:
        raise ValueError(f"not supported: realized_speed_map lacks speeds {missing}")
    tab = pd.DataFrame({"step": slices["step"], "r_coord": slices["lam"], "speed": slices["speed"],
                        "estimator": "force", "dG": slices["Wmean"]})
    v0 = extrapolate_to_v0(tab, param="dG", min_speeds=2, realized_speed_map=realized_speed_map)
    if v0.empty:
        raise ValueError("not supported: force F has no v->0 intercept at any step")
    return slices["step"].map(v0.set_index("step")["dG"]).to_numpy(float)


def slice_weights(slices: pd.DataFrame, mode: str) -> np.ndarray:
    """Slice weights c: Kish ESS or uniform.

    Parameters
    ----------
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    mode : {"ess", "uniform"}
        Weighting scheme.

    Returns
    -------
    numpy.ndarray
        c per slice.
    """
    if mode == "ess":
        return slices["ess"].to_numpy(float)
    if mode == "uniform":
        return np.ones(len(slices))
    raise ValueError(f"not supported: slice weight mode {mode!r}")


def log_denominator(r, slices, F, c, k, beta, dr=None, max_block=2_000_000):
    """ln sum_tau c_tau exp(-beta[u(r, lam_tau) - F_tau]) at each r.

    Evaluated on a uniform grid over [min r, max r] and linearly interpolated.

    Parameters
    ----------
    r : array_like
        Positions, nm.
    slices : pandas.DataFrame
        Output of :func:`slice_table` (uses ``lam``).
    F : array_like
        Free energy per slice, kJ/mol.
    c : array_like
        Slice weights.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    dr : float, optional
        Grid spacing, nm; default sqrt(1/(beta k))/10.
    max_block : int, optional
        Maximum slices-by-grid block size per logsumexp call.

    Returns
    -------
    numpy.ndarray
        Log denominator at each r.
    """
    r = np.asarray(r, float)
    lo, hi = r.min(), r.max()
    dr = np.sqrt(1.0 / (beta * k)) / 10.0 if dr is None else dr
    # Tolerance keeps float noise from adding a node when the span is a multiple of dr.
    grid = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / dr - 1e-9)) + 1))
    lam = slices["lam"].to_numpy(float)
    with np.errstate(divide="ignore"):
        a = np.log(np.asarray(c, float)) + beta * np.asarray(F, float)
    chunk = max(1, max_block // len(grid))
    acc = np.full(len(grid), -np.inf)
    for i in range(0, len(lam), chunk):
        e = a[i:i + chunk, None] - 0.5 * beta * k * (grid[None, :] - lam[i:i + chunk, None]) ** 2
        acc = np.logaddexp(acc, logsumexp(e, axis=0))
    return np.interp(r, grid, acc)


def _slice_pos(rows, slices) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(slices[_KEYS])
    pos = idx.get_indexer(pd.MultiIndex.from_frame(rows[_KEYS]))
    if (pos < 0).any():
        raise ValueError("not supported: rows outside the slice table")
    return pos


def _log_p_slice(rows, slices, pos, beta) -> np.ndarray:
    # ln of e^{-beta W} / sum_slice e^{-beta W}: each slice's p_t sums to 1 whatever its n_pulls.
    lnz = slices["lnZ"].to_numpy()[pos] + np.log(slices["n_pulls"].to_numpy(float)[pos])
    return -beta * rows["work"].to_numpy() - lnz


def hs_log_weights(rows, slices, F, c, k, beta, dr=None) -> np.ndarray:
    """Per-row ln c_s - beta W - ln(n_s Z_s) - ln denominator(r_after).

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``speed, step, work, r_after``.
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    F, c : array_like
        Free energy and weight per slice, aligned with ``slices``.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    dr : float, optional
        Denominator grid spacing, nm; see :func:`log_denominator`.

    Returns
    -------
    numpy.ndarray
        Log weight per row (unnormalised).
    """
    pos = _slice_pos(rows, slices)
    with np.errstate(divide="ignore"):
        lnc = np.log(np.asarray(c, float))[pos]
    den = log_denominator(rows["r_after"].to_numpy(), slices, F, c, k, beta, dr=dr)
    return lnc + _log_p_slice(rows, slices, pos, beta) - den


def g0_profile(r, logw, pull, beta, bin_nm=0.02) -> pd.DataFrame:
    """Binned unbiased free energy from log weights, minimum set to 0.

    Parameters
    ----------
    r : array_like
        Row positions, nm.
    logw : array_like
        Log weight per row.
    pull : array_like
        Pull identifier per row.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    bin_nm : float, optional
        Bin width, nm.

    Returns
    -------
    pandas.DataFrame
        ``r_bin, G0, n_rows, ess_pulls`` per occupied bin.
    """
    r = np.asarray(r, float)
    w = np.exp(np.asarray(logw, float) - np.max(logw))
    b = np.floor(r / bin_nm).astype(int)
    df = pd.DataFrame({"b": b, "w": w, "pull": np.asarray(pull)})
    per = df.groupby(["b", "pull"])["w"].sum()
    s1 = per.groupby("b").sum()
    s2 = (per ** 2).groupby("b").sum()
    n_rows = df.groupby("b").size()
    out = pd.DataFrame({
        "r_bin": (s1.index.to_numpy() + 0.5) * bin_nm,
        "weight": s1.to_numpy(),
        "n_rows": n_rows.loc[s1.index].to_numpy(),
        "ess_pulls": (s1 ** 2 / s2).to_numpy(),
    })
    with np.errstate(divide="ignore"):
        g = -np.log(out["weight"].to_numpy() / bin_nm) / beta
    out["G0"] = g - g[np.isfinite(g)].min()
    return out[["r_bin", "G0", "n_rows", "ess_pulls"]]


def _wham_setup(rows, slices, c, k, beta, bin_nm):
    # Numerator sum_t c_t p_t(b) is fixed; only the occupied bins enter the iteration.
    pos = _slice_pos(rows, slices)
    r = rows["r_after"].to_numpy(float)
    i0 = int(np.floor(r.min() / bin_nm))
    ib = np.floor(r / bin_nm).astype(int) - i0
    nb = ib.max() + 1
    edges = (i0 + np.arange(nb + 1)) * bin_nm
    w = np.asarray(c, float)[pos] * np.exp(_log_p_slice(rows, slices, pos, beta))
    num = np.bincount(ib, weights=w, minlength=nb)
    occ = num > 0
    centres = (i0 + np.flatnonzero(occ) + 0.5) * bin_nm
    lam = slices["lam"].to_numpy(float)
    L = -0.5 * beta * k * (centres[None, :] - lam[:, None]) ** 2
    # Column-shifted Boltzmann factors: the iteration becomes two matmuls per step.
    Lmax = L.max(axis=0)
    E = np.exp(L - Lmax[None, :])
    with np.errstate(divide="ignore"):
        lnc = np.log(np.asarray(c, float))
    return edges, occ, np.log(num[occ]), (E, Lmax), lnc


def _log_rho(lnum, EL, lnc, F, beta):
    E, Lmax = EL
    a = lnc + beta * np.asarray(F, float)
    m = a.max()
    with np.errstate(divide="ignore"):
        lr = lnum - (m + Lmax + np.log(np.exp(a - m) @ E))
    if not np.isfinite(lr).all():
        raise ValueError("not supported: WHAM denominator underflows at an occupied bin")
    return lr


def _wham_F(lr, EL, beta):
    E, Lmax = EL
    q = lr + Lmax
    m = q.max()
    with np.errstate(divide="ignore"):
        return -(m + np.log(E @ np.exp(q - m))) / beta


def wham_density(rows, slices, F, c, k, beta, bin_nm=0.002):
    """One binned WHAM density step at fixed F.

    rho(b) = sum_t c_t p_t(b) / sum_t c_t exp(-beta[u_t(b) - F_t]), with p_t the
    work-weighted histogram of ``r_after`` in slice t (sums to 1) and u_t at bin centres.

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``speed, step, work, r_after``.
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    F, c : array_like
        Free energy and weight per slice, aligned with ``slices``.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    bin_nm : float, optional
        Bin width, nm.

    Returns
    -------
    edges : numpy.ndarray
        Bin edges, nm.
    rho : numpy.ndarray
        Bin probabilities, summing to 1.
    """
    edges, occ, lnum, EL, lnc = _wham_setup(rows, slices, c, k, beta, bin_nm)
    lr = _log_rho(lnum, EL, lnc, F, beta)
    rho = np.zeros(len(occ))
    rho[occ] = np.exp(lr - logsumexp(lr))
    return edges, rho


def f_self_consistent(rows, slices, c, k, beta, bin_nm=0.002, stride=1, tol=1e-6, max_iter=200000):
    """Slice free energies solved by binned WHAM iteration.

    Starts from :func:`f_jarzynski`; F of the first slice is 0.

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``speed, step, work, r_after``.
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    c : array_like
        Slice weights, aligned with ``slices``.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    bin_nm : float, optional
        Bin width, nm.
    stride : int, optional
        Solve on every stride-th step per speed (plus the last); interpolate F in lam elsewhere.
    tol : float, optional
        Convergence threshold on max |dF|, kJ/mol.
    max_iter : int, optional
        Maximum iterations.

    Returns
    -------
    numpy.ndarray
        F per slice, kJ/mol.
    """
    if int(stride) != stride or stride < 1:
        raise ValueError(f"not supported: stride {stride!r}; use a positive integer")
    slices = slices.reset_index(drop=True)
    # Stride interpolation in lam assumes lam increases with step within a speed.
    inc = slices.sort_values(_KEYS).groupby("speed")["lam"].apply(lambda x: bool(np.all(np.diff(x) > 0)))
    if not inc.all():
        raise ValueError("not supported: lam must increase strictly with step within each speed")
    step0 = slices.groupby("speed")["step"].transform("min")
    last = slices.groupby("speed")["step"].transform("max")
    sel = (((slices["step"] - step0) % stride) == 0) | (slices["step"] == last)
    sub = slices[sel].reset_index(drop=True)
    rsub = rows.merge(sub[_KEYS], on=_KEYS)
    csub = np.asarray(c, float)[sel.to_numpy()]
    _, _, lnum, EL, lnc = _wham_setup(rsub, sub, csub, k, beta, bin_nm)
    F = f_jarzynski(sub, beta)
    F = F - F[0]
    for _ in range(max_iter):
        lr = _log_rho(lnum, EL, lnc, F, beta)
        Fn = _wham_F(lr, EL, beta)
        if not np.isfinite(Fn).all():
            raise ValueError("not supported: WHAM slice with no overlap with any occupied bin")
        Fn -= Fn[0]
        done = np.max(np.abs(Fn - F)) < tol
        F = Fn
        if done:
            break
    else:
        raise ValueError(f"not supported: WHAM did not converge to {tol} kJ/mol in {max_iter} iterations")
    out = np.empty(len(slices))
    for speed, idx in slices.groupby("speed").groups.items():
        m = (sub["speed"] == speed).to_numpy()
        out[idx] = np.interp(slices.loc[idx, "lam"], sub.loc[m, "lam"], F[m])
    return out


def _source_F(rows, slices, c, k, beta, f_source, realized_speed_map, wham_stride,
              wham_max_iter=200000):
    if f_source == "jarzynski":
        return f_jarzynski(slices, beta)
    if f_source == "cumulant":
        return f_cumulant(slices, beta)
    if f_source == "force":
        if realized_speed_map is None:
            raise ValueError("not supported: force F needs realized_speed_map")
        return f_force_v0(slices, realized_speed_map)
    if f_source == "self_consistent":
        return f_self_consistent(rows, slices, c, k, beta, stride=wham_stride, max_iter=wham_max_iter)
    raise ValueError(f"not supported: F source {f_source!r}; use one of {F_SOURCES}")


def estimate(rows, grids, k, beta, f_source, c_mode="ess", bin_nm=0.02,
             realized_speed_map=None, wham_stride=1, wham_max_iter=200000):
    """HS unbiased profile for one F source.

    Slices with NaN F are dropped from the denominator and their rows get log weight -inf.

    Parameters
    ----------
    rows : pandas.DataFrame
        Full-support conditioned rows.
    grids : dict
        ``grids[speed]`` DataFrame with ``step`` and ``r_target_protocol``.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).
    f_source : {"jarzynski", "cumulant", "force", "self_consistent"}
        Slice free-energy source.
    c_mode : {"ess", "uniform"}, optional
        Slice weighting.
    bin_nm : float, optional
        Profile bin width, nm.
    realized_speed_map : dict, optional
        Nominal -> realized speed; required for ``force``.
    wham_stride : int, optional
        Step stride for ``self_consistent``.
    wham_max_iter : int, optional
        Iteration cap for ``self_consistent``.

    Returns
    -------
    g0 : pandas.DataFrame
        ``r_bin, G0, n_rows, ess_pulls``.
    logw : numpy.ndarray
        Log weight per input row.
    F : numpy.ndarray
        F per slice (rows of :func:`slice_table`), kJ/mol.
    """
    rows = rows.reset_index(drop=True)
    slices = slice_table(rows, grids, beta)
    c = slice_weights(slices, c_mode)
    F = _source_F(rows, slices, c, k, beta, f_source, realized_speed_map, wham_stride, wham_max_iter)
    keep = np.isfinite(F)
    if not keep.any():
        raise ValueError("not supported: no slice has a finite F")
    pos = _slice_pos(rows, slices)
    row_ok = keep[pos]
    sub = rows[row_ok]
    sl = slices[keep].reset_index(drop=True)
    lw = hs_log_weights(sub, sl, F[keep], c[keep], k, beta)
    logw = np.full(len(rows), -np.inf)
    logw[row_ok] = lw
    g0 = g0_profile(sub["r_after"].to_numpy(), lw, sub["trajname"].to_numpy(), beta, bin_nm)
    return g0, logw, F


def bootstrap_g0(rows, grids, k, beta, f_source, n_boot=200, seed=0, point=None, **estimate_kw):
    """Point-estimate profile with a pull-resampling bootstrap sd.

    Whole pulls are resampled with replacement within each speed; F is re-solved per resample.

    Parameters
    ----------
    rows, grids, k, beta, f_source
        As in :func:`estimate`.
    n_boot : int, optional
        Resamples; 0 returns the point estimate with NaN ``G0_sd``.
    seed : int, optional
        RNG seed.
    point : pandas.DataFrame, optional
        Profile already returned by :func:`estimate` for these inputs; computed when omitted.
    **estimate_kw
        Passed to :func:`estimate`.

    Returns
    -------
    pandas.DataFrame
        Point-estimate ``r_bin, G0, n_rows, ess_pulls`` plus ``G0_sd`` per ``r_bin``.
    """
    if point is None:
        point, _, _ = estimate(rows, grids, k, beta, f_source, **estimate_kw)
    point = point.copy()
    if n_boot == 0:
        point["G0_sd"] = np.nan
        return point
    rng = np.random.default_rng(seed)
    pulls = {s: [g for _, g in sp.groupby("trajname", sort=False)] for s, sp in rows.groupby("speed")}
    ref = point.set_index("r_bin")["G0"]
    prof = []
    for b in range(n_boot):
        parts = []
        for speed, gs in pulls.items():
            for j, i in enumerate(rng.integers(0, len(gs), len(gs))):
                parts.append(gs[i].assign(trajname=f"{gs[i]['trajname'].iloc[0]}#b{j}"))
        g, _, _ = estimate(pd.concat(parts, ignore_index=True), grids, k, beta, f_source, **estimate_kw)
        s = g.set_index("r_bin")["G0"].reindex(ref.index)
        ok = np.isfinite(s.to_numpy()) & np.isfinite(ref.to_numpy())
        if ok.any():
            s = s - (s - ref)[ok].mean()
        prof.append(s.to_numpy())
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(prof).sum(axis=0)
        sd = np.nanstd(np.array(prof), axis=0, ddof=1)
    point["G0_sd"] = np.where(cnt >= 2, sd, np.nan)
    return point


def slice_diagnostics(rows, slices, F_by_source, k, beta):
    """Per-slice F by source, source spread and spring-lag statistics.

    Parameters
    ----------
    rows : pandas.DataFrame
        Conditioned rows with ``speed, step, r_after``.
    slices : pandas.DataFrame
        Output of :func:`slice_table`.
    F_by_source : dict
        Source name -> F per slice, aligned with ``slices``.
    k : float
        Spring constant, kJ/mol/nm^2.
    beta : float
        Inverse temperature, 1/(kJ/mol).

    Returns
    -------
    pandas.DataFrame
        ``speed, step, lam, n_pulls, ess, F_<source>..., F_spread, lag_mean, lag_eq,
        lag_sd, lag_sd_eq``.
    """
    d = slices[["speed", "step", "lam", "n_pulls", "ess"]].reset_index(drop=True)
    for name, F in F_by_source.items():
        d[f"F_{name}"] = np.asarray(F, float)
    pos = _slice_pos(rows, slices)
    lag = slices["lam"].to_numpy(float)[pos] - rows["r_after"].to_numpy(float)
    d["lag_mean"] = np.bincount(pos, lag, len(d)) / np.bincount(pos, minlength=len(d))
    d["lag_sd"] = np.sqrt(np.maximum(
        np.bincount(pos, lag ** 2, len(d)) / np.bincount(pos, minlength=len(d)) - d["lag_mean"] ** 2, 0))
    d["lag_sd_eq"] = np.sqrt(1.0 / (beta * k))
    Fj = f_jarzynski(slices.reset_index(drop=True), beta)
    d["lag_eq"] = np.nan
    cols = [f"F_{n}" for n in F_by_source]
    spread = np.full(len(d), np.nan)
    for _, idx in d.sort_values(_KEYS).groupby("speed").groups.items():
        idx = np.asarray(idx)
        if len(idx) > 1:
            d.loc[idx, "lag_eq"] = np.gradient(Fj[idx], d["lam"].to_numpy()[idx]) / k
        if cols:
            A = d.loc[idx, cols].to_numpy()
            fin = np.isfinite(A).all(axis=1)
            if fin.any():
                A = A - A[np.argmax(fin)]
                spread[idx] = A.max(axis=1) - A.min(axis=1)
    d["F_spread"] = spread
    return d


@dataclass
class HSResult:
    """Output of :meth:`HummerSzaboEstimator.fit`.

    Attributes
    ----------
    g0 : pandas.DataFrame
        ``f_source, scope, path, speed, r_bin, G0, G0_sd, n_rows, ess_pulls``. ``scope`` is
        ``"all"`` (all pulls jointly), ``"speed"`` (one speed, all paths) or ``"path"``
        (one path, all speeds); ``path``/``speed`` are NaN where not applicable.
        ``force`` is run only for groups with at least 2 speeds.
    row_weights : pandas.DataFrame
        ``trajname, speed, step, r, path, f_source, logw, logw_path``. ``logw`` is from scope
        ``"all"`` with sum(exp) = 1 per source; ``logw_path`` is from the row's path with
        sum(exp) = 1 per (path, source). -inf for rows without support or F in that scope,
        NaN where the source was not run for the path.
    slices : pandas.DataFrame
        :func:`slice_diagnostics` for scope ``"all"``.
    excluded : dict
        ``n_rows_unsupported``, ``n_rows_no_F`` (per source), ``n_pulls_dropped``,
        ``lam0_spread_nm``, ``lam0_sd_nm``, ``skipped_sources``, ``bootstrap_failed``,
        ``skipped_groups``. ``skipped_sources[src][scope]`` and ``bootstrap_failed[src][scope]``
        hold the error message; scope labels are ``"all"``, ``"speed=<v>"``, ``"path=<p>"``.
        ``n_pulls_dropped`` is loaded logs minus pulls in the joint estimate: pulls
        :class:`SMDData` dropped as incomplete or in excluded speeds, plus any removed upstream
        of HS. ``lam0_*`` use per-pull lambda0 = first ``r_target`` minus its speed's median dx;
        ``lam0_spread_nm`` is the range of the per-speed medians, ``lam0_sd_nm`` the largest
        per-speed sd.
    """

    g0: pd.DataFrame
    row_weights: pd.DataFrame
    slices: pd.DataFrame
    excluded: dict


_ROW_COLS = ["trajname", "speed", "step", "work", "r_after", "path"]


def _trim(rows: pd.DataFrame) -> pd.DataFrame:
    from .Estimators import trim_results_by_n_samples_support

    out = trim_results_by_n_samples_support(rows, min_samples=2, min_support_ratio=1.0,
                                            add_support_columns=False)
    return out.drop(columns="n_samples", errors="ignore").reset_index(drop=True)


def _check_step_size(raw: pd.DataFrame, tol: float = 1e-6) -> None:
    r = raw.sort_values(["trajname", "step"])
    dx = r.groupby("trajname")["r_target"].diff()
    med = dx.groupby(r["speed"]).transform("median")
    bad = (dx - med).abs() > tol
    if bad.any():
        names = sorted(set(r.loc[bad, "trajname"]))
        raise ValueError(f"not supported: {len(names)} pulls (first {names[0]}) have a per-move dx "
                         f"differing from their speed's median by > {tol} nm")


def _speed_dx(raw: pd.DataFrame) -> pd.Series:
    # Median per-move target increment of each speed.
    r = raw.sort_values(["trajname", "step"])
    return r.groupby("trajname")["r_target"].diff().groupby(r["speed"]).median()


def _lam0_stats(raw: pd.DataFrame, pulls) -> tuple[float, float]:
    # Per-pull protocol lambda0: first logged target minus its speed's move.
    # Returns (spread of per-speed medians, largest per-speed sd).
    r = raw[raw["trajname"].isin(pulls)].sort_values(["trajname", "step"])
    first = r.groupby("trajname")[["speed", "r_target"]].first()
    lam0 = first["r_target"] - first["speed"].map(_speed_dx(r))
    by = lam0.groupby(first["speed"])
    med = by.median()
    return float(med.max() - med.min()), float(by.std().max())


class HummerSzaboEstimator:
    """Hummer-Szabo unbiased profile G0(r) from the frames of an :class:`SMDData`.

    Parameters
    ----------
    f_sources : tuple of str, optional
        Slice free-energy sources, a subset of :data:`F_SOURCES`.
    slice_weights : {"ess", "uniform"}, optional
        Slice weighting c.
    bin_nm : float, optional
        Profile bin width, nm.
    n_boot : int, optional
        Pull-bootstrap resamples per profile; 0 gives NaN ``G0_sd``.
    seed : int, optional
        Bootstrap RNG seed.
    wham_stride : int, optional
        Step stride for the ``self_consistent`` source.
    wham_max_iter : int, optional
        Iteration cap for the ``self_consistent`` source.

    A source whose point estimate raises ``ValueError`` in a scope has no output for that scope
    and is recorded in ``excluded["skipped_sources"]``; failing in scope ``"all"`` drops it
    everywhere, and :meth:`fit` raises if every source fails there. A failed bootstrap keeps
    the point estimate with NaN ``G0_sd`` and is recorded in ``excluded["bootstrap_failed"]``.
    """

    name = "hummer_szabo"

    def __init__(self, f_sources=F_SOURCES, slice_weights="ess", bin_nm=0.02, n_boot=200, seed=0,
                 wham_stride=1, wham_max_iter=200000):
        unknown = [s for s in f_sources if s not in F_SOURCES]
        if unknown:
            raise ValueError(f"not supported: F sources {unknown}; use a subset of {F_SOURCES}")
        self.f_sources = tuple(f_sources)
        self.slice_weights = slice_weights
        self.bin_nm = bin_nm
        self.n_boot = n_boot
        self.seed = seed
        self.wham_stride = wham_stride
        self.wham_max_iter = wham_max_iter

    def _run(self, rows, grids, k, beta, rsm, sources, skipped, boot_failed, scope):
        # Per source: bootstrap profile, log weights aligned with rows, F per slice.
        # ValueErrors are recorded per scope: a failed point estimate leaves the source out,
        # a failed bootstrap keeps the point estimate with NaN G0_sd.
        kw = dict(c_mode=self.slice_weights, bin_nm=self.bin_nm, realized_speed_map=rsm,
                  wham_stride=self.wham_stride, wham_max_iter=self.wham_max_iter)
        out = {}
        for src in sources:
            if "all" in skipped.get(src, {}):
                continue
            try:
                point, logw, F = estimate(rows, grids, k, beta, src, **kw)
            except ValueError as e:
                logger.warning(f"Skipping F source {src!r} in scope {scope}: point estimate failed: {e}")
                skipped.setdefault(src, {})[scope] = str(e)
                continue
            try:
                g0 = bootstrap_g0(rows, grids, k, beta, src, n_boot=self.n_boot, seed=self.seed,
                                  point=point, **kw)
            except ValueError as e:
                logger.warning(f"F source {src!r} in scope {scope}: bootstrap failed, keeping the "
                               f"point estimate with NaN G0_sd: {e}")
                boot_failed.setdefault(src, {})[scope] = str(e)
                g0 = point.assign(G0_sd=np.nan)
            out[src] = (g0, logw - logsumexp(logw), F)
        return out

    def fit(self, smd_data) -> HSResult:
        """Estimate G0 for every F source and scope.

        Pulls whose spring starts at their own position bias G0 as ``lam0_sd_nm`` grows.

        Parameters
        ----------
        smd_data : SMDData
            Loaded forward pulls; ``raw_data`` and ``results`` are not modified.

        Returns
        -------
        HSResult
            Profiles, per-row weights, slice diagnostics and exclusion counts.
        """
        from .Estimators import recover_spring_constant

        if smd_data.pulling_direction != "forward":
            raise ValueError(f"not supported: {smd_data.pulling_direction!r} pulls; HS needs forward pulls")
        mode = getattr(smd_data, "work_mode", None)
        col = getattr(smd_data, "protocol_work_column", "dW_protocol")
        if not (mode == "protocol" or (mode == "auto" and col in smd_data.raw_data.columns)):
            raise ValueError(f"not supported: work_mode {mode!r}; HS needs protocol work "
                             f"(work_mode 'protocol', or 'auto' with a {col!r} column)")
        raw = smd_data.raw_data.reset_index(drop=True)
        _check_step_size(raw)
        k = recover_spring_constant(raw, logged_k=smd_data.spring_constant)
        beta, grids, rsm = smd_data.beta, smd_data.protocol_grids, smd_data.realized_speed_map
        rows_in = raw[_ROW_COLS].copy()

        # One path label so support is trimmed over all pulls of each speed.
        joint = _trim(rows_in.assign(path=0)).drop(columns="path")
        if joint.empty:
            raise ValueError("not supported: no (speed, step) has rows from at least 2 pulls of its speed")
        joint = joint.merge(rows_in[["trajname", "step", "path"]], on=["trajname", "step"], how="left")
        speeds = sorted(joint["speed"].unique())
        skipped = {}
        sources = list(self.f_sources)
        if "force" in sources and len(speeds) < 2:
            logger.warning("Skipping F source 'force': it needs at least 2 speeds")
            skipped["force"] = {"all": "single speed"}
            sources.remove("force")
        if "force" in sources:
            dx = _speed_dx(raw)
            if dx.max() - dx.min() > 1e-6:
                raise ValueError(f"not supported: force F joins speeds on step, but the per-speed median "
                                 f"dx differ ({dx.min():.6g}-{dx.max():.6g} nm); drop 'force'")
        if not sources:
            raise ValueError("not supported: no F source can run on this dataset")

        g0_parts, skipped_groups, boot_failed = [], [], {}

        def _collect(res, scope, path=np.nan, speed=np.nan):
            for src, (g0, _, _) in res.items():
                g0_parts.append(g0.assign(f_source=src, scope=scope, path=path, speed=speed))

        res_all = self._run(joint, grids, k, beta, rsm, sources, skipped, boot_failed, "all")
        if not res_all:
            raise ValueError("not supported: every F source failed: "
                             + "; ".join(f"{s}: {skipped[s]['all']}" for s in sources))
        _collect(res_all, "all")

        # The joint trim already holds every pull of a speed, so it is that speed's own trim.
        for speed in speeds:
            sub = joint[joint["speed"] == speed].reset_index(drop=True)
            _collect(self._run(sub, grids, k, beta, rsm, [s for s in sources if s != "force"],
                               skipped, boot_failed, f"speed={speed}"), "speed", speed=speed)

        path_rw = []
        for path, g in rows_in.groupby("path"):
            sub = _trim(g)
            if sub.empty:
                skipped_groups.append(("path", path))
                continue
            srcs = [s for s in sources if s != "force" or sub["speed"].nunique() > 1]
            res = self._run(sub, grids, k, beta, rsm, srcs, skipped, boot_failed, f"path={path}")
            _collect(res, "path", path=path)
            for src, (_, lw, _) in res.items():
                path_rw.append(sub[["trajname", "step"]].assign(path=path, f_source=src, logw_path=lw))

        rw = pd.concat([joint[["trajname", "speed", "step", "r_after", "path"]].assign(f_source=src, logw=lw)
                        for src, (_, lw, _) in res_all.items()], ignore_index=True)
        pw = pd.concat(path_rw, ignore_index=True) if path_rw else pd.DataFrame(
            columns=["trajname", "step", "path", "f_source", "logw_path"])
        keys = ["trajname", "step", "path", "f_source"]
        rw = rw.merge(pw, on=keys, how="outer")
        # Path-only rows (past the joint cutoff) take speed and r from the input rows.
        rw = rw.drop(columns=["speed", "r_after"]).merge(
            rows_in[["trajname", "step", "speed", "r_after"]], on=["trajname", "step"], how="left")
        rw["logw"] = rw["logw"].fillna(-np.inf)
        ran = rw.groupby(["path", "f_source"])["logw_path"].transform(lambda x: x.notna().any())
        rw.loc[ran & rw["logw_path"].isna(), "logw_path"] = -np.inf
        rw = rw.rename(columns={"r_after": "r"})[
            ["trajname", "speed", "step", "r", "path", "f_source", "logw", "logw_path"]]
        rw = rw.sort_values(["f_source", "speed", "trajname", "step"]).reset_index(drop=True)

        g0 = pd.concat(g0_parts, ignore_index=True)[
            ["f_source", "scope", "path", "speed", "r_bin", "G0", "G0_sd", "n_rows", "ess_pulls"]]
        sl = slice_table(joint, grids, beta)
        slices = slice_diagnostics(joint, sl, {s: F for s, (_, _, F) in res_all.items()}, k, beta)

        lam0_spread, lam0_sd = _lam0_stats(raw, joint["trajname"].unique())
        # Loaded logs, including those SMDData dropped as incomplete.
        n_loaded = len(smd_data.log_metadata)
        excluded = {
            "n_rows_unsupported": int(len(rows_in) - len(joint)),
            "n_rows_no_F": {s: int(np.isneginf(lw).sum()) for s, (_, lw, _) in res_all.items()},
            "n_pulls_dropped": int(n_loaded - joint["trajname"].nunique()),
            "lam0_spread_nm": lam0_spread,
            "lam0_sd_nm": lam0_sd,
            "skipped_sources": skipped,
            "bootstrap_failed": boot_failed,
            "skipped_groups": skipped_groups,
        }
        return HSResult(g0=g0, row_weights=rw, slices=slices, excluded=excluded)


def frame_steps(n_rows: int, n_frames: int, save_freq: int) -> np.ndarray:
    """Log step each DCD frame corresponds to.

    Frame j is the state at the end of row ``(j + 1) * save_freq - 1``.

    Parameters
    ----------
    n_rows : int
        Rows in the pull's log.
    n_frames : int
        Frames in its DCD.
    save_freq : int
        Moves per saved frame.

    Returns
    -------
    numpy.ndarray
        Step per frame.
    """
    if n_frames != n_rows // save_freq:
        raise ValueError(f"not supported: {n_frames} frames for {n_rows} rows at save_freq={save_freq}; "
                         f"expected {n_rows // save_freq}")
    return (np.arange(n_frames) + 1) * save_freq - 1


def hs_frame_weights(dat_files: list[str], temperature: float = 300.0, f_source: str = "jarzynski",
                     save_freq: int = 5, slice_weights: str = "ess", bin_nm: float = 0.02) -> pd.DataFrame:
    """Hummer-Szabo weight of each saved DCD frame, all pulls jointly (one path).

    Parameters
    ----------
    dat_files : list of str
        sMD logs; each ``_aligned.dcd`` sits at the path :class:`SMDData` derives.
    temperature : float, optional
        K.
    f_source : str, optional
        Slice free-energy source.
    save_freq : int, optional
        Moves per saved frame.
    slice_weights : {"ess", "uniform"}, optional
        Slice weighting c.
    bin_nm : float, optional
        Profile bin width, nm.

    Returns
    -------
    pandas.DataFrame
        ``trajname, frame, step, logw`` with sum(exp(logw)) = 1 over the frames kept. The
        estimate uses only the frame-step rows and slices (numerator and denominator). Frames
        without a weighted row are omitted; their count, including every frame of pulls that
        :class:`SMDData` dropped, is ``attrs["n_frames_excluded"]``, and the dropped pulls'
        trajnames are ``attrs["pulls_dropped"]``.
    """
    import copy
    import os

    from MDAnalysis.lib.formats.libdcd import DCDFile

    from .SMDData import SMDData

    smd = SMDData(dat_files, temperature=temperature)
    smd.raw_data["path"] = 0
    raw = smd.raw_data
    for name, st in raw.groupby("trajname", sort=False)["step"]:
        if not np.array_equal(st.to_numpy(), np.arange(len(st))):
            raise ValueError(f"not supported: {name} log steps are not 0..n_rows-1 in order")
    # Frame j is the state after step (j + 1) * save_freq - 1; HS on those slices only.
    view = copy.copy(smd)
    view.raw_data = raw[(raw["step"] + 1) % save_freq == 0].reset_index(drop=True)
    res = HummerSzaboEstimator(f_sources=(f_source,), n_boot=0, slice_weights=slice_weights,
                               bin_nm=bin_nm).fit(view)
    rw = res.row_weights
    n_rows = raw.groupby("trajname").size()
    parts, n_excl, dropped = [], 0, []
    for log, dcd in zip(smd.log_files, smd.traj_files):
        name = os.path.basename(log)[:-4]
        with DCDFile(dcd) as f:
            n_frames = f.n_frames
        if name not in n_rows.index:
            dropped.append(name)
            n_excl += n_frames
            continue
        steps = frame_steps(int(n_rows[name]), n_frames, save_freq)
        w = rw[rw["trajname"] == name].set_index("step")["logw"]
        lw = w.reindex(steps)
        keep = np.isfinite(lw.to_numpy())
        n_excl += int((~keep).sum())
        parts.append(pd.DataFrame({"trajname": name, "frame": np.arange(n_frames)[keep],
                                   "step": steps[keep], "logw": lw.to_numpy()[keep]}))
    out = pd.concat(parts, ignore_index=True)
    out["logw"] -= logsumexp(out["logw"].to_numpy())
    out.attrs["n_frames_excluded"] = n_excl
    out.attrs["pulls_dropped"] = sorted(dropped)
    return out
