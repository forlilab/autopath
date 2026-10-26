"""Hummer-Szabo reweighting of steered-MD frames to the unbiased density."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

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
    out = []
    for (speed, step), w in rows.groupby(_KEYS, sort=True)["work"]:
        a = -beta * w.to_numpy()
        n = len(a)
        out.append((speed, step, n, logsumexp(a) - np.log(n),
                    np.exp(2 * logsumexp(a) - logsumexp(2 * a)),
                    w.mean(), w.var(ddof=1) if n > 1 else np.nan))
    t = pd.DataFrame(out, columns=["speed", "step", "n_pulls", "lnZ", "ess", "Wmean", "Wvar"])
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


def f_self_consistent(rows, slices, c, k, beta, bin_nm=0.002, stride=1, tol=1e-6, max_iter=10000):
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
