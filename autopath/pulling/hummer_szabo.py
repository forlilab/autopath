"""Hummer-Szabo reweighting of steered-MD frames to the unbiased density."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

_KEYS = ["speed", "step"]


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
    grid = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / dr)) + 1))
    lam = slices["lam"].to_numpy(float)
    with np.errstate(divide="ignore"):
        a = np.log(np.asarray(c, float)) + beta * np.asarray(F, float)
    chunk = max(1, max_block // len(grid))
    acc = np.full(len(grid), -np.inf)
    for i in range(0, len(lam), chunk):
        e = a[i:i + chunk, None] - 0.5 * beta * k * (grid[None, :] - lam[i:i + chunk, None]) ** 2
        acc = np.logaddexp(acc, logsumexp(e, axis=0))
    return np.interp(r, grid, acc)


def hs_log_weights(rows, slices, F, c, k, beta) -> np.ndarray:
    """Per-row ln c_s - beta W - lnZ_s - ln denominator(r_after).

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

    Returns
    -------
    numpy.ndarray
        Log weight per row (unnormalised).
    """
    idx = pd.MultiIndex.from_frame(slices[_KEYS])
    pos = idx.get_indexer(pd.MultiIndex.from_frame(rows[_KEYS]))
    if (pos < 0).any():
        raise ValueError("not supported: rows outside the slice table")
    with np.errstate(divide="ignore"):
        lnc = np.log(np.asarray(c, float))[pos]
    lnz = slices["lnZ"].to_numpy()[pos]
    den = log_denominator(rows["r_after"].to_numpy(), slices, F, c, k, beta)
    return lnc - beta * rows["work"].to_numpy() - lnz - den


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
