# autopath/metadynamics/learned/sw.py
"""s_w: the TS-separating HLDA component, refitted on z-residualised features when it tracks z."""
import numpy as np

from .reduce import HLDA
from .select import fisher

SW_CORR_MAX = 0.9
N_KNOTS, N_TABLE = 12, 200


def choose_component(P, labels, order):
    """Column of P with the largest TS vs B∪U two-group Fisher ratio.

    Parameters
    ----------
    P : (n, k) array, projections
    labels : (n,) object array; frames outside ``order`` are ignored
    order : [B, TS, U]

    Returns
    -------
    index : int
    ratios : (k,) array
    """
    m = np.isin(labels, order)
    ts = labels[m] == order[1]
    ratios = np.array([fisher(P[m, k], ts) for k in range(P.shape[1])])
    return int(np.argmax(ratios)), ratios


def orient(w, X, labels, order):
    """w signed so that mean(B) < mean(TS)."""
    s = X @ w
    return -w if s[labels == order[0]].mean() > s[labels == order[1]].mean() else w


def fit_z_splines(Y, z, n_knots=N_KNOTS, n_table=N_TABLE):
    """Least-squares cubic B-spline of each column of Y on z, tabulated on a uniform grid.

    Parameters
    ----------
    Y : (n, m) array
    z : (n,) array
    n_knots : int, interior knots at z quantiles
    n_table : int, table points

    Returns
    -------
    z_min, z_max : float
    tables : (m, n_table) array, spline values on linspace(z_min, z_max, n_table)
    """
    from scipy.interpolate import BSpline
    Y, z = np.asarray(Y, float), np.asarray(z, float)
    _check_finite_z(z)
    lo, hi = float(z.min()), float(z.max())
    inner = np.quantile(z, np.linspace(0.0, 1.0, n_knots + 2)[1:-1])
    if not (inner[0] > lo and inner[-1] < hi and np.all(np.diff(inner) > 0)):
        raise ValueError(f"z has too few distinct values for {n_knots} interior spline knots")
    t = np.r_[[lo] * 4, inner, [hi] * 4]
    C = np.linalg.lstsq(BSpline.design_matrix(z, t, 3).toarray(), Y, rcond=None)[0]
    grid = np.linspace(lo, hi, n_table)
    return lo, hi, (BSpline.design_matrix(grid, t, 3).toarray() @ C).T


def z_spline_values(tables, z_min, z_max, z):
    """Natural cubic spline through each table row at clamp(z, z_min, z_max) (as OpenMM's table).

    Returns
    -------
    (len(z), m) array, or (m,) for scalar z
    """
    from scipy.interpolate import CubicSpline
    tables = np.atleast_2d(np.asarray(tables, float))
    x = np.linspace(z_min, z_max, tables.shape[1])
    return CubicSpline(x, tables, axis=1, bc_type="natural")(np.clip(z, z_min, z_max)).T


def _check_finite_z(z):
    if not np.all(np.isfinite(z)):
        raise ValueError("z has non-finite values")


def _corr(s, z, m):
    r = float(np.corrcoef(s[m], z[m])[0, 1])
    if not np.isfinite(r):
        raise ValueError("s_w or z is constant over the labelled frames")
    return r


def _fit(X, labels, order, reducer):
    W = reducer.fit(X, labels, order)
    k, ratios = choose_component(X @ W, labels, order)
    return W, k, ratios, orient(W[:, k], X, labels, order)


def fit_sw(Z, z, labels, order, reducer=None, corr_max=SW_CORR_MAX):
    """s_w = X @ w on the standardised features Z, or on Z - spline(z) when |corr(Z @ w, z)| > corr_max.

    Parameters
    ----------
    Z : (n, m) standardised features
    z : (n,) funnel z per frame
    labels : (n,) object array; "" frames are ignored by the HLDA fit and the correlations
        (the z-spline is fitted on all frames, pooled)
    order : [B, TS, U]
    reducer : object with fit(X, labels, order) -> (m, k); default HLDA(n_components=2)
    corr_max : float

    Returns
    -------
    dict
        weights, component, w, fisher_ts, X of the fit used; plain_weights, plain_component, plain_w,
        plain_fisher_ts, corr_plain of the fit on Z; corr_resid (NaN unless refitted); residualised; z_min, z_max, tables
        (NaN / empty unless residualised).
    """
    reducer = reducer or HLDA(n_components=2)
    _check_finite_z(np.asarray(z, float))
    m = np.isin(labels, order)
    W, k, ratios, w = _fit(Z, labels, order, reducer)
    out = dict(weights=W, component=k, w=w, fisher_ts=ratios, X=Z, plain_weights=W, plain_component=k,
               plain_w=w, plain_fisher_ts=ratios,
               corr_plain=_corr(Z @ w, z, m), corr_resid=np.nan, residualised=False,
               z_min=np.nan, z_max=np.nan, tables=np.zeros((0, 0)))
    if not abs(out["corr_plain"]) > corr_max:
        return out
    lo, hi, tables = fit_z_splines(Z, z)
    R = Z - z_spline_values(tables, lo, hi, z)
    W, k, ratios, w = _fit(R, labels, order, reducer)
    out.update(weights=W, component=k, w=w, fisher_ts=ratios, X=R, corr_resid=_corr(R @ w, z, m),
               residualised=True, z_min=lo, z_max=hi, tables=tables)
    return out
