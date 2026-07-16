import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import statsmodels.formula.api as smf
from scipy import special
from scipy.stats import linregress
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

from autopath.pulling.SMDData import SMDData

import logging
logger = logging.getLogger("autopath.pulling.Estimators")


class BaseEstimator(ABC):
    """
    Base class for all estimators in SMDAnalysis.
    """

    def __init__(self):
        pass

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        """Estimate free energy from raw work values.

        Returns a dict with at least 'Wmean', 'dG', 'Wdiss' keys,
        or None if estimation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, smd_data: SMDData):
        """Fit the estimator to the provided SMD data."""
        pass
    
class JarzynskiEstimator(BaseEstimator):
    
    @property
    def name(self):
        return 'jarzynski'

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        raw_W = np.asarray(raw_W, dtype=float)
        if raw_W.size == 0:
            return None
        Wmean = float(raw_W.mean())
        dG = float(-(1.0 / beta) * (
            special.logsumexp(-beta * raw_W) - np.log(raw_W.size)
        ))
        return {'Wmean': Wmean, 'dG': dG, 'Wdiss': Wmean - dG}

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]

            raw_W = group['work'].astype(float).values
            result = self.estimate_dG(raw_W, smd_data.beta)
            if result is None:
                continue

            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                **result,
            })
        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data
    
class CumulantEstimator(BaseEstimator):
    
    @property
    def name(self):
        return 'cumulant'

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        raw_W = np.asarray(raw_W, dtype=float)
        if raw_W.size == 0:
            return None
        Wmean = float(raw_W.mean())
        # ddof=1 variance is undefined for a single sample → NaN (propagates to dG);
        # guard to avoid a spurious "Degrees of freedom <= 0" RuntimeWarning.
        Wvar = float(raw_W.var(ddof=1)) if raw_W.size > 1 else np.nan
        dG = Wmean - (beta * Wvar) / 2.0
        return {'Wmean': Wmean, 'dG': dG, 'Wdiss': Wmean - dG}

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]

            raw_W = group['work'].astype(float).values
            result = self.estimate_dG(raw_W, smd_data.beta)
            if result is None:
                continue

            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                **result,
            })
        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data


class ForceEstimator(BaseEstimator):
    """Force / raw-work estimator.

    Per-speed ``dG`` is the raw mean cumulative work ``Wmean`` (uncorrected);
    its v→0 intercept via :func:`extrapolate_to_v0` is the reversible work = ΔG.
    Also records ``Fmean`` (mean restraint force per step/speed/path) which the
    force-based friction (``FrictionEstimator.gamma_from_force_*``) turns into
    Γ = dF/dv.  Inherently multi-speed: a single speed gives no v→0 intercept.
    """

    @property
    def name(self):
        return 'force'

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        raw_W = np.asarray(raw_W, dtype=float)
        if raw_W.size == 0:
            return None
        Wmean = float(raw_W.mean())
        # Per-speed "PMF" is the raw work; dissipation is removed by the v→0
        # extrapolation, not per-speed.  Wdiss is NaN (force friction uses force).
        return {'Wmean': Wmean, 'dG': Wmean, 'Wdiss': np.nan}

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]

            raw_W = group['work'].astype(float).values
            result = self.estimate_dG(raw_W, smd_data.beta)
            if result is None:
                continue

            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                'Fmean': float(group['force'].astype(float).mean()),
                **result,
            })
        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data


class FrictionEstimator(BaseEstimator):
    """Estimate friction profiles from Wdiss.

    Two complementary estimators are provided:
    1) Derivative method (per speed): Gamma(r) = (dWdiss/dr) / v
    2) Regression method (across speeds): Wdiss(r, v) ≈ Gamma(r) * v + b(r)
    """

    def __init__(
        self,
        w_col: str = 'Wdiss',
        use_spline: bool = False,
        smooth_window: int | None = 11,
        smooth_method: str = 'savgol',
        smooth_polyorder: int = 3,
        speeds: list[float] | None = None,
    ):
        self.w_col = w_col
        self.use_spline = use_spline
        self.smooth_window = smooth_window
        self.smooth_method = smooth_method
        self.smooth_polyorder = smooth_polyorder
        self.speeds = speeds

    @property
    def name(self):
        return 'friction'

    def fit_transform(self, smd_data: SMDData):
        return super().fit_transform(smd_data)

    @staticmethod
    def _cumulative_integral(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.array([])
        if len(x) == 1:
            return np.array([0.0])
        dx = np.diff(x)
        trap = 0.5 * (y[1:] + y[:-1]) * dx
        out = np.zeros_like(x, dtype=float)
        out[1:] = np.cumsum(trap)
        return out

    def _compute_gamma_derivative(
        self,
        r: np.ndarray,
        wdiss: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Return dWdiss/dr / speed after optionally smoothing Wdiss."""
        if len(r) < 2 or speed <= 0:
            return np.full_like(r, np.nan, dtype=float)

        wd = self._smooth_profile(np.asarray(wdiss, dtype=float))

        dwd_dr = np.gradient(wd, r)
        
        return dwd_dr / speed

    def _smooth_profile(self, y: np.ndarray) -> np.ndarray:
        """Optionally smooth a 1D profile using Savitzky–Golay or Gaussian filter."""
        arr = np.asarray(y, dtype=float)
        n = len(arr)
        if n < 3 or self.smooth_window is None:
            return arr

        method = str(self.smooth_method).lower()
        if method == 'savgol':
            win = int(self.smooth_window)
            if win < 3:
                return arr
            if win % 2 == 0:
                win += 1
            if win > n:
                win = n if n % 2 == 1 else n - 1
            if win < 3:
                return arr

            poly = int(self.smooth_polyorder)
            if poly >= win:
                poly = win - 1
            if poly < 1:
                poly = 1

            return savgol_filter(arr, window_length=win, polyorder=poly, mode='interp')

        if method == 'gaussian':
            sigma = float(self.smooth_window)
            if sigma <= 0:
                return arr
            return gaussian_filter1d(arr, sigma=sigma, mode='nearest')

        raise ValueError(
            f"Unknown smooth_method '{self.smooth_method}'. Allowed values: 'savgol', 'gaussian'."
        )

    def gamma_from_wdiss_derivative(
        self,
        df: pd.DataFrame,
        estimator: str | None = None,
    ) -> pd.DataFrame:
        """Compute the friction profile Γ(r) = dWdiss/dr / v using the derivative method.

        For each (estimator, speed[, path]) group, sorts by r_coord, smooths Wdiss,
        and takes a numerical gradient to obtain the local friction coefficient Γ(r)
        at each grid point.  Also returns the cumulative integral Γ_integrated(r).

        Parameters
        ----------
        df : pd.DataFrame
            Estimator results table with columns ``r_coord``, ``speed``,
            ``estimator``, and the configured ``w_col`` (default ``'Wdiss'``).
        estimator : str or None
            If provided, filter to a single estimator label before computing.

        Returns
        -------
        pd.DataFrame
            One row per grid point per (estimator, speed[, path]) with columns
            ``r_coord``, ``speed``, ``Gamma``, ``Gamma_integrated``, ``method``
            (= ``'derivative'``), ``estimator``, and optionally ``path``.

        Note
        ----
        For backward pulling, r_coord decreases with step.  After sorting by
        r_coord the step index is reversed: the small-r end (center) comes first
        and carries the largest accumulated Wdiss, making dWdiss/dr negative.
        The correct formula for backward pulling is γ = −dWdiss/dr / v (sign flip
        because dr < 0 during the pull).  Direction is detected from the ``step``
        column when available, otherwise from the sign of Wdiss[0] − Wdiss[−1].
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if self.w_col not in df.columns:
            raise ValueError(f"Column '{self.w_col}' not found in input DataFrame.")

        data = df.copy()
        if estimator is not None and 'estimator' in data.columns:
            data = data[data['estimator'] == estimator]
        if self.speeds is not None:
            data = data[data['speed'].isin(self.speeds)]

        group_cols = ['estimator', 'speed']
        if 'path' in data.columns:
            group_cols.append('path')

        rows = []
        for keys, g in data.groupby(group_cols, dropna=False):
            g = g.sort_values('r_coord').dropna(subset=['r_coord', self.w_col])
            if g.empty:
                continue

            r = g['r_coord'].to_numpy(dtype=float)
            wdiss = g[self.w_col].to_numpy(dtype=float)
            speed = float(g['speed'].iloc[0])

            # Detect pulling direction: if step at r_min > step at r_max, it's backward.
            if 'step' in g.columns:
                is_backward = float(g.iloc[0]['step']) > float(g.iloc[-1]['step'])
            else:
                is_backward = wdiss[0] > wdiss[-1]
            direction_sign = -1.0 if is_backward else 1.0

            gamma = self._compute_gamma_derivative(r, wdiss, speed) * direction_sign
            gamma_int = self._cumulative_integral(r, gamma)

            out = g[['step', 'r_coord', 'speed']].copy() if 'step' in g.columns else g[['r_coord', 'speed']].copy()
            out['Gamma'] = gamma
            out['Gamma_integrated'] = gamma_int
            out['method'] = 'derivative'
            out['estimator'] = g['estimator'].iloc[0] if 'estimator' in g.columns else estimator
            if 'path' in g.columns:
                out['path'] = g['path'].iloc[0]

            rows.append(out)

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def gamma_from_wdiss_regression(
        self,
        df: pd.DataFrame,
        estimator: str | None = None,
    ) -> pd.DataFrame:
        """Compute the friction profile Γ(r) by linear regression of Wdiss across speeds.

        Fits Wdiss(r, v) ≈ Γ(r)·v + b(r) at each grid point using
        :func:`extrapolate_to_v0`, then differentiates the resulting
        Gamma_integrated(r) = slope(r) to obtain the local friction Γ(r).

        This cross-speed approach averages over pulling-speed noise and is
        complementary to the per-speed derivative method
        (:meth:`gamma_from_wdiss_derivative`).

        Parameters
        ----------
        df : pd.DataFrame
            Estimator results table containing at least ``r_coord``, ``speed``,
            ``estimator``, and the configured ``w_col`` (default ``'Wdiss'``).
        estimator : str or None
            If provided, filter to a single estimator label before fitting.

        Returns
        -------
        pd.DataFrame
            One row per grid point per estimator with columns ``r_coord``,
            ``estimator``, ``Gamma``, ``Gamma_integrated``, ``method``
            (= ``'regression'``), and ``step`` when available.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if self.w_col not in df.columns:
            raise ValueError(f"Column '{self.w_col}' not found in input DataFrame.")

        data = df.copy()
        if estimator is not None and 'estimator' in data.columns:
            data = data[data['estimator'] == estimator]
        if self.speeds is not None:
            data = data[data['speed'].isin(self.speeds)]

        if 'estimator' not in data.columns:
            data['estimator'] = estimator if estimator is not None else 'unknown'

        v0_df = extrapolate_to_v0(
            results=data,
            param=self.w_col,
            speeds=self.speeds,
        )

        if v0_df is None or v0_df.empty:
            return pd.DataFrame()

        slope_col = f"{self.w_col}_slope"

        if slope_col not in v0_df.columns:
            raise ValueError(
                f"Expected slope column '{slope_col}' in extrapolation output. "
                f"Available: {list(v0_df.columns)}"
            )

        out = v0_df.copy()
        out['Gamma_integrated'] = out[slope_col].astype(float)
        out['method'] = 'regression'

        key_col = 'step' if 'step' in out.columns else 'r_coord'
        out = out.sort_values(['estimator', key_col]).reset_index(drop=True)

        # Local friction: Gamma(r) = d/dr [Gamma_integrated(r)]
        out['Gamma'] = np.nan
        for est, g in out.groupby('estimator', dropna=False):
            idx = g.index
            gg = g.sort_values('r_coord')
            r = gg['r_coord'].to_numpy(dtype=float)
            gint = gg['Gamma_integrated'].to_numpy(dtype=float)

            gint = self._smooth_profile(gint)

            if len(r) < 2:
                gamma_local = np.full(len(r), np.nan, dtype=float)
            else:
                gamma_local = np.gradient(gint, r)

            out.loc[gg.index, 'Gamma'] = gamma_local

        drop_cols = [c for c in out.columns if c.endswith('_se') or c in {'R2', 'n_speeds'}]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        return out

    @staticmethod
    def _mix_force(force_results: pd.DataFrame, weights: dict) -> pd.DataFrame:
        """Path-mix Fmean into <F>(step, speed) using p_eq weights.

        ``weights`` is ``{speed: {path: p_eq}}``.  For single-path systems this
        reduces to the plain mean.  Returns columns
        ``estimator, step, r_coord, speed, Fbar``.
        """
        rows = []
        for (speed, step), g in force_results.groupby(['speed', 'step']):
            sw = weights.get(speed, {}) if weights else {}
            num = den = 0.0
            rcoords = []
            for _, row in g.iterrows():
                w = sw.get(row['path'], 0.0) if sw else 1.0
                if w <= 0:
                    continue
                num += w * float(row['Fmean'])
                den += w
                rcoords.append(float(row['r_coord']))
            if den <= 0:
                continue
            rows.append(dict(estimator='force', step=step, speed=speed,
                             r_coord=float(np.mean(rcoords)), Fbar=num / den))
        return pd.DataFrame(rows)

    def gamma_from_force_regression(self, force_results: pd.DataFrame,
                                    weights: dict) -> pd.DataFrame:
        """Γ(r) = dF/dv (slope) and Feq(r) (intercept) via across-speed OLS of <F>."""
        if force_results is None or force_results.empty:
            return pd.DataFrame()
        fbar = self._mix_force(force_results, weights)
        if fbar.empty or fbar['speed'].nunique() < 2:
            return pd.DataFrame()

        v0 = extrapolate_to_v0(results=fbar, param='Fbar', speeds=self.speeds)
        if v0 is None or v0.empty:
            return pd.DataFrame()

        out = v0.copy()
        out['Feq'] = out['Fbar'].astype(float)          # intercept
        out['Gamma'] = out['Fbar_slope'].astype(float)  # local friction dF/dv
        out = out.sort_values('r_coord').reset_index(drop=True)
        r = out['r_coord'].to_numpy(dtype=float)
        out['Gamma_integrated'] = self._cumulative_integral(r, out['Gamma'].to_numpy(dtype=float))
        out['method'] = 'regression'
        out['estimator'] = 'force'
        keep = ['r_coord', 'step', 'speed', 'Gamma', 'Gamma_integrated',
                'Feq', 'method', 'estimator']
        return out[[c for c in keep if c in out.columns]]

    def gamma_from_force_derivative(self, force_results: pd.DataFrame,
                                    weights: dict) -> pd.DataFrame:
        """Per-speed friction Γ_sp(r) = (<F>(r;v) − Feq(r)) / v."""
        if force_results is None or force_results.empty:
            return pd.DataFrame()
        fbar = self._mix_force(force_results, weights)
        if fbar.empty or fbar['speed'].nunique() < 2:
            return pd.DataFrame()

        reg = self.gamma_from_force_regression(force_results, weights)
        if reg.empty:
            return pd.DataFrame()
        feq_by_step = reg.set_index('step')['Feq'].to_dict()

        rows = []
        for speed, g in fbar.groupby('speed'):
            if speed <= 0:
                continue
            g = g.sort_values('r_coord')
            # Friction is undefined at steps the regression trimmed (no Feq);
            # drop them rather than zero-filling, which would bias the running
            # cumulative integral for all subsequent steps.
            g = g[g['step'].isin(feq_by_step.keys())]
            if g.empty:
                continue
            r = g['r_coord'].to_numpy(dtype=float)
            gamma = np.array([(fb - feq_by_step[st]) / speed
                              for fb, st in zip(g['Fbar'], g['step'])], dtype=float)
            gamma_int = self._cumulative_integral(r, gamma)
            rows.append(pd.DataFrame({
                'r_coord': r, 'step': g['step'].to_numpy(), 'speed': speed,
                'Gamma': gamma, 'Gamma_integrated': gamma_int,
                'method': 'derivative', 'estimator': 'force',
            }))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _find_pmf_peak(
    r: np.ndarray,
    dG: np.ndarray,
    kBT: float,
    *,
    smooth: str = 'savgol',
    smooth_window: int = 5,
    smooth_polyorder: int = 3,
    prominence_factor: float = 0.5,
) -> tuple[float | None, float | None]:
    """Return (r_ts_nm, barrier_height_kJ) of the most prominent PMF peak.

    Smooths the profile then locates the highest-prominence peak via
    ``find_peaks``.  Returns ``(None, None)`` when no peak exceeds
    ``prominence_factor * kBT``.  ``barrier_height = smoothed_dG[peak] - smoothed_dG[0]``.

    Parameters
    ----------
    smooth : 'savgol' (default) or 'gaussian'
        Smoothing method.  For 'savgol', ``smooth_window`` is the filter
        window (odd int, auto-adjusted); for 'gaussian', it is σ in grid points.
    prominence_factor : float
        Peak prominence threshold as a multiple of kBT (default 0.5).

    Both ``KramersEstimator.detect_ts`` and ``SMDAnalysis._compute_barrier_rts``
    delegate to this function so the algorithm stays in one place.
    """
    arr = dG.astype(float)
    if smooth == 'savgol':
        win = int(smooth_window)
        if win % 2 == 0:
            win += 1
        win = min(win, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if win >= 3:
            poly = min(int(smooth_polyorder), win - 1)
            poly = max(poly, 1)
            smoothed = savgol_filter(arr, window_length=win, polyorder=poly, mode='interp')
        else:
            smoothed = arr
    else:
        smoothed = gaussian_filter1d(arr, sigma=float(smooth_window), mode='nearest')

    prominence = prominence_factor * kBT
    peak_idxs, props = find_peaks(smoothed, prominence=prominence)
    if len(peak_idxs) == 0:
        return None, None
    best = peak_idxs[int(np.argmax(props["prominences"]))]
    return float(r[best]), float(smoothed[best] - smoothed[0])


class KramersEstimator:
    """Kramers/Pontryagin MFPT k_off from pre-computed PMF and friction profiles.

    NOT a BaseEstimator subclass — operates on mixture_pmfs + friction DataFrames
    produced by the sMD analysis pipeline rather than on raw per-step work values.

    Called automatically at the end of ``SMDAnalysis.run()`` after ``mixture_pmfs.csv``
    and ``friction.csv`` have been written.

    Output (one row per dG estimator present in *mixture_pmfs*) columns:
      estimator, source, speed_nm_per_ps, T_K, start_r_nm, abs_r_nm,
      reflect_r_nm, tau_mfpt_ps, tau_mfpt_ns, k_off_per_s, barrier_kjmol,
      barrier_r_nm, max_beta_dF, attempt_time_ps, dG_apparent_kjmol
    """

    R_GAS_KJ_MOL_K = 8.314e-3   # kJ/(mol·K)
    PS_PER_S = 1.0e12

    def __init__(
        self,
        temperature: float = 310.0,
        gamma_floor: float = 1.0,
        attempt_time_ps: float = 1.0,
        barrier_warn_kjmol: float = 80.0,
    ):
        self.temperature = temperature
        self.gamma_floor = gamma_floor
        self.attempt_time_ps = attempt_time_ps
        self.barrier_warn_kjmol = barrier_warn_kjmol
        self.kBT = self.R_GAS_KJ_MOL_K * temperature

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge_pmf_friction(
        pmf_df: pd.DataFrame,
        fric_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inner join of PMF and friction.

        Merges on ``step`` (integer, protocol-anchored) when both DataFrames
        carry the column, avoiding float jitter in ``r_coord`` that causes
        spurious row drops after ``extrapolate_to_v0()`` end-trimming.
        Falls back to ``r_coord`` for the per-path codepath where ``step``
        is not available.  ``r_coord`` in the output always comes from
        *pmf_df* (authoritative source).
        """
        use_step = "step" in pmf_df.columns and "step" in fric_df.columns
        key_label = "step" if use_step else "r_coord"

        if use_step:
            left = pmf_df[["step", "r_coord", "dG"]].copy()
            right = fric_df[["step", "Gamma"]].copy()
            # Coerce to int64 defensively (float step can arise after CSV round-trip)
            left["step"] = left["step"].astype("int64")
            right["step"] = right["step"].astype("int64")
            merged = pd.merge(left, right, on="step", how="inner")
            # r_coord comes from pmf_df (left); no r_coord_x/r_coord_y ambiguity
        else:
            merged = pd.merge(
                pmf_df[["r_coord", "dG"]],
                fric_df[["r_coord", "Gamma"]],
                on="r_coord",
                how="inner",
            )

        dropped = max(len(pmf_df), len(fric_df)) - len(merged)
        if dropped > 3 and len(merged) > 0:
            logger.warning(
                f"[Kramers] merge key='{key_label}': PMF rows={len(pmf_df)}, "
                f"friction rows={len(fric_df)}, merged rows={len(merged)} "
                f"({dropped} dropped)"
            )
        return merged

    @staticmethod
    def _select_profile(
        pmf_df: pd.DataFrame,
        fric_df: pd.DataFrame,
        dG_extrapolated: pd.DataFrame | None,
        estimator_name: str,
        prefer_v0: bool,
        speed_override: float | None,
    ) -> tuple[pd.DataFrame, str, float]:
        """Choose the best (merged PMF+friction) profile for *estimator_name*.

        Preference order (when *prefer_v0* is True):
        1. v→0 extrapolated dG  +  regression friction  (source: ``v0_extrapolation``)
        2. Slowest per-speed dG  +  derivative friction  (source: ``per_speed(v=…)``)

        Returns ``(merged_df, source_label, speed_value)``.
        """
        pmf_sub = pmf_df[pmf_df["estimator"] == estimator_name]
        fric_sub = fric_df[fric_df["estimator"] == estimator_name]

        if pmf_sub.empty:
            raise RuntimeError(
                f"[Kramers] Estimator '{estimator_name}' not found in mixture_pmfs."
            )

        # --- try v→0 extrapolation ---
        if prefer_v0 and dG_extrapolated is not None:
            v0_sub = dG_extrapolated[dG_extrapolated["estimator"] == estimator_name]
            fric_reg = fric_sub[fric_sub["method"] == "regression"]
            if not v0_sub.empty and not fric_reg.empty:
                merged = KramersEstimator._merge_pmf_friction(v0_sub, fric_reg)
                if not merged.empty:
                    return merged, "v0_extrapolation", 0.0
            logger.warning(
                "[Kramers] v→0 profile empty or missing regression friction; "
                "falling back to per-speed profile."
            )

        # --- per-speed fallback ---
        fric_deriv = fric_sub[fric_sub["method"] == "derivative"]
        common_speeds = sorted(
            set(pmf_sub["speed"].unique()) & set(fric_deriv["speed"].unique())
        )
        if not common_speeds:
            raise RuntimeError(
                f"[Kramers] No common (speed, estimator='{estimator_name}') "
                "between mixture_pmfs and friction."
            )

        if speed_override is not None:
            chosen = min(common_speeds, key=lambda s: abs(s - speed_override))
        else:
            chosen = min(common_speeds)
            logger.info(f"[Kramers] Using slowest speed: {chosen} nm/ps")

        merged = KramersEstimator._merge_pmf_friction(
            pmf_sub[pmf_sub["speed"] == chosen],
            fric_deriv[fric_deriv["speed"] == chosen],
        )
        if merged.empty:
            raise RuntimeError(
                f"[Kramers] Empty merge for estimator={estimator_name}, speed={chosen}"
            )
        return merged, f"per_speed(v={chosen})", float(chosen)

    @staticmethod
    def _iter_profiles(
        pmf_df: pd.DataFrame,
        fric_df: pd.DataFrame,
        dG_extrapolated: pd.DataFrame | None,
        estimator_name: str,
        prefer_v0: bool,
    ) -> list[tuple[pd.DataFrame, str, float]]:
        """Return all available (merged, source_label, speed) profiles for *estimator_name*.

        Includes:
        - v→0 extrapolated profile (when *prefer_v0* is True and data are available)
        - One profile per pulling speed found in both *pmf_df* and derivative friction
        """
        profiles: list[tuple[pd.DataFrame, str, float]] = []
        pmf_sub = pmf_df[pmf_df["estimator"] == estimator_name]
        fric_sub = fric_df[fric_df["estimator"] == estimator_name]

        if pmf_sub.empty:
            raise RuntimeError(
                f"[Kramers] Estimator '{estimator_name}' not found in mixture_pmfs."
            )

        # v→0 extrapolated profile
        if prefer_v0 and dG_extrapolated is not None:
            v0_sub = dG_extrapolated[dG_extrapolated["estimator"] == estimator_name]
            fric_reg = fric_sub[fric_sub["method"] == "regression"]
            if not v0_sub.empty and not fric_reg.empty:
                merged = KramersEstimator._merge_pmf_friction(v0_sub, fric_reg)
                if not merged.empty:
                    profiles.append((merged, "v0_extrapolation", 0.0))

        # per-speed profiles (derivative friction)
        fric_deriv = fric_sub[fric_sub["method"] == "derivative"]
        common_speeds = sorted(
            set(pmf_sub["speed"].unique()) & set(fric_deriv["speed"].unique())
        )
        for speed in common_speeds:
            merged = KramersEstimator._merge_pmf_friction(
                pmf_sub[pmf_sub["speed"] == speed],
                fric_deriv[fric_deriv["speed"] == speed],
            )
            if not merged.empty:
                profiles.append((merged, f"per_speed(v={speed})", float(speed)))

        if not profiles:
            raise RuntimeError(
                f"[Kramers] No usable profiles found for estimator='{estimator_name}'."
            )
        return profiles

    @staticmethod
    def detect_ts(r: np.ndarray, dG: np.ndarray, kBT: float) -> float | None:
        """Return r_ts (nm) of the highest-prominence PMF peak, or None."""
        r_ts, _ = _find_pmf_peak(r, dG, kBT)
        return r_ts

    @staticmethod
    def force_plateau_boundary(
        force_df: pd.DataFrame,
        speed: float,
        start_r: float,
        r_max: float,
        frac: float = 0.3,
    ) -> float | None:
        """Absorbing boundary from the pulling restraint force (kinetics-motivated).

        The mean |force| along the reaction coordinate peaks at the rupture and
        decays as the ligand leaves the pocket. The boundary is placed where the
        (speed-matched) mean |force| has decayed to *frac* of its peak past that
        peak. Estimator-independent (raw pulling data), so it avoids the failure
        mode of :meth:`detect_ts` where a noisy/monotonic reconstructed PMF has no
        clean interior barrier (jarzynski) or a spurious far peak (cumulant at
        higher speed). Returns ``None`` when it cannot be located, in which case
        the caller should fall back to :meth:`detect_ts`.

        Validated on the HSP90 kinetics set (koff↔pKoff): with ``frac=0.3`` it
        rescues the cumulant boundary failures and matches jarzynski. NOTE: this is
        a *kinetic* boundary for the Kramers MFPT — it is not a thermodynamic
        affinity readout (use max/end dG for that).

        Parameters
        ----------
        force_df : DataFrame
            Per-frame processed sMD data (``sMD_processed_data``) with columns
            ``r_coord``, ``force`` and ``speed``.
        speed : float
            Pulling speed (nm/ps) of the profile. ``0.0`` (v→0 extrapolation) uses
            the slowest available speed's force.
        start_r, r_max : float
            Reaction-coordinate bounds of the PMF profile (nm).
        frac : float
            Decay fraction of the peak mean |force| defining the boundary.
        """
        prof = KramersEstimator._binned_mean_abs_force(force_df, speed, start_r, r_max)
        if prof is None:
            return None
        r_bc, mF, pk = prof
        after = np.where(mF[pk:] < frac * mF[pk])[0]
        if len(after) == 0:
            return None
        return float(min(r_bc[pk + after[0]], r_max - 1e-6))

    @staticmethod
    def _binned_mean_abs_force(force_df, speed, start_r, r_max):
        """Shared binning for the force-based boundaries.

        Returns ``(r_centers, smoothed_mean_abs_force, peak_index)`` with
        ``r_centers`` in absolute reaction-coordinate units
        (``start_r`` + displacement), or ``None`` when the force profile is
        unusable (missing columns, too few frames, degenerate range).
        """
        if force_df is None or not {"r_coord", "force", "speed"}.issubset(force_df.columns):
            return None
        speeds = sorted(force_df["speed"].dropna().unique())
        if not speeds:
            return None
        use = speed if (speed > 0 and speed in speeds) else speeds[0]
        d = force_df[force_df["speed"] == use].dropna(subset=["r_coord", "force"])
        if len(d) < 100:
            return None
        disp = d["r_coord"].to_numpy(dtype=np.float64) - float(d["r_coord"].min())
        F = np.abs(d["force"].to_numpy(dtype=np.float64))
        hi = float(np.quantile(disp, 0.98))
        if hi <= 0:
            return None
        bins = np.linspace(0.0, hi, 60)
        bc = 0.5 * (bins[:-1] + bins[1:])
        idx = np.clip(np.digitize(disp, bins) - 1, 0, len(bc) - 1)
        mF = np.array([F[idx == i].mean() if (idx == i).any() else np.nan
                       for i in range(len(bc))])
        ok = ~np.isnan(mF)
        if ok.sum() < 5:
            return None
        mF = np.interp(np.arange(len(bc)), np.where(ok)[0], mF[ok])
        mF = pd.Series(mF).rolling(5, center=True, min_periods=1).mean().to_numpy()
        pk = int(np.argmax(mF))
        return (start_r + bc, mF, pk)

    @staticmethod
    def force_peak_r(force_df, speed, start_r, r_max):
        """Reaction coordinate (nm) of the mean|force| peak — the rupture / TS.

        This is the barrier the ligand crosses; the Kramers absorbing boundary
        (:meth:`force_plateau_boundary`) sits past it. ``None`` if the force
        profile is unusable.
        """
        prof = KramersEstimator._binned_mean_abs_force(force_df, speed, start_r, r_max)
        if prof is None:
            return None
        r_bc, _mF, pk = prof
        return float(r_bc[pk])

    @staticmethod
    def kramers_mfpt(
        profile_df: pd.DataFrame,
        start_r: float,
        abs_r: float,
        reflect_r: float | None,
        kBT: float,
        gamma_floor: float,
    ) -> dict:
        """Compute the mean first-passage time (MFPT) via the Pontryagin/Kramers double integral.

        Evaluates τ = ∫_{start_r}^{abs_r} [exp(+βΔF(r)) / D(r)]
                           · ∫_{reflect_r}^{r} exp(−βΔF(r')) dr' dr
        where D(r) = kBT / Γ(r) is the position-dependent diffusivity.

        Parameters
        ----------
        profile_df : pd.DataFrame
            Merged PMF + friction table with columns ``r_coord``, ``dG``, and
            ``Gamma`` (produced by :meth:`_merge_pmf_friction`).
        start_r : float
            Starting coordinate (nm) — the reflecting/equilibrium position.
        abs_r : float
            Absorbing boundary coordinate (nm) — typically the barrier peak
            located by :meth:`detect_ts` (calls :func:`_find_pmf_peak`).
        reflect_r : float or None
            Reflecting boundary for the inner integral.  ``None`` defaults to
            ``r_coord.min()`` (forward pull) or ``r_coord.max()`` (backward pull).
        kBT : float
            Thermal energy in kJ/mol.
        gamma_floor : float
            Minimum allowed Γ value (kJ·ps/mol/nm²); clips non-physical negatives.

        Returns
        -------
        dict
            Diagnostics dictionary with keys:
            ``tau_ps`` (MFPT in ps), ``barrier_kjmol``, ``barrier_r_nm``,
            ``max_beta_dF``, ``reflect_r_canon``, ``mirrored``, ``start_r_input``.

        Note
        ----
        ΔF is first shifted so ΔF(start_r) = 0, then shifted again so its
        minimum is ≥ 0 (to clip noise-induced dips below zero).  This second
        shift is valid because the Pontryagin MFPT integral is invariant to a
        constant shift of ΔF — the constant cancels between the exp(+βΔF) and
        exp(−βΔF) factors in the double integral.

        A soft warning is emitted when max(β·ΔF) > 100 (precision degrades).
        A hard guard triggers at max(β·ΔF) ≈ 708 (float64 overflow); in that
        case ``tau_ps = inf`` is returned.
        """
        df = (
            profile_df.sort_values("r_coord")
            .drop_duplicates("r_coord")
            .reset_index(drop=True)
        )
        r = df["r_coord"].to_numpy(dtype=np.float64)
        dF = df["dG"].to_numpy(dtype=np.float64)
        G = df["Gamma"].to_numpy(dtype=np.float64)

        r_min, r_max = float(r.min()), float(r.max())
        if not (r_min <= start_r <= r_max):
            raise ValueError(
                f"[Kramers] start_r={start_r:.4f} outside data range "
                f"[{r_min:.4f}, {r_max:.4f}]"
            )
        if not (r_min <= abs_r <= r_max):
            raise ValueError(
                f"[Kramers] abs_r={abs_r:.4f} outside data range "
                f"[{r_min:.4f}, {r_max:.4f}]"
            )
        if abs(abs_r - start_r) < 1e-6:
            raise ValueError("[Kramers] start_r and abs_r coincide — MFPT is trivially zero")

        if reflect_r is None:
            reflect_r = r_min if abs_r > start_r else r_max
        if (abs_r > start_r and reflect_r > start_r) or (
            abs_r < start_r and reflect_r < start_r
        ):
            raise ValueError(
                f"[Kramers] reflect_r must lie on the opposite side of start_r from abs_r "
                f"(start={start_r}, abs={abs_r}, reflect={reflect_r})"
            )

        n_neg = int((G < gamma_floor).sum())
        if n_neg:
            logger.warning(
                f"[Kramers] clipped {n_neg}/{G.size} Γ values to floor={gamma_floor}"
            )
        G = np.clip(G, gamma_floor, None)

        # Reference ΔF so ΔF(start_r) = 0
        dF = dF - float(np.interp(start_r, r, dF))

        # Shift dF so its minimum is ≥ 0 (clips noise-induced dips below zero).
        dF_min = float(np.nanmin(dF))
        if dF_min < 0.0:
            dF = dF - dF_min

        # Canonical ascending form: mirror if absorbing boundary is to the left
        mirrored = abs_r < start_r
        if mirrored:
            r = 2.0 * start_r - r[::-1]
            dF = dF[::-1]
            G = G[::-1]
            reflect_r = 2.0 * start_r - reflect_r
            abs_r = 2.0 * start_r - abs_r

        beta_dF = dF / kBT
        max_bdg = float(np.nanmax(beta_dF))

        # Compute mask and barrier early so the early-return path can populate them
        mask = (r >= start_r) & (r <= abs_r)
        if mask.sum() < 2:
            raise RuntimeError(
                "[Kramers] Fewer than 2 grid points between start_r and abs_r"
            )
        barrier_mask_dF = dF[mask]
        barrier_kjmol = float(np.max(barrier_mask_dF))
        barrier_r = float(r[mask][int(np.argmax(barrier_mask_dF))])
        if mirrored:
            barrier_r = 2.0 * start_r - barrier_r

        # Soft warning: precision degrades for β·ΔF > 100
        if max_bdg > 100.0:
            logger.warning(
                f"[Kramers] max(β·ΔF) = {max_bdg:.1f} — exp(β·ΔF) may overflow; "
                "τ estimate may be unreliable."
            )

        # Hard guard: float64 saturates at exp(≈709); beyond this τ = ∞
        _FLOAT64_MAX_EXP = np.log(np.finfo(np.float64).max) - 1.0  # ≈ 708.4
        if max_bdg > _FLOAT64_MAX_EXP:
            logger.warning(
                f"[Kramers] max(β·ΔF) = {max_bdg:.1f} exceeds float64 limit "
                f"({_FLOAT64_MAX_EXP:.0f}); τ = ∞. "
                "Use more replicas or slower pulling speeds."
            )
            return {
                "tau_ps": float("inf"),
                "barrier_kjmol": barrier_kjmol,
                "barrier_r_nm": barrier_r,
                "max_beta_dF": max_bdg,
                "reflect_r_canon": reflect_r,
                "mirrored": mirrored,
                "start_r_input": start_r,
            }

        exp_neg = np.exp(-beta_dF)
        exp_pos = np.exp(+beta_dF)
        D = kBT / G  # nm²/ps

        # Inner cumulative integral: I_inner(r_i) = ∫_{reflect_r}^{r_i} exp(-βΔF) dr'
        dr = np.diff(r)
        inner_cum = np.concatenate(
            ([0.0], np.cumsum(0.5 * (exp_neg[:-1] + exp_neg[1:]) * dr))
        )
        inner_cum -= float(np.interp(reflect_r, r, inner_cum))

        # Outer integrand: exp(+βΔF)/D · I_inner
        outer = exp_pos / D * inner_cum

        # Integrate from start_r to abs_r
        r_int = r[mask]
        outer_int = outer[mask]
        if r_int[0] > start_r + 1e-9:
            r_int = np.concatenate(([start_r], r_int))
            outer_int = np.concatenate(([float(np.interp(start_r, r, outer))], outer_int))
        if r_int[-1] < abs_r - 1e-9:
            r_int = np.concatenate((r_int, [abs_r]))
            outer_int = np.concatenate((outer_int, [float(np.interp(abs_r, r, outer))]))

        tau_ps = float(np.trapz(outer_int, r_int))

        return {
            "tau_ps": tau_ps,
            "barrier_kjmol": barrier_kjmol,
            "barrier_r_nm": barrier_r,
            "max_beta_dF": max_bdg,
            "reflect_r_canon": reflect_r,
            "mirrored": mirrored,
            "start_r_input": start_r,
        }

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def compute(
        self,
        mixture_pmfs: pd.DataFrame,
        friction_df: pd.DataFrame,
        dG_extrapolated: pd.DataFrame | None = None,
        start_r: float | None = None,
        abs_r: float | None = None,
        reflect_r: float | None = None,
        prefer_v0: bool = True,
        speed_override: float | None = None,
        all_speeds: bool = True,
        force_df: pd.DataFrame | None = None,
        boundary_method: str = "pmf_peak",
        plateau_frac: float = 0.3,
    ) -> pd.DataFrame:
        """Run Kramers MFPT for every dG estimator present in *mixture_pmfs*.

        Parameters
        ----------
        mixture_pmfs : DataFrame
            Output of ``calculate_weighted_pmf()``; columns include ``estimator``,
            ``r_coord``, ``dG``, ``Wdiss``, ``speed``.
        friction_df : DataFrame
            Output of ``FrictionEstimator``; columns include ``estimator``,
            ``r_coord``, ``Gamma``, ``method``, ``speed``.
        dG_extrapolated : DataFrame or None
            v→0 extrapolated PMF (``dG_extrapolated.csv``).  When provided and
            matching regression friction is available, it is included as an
            additional speed=0 row (when *all_speeds* is True) or preferred
            over per-speed profiles (when *all_speeds* is False).
        start_r : float or None
            Reflecting-wall position (nm).  ``None`` → ``r_coord.min()`` of each
            profile.
        abs_r : float or None
            Absorbing-boundary position (nm).  ``None`` → auto-detect via
            *boundary_method*.
        force_df : DataFrame or None
            Per-frame processed sMD data (``sMD_processed_data``) with ``r_coord``,
            ``force``, ``speed``.  Required for ``boundary_method="force_plateau"``.
        boundary_method : str
            How to place ``abs_r`` when it is not given explicitly:
            ``"pmf_peak"`` (default) → :meth:`detect_ts` (highest-prominence PMF
            peak, falling back to ``r_coord.max()``);
            ``"force_plateau"`` → :meth:`force_plateau_boundary` (restraint-force
            decay; falls back to ``detect_ts`` if the force profile is unusable).
            Force-plateau (``plateau_frac=0.3``) is validated on HSP90 kinetics to
            rescue cumulant boundary failures; it is a *kinetic* boundary only.
        plateau_frac : float
            Decay fraction for ``boundary_method="force_plateau"`` (default 0.3).
        reflect_r : float or None
            Explicit reflecting boundary (nm); ``None`` → same as *start_r*.
        prefer_v0 : bool
            Include the v→0 extrapolated profile (when *all_speeds* is True) or
            prefer it over per-speed profiles (when *all_speeds* is False).
        speed_override : float or None
            Force a specific pulling speed (nm/ps).  Implies ``all_speeds=False``.
        all_speeds : bool
            When True (default), compute Kramers for the v→0 extrapolated profile
            (if available) *and* every per-speed profile — one output row each.
            When False, use the single best profile (v→0 preferred, or slowest
            per-speed as fallback); *speed_override* selects a specific speed.

        Returns
        -------
        pd.DataFrame
            One row per (estimator, speed) with columns matching ``koff_kramers.csv``.
        """
        # speed_override implies single-speed mode
        if speed_override is not None:
            all_speeds = False

        records = []
        for est_name in mixture_pmfs["estimator"].unique():
            try:
                if all_speeds:
                    profile_list = self._iter_profiles(
                        mixture_pmfs, friction_df, dG_extrapolated,
                        est_name, prefer_v0,
                    )
                else:
                    profile, source, speed = self._select_profile(
                        mixture_pmfs, friction_df, dG_extrapolated,
                        est_name, prefer_v0, speed_override,
                    )
                    profile_list = [(profile, source, speed)]
            except (RuntimeError, ValueError) as exc:
                logger.warning(f"[Kramers] Skipping estimator '{est_name}': {exc}")
                continue

            for profile, source, speed in profile_list:
                r_arr = profile["r_coord"].to_numpy(dtype=np.float64)
                dG_arr = profile["dG"].to_numpy(dtype=np.float64)

                _start_r = start_r if start_r is not None else float(r_arr.min())

                if abs_r is not None:
                    _abs_r = abs_r
                elif boundary_method == "force_plateau" and force_df is not None:
                    _abs_r = self.force_plateau_boundary(
                        force_df, speed, _start_r, float(r_arr.max()), plateau_frac
                    )
                    if _abs_r is not None:
                        logger.info(
                            f"[Kramers] Force-plateau boundary abs_r={_abs_r:.4f} nm "
                            f"(estimator={est_name}, speed={speed}, frac={plateau_frac})."
                        )
                    else:
                        # force profile unusable → fall back to PMF peak / r_coord.max()
                        r_ts = self.detect_ts(r_arr, dG_arr, self.kBT)
                        _abs_r = r_ts if r_ts is not None else float(r_arr.max())
                        logger.info(
                            f"[Kramers] Force-plateau unavailable (estimator={est_name}, "
                            f"speed={speed}); fell back to "
                            f"{'PMF peak' if r_ts is not None else 'r_coord.max()'} "
                            f"abs_r={_abs_r:.4f} nm."
                        )
                else:
                    r_ts = self.detect_ts(r_arr, dG_arr, self.kBT)
                    if r_ts is not None:
                        logger.info(
                            f"[Kramers] Detected TS at r_ts={r_ts:.4f} nm "
                            f"(estimator={est_name}, speed={speed}); using as abs_r."
                        )
                        _abs_r = r_ts
                    else:
                        logger.info(
                            f"[Kramers] No PMF peak found for estimator='{est_name}', "
                            f"speed={speed}; using r_coord.max() as abs_r."
                        )
                        _abs_r = float(r_arr.max())

                try:
                    res = self.kramers_mfpt(
                        profile, _start_r, _abs_r, reflect_r, self.kBT, self.gamma_floor
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        f"[Kramers] MFPT integral failed for estimator='{est_name}', "
                        f"speed={speed}: {exc}"
                    )
                    continue

                if res["barrier_kjmol"] > self.barrier_warn_kjmol:
                    logger.warning(
                        f"[Kramers] Large barrier detected: {res['barrier_kjmol']:.1f} kJ/mol "
                        f"(estimator={est_name}, source={source}, speed={speed}). "
                        f"Barriers >{self.barrier_warn_kjmol:.0f} kJ/mol are likely artifacts "
                        "from insufficient sampling. "
                        "Recommended: increase replicas to ≥25 per speed, add slower pulling "
                        "speeds (v ≤ 0.0005 nm/ps), and verify Wdiss ∝ v (linear-response check)."
                    )

                tau_ps = res["tau_ps"]
                if not np.isfinite(tau_ps):
                    k_off = float("nan")
                elif tau_ps > 0:
                    k_off = self.PS_PER_S / tau_ps
                else:
                    k_off = float("nan")

                attempt_time_s = self.attempt_time_ps / self.PS_PER_S
                if np.isfinite(k_off) and k_off > 0:
                    x = k_off * attempt_time_s
                    dG_apparent = -self.kBT * np.log(x) if x > 0 else float("nan")
                else:
                    dG_apparent = float("nan")

                # Recover original reflect_r in un-mirrored coordinates
                reflect_r_out = res["reflect_r_canon"]
                if res["mirrored"]:
                    reflect_r_out = 2.0 * res["start_r_input"] - reflect_r_out

                records.append(
                    {
                        "estimator": est_name,
                        "source": source,
                        "speed_nm_per_ps": speed,
                        "T_K": self.temperature,
                        "start_r_nm": _start_r,
                        "abs_r_nm": _abs_r,
                        "reflect_r_nm": reflect_r_out,
                        "tau_mfpt_ps": tau_ps,
                        "tau_mfpt_ns": tau_ps / 1e3,
                        "k_off_per_s": k_off,
                        "barrier_kjmol": res["barrier_kjmol"],
                        "barrier_r_nm": res["barrier_r_nm"],
                        "max_beta_dF": res["max_beta_dF"],
                        "attempt_time_ps": self.attempt_time_ps,
                        "dG_apparent_kjmol": dG_apparent,
                    }
                )

        return pd.DataFrame(records)

    def compute_per_path(
        self,
        per_path_results: pd.DataFrame,
        friction_df: pd.DataFrame,
        weights_by_estimator: dict,
    ) -> pd.DataFrame:
        """Compute Kramers k_off for each (estimator, speed, path) independently.

        Uses each path's own dG(r) profile for TS detection and MFPT integration.
        Uses per-speed (derivative) friction pooled across all paths at that speed.

        Returns one row per (estimator, speed, path) with ``source='per_path'``,
        plus one combined row per (estimator, speed) with ``source='per_path_combined'``
        computed as ``k_off_total = Σ_k p_eq(k) × k_off(k)`` (parallel pathways).
        """
        _PS_PER_S = 1e12
        fric_deriv = friction_df[friction_df["method"] == "derivative"]
        rows: list[dict] = []

        for (est_name, speed, path), grp in per_path_results.groupby(
            ["estimator", "speed", "path"]
        ):
            grp_s = grp.sort_values("r_coord").drop_duplicates("r_coord")
            r_arr  = grp_s["r_coord"].to_numpy(dtype=np.float64)
            dG_arr = grp_s["dG"].to_numpy(dtype=np.float64)
            if r_arr.size < 5:
                continue

            fric_sub = fric_deriv[
                (fric_deriv["estimator"] == est_name) & (fric_deriv["speed"] == speed)
            ]
            if fric_sub.empty:
                logger.warning(
                    f"[Kramers/per-path] No friction for estimator={est_name} "
                    f"speed={speed}; skipping path {path}"
                )
                continue

            pmf_mini = grp_s[["r_coord", "dG"]].copy()
            try:
                profile = KramersEstimator._merge_pmf_friction(pmf_mini, fric_sub)
            except Exception:
                continue
            if profile.empty:
                continue

            r_ts   = self.detect_ts(r_arr, dG_arr, self.kBT)
            start_r = float(r_arr[0])
            abs_r   = r_ts if r_ts is not None else float(r_arr[-1])

            try:
                res = KramersEstimator.kramers_mfpt(
                    profile, start_r, abs_r,
                    reflect_r=None, kBT=self.kBT, gamma_floor=self.gamma_floor,
                )
            except Exception as exc:
                logger.warning(
                    f"[Kramers/per-path] kramers_mfpt failed for "
                    f"{est_name}/{speed}/{path}: {exc}"
                )
                continue

            if float(res["barrier_kjmol"]) > self.barrier_warn_kjmol:
                logger.warning(
                    f"[Kramers/per-path] Large barrier: {float(res['barrier_kjmol']):.1f} kJ/mol "
                    f"(estimator={est_name}, speed={speed}, path={path}). "
                    f"Likely a sampling artifact (threshold: {self.barrier_warn_kjmol:.0f} kJ/mol)."
                )

            p_eq   = weights_by_estimator.get(est_name, {}).get(speed, {}).get(path, float("nan"))
            tau_ps = float(res["tau_ps"])
            rows.append({
                "estimator":     est_name,
                "speed":         speed,
                "path":          path,
                "source":        "per_path",
                "p_eq":          p_eq,
                "start_r_nm":    float(res["start_r_input"]),
                "abs_r_nm":      abs_r,
                "reflect_r_nm":  float(res["reflect_r_canon"]),
                "tau_mfpt_ps":   tau_ps,
                "tau_mfpt_ns":   tau_ps / 1e3,
                "k_off_per_s":   _PS_PER_S / tau_ps if tau_ps > 0 else float("nan"),
                "barrier_kjmol": float(res["barrier_kjmol"]),
                "barrier_r_nm":  float(res["barrier_r_nm"]),
                "max_beta_dF":   float(res["max_beta_dF"]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Combined: k_off_total = Σ p_eq(k) × k_off(k)  (parallel independent pathways)
        combined: list[dict] = []
        for (est_name, speed), grp in df.groupby(["estimator", "speed"]):
            valid = grp.dropna(subset=["k_off_per_s", "p_eq"])
            if valid.empty:
                continue
            p_sum      = float(valid["p_eq"].sum())
            k_combined = float((valid["p_eq"] * valid["k_off_per_s"]).sum() / p_sum) if p_sum > 0 else float("nan")
            tau_comb   = (_PS_PER_S / k_combined) if (k_combined > 0 and not np.isnan(k_combined)) else float("nan")
            combined.append({
                "estimator":     est_name,
                "speed":         speed,
                "path":          "combined_weighted",
                "source":        "per_path_combined",
                "p_eq":          p_sum,
                "start_r_nm":    float(valid["start_r_nm"].mean()),
                "abs_r_nm":      float("nan"),
                "reflect_r_nm":  float(valid["reflect_r_nm"].mean()),
                "tau_mfpt_ps":   tau_comb,
                "tau_mfpt_ns":   tau_comb / 1e3 if not np.isnan(tau_comb) else float("nan"),
                "k_off_per_s":   k_combined,
                "barrier_kjmol": float(valid["barrier_kjmol"].mean()),
                "barrier_r_nm":  float(valid["barrier_r_nm"].mean()),
                "max_beta_dF":   float(valid["max_beta_dF"].max()),
            })

        return pd.concat([df, pd.DataFrame(combined)], ignore_index=True)


ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    'jarzynski': JarzynskiEstimator,
    'cumulant': CumulantEstimator,
    'force': ForceEstimator,
}


def trim_results_by_n_samples_support(
    results: pd.DataFrame,
    min_samples: int = 3,
    min_support_ratio: float = 1.0,
    keep_prefix: bool = True,
    add_support_columns: bool = True,
) -> pd.DataFrame:
    """Trim low-support regions using per-step sample support.

    Works for:
    - fitted estimator tables (already containing ``n_samples``), and
    - raw trajectory tables (computes ``n_samples`` from unique ``trajname``
      per ``group_cols + step_col``).

    Parameters
    ----------
    results : pd.DataFrame
        Fitted estimator table (typically ``smd_data.results``) containing at
        least ``group_cols``, ``step_col`` and ``n_samples_col``.
    min_samples : int
        Absolute minimum number of replicas required at a point.
    min_support_ratio : float
        Relative support threshold, defined as ``n_samples / n_ref``.
        Default 1.0 trims at the first step where any replica stops (e.g.
        autostop), eliminating survivorship bias in the cumulant estimator.
    keep_prefix : bool
        If True, keep the contiguous prefix up to first failing point.
        If False, keep all points that pass support criteria.
    add_support_columns : bool
        If True, include ``support_frac``, ``support_ok`` and ``n_ref`` in output.

    Returns
    -------
    pd.DataFrame
        Trimmed results table.
    """
    group_cols= ["speed", "path"]
    step_col = "step"
    n_samples_col = "n_samples"
    traj_col = "trajname"
    
    if results is None or results.empty:
        return pd.DataFrame(columns=[] if results is None else results.columns)

    data = results.copy()
    if n_samples_col not in data.columns:
        if traj_col not in data.columns:
            raise ValueError(
                f"'{n_samples_col}' not found and '{traj_col}' is missing; cannot infer per-step support."
            )
        n_by_step = (
            data.groupby(list(group_cols) + [step_col], dropna=False)[traj_col]
            .nunique()
            .reset_index(name=n_samples_col)
        )
        data = data.merge(n_by_step, on=list(group_cols) + [step_col], how='left')

    kept_groups = []

    for _, group in data.groupby(list(group_cols), dropna=False):
        g = group.sort_values(step_col).copy()

        # Support is defined at step-level (same support for all rows in a step)
        nvals_by_step = (
            g.groupby(step_col, dropna=False)[n_samples_col]
            .max()
            .sort_index()
            .astype(float)
        )

        n_ref = float(nvals_by_step.max())

        if not np.isfinite(n_ref) or n_ref <= 0:
            support_frac_by_step = pd.Series(
                np.zeros(len(nvals_by_step), dtype=float),
                index=nvals_by_step.index,
            )
        else:
            support_frac_by_step = nvals_by_step / n_ref

        support_ok_by_step = (
            (nvals_by_step >= float(min_samples))
            & (support_frac_by_step >= float(min_support_ratio))
        )

        if keep_prefix:
            fail_idx = np.flatnonzero((~support_ok_by_step).to_numpy())
            if fail_idx.size > 0:
                keep_steps = nvals_by_step.index[:fail_idx[0]]
            else:
                keep_steps = nvals_by_step.index
        else:
            keep_steps = support_ok_by_step[support_ok_by_step].index

        g_keep = g[g[step_col].isin(keep_steps)].copy()

        if add_support_columns and not g_keep.empty:
            g_keep["support_frac"] = g_keep[step_col].map(support_frac_by_step.to_dict())
            g_keep["support_ok"] = g_keep[step_col].map(support_ok_by_step.to_dict())
            g_keep["n_ref"] = n_ref

        kept_groups.append(g_keep)

    if len(kept_groups) == 0:
        return results.iloc[0:0].copy()

    out = pd.concat(kept_groups, ignore_index=True)
    if step_col in out.columns:
        out = out.sort_values(list(group_cols) + [step_col]).reset_index(drop=True)
    return out


def calculate_weighted_pmf(
    smd_data: SMDData,
    weight_cols: list[str] = ['dG', 'Wdiss'],
    grid_col: str = "step",
    weights_by_estimator: dict | None = None,
) -> pd.DataFrame:
    """Build weighted PMFs over paths for each estimator and speed.

    Parameters
    ----------
    smd_data : SMDData
        Data object with fitted estimator results.
    weight_cols : list[str]
        Columns to weight.
    grid_col : str
        Grid column (default 'step').
    weights_by_estimator : dict
        Pre-computed weights ``{estimator: {speed: {path: p_eq}}}``.
        Obtain these from :meth:`SMDAnalysis.compute_p_eq`.

    Note
    ----
    The grid is restricted to the shortest path's last step so that all paths
    contribute at every grid point.  Without this restriction, when a shorter
    path terminates the p_eq renormalization denominator changes abruptly,
    producing a visible discontinuity in the weighted PMF.
    """

    results = smd_data.results.copy()
    if results is None or results.empty:
        return pd.DataFrame()

    if 'estimator' not in results.columns:
        raise ValueError("results must include an 'estimator' column for weighted PMF calculation.")

    if weights_by_estimator is None:
        raise ValueError(
            "weights_by_estimator is required. Compute it with SMDAnalysis.compute_p_eq()."
        )
    estimators = results['estimator'].dropna().unique()

    weighted_pmfs = []

    for estimator in estimators:
        df_est = results[results['estimator'] == estimator]
        est_weights = weights_by_estimator.get(estimator, {})

        for speed, speedg in df_est.groupby('speed'):
            speed_weights = est_weights.get(speed, {})
            if not speed_weights:
                continue

            path_last = speedg.groupby("path")[grid_col].max()
            max_grid_step = path_last.min()
            grid_vals = sorted(speedg.loc[speedg[grid_col] <= max_grid_step, grid_col].dropna().unique())
            
            rows = []

            for gval in grid_vals:
                vals_per_col = {col: [] for col in weight_cols}
                w_per_col = {col: [] for col in weight_cols}

                slice_g = speedg[speedg[grid_col] == gval]

                for path, g in slice_g.groupby('path'):
                    if path not in speed_weights:
                        continue

                    w = speed_weights[path]

                    for col in weight_cols:
                        if col in g.columns:
                            v = g[col].values[0]
                            if pd.notna(v):
                                vals_per_col[col].append(v)
                                w_per_col[col].append(w)

                if any(len(vals_per_col[col]) > 0 for col in weight_cols):
                    row = {
                        "step": gval,
                        "speed": speed,
                        "estimator": estimator,
                        # protocol-anchored coordinate
                        "r_coord": smd_data.protocol_grids[speed].loc[
                            smd_data.protocol_grids[speed]["step"] == gval,
                            "r_target_protocol"
                        ].values[0],
                    }

                    for col in weight_cols:
                        if len(vals_per_col[col]) > 0 and np.sum(w_per_col[col]) > 0:
                            row[f"{col}"] = (
                                np.sum(np.array(vals_per_col[col]) * np.array(w_per_col[col]))
                                / np.sum(w_per_col[col])
                            )
                        else:
                            row[f"{col}"] = np.nan

                    rows.append(row)

            if rows:
                weighted_pmfs.append(pd.DataFrame(rows))

    if not weighted_pmfs:
        return pd.DataFrame()

    return pd.concat(weighted_pmfs, ignore_index=True)

def recover_spring_constant(raw_data: pd.DataFrame) -> float:
    """Exact per-ligand SMD spring constant from the identity force = -k*(r_before - r_target)."""
    d = raw_data.dropna(subset=['force', 'r_before', 'r_target'])
    delta = (d['r_before'] - d['r_target']).to_numpy(dtype=float)
    f = d['force'].to_numpy(dtype=float)
    m = np.abs(delta) > 1e-12
    if not m.any():
        raise ValueError("recover_spring_constant: no rows with r_before != r_target")
    return float(np.median(-f[m] / delta[m]))

def monotone_z_filter(z: np.ndarray) -> np.ndarray:
    """Greedy mask: keep points whose z strictly exceeds the last kept z (drops folds/NaN)."""
    z = np.asarray(z, dtype=float)
    keep = np.zeros(len(z), dtype=bool); last = -np.inf
    for i, zi in enumerate(z):
        if np.isfinite(zi) and zi > last:
            keep[i] = True; last = zi
    return keep

def extrapolate_to_v0(
    results: pd.DataFrame,
    param: str = 'dG_weighted',
    speeds: list[float] | None = None,
    mixed_models: bool = False,
    min_speeds: int = 2,
) -> pd.DataFrame:
    """
    Extrapolate a single parameter to zero pulling speed (v → 0)
    using per-step regression across speeds.

    Because dx_per_move is fixed across all pulling speeds, the
    protocol step index maps to the same r_target regardless of
    speed.  The natural join key is therefore ``step``, not
    ``r_coord`` (which may carry tiny r₀ jitter between speeds).
    Only steps present at **all** included speeds are used.

    Parameters
    ----------
    results : pd.DataFrame
        Weighted PMF table produced by :func:`calculate_weighted_pmf`.
        Required columns: ``['step', 'r_coord', 'speed', 'estimator',
        param]``.
    param : str
        Column name to extrapolate, e.g. ``'dG_weighted'`` or
        ``'Wdiss_weighted'``.
    speeds : list[float] | None
        Optional subset of speeds to include.  If *None*, all speeds
        are used.
    mixed_models : bool
        If *True*, fit a mixed-effects linear model (random
        intercept + slope per step); otherwise simple OLS per step.
    min_speeds : int
        Minimum number of distinct speeds required at a step to include
        it in the extrapolation.  Default 2 (need at least two points
        for a linear fit).  Setting this lower than the total number of
        speeds lets the extrapolated PMF extend beyond where the fastest
        speed's profile ends.

    Returns
    -------
    pd.DataFrame
        One row per common step per estimator, with columns:
        ``r_coord``, ``step``, ``estimator``, ``{param}`` (v→0
        intercept), ``{param}_slope``, ``{param}_se``,
        ``{param}_slope_se``, ``R2``, ``n_speeds``, ``model``,
        ``speed`` (= 0.0).

        This output is directly compatible with
        :func:`Diagnostics.plot_extrapolated_param`.

    Note
    ----
    **Step selection trade-off (min_speeds):** requiring all speeds (equivalent
    to setting ``min_speeds`` equal to the total number of speeds) cuts the
    extrapolated PMF to whichever speed has the shortest profile.  Allowing
    fewer speeds via a lower ``min_speeds`` lets the tail extend further at the
    cost of a less constrained fit (fewer data points per step).

    **Edge trimming:** after the per-step regressions, leading and trailing steps
    where fewer than the maximum number of available speeds contributed are
    stripped.  Edge rows with only ``min_speeds`` data points have R²=1.0
    trivially (zero degrees of freedom) and produce unconstrained intercept
    spikes.
    """

    df = results.copy()

    if param not in df.columns:
        raise ValueError(
            f"Column '{param}' not found in results. "
            f"Available: {list(df.columns)}"
        )

    # Optional speed filtering
    if speeds is not None:
        speeds = [float(s) for s in speeds]
        df = df[df['speed'].isin(speeds)]

    if df['speed'].nunique() < 2:
        logger.error("Need at least two distinct speeds for extrapolation.")
        return pd.DataFrame()
    
    out_rows: list[dict] = []

    for estimator, est_group in df.groupby('estimator'):
        sub = est_group.dropna(subset=[param])

        step_speed_counts = sub.groupby('step')['speed'].nunique()
        common_steps = step_speed_counts[step_speed_counts >= min_speeds].index
        sub = sub[sub['step'].isin(common_steps)].copy()

        if sub.empty:
            continue

        # Mean r_coord per step across speeds (should be nearly identical)
        step_r_coord = sub.groupby('step')['r_coord'].mean()

        if mixed_models:
            model = smf.mixedlm(
                f"{param} ~ speed",
                sub,
                groups=sub["step"],
                re_formula="~speed",
            )
            res = model.fit(reml=False)

            fe_int = res.fe_params["Intercept"]
            fe_slope = res.fe_params["speed"]

            for step, re in res.random_effects.items():
                out_rows.append({
                    "step": step,
                    "r_coord": step_r_coord.loc[step],
                    "estimator": estimator,
                    param: fe_int + re.get("Intercept", 0.0),
                    f"{param}_slope": fe_slope + re.get("speed", 0.0),
                    f"{param}_se": np.nan,
                    f"{param}_slope_se": np.nan,
                    "R2": np.nan,
                    "n_speeds": n_speeds_total,
                    "model": "mixedlm",
                    "speed": 0.0,
                })
        else:
            for step, g in sub.groupby("step"):
                if g['speed'].nunique() < 2:
                    continue

                lr = linregress(g['speed'].values, g[param].values)

                out_rows.append({
                    "step": step,
                    "r_coord": step_r_coord.loc[step],
                    "estimator": estimator,
                    param: lr.intercept,
                    f"{param}_slope": lr.slope,
                    f"{param}_se": lr.intercept_stderr,
                    f"{param}_slope_se": lr.stderr,
                    "R2": lr.rvalue ** 2,
                    "n_speeds": g['speed'].nunique(),
                    "model": "linear",
                    "speed": 0.0,
                })

    v0_df = pd.DataFrame(out_rows)
    if not v0_df.empty:
        v0_df = v0_df.sort_values(['estimator', 'step']).reset_index(drop=True)
        trimmed = []
        for _est, eg in v0_df.groupby('estimator', sort=False):
            eg = eg.sort_values('step').reset_index(drop=True)
            max_n = int(eg['n_speeds'].max())
            full_support = eg.index[eg['n_speeds'] >= max_n]
            if len(full_support) > 0:
                trimmed.append(eg.loc[full_support[0]:full_support[-1]])
            else:
                trimmed.append(eg)
        v0_df = pd.concat(trimmed).sort_values(['estimator', 'step']).reset_index(drop=True)
    else:
        logger.error("No valid data for extrapolation to v=0. Returning empty DataFrame.")
    return v0_df