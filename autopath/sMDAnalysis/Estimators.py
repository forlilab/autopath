import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import statsmodels.formula.api as smf
from scipy import special
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from autopath.sMDAnalysis.SMDData import SMDData

import logging
logger = logging.getLogger("autopath.sMDAnalysis.Estimators")

def _fit_gmm_to_work_values(
    raw_work: np.ndarray,
    max_components: int = 3,
    max_iter: int = 100,
    covariance_type: str = "diag",
    random_state: int = 42,
):
    """Fit a Gaussian mixture to 1D work samples and return best model by BIC."""
    raw_work = np.asarray(raw_work, dtype=float).reshape(-1, 1)
    n_samples = raw_work.shape[0]

    if n_samples < 2:
        return None

    k_max = min(max_components, n_samples)
    models = []
    scores = []

    for n_comp in range(1, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                max_iter=max_iter,
                covariance_type=covariance_type,
                random_state=random_state,
                init_params="k-means++",
            )
            gmm.fit(raw_work)
            models.append(gmm)
            scores.append(gmm.bic(raw_work))
        except Exception:
            continue

    if len(models) == 0:
        return None

    best_model = models[int(np.argmin(scores))]
    variances = best_model.covariances_.reshape(best_model.n_components, -1).flatten()

    return {
        "weights": np.asarray(best_model.weights_, dtype=float),
        "means": np.asarray(best_model.means_, dtype=float).ravel(),
        "variances": np.asarray(variances, dtype=float),
        "n_components": int(best_model.n_components),
        "bic": float(np.min(scores)),
    }


def _fit_gmm_and_get_components(
    raw_W: np.ndarray,
    max_components: int = 3,
    max_iter: int = 100,
    covariance_type: str = "diag",
    random_state: int = 42,
    min_samples: int = 2,
    weight_cutoff: float = 0.0,
    **kwargs,
) -> dict | None:
    """Fit GMM to 1-D work samples, normalize weights, and return components.

    Returns a dict with keys: w, mu, sig2, Wmean, Wvar,
    gmm_n_components, gmm_bic — or None on failure.
    """
    raw_W = np.asarray(raw_W, dtype=float)
    if raw_W.size < min_samples:
        return None

    gmm_dict = _fit_gmm_to_work_values(
        raw_work=raw_W,
        max_components=max_components,
        max_iter=max_iter,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    if gmm_dict is None:
        return None

    w = np.asarray(gmm_dict["weights"], dtype=float)
    mu = np.asarray(gmm_dict["means"], dtype=float)
    sig2 = np.asarray(gmm_dict["variances"], dtype=float)

    if np.any(w < weight_cutoff):
        w = np.where(w < weight_cutoff, 0.0, w)
    wsum = w.sum()
    if wsum <= 0.0:
        return None
    w = w / wsum

    Wmean = float(np.dot(w, mu))
    Wvar = float(np.dot(w, sig2 + (mu - Wmean) ** 2))

    return {
        "w": w,
        "mu": mu,
        "sig2": sig2,
        "Wmean": Wmean,
        "Wvar": Wvar,
        "gmm_n_components": gmm_dict["n_components"],
        "gmm_bic": gmm_dict["bic"],
    }


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
        Wvar = float(raw_W.var())
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


class JarzynskiGMMEstimator(BaseEstimator):

    def __init__(
        self,
        max_components: int = 3,
        max_iter: int = 100,
        covariance_type: str = "diag",
        random_state: int = 42,
        min_samples: int = 3,
        weight_cutoff: float = 0.0,
    ):
        self.max_components = max_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.min_samples = min_samples
        self.weight_cutoff = weight_cutoff

    @property
    def name(self):
        return "jarzynski_gmm"

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        comp = _fit_gmm_and_get_components(raw_W, **kwargs)
        if comp is None:
            return None
        log_terms = -beta * comp["mu"] + 0.5 * (beta ** 2) * comp["sig2"]
        log_Z = special.logsumexp(log_terms, b=comp["w"])
        dG = float(-(1.0 / beta) * log_Z)
        return {
            "Wmean": comp["Wmean"],
            "Wvar": comp["Wvar"],
            "dG": dG,
            "Wdiss": comp["Wmean"] - dG,
            "gmm_n_components": comp["gmm_n_components"],
            "gmm_bic": comp["gmm_bic"],
        }

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ["step", "speed", "path"]
        results = []

        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]["step"] == step,
                "r_target_protocol",
            ].values[0]

            raw_W = group["work"].astype(float).values
            result = self.estimate_dG(
                raw_W, smd_data.beta,
                max_components=self.max_components,
                max_iter=self.max_iter,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                min_samples=self.min_samples,
                weight_cutoff=self.weight_cutoff,
            )
            if result is None:
                continue

            results.append({
                "r_coord": r_coord,
                "step": step,
                "speed": speed,
                "path": path,
                "n_samples": raw_W.size,
                **result,
            })

        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data


class CumulantGMMEstimator(BaseEstimator):

    def __init__(
        self,
        max_components: int = 3,
        max_iter: int = 100,
        covariance_type: str = "diag",
        random_state: int = 42,
        min_samples: int = 3,
        weight_cutoff: float = 0.0,
    ):
        self.max_components = max_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.min_samples = min_samples
        self.weight_cutoff = weight_cutoff

    @property
    def name(self):
        return "cumulant_gmm"

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        comp = _fit_gmm_and_get_components(raw_W, **kwargs)
        if comp is None:
            return None
        dG = comp["Wmean"] - (beta * comp["Wvar"]) / 2.0
        return {
            "Wmean": comp["Wmean"],
            "Wvar": comp["Wvar"],
            "dG": dG,
            "Wdiss": comp["Wmean"] - dG,
            "gmm_n_components": comp["gmm_n_components"],
            "gmm_bic": comp["gmm_bic"],
        }

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ["step", "speed", "path"]
        results = []

        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]["step"] == step,
                "r_target_protocol",
            ].values[0]

            raw_W = group["work"].astype(float).values
            result = self.estimate_dG(
                raw_W, smd_data.beta,
                max_components=self.max_components,
                max_iter=self.max_iter,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                min_samples=self.min_samples,
                weight_cutoff=self.weight_cutoff,
            )
            if result is None:
                continue

            results.append({
                "r_coord": r_coord,
                "step": step,
                "speed": speed,
                "path": path,
                "n_samples": raw_W.size,
                **result,
            })

        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data


class CumulantGMMComponentwiseEstimator(BaseEstimator):

    def __init__(
        self,
        max_components: int = 3,
        max_iter: int = 100,
        covariance_type: str = "diag",
        random_state: int = 42,
        min_samples: int = 3,
        weight_cutoff: float = 0.0,
    ):
        self.max_components = max_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.min_samples = min_samples
        self.weight_cutoff = weight_cutoff

    @property
    def name(self):
        return "cumulant_gmm_componentwise"

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        comp = _fit_gmm_and_get_components(raw_W, **kwargs)
        if comp is None:
            return None
        dG_k = comp["mu"] - 0.5 * beta * comp["sig2"]
        dG = float(np.dot(comp["w"], dG_k))
        return {
            "Wmean": comp["Wmean"],
            "Wvar": comp["Wvar"],
            "dG": dG,
            "Wdiss": comp["Wmean"] - dG,
            "gmm_n_components": comp["gmm_n_components"],
            "gmm_bic": comp["gmm_bic"],
        }

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ["step", "speed", "path"]
        results = []

        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]["step"] == step,
                "r_target_protocol",
            ].values[0]

            raw_W = group["work"].astype(float).values
            result = self.estimate_dG(
                raw_W, smd_data.beta,
                max_components=self.max_components,
                max_iter=self.max_iter,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                min_samples=self.min_samples,
                weight_cutoff=self.weight_cutoff,
            )
            if result is None:
                continue

            results.append({
                "r_coord": r_coord,
                "step": step,
                "speed": speed,
                "path": path,
                "n_samples": raw_W.size,
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

            gamma = self._compute_gamma_derivative(r, wdiss, speed)
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


ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    'jarzynski': JarzynskiEstimator,
    'cumulant': CumulantEstimator,
    'jarzynski_gmm': JarzynskiGMMEstimator,
    'cumulant_gmm': CumulantGMMEstimator,
    'cumulant_gmm_componentwise': CumulantGMMComponentwiseEstimator,
}


def trim_results_by_n_samples_support(
    results: pd.DataFrame,
    min_samples: int = 3,
    min_support_ratio: float = 0.7,
    keep_prefix: bool = True,
    add_support_columns: bool = False,
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
    reference = "max"
    
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
) -> pd.DataFrame:

    results = smd_data.results.copy()
    if results is None or results.empty:
        return pd.DataFrame()

    if 'estimator' not in results.columns:
        raise ValueError("results must include an 'estimator' column for weighted PMF calculation.")

    # Build p_eq independently for each estimator so each PMF is weighted
    # with its own thermodynamic model.
    weights_by_estimator = smd_data.get_p_eq(
        byspeed=True,
        results=results,
        per_estimator=True,
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

            # grid_vals = sorted(speedg[grid_col].dropna().unique())
            # restrict to steps where ALL paths have data (common support)
            path_last = speedg.groupby("path")[grid_col].max()
            max_common_step = path_last.min()
            grid_vals = sorted(speedg.loc[speedg[grid_col] <= max_common_step, grid_col].dropna().unique())
            # print(f'Speed {speed}: using {len(grid_vals)} common grid points up to step {max_common_step} for weighted PMF calculation.')
            
            rows = []

            for gval in grid_vals:
                vals_per_col = {col: [] for col in weight_cols}
                w_per_col = {col: [] for col in weight_cols}

                slice_g = speedg[speedg[grid_col] == gval]

                for path, g in slice_g.groupby('path'):
                    if path not in speed_weights:
                        continue
                    
                    # # skip terminal or undersampled points
                    # if g['n_samples'].iloc[0] < 3:
                    #     continue
                    
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
    
def extrapolate_to_v0(
    results: pd.DataFrame,
    param: str = 'dG_weighted',
    speeds: list[float] | None = None,
    mixed_models: bool = False,
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

        # Keep only steps that appear in ALL speeds (common support)
        step_speed_counts = sub.groupby('step')['speed'].nunique()
        n_speeds_total = sub['speed'].nunique()
        common_steps = step_speed_counts[step_speed_counts == n_speeds_total].index
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
    else:
        logger.error("No valid data for extrapolation to v=0. Returning empty DataFrame.")
    return v0_df