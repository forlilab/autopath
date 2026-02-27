import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import statsmodels.formula.api as smf
from scipy import special
from scipy.stats import linregress
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
    @property
    def name(self):
        return 'friction'
    
    def fit_transform(self, smd_data: SMDData):
        return super().fit_transform(smd_data)


ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    'jarzynski': JarzynskiEstimator,
    'cumulant': CumulantEstimator,
    'jarzynski_gmm': JarzynskiGMMEstimator,
    'cumulant_gmm': CumulantGMMEstimator,
    'cumulant_gmm_componentwise': CumulantGMMComponentwiseEstimator,
}


def calculate_weighted_pmf(
    smd_data: SMDData,
    weight_cols: list[str] = ['dG', 'Wdiss'],
    grid_col: str = "step",
) -> pd.DataFrame:

    results = smd_data.results.copy()
    weights = smd_data.get_p_eq(byspeed=True, results=results)
    estimators = results['estimator'].unique()

    weighted_pmfs = []

    for estimator in estimators:
        df_est = results[results['estimator'] == estimator]

        for speed, speedg in df_est.groupby('speed'):
            speed_weights = weights[speed]

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
                            row[f"{col}_weighted"] = (
                                np.sum(np.array(vals_per_col[col]) * np.array(w_per_col[col]))
                                / np.sum(w_per_col[col])
                            )
                        else:
                            row[f"{col}_weighted"] = np.nan

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
        raise ValueError("Need at least two distinct speeds for extrapolation.")

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