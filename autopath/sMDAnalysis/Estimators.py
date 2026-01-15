import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import statsmodels.formula.api as smf
from scipy import special
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline

from autopath.sMDAnalysis.SMDData import SMDData

class BaseEstimator(ABC):
    """
    Base class for all estimators in SMDAnalysis.
    """

    def __init__(self):
        pass
    
    @abstractmethod
    def fit_transform(self, smd_data: SMDData):
        """Fit the estimator to the provided SMD data."""
        pass
    
class JarzynskiEstimator(BaseEstimator):
    
    @property
    def name(self):
        return 'jarzynski'
    
    def fit_transform(self, smd_data: SMDData) -> SMDData:

        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            
            # protocol-anchored coordinate
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]
                        
            # print(f'analyzing coord={coord:.2f}, r_bin={r_coord:.2f}, speed={speed:.5f}, path={path} with {len(group)} points.')
            raw_W = group['work'].astype(float).values

            Wmean = raw_W.mean()

            # plain Jarzynski on the samples
            dG_Jarzynski = -(1.0/smd_data.beta) * (
                special.logsumexp(-smd_data.beta * raw_W) - np.log(raw_W.size)
            )
            
            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                'Wmean': Wmean,
                'Wdiss': Wmean - dG_Jarzynski,
                'dG': dG_Jarzynski
            })
        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data
    
class CumulantEstimator(BaseEstimator):
    
    @property
    def name(self):
        return 'cumulant'
    
    def fit_transform(self, smd_data: SMDData) -> SMDData:

        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            
            # protocol-anchored coordinate
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]
                  
            # print(f'analyzing coord={coord:.2f}, r_bin={r_coord:.2f}, speed={speed:.5f}, path={path} with {len(group)} points.')
            raw_W = group['work'].astype(float).values

            Wmean = raw_W.mean()
            Wvar = raw_W.var()

            # cumulant expansion to 2nd order
            dG_cumulant = Wmean - (smd_data.beta * Wvar) / 2.0
            
            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                'Wmean': Wmean,
                'Wdiss': Wmean - dG_cumulant,
                'dG': dG_cumulant
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
    param_cols: list[str] = ['Wdiss_weighted', 'dG_weighted'],
    speeds: list[float] | None = None,
    mixed_models: bool = False,
) -> pd.DataFrame:
    """
    #FIXME this wont work because now r_coord doesnt match across speeds
    would need to bin or interpolate first or do windowed regression per r_coord
    
    Extrapolate parameters to zero pulling speed (v -> 0)
    using per-r_coord regression across speeds.

    Expected columns in `results`:
    ['r_coord', <param_cols>, 'estimator', 'speed']
    """

    df = results.copy()

    # Optional speed filtering (e.g. keep only slow speeds)
    if speeds is not None:
        speeds = [float(s) for s in speeds]
        df = df[df['speed'].isin(speeds)]

    if df['speed'].nunique() < 2:
        raise ValueError("Need at least two distinct speeds for extrapolation.")

    out_rows = []

    # Loop over estimator and parameter independently
    for estimator, est_group in df.groupby('estimator'):
        for param in param_cols:

            sub = est_group.dropna(subset=[param])

            if mixed_models:
                # Mixed-effects: random intercept & slope per r_coord
                model = smf.mixedlm(
                    f"{param} ~ speed",
                    sub,
                    groups=sub["r_coord"],
                    re_formula="~speed",
                )
                res = model.fit(reml=False)

                fe_int = res.fe_params["Intercept"]
                fe_slope = res.fe_params["speed"]

                for r_coord, re in res.random_effects.items():
                    out_rows.append({
                        "r_coord": r_coord,
                        "estimator": estimator,
                        "param": param,
                        "v0_value": fe_int + re.get("Intercept", 0.0),
                        "slope": fe_slope + re.get("speed", 0.0),
                        "model": "mixedlm",
                        "n_speeds": sub[sub["r_coord"] == r_coord]["speed"].nunique(),
                        "R2": np.nan,  # not well-defined for MixedLM
                        "speed": 0.0,
                    })

            else:
                # Simple linear regression per r_coord
                for r_coord, g in sub.groupby("r_coord"):
                    if g["speed"].nunique() < 2:
                        continue

                    lr = linregress(g["speed"].values, g[param].values)

                    out_rows.append({
                        "r_coord": r_coord,
                        "estimator": estimator,
                        "param": param,
                        "v0_value": lr.intercept,
                        "slope": lr.slope,
                        "intercept_se": lr.intercept_stderr,
                        "slope_se": lr.stderr,
                        "R2": lr.rvalue**2,
                        "n_speeds": g["speed"].nunique(),
                        "model": "linear",
                        "speed": 0.0,
                    })

    v0_df = pd.DataFrame(out_rows)

    return v0_df