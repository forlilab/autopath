import os
import logging
import numpy as np
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")
plt.rcParams["savefig.facecolor"] = 'white'
plt.rcParams["savefig.edgecolor"] = 'white'
plt.rcParams["axes.facecolor"] = 'white'

from autopath.sMDAnalysis.Estimators import _find_pmf_peak

logger = logging.getLogger("autopath")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_colvar(out_dir: str = None, colvar_name: str = None):
    sys_name = out_dir.split("/")[0]
    files = glob(f"{out_dir}/COLVAR_*")
    data = []
    for f in files:
        walker_name = f.split("/")[2].split(".")[0]
        np_data = np.load(f)
        df = pd.DataFrame(np_data, columns=[colvar_name])
        df["walker"] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data, x=data.index, y=colvar_name, hue="walker")
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.ylabel(colvar_name)
    plt.xlabel("Frame #")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_COLVAR.png")
    plt.close()
    return


def plot_bias(out_dir: str = None, x_min: float = None, x_max: float = None, grid_points: int = None, colvar_name: str = None):
    sys_name = out_dir.split("/")[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    files = glob(f"{out_dir}/bias_*")
    data = []
    for f in files:
        walker_name = f.split("/")[2].split(".")[0]
        np_data = np.load(f)
        np_data = np_data * 0.239006  # KJ to Kcal
        df = pd.DataFrame(np_data, columns=["bias"])
        df.index = axis_values
        df["walker"] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)

    plt.figure(figsize=(10, 4))
    sns.lineplot(data, x=data.index, y="bias", hue="walker")
    plt.ylabel("Bias (Kcal/mol)")
    plt.xlabel(colvar_name)
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.title(f"Bias deposited - {sys_name}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_BIAS.png")
    plt.close()
    return


def plot_FE(out_dir, x_min, x_max, grid_points, colvar_name):
    """Plot biased FE profiles (raw FE_*.npy, excluding _rw files).

    Each file is a sequential walker that also accumulated all previous walkers'
    bias (shared biasDir), so later files are more converged estimates.  All
    walker traces are shown together to visualise convergence.
    """
    sys_name = out_dir.split("/")[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    files = sorted(f for f in glob(f"{out_dir}/FE_*.npy") if not f.endswith("_rw.npy"))
    if not files:
        logger.warning(f"plot_FE: no FE_*.npy files found in {out_dir}")
        return

    data = []
    for f in files:
        walker_name = os.path.splitext(os.path.basename(f))[0]
        np_data = np.load(f) * 0.239006  # kJ/mol → kcal/mol
        df = pd.DataFrame(np_data, columns=["FE"])
        df.index = axis_values
        df["walker"] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)
    plt.figure(figsize=(10, 4))
    sns.lineplot(data, x=data.index, y="FE", hue="walker")
    plt.ylabel("FE (kcal/mol)")
    plt.xlabel(colvar_name)
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="10")
    plt.title(f"Free Energy (biased) - {sys_name}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_FE.png")
    plt.close()
    return


def plot_FE_rw(out_dir, x_min, x_max, grid_points, colvar_name):
    """Plot standard-state-corrected FE profiles (FE_*_rw.npy).

    Each _rw file is the corresponding biased FE normalised so the unbound plateau
    is zero, then shifted by the Limongelli 2013 standard-state correction so that
    the bound-state minimum directly reads as ΔG°_b.  Later files are more
    converged; the last file is the best current estimate and is highlighted.
    """
    sys_name = out_dir.split("/")[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    files = sorted(glob(f"{out_dir}/FE_*_rw.npy"))
    if not files:
        return

    data = []
    for f in files:
        walker_name = os.path.splitext(os.path.basename(f))[0]
        np_data = np.load(f) * 0.239006  # kJ/mol → kcal/mol
        df = pd.DataFrame(np_data, columns=["FE"])
        df.index = axis_values
        df["walker"] = walker_name
        data.append(df)

    last_fe_kcal = np.load(files[-1]) * 0.239006
    dG_std_kcal = float(last_fe_kcal.min())
    dG_std_kj = dG_std_kcal * 4.184
    pKd = -dG_std_kj / (8.314e-3 * 298.15 * np.log(10))

    data = pd.concat(data, axis=0)
    plt.figure(figsize=(10, 4))
    sns.lineplot(data, x=data.index, y="FE", hue="walker")
    plt.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    plt.ylabel("ΔG°_b (kcal/mol)")
    plt.xlabel(colvar_name)
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="10")
    plt.title(f"FE standard-state corrected - {sys_name}\n"
              f"ΔG°_b (last walker) ≈ {dG_std_kcal:.1f} kcal/mol | pKd ≈ {pKd:.2f}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_FE_rw.png")
    plt.close()
    return


def plot_FE_2D(
    out_dir, x_min, x_max, x_grid_points, xCV_name, y_min, y_max, y_grid_points, yCV_name
):
    sys_name = out_dir.split("/")[0]

    x_values = np.linspace(x_min, x_max, x_grid_points)
    x_values = [round(i, 2) for i in x_values]

    y_values = np.linspace(y_min, y_max, y_grid_points)
    y_values = [round(i, 2) for i in y_values]

    files_fe = glob(f"{out_dir}/FE_*.npy")
    for i, f in enumerate(files_fe):
        walker_name = os.path.splitext(os.path.basename(f))[0]

        np_data = np.load(f)
        np_data = np_data * 0.239006  # KJ to Kcal
        df = pd.DataFrame(np_data)

        df.index, df.columns = y_values, x_values

        sns.heatmap(df, cmap="Spectral")
        plt.title(f"Metadynamics - {sys_name} - {walker_name}", fontsize=12)
        plt.xlabel(xCV_name)
        plt.ylabel(yCV_name)
        plt.title(f"Free Energy - {sys_name}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{sys_name}_{walker_name}.png")
        plt.close()

    return


def plot_colvar_2D(out_dir, xCV_name, yCV_name):
    sys_name = out_dir.split("/")[0]
    files = glob(f"{out_dir}/COLVAR_*")
    data = []
    for f in files:
        walker_name = os.path.basename(f).split(".")[0]
        np_data = np.load(f)
        df = pd.DataFrame(np_data, columns=[xCV_name, yCV_name])
        df["walker"] = walker_name
        data.append(df)
    data = pd.concat(data, axis=0)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data, x=data.index, y=xCV_name, hue="walker")
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.xlabel('Step #')
    plt.ylabel(xCV_name)
    plt.title(f'{xCV_name} vs. Step # - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_{xCV_name}_COLVAR.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data, x=data.index, y=yCV_name, hue="walker")
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.xlabel('Step #')
    plt.ylabel(yCV_name)
    plt.title(f'{yCV_name} vs. Step # - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_{yCV_name}_COLVAR.png")
    plt.close()

    return


# ---------------------------------------------------------------------------
# Free energy correction
# ---------------------------------------------------------------------------

def correct_fe_for_funnel(
    fe_files: "str | list[str] | np.ndarray",
    funnel_params: dict,
    temperature: float = 298.15,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
    n_plateau_points: int = 10,
    C0_per_A3: float = 1.0 / 1660.0,
) -> dict:
    """Apply the Limongelli 2013 standard-state correction to a funnel metadynamics FE profile.

    Accepts a single FE_*.npy path, a numpy array (kJ/mol), or a list of paths.
    When a list is given the last file is used, as each sequential walker already
    contains all previous walkers' accumulated bias (shared biasDir).

    The function normalises the profile so the unbound-state plateau is 0 in the biased
    simulation, then applies the standard-state correction (Eq. 3, Limongelli 2013):

        FE_rw = FE_norm − correction
        correction = kT × ln(πR²_cyl × C°)          [typically negative]

    so the bound-state minimum of ``FE_rw`` directly reads as ΔG°_b and the unbound
    plateau sits at |correction| above zero.

    Parameters
    ----------
    fe_files : str, list of str, or np.ndarray
        Path(s) to FE_*.npy (kJ/mol). If a list, the last file is used (most converged,
        since each sequential walker accumulates all previous bias). A numpy array can
        be passed directly.
    funnel_params : dict
        From ``generate_funnel_parameters_from_trajectory()``. Must contain
        ``"R_cylinder"``, ``"z_cc"``, and ``"z_max"`` as OpenMM Quantities in angstroms.
    temperature : float
        Kelvin (default 298.15 K).
    grid_min, grid_max : float
        CV grid bounds — default 0.0/1.0 for the PathCV progress coordinate.
    n_plateau_points : int
        Last N grid points averaged to define the unbound-state reference.
    C0_per_A3 : float
        Standard concentration in Å⁻³ (default 1/1660 = 1 M).

    Returns
    -------
    dict with keys:
        fe_corrected        — numpy array (kJ/mol): FE_norm − correction; save as _rw.npy
        dG_bind_sim_kj_mol  — min(FE_norm); binding ΔG from simulation (negative)
        correction_kj_mol   — kT·ln(πR²_cyl·C°); negative for typical R values
        dG_bind_std_kj_mol  — min(FE_rw) = ΔG°_b (standard-state binding free energy)
        R_cylinder_ang      — cylinder radius used (Å)
        pKd                 — predicted pKd
        cv_at_minimum       — PathCV progress at the FE minimum
        n_fe_files          — number of FE files/arrays combined
        temperature_K       — temperature used
    """
    import math
    from openmm import unit as openmmunit

    kT = 8.314e-3 * temperature  # kJ/mol

    if isinstance(fe_files, np.ndarray):
        fe = fe_files.astype(float)
        n_files = 1
    else:
        if isinstance(fe_files, str):
            fe_files = [fe_files]
        fe = np.load(fe_files[-1]).astype(float)
        n_files = len(fe_files)

    cv = np.linspace(grid_min, grid_max, len(fe))

    tail = fe[int(0.7 * len(fe)):]
    n = min(n_plateau_points, len(tail))
    if len(tail) > n:
        stds = [float(tail[i : i + n].std()) for i in range(len(tail) - n + 1)]
        best_start = int(np.argmin(stds))
        plateau_val = float(tail[best_start : best_start + n].mean())
    else:
        plateau_val = float(tail.mean())

    fe_max = float(fe.max())
    unsampled_gap = fe_max - plateau_val
    if unsampled_gap < 10.0:
        raise ValueError(
            f"Unbound state not sampled: plateau ({plateau_val:.1f} kJ/mol) is only "
            f"{unsampled_gap:.1f} kJ/mol below the FE maximum ({fe_max:.1f} kJ/mol). "
            "The walker did not reach the unbound end of the CV; ΔG would be meaningless."
        )

    fe_norm = fe - plateau_val

    min_idx = int(np.argmin(fe_norm))
    dG_bind_sim = float(fe_norm[min_idx])
    cv_at_min = float(cv[min_idx])

    R_cyl_ang = float(funnel_params["R_cylinder"].value_in_unit(openmmunit.angstrom))
    z_cc_ang = float(funnel_params["z_cc"].value_in_unit(openmmunit.angstrom))
    z_max_ang = float(funnel_params["z_max"].value_in_unit(openmmunit.angstrom))
    L_cyl_ang = z_max_ang - z_cc_ang

    V_cyl = math.pi * R_cyl_ang ** 2 * L_cyl_ang  # Å³
    correction = kT * math.log(V_cyl * C0_per_A3)  # kJ/mol

    fe_rw = fe_norm - correction

    dG_bind_std = float(fe_rw.min())
    pKd = -dG_bind_std / (kT * math.log(10))

    logger.info(
        f"Funnel correction: ΔG_sim={dG_bind_sim:.2f} kJ/mol, "
        f"correction={correction:.2f} kJ/mol "
        f"(R_cyl={R_cyl_ang:.1f} Å, L_cyl={L_cyl_ang:.1f} Å, V_cyl={V_cyl:.1f} Å³), "
        f"ΔG°_b={dG_bind_std:.2f} kJ/mol, pKd={pKd:.2f}"
    )

    return {
        "fe_corrected": fe_rw,
        "dG_bind_sim_kj_mol": dG_bind_sim,
        "correction_kj_mol": correction,
        "dG_bind_std_kj_mol": dG_bind_std,
        "R_cylinder_ang": R_cyl_ang,
        "z_cc_ang": z_cc_ang,
        "z_max_ang": z_max_ang,
        "L_cylinder_ang": L_cyl_ang,
        "V_cylinder_ang3": V_cyl,
        "pKd": pKd,
        "cv_at_minimum": cv_at_min,
        "n_fe_files": n_files,
        "temperature_K": temperature,
    }


# ---------------------------------------------------------------------------
# sMD preseed
# ---------------------------------------------------------------------------

def write_metad_preseed(
    smd_analysis_outdir: str,
    milestone_com_distances: list,
    bias_dir: str,
    gamma: float,
    temperature: float = 298.15,
    alpha: float = 0.3,
    grid_points: int = 125,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
    speed: float = None,
    estimator: str = 'jarzynski',
) -> str | None:
    """Pre-seed the metadynamics bias surface from a sMD PMF profile.

    Writes ``bias_0_0.npy`` into ``bias_dir`` so all walkers load it
    via ``_syncWithDisk()`` at initialisation (before any hills are deposited).

    The preseed is a fraction (``alpha``) of the well-tempered-metadynamics bias
    that would be deposited at full convergence:
        V_preseed(s) = alpha × (gamma-1)/gamma × (ΔG_unbind − F_sMD(r(s)))
    where F_sMD is the sMD PMF in kJ/mol (0 at bound state) mapped onto the
    path-CV progress coordinate s ∈ [0,1] using the milestone COM distances.

    The preseed is truncated at the transition state (PMF peak): grid points
    beyond s_TS are set to zero so metadynamics explores the post-barrier region
    without bias from the sMD profile.

    Parameters
    ----------
    smd_analysis_outdir : str
        sMD analysis output directory (contains ``mixture_pmfs.csv`` and
        optionally ``dG_extrapolated.csv``).
    milestone_com_distances : list of float
        COM distances (nm) for each milestone, in ascending order from
        bound (s=0) to unbound (s=1).  These define the r_coord → s mapping.
    bias_dir : str
        MetadynamicsMD out_dir (same directory passed to Metadynamics as biasDir).
    gamma : float
        Well-tempered bias factor γ (= bias_factor parameter of MetadynamicsMD).
    temperature : float
        Temperature in K (default 298.15).
    alpha : float
        Fraction of the converged bias to pre-seed (default 0.3; values 0.1–0.5
        are reasonable — too large risks over-constraining the walker).
    grid_points : int
        Must match the CVSpec.grid_points used in the run (default 125).
    grid_min, grid_max : float
        Path CV grid bounds (default 0.0/1.0 for PathCV progress).
    speed : float or None
        sMD pulling speed to use.  ``None`` (default) picks the minimum available:
        ``dG_extrapolated.csv`` (speed=0, v→0) if present, otherwise the slowest
        speed in ``mixture_pmfs.csv``.  Pass an explicit ``0.0`` to force the
        extrapolated PMF, or a non-zero float to select a specific pulling speed.
    estimator : str
        Estimator to use (``'cumulant'`` or ``'jarzynski'``, default ``'cumulant'``).

    Returns
    -------
    str
        Path to the written preseed file.
    """
    os.makedirs(bias_dir, exist_ok=True)

    extrap_path = os.path.join(smd_analysis_outdir, 'dG_extrapolated.csv')
    mixture_path = os.path.join(smd_analysis_outdir, 'mixture_pmfs.csv')

    use_extrapolated = (
        (speed is None and os.path.exists(extrap_path))
        or (speed is not None and float(speed) == 0.0)
    )

    if use_extrapolated:
        if not os.path.exists(extrap_path):
            raise FileNotFoundError(
                f"v=0 extrapolated PMF not found: {extrap_path}. "
                "Run SMDAnalysis with at least 2 speeds to enable extrapolation."
            )
        pmf_df = pd.read_csv(extrap_path)
        speed_label = "v→0 extrapolated"
    else:
        if not os.path.exists(mixture_path):
            raise FileNotFoundError(f"mixture_pmfs.csv not found: {mixture_path}")
        pmf_df = pd.read_csv(mixture_path)
        use_speed = float(pmf_df['speed'].min()) if speed is None else float(speed)
        pmf_df = pmf_df[pmf_df['speed'] == use_speed]
        speed_label = f"speed={use_speed}"

    if 'estimator' in pmf_df.columns:
        if estimator in pmf_df['estimator'].values:
            pmf_df = pmf_df[pmf_df['estimator'] == estimator]
        else:
            fallback = next(
                (e for e in ('cumulant', 'jarzynski') if e in pmf_df['estimator'].values),
                None,
            )
            if fallback is None:
                raise ValueError(
                    f"Estimator '{estimator}' not found in PMF data and no fallback available. "
                    f"Available estimators: {pmf_df['estimator'].unique().tolist()}"
                )
            logger.warning(
                f"Estimator '{estimator}' not found; falling back to '{fallback}'."
            )
            pmf_df = pmf_df[pmf_df['estimator'] == fallback]

    pmf_df = pmf_df.dropna(subset=['r_coord', 'dG']).sort_values('r_coord').reset_index(drop=True)

    if pmf_df.empty:
        raise ValueError(
            f"No PMF data after filtering (speed={speed}, estimator={estimator}). "
            "Check the smd_analysis_outdir and parameters."
        )

    r_pmf = pmf_df['r_coord'].values
    F_pmf = pmf_df['dG'].values

    milestone_com = np.array(sorted(milestone_com_distances))
    n_ms = len(milestone_com)
    milestone_s = np.linspace(0.0, 1.0, n_ms)

    s_pmf = np.interp(r_pmf, milestone_com, milestone_s, left=0.0, right=1.0)

    sort_idx = np.argsort(s_pmf)
    s_pmf = s_pmf[sort_idx]
    F_pmf = F_pmf[sort_idx]

    s_grid = np.linspace(grid_min, grid_max, grid_points)
    F_grid = np.interp(s_grid, s_pmf, F_pmf, left=float(F_pmf[0]), right=float(F_pmf[-1]))

    dG_unbind = float(F_grid[-1])
    F_meta = F_grid - dG_unbind

    wtf = (gamma - 1.0) / gamma
    V_preseed = -alpha * wtf * F_meta

    kBT = 8.314e-3 * temperature
    s_ts, barrier_kJ = _find_pmf_peak(s_grid, F_grid, kBT, prominence_factor=1.0)
    if s_ts is None:
        logger.warning(
            "No PMF barrier detected in the sMD profile (monotonic or flat PMF). "
            "This can indicate a non-binder, insufficient sMD sampling, or a pulling "
            "speed too fast to resolve the barrier. Preseed will NOT be written — "
            "walkers will start from their milestone conformations without bias. "
            "To apply a preseed anyway, lower prominence_factor or verify sMD quality."
        )
        return None

    n_zeroed = int(np.sum(s_grid > s_ts))
    V_preseed[s_grid > s_ts] = 0.0
    logger.info(
        f"Preseed truncated at s_TS={s_ts:.3f} (barrier={barrier_kJ:.1f} kJ/mol); "
        f"zeroed {n_zeroed}/{grid_points} post-TS grid points."
    )
    logger.info(
        f"Preseed bias ({speed_label}): "
        f"ΔG_unbind={dG_unbind:.2f} kJ/mol, "
        f"V(s=0)={float(V_preseed[0]):.2f} kJ/mol "
        f"(alpha={alpha}, gamma={gamma})"
    )

    out_path = os.path.join(bias_dir, 'bias_0_0.npy')
    np.save(out_path, V_preseed)
    logger.info(f"Preseed bias written → {out_path}")

    try:
        plot_bias(
            out_dir=bias_dir,
            x_min=grid_min,
            x_max=grid_max,
            grid_points=grid_points,
            colvar_name="PathCV progress (s)",
        )
        logger.info(f"Preseed bias plot saved in {bias_dir}/")
    except Exception as _plot_err:
        logger.warning(f"Preseed bias plot failed (non-fatal): {_plot_err}")

    return out_path


# ---------------------------------------------------------------------------
# MetadynamicsAnalysis class
# ---------------------------------------------------------------------------

class MetadynamicsAnalysis:
    """Post-run analysis for multi-walker WT-MetaD simulations.

    Parameters
    ----------
    out_dir : str
        Directory where MetadynamicsMD wrote its output files
        (bias_*.npy, COLVAR_*.npy, FE_*.npy).
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def load_bias_files(bias_dir: str) -> dict:
        """Load all walker self-bias arrays from bias_dir.

        Returns {walker_id (int): bias_array (np.ndarray, kJ/mol)}.
        Only the latest snapshot for each walker is loaded (OpenMM
        overwrites with a higher save-index at each sync interval).
        """
        import re
        pattern = re.compile(r'bias_(\d+)_(\d+)\.npy')
        latest = {}
        for fname in sorted(os.listdir(bias_dir)):
            m = pattern.match(fname)
            if m:
                wid, idx = int(m.group(1)), int(m.group(2))
                if wid not in latest or idx > latest[wid][0]:
                    latest[wid] = (idx, os.path.join(bias_dir, fname))
        return {wid: np.load(fp) for wid, (_, fp) in latest.items()}

    @staticmethod
    def compute_fes(bias_arrays, temperature: float, bias_factor: float,
                    grid_min: float = 0.0, grid_max: float = 1.0) -> tuple:
        """Compute the WT-MetaD FES from a collection of bias arrays.

        Uses the standard estimator:
            FE(s) = -(gamma / (gamma − 1)) × V_total(s)
        where V_total = sum of all provided arrays.  Pass a subset of
        walkers' bias arrays to get a FES that excludes stuck walkers.

        Parameters
        ----------
        bias_arrays : list/array of np.ndarray
            Self-bias arrays in kJ/mol from load_bias_files().
        temperature : float
            Simulation temperature in K (not used in the formula but
            kept for API completeness / future extensions).
        bias_factor : float
            WT-MetaD bias factor γ.
        grid_min, grid_max : float
            CV grid bounds — used only to build the returned axis.

        Returns
        -------
        axis : np.ndarray   — CV values at each grid point
        fes  : np.ndarray   — Free energy in kJ/mol
        """
        V = np.sum(np.stack(bias_arrays), axis=0)
        gamma = float(bias_factor)
        fes = -(gamma / (gamma - 1.0)) * V
        axis = np.linspace(grid_min, grid_max, fes.shape[0])
        return axis, fes

    @staticmethod
    def analyze_colvar_transitions(colvar_array: np.ndarray,
                                    bound_threshold: float = 0.1,
                                    unbound_threshold: float = 0.9) -> dict:
        """Detect basin transitions in a 1D CV time series.

        A bound→unbound transition is counted each time the CV rises above
        unbound_threshold after having been below bound_threshold.  A round
        trip additionally requires a return below bound_threshold.

        Returns
        -------
        dict with keys:
            n_transitions        — bound→unbound crossings
            n_round_trips        — complete B→U→B cycles
            first_transition_frame — frame index of first B→U crossing (-1 = never)
            max_cv               — maximum CV value reached
            frac_bound           — fraction of frames in bound basin
            frac_unbound         — fraction of frames in unbound basin
            reached_unbound      — bool: crossed unbound_threshold at least once
        """
        cv = colvar_array[:, 0] if np.ndim(colvar_array) > 1 else np.asarray(colvar_array)
        n_trans, n_rt, first_frame = 0, 0, -1

        # Determine starting basin.  A walker seeded from an unbound sMD
        # milestone starts with CV≈1; we set in_unbound=True so that the
        # first return to the bound state is correctly counted as a round trip.
        in_unbound = cv[0] > unbound_threshold
        in_bound   = cv[0] < bound_threshold

        for i, v in enumerate(cv):
            if not in_unbound and v > unbound_threshold:
                n_trans += 1
                if first_frame < 0:
                    first_frame = i
                in_bound   = False
                in_unbound = True
            elif in_unbound and v < bound_threshold:
                n_rt += 1
                in_unbound = False
                in_bound   = True

        started_unbound = bool(cv[0] > unbound_threshold)
        return {
            'n_transitions':          n_trans,
            'n_round_trips':          n_rt,
            'first_transition_frame': first_frame,
            'max_cv':                 float(cv.max()),
            'frac_bound':             float(np.mean(cv < bound_threshold)),
            'frac_unbound':           float(np.mean(cv > unbound_threshold)),
            'reached_unbound':        bool(cv.max() > unbound_threshold),
            'started_unbound':        started_unbound,
        }

    @staticmethod
    def bias_coverage(bias_array: np.ndarray, threshold_frac: float = 1e-3) -> float:
        """Fraction of CV grid bins with non-negligible bias.

        A stuck walker deposits hills only near CV=0, leaving most of the
        grid at zero.  This metric is a proxy for convergence that can be
        computed from bias files alone, without needing COLVAR data.
        """
        if bias_array.max() <= 0:
            return 0.0
        return float(np.mean(bias_array > threshold_frac * bias_array.max()))

    # ── Instance methods ───────────────────────────────────────────────────

    def select_converged_biases(self, min_coverage: float = 0.6) -> list:
        """Return bias arrays from walkers that covered ≥ min_coverage of the CV grid.

        Stuck walkers (deposited hills only in the bound basin) have low
        grid coverage and are excluded.  This is the bias-file proxy for
        the transition criterion when COLVAR data is not available.

        Returns
        -------
        list of (walker_id, bias_array) tuples that passed the threshold.
        """
        biases = self.load_bias_files(self.out_dir)
        selected = []
        for wid, barr in sorted(biases.items()):
            cov = self.bias_coverage(barr)
            if cov >= min_coverage:
                selected.append((wid, barr))
            else:
                logging.warning(
                    f"Walker id={wid}: bias coverage {100*cov:.1f}% < "
                    f"{100*min_coverage:.0f}% — likely stuck, excluded from converged set."
                )
        if not selected:
            logging.warning("No walkers passed the coverage threshold — returning all biases.")
            selected = list(sorted(biases.items()))
        return selected

    def diagnose(
        self,
        colvar_files: list,
        temperature: float,
        bias_factor: float,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        cv_name: str = "PathCV_progress",
        bound_threshold: float = 0.1,
        unbound_threshold: float = 0.9,
        coverage_threshold: float = 0.6,
        out_prefix: str = None,
    ) -> dict:
        """Comprehensive multi-walker WT-MetaD diagnostic.

        Produces a four-panel PNG figure (PLUMED-community style) and returns
        a summary dict.  The four panels are:

        1. CV time traces — each walker, annotated with transition counts.
           Solid lines = reached unbound state; dashed = stuck.
        2. Self-bias per walker — what each walker actually deposited on the
           CV grid (only its own Gaussians, not the total).
        3. Per-walker FES — FES computed from each walker's self-bias alone
           (normalized min=0).  Reveals whether a single walker gives a
           physically reasonable shape.
        4. Combined vs converged-only FES — FES from all biases summed vs
           FES from walkers that passed the coverage threshold, both
           normalized to zero at the unbound plateau.  ΔG estimates are
           annotated directly on the plot.

        Parameters
        ----------
        colvar_files : list of str
            Paths to COLVAR_*.npy files (one per walker).
        temperature : float
            Simulation temperature in K.
        bias_factor : float
            WT-MetaD bias factor γ.
        grid_min, grid_max : float
            CV grid bounds.
        cv_name : str
            Label for the CV axis.
        bound_threshold, unbound_threshold : float
            CV values defining bound and unbound basins.
        coverage_threshold : float
            Minimum grid coverage fraction to classify a walker as converged.
        out_prefix : str, optional
            Output filename prefix.  Defaults to <out_dir>/diagnosis.

        Returns
        -------
        dict with keys:
            walker_stats    — {run_id: analyze_colvar_transitions result}
            fes_all_kcal    — FES from all walkers (kcal/mol, unbound=0)
            fes_conv_kcal   — FES from coverage-filtered walkers (kcal/mol, unbound=0)
            cv_axis         — grid axis
            n_converged_cv  — walkers that reached unbound_threshold
            n_walkers       — total walkers in colvar_files
        """
        import matplotlib.gridspec as gridspec

        if out_prefix is None:
            out_prefix = os.path.join(self.out_dir, "diagnosis")

        all_biases = self.load_bias_files(self.out_dir)

        walker_data = []
        for colvar_path in sorted(colvar_files):
            colvar = np.load(colvar_path)
            run_id = os.path.splitext(os.path.basename(colvar_path))[0].replace('COLVAR_', '')
            ts = self.analyze_colvar_transitions(colvar, bound_threshold, unbound_threshold)
            walker_data.append((run_id, colvar, ts))

        walker_stats = {run_id: ts for run_id, _, ts in walker_data}
        n_converged_cv = sum(1 for ts in walker_stats.values() if ts['reached_unbound'])
        converged_biases = self.select_converged_biases(coverage_threshold)

        all_bias_list = list(all_biases.values())
        axis, fes_all_kj = self.compute_fes(all_bias_list, temperature, bias_factor,
                                              grid_min, grid_max)
        n = len(axis)

        def _norm_plateau(fes_kj):
            plateau = fes_kj[int(0.85 * n):].mean()
            return (fes_kj - plateau) * 0.239006  # kcal/mol, unbound≈0

        fes_all_kcal = _norm_plateau(fes_all_kj)

        fes_conv_kcal = None
        if len(converged_biases) < len(all_biases):
            _, fes_conv_kj = self.compute_fes(
                [b for _, b in converged_biases], temperature, bias_factor, grid_min, grid_max
            )
            fes_conv_kcal = _norm_plateau(fes_conv_kj)

        n_w = len(walker_data)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_w, 1)))

        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)

        # Panel 1 — CV traces
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axhspan(0, bound_threshold, alpha=0.07, color='royalblue')
        ax1.axhspan(unbound_threshold, 1.0, alpha=0.07, color='tomato')
        ax1.axhline(bound_threshold,   color='royalblue', lw=0.7, ls='--', alpha=0.5)
        ax1.axhline(unbound_threshold, color='tomato',    lw=0.7, ls='--', alpha=0.5)
        for k, (run_id, colvar, ts) in enumerate(walker_data):
            cv = colvar[:, 0] if np.ndim(colvar) > 1 else colvar
            converged = ts['reached_unbound']
            label = (f"{run_id}  T={ts['n_transitions']} RT={ts['n_round_trips']}"
                     f"  max={ts['max_cv']:.3f}")
            ax1.plot(np.arange(len(cv)), cv, color=colors[k],
                      lw=1.8 if converged else 0.9,
                      ls='-' if converged else '--',
                      alpha=0.9 if converged else 0.6,
                      label=label)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Frame #")
        ax1.set_ylabel(cv_name)
        ax1.set_title("CV traces  (T=transitions  RT=round-trips  |  solid=reached unbound  dashed=stuck)")
        ax1.legend(fontsize=8, loc='upper left', framealpha=0.85)

        # Panel 2 — Self-bias per walker
        ax2 = fig.add_subplot(gs[1, 0])
        for k, (wid, barr) in enumerate(sorted(all_biases.items())):
            cov = self.bias_coverage(barr)
            ax2.plot(axis, barr * 0.239006, color=colors[k % len(colors)], lw=1.5,
                      label=f"id={wid}  cov={100*cov:.0f}%", alpha=0.85)
        ax2.set_xlabel(cv_name)
        ax2.set_ylabel("Self-bias (kcal/mol)")
        ax2.set_title("Self-bias per walker\n(only this walker's Gaussians)")
        ax2.legend(fontsize=8)

        # Panel 3 — Per-walker FES from self-bias only
        ax3 = fig.add_subplot(gs[1, 1])
        for k, (wid, barr) in enumerate(sorted(all_biases.items())):
            cov = self.bias_coverage(barr)
            _, fes_s_kj = self.compute_fes([barr], temperature, bias_factor, grid_min, grid_max)
            fes_s = fes_s_kj * 0.239006
            fes_s -= fes_s.min()
            ax3.plot(axis, fes_s, color=colors[k % len(colors)], lw=1.5,
                      label=f"id={wid}  cov={100*cov:.0f}%", alpha=0.85)
        ax3.set_xlabel(cv_name)
        ax3.set_ylabel("FE from self-bias (kcal/mol, min=0)")
        ax3.set_title("Per-walker FES\n(self-bias only — shape reflects individual sampling)")
        ax3.legend(fontsize=8)

        # Panel 4 — Combined vs converged-only FES
        ax4 = fig.add_subplot(gs[2, :])
        dG_all = float(fes_all_kcal.min())
        ax4.plot(axis, fes_all_kcal, 'k-', lw=2.0,
                  label=f"All walkers (n={len(all_biases)})  ΔG={dG_all:.1f} kcal/mol")
        if fes_conv_kcal is not None:
            dG_conv = float(fes_conv_kcal.min())
            ax4.plot(axis, fes_conv_kcal, 'g--', lw=2.0,
                      label=f"Coverage-filtered (n={len(converged_biases)})  ΔG={dG_conv:.1f} kcal/mol")
        ax4.axhline(0, color='gray', lw=0.7, ls=':')
        ax4.axvline(bound_threshold,   color='royalblue', lw=0.7, ls='--', alpha=0.4)
        ax4.axvline(unbound_threshold, color='tomato',    lw=0.7, ls='--', alpha=0.4)
        idx_min = int(np.argmin(fes_all_kcal))
        ax4.annotate(f"ΔG={dG_all:.1f}", xy=(axis[idx_min], dG_all),
                      xytext=(axis[idx_min] + 0.1, dG_all + 1.5), fontsize=9,
                      arrowprops=dict(arrowstyle='->', lw=0.8))
        ax4.set_xlabel(cv_name)
        ax4.set_ylabel("ΔG (kcal/mol, unbound plateau = 0)")
        ax4.set_title(
            f"FES comparison — {n_converged_cv}/{n_w} walkers reached unbound state\n"
            f"Use coverage-filtered FES when walkers are compartmentalized (Raiteri 2006)"
        )
        ax4.legend(fontsize=9)

        plt.suptitle(
            f"WT-MetaD Diagnosis — {os.path.basename(self.out_dir)}",
            fontsize=12, fontweight='bold'
        )
        out_path = f"{out_prefix}_diagnosis.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Diagnostic plot saved to {out_path}")

        logging.info("── Walker summary ──────────────────────────────────────")
        for run_id, _, ts in walker_data:
            status = "✓ converged" if ts['reached_unbound'] else "✗ stuck"
            logging.info(
                f"  {run_id}: {status}  T={ts['n_transitions']}  RT={ts['n_round_trips']}"
                f"  max_CV={ts['max_cv']:.3f}"
                f"  bound={100*ts['frac_bound']:.0f}%  unbound={100*ts['frac_unbound']:.0f}%"
            )
        logging.info(f"  ΔG(all walkers)     = {dG_all:.2f} kcal/mol")
        if fes_conv_kcal is not None:
            logging.info(f"  ΔG(coverage-filter) = {float(fes_conv_kcal.min()):.2f} kcal/mol")
        logging.info("────────────────────────────────────────────────────────")

        return {
            'walker_stats':   walker_stats,
            'fes_all_kcal':   fes_all_kcal,
            'fes_conv_kcal':  fes_conv_kcal,
            'cv_axis':        axis,
            'n_converged_cv': n_converged_cv,
            'n_walkers':      n_w,
        }
