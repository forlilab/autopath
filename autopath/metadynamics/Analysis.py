import os
import logging
from pathlib import Path
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

from autopath.pulling.Estimators import _find_pmf_peak

from autopath.metadynamics.Funnel import load_funnel_params, funnel_basins

logger = logging.getLogger("autopath")

KJ_TO_KCAL = 0.239006  # kJ/mol → kcal/mol conversion factor


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_colvar(out_dir: str = None, colvar_name: str = None):
    """Plot 1D collective variable (CV) time traces for each walker.

    Parameters
    ----------
    out_dir : str
        Directory containing ``COLVAR_*.npy`` output files from MetadynamicsMD.
    colvar_name : str
        Label for the CV axis (e.g. ``"PathCV_progress"``).

    Output
    ------
    Saves ``<sys_name>_COLVAR.png`` into ``out_dir``.
    """
    sys_name = Path(out_dir).parts[0]
    files = glob(f"{out_dir}/COLVAR_*")
    data = []
    for f in files:
        walker_name = os.path.splitext(os.path.basename(f))[0]
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
    plt.suptitle("Quick look (per walker, provisional) — final: *_FES / *_COLVAR / *_convergence.png", fontsize=8)
    plt.savefig(f"{out_dir}/{sys_name}_quicklook_COLVAR.png")
    plt.close()
    return


def plot_bias(out_dir: str = None, x_min: float = None, x_max: float = None, grid_points: int = None, colvar_name: str = None):
    """Plot per-walker accumulated bias profiles (hill deposition) over the CV grid.

    Parameters
    ----------
    out_dir : str
        Directory containing ``bias_*.npy`` output files from MetadynamicsMD.
    x_min, x_max : float
        CV grid bounds used to reconstruct the axis values.
    grid_points : int
        Number of grid points on the CV axis (must match the MetadynamicsMD run).
    colvar_name : str
        Label for the CV axis (e.g. ``"PathCV_progress"``).

    Output
    ------
    Saves ``<sys_name>_BIAS.png`` into ``out_dir``.
    """
    sys_name = Path(out_dir).parts[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    # Shared bias: bias_*.npy in out_dir; isolated: one bias_<run_id>/ subdir per walker.
    entries = []
    for f in sorted(glob(f"{out_dir}/bias_*.npy")):
        if os.path.isfile(f):
            entries.append((os.path.splitext(os.path.basename(f))[0], f))
    for d in sorted(glob(f"{out_dir}/bias_*")):
        if not os.path.isdir(d):
            continue
        walker = os.path.basename(d)[len("bias_"):]
        for f in sorted(glob(f"{d}/bias_*.npy")):
            entries.append((walker, f))

    if not entries:
        logger.warning("No bias arrays found under %s; skipping bias plot.", out_dir)
        return

    data = []
    for walker_name, f in entries:
        np_data = np.load(f)
        np_data = np_data * KJ_TO_KCAL
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
    plt.suptitle("Quick look (per walker, provisional) — final: *_FES / *_COLVAR / *_convergence.png", fontsize=8)
    plt.savefig(f"{out_dir}/{sys_name}_quicklook_BIAS.png")
    plt.close()
    return


def plot_FE(out_dir, x_min, x_max, grid_points, colvar_name):
    """Plot biased FE profiles (raw FE_*.npy, excluding _rw files), one line per walker."""
    sys_name = Path(out_dir).parts[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    files = sorted(f for f in glob(f"{out_dir}/FE_*.npy") if not f.endswith("_rw.npy"))
    if not files:
        logger.warning(f"plot_FE: no FE_*.npy files found in {out_dir}")
        return

    data = []
    for f in files:
        walker_name = os.path.splitext(os.path.basename(f))[0]
        np_data = np.load(f) * KJ_TO_KCAL
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
    plt.suptitle("Quick look (per walker, provisional) — final: *_FES / *_COLVAR / *_convergence.png", fontsize=8)
    plt.savefig(f"{out_dir}/{sys_name}_quicklook_FE.png")
    plt.close()
    return


def rct_weights(bias_kj, rct_kj, temperature: float) -> np.ndarray:
    """Normalised frame weights exp[(V(s_t, t) - c(t)) / kT] (Tiwary & Parrinello 2015).

    Inputs are the ``bias_kj`` and ``rct_kj`` arrays of ``REWEIGHT_<run_id>.npz``;
    drop the early, still-filling frames before using them.
    """
    lw = (np.asarray(bias_kj, float) - np.asarray(rct_kj, float)) / (0.0083144626 * temperature)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def dcd_frame_of_rows(n_rows: int, record_dt_ns: float, dcd_interval_ns: float) -> np.ndarray:
    """Trajectory frame of each COLVAR/REWEIGHT row, -1 where none.

    Row k is at k*record_dt; OpenMM writes DCD frames at D, 2D, ..., so a row
    maps to frame k*record_dt/D - 1 when that is a whole number.
    """
    r = np.arange(n_rows) * float(record_dt_ns) / float(dcd_interval_ns)
    hit = (r > 0.5) & (np.abs(r - np.round(r)) < 1e-6)
    return np.where(hit, np.round(r).astype(int) - 1, -1)


def reweighted_fes(values, weights, bins, temperature: float):
    """Unbiased free energy (kJ/mol, min 0) of any per-frame quantity from rct_weights.

    Returns (bin centres, F); empty bins are NaN.
    """
    p, edges = np.histogram(values, bins=bins, weights=weights)
    with np.errstate(divide="ignore"):
        f = -0.0083144626 * temperature * np.log(p)
    f[~np.isfinite(f)] = np.nan
    return 0.5 * (edges[1:] + edges[:-1]), f - np.nanmin(f)


def bound_site_bins(grid_min, grid_max, n_points, origin=None, frac=0.20) -> int:
    """Grid points from grid_min up to ``frac`` of the way from the origin to grid_max."""
    origin = grid_min if origin is None else float(origin)
    site_hi = origin + frac * (grid_max - origin)
    return max(1, int((site_hi - grid_min) / (grid_max - grid_min) * n_points + 1e-9))


def plot_FE_rw(out_dir, x_min, x_max, grid_points, colvar_name, temperature: float = 298.15,
               origin=None):
    """Plot standard-state-corrected FE profiles (FE_*_rw.npy).

    ΔG°_b per walker is the minimum over the bound site (see ``bound_site_bins``).

    Parameters
    ----------
    out_dir : str
        Directory containing ``FE_*_rw.npy`` output files from MetadynamicsMD.
    x_min, x_max : float
        CV grid bounds used to reconstruct the axis values.
    grid_points : int
        Number of grid points on the CV axis (must match the MetadynamicsMD run).
    colvar_name : str
        Label for the CV axis (e.g. ``"PathCV_progress"``).
    temperature : float
        Simulation temperature in K used to convert ΔG to pKd (default 298.15 K).
    origin : float, optional
        Bound end of the CV when the grid extends below it (default ``x_min``).

    Output
    ------
    Saves ``<sys_name>_FE_rw.png`` into ``out_dir``.
    """
    sys_name = Path(out_dir).parts[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

    files = sorted(glob(f"{out_dir}/FE_*_rw.npy"))
    if not files:
        return

    bound_bins = bound_site_bins(x_min, x_max, grid_points, origin)
    kT_log10 = 8.314e-3 * temperature * np.log(10)
    best_dG, best_pKd = None, None

    data = []
    for f in files:
        raw_name = os.path.splitext(os.path.basename(f))[0]
        short = raw_name.replace("FE_", "").replace("_rw", "")
        fe_kcal = np.load(f) * KJ_TO_KCAL
        dG = float(fe_kcal[:bound_bins].min())
        pKd_w = -dG * 4.184 / kT_log10
        dG_str = f"{dG:.1f}" if dG < 0 else f"+{dG:.1f}"
        label = f"{short}  ({dG_str} kcal/mol)"
        df = pd.DataFrame(fe_kcal, columns=["FE"])
        df.index = axis_values
        df["walker"] = label
        data.append(df)
        # Best = last walker with a negative ΔG and a flat mid-CV plateau (s=0.4-0.8).
        if dG < 0:
            mid_lo = int(0.40 * len(fe_kcal))
            mid_hi = int(0.80 * len(fe_kcal))
            plateau_ok = abs(float(fe_kcal[mid_lo:mid_hi].mean())) < 5.0
            if plateau_ok:
                best_dG, best_pKd = dG, pKd_w   # overwrite → last valid wins

    if best_dG is None:        # no walker passed — fall back to last negative ΔG
        for f in files:
            dG_f = float(np.load(f)[:bound_bins].min() * KJ_TO_KCAL)
            if dG_f < 0:
                best_dG = dG_f
                best_pKd = -best_dG * 4.184 / kT_log10

    data = pd.concat(data, axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data, x=data.index, y="FE", hue="walker", ax=ax)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_ylabel("ΔG°_b (kcal/mol)")
    ax.set_xlabel(colvar_name)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(f"FE standard-state corrected — {sys_name}   " + (
        f"ΔG°_b ≈ {best_dG:.1f} kcal/mol  |  pKd ≈ {best_pKd:.2f}" if best_dG is not None
        else "no walker resolved the bound site"))
    plt.tight_layout()
    plt.suptitle("Quick look (per walker, provisional) — final: *_FES / *_COLVAR / *_convergence.png", fontsize=8)
    plt.savefig(f"{out_dir}/{sys_name}_quicklook_FE_rw.png")
    plt.close()
    return


def plot_FE_2D(
    out_dir, x_min, x_max, x_grid_points, xCV_name, y_min, y_max, y_grid_points, yCV_name
):
    """Plot a 2D free energy surface (FES) heatmap from 2D metadynamics.

    Each ``FE_*.npy`` is a ``(y_grid_points, x_grid_points)`` array in kJ/mol.

    Parameters
    ----------
    out_dir : str
        Directory containing ``FE_*.npy`` output files from MetadynamicsMD.
    x_min, x_max : float
        Grid bounds for the first (x) collective variable.
    x_grid_points : int
        Number of grid points along the x CV axis.
    xCV_name : str
        Label for the x CV axis.
    y_min, y_max : float
        Grid bounds for the second (y) collective variable.
    y_grid_points : int
        Number of grid points along the y CV axis.
    yCV_name : str
        Label for the y CV axis.

    Output
    ------
    Saves one PNG per walker as ``<sys_name>_<walker_name>.png`` in ``out_dir``.
    """
    sys_name = Path(out_dir).parts[0]

    x_values = np.linspace(x_min, x_max, x_grid_points)
    x_values = [round(i, 2) for i in x_values]

    y_values = np.linspace(y_min, y_max, y_grid_points)
    y_values = [round(i, 2) for i in y_values]

    files_fe = glob(f"{out_dir}/FE_*.npy")
    for i, f in enumerate(files_fe):
        walker_name = os.path.splitext(os.path.basename(f))[0]

        np_data = np.load(f)
        np_data = np_data * KJ_TO_KCAL
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
    """Plot 2D CV time traces for each walker as separate line plots.

    Parameters
    ----------
    out_dir : str
        Directory containing ``COLVAR_*.npy`` output files from MetadynamicsMD.
    xCV_name : str
        Name of the first collective variable (column 0 of the COLVAR array).
    yCV_name : str
        Name of the second collective variable (column 1 of the COLVAR array).

    Output
    ------
    Saves ``<sys_name>_<xCV_name>_COLVAR.png`` and
    ``<sys_name>_<yCV_name>_COLVAR.png`` into ``out_dir``.
    """
    sys_name = Path(out_dir).parts[0]
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

def limongelli_dG(
    site_G_kJ: np.ndarray,
    dz_ang: float,
    R_cyl_ang: float,
    kT: float,
    C0_per_A3: float = 1.0 / 1660.0,
) -> dict:
    """Limongelli 2013 Eq. 15 standard-state binding free energy.

    ΔG°_b = -kT ln ∫_site exp(-βW) dz − kT ln(πR²_cyl C°).

    Parameters
    ----------
    site_G_kJ : np.ndarray
        Site profile W(z) in kJ/mol on a uniform grid, already zeroed on the
        unbound reference.
    dz_ang : float
        Grid spacing in Å.
    R_cyl_ang : float
        Cylinder radius in Å.
    kT : float
        kJ/mol.
    C0_per_A3 : float
        Standard concentration in Å⁻³ (default 1/1660 = 1 M).

    Returns
    -------
    dict
        ``integral_ang`` (Å), ``dG_basin`` (-kT ln integral), ``correction``
        (kT ln πR²C°), ``dG_bind_std`` (kJ/mol), ``pKd``.
    """
    import math

    # Only the cross-section is analytic; the length comes from the site integral.
    integral_ang = float(np.trapz(np.exp(-np.asarray(site_G_kJ) / kT), dx=dz_ang))
    if not integral_ang > 0:
        raise ValueError("bound-site integral is not positive; cannot take its log")
    dG_basin = -kT * math.log(integral_ang)
    correction = kT * math.log(math.pi * R_cyl_ang ** 2 * C0_per_A3)
    dG_bind_std = dG_basin - correction
    return {
        "integral_ang": integral_ang,
        "dG_basin": dG_basin,
        "correction": correction,
        "dG_bind_std": dG_bind_std,
        "pKd": -dG_bind_std / (kT * math.log(10)),
    }


def correct_fe_for_funnel(
    fe_files: "str | list[str] | np.ndarray",
    funnel_params: dict,
    temperature: float = 298.15,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
    n_plateau_points: int = 10,
    C0_per_A3: float = 1.0 / 1660.0,
    cv_length_per_unit_ang: float = None,
    strict: bool = True,
    verbose: bool = True,
    reference_range: tuple = None,
    cv_origin: float = None,
) -> dict:
    """Apply the Limongelli 2013 standard-state correction to a funnel metadynamics FE profile.

    The FES is zeroed on the unbound reference, then (Limongelli 2013 Eq. 15)
    ΔG°_b = -kT ln ∫_site exp(-βW) dz − kT ln(πR²_cyl C°).

    Parameters
    ----------
    fe_files : str, list of str, or np.ndarray
        Path(s) to FE_*.npy (kJ/mol), or the array itself. For a list only the last
        file is used, which is the full surface only with a shared biasDir.
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
    cv_length_per_unit_ang : float
        Å spanned per CV unit; required, the site integral is in length units.
    strict : bool
        Raise if the bound basin is not resolved; otherwise only flag it.
    reference_range : tuple, optional
        CV window for the unbound zero. Without it the last ``n_plateau_points`` are
        used, which sit on the hill-deficit artifact at a hard CV edge.
    cv_origin : float, optional
        Bound end of the CV when the grid extends below it (funnel z: 0).
        Default ``grid_min``.

    Returns
    -------
    dict with keys:
        fe_corrected        — FE_norm shifted so its minimum is ΔG°_b (kJ/mol)
        dG_bind_sim_kj_mol  — min(FE_norm) in the bound basin (well depth)
        correction_kj_mol   — kT·ln(πR²_cyl·C°); negative for typical R values
        dG_bind_std_kj_mol  — ΔG°_b (standard-state binding free energy)
        R_cylinder_ang      — cylinder radius used (Å)
        pKd                 — predicted pKd
        cv_at_minimum       — PathCV progress at the FE minimum
        n_fe_files          — number of FE files/arrays combined
        temperature_K       — temperature used

    Notes
    -----
    The bound site is the grid up to 20 % of the way from ``cv_origin`` to
    ``grid_max``, so an over-biased unbound
    region cannot pass as the binding minimum; the site must sit ≥10 kJ/mol below
    the unbound reference to count as sampled.
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
        if len(fe_files) > 1:
            logger.warning(
                "%d FE files given; using only the last (%s). That is the accumulated "
                "surface only when the walkers shared a biasDir -- with per-walker "
                "biasDirs each file is one walker's own estimate.",
                len(fe_files), os.path.basename(fe_files[-1]))
        fe = np.load(fe_files[-1]).astype(float)
        n_files = len(fe_files)

    cv = np.linspace(grid_min, grid_max, len(fe))

    n_tail = min(n_plateau_points, len(fe))
    if reference_range is not None:
        ref_mask = (cv >= reference_range[0]) & (cv <= reference_range[1])
        if ref_mask.sum() < 3:
            raise ValueError(f"reference_range {reference_range} covers fewer than 3 grid points")
        ref_vals = fe[ref_mask][np.isfinite(fe[ref_mask])]
        if not len(ref_vals):
            raise ValueError("unbound reference window has no finite FES values")
        plateau_val = float(ref_vals.mean())
        _plateau_range = (float(cv[ref_mask][0]), float(cv[ref_mask][-1]))
    else:
        tail = fe[-n_tail:][np.isfinite(fe[-n_tail:])]
        if not len(tail):
            raise ValueError("unbound reference tail has no finite FES values")
        plateau_val = float(tail.mean())
        _plateau_range = (float(cv[-n_tail]), float(cv[-1]))

    fe_norm = fe - plateau_val

    # Bound basin: up to 20 % of the way from the origin to grid_max.
    origin = grid_min if cv_origin is None else float(cv_origin)
    site_hi = origin + 0.20 * (grid_max - origin)
    bound_bins = bound_site_bins(grid_min, grid_max, len(fe_norm), origin)

    bound_depth = float(fe_norm[:bound_bins].min())
    bound_state_sampled = bound_depth <= -10.0
    if not bound_state_sampled:
        sign = "above" if bound_depth >= 0 else "only"
        msg = (
            f"Bound state not sampled: the FES in the bound basin (CV < {site_hi:.3g}) "
            f"sits {sign} {bound_depth:.1f} kJ/mol relative to the unbound reference. "
            "The walker may not have visited the binding site; ΔG would be meaningless."
        )
        # strict=False (convergence curves): an empty site still gives a finite, large ΔG.
        if strict:
            raise ValueError(msg)
        logger.debug("%s", msg)
    min_idx = int(np.argmin(fe_norm[:bound_bins]))
    dG_bind_sim = float(fe_norm[min_idx])
    cv_at_min = float(cv[min_idx])

    if cv_length_per_unit_ang is None or not np.isfinite(cv_length_per_unit_ang) \
            or cv_length_per_unit_ang <= 0:
        raise ValueError(
            "cv_length_per_unit_ang is required: Limongelli Eq. 15 integrates "
            "exp(-beta W) over the bound site in LENGTH units, and a progress "
            "coordinate is dimensionless. Pass the Angstrom spanned per CV unit "
            "(CVSpec.cv_length_per_unit_ang).")

    R_cyl_ang = float(funnel_params["R_cylinder"].value_in_unit(openmmunit.angstrom))
    z_cc_ang = float(funnel_params["z_cc"].value_in_unit(openmmunit.angstrom))
    z_max_ang = float(funnel_params["z_max"].value_in_unit(openmmunit.angstrom))
    d_cv = (grid_max - grid_min) / max(1, len(fe_norm) - 1)
    res = limongelli_dG(fe_norm[:bound_bins], d_cv * cv_length_per_unit_ang,
                        R_cyl_ang, kT, C0_per_A3)
    integral_ang = res["integral_ang"]
    dG_basin = res["dG_basin"]
    correction = res["correction"]
    dG_bind_std = res["dG_bind_std"]

    # Normalised FES shifted so its minimum sits at ΔG°_b.
    fe_rw = fe_norm + (dG_bind_std - dG_bind_sim)
    L_cyl_ang = z_max_ang - z_cc_ang
    pKd = res["pKd"]

    (logger.info if verbose else logger.debug)(
        "Funnel correction (Limongelli Eq. 15): well depth %.2f, basin integral "
        "%.3f Å -> ΔG=%.2f, analytic term kT·ln(πR²C°)=%.2f "
        "(R_cyl=%.1f Å), ΔG°_b=%.2f kJ/mol, pKd=%.2f",
        dG_bind_sim, integral_ang, dG_basin, correction,
        R_cyl_ang, dG_bind_std, pKd)

    return {
        "fe_corrected": fe_rw,
        "dG_bind_sim_kj_mol": dG_bind_sim,
        "correction_kj_mol": correction,
        "dG_bind_std_kj_mol": dG_bind_std,
        "R_cylinder_ang": R_cyl_ang,
        "z_cc_ang": z_cc_ang,
        "z_max_ang": z_max_ang,
        "L_cylinder_ang": L_cyl_ang,       # geometry only; Eq. 15 does not use it
        "bound_integral_ang": integral_ang,
        "dG_basin_kj_mol": dG_basin,
        "pKd": pKd,
        "bound_state_sampled": bool(bound_state_sampled),
        "bound_depth_kj_mol": bound_depth,
        "cv_at_minimum": cv_at_min,
        # CV ranges of the integrated bound site and the unbound reference.
        "site_range": (float(cv[0]), float(cv[bound_bins - 1])),
        "plateau_range": _plateau_range,
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
    taper: str = 'cosine',
) -> str | None:
    """Pre-seed the metadynamics bias surface from a sMD PMF profile.

    Writes ``bias_0_0.npy`` into ``bias_dir``, loaded by every walker via
    ``_syncWithDisk()`` before any hills are deposited:

        V_preseed(s) = alpha(s) × (gamma-1)/gamma × (ΔG_unbind − F_sMD(r(s)))

    with F_sMD mapped onto s ∈ [0,1] through the milestone COM distances. alpha(s)
    tapers to zero at the PMF peak (s_TS), since sMD variance grows along the path.

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
        Fraction of the converged bias pre-seeded at s=0 (default 0.3; 0.1–0.5 is
        reasonable, larger risks over-constraining the walker).
    grid_points : int
        Must match the CVSpec.grid_points used in the run (default 125).
    grid_min, grid_max : float
        Path CV grid bounds (default 0.0/1.0 for PathCV progress).
    speed : float or None
        sMD pulling speed. ``None``: the v→0 ``dG_extrapolated.csv`` if present, else
        the slowest speed in ``mixture_pmfs.csv``; ``0.0`` forces the extrapolated PMF.
    estimator : str
        ``'cumulant'`` or ``'jarzynski'`` (default ``'jarzynski'``).
    taper : str
        alpha(s) from ``alpha`` at s=0 to 0 at s_TS: ``'cosine'`` (default)
        cos²(π/2 · s/s_TS), ``'linear'`` 1 − s/s_TS, or ``'step'`` (uniform up to s_TS).

    Returns
    -------
    str or None
        Path to the written preseed file; None if the PMF has no barrier.
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

    # Seed strongest near the bound state, where the sMD PMF is most reliable.
    if taper == 'cosine':
        alpha_s = np.where(
            s_grid <= s_ts,
            alpha * np.cos(0.5 * np.pi * s_grid / s_ts) ** 2,
            0.0,
        )
    elif taper == 'linear':
        alpha_s = np.where(
            s_grid <= s_ts,
            alpha * (1.0 - s_grid / s_ts),
            0.0,
        )
    elif taper == 'step':
        alpha_s = np.where(s_grid <= s_ts, float(alpha), 0.0)
    else:
        raise ValueError(
            f"Unknown taper '{taper}'; choose 'cosine' (default), 'linear', or 'step'."
        )

    V_preseed = -alpha_s * wtf * F_meta

    logger.info(
        f"Preseed truncated at s_TS={s_ts:.3f} (barrier={barrier_kJ:.1f} kJ/mol); "
        f"taper='{taper}', alpha(s=0)={float(alpha_s[0]):.3f}."
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
    def dG_plateau(time_ns, dG_kj, sampled=None, tail_frac: float = 0.4,
                   drift_tol_kj: float = 4.0, both_basins: bool = True) -> dict:
        """Has ΔG°(t) settled?

        Settled = peak-to-peak spread over the trailing ``tail_frac`` ≤ ``drift_tol_kj``,
        the last snapshot resolved, and ``both_basins`` True (a walker that never
        enters the site gives a flat curve too).
        """
        t = np.asarray(time_ns, dtype=float)
        g = np.asarray(dG_kj, dtype=float)
        ok = np.isfinite(g)
        if sampled is not None:
            ok = ok & np.asarray(sampled, dtype=bool)
        # Settling is judged at the end of the run.
        ends_resolved = bool(ok[-1]) if ok.size else False
        if ok.sum() < 3:
            return {"settled": False, "n_points": int(ok.sum()), "drift_kj": float("nan"),
                    "reason": "too few snapshots with a sampled bound basin"}
        t, g = t[ok], g[ok]
        n_tail = max(3, int(tail_frac * len(g)))
        tail = g[-n_tail:]
        drift = float(np.ptp(tail))
        settled = bool(drift <= drift_tol_kj and both_basins and ends_resolved)
        return {
            "settled": settled,
            "both_basins": bool(both_basins),
            "ends_resolved": ends_resolved,
            "drift_kj": drift,
            "n_points": int(len(g)),
            "tail_frac": tail_frac,
            "dG_final_kj": float(tail.mean()),
            "reason": "" if settled else (
                f"ΔG still moves by {drift:.1f} kJ/mol over the last "
                f"{100*tail_frac:.0f}% of the run (tol {drift_tol_kj:.1f})"
                if drift > drift_tol_kj else
                "the walker did not visit both basins, so a flat ΔG only means "
                "nothing happened" if not both_basins else
                "the bound basin was not resolved at the end of the run, so the "
                "flat stretch is earlier history rather than a settled value"),
        }

    @staticmethod
    def assess_convergence(walker_transitions: list, fes_kj, grid_min: float = 0.0,
                           grid_max: float = 1.0, plateau_frac: float = 0.25,
                           min_round_trips: int = 1,
                           flatness_tol_kj: float = 5.0,
                           dG_plateau_result: dict = None,
                           reference_range: tuple = None) -> dict:
        """Limongelli 2013's two convergence conditions, as a verdict.

        (i) bound<->unbound recrossings, and (ii) a flat FES in the solvated region;
        optionally also a settled ΔG(t).

        Parameters
        ----------
        walker_transitions : list of dict
            ``analyze_colvar_transitions`` output, one per walker.
        fes_kj : array
            Free energy on the CV grid, kJ/mol.
        plateau_frac : float
            Trailing fraction of the grid treated as the solvated region.
        min_round_trips : int
            Recrossings required, summed over walkers.
        flatness_tol_kj : float
            Peak-to-peak spread allowed across the plateau.
    dG_plateau_result : dict, optional
        ``dG_plateau`` output; must be settled to converge.
    reference_range : tuple, optional
        CV window used as the plateau instead of ``plateau_frac``.

        Returns
        -------
        dict with ``converged`` and the two measurements behind it.
        """
        fes = np.asarray(fes_kj, dtype=float)
        if reference_range is not None:
            _cv = np.linspace(grid_min, grid_max, len(fes))
            tail = fes[(_cv >= reference_range[0]) & (_cv <= reference_range[1])]
        else:
            n_tail = max(2, int(plateau_frac * len(fes)))
            tail = fes[-n_tail:]
        flatness = float(np.ptp(tail))
        round_trips = int(sum(t.get("n_round_trips", 0) for t in walker_transitions))

        reasons = []
        if round_trips < min_round_trips:
            reasons.append(
                f"{round_trips} bound<->unbound round trip(s) across "
                f"{len(walker_transitions)} walker(s), need >= {min_round_trips}")
        if flatness > flatness_tol_kj:
            reasons.append(
                f"the solvated plateau spans {flatness:.1f} kJ/mol "
                f"(tol {flatness_tol_kj:.1f}); the unbound state is not flat")
        if dG_plateau_result is not None and not dG_plateau_result.get("settled"):
            reasons.append("ΔG(t) has not settled: "
                           + dG_plateau_result.get("reason", "still drifting"))
        return {
            "converged": not reasons,
            "dG_plateau": dG_plateau_result,
            "n_round_trips": round_trips,
            "plateau_span_kj": flatness,
            "plateau_frac": plateau_frac,
            "reasons": reasons,
        }

    def walker_transitions(self, colvar_path, basins, bound_threshold=0.1,
                           unbound_threshold=0.9, grid_min=None, grid_max=None,
                           cv_is_funnel_z=False) -> dict:
        """Transitions of one walker.

        A funnel z CV (nm) is tested in Å against ``basins = (bound_max, unbound_min)``;
        any other CV uses grid-fraction thresholds.
        """
        col = np.load(colvar_path)
        if cv_is_funnel_z and basins is not None:
            cv = col[:, 0] if col.ndim > 1 else col
            ts = self.analyze_colvar_transitions(cv[:, None] * 10.0, bound_threshold=basins[0],
                                                 unbound_threshold=basins[1])
            ts["coordinate"] = "funnel_z"
            return ts
        ts = self.analyze_colvar_transitions(col, bound_threshold,
                                             unbound_threshold, grid_min=grid_min,
                                             grid_max=grid_max)
        ts["coordinate"] = "cv"
        return ts

    @staticmethod
    def analyze_colvar_transitions(colvar_array: np.ndarray,
                                    bound_threshold: float = 0.1,
                                    unbound_threshold: float = 0.9,
                                    grid_min: float = None,
                                    grid_max: float = None) -> dict:
        """Detect basin transitions in a 1D CV time series.

        A transition is a rise above ``unbound_threshold``; a round trip adds a return
        below ``bound_threshold``. With ``grid_min``/``grid_max`` the thresholds are
        fractions of the CV grid.

        Returns
        -------
        dict with keys:
            n_transitions        — bound→unbound crossings
            n_round_trips        — complete B→U→B cycles
            exit_frames, return_frames — frame indices of those crossings
            bound_threshold, unbound_threshold — thresholds in CV units
            first_transition_frame — frame index of first B→U crossing (-1 = never)
            max_cv               — maximum CV value reached
            frac_bound           — fraction of frames in bound basin
            frac_unbound         — fraction of frames in unbound basin
            reached_unbound      — bool: crossed unbound_threshold at least once
            started_unbound      — bool: first frame already unbound

        A walker starting unbound counts its first return as a round trip.
        """
        cv = colvar_array[:, 0] if np.ndim(colvar_array) > 1 else np.asarray(colvar_array)
        if grid_min is not None and grid_max is not None:
            span = float(grid_max) - float(grid_min)
            bound_threshold = float(grid_min) + bound_threshold * span
            unbound_threshold = float(grid_min) + unbound_threshold * span
        n_trans, n_rt, first_frame = 0, 0, -1
        exit_frames, return_frames = [], []

        in_unbound = cv[0] > unbound_threshold
        in_bound   = cv[0] < bound_threshold

        for i, v in enumerate(cv):
            if not in_unbound and v > unbound_threshold:
                n_trans += 1
                exit_frames.append(i)
                if first_frame < 0:
                    first_frame = i
                in_bound   = False
                in_unbound = True
            elif in_unbound and v < bound_threshold:
                n_rt += 1
                return_frames.append(i)
                in_unbound = False
                in_bound   = True

        started_unbound = bool(cv[0] > unbound_threshold)
        return {
            'n_transitions':          n_trans,
            'n_round_trips':          n_rt,
            'exit_frames':            exit_frames,
            'return_frames':          return_frames,
            'bound_threshold':        float(bound_threshold),
            'unbound_threshold':      float(unbound_threshold),
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

        Convergence proxy from bias files alone: a stuck walker biases only near CV=0.
        """
        if bias_array.max() <= 0:
            return 0.0
        return float(np.mean(bias_array > threshold_frac * bias_array.max()))

    # ── Instance methods ───────────────────────────────────────────────────

    def diagnose(
        self,
        colvar_files: list = None,
        temperature: float = 300.0,
        bias_factor: float = 15.0,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        cv_name: str = "CV",
        bound_threshold: float = 0.1,
        unbound_threshold: float = 0.9,
        coverage_threshold: float = 0.6,
        out_prefix: str = None,
        cv_length_per_unit_ang: float = None,
        cv_is_funnel_z: bool = False,
        hill_width: float = None,
    ) -> dict:
        """End-of-run analysis: see :func:`autopath.metadynamics.report.report`.

        Walkers are discovered from the HILLS v2 files in ``out_dir``;
        ``colvar_files``, ``bias_factor`` (HILLS carries the tempering) and the
        threshold/coverage arguments are ignored.
        """
        from autopath.metadynamics.report import report
        return report(self.out_dir, temperature, grid_min=grid_min,
                      grid_max=grid_max, cv_name=cv_name,
                      cv_length_per_unit_ang=cv_length_per_unit_ang,
                      out_prefix=out_prefix, cv_is_funnel_z=cv_is_funnel_z,
                      hill_width=hill_width)
