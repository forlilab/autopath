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

    Loads all ``COLVAR_*.npy`` files from ``out_dir`` and draws one line per
    walker on a shared axes, using walker filename as the legend label.

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
    plt.savefig(f"{out_dir}/{sys_name}_COLVAR.png")
    plt.close()
    return


def plot_bias(out_dir: str = None, x_min: float = None, x_max: float = None, grid_points: int = None, colvar_name: str = None):
    """Plot per-walker accumulated bias profiles (hill deposition) over the CV grid.

    Loads all ``bias_*.npy`` files from ``out_dir``, converts from kJ/mol to
    kcal/mol, and draws one line per walker on a shared axes.

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

    # Shared-bias mode writes bias_*.npy straight into out_dir. Isolated mode
    # (mMD_multiple_walkers=False) gives each walker its own bias_<run_id>/
    # subdirectory, so out_dir holds DIRECTORIES named bias_* and no .npy at all --
    # a bare bias_* glob then hands np.load a directory (IsADirectoryError).
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
    plt.savefig(f"{out_dir}/{sys_name}_BIAS.png")
    plt.close()
    return


def plot_FE(out_dir, x_min, x_max, grid_points, colvar_name):
    """Plot biased FE profiles (raw FE_*.npy, excluding _rw files).

    Each file is a sequential walker that also accumulated all previous walkers'
    bias (shared biasDir), so later files are more converged estimates.  All
    walker traces are shown together to visualise convergence.
    """
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
    plt.savefig(f"{out_dir}/{sys_name}_FE.png")
    plt.close()
    return


def plot_FE_rw(out_dir, x_min, x_max, grid_points, colvar_name, temperature: float = 298.15):
    """Plot standard-state-corrected FE profiles (FE_*_rw.npy).

    Each _rw file shows the FES normalised to the unbound endpoint (s=1 ≈ 0) with
    the Limongelli 2013 standard-state correction applied.  ΔG°_b is read from the
    bound-basin minimum (first 20 % of the CV).  Each legend entry includes the
    per-walker ΔG°_b.  The title reports the best (most negative valid) estimate.

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

    bound_bins = max(1, int(0.20 * grid_points))
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
        # Best = last walker (most accumulated bias = most converged) with a
        # valid negative bound-basin ΔG AND a roughly flat plateau (|mean| of
        # mid-CV region s=0.4–0.8 within 5 kcal/mol of zero).
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
    ax.set_title(f"FE standard-state corrected — {sys_name}   "
                 f"ΔG°_b ≈ {best_dG:.1f} kcal/mol  |  pKd ≈ {best_pKd:.2f}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_FE_rw.png")
    plt.close()
    return


def plot_FE_2D(
    out_dir, x_min, x_max, x_grid_points, xCV_name, y_min, y_max, y_grid_points, yCV_name
):
    """Plot a 2D free energy surface (FES) heatmap from 2D metadynamics.

    Loads all ``FE_*.npy`` files from ``out_dir`` — each expected to be a 2D
    array of shape ``(y_grid_points, x_grid_points)`` in kJ/mol — converts to
    kcal/mol, and saves one heatmap per walker using a Spectral colormap.

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

    Loads all ``COLVAR_*.npy`` files from ``out_dir``, expecting each array
    to have two columns (one per CV).  Produces two figures: one for
    ``xCV_name`` vs frame index and one for ``yCV_name`` vs frame index,
    with one line per walker on each.

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
        Path(s) to FE_*.npy (kJ/mol), or the array itself. Each FE file holds
        ``getFreeEnergy()`` for one walker, i.e. its ``_totalBias``. With a SHARED
        biasDir that total already includes every walker, so the last file is the
        accumulated surface; with a per-walker biasDir it is only that walker's own
        estimate and the last file is not "the most converged". Passing a list
        therefore assumes shared accumulation and warns.
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

    Notes
    -----
    **Unbound-state normalization:** The last ``n_plateau_points`` grid points
    (s ≈ 1, funnel endpoint) are averaged and subtracted to set the unbound
    reference to zero — the assumption underlying the Limongelli 2013
    standard-state correction (Eq. 3).  Walkers that over-biased s ≈ 1 produce
    a deeply-negative endpoint; the bound-depth check below catches them
    (their bound basin stays positive relative to the reference) and raises
    a ``ValueError`` before a spurious ΔG is reported.

    **Bound-basin ΔG (20 % rule):** ΔG is read from the minimum within the first
    20 % of the CV grid (s < 0.20), so an over-biased unbound region can never
    masquerade as the binding minimum.  This replaces an older ``unsampled_gap``
    check (``fe_max − plateau < 10``) that misfired when the entire FES was
    deeply negative (heavily biased at both ends — more sampled, not less).
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

    # Last N points → unbound reference (s ≈ 1); see Notes in docstring.
    n_tail = min(n_plateau_points, len(fe))
    plateau_val = float(fe[-n_tail:].mean())

    fe_norm = fe - plateau_val

    # First 20 % of CV = bound basin; see Notes in docstring.
    bound_bins = max(1, int(0.20 * len(fe_norm)))

    # Sanity check: bound basin must be meaningfully below the unbound reference.
    bound_depth = float(fe_norm[:bound_bins].min())   # should be negative
    bound_state_sampled = bound_depth <= -10.0
    if not bound_state_sampled:
        sign = "above" if bound_depth >= 0 else "only"
        msg = (
            f"Bound state not sampled: the FES in the bound basin (s < 0.20) "
            f"sits {sign} {bound_depth:.1f} kJ/mol relative to the unbound reference. "
            "The walker may not have visited the binding site; ΔG would be meaningless."
        )
        # Reported numbers stay gated. Convergence curves pass strict=False: a
        # walker seeded outside the site has no basin yet by construction, and
        # the integral is Boltzmann-weighted, so an empty site yields a small
        # integral and a large positive ΔG -- well defined, just not an estimate.
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
    # Limongelli 2013 Eq. 15:  K_b = pi*R_cyl^2 * INTEGRAL_site dz exp(-beta W(z))
    # Only the CROSS-SECTION enters the analytic term; the length comes from the
    # integral over the BOUND SITE. Using the cylinder length instead, with the
    # well depth in place of the integral, treats the bound basin as if it were as
    # wide as the whole cylinder and overestimates affinity by kT*ln(L/dz_site).
    d_cv = (grid_max - grid_min) / max(1, len(fe_norm) - 1)
    site = fe_norm[:bound_bins]
    integral_ang = float(np.trapz(np.exp(-site / kT), dx=d_cv)
                         * cv_length_per_unit_ang)
    if integral_ang <= 0:
        raise ValueError("bound-site integral is not positive; cannot take its log")
    dG_basin = -kT * math.log(integral_ang)          # = -kT ln INTEGRAL_site
    correction = kT * math.log(math.pi * R_cyl_ang ** 2 * C0_per_A3)
    dG_bind_std = dG_basin - correction

    # Reported surface keeps the old meaning: the normalised FES shifted so its
    # minimum sits at the standard-state binding free energy.
    fe_rw = fe_norm + (dG_bind_std - dG_bind_sim)
    L_cyl_ang = z_max_ang - z_cc_ang
    pKd = -dG_bind_std / (kT * math.log(10))

    logger.info(
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

    Writes ``bias_0_0.npy`` into ``bias_dir`` so all walkers load it
    via ``_syncWithDisk()`` at initialisation (before any hills are deposited).

    The preseed is a fraction (``alpha``) of the well-tempered-metadynamics bias
    that would be deposited at full convergence:
        V_preseed(s) = alpha × (gamma-1)/gamma × (ΔG_unbind − F_sMD(r(s)))
    where F_sMD is the sMD PMF in kJ/mol (0 at bound state) mapped onto the
    path-CV progress coordinate s ∈ [0,1] using the milestone COM distances.

    The preseed is tapered toward the transition state (PMF peak) and set to zero
    beyond it, so metadynamics explores the post-barrier region without bias from
    the sMD profile.  The taper accounts for the heteroskedastic reliability of
    the sMD PMF: the bound-state region (s ≈ 0) has low cumulative variance and
    is seeded at full ``alpha``; variance accumulates toward the barrier, so the
    effective seeding fraction smoothly declines to zero at s_TS (``taper='cosine'``
    default) rather than applying a uniform bias up to a hard cutoff.

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
        Maximum fraction of the converged bias to pre-seed, applied at s=0
        (default 0.3; values 0.1–0.5 are reasonable — too large risks
        over-constraining the walker).  With a taper, the effective fraction
        declines toward zero at the TS.
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
    taper : str
        Shape of the position-dependent alpha(s) decay from ``alpha`` at s=0 to
        0 at s=s_TS.  ``'cosine'`` (default): smooth cos²(π/2 · s/s_TS) ramp —
        no discontinuity at s_TS and steeper near the bound state.
        ``'linear'``: alpha × (1 − s/s_TS) — linear decline.
        ``'step'``: original flat-then-zero behaviour (uniform alpha up to s_TS).

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

    # Position-dependent taper: alpha_s(s) declines from alpha at s=0 to 0 at s_ts.
    # The sMD PMF (Jarzynski/cumulant) accumulates variance along the path, so it is
    # most reliable near the bound state and least reliable near the barrier.  Grading
    # the preseed strength by position seeds most strongly where sMD statistics are best.
    if taper == 'cosine':
        # Smooth cos²(π/2 · s/s_ts) ramp — no kink at s_ts, steeper near bound state.
        alpha_s = np.where(
            s_grid <= s_ts,
            alpha * np.cos(0.5 * np.pi * s_grid / s_ts) ** 2,
            0.0,
        )
    elif taper == 'linear':
        # Linear ramp: alpha × (1 − s/s_ts).
        alpha_s = np.where(
            s_grid <= s_ts,
            alpha * (1.0 - s_grid / s_ts),
            0.0,
        )
    elif taper == 'step':
        # Original behaviour: uniform alpha up to s_ts, hard zero beyond.
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
    def load_bias_files(bias_dir: str) -> dict:
        """Load all walker self-bias arrays from bias_dir.

        Returns {walker_id (int): bias_array (np.ndarray, kJ/mol)}.
        Only the latest snapshot for each walker is loaded (OpenMM
        overwrites with a higher save-index at each sync interval).
        """
        import re
        pattern = re.compile(r'bias_(\d+)_(\d+)\.npy')
        # Shared-bias mode writes bias_<id>_<idx>.npy directly into bias_dir.
        # Isolated mode (mMD_multiple_walkers=False) gives each walker its own
        # bias_<run_id>/ subdirectory, so bias_dir itself holds no matching files
        # and a flat listdir silently returns nothing.
        search_dirs = [bias_dir]
        search_dirs += sorted(
            d for d in glob(os.path.join(bias_dir, "bias_*")) if os.path.isdir(d))
        latest = {}
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                m = pattern.match(fname)
                if m:
                    wid, idx = int(m.group(1)), int(m.group(2))
                    if wid not in latest or idx > latest[wid][0]:
                        latest[wid] = (idx, os.path.join(d, fname))
        return {wid: np.load(fp) for wid, (_, fp) in latest.items()}

    @staticmethod
    def _plot_dG_convergence(curves: dict, out_prefix: str = None) -> str:
        """One line per walker: ΔG° against simulation time.

        Dotted where the bound basin is not yet resolved. Those points are
        sampling transients rather than estimates, but hiding them would break
        the curve where a path-seeded walker begins.
        """
        if not curves:
            return None
        plt.figure(figsize=(7, 4))
        any_unresolved = False
        for rid, entry in sorted(curves.items()):
            t = np.asarray(entry[0], dtype=float)
            g = np.asarray(entry[1], dtype=float) * KJ_TO_KCAL
            mask = (np.asarray(entry[2], dtype=bool) if len(entry) > 2
                    else np.isfinite(g))
            any_unresolved = any_unresolved or bool((~mask).any())
            line, = plt.plot(np.where(mask, t, np.nan), np.where(mask, g, np.nan),
                             marker="o", ms=3, lw=1.4, label=rid)
            plt.plot(t, g, color=line.get_color(), ls=":", lw=0.9,
                     alpha=0.5, zorder=0)
        if any_unresolved:
            plt.plot([], [], color="0.5", ls=":", lw=0.9,
                     label="bound basin not resolved")
        plt.xlabel("simulation time (ns)")
        plt.ylabel("ΔG°$_b$ (kcal/mol)")
        plt.title("Binding free energy vs time (convergence)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = f"{out_prefix}_dG_convergence.png" if out_prefix else "dG_convergence.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Wrote the ΔG(t) convergence plot: %s", path)
        return path

    @staticmethod
    def shared_bias_run_order(hist_files) -> list:
        """Execution order of walkers that shared one bias directory.

        Shared-bias walkers run sequentially, so each inherits its predecessors'
        hills and its history continues theirs rather than restarting. Alphabetical
        order is not run order -- the loop starts from the most unbound milestone --
        so the files are ordered by the time each walker finished writing them.
        """
        def _key(f):
            try:
                d = np.load(f)
                if "started_at" in getattr(d, "files", []):
                    return float(d["started_at"])
            except Exception:
                pass
            return os.path.getmtime(f)
        return sorted(hist_files, key=_key)

    @staticmethod
    def dG_timeseries(fe_hist_npz, funnel_params: dict, temperature: float = 298.15,
                      grid_min: float = 0.0, grid_max: float = 1.0,
                      cv_length_per_unit_ang: float = None,
                      strict: bool = False) -> tuple:
        """ΔG°(t) from a saved FES history — the Box 2 convergence curve.

        Raniolo & Limongelli judge convergence on the free-energy difference
        between bound and unbound as a function of simulation time: it is
        converged once that value fluctuates about a constant. A single final
        number cannot show that, so each snapshot is corrected independently.

        Parameters
        ----------
        fe_hist_npz : str or dict-like
            ``FEhist_<run_id>.npz`` holding ``fes_kj`` (n_snapshots, n_grid) and
            ``time_ns``.

        strict : bool
            Default False, so every snapshot gets a value and the third array
            flags which ones had a resolved bound basin. Dropping the rest cuts
            the curve exactly where a path-seeded walker starts, which is the
            part worth seeing.

        Returns
        -------
        (time_ns, dG_kj, bound_state_sampled) arrays.
        """
        data = np.load(fe_hist_npz) if isinstance(fe_hist_npz, str) else fe_hist_npz
        fes = np.asarray(data["fes_kj"], dtype=float)
        t = np.asarray(data["time_ns"], dtype=float)
        out = np.full(len(fes), np.nan)
        sampled = np.zeros(len(fes), dtype=bool)
        for i, fe in enumerate(fes):
            try:
                r = correct_fe_for_funnel(
                    fe, funnel_params, temperature=temperature,
                    grid_min=grid_min, grid_max=grid_max,
                    cv_length_per_unit_ang=cv_length_per_unit_ang,
                    strict=strict,
                )
                out[i] = r["dG_bind_std_kj_mol"]
                sampled[i] = r["bound_state_sampled"]
            except Exception:
                pass          # strict=True, or a surface with no usable integral
        return t, out, sampled

    @staticmethod
    def dG_plateau(time_ns, dG_kj, sampled=None, tail_frac: float = 0.4,
                   drift_tol_kj: float = 4.0, both_basins: bool = True) -> dict:
        """Has ΔG°(t) settled?

        Measures the spread of ΔG over the trailing ``tail_frac`` of the run.
        This is the primary convergence test: a surface can look flat at one
        instant and still be drifting.

        ``both_basins`` must be True for the plateau to count. A walker that never
        enters the binding site produces the flattest curve of all -- it is flat
        near zero because nothing is happening -- so flatness alone would rank the
        least informative walker as the best converged.
        """
        t = np.asarray(time_ns, dtype=float)
        g = np.asarray(dG_kj, dtype=float)
        ok = np.isfinite(g)
        if sampled is not None:
            # Values now exist for unsampled snapshots too; only the resolved
            # ones can settle onto a ΔG.
            ok = ok & np.asarray(sampled, dtype=bool)
        # Settling is a statement about the END of the run. A walker that
        # resolved a basin early and then left has not settled, however flat the
        # points it did produce.
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
                           dG_plateau_result: dict = None) -> dict:
        """Limongelli 2013's two convergence conditions, as a verdict.

        The paper reports a binding free energy only once (i) the system has
        recrossed between bound and unbound several times, and (ii) the FES is
        flat in the fully solvated region -- "the flatness of the FES gives a
        measure of the convergence of our calculations". A single outward pass
        with a sloping plateau is not a converged surface, and a dG read off one
        is not interpretable.

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

        Returns
        -------
        dict with ``converged`` and the two measurements behind it.
        """
        fes = np.asarray(fes_kj, dtype=float)
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

    @staticmethod
    def bias_layout(bias_dir: str) -> str:
        """'isolated' if each walker owns a bias_<run_id>/ subdirectory, else 'shared'.

        Decides how walker biases combine: summed when they are partial
        contributions to one shared bias, averaged when each is complete.
        """
        import re
        pattern = re.compile(r'bias_(\d+)_(\d+)\.npy')
        if any(pattern.match(f) for f in os.listdir(bias_dir)):
            return "shared"
        if any(os.path.isdir(os.path.join(bias_dir, d))
               for d in os.listdir(bias_dir) if d.startswith("bias_")):
            return "isolated"
        return "shared"

    @staticmethod
    def compute_fes(bias_arrays, temperature: float, bias_factor: float,
                    grid_min: float = 0.0, grid_max: float = 1.0,
                    combine: str = "sum") -> tuple:
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
            WT-MetaD bias factor γ.  Must be > 1; γ = 1 corresponds to
            unbiased MD where the FES estimator is undefined (divide-by-zero).
        grid_min, grid_max : float
            CV grid bounds — used only to build the returned axis.

        Returns
        -------
        axis : np.ndarray   — CV values at each grid point
        fes  : np.ndarray   — Free energy in kJ/mol
        """
        # OpenMM writes each walker's SELF bias (metadynamics.py: np.save(..., 
        # self._selfBias)) and rebuilds the total by summing every walker's file.
        # So with a shared biasDir the arrays are partial contributions to ONE bias
        # and must be summed. With a per-walker biasDir each array is already that
        # walker's complete bias -- an independent estimate of the same surface --
        # and summing them scales the free energy by the number of walkers.
        stacked = np.stack(bias_arrays)
        if combine == "sum":
            V = np.sum(stacked, axis=0)
        elif combine == "mean":
            V = np.mean(stacked, axis=0)
        else:
            raise ValueError(
                f"combine must be 'sum' (shared biasDir) or 'mean' (per-walker "
                f"biasDir); got {combine!r}")
        gamma = float(bias_factor)
        if abs(gamma - 1.0) < 1e-9:
            raise ValueError(
                f"bias_factor (γ) must be > 1 for well-tempered metadynamics; got {gamma}. "
                f"γ = 1 corresponds to unbiased MD where the FES estimator is undefined."
            )
        fes = -(gamma / (gamma - 1.0)) * V
        axis = np.linspace(grid_min, grid_max, fes.shape[0])
        return axis, fes

    def walker_transitions(self, colvar_path, basins, bound_threshold=0.1,
                           unbound_threshold=0.9, grid_min=None, grid_max=None) -> dict:
        """Transitions of one walker: on its recorded funnel z (``FUNNELZ_<run_id>.npy``,
        Å) with ``basins = (bound_max, unbound_min)`` when both exist, else on the CV
        with grid-fraction thresholds."""
        run_id = os.path.splitext(os.path.basename(colvar_path))[0].replace('COLVAR_', '')
        z_path = os.path.join(os.path.dirname(colvar_path), f"FUNNELZ_{run_id}.npy")
        if basins is not None and os.path.exists(z_path):
            ts = self.analyze_colvar_transitions(np.load(z_path)[:, None],
                                                 bound_threshold=basins[0],
                                                 unbound_threshold=basins[1])
            ts["coordinate"] = "funnel_z"
            return ts
        ts = self.analyze_colvar_transitions(np.load(colvar_path), bound_threshold,
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

        A bound→unbound transition is counted each time the CV rises above
        unbound_threshold after having been below bound_threshold.  A round
        trip additionally requires a return below bound_threshold. When
        ``grid_min``/``grid_max`` are given, both thresholds are fractions of the
        CV grid, so a length CV and the [0, 1] path progress are judged alike.

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

        Notes
        -----
        **Asymmetric seeding:** Walkers seeded from an unbound sMD milestone
        start with CV ≈ 1 (already in the unbound basin).  ``in_unbound`` is
        initialised to ``True`` in that case so the first return below
        ``bound_threshold`` is correctly counted as a round trip rather than
        discarded.
        """
        cv = colvar_array[:, 0] if np.ndim(colvar_array) > 1 else np.asarray(colvar_array)
        if grid_min is not None and grid_max is not None:
            span = float(grid_max) - float(grid_min)
            bound_threshold = float(grid_min) + bound_threshold * span
            unbound_threshold = float(grid_min) + unbound_threshold * span
        n_trans, n_rt, first_frame = 0, 0, -1

        # Initialise starting basin; unbound-seeded walkers get in_unbound=True.
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
                logger.warning(
                    f"Walker id={wid}: bias coverage {100*cov:.1f}% < "
                    f"{100*min_coverage:.0f}% — likely stuck, excluded from converged set."
                )
        if not selected:
            logger.warning("No walkers passed the coverage threshold — returning all biases.")
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
        cv_length_per_unit_ang: float = None,
    ) -> dict:
        """Comprehensive multi-walker WT-MetaD diagnostic.

        Produces a three-panel PNG figure and returns a summary dict.
        The three panels are:

        1. CV time traces — each walker, annotated with transition counts.
           Solid lines = reached unbound state; dashed = stuck.
        2. Self-bias per walker — what each walker actually deposited on the
           CV grid (only its own Gaussians, not the total).
        3. Combined vs converged-only FES — FES from all biases summed vs
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

        Notes
        -----
        **Standard-state correction uses all walkers:** The Limongelli 2013
        correction is applied to the combined FES from *all* walkers, not only
        the converged subset.  Stuck walkers accumulate hills in the bound basin
        (supplying bound-state bias); converged walkers explore the full CV and
        supply the unbound-basin bias.  Together they produce the best available
        estimate of both FES endpoints, even when no individual walker traversed
        the complete CV range.
        """
        import matplotlib.gridspec as gridspec

        if out_prefix is None:
            out_prefix = os.path.join(self.out_dir, "diagnosis")

        all_biases = self.load_bias_files(self.out_dir)

        # Round trips are counted on the funnel axis when it was recorded: pocket and
        # cylinder are physical basins, whereas the ends of a CV grid are not.
        basins = None
        _fp_npz = os.path.join(self.out_dir, 'funnel_params.npz')
        if os.path.exists(_fp_npz):
            basins = funnel_basins(load_funnel_params(_fp_npz))
            logger.info("Round trips on the funnel axis: bound z < %.1f A, unbound z > %.1f A "
                        "(walkers without FUNNELZ fall back to CV-grid fractions).", *basins)

        walker_data = []
        for colvar_path in sorted(colvar_files):
            colvar = np.load(colvar_path)
            run_id = os.path.splitext(os.path.basename(colvar_path))[0].replace('COLVAR_', '')
            ts = self.walker_transitions(colvar_path, basins, bound_threshold,
                                         unbound_threshold, grid_min, grid_max)
            walker_data.append((run_id, colvar, ts))

        walker_stats = {run_id: ts for run_id, _, ts in walker_data}
        n_converged_cv = sum(1 for ts in walker_stats.values() if ts['reached_unbound'])
        converged_biases = self.select_converged_biases(coverage_threshold)

        all_bias_list = list(all_biases.values())
        # Per-walker biasDirs hold complete, independent biases; a shared one holds
        # partial contributions. Summing the former multiplies the FES by n_walkers.
        _combine = "mean" if self.bias_layout(self.out_dir) == "isolated" else "sum"
        if len(all_bias_list) > 1:
            logger.info("Combining %d walker bias arrays by %s (%s biasDir layout).",
                        len(all_bias_list), _combine, self.bias_layout(self.out_dir))
        axis, fes_all_kj = self.compute_fes(all_bias_list, temperature, bias_factor,
                                              grid_min, grid_max, combine=_combine)
        n = len(axis)

        def _norm_plateau(fes_kj):
            plateau = fes_kj[-10:].mean()          # last 10 bins = unbound endpoint
            return (fes_kj - plateau) * KJ_TO_KCAL  # kcal/mol, unbound plateau = 0

        fes_all_kcal = _norm_plateau(fes_all_kj)

        # Coverage-filtered FES: useful for visual comparison (shows what walkers that
        # reached unbound contribute), but may have poor bound-basin coverage.
        _, fes_conv_kj = self.compute_fes(
            [b for _, b in converged_biases], temperature, bias_factor,
            grid_min, grid_max, combine=_combine
        )
        fes_conv_kcal = _norm_plateau(fes_conv_kj) if len(converged_biases) < len(all_biases) else None

        # All-walkers FES for Limongelli correction; see Notes in docstring.
        dG_result = None
        conv = None
        funnel_npz = os.path.join(self.out_dir, 'funnel_params.npz')
        if os.path.exists(funnel_npz):
            try:
                from openmm import unit as openmmunit
                _raw = np.load(funnel_npz, allow_pickle=True)
                _fp = {
                    'R_cylinder': float(_raw['R_cylinder_ang']) * openmmunit.angstrom,
                    'z_cc':       float(_raw['z_cc_ang'])       * openmmunit.angstrom,
                    'z_max':      float(_raw['z_max_ang'])       * openmmunit.angstrom,
                }
                # Per-walker ΔG(t) from the retained FES history: the Box 2 test.
                _plateau, _curves = None, {}
                _hists = sorted(glob(os.path.join(self.out_dir, "FEhist_*.npz")))
                _shared = self.bias_layout(self.out_dir) == "shared"
                if _shared and len(_hists) > 1:
                    _hists = self.shared_bias_run_order(_hists)
                    logger.info(
                        "Shared bias: laying %d walker curves end to end in run "
                        "order (%s); ΔG(t) spans the accumulated run, not 0-%s ns.",
                        len(_hists),
                        ", ".join(os.path.basename(h)[len("FEhist_"):-len(".npz")]
                                  for h in _hists), "mMD_time")
                _offset = 0.0
                _cat_t, _cat_g, _cat_s = [], [], []
                for _h in _hists:
                    _rid = os.path.basename(_h)[len("FEhist_"):-len(".npz")]
                    try:
                        _t, _g, _s = self.dG_timeseries(
                            _h, _fp, temperature=temperature,
                            grid_min=grid_min, grid_max=grid_max,
                            cv_length_per_unit_ang=cv_length_per_unit_ang)
                        if _shared:
                            _t = np.asarray(_t, dtype=float) + _offset
                            _offset = float(_t[-1]) if _t.size else _offset
                            _cat_t.append(_t); _cat_g.append(_g); _cat_s.append(_s)
                        _curves[_rid] = (_t, _g, _s)
                        # A walker only informs convergence if it saw both basins.
                        _ws = walker_stats.get(_rid, {})
                        _both = bool(_ws.get("frac_bound", 0) > 0.01
                                     and _ws.get("frac_unbound", 0) > 0.01)
                        _p = self.dG_plateau(_t, _g, sampled=_s,
                                             both_basins=_both)
                        logger.info(
                            "ΔG(t) for %s: %d usable snapshot(s), drift over the last "
                            "40%% = %.1f kJ/mol, both basins visited: %s -> %s",
                            _rid, _p["n_points"], _p["drift_kj"], _both,
                            "settled" if _p["settled"] else "NOT settled")
                        # the longest-running walker sets the verdict
                        _better = (_plateau is None
                                   or (_p["both_basins"] and not _plateau["both_basins"])
                                   or (_p["both_basins"] == _plateau["both_basins"]
                                       and _p["n_points"] > _plateau["n_points"]))
                        if _better:
                            _plateau = _p
                    except Exception as _hist_err:
                        logger.warning("ΔG(t) for %s failed: %s", _rid, _hist_err)
                if _shared and _cat_t:
                    # One bias, one estimate: the plateau test belongs on the whole
                    # accumulated curve, not on any single walker's slice of it.
                    _ct = np.concatenate(_cat_t)
                    _cg = np.concatenate(_cat_g)
                    _cs = np.concatenate(_cat_s)
                    _any_both = any(
                        bool(_w.get("frac_bound", 0) > 0.01
                             and _w.get("frac_unbound", 0) > 0.01)
                        for _w in walker_stats.values())
                    _plateau = self.dG_plateau(_ct, _cg, sampled=_cs,
                                               both_basins=_any_both)
                    logger.info(
                        "ΔG(t) over the accumulated shared-bias run (%.2f ns, %d "
                        "resolved of %d): drift over the last 40%% = %.1f kJ/mol -> %s",
                        float(_ct[-1]), int(_cs.sum()), _cs.size,
                        _plateau["drift_kj"],
                        "settled" if _plateau["settled"] else "NOT settled")
                if _curves:
                    self._plot_dG_convergence(_curves, out_prefix)

                conv = self.assess_convergence(
                    [ts for _, _, ts in walker_data], fes_all_kj,
                    grid_min=grid_min, grid_max=grid_max,
                    dG_plateau_result=_plateau)
                if conv["converged"]:
                    logger.info(
                        "Convergence OK: %d round trip(s), solvated plateau flat to "
                        "%.1f kJ/mol.", conv["n_round_trips"], conv["plateau_span_kj"])
                else:
                    logger.warning(
                        "NOT CONVERGED by the Limongelli criteria -- treat the ΔG "
                        "below as indicative only: %s", "; ".join(conv["reasons"]))
                dG_result = correct_fe_for_funnel(
                    fes_all_kj, _fp,
                    temperature=temperature,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    cv_length_per_unit_ang=cv_length_per_unit_ang,
                )
            except Exception as _fe_err:
                logger.warning(f"Limongelli correction failed (non-fatal): {_fe_err}")

        n_w = len(walker_data)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_w, 1)))

        fig = plt.figure(figsize=(14, 11))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50,
                               height_ratios=[2, 1.5, 2])

        # Panel 1 — CV traces
        ax1 = fig.add_subplot(gs[0])
        _bt = grid_min + bound_threshold * (grid_max - grid_min)
        _ut = grid_min + unbound_threshold * (grid_max - grid_min)
        ax1.axhspan(grid_min, _bt, alpha=0.07, color='royalblue')
        ax1.axhspan(_ut, grid_max, alpha=0.07, color='tomato')
        ax1.axhline(_bt, color='royalblue', lw=0.7, ls='--', alpha=0.5)
        ax1.axhline(_ut, color='tomato',    lw=0.7, ls='--', alpha=0.5)
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
        ax1.set_title("CV traces  (solid = converged,  dashed = stuck)")
        ax1.legend(fontsize=8, loc='upper left', framealpha=0.85)

        # Panel 2 — Self-bias per walker
        ax2 = fig.add_subplot(gs[1])
        for k, (wid, barr) in enumerate(sorted(all_biases.items())):
            cov = self.bias_coverage(barr)
            ax2.plot(axis, barr * KJ_TO_KCAL, color=colors[k % len(colors)], lw=1.5,
                      label=f"id={wid}  cov={100*cov:.0f}%", alpha=0.85)
        ax2.set_xlabel(cv_name)
        ax2.set_ylabel("Self-bias (kcal/mol)")
        ax2.set_title("Self-bias per walker")
        ax2.legend(fontsize=8)

        # Panel 3 — Combined vs converged-only FES + ΔG°_b annotation
        ax4 = fig.add_subplot(gs[2])
        dG_all = float(fes_all_kcal.min())
        ax4.plot(axis, fes_all_kcal, 'k-', lw=2.0,
                  label=f"All walkers (n={len(all_biases)})  ΔG_sim={dG_all:.1f} kcal/mol")
        if fes_conv_kcal is not None:
            dG_conv = float(fes_conv_kcal.min())
            ax4.plot(axis, fes_conv_kcal, 'g--', lw=2.0,
                      label=f"Coverage-filtered (n={len(converged_biases)})  ΔG_sim={dG_conv:.1f} kcal/mol")
        ax4.axhline(0, color='gray', lw=0.7, ls=':')
        ax4.axvline(_bt, color='royalblue', lw=0.7, ls='--', alpha=0.4)
        ax4.axvline(_ut, color='tomato',    lw=0.7, ls='--', alpha=0.4)
        if dG_result is not None:
            dG_std_kcal = dG_result['dG_bind_std_kj_mol'] * KJ_TO_KCAL
            pKd = dG_result['pKd']
            cv_min = dG_result['cv_at_minimum']
            bound_bins_plot = max(1, int(0.20 * len(fes_all_kcal)))
            idx_best = int(np.argmin(fes_all_kcal[:bound_bins_plot]))
            ax4.annotate(
                f"ΔG°_b={dG_std_kcal:.1f} kcal/mol  pKd={pKd:.2f}",
                xy=(axis[idx_best], float(fes_all_kcal[idx_best])),
                xytext=(max(0.05, axis[idx_best] + 0.15),
                        float(fes_all_kcal[idx_best]) + abs(float(fes_all_kcal.ptp())) * 0.12),
                fontsize=9, color='darkgreen',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='darkgreen'),
            )
        else:
            idx_min = int(np.argmin(fes_all_kcal))
            ax4.annotate(f"ΔG_sim={dG_all:.1f}", xy=(axis[idx_min], dG_all),
                          xytext=(axis[idx_min] + 0.1, dG_all + 1.5), fontsize=9,
                          arrowprops=dict(arrowstyle='->', lw=0.8))
        ax4.set_xlabel(cv_name)
        ax4.set_ylabel("ΔG (kcal/mol, unbound plateau = 0)")
        title_dg = (f"  ·  ΔG°_b={dG_result['dG_bind_std_kj_mol']*KJ_TO_KCAL:.1f} kcal/mol,"
                    f"  pKd={dG_result['pKd']:.2f}" if dG_result else "")
        ax4.set_title(
            f"FES  ·  {n_converged_cv}/{n_w} walkers converged{title_dg}"
        )
        ax4.legend(fontsize=9)

        plt.suptitle(
            f"WT-MetaD Diagnosis — {os.path.basename(self.out_dir)}",
            fontsize=11, fontweight='bold'
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
        logging.info(f"  ΔG_sim (all walkers)              = {dG_all:.2f} kcal/mol")
        if fes_conv_kcal is not None:
            logging.info(f"  ΔG_sim (coverage-filtered)        = {float(fes_conv_kcal.min()):.2f} kcal/mol")
        if dG_result is not None:
            dG_std_kcal = dG_result['dG_bind_std_kj_mol'] * KJ_TO_KCAL
            conv_note = " [⚠ FES min not in bound basin]" if dG_result['cv_at_minimum'] > 0.3 else ""
            logging.info(
                f"  ΔG°_b (Limongelli, {len(converged_biases)} converged walkers) "
                f"= {dG_std_kcal:.2f} kcal/mol  pKd={dG_result['pKd']:.2f}{conv_note}"
            )
        logging.info("────────────────────────────────────────────────────────")

        return {
            'walker_stats':     walker_stats,
            'fes_all_kcal':     fes_all_kcal,
            'fes_conv_kcal':    fes_conv_kcal,
            'cv_axis':          axis,
            'n_converged_cv':   n_converged_cv,
            'n_walkers':        n_w,
            'dG_bind_std_kcal': dG_result['dG_bind_std_kj_mol'] * KJ_TO_KCAL if dG_result else None,
            'pKd':              dG_result['pKd'] if dG_result else None,
            'fes_min_at_cv':    dG_result['cv_at_minimum'] if dG_result else None,
            # Carries the Limongelli convergence verdict alongside the number, so a
            # caller cannot quote ΔG without seeing whether it is interpretable.
            'convergence':      conv,
        }
