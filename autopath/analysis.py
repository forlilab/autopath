import os
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
# plt.rcParams["axes.edgecolor"] = 'black'

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF

import logging
logger = logging.getLogger("autopath")

def plot_colvar(out_dir:str=None, colvar_name:str=None):
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


def plot_bias(out_dir:str=None, x_min:float=None, x_max:float=None, grid_points:int=None, colvar_name:str=None):
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
        return  # funnel not used or correction not applied yet

    data = []
    for f in files:
        walker_name = os.path.splitext(os.path.basename(f))[0]
        np_data = np.load(f) * 0.239006  # kJ/mol → kcal/mol
        df = pd.DataFrame(np_data, columns=["FE"])
        df.index = axis_values
        df["walker"] = walker_name
        data.append(df)

    # Best estimate from the last (most converged) walker
    last_fe_kcal = np.load(files[-1]) * 0.239006
    #TODO this can be improved by fitting the plateau and minimum to get a more robust estimate of the ΔG range, but this is a quick approximation
    mindG_std_kcal = float(last_fe_kcal.min())
    maxdG_std_kcal = float(last_fe_kcal.max())
    dG_std_kcal = maxdG_std_kcal - mindG_std_kcal
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

        sns.heatmap(df, cmap="Spectral")  # , vmin=0, vmax=-8)
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
    plt.xlabel('Step #'); plt.ylabel(xCV_name)
    plt.title(f'{xCV_name} vs. Step # - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_{xCV_name}_COLVAR.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data, x=data.index, y=yCV_name, hue="walker")
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.xlabel('Step #'); plt.ylabel(yCV_name)
    plt.title(f'{yCV_name} vs. Step # - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_{yCV_name}_COLVAR.png")
    plt.close()

    return


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
        ``"R_cylinder"`` as an OpenMM Quantity in angstroms.
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

    # Accept raw array, single path, or list of paths.
    # When multiple paths are given each file already contains the accumulated bias
    # from all previous sequential walkers (shared biasDir), so the last file is the
    # most converged estimate and is used for the correction.
    if isinstance(fe_files, np.ndarray):
        fe = fe_files.astype(float)
        n_files = 1
    else:
        if isinstance(fe_files, str):
            fe_files = [fe_files]
        fe = np.load(fe_files[-1]).astype(float)   # last = most converged
        n_files = len(fe_files)

    cv = np.linspace(grid_min, grid_max, len(fe))

    # Normalise: simulation unbound plateau → 0
    plateau_val = float(fe[-n_plateau_points:].mean())

    # Guard: if the plateau is within ~10 kJ/mol of the global FE maximum, the unbound
    # state was never sampled (no bias deposited there, so FE ≈ 0).  The normalization
    # would be meaningless and ΔG would be the raw bias depth, not a binding free energy.
    fe_max = float(fe.max())
    unsampled_gap = fe_max - plateau_val
    if unsampled_gap < 10.0:
        raise ValueError(
            f"Unbound state not sampled: plateau ({plateau_val:.1f} kJ/mol) is only "
            f"{unsampled_gap:.1f} kJ/mol below the FE maximum ({fe_max:.1f} kJ/mol). "
            "The walker did not reach the unbound end of the CV; ΔG would be meaningless."
        )

    fe_norm = fe - plateau_val  # bound minimum → negative; unbound → ≈0

    # Simulation binding ΔG
    min_idx = int(np.argmin(fe_norm))
    dG_bind_sim = float(fe_norm[min_idx])
    cv_at_min = float(cv[min_idx])

    # R_cylinder: OpenMM Quantity → Å
    R_cyl_qty = funnel_params["R_cylinder"]
    R_cyl_ang = float(R_cyl_qty.value_in_unit(openmmunit.angstrom))

    # Standard-state correction (Limongelli 2013 Eq. 3)
    A_cyl = math.pi * R_cyl_ang ** 2            # Å²
    correction = kT * math.log(A_cyl * C0_per_A3)  # kJ/mol; negative for typical R

    # Corrected FE array: unbound plateau → |correction|; bound min → ΔG°_b
    fe_rw = fe_norm - correction

    dG_bind_std = float(fe_rw.min())            # == dG_bind_sim - correction
    pKd = -dG_bind_std / (kT * math.log(10))

    logger.info(
        f"Funnel correction: ΔG_sim={dG_bind_sim:.2f} kJ/mol, "
        f"correction={correction:.2f} kJ/mol (R_cyl={R_cyl_ang:.1f} Å), "
        f"ΔG°_b={dG_bind_std:.2f} kJ/mol, pKd={pKd:.2f}"
    )

    return {
        "fe_corrected": fe_rw,
        "dG_bind_sim_kj_mol": dG_bind_sim,
        "correction_kj_mol": correction,
        "dG_bind_std_kj_mol": dG_bind_std,
        "R_cylinder_ang": R_cyl_ang,
        "pKd": pKd,
        "cv_at_minimum": cv_at_min,
        "n_fe_files": n_files,
        "temperature_K": temperature,
    }
