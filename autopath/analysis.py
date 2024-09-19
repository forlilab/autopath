import os
import numpy as np
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF


def plot_atomic_rmsf(u, lig_resname:str='UNK', outname:str='rmsf.png', log_rmsf:bool=False):
    """
    Draws a RMSF (Root Mean Square Fluctuation) plot for a specified ligand and saves it as an image file.
    Parameters:
    -----------
    u : MDAnalysis.Universe
        The MDAnalysis universe object containing the molecular dynamics trajectory and topology.
    lig_resname : str, optional
        The residue name of the ligand to analyze (default is 'UNK').
    outname : str, optional
        The name of the output image file where the RMSF plot will be saved (default is 'rmsf.png').
    log_rmsf : bool, optional
        If True, logs the RMSF values to a CSV file with the same name as the output image (default is False).
    Returns:
    --------
    None
        This function does not return any value. It saves the RMSF plot and optionally logs the RMSF values.
    """

    lig_select = u.select_atoms(f'resname {lig_resname}')
    r = RMSF(atomgroup=lig_select).run()
    probe_mol = lig_select.convert_to('RDKIT')
    probe_mol.Compute2DCoords()
    probe_mol = Chem.RemoveHs(probe_mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(probe_mol, r.rmsf, step=0.01, alpha=0.3, contourLines=5) 
    fig.savefig(outname, bbox_inches='tight')
    
    # Optionally, log the RMSF values for further analysis
    if log_rmsf:
        log_fname = os.path.splitext(outname)[0]
        with open(f'{log_fname}.csv', 'w') as f:
            for res_id, rmsf_value in enumerate(r.rmsf):
                f.write(f'{res_id},{rmsf_value:.3f}\n')

    return

def plot_rmsd(rmsd_df:pd.DataFrame=None,
              sys_name:str=None,
              out_dir:str=None) -> None:
    rmsd_df["rmsd"] = rmsd_df["rmsd"] * 10  # nM to A
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rmsd_df, y="rmsd", x=rmsd_df.index)
    plt.ylabel("RMSD (A)");     plt.xlabel("Frame #")
    plt.title(f"RMSD {sys_name}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_rmsd.png")
    plt.close()
    return


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
    axis_values = [round(i * 10, 2) for i in axis_values]

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

    sys_name = out_dir.split("/")[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i * 10, 2) for i in axis_values]

    files = glob(f"{out_dir}/FE_*.npy")
    data = []
    for i, f in enumerate(files):
        walker_name = f.split("/")[2].split(".")[0]  # +str(i)
        np_data = np.load(f)
        np_data = np_data * 0.239006  # KJ to Kcal
        df = pd.DataFrame(np_data, columns=["FE"])
        df.index = axis_values
        df["walker"] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)
    plt.figure(figsize=(10, 4))
    sns.lineplot(data, x=data.index, y="FE", hue="walker")
    plt.ylabel("FE (Kcal/mol)")
    plt.xlabel(colvar_name)
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.title(f"Free Energy - {sys_name}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_FE.png")
    plt.close()
    return


def plot_FE_2D(
    out_dir, x_min, x_max, x_grid_points, x_name, y_min, y_max, y_grid_points, y_name
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
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(f"Free Energy - {sys_name}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{sys_name}_{walker_name}.png")
        # plt.show()
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

def plot_sMD_statistics(data:pd.DataFrame=None, sys_name:str=None, out_dir:str=None) -> None:

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="r0", y="work", hue="replica")
    plt.xlabel("r0 dist (nm)")
    plt.ylabel("Work (KJ/mol)")
    plt.title(f'r0 vs Work - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-r0_vs_work.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="COMDist", y="work", hue="replica")
    plt.xlabel("COM dist (nm)")
    plt.ylabel("Work (KJ/mol)")
    plt.title(f'COM vs Work - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-com_vs_work.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="r0", y="force")  # , hue='replica')
    plt.xlabel("r0 dist (nm)")
    plt.ylabel("Force (KJ/mol)")
    plt.title(f'r0 vs Force - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-r0_vs_force.png")
    plt.close()

    # plt.figure(figsize=(6, 5))
    # sns.lineplot(data, x="COMDist", y="force")  # , hue='replica')
    # plt.xlabel("COM dist (nm)")
    # plt.ylabel("Force (KJ/mol)")
    # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f"{out_dir}/{sys_name}-com_vs_force.png")
    # plt.close()

    # plt.figure(figsize=(6,4))
    # sns.relplot(data, x='COMDist', y='work', hue='replica', col='replica', kind='line')
    # plt.xlabel('COM dist (nm)'); plt.ylabel('Work (KJ/mol)')
    # # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/{sys_name}-com_vs_work.png')
    # plt.close()

    # plt.figure(figsize=(6,5))
    # sns.lineplot(data, x=data['time'], y='work', hue='replica')
    # plt.xlabel('time (ns)'); plt.ylabel('Work (KJ/mol)')
    # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/{sys_name}-time_vs_work.png')
    # plt.close()

    # plt.figure(figsize=(6,5))
    # sns.lineplot(data, x=data['time'], y='COMDist', hue='replica')
    # plt.xlabel('time (ns)'); plt.ylabel('COM dist (nm)')
    # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/{sys_name}-time_vs_com.png')
    # plt.close()

    return


def plot_clusters(df_clustered, closest_points, sys_name, out_dir):

    plt.scatter(
        df_clustered["rmsd"],
        df_clustered["cog_d"],
        marker="o",
        c=df_clustered["cluster"],
        alpha=0.5,
    )
    for idx, row in closest_points.iterrows():
        plt.scatter(row["rmsd"], row["cog_d"], marker="x", c="black", alpha=1, zorder=3)

    plt.xlabel("RMSD (nm)");    plt.ylabel("COG dist (nm)")
    plt.title(f"{sys_name} sMD centroids")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_sMD_cluster_centroids.png")
    plt.close()
    return
