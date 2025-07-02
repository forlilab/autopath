import os
import numpy as np
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF
from rdkit.Chem import rdMolDescriptors
import numpy as np

from scipy.stats import pearsonr

def get_crippen_contributions(ligand_mol):
    """Calculate the CLogP contribution of each atom using RDKit's Crippen module."""
    atom_contribs = rdMolDescriptors._CalcCrippenContribs(ligand_mol)
    atom_coords = [ligand_mol.GetConformer().GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())]
    return [(atom_contribs[i][0], atom_coords[i]) for i in range(ligand_mol.GetNumAtoms())]

def calculate_lipophilicity_moment(ligand_input):
    """Calculate LMZ and centroid Z from an SDF file or RDKit molecule object."""
    if isinstance(ligand_input, str):  # If input is an SDF file path
        supplier = Chem.SDMolSupplier(ligand_input, removeHs=False)
        ligand_mol = supplier[0]
        if ligand_mol is None:
            raise ValueError("Failed to read molecule from SDF file.")
    elif isinstance(ligand_input, Chem.Mol):  # If input is an RDKit Mol object
        ligand_mol = ligand_input
    else:
        raise TypeError("ligand_input must be an SDF file path or an RDKit Mol object.")

    # Get the ClogP contributions
    clogp_contributions = get_crippen_contributions(ligand_mol)
    clogp_values, atom_coords = zip(*clogp_contributions)

    # Compute centroid coordinates
    N = len(ligand_mol.GetAtoms())
    centroid_x = sum(atom.x for atom in atom_coords) / N
    centroid_y = sum(atom.y for atom in atom_coords) / N
    centroid_z = sum(atom.z for atom in atom_coords) / N

    # Compute lipophilicity moments
    delta_lipophilicity_x = sum((atom.x - centroid_x) * clogp for atom, clogp in zip(atom_coords, clogp_values))
    delta_lipophilicity_y = sum((atom.y - centroid_y) * clogp for atom, clogp in zip(atom_coords, clogp_values))
    delta_lipophilicity_z = sum((atom.z - centroid_z) * clogp for atom, clogp in zip(atom_coords, clogp_values))

    dot_product = delta_lipophilicity_z
    vector_magnitude = np.sqrt(delta_lipophilicity_x**2 + delta_lipophilicity_y**2 + delta_lipophilicity_z**2)

    # Compute LMZ (avoid division by zero)
    LMZ = dot_product / vector_magnitude if vector_magnitude != 0 else 0

    return LMZ, centroid_z*0.1



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
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rmsd_df, y="rmsd", x=rmsd_df.index)
    plt.ylabel("RMSD (nm)");     plt.xlabel("Frame #")
    plt.title(f"RMSD {sys_name}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_rmsd.png")
    plt.close()
    return


def plot_colvar(out_dir:str=None, colvar_name:str=None):
    sys_name = out_dir.split("/")[0]
    files = glob(f"{out_dir}/COLVAR_*npy")
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

    sys_name = out_dir.split("/")[0]
    axis_values = np.linspace(x_min, x_max, grid_points)
    axis_values = [round(i, 2) for i in axis_values]

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

def plot_bias_2D(
    out_dir, x_min, x_max, x_grid_points, xCV_name, y_min, y_max, y_grid_points, yCV_name
):
    sys_name = out_dir.split("/")[0]
    x_values = np.linspace(x_min, x_max, x_grid_points)
    y_values = np.linspace(y_min, y_max, y_grid_points)

    files_bias = glob(f"{out_dir}/bias_*")
    for i, f in enumerate(files_bias):
        walker_name = os.path.splitext(os.path.basename(f))[0]

        np_data = np.load(f)
        np_data = np_data * 0.239006  # Convert KJ to Kcal
        df = pd.DataFrame(np_data)

        df.index, df.columns = y_values, x_values

        sns.heatmap(df, cmap="Spectral")

        plt.title(f"Metadynamics Bias - {sys_name} - {walker_name}", fontsize=12)
        plt.xlabel(xCV_name)
        plt.ylabel(yCV_name)

        # Exactly 10 ticks, nicely spaced:
        x_tick_vals = np.round(np.linspace(x_min, x_max, 10), 1)
        y_tick_vals = np.round(np.linspace(y_min, y_max, 10), 1)

        # Add 0.0 if it’s not already in tick_vals and in range:
        if (x_min < 0 < x_max) and 0.0 not in x_tick_vals:
            x_tick_vals = np.append(x_tick_vals, 0.0)
            x_tick_vals = np.sort(np.unique(x_tick_vals))

        if (y_min < 0 < y_max) and 0.0 not in y_tick_vals:
            y_tick_vals = np.append(y_tick_vals, 0.0)
            y_tick_vals = np.sort(np.unique(y_tick_vals))

        # Set ticks on heatmap axes
        plt.xticks(
            ticks=[np.argmin(np.abs(np.array(x_values) - val)) for val in x_tick_vals],
            labels=[f"{val:.1f}" for val in x_tick_vals],
        )
        plt.yticks(
            ticks=[np.argmin(np.abs(np.array(y_values) - val)) for val in y_tick_vals],
            labels=[f"{val:.1f}" for val in y_tick_vals],
        )

        # Make ticks show up big and clear
        plt.tick_params(axis='both', length=8, width=2, direction='out')

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{sys_name}_{walker_name}_bias.png")
        plt.close()

    return
def assess_symmetry(out_dir, x_min, x_max, x_grid_points, xCV_name, 
               y_min, y_max, y_grid_points, yCV_name):
    """
    Plots the 2D free energy surface and optionally adds a point for LMZ vs. centroid Z 
    if an SDF file is provided.
    """
    sys_name = out_dir.split("/")[0]

    x_values = np.linspace(x_min, x_max, x_grid_points)
    x_values = [round(i, 2) for i in x_values]

    y_values = np.linspace(y_min, y_max, y_grid_points)
    y_values = [round(i, 2) for i in y_values]



    files_fe = glob(f"{out_dir}/FE_*.npy")
    for i, f in enumerate(files_fe):
        walker_name = os.path.splitext(os.path.basename(f))[0]

        np_data = np.load(f) * 0.239006  # Convert KJ to Kcal
        df = pd.DataFrame(np_data, index=y_values, columns=x_values)
        # Determine the number of rows
        num_rows = df.shape[0]

        # Split the DataFrame into top and bottom halves
        if num_rows % 2 == 0:
            top_half = df.iloc[:num_rows // 2, :]
            bottom_half = df.iloc[num_rows // 2:, :]
        else:
            top_half = df.iloc[:num_rows // 2, :]
            bottom_half = df.iloc[num_rows // 2 + 1:, :]

        # Flip the bottom half vertically to align with the top half
        bottom_half_flipped = bottom_half.iloc[::-1].reset_index(drop=True)

        # Ensure both halves have the same number of rows
        if top_half.shape[0] != bottom_half_flipped.shape[0]:
            raise ValueError("The two halves have mismatched dimensions.")

        # Compute the absolute differences between corresponding elements
        abs_diff = np.abs(top_half.to_numpy() - bottom_half_flipped.to_numpy())

        top = top_half.to_numpy().flatten()
        bottom = bottom_half_flipped.to_numpy().flatten()
        r, _ = pearsonr(top, bottom)
        print(f"Symmetry (Pearson r): {r:.4f}")

        # Calculate the Mean Absolute Difference
        mad = np.mean(abs_diff)
        print(f"Horizontal Symmetry Metric (MAD) {sys_name} - {walker_name}: {mad:.4f}")


def plot_FE_2D(out_dir, x_min, x_max, x_grid_points, xCV_name, 
               y_min, y_max, y_grid_points, yCV_name, sdf_file=None):
    """
    Plots the 2D free energy surface and optionally adds a point for LMZ vs. centroid Z 
    if an SDF file is provided.
    """
    sys_name = out_dir.split("/")[0]

    x_values = np.linspace(x_min, x_max, x_grid_points)
    x_values = [round(i, 2) for i in x_values]

    y_values = np.linspace(y_min, y_max, y_grid_points)
    y_values = [round(i, 2) for i in y_values]

    # Calculate LMZ and centroid Z if an SDF file is provided
    lmz, centroid_z = None, None
    if sdf_file:
        lmz, centroid_z = calculate_lipophilicity_moment(sdf_file)
        print(f'lmz: {lmz}')
        print(f'centroid_z: {centroid_z}')

    files_fe = glob(f"{out_dir}/FE_*.npy")
    for i, f in enumerate(files_fe):
        walker_name = os.path.splitext(os.path.basename(f))[0]

        np_data = np.load(f) * 0.239006  # Convert KJ to Kcal
        df = pd.DataFrame(np_data, index=y_values, columns=x_values)


        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, cmap="Spectral")
        plt.title(f"Free Energy - {sys_name} - {walker_name}", fontsize=12)
        plt.xlabel(xCV_name)
        plt.ylabel(yCV_name)

        # Plot LMZ vs. Centroid Z point
        if sdf_file and lmz is not None and centroid_z is not None:
            x_idx = np.argmin(np.abs(np.array(x_values) - centroid_z))

            # Find the closest index of lmz in y_values
            y_idx = np.argmin(np.abs(np.array(y_values) - lmz))

            # Plot LMZ vs. Centroid Z point using the found indices
            plt.scatter(x_idx, y_idx, color='black', label=f"LMZ: {lmz:.2f}, Centroid Z: {centroid_z:.2f}")

            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{sys_name}_{walker_name}.png")
        plt.close()


def compute_1d_pmf(F_2D, axis, T=310):
    """
    Compute 1D PMF from a 2D PMF by integrating along one axis.

    Parameters:
        F_2D (np.ndarray): 2D PMF array
        axis (int): Axis to integrate over (0 for y, 1 for x)
        T (float): Temperature in Kelvin (default: 310 K)

    Returns:
        np.ndarray: 1D PMF along the chosen axis
    """
    k_B_kcal = 0.0019872041  # Boltzmann constant in kcal/mol·K
    kT = k_B_kcal * T

    P = np.exp(-F_2D / kT)  # Convert free energy to probability
    P_1D = np.trapz(P, axis=axis)  # Integrate over selected axis
    F_1D = -kT * np.log(P_1D)  # Convert back to free energy
    return F_1D

def plot_FE_2D_with_marginals(out_dir, x_min, x_max, x_grid_points, xCV_name,
                               y_min, y_max, y_grid_points, yCV_name, sdf_file=None, symmetrize=False):
    """
    Plots the 2D free energy surface with 1D PMFs along the x (top) and y (right) axes.
    """
    sys_name = out_dir.split("/")[0]

    # Define independent axes for each plot
    x_values = np.linspace(x_min, x_max, x_grid_points)
    y_values = np.linspace(y_min, y_max, y_grid_points)



    files_fe = glob(f"{out_dir}/FE_*.npy")
    for f in files_fe:
        walker_name = os.path.splitext(os.path.basename(f))[0]
        F_2D = np.load(f) * 0.239006  # Convert KJ to Kcal

        if symmetrize:
            # Flip F_2D over x = 0 (axis=1)
            flipped_F2D = np.flip(F_2D, axis=1)
            sym_F2D = 0.5 * (F_2D + flipped_F2D)
            
            F_x = compute_1d_pmf(sym_F2D, axis=0)
            F_x_orig = compute_1d_pmf(F_2D, axis=0)
            F_x_flip = compute_1d_pmf(flipped_F2D, axis=0)
            F_x_diff = np.abs(F_x_orig - F_x_flip)
        else:
            F_x = compute_1d_pmf(F_2D, axis=0)
        F_y = compute_1d_pmf(F_2D, axis=1)  # 1D PMF along y

        # Adjusted figure size for better visibility
        fig = plt.figure(figsize=(14, 10))  # Increase figure size for better clarity
        gs = fig.add_gridspec(2, 3, width_ratios=[0.3, 4, 1], height_ratios=[1, 4],
                              wspace=0.4, hspace=0.1)  

        ### Main 2D Heatmap ###
        ax_main = fig.add_subplot(gs[1, 1])
        cbar_ax = fig.add_subplot(gs[1, 0])  # Separate axis for color bar

        # Create the heatmap
        heatmap = sns.heatmap(F_2D, cmap="Spectral", cbar=True, ax=ax_main, 
                              cbar_ax=cbar_ax, cbar_kws={'label': 'Free Energy (kcal/mol)', 'location': 'left'})
        # Increase font size of the colorbar label and ticks

        # Add contours for better visualization
        contour_levels = np.linspace(np.min(F_2D), np.max(F_2D), 10)
        X, Y = np.meshgrid(np.arange(x_grid_points), np.arange(y_grid_points))
        ax_main.contour(X, Y, F_2D, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.6)


        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # Increase font size for the ticks
        cbar.set_label('Free Energy (kcal/mol)', fontsize=16)  # Increase font size for the label


        # Set x and y axis labels with larger fonts
        ax_main.set_xlabel(xCV_name, fontsize=16)
        ax_main.set_ylabel(yCV_name, fontsize=16)

        # Desired tick labels for the x and y axes
        # x_tick_labels = [-3, -2, -1, 0, 1, 2, 3]
        # y_tick_labels = [1, 0.5, 0, -0.5, -1]
        # Define nice min/max ranges for ticks
        x_tick_min = np.floor(x_min)
        x_tick_max = np.ceil(x_max)
        y_tick_min = np.floor(y_min)
        y_tick_max = np.ceil(y_max)

        # Create nice evenly spaced tick labels
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, 9)
        y_tick_labels = np.linspace(y_tick_max, y_tick_min, 5)  # Flip for heatmap orientation


        # Map these labels to indices in the heatmap data
        x_tick_positions = np.interp(x_tick_labels, np.linspace(x_min, x_max, x_grid_points), np.arange(x_grid_points))
        y_tick_positions = np.interp(y_tick_labels, np.linspace(y_min, y_max, y_grid_points), np.arange(y_grid_points))

        # Set the positions and labels for the x and y axes with larger fonts
        ax_main.set_xticks(x_tick_positions)
        ax_main.set_xticklabels(x_tick_labels, fontsize=14)

        ax_main.set_yticks(y_tick_positions)
        ax_main.set_yticklabels(y_tick_labels, fontsize=14)

        # Crop the heatmap's y-axis from -1 to 1 (adjust based on the position of the labels)
        ax_main.set_ylim(y_tick_positions[4], y_tick_positions[0])  # y_tick_positions[0] is -1 and [4] is 1

        ### X-axis Marginal PMF (Top) ###
        ax_top = fig.add_subplot(gs[0, 1])
        if symmetrize:
            ax_top.plot(x_values, F_x, color='black', label='Symmetrized PMF')
            ax_top.fill_between(x_values, F_x - F_x_diff/2, F_x + F_x_diff/2,
                                color='gray', alpha=0.3, label='Symmetry error')
            ax_top.legend(fontsize=12)
        else:
            ax_top.plot(x_values, F_x, color='black')

        ax_top.set_xlim(x_values[1], x_values[-1])  # Independent x-scale
        ax_top.set_ylabel("Free Energy (kcal/mol)", fontsize=16)
        # Set more y-ticks and enable grid lines
        y_ticks_dense = np.linspace(np.min(F_x), np.max(F_x), 4)  # Increase to 10 tick levels
        ax_top.set_yticks(y_ticks_dense)

        # Set tick labels on the top plot with larger font
        ax_top.tick_params(axis='both', labelsize=14)

        ### Y-axis Marginal PMF (Right) ###
        ax_right = fig.add_subplot(gs[1, 2])
        ax_right.plot(F_y, y_values, color='black')
        ax_right.set_xlabel("Free Energy (kcal/mol)", fontsize=16)

        # Crop the y-axis of the marginal PMF plot from -1 to 1
        ax_right.set_ylim(-1, 1)  # Set y-axis limits to the desired range
        #ax_right.set_ylabel("Free Energy (kcal/mol)", fontsize=16)

        # Set tick labels on the right plot with larger font
        ax_right.tick_params(axis='both', labelsize=14)

        if sdf_file:
            lm_z, centroid_z = calculate_lipophilicity_moment(sdf_file)

            # Plot horizontal line on the right plot at lm_z
            ax_right.axhline(y=lm_z, color='red', linestyle='--', linewidth=2)

            # Plot vertical line on the top plot at centroid_z
            ax_top.axvline(x=centroid_z, color='red', linestyle='--', linewidth=2)

            # Plot point on the heatmap at (centroid_z, lm_z)
            ax_main.plot(np.interp(centroid_z, np.linspace(x_min, x_max, x_grid_points), np.arange(x_grid_points)),
                         np.interp(lm_z, np.linspace(y_min, y_max, y_grid_points), np.arange(y_grid_points)),
                         'ko', markersize=8)

        # Adjust title font size and position
        plt.suptitle(f"Free Energy - {sys_name} - {walker_name}", fontsize=18, y=0.95)

        # Save and close
        plt.savefig(f"{out_dir}/{sys_name}_{walker_name}_pmf.png")
        plt.close()


def plot_colvar_2D(out_dir, xCV_name, yCV_name):
    sys_name = out_dir.split("/")[0]
    files = glob(f"{out_dir}/COLVAR_*npy")
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

def plot_colvar_scatter(out_dir, xCV_name, yCV_name):
    sys_name = out_dir.split("/")[0]
    files = glob(f"{out_dir}/COLVAR_*npy")
    data = []

    for f in files:
        walker_name = os.path.basename(f).split(".")[0]
        np_data = np.load(f)
        df = pd.DataFrame(np_data, columns=[xCV_name, yCV_name])
        df["walker"] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)

    plt.figure(figsize=(8, 6))
    walkers = data["walker"].unique()
    palette = sns.color_palette("husl", len(walkers))

    for i, walker in enumerate(walkers):
        walker_data = data[data["walker"] == walker]
        plt.scatter(
            walker_data[xCV_name],
            walker_data[yCV_name],
            label=walker,
            alpha=0.05,
            edgecolors="black",
            linewidths=0.25,
            color=palette[i],
            s=20
        )

    plt.xlabel(xCV_name)
    plt.ylabel(yCV_name)
    plt.title(f"{xCV_name} vs. {yCV_name} - {sys_name}")
    plt.legend(title="Walker", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_{xCV_name}_vs_{yCV_name}_scatter.png", dpi=300)
    plt.close()

def plot_colvar_lines_by_time(out_dir, xCV_name, yCV_name):
    files = glob(f"{out_dir}/COLVAR_*.npy")
    if not files:
        raise FileNotFoundError(f"No COLVAR_*.npy files found in {out_dir}")

    num_files = len(files)
    ncols = 2
    nrows = (num_files + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()  # Make it easy to index

    for idx, f in enumerate(files):
        np_data = np.load(f)
        df = pd.DataFrame(np_data, columns=[xCV_name, yCV_name])
        ax = axes[idx]

        # Normalize time steps to [0, 1] for colormap
        timesteps = np.arange(len(df))
        norm = mcolors.Normalize(vmin=timesteps.min(), vmax=timesteps.max())
        cmap = cm.get_cmap("viridis")

        # Plot colored line segments by time
        for i in range(len(df) - 1):
            x_vals = [df.iloc[i][xCV_name], df.iloc[i + 1][xCV_name]]
            y_vals = [df.iloc[i][yCV_name], df.iloc[i + 1][yCV_name]]
            color = cmap(norm(timesteps[i]))
            ax.plot(x_vals, y_vals, color=color, linewidth=1)

        walker_name = os.path.basename(f).split(".")[0]
        ax.set_title(walker_name)
        ax.set_xlabel(xCV_name)
        ax.set_ylabel(yCV_name)

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{xCV_name} vs. {yCV_name} - colored by timestep", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(out_dir, f"COLVAR_lines_by_time_{xCV_name}_vs_{yCV_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()





def _extract_sMD_statistics(files: list = None) -> pd.DataFrame:
    data = []
    for f in files:
        run_n = os.path.splitext(os.path.basename(f))[0].split("_")[-2]

        df = pd.read_csv(f, names=["r0", "com_dist", "force", "work"])  # [:500]
        df["replica"] = f"rep_{run_n}"
        df.reset_index(inplace=True, drop=False)
        data.append(df)

    data = pd.concat(data, axis=0)
    data["time"] = data["index"] / 1000  # ps to ns
    data.reset_index(inplace=True, drop=True)

    return data

def plot_sMD_statistics(files: list = None, sys_name:str=None, out_dir:str=None,return_data:bool=False) -> None:

    # Get statistics from sMD .dat files
    data = _extract_sMD_statistics(files)

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="r0", y="work", hue="replica")
    plt.xlabel("r0 dist (nm)")
    plt.ylabel("Work (KJ/mol/nm)")
    plt.title(f'r0 vs Work - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-r0_vs_work.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="r0", y="force")  # , hue='replica')
    plt.xlabel("r0 dist (nm)")
    plt.ylabel("Force (KJ/mol/nm^2)")
    plt.title(f'r0 vs Force - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-r0_vs_force.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.lineplot(data, x="com_dist", y="work", hue="replica")
    plt.xlabel("COM dist (nm)")
    plt.ylabel("Work (KJ/mol/nm)")
    plt.title(f'COM vs Work - {sys_name}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}-com_vs_work.png")
    plt.close()

    # plt.figure(figsize=(6, 5))
    # sns.lineplot(data, x="com_dist", y="force")  # , hue='replica')
    # plt.xlabel("COM dist (nm)")
    # plt.ylabel("Force (KJ/mol)")
    # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f"{out_dir}/{sys_name}-com_vs_force.png")
    # plt.close()

    # plt.figure(figsize=(6,4))
    # sns.relplot(data, x='com_dist', y='work', hue='replica', col='replica', kind='line')
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
    # sns.lineplot(data, x=data['time'], y='com_dist', hue='replica')
    # plt.xlabel('time (ns)'); plt.ylabel('COM dist (nm)')
    # plt.title(sys_name)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/{sys_name}-time_vs_com.png')
    # plt.close()

    if return_data:
        return data
    else:
        return None