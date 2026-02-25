from autopath.sMDAnalysis.SMDData import SMDData
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
style.use("fivethirtyeight")
plt.rcParams["savefig.facecolor"] = 'white'
plt.rcParams["savefig.edgecolor"] = 'white'
plt.rcParams["axes.facecolor"] = 'white'
# plt.rcParams["axes.edgecolor"] = 'black'

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import MDAnalysis as mda
from MDAnalysis.analysis import align, density
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger("autopath.sMDAnalysis.Diagnostics")

def plot_work_profiles(
    results: pd.DataFrame,
    r_coord: str = 'r_coord',
    cols_to_plot: list = ['Wmean', 'dG', 'Wdiss'],
    estimator: str = 'cumulant',
    outdir: str = 'analysis',
):

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"work_profiles_{estimator}.png")

    results = results[results['estimator'] == estimator]
    style_order = results['path'].unique().tolist()
    speeds = sorted(results['speed'].unique())
    
    fig, ax = plt.subplots(
        figsize=(12, 4),
        ncols=len(speeds),
        nrows=1,
        sharey=True,
        sharex=True
    )
    axes = ax.flatten() if len(speeds) > 1 else [ax]

    legend_handles, legend_labels = None, None

    for i, speed in enumerate(speeds):
        speed_df = results.loc[results['speed'] == speed, [r_coord, 'path', 'speed'] + cols_to_plot]

        # Long format: metric is {Wmean, dG, Wdiss}, value = corresponding y
        long_df = speed_df.melt(
            id_vars=[r_coord, 'path', 'speed'],
            value_vars=cols_to_plot,
            var_name=estimator,
            value_name='value'
        )

        sns.lineplot(
            data=long_df,
            x=r_coord, y='value',
            hue=estimator, style='path', style_order=style_order,
            estimator=None, errorbar=None,  # don't aggregate across paths
            ax=axes[i]
        )
        axes[i].grid(True)
        
        # # just reference for trypsin
        # axes[i].axhline(27, color='k', lw=1, ls='--')
        
        axes[i].set_title(f'Speed={speed} nm/ps', fontsize=12)
        axes[i].set_xlabel(f'{r_coord} (nm)')
        if i == 0:
            axes[i].set_ylabel('Energy (kJ/mol)')
        else:
            axes[i].set_ylabel('')

        # Capture legend once, then remove per-axes legends
        if legend_handles is None:
            legend_handles, legend_labels = axes[i].get_legend_handles_labels()
        axes[i].legend_.remove()

    # Figure-level legend combining hue (metrics) and style (paths)
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            title='',
            bbox_to_anchor=(1.01, 0.8), loc='upper left',
            borderaxespad=0.0
        )

    # plt.title(f'Work Profiles {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    return

def plot_weighted_pmf(df: pd.DataFrame,
                    r_coord: str = 'r_coord',
                    cols_to_plot: list = ['Wdiss_weighted', 'dG_weighted'],
                    outdir: str = 'analysis',
                        ):

    """Plot weighted PMF from SMDAnalysis data."""

    os.makedirs(outdir, exist_ok=True)
    
    if 'estimator' in df.columns:
        hue_col = 'estimator'
    else:
        hue_col = None
    
    for col in cols_to_plot:
        outfname = os.path.join(outdir, f"{col}.png")
        # plt.figure(figsize=(10,6))
        g = sns.FacetGrid(df, col="speed", hue=hue_col, sharey=True, sharex=True, height=3.5, aspect=1.5)
        g.map(plt.plot, r_coord, col)#.add_legend()
        # g.map(plt.plot, "r_coord", "Wdiss_weighted").add_legend()
        # g.map(plt.fill_between, "r_coord", "dG_weighted_lower", "dG_weighted_upper", alpha=0.3)    
        g.set_titles(col_template="Speed = {col_name} nm/ps")
        plt.legend(
            # legend_handles, legend_labels,
            title='Estimator',
            bbox_to_anchor=(1.02, 0.8), loc='upper left',
            borderaxespad=0.0, fontsize=10
        )
        plt.tight_layout()
        plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close()
    return

def plot_convergence_traces(smd_conv_traces:list[str], outdir: str):
    """
    Plot convergence traces from sMD convergence analysis.
    Each cluster/path is shown with a different line style,
    while color encodes the number of replicas (per speed).
    """
    if len(smd_conv_traces) == 0:
        logger.warning("No data files provided for convergence traces plotting.")
        return

    # load all data
    all_data = []
    for f in smd_conv_traces:
        df = pd.read_csv(f)
        all_data.append(df)

    traces_df = pd.concat(all_data, ignore_index=True)

    # define line styles for clusters
    LINESTYLES = [
        "-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))
    ]

    for quantity, group in traces_df.groupby("quantity"):

        speeds = sorted(group["speed"].unique())
        fig, axes = plt.subplots(
            1, len(speeds),
            figsize=(5 * len(speeds), 4),
            sharey=True
        )

        if len(speeds) == 1:
            axes = [axes]

        for ax, speed in zip(axes, speeds):

            g_speed = group[group["speed"] == speed]

            # per-speed color normalization
            norm = mcolors.Normalize(
                vmin=g_speed["n_replicas"].min(),
                vmax=g_speed["n_replicas"].max()
            )
            cmap = cm.get_cmap("coolwarm_r")

            # iterate over clusters / paths
            paths = sorted(g_speed["path"].unique())

            for i, path in enumerate(paths):

                g_path = g_speed[g_speed["path"] == path]
                linestyle = LINESTYLES[i % len(LINESTYLES)]

                for n_rep, gg in g_path.groupby("n_replicas"):
                    color = cmap(norm(n_rep))

                    ax.plot(
                        gg["r_coord"],
                        gg["value"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.0,
                        alpha=0.9,
                    )

            ax.set_title(f"speed = {speed} nm/ps")
            ax.set_xlabel("r_coord (nm)")
            ax.grid(True)

        axes[0].set_ylabel(f"{quantity} (kJ/mol)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/sMD_convergence_{quantity}.png", dpi=300)
        plt.close()

    return

def plot_convergence_metrics(smd_conv_metrics: list[str], outdir: str):
    """
    Plot convergence metrics in a single combined figure.
    One subplot column per metric, one color per speed.
    Values are aggregated across paths for each (speed, n_replicas).
    """
    if len(smd_conv_metrics) == 0:
        logger.warning("No data files provided for convergence metrics plotting.")
        return
    
    # load all data
    all_data = []
    for f in smd_conv_metrics:
        df = pd.read_csv(f)
        all_data.append(df)

    metrics_df = pd.concat(all_data, ignore_index=True)
    
    if metrics_df.empty:
        logger.warning("No data available for convergence metrics plotting.")
        return
    
    os.makedirs(outdir, exist_ok=True)

    # metrics to plot
    metrics = [
        c for c in metrics_df.columns
        if c not in ['speed', 'path', 'n_replicas', 'converged', 'decision_quantity', 'n_common_points']
    ]

    if len(metrics) == 0:
        logger.warning("No numeric convergence metrics found to plot.")
        return

    speeds = sorted(metrics_df['speed'].unique())

    # color map for speeds
    cmap = cm.get_cmap("tab10")
    speed_colors = {s: cmap(i % cmap.N) for i, s in enumerate(speeds)}

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(5 * len(metrics), 4),
        sharex=False,
        sharey=False
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_df = metrics_df[['speed', 'n_replicas', metric]].dropna()

        if metric_df.empty:
            ax.set_title(metric)
            ax.set_xlabel("Number of replicas")
            ax.set_ylabel(metric)
            ax.grid(True)
            continue

        # aggregate across paths to produce one trend per speed
        agg_df = (
            metric_df
            .groupby(['speed', 'n_replicas'], as_index=False)[metric]
            .mean()
            .sort_values(['speed', 'n_replicas'])
        )

        for speed in speeds:
            subset = agg_df[agg_df['speed'] == speed]
            if subset.empty:
                continue

            ax.plot(
                subset['n_replicas'],
                subset[metric],
                color=speed_colors[speed],
                marker='o',
                linewidth=2.0,
                alpha=0.95,
                label=f"speed {speed}",
            )

        ax.set_title(metric)
        ax.set_xlabel("Number of replicas")
        ax.set_ylabel(metric)
        ax.grid(True)

    # single legend for speeds
    speed_handles = [
        Line2D([0], [0], color=speed_colors[s], marker='o', lw=2, label=f"speed {s}")
        for s in speeds
    ]
    fig.legend(
        handles=speed_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
        ncol=min(len(speeds), 4),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{outdir}/sMD_convergence_metrics.png", dpi=300)
    plt.close()
    return
    
def plot_extrapolated_param(df: pd.DataFrame = None, 
                            param: str = 'dG_weighted_slope',
                            outfname: str = None
                            ):
    """Plot the extrapolated parameter vs x_col with R2 color mapping and error bands.
    
    If a 'path' column is present, creates one subplot per path in a single figure.
    """
    if outfname is None:
        outfname = f'{param}_extrapolated.png'
        
    color_col = 'R2'  # Column for color mapping
    se_col = f'{param}_se'

    if df is None or df.empty:
        return

    # Normalize R2 for colormap across all paths
    vmin, vmax = df[color_col].min(), df[color_col].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')

    # Determine paths
    if 'path' in df.columns:
        paths = sorted(df['path'].unique())
    else:
        paths = [None]  # single "path" (no splitting)

    n_paths = len(paths)
    fig, axes = plt.subplots(
        n_paths, 1,
        figsize=(6, 4 * n_paths),
        sharex=True
    )
    if n_paths == 1:
        axes = [axes]  # make iterable

    for ax, path_val in zip(axes, paths):
        if path_val is not None:
            df_p = df[df['path'] == path_val].copy()
        else:
            df_p = df.copy()

        x = df_p['r_coord'].values
        y = df_p[param].values
        yerr = df_p[se_col].values if se_col in df_p.columns else None

        # Plot shaded error bands and colored lines segment-wise
        for i in range(len(df_p) - 1):
            # x may be Series or Index
            xi = x.iloc[i:i+2] if hasattr(x, 'iloc') else x[i:i+2]
            yi = y[i:i+2]
            yerri = yerr[i:i+2] if yerr is not None else None
            r2_val = df_p[color_col].iloc[i]

            color = cmap(norm(r2_val))
            ax.plot(xi, yi, color=color, lw=4)

            if yerri is not None:
                ax.fill_between(xi, yi - yerri, yi + yerri, color=color, alpha=0.3)

        # Ax labels/titles per subplot
        # if param == 'dG_v0_intercept':
        #     ax.axhline(27, color='k', lw=2, ls='--')

        ax.set_xlabel('r_coord (nm)');            ax.set_ylabel(param)
        if path_val is not None:
            ax.set_title(f'path {path_val}')
        else:
            ax.set_title(f'combined paths')
        ax.grid(True)

    # Add a single colorbar for the whole figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])  # Position for colorbar
    cbar = fig.colorbar(sm, orientation='vertical', cax=cbar_ax)

    cbar.set_label('$R^2$ of extrapolation')

    plt.tight_layout()
    plt.savefig(outfname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return
    
def make_unbinding_paths_pml( 
    paths: Dict[str, List[Tuple[str, str]]],
    reference_pdb: str,
    ligand_select: str,
    outdir: str = "unbinding_paths",
    align_sel: str = "protein and backbone",
    grid_spacing: float = 1.0,
    cartoon_color: str = "palecyan",
    sample_stride: int = 1,
) -> str:
    """Generate ligand-path density maps and a PyMOL .pml that uses only relative paths."""
            
    level = 0.000002
    surface_transparency = 0.35
    cartoon_transparency = 0.25
    
    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)
    protein_pdb = reference_pdb
    ligand_sel = ligand_select
    # Copy the reference PDB into OUTDIR so the .pml can run anywhere
    prot_copy = outdir / os.path.basename(protein_pdb)
    shutil.copy2(protein_pdb, prot_copy)

    protein_abs = str(prot_copy.resolve())
    u_ref = mda.Universe(protein_abs)

    default_palette = ["violetpurple", "marine", "forest", "deepsalmon", "gold", "tv_red", "tv_blue"]
    path_colors = {name: default_palette[i % len(default_palette)] for i, name in enumerate(paths)}

    dx_files_rel = {}
    for path_name, traj_list in paths.items():
        # Only use the first (medoid) trajectory for each path
        if not traj_list:
            logger.warning(f"No trajectories found for path {path_name}")
            continue
        
        top, traj = traj_list[0]  # Use only the medoid trajectory
        
        # if not aligned, align to reference
        u = mda.Universe(top, traj)
        align.AlignTraj(u, u_ref, select=align_sel, in_memory=True).run()

        lig = u.select_atoms(ligand_sel)
        if lig.n_atoms == 0:
            raise ValueError(f"No atoms found for '{ligand_sel}' in {traj}.")

        da = density.DensityAnalysis(lig, delta=grid_spacing, padding=50.0)
        da.run(step=sample_stride)
        dens_sum = da.results.density

        # Smooth the density
        dens_sum.grid = gaussian_filter(dens_sum.grid, sigma=1.0)
        
        dx_path = str((outdir / f"{path_name}_density.dx").resolve())
        dens_sum.export(dx_path)
        dx_files_rel[path_name] = dx_path

    # dx_05 = np.quantile(dens_sum.grid, 0.05)
    # print(f"0.05 quantile of last path density: {dx_05}")
    
    # Write the .pml using ONLY filenames (relative to outdir)
    pml_path = os.path.join(outdir, "unbinding_paths.pml")
    with open(pml_path, "w") as pml:
        pml.write("# Relative-path PyMOL visualization for ligand unbinding paths\n")
        pml.write("reinitialize\n")
        pml.write("bg_color white\n")
        pml.write("set ray_opaque_background, 0\n")
        pml.write("set antialias, 2\n")
        pml.write("set specular, 0.2\n")
        pml.write("set ray_shadow, off\n")
        pml.write(f"set cartoon_transparency, {cartoon_transparency:.2f}\n")
        pml.write(f"load {protein_abs}, prot\n")
        pml.write("hide everything, prot\n")
        pml.write("show cartoon, prot\n")
        pml.write(f"color {cartoon_color}, prot\n")

        for path_name, dx_filename in dx_files_rel.items():
            map_obj = f"map_{path_name}"
            surf_obj = f"surf_{path_name}"
            col = path_colors[path_name]
            pml.write(f"load {dx_filename}, {map_obj}\n")
            pml.write(f'map_double {map_obj}\n')
            pml.write(f"isosurface {surf_obj}, {map_obj}, {level}\n")
            pml.write(f"color {col}, {surf_obj}\n")
            pml.write(f"set transparency, {surface_transparency:.2f}, {surf_obj}\n")
            pml.write(f"set two_sided_lighting, on, {surf_obj}\n")

        pml.write(f"select lig_ref, ({ligand_sel}) and prot\n")
        pml.write("if sele count lig_ref > 0:\n")
        pml.write("    create lig, lig_ref\n")
        # pml.write("    show stick, lig\n")
        pml.write("    show sphere, lig\n")
        pml.write("    color yellow, lig\n")
        pml.write("    set sphere_transparency, 0.9, lig\n")
        pml.write("orient lig\n")
        pml.write("zoom prot, 10.0\n")

    return str(pml_path)