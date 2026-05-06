from autopath.sMDAnalysis.SMDData import SMDData
import pandas as pd
import os
import numpy as np
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
    mask_negative_dG: bool = False,
):

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"work_profiles_{estimator}.svg")

    results = results[results['estimator'] == estimator]
    if mask_negative_dG and 'dG' in results.columns:
        results = results.copy()
        results.loc[results['dG'] < 0, 'dG'] = np.nan
    style_order = results['path'].unique().tolist()
    speeds = sorted(results['speed'].unique())
        
    fig, ax = plt.subplots(
        figsize=(11, 3),
        ncols=len(speeds),
        nrows=1,
        sharey=True,
        sharex=True
    )
    axes = ax.flatten() if len(speeds) > 1 else [ax]

    hue_legend_handles, hue_legend_labels = None, None

    for i, speed in enumerate(speeds):
        speed_df = results.loc[results['speed'] == speed, [r_coord, 'path', 'speed'] + cols_to_plot]

        # Style order scoped to paths in this speed only
        speed_style_order = [p for p in style_order if p in speed_df['path'].unique()]

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
            hue=estimator, style='path', style_order=speed_style_order,
            estimator=None, errorbar=None,  # don't aggregate across paths
            ax=axes[i]
        )
        axes[i].grid(True)
        
        # # just reference for trypsin
        # axes[i].axhline(27, color='k', lw=1, ls='--')
        
        axes[i].set_title(f'Speed={speed} nm/ps', fontsize=14)
        axes[i].set_xlabel(f'{r_coord} (nm)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Energy (kJ/mol)', fontsize=12)
        else:
            axes[i].set_ylabel('')

        # Separate legend into hue (metrics) and style (paths)
        handles, labels = axes[i].get_legend_handles_labels()
        style_handles, style_labels = [], []
        hue_handles, hue_labels = [], []
        for h, l in zip(handles, labels):
            if l in cols_to_plot:
                hue_handles.append(h)
                hue_labels.append(l)
            elif l in speed_style_order:
                style_handles.append(h)
                style_labels.append(l)

        axes[i].legend_.remove()

        # Path legend (line style) inside each subplot — black lines, strip speed from labels
        if style_handles:
            clean_handles = []
            seen_labels = set()
            clean_labels = []
            for h, l in zip(style_handles, style_labels):
                # Strip speed suffix (e.g. path-0_v0.005 → path-0)
                display = l.rsplit('_v', 1)[0] if '_v' in l else l
                if display in seen_labels:
                    continue
                seen_labels.add(display)
                clean_labels.append(display)
                clean_handles.append(
                    Line2D([0], [0], color='black', linestyle=h.get_linestyle(), lw=2)
                )
            axes[i].legend(
                clean_handles, clean_labels,
                title='Path', fontsize=8, title_fontsize=9,
                loc='best', framealpha=0.8,
            )

        # Capture hue (metric) handles once for figure-level legend — solid colored lines
        if hue_legend_handles is None and hue_handles:
            hue_legend_handles = [
                Line2D([0], [0], color=h.get_color(), linestyle='-', lw=2)
                for h in hue_handles
            ]
            hue_legend_labels = hue_labels

    # Figure-level legend for metrics (color) at the bottom
    if hue_legend_handles:
        fig.legend(
            hue_legend_handles, hue_legend_labels,
            title='',
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(hue_legend_handles), 4),
            borderaxespad=0.0,
        )

    # plt.title(f'Work Profiles {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    return

def plot_profile(df: pd.DataFrame,
                 value_col: str = 'dG',
                 hue: str | None = None,
                 row: str | None = None,
                 col: str = 'speed',
                 show_error_bands: bool = False,
                 ylabel: str | None = None,
                 outdir: str = 'analysis',
                 prefix: str = '',
                 mask_negative_dG: bool = False,
                 ):
    """Plot a single quantity vs reaction coordinate, faceted by speed.

    Parameters
    ----------
    df : pd.DataFrame
        Table with at least ``r_coord``, *col*, and *value_col* columns.
    value_col : str
        Column to plot on the y-axis (e.g. ``'dG'``, ``'Wdiss'``).
    hue : str or None
        Column used for colour encoding.  If *None*, auto-detected:
        ``'source_estimator'`` if present, else ``'estimator'``,
        else no hue.
    row : str or None
        Optional row facet variable (e.g. ``'path'``).
    col : str
        Column facet variable (default ``'speed'``).
    show_error_bands : bool
        If True and a column named ``{value_col}_se`` exists,
        draw ±1 SE shading around each line.
    ylabel : str or None
        Shared y-axis label.  Defaults to *value_col*.
    outdir : str
        Directory for saved figures.
    prefix : str
        Optional filename prefix.
    """
    os.makedirs(outdir, exist_ok=True)

    if mask_negative_dG and value_col == 'dG' and value_col in df.columns:
        df = df.copy()
        df.loc[df[value_col] < 0, value_col] = np.nan

    # Auto-detect hue
    if hue is None:
        if 'estimator' in df.columns and df['estimator'].nunique() > 1:
            hue = 'estimator'

    outfname = os.path.join(outdir, f"{value_col}{prefix}.svg")
    se_col = f"{value_col}_se"
    has_se = show_error_bands and se_col in df.columns

    g = sns.FacetGrid(
        df, col=col, row=row, hue=hue,
        sharey=True, sharex=True,
        height=4.0, aspect=1,
        margin_titles=True,
    )
    g.map_dataframe(plt.plot, 'r_coord', value_col)

    # Optional error bands
    if has_se:
        def _fill_band(data, x, y, se, **kwargs):
            data = data.sort_values(x)
            color = kwargs.get('color', 'C0')
            plt.fill_between(
                data[x], data[y] - data[se], data[y] + data[se],
                color=color, alpha=0.2,
            )
        g.map_dataframe(_fill_band, x='r_coord', y=value_col, se=se_col)

    col_template = f"{col.replace('_', ' ').title()} = {{col_name}}"
    if col == 'speed':
        col_template = "Speed = {col_name} nm/ps"
    g.set_titles(col_template=col_template, row_template="{row_name}")

    y_label = ylabel if ylabel else value_col
    g.set_axis_labels('r_coord (nm)', y_label)

    g.add_legend(title=hue.replace('_', ' ').title() if hue else '',
                bbox_to_anchor=(1.01, 0.8), loc='upper left',
                 )

    plt.tight_layout()
    plt.savefig(outfname, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    return


def plot_friction(df: pd.DataFrame,
                  outdir: str = 'analysis',
                  ):
    """Plot friction profiles from FrictionEstimator output.

    Creates one figure per friction method (derivative / regression),
    faceted by speed, coloured by estimator. Regression panels 
    include ±1 SE error bands when available.
    """
    if df is None or df.empty:
        logger.warning("No friction data to plot.")
        return

    for method, mdf in df.groupby('method'):
        # Local friction (both methods)
        plot_profile(
            mdf,
            value_col='Gamma',
            hue='estimator',
            show_error_bands=False,
            ylabel='Γ (kJ·ps/nm²)',
            outdir=outdir,
            prefix=f'_{method}',
        )
        # Integrated friction (regression only)
        if 'Gamma_integrated' in mdf.columns and mdf['Gamma_integrated'].notna().any():
            plot_profile(
                mdf,
                value_col='Gamma_integrated',
                hue='estimator',
                show_error_bands=False,
                ylabel='∫Γ dr (kJ·ps/nm)',
                outdir=outdir,
                prefix=f'_{method}',
            )
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
        plt.savefig(f"{outdir}/sMD_convergence_{quantity}.svg", dpi=300)
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
                label=f"speed {speed} nm/ps",
            )

        speed_handles = [
            Line2D([0], [0], color=speed_colors[s], marker='o', lw=2, label=f"speed {s}")
            for s in speeds
        ]
        ax.legend(handles=speed_handles, loc="best")
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel("Number of replicas")
        ax.set_ylabel(metric)
        ax.grid(True)

    plt.tight_layout()#rect=[0, 0, 1, 0.95])
    plt.savefig(f"{outdir}/sMD_convergence_metrics.svg", dpi=300, bbox_inches='tight')
    plt.close()
    return
   
def plot_extrapolated_param(df: pd.DataFrame = None,
                            param: str = 'dG',
                            outfname: str = None,
                            mask_negative_dG: bool = False,
                            ):
    """Plot the v→0 extrapolated parameter vs r_coord with R² color mapping and error bands.
    
    Works with the output of :func:`Estimators.extrapolate_to_v0`, which
    produces columns ``{param}``, ``{param}_se``, and ``R2``.
    
    Creates one subplot column per estimator (values in ``'estimator'``).
    """
    if outfname is None:
        outfname = f'{param}_extrapolated.svg'
        
    color_col = 'R2'
    se_col = f'{param}_se'

    if df is None or df.empty:
        return

    if mask_negative_dG and param == 'dG' and param in df.columns:
        df = df.copy()
        df.loc[df[param] < 0, param] = np.nan

    # Determine estimators
    if 'estimator' in df.columns:
        estimators = sorted(df['estimator'].unique())
    else:
        estimators = [None]

    n_panels = len(estimators)

    # Normalize R2 for colormap across all data
    vmin = max(df[color_col].min(), 0.0)
    vmax = min(df[color_col].max(), 1.0)
    if vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(6 * n_panels, 4),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, est_val in zip(axes, estimators):
        if est_val is not None:
            df_e = df[df['estimator'] == est_val].copy()
        else:
            df_e = df.copy()

        # Sort by r_coord so segments connect in order
        df_e = df_e.sort_values('r_coord').reset_index(drop=True)

        x = df_e['r_coord'].values
        y = df_e[param].values
        yerr = df_e[se_col].values if se_col in df_e.columns else None
        r2 = df_e[color_col].values

        # Draw R²-colored segments with optional error bands
        for i in range(len(df_e) - 1):
            xi = x[i:i+2]
            yi = y[i:i+2]
            color = cmap(norm(np.clip(r2[i], vmin, vmax)))

            ax.plot(xi, yi, color=color, lw=3)

            if yerr is not None:
                yerri = yerr[i:i+2]
                ax.fill_between(xi, yi - yerri, yi + yerri, color=color, alpha=0.25)

        ax.set_xlabel('r_coord (nm)')
        # ax.set_ylabel(f'{param.split("_")[0]} (kJ/mol)')
        title = est_val if est_val is not None else 'combined'
        ax.set_title(f'{title}  (v→0 extrapolation)')
        ax.grid(True)

    # Single colorbar outside the subplot area (bottom)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label='$R^2$')

    plt.savefig(outfname, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    return
    
def make_unbinding_paths_pml( 
    paths: Dict[str, List[Tuple[str, str]]],
    reference_pdb: str,
    ligand_select: str,
    outdir: str = "unbinding_paths",
    align_sel: str = "protein and backbone",
    grid_spacing: float = 0.5,
    cartoon_color: str = "palecyan",
    sample_stride: int = 2,
    pocket_select: str = None
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

        da = density.DensityAnalysis(lig, delta=grid_spacing, padding=25.0)
        da.run(step=sample_stride)
        dens_sum = da.results.density

        # Smooth the density
        dens_sum.grid = gaussian_filter(dens_sum.grid, sigma=2.0)
        
        dx_path = str((outdir / f"{path_name}_density.dx").resolve())
        dens_sum.export(dx_path)
        dx_files_rel[path_name] = dx_path

    # dx_05 = np.quantile(dens_sum.grid, 0.05)
    # print(f"0.05 quantile of last path density: {dx_05}")
    
    pocket_resids = []
    if pocket_select is not None:
        # show pocket atoms in the .pml if a selection is provided and valid in the reference PDB
        pocket = u_ref.select_atoms(pocket_select)
        if pocket.n_atoms == 0:
            raise ValueError(f"No atoms found for pocket selection '{pocket_select}' in reference PDB.")
        pocket_resids = sorted(set(pocket.resids))
        logger.info(f"Found pocket residues: {pocket_resids}")
    
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
        
        if pocket_resids:
            pml.write(f"select pocket, resi {'+'.join(map(str, pocket_resids))} and prot\n")
            pml.write("show sticks, pocket\n")
            pml.write("color orange, pocket\n")
                
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