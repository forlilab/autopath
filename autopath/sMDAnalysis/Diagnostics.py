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
                 sharey: bool = True,
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
        sharey=sharey, sharex=True,
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
            ylabel='Γ (kJ·ps/mol/nm²)',
            outdir=outdir,
            prefix=f'_{method}',
            sharey=False
        )
        # Integrated friction (regression only)
        if 'Gamma_integrated' in mdf.columns and mdf['Gamma_integrated'].notna().any():
            plot_profile(
                mdf,
                value_col='Gamma_integrated',
                hue='estimator',
                show_error_bands=False,
                ylabel='∫Γ dr (kJ·ps/mol/nm)',
                outdir=outdir,
                prefix=f'_{method}',
                sharey=False,   # each speed has its own scale — low-speed Γ dominates if shared
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
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty traces file: {f}")
            continue
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

def plot_convergence_metrics(smd_conv_metrics: list[str], outdir: str, tolerances: dict = None):
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
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty metrics file: {f}")
            continue
        all_data.append(df)

    if not all_data:
        logger.warning("No convergence metrics data available for plotting.")
        return

    metrics_df = pd.concat(all_data, ignore_index=True)

    if metrics_df.empty:
        logger.warning("No data available for convergence metrics plotting.")
        return
    
    os.makedirs(outdir, exist_ok=True)

    # metrics to plot
    metrics = [
        c for c in metrics_df.columns
        if c not in ['speed', 'path', 'n_replicas', 'converged', 'decision_quantity', 'n_common_points', 'reason']
    ]

    if len(metrics) == 0:
        logger.warning("No numeric convergence metrics found to plot.")
        return

    speeds = sorted(metrics_df['speed'].unique())

    # use the fivethirtyeight color cycle (applied at module level)
    color_list = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    speed_colors = {s: color_list[i % len(color_list)] for i, s in enumerate(speeds)}

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

        legend_handles = [
            Line2D([0], [0], color=speed_colors[s], marker='o', lw=2, label=f"speed {s}")
            for s in speeds
        ]
        if tolerances and metric in tolerances:
            tol_val = tolerances[metric]
            ax.axhline(tol_val, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
            legend_handles.append(
                Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label=f"tol = {tol_val}")
            )
        ax.legend(handles=legend_handles, loc="best")
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel("Number of replicas")
        ax.set_ylabel(metric)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
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

def _build_friction_grid(dens, friction_profile: pd.Series, center_pos_nm: np.ndarray):
    """Build a volumetric friction map on the same grid as a density object.

    For each voxel, computes Euclidean distance from *center_pos_nm* (the
    membrane anchor, e.g. DUM atom) and looks up Gamma(r) from the friction
    profile via linear interpolation.  Values are clipped at 0 (negative
    Gamma arises from noise, not physics).

    Parameters
    ----------
    dens : gridData.Grid
        Density object whose grid geometry (origin, delta, shape) is reused.
    friction_profile : pd.Series
        Index = r_coord (nm), values = Gamma (kJ·ps/mol/nm²).
    center_pos_nm : (3,) array
        3-D position of the membrane center anchor in **nm**.

    Returns
    -------
    gridData.Grid
        Friction map on the same voxel grid as *dens*.
    """
    from scipy.interpolate import interp1d
    import gridData

    r_vals = friction_profile.index.to_numpy(dtype=float)
    g_vals = np.clip(friction_profile.to_numpy(dtype=float), 0, None)
    order = np.argsort(r_vals)
    r_vals, g_vals = r_vals[order], g_vals[order]

    gamma_interp = interp1d(
        r_vals, g_vals, kind="linear",
        bounds_error=False,
        fill_value=(g_vals[0], g_vals[-1]),
    )

    shape = dens.grid.shape
    origin_A = np.asarray(dens.origin)   # Å
    delta_A  = np.asarray(dens.delta)    # Å / voxel

    ix, iy, iz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x = origin_A[0] + ix * delta_A[0]
    y = origin_A[1] + iy * delta_A[1]
    z = origin_A[2] + iz * delta_A[2]

    center_A = center_pos_nm * 10.0      # nm → Å
    r_nm = np.sqrt((x - center_A[0])**2 + (y - center_A[1])**2 + (z - center_A[2])**2) / 10.0

    gamma_grid = gamma_interp(r_nm)
    return gridData.Grid(gamma_grid, origin=dens.origin, delta=dens.delta)


def make_unbinding_paths_visualization(
    paths: Dict[str, List[Tuple[str, str]]],
    reference_pdb: str,
    ligand_select: str,
    outdir: str = "path_analysis",
    align_sel: str = "protein and backbone",
    grid_spacing: float = 0.5,
    cartoon_color: str = "grey90",
    sample_stride: int = 2,
    pocket_select: str = None,
    n_lig_conformations: int = 20,
    output_format: str = "pse",
    friction_csv: str = None,
    friction_estimator: str = "cumulant",
    friction_center_select: str = "resname DUM",
) -> str:
    """Generate ligand-path density maps as a PyMOL session or script.

    Args:
        output_format: ``"pse"`` saves a portable, self-contained session file;
            ``"pml"`` writes a script + auxiliary .dx files next to it.
        friction_csv: Path to ``friction.csv`` produced by
            :class:`~autopath.sMDAnalysis.Estimators.FrictionEstimator`.
            When provided, the isosurfaces are coloured by local friction
            Γ(r) as a blue (low) → white → red (high) gradient.
        friction_estimator: Which estimator row to use from *friction_csv*
            (``"cumulant"`` or ``"jarzynski"``).  Default ``"cumulant"``.
        friction_center_select: MDAnalysis selection for the CV reference
            anchor used to map voxel distances to r_coord values.
            Default ``"resname DUM"`` (membrane-pull convention).  If this
            selection matches 0 atoms and *pocket_select* is provided, the
            pocket COM is used as fallback (pocket-unbinding convention).
    """
    if output_format not in ("pse", "pml"):
        raise ValueError(f"output_format must be 'pse' or 'pml', got '{output_format}'")

    level = 0.000002
    surface_transparency = 0.45
    cartoon_transparency = 0.25
    stick_transparency = 0.25

    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)
    protein_abs = str(Path(reference_pdb).resolve())
    u_ref = mda.Universe(protein_abs)

    # ---------------------------------------------------------------------- #
    # Load friction profile (optional)                                         #
    # ---------------------------------------------------------------------- #
    friction_profile: Optional[pd.Series] = None
    friction_center_nm: Optional[np.ndarray] = None
    if friction_csv is not None:
        try:
            fdf = pd.read_csv(friction_csv)
            fdf = fdf[(fdf["method"] == "derivative") & (fdf["estimator"] == friction_estimator)]
            if fdf.empty:
                logger.warning(
                    f"No derivative/{friction_estimator} rows in {friction_csv}. "
                    "Friction colouring disabled."
                )
            else:
                fdf = fdf.sort_values("r_coord")
                friction_profile = pd.Series(
                    fdf["Gamma"].to_numpy(), index=fdf["r_coord"].to_numpy()
                )
                center_ag = u_ref.select_atoms(friction_center_select)
                if center_ag.n_atoms == 0:
                    if pocket_select is not None:
                        pocket_ag = u_ref.select_atoms(pocket_select)
                        if pocket_ag.n_atoms > 0:
                            logger.info(
                                f"Friction centre selection '{friction_center_select}' found 0 atoms. "
                                "Falling back to pocket_select COM as friction anchor."
                            )
                            friction_center_nm = pocket_ag.center_of_geometry() / 10.0  # Å → nm
                        else:
                            logger.warning(
                                f"Friction centre selection '{friction_center_select}' found 0 atoms "
                                "and pocket_select also found 0 atoms. Friction colouring disabled."
                            )
                            friction_profile = None
                    else:
                        logger.warning(
                            f"Friction centre selection '{friction_center_select}' found 0 atoms "
                            "and no pocket_select provided. Friction colouring disabled."
                        )
                        friction_profile = None
                else:
                    friction_center_nm = center_ag.center_of_geometry() / 10.0  # Å → nm
                if friction_center_nm is not None:
                    logger.info(
                        f"Friction colouring enabled ({friction_estimator}). "
                        f"Centre: {friction_center_nm} nm"
                    )
        except Exception as exc:
            logger.warning(f"Could not load friction data from {friction_csv}: {exc}. Skipping.")

    default_palette = ["violetpurple", "marine", "forest", "deepsalmon", "gold", "tv_red", "tv_blue"]
    path_colors = {name: default_palette[i % len(default_palette)] for i, name in enumerate(paths)}

    # ------------------------------------------------------------------ #
    # Shared: compute density maps and extract ligand conformations        #
    # ------------------------------------------------------------------ #
    import tempfile
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = Path(tmpdir_obj.name)

    dx_files: Dict[str, str] = {}
    friction_dx_files: Dict[str, str] = {}
    lig_pdb_files: Dict[str, str] = {}

    for path_name, traj_list in paths.items():
        if not traj_list:
            logger.warning(f"No trajectories found for path {path_name}")
            continue

        top, traj = traj_list[0]
        u = mda.Universe(top, traj)
        if u_ref.select_atoms(align_sel).n_atoms > 0:
            align.AlignTraj(u, u_ref, select=align_sel, in_memory=True).run()
        else:
            logger.warning(
                f"Alignment selection '{align_sel}' matched 0 atoms in reference PDB "
                f"(membrane/no-protein system?). Skipping re-alignment — trajectories "
                f"are assumed to be pre-aligned."
            )

        lig = u.select_atoms(ligand_select)
        if lig.n_atoms == 0:
            raise ValueError(f"No atoms found for '{ligand_select}' in {traj}.")

        da = density.DensityAnalysis(lig, delta=grid_spacing, padding=25.0)
        da.run(step=sample_stride)
        dens = da.results.density
        dens.grid = gaussian_filter(dens.grid, sigma=2.0)

        # For PML the .dx files are placed in outdir (referenced by the script).
        # For PSE they go to the temp dir and are cleaned up after saving.
        dx_dest = outdir if output_format == "pml" else tmpdir
        dx_path = str(dx_dest / f"{path_name}_density.dx")
        dens.export(dx_path)
        dx_files[path_name] = dx_path

        # Build friction volumetric map on the same grid as the density
        if friction_profile is not None and friction_center_nm is not None:
            try:
                fric_grid = _build_friction_grid(dens, friction_profile, friction_center_nm)
                fric_dx_path = str(dx_dest / f"{path_name}_friction.dx")
                fric_grid.export(fric_dx_path)
                friction_dx_files[path_name] = fric_dx_path
            except Exception as exc:
                logger.warning(f"Could not build friction grid for {path_name}: {exc}")

        n_frames = len(u.trajectory)
        if n_lig_conformations >= n_frames:
            frame_indices = list(range(n_frames))
        else:
            frame_indices = [int(i * n_frames / n_lig_conformations) for i in range(n_lig_conformations)]

        lig_pdb = str(tmpdir / f"{path_name}_lig.pdb")
        with mda.Writer(lig_pdb, multiframe=True, n_atoms=lig.n_atoms) as W:
            for idx in frame_indices:
                u.trajectory[idx]
                W.write(lig)
        lig_pdb_files[path_name] = lig_pdb

    # Determine friction colour range from all paths combined (use 5th–95th percentile)
    friction_ramp_vals: Optional[List[float]] = None
    if friction_dx_files:
        all_gamma: List[float] = []
        for path_name in friction_dx_files:
            fdf = pd.read_csv(friction_csv)
            fdf = fdf[(fdf["method"] == "derivative") & (fdf["estimator"] == friction_estimator)]
            all_gamma.extend(np.clip(fdf["Gamma"].to_numpy(), 0, None).tolist())
        if all_gamma:
            g_lo  = float(np.percentile(all_gamma, 5))
            g_mid = float(np.percentile(all_gamma, 60))
            g_hi  = float(np.percentile(all_gamma, 95))
            friction_ramp_vals = [g_lo, g_mid, g_hi]
            logger.info(
                f"Friction ramp: low={g_lo:.0f}, mid={g_mid:.0f}, "
                f"high={g_hi:.0f} kJ·ps/mol/nm²"
            )

    pocket_resids: List[int] = []
    if pocket_select is not None:
        pocket = u_ref.select_atoms(pocket_select)
        if pocket.n_atoms == 0:
            raise ValueError(f"No atoms found for pocket selection '{pocket_select}' in reference PDB.")
        pocket_resids = sorted(set(pocket.resids))
        logger.info(f"Found pocket residues: {pocket_resids}")

    # ------------------------------------------------------------------ #
    # PSE branch: build session via PyMOL Python API                      #
    # ------------------------------------------------------------------ #
    if output_format == "pse":
        try:
            import pymol2
        except ImportError:
            raise ImportError("pymol2 is required. Install open-source PyMOL into your environment.")

        out_path = str(outdir / "unbinding_paths.pse")

        with pymol2.PyMOL() as pymol:
            cmd = pymol.cmd
            cmd.bg_color("white")
            cmd.set("antialias", 2)
            cmd.set("specular", 0.2)
            cmd.set("ray_shadow", 0)
            cmd.set("ray_opaque_background", 0)
            cmd.set("cartoon_transparency", cartoon_transparency)

            cmd.load(protein_abs, "prot")
            cmd.hide("everything", "prot")
            cmd.show("cartoon", "prot")
            cmd.color(cartoon_color, "prot")

            if pocket_resids:
                resi_str = "+".join(map(str, pocket_resids))
                cmd.select("pocket", f"resi {resi_str} and prot")
                cmd.show("sticks", "pocket")
                cmd.color("orange", "pocket")

            # Build shared friction ramp (one ramp covers all paths)
            use_friction_coloring = bool(friction_dx_files and friction_ramp_vals)
            if use_friction_coloring:
                g_lo, g_mid, g_hi = friction_ramp_vals
                # Load one friction map (they all share the same Gamma(r) function;
                # we use path-0's map for the ramp calibration, then reload per path)
                first_path = next(iter(friction_dx_files))
                cmd.load(friction_dx_files[first_path], "friction_ramp_ref")
                cmd.do("map_double friction_ramp_ref")
                cmd.ramp_new(
                    "friction_ramp", "friction_ramp_ref",
                    [g_lo, g_mid, g_hi],
                    ["blue", "white", "red"],
                )
                cmd.hide("everything", "friction_ramp_ref")

            for path_name in dx_files:
                col = path_colors[path_name]
                map_obj  = f"map_{path_name}"
                surf_obj = f"surf_{path_name}"
                lig_obj  = f"lig_{path_name}"

                cmd.load(dx_files[path_name], map_obj)
                cmd.do(f"map_double {map_obj}")
                cmd.isosurface(surf_obj, map_obj, level)
                cmd.set("transparency", surface_transparency, surf_obj)
                cmd.set("two_sided_lighting", 1, surf_obj)
                cmd.hide("everything", map_obj)

                if use_friction_coloring and path_name in friction_dx_files:
                    # Load this path's friction map and redefine the ramp against it,
                    # then colour the surface.  Using per-path maps ensures the ramp
                    # samples the correct spatial region for this isosurface.
                    fric_obj = f"fric_{path_name}"
                    cmd.load(friction_dx_files[path_name], fric_obj)
                    cmd.do(f"map_double {fric_obj}")
                    cmd.ramp_new(
                        f"ramp_{path_name}", fric_obj,
                        [g_lo, g_mid, g_hi],
                        ["blue", "white", "red"],
                    )
                    cmd.color(f"ramp_{path_name}", surf_obj)
                    cmd.hide("everything", fric_obj)
                else:
                    cmd.color(col, surf_obj)

                cmd.load(lig_pdb_files[path_name], lig_obj)
                cmd.hide("everything", lig_obj)
                cmd.show("sticks", lig_obj)
                cmd.color(col, lig_obj)
                cmd.set("stick_transparency", stick_transparency, lig_obj)

            if cmd.count_atoms(f"({ligand_select}) and prot") > 0:
                cmd.create("lig_ref", f"({ligand_select}) and prot")
                cmd.hide("everything", "lig_ref")
                cmd.show("sticks", "lig_ref")
                cmd.color("yellow", "lig_ref")

            cmd.zoom("prot", 10.0)
            cmd.save(out_path)

        tmpdir_obj.cleanup()
        return out_path

    # ------------------------------------------------------------------ #
    # PML branch: write a script; .dx files already in outdir             #
    # ------------------------------------------------------------------ #
    # Copy the ligand PDB conformations into outdir so the .pml can reference them
    for path_name, lig_pdb in lig_pdb_files.items():
        dest = str(outdir / f"{path_name}_lig.pdb")
        shutil.copy2(lig_pdb, dest)
        lig_pdb_files[path_name] = dest

    # Copy the reference PDB into outdir for portability
    prot_copy = outdir / os.path.basename(reference_pdb)
    shutil.copy2(reference_pdb, prot_copy)

    tmpdir_obj.cleanup()

    pml_path = str(outdir / "unbinding_paths.pml")
    with open(pml_path, "w") as pml:
        pml.write("reinitialize\n")
        pml.write("bg_color white\n")
        pml.write("set ray_opaque_background, 0\n")
        pml.write("set antialias, 2\n")
        pml.write("set specular, 0.2\n")
        pml.write("set ray_shadow, off\n")
        pml.write(f"set cartoon_transparency, {cartoon_transparency:.2f}\n")
        pml.write(f"load {os.path.basename(str(prot_copy))}, prot\n")
        pml.write("hide everything, prot\n")
        pml.write("show cartoon, prot\n")
        pml.write(f"color {cartoon_color}, prot\n")

        if pocket_resids:
            pml.write(f"select pocket, resi {'+'.join(map(str, pocket_resids))} and prot\n")
            pml.write("show sticks, pocket\n")
            pml.write("color orange, pocket\n")

        use_friction_coloring = bool(friction_dx_files and friction_ramp_vals)

        for path_name, dx_path in dx_files.items():
            col = path_colors[path_name]
            map_obj  = f"map_{path_name}"
            surf_obj = f"surf_{path_name}"
            lig_obj  = f"lig_{path_name}"
            pml.write(f"load {os.path.basename(dx_path)}, {map_obj}\n")
            pml.write(f"map_double {map_obj}\n")
            pml.write(f"isosurface {surf_obj}, {map_obj}, {level}\n")
            pml.write(f"set transparency, {surface_transparency:.2f}, {surf_obj}\n")
            pml.write(f"set two_sided_lighting, on, {surf_obj}\n")

            if use_friction_coloring and path_name in friction_dx_files:
                g_lo, g_mid, g_hi = friction_ramp_vals
                fric_obj  = f"fric_{path_name}"
                ramp_name = f"ramp_{path_name}"
                fric_base = os.path.basename(friction_dx_files[path_name])
                pml.write(f"load {fric_base}, {fric_obj}\n")
                pml.write(f"map_double {fric_obj}\n")
                pml.write(
                    f"ramp_new {ramp_name}, {fric_obj}, "
                    f"[{g_lo:.2f}, {g_mid:.2f}, {g_hi:.2f}], "
                    f"[blue, white, red]\n"
                )
                pml.write(f"color {ramp_name}, {surf_obj}\n")
                pml.write(f"hide everything, {fric_obj}\n")
            else:
                pml.write(f"color {col}, {surf_obj}\n")

            pml.write(f"load {os.path.basename(lig_pdb_files[path_name])}, {lig_obj}\n")
            pml.write(f"hide everything, {lig_obj}\n")
            pml.write(f"show sticks, {lig_obj}\n")
            pml.write(f"color {col}, {lig_obj}\n")
            pml.write(f"set stick_transparency, {stick_transparency:.2f}, {lig_obj}\n")

        pml.write(f"select lig_ref_sel, ({ligand_select}) and prot\n")
        pml.write("if cmd.count_atoms('lig_ref_sel') > 0:\n")
        pml.write("    create lig_ref, lig_ref_sel\n")
        pml.write("    hide everything, lig_ref\n")
        pml.write("    show sticks, lig_ref\n")
        pml.write("    color yellow, lig_ref\n")
        pml.write("zoom prot, 10.0\n")

    return pml_path