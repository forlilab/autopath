from autopath.sMDAnalysis.SMDData import SMDData
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

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
        
        axes[i].set_title(f'Speed: {speed} nm/ps', fontsize=10)
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
        # plt.figure(figsize=(8,6))
        g = sns.FacetGrid(df, col="speed", hue=hue_col)
        g.map(plt.plot, r_coord, col).add_legend()
        # g.map(plt.plot, "r_coord", "Wdiss_weighted").add_legend()
        # g.map(plt.fill_between, "r_coord", "dG_weighted_lower", "dG_weighted_upper", alpha=0.3)    
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
    Plot convergence metrics from sMD convergence analysis.
    Each path/cluster is shown with a different line style,
    while color encodes pulling speed.
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
    
    # metrics to plot
    metrics = [
        c for c in metrics_df.columns
        if c not in ['speed', 'path', 'n_replicas', 'converged', 'decision_quantity', 'n_common_points']
    ]

    speeds = sorted(metrics_df['speed'].unique())
    paths = sorted(metrics_df['path'].unique())

    # color map for speeds
    cmap = cm.get_cmap("tab10")
    speed_colors = {s: cmap(i % cmap.N) for i, s in enumerate(speeds)}

    # line styles for paths
    LINESTYLES = [
        "-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))
    ]

    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(8, 4 * len(metrics)),
        sharex=True
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):

        for i, path in enumerate(paths):
            linestyle = LINESTYLES[i % len(LINESTYLES)]

            for speed in speeds:
                subset = metrics_df[
                    (metrics_df['speed'] == speed) &
                    (metrics_df['path'] == path)
                ]

                if subset.empty:
                    continue

                ax.plot(
                    subset['n_replicas'],
                    subset[metric],
                    color=speed_colors[speed],
                    linestyle=linestyle,
                    linewidth=2.0,
                    alpha=0.9,
                )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True)

    axes[-1].set_xlabel("Number of replicas")

    # build legend (speed colors + path styles)
    speed_handles = [
        Line2D([0], [0], color=speed_colors[s], lw=2, label=f"speed {s}")
        for s in speeds
    ]

    path_handles = [
        Line2D([0], [0], color="black",
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            lw=2, label=f"path {p}")
        for i, p in enumerate(paths)
    ]

    axes[0].legend(
        handles=speed_handles + path_handles,
        loc="best",
        frameon=False,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(f"{outdir}/sMD_convergence_metrics.png", dpi=300)
    plt.close()
    return
    
def plot_extrapolated_param(self, 
                            df: pd.DataFrame = None, 
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