from autopath.SMDAnalysis.SMDData import SMDData
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger("autopath")

class Diagnostics:
    """A class to perform diagnostic checks, test, and plots on SMDAnalysis data."""
    def __init__(self, smd_data: SMDData,
                 outdir: str = './diagnostics'
                 
                 ):
        self.raw_data = smd_data.raw_data
        self.results = smd_data.results
        if self.results is None:
            logger.warning('SMDData results not found None. Some diagnostics may not work.')
        
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        return None
    
    def plot_work_profiles(
        self,
        results: pd.DataFrame,
        r_coord: str = 'r_coord',
        cols_to_plot: list = ['Wmean', 'dG', 'Wdiss'],
        title_suffix: str = 'dcTMD',
        outdir: str = None,
    ):
        if outdir is None:
            outdir = self.outdir
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"work_profiles_{title_suffix}.png")
        
        if results is None:
            results = self.results
        
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
                var_name=title_suffix,
                value_name='value'
            )

            sns.lineplot(
                data=long_df,
                x=r_coord, y='value',
                hue=title_suffix, style='path',
                estimator=None, errorbar=None,  # don't aggregate across paths
                ax=axes[i]
            )
            axes[i].grid(True)
            
            # # just reference for trypsin
            # axes[i].axhline(27, color='k', lw=1, ls='--')
            
            axes[i].set_title(f'Speed: {speed} nm/ps', fontsize=10)
            axes[i].set_xlabel(f'{r_coord} (nm)')
            if i == 0:
                axes[i].set_ylabel('dG (kJ/mol)')
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
    
    def plot_extrapolated_param(self, 
                                df: pd.DataFrame = None, 
                                param: str = 'Wdiss_slope',
                                x_col: str = 'r_coord',
                                ):
        """Plot the extrapolated parameter vs x_col with R2 color mapping and error bands.
        
        If a 'path' column is present, creates one subplot per path in a single figure.
        """
        outfname = os.path.join(self.outdir, f'{self.sysname}_{param}_extrapolated.png')
        color_col = 'R2'  # Column for color mapping
        se_col = f'{param}_se'

        if df is None or df.empty:
            return

        # Figure out x for the whole df (only used if x_col not present)
        if x_col in df.columns:
            x_global = df[x_col]
        else:
            x_global = df.index

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
            sharex=True if x_col in df.columns else False
        )
        if n_paths == 1:
            axes = [axes]  # make iterable

        for ax, path_val in zip(axes, paths):
            if path_val is not None:
                df_p = df[df['path'] == path_val].copy()
            else:
                df_p = df.copy()

            # Sort by x for nice plotting
            if x_col in df_p.columns:
                df_p = df_p.sort_values(x_col)
                x = df_p[x_col]
            else:
                df_p = df_p.sort_index()
                x = df_p.index

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

            ax.set_xlabel(x_col)
            ax.set_ylabel(param)
            if path_val is not None:
                ax.set_title(f'{self.sysname} - path {path_val}')
            else:
                ax.set_title(f'{self.sysname} - path 1')
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