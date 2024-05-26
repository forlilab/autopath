import numpy as np
import pandas as pd
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')

def plot_rmsd(rmsd_df, sys_name, prefix):
    plt.figure(figsize=(6,4))
    sns.lineplot(data=rmsd_df, y='rmsd', x=rmsd_df.index)
    plt.ylabel('RMSD (A)'); plt.xlabel('Frame #')
    plt.title(f'RMSD {prefix}')
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{prefix}_rmsd.png')
    plt.close()
    return

def plot_colvar(dir_path, colvar_name):
    sys_name = dir_path.split('/')[0]
    files = glob(f'{dir_path}/COLVAR_*')
    data=[]
    for f in files:
        walker_name = f.split('/')[2].split('.')[0]
        np_data = np.load(f)
        df = pd.DataFrame(np_data, columns=[colvar_name])
        df['walker'] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)
    
    plt.figure(figsize=(10,5))
    sns.lineplot(data, x=data.index, y=colvar_name, hue='walker')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.ylabel('COM distance (nm)');    plt.xlabel('Frame #')
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}_COLVAR.png')
    plt.show()
    plt.close()

def plot_bias(dir_path, x_min, x_max, grid_points):
    sys_name = dir_path.split('/')[0]
    axis_values = np.linspace(x_min,x_max,grid_points)
    axis_values = [round(i,2) for i in axis_values]

    files = glob(f'{dir_path}/bias_*')
    data=[]
    for f in files:
        walker_name = f.split('/')[2].split('.')[0]
        np_data = np.load(f)
        # np_data = np_data * 10 #nM to A
        np_data = np_data * 0.239006 #KJ to Kcal
        df = pd.DataFrame(np_data, columns=['bias'])
        df.index = axis_values
        df['walker'] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)
    
    plt.figure(figsize=(10,4))
    sns.lineplot(data, x=data.index, y='bias', hue='walker')
    plt.ylabel('Bias (Kcal/mol)');    plt.xlabel('COM distance (nm)')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}_BIAS.png')
    plt.close()

def plot_FE(dir_path, x_min, x_max, grid_points):

    sys_name = dir_path.split('/')[0]
    axis_values = np.linspace(x_min,x_max,grid_points)
    axis_values = [round(i,2) for i in axis_values]

    files = glob(f'{dir_path}/FE_*.npy')
    data=[]
    for i, f in enumerate(files):
        walker_name = f.split('/')[2].split('.')[0]#+str(i)
        np_data = np.load(f)
        np_data = np_data * 0.239006 #KJ to Kcal
        df = pd.DataFrame(np_data, columns=['FE'])
        df.index = axis_values
        df['walker'] = walker_name
        data.append(df)

    data = pd.concat(data, axis=0)
    plt.figure(figsize=(10,4))
    sns.lineplot(data, x=data.index, y='FE', hue='walker')
    # plt.vlines(ymin=-22, ymax=0, x=0.31, colors='black')
    plt.ylabel('FE (Kcal/mol)');  plt.xlabel('COM distance (nm)')
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize="12")
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}_FE.png')
    plt.close()

def plot_sMD_statistics(data, sys_name):

    plt.figure(figsize=(6,5))
    sns.lineplot(data, x='r0', y='work', hue='replica')
    plt.xlabel('r0 dist (nm)'); plt.ylabel('Work (KJ/mol)')
    plt.title(sys_name)
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}-r0_vs_work.png')
    plt.close()

    plt.figure(figsize=(6,5))
    sns.lineplot(data, x='COMDist', y='work', hue='replica')
    plt.xlabel('COM dist (nm)'); plt.ylabel('Work (KJ/mol)')
    plt.title(sys_name)
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}-com_vs_work.png')
    plt.close()

    plt.figure(figsize=(6,5))
    sns.lineplot(data, x='r0', y='force')#, hue='replica')
    plt.xlabel('r0 dist (nm)'); plt.ylabel('Force (KJ/mol)')
    plt.title(sys_name)
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}-r0_vs_force.png')
    plt.close()

    plt.figure(figsize=(6,5))
    sns.lineplot(data, x=data['time'], y='work', hue='replica')
    plt.xlabel('time (ns)'); plt.ylabel('Work (KJ/mol)')
    plt.title(sys_name)
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}-time_vs_work.png')
    plt.close()

    plt.figure(figsize=(6,5))
    sns.lineplot(data, x=data['time'], y='COMDist', hue='replica')
    plt.xlabel('time (ns)'); plt.ylabel('COM dist (nm)')
    plt.title(sys_name)
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/{sys_name}-time_vs_com.png')
    plt.close()

    # plt.figure(figsize=(6,5))
    # sns.displot(data=data, x='COMDist', kind='kde', hue='replica')
    # plt.ylabel('fraction'); plt.xlabel('COM dist (nm)')
    # plt.tight_layout()
    # plt.savefig(f'{sys_name}/COM_dist.png')

    # plt.figure(figsize=(6,5))
    # sns.displot(data=data, x='force', kind='kde', hue='replica')
    # plt.ylabel('fraction'); plt.xlabel('Force (KJ/mol)')
    # plt.tight_layout()
    # plt.savefig(f'{sys_name}/force_dist.png')

    return None

def plot_clusters(df_clustered, closest_points, sys_name):
    
    plt.scatter(df_clustered['rmsd'], df_clustered['cog_d'],
                marker='o', c=df_clustered['cluster'], alpha=0.5)
    for idx, row in closest_points.iterrows():
        plt.scatter(row['rmsd'], row['cog_d'], marker='x',c='black', alpha=1, zorder=3)

    plt.xlabel('RMSD (A)'); plt.ylabel('COG dist(nm)')
    plt.title(f'{sys_name} sMD centroids')
    plt.tight_layout()
    plt.savefig(f'{sys_name}/plots/sMD_cluster_centroids.png')
    plt.close()