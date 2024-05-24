import pdbfixer
from pdbfixer.pdbfixer import PDBFixer
import openmm.app as app
import openmm.unit as openmmunit
import parmed
from openmmtools.utils import get_fastest_platform

from openmm import *
from openmm.app import *
import os
from sys import stdout
import numpy as np
import pandas as pd
import shutil
import logging

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD, RMSF
from MDAnalysis.analysis import align

import pytraj as pt

def align_trajectory(prmtop_file, traj_file, overwrite:bool=False, strip_mask:str=':HOH,NA,CL,K,POP'):
    basename = traj_file.split('.')[0]
    traj_name = f"{basename}_aligned.dcd"
    if overwrite:
        traj_name = traj_file

    ptraj = pt.iterload(traj_file, prmtop_file)
    ptraj = ptraj.autoimage()
    ptraj = ptraj.center()
    ptraj = ptraj.superpose(ref=0, mask='@CA,C,N')
    
    if strip_mask is not None:
        ptraj = ptraj.strip(strip_mask)
        pt.save(f'{basename}_dry.prmtop', ptraj.top, overwrite=True)
    ptraj.save(traj_name)

    return

def fix_pdb(pdbfile: str,keep_heterogens: bool=False, 
            ignore_terminal_missing_residues:bool=True, 
            pH: float=7.4) -> PDBFixer:
    """ Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - remove nonstandard residues
    
    Args:
        pdbfile (str): pdb string old format
        pdbxfile (str): pdb string new format
        keep_heterogens (bool): if False all the heterogen atoms but waters are deleted.
        ignore_terminal_missing_residues (bool): If missing residues at the beginning and the end of a chain should be ignored or built.
        pH (float):  pH value used to determine protonation state of residues
    """

    fixer = pdbfixer.PDBFixer(str(pdbfile))
    fixer.findMissingResidues()

    if ignore_terminal_missing_residues:
        # if missing terminal residues shall be ignored, remove them from the dictionary
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms() 
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)

    return fixer

def save_pdb(topology: app.Topology, positions: list, file_path: str):
    """Saves the specified topology and position to the out_path file.

    Args:
        topology (app.Topology): topology used
        positions (list): list of 3D coords
        out_path (str): path to where to save the file
    """
    app.PDBFile.writeFile(
        topology,
        positions,
        file_path,
        keepIds=True
    )
    return

def save_system(system: System, out_file: str):
    """Saves the openmm system to the desired out path.

    Args:
        out_path (str): path where to save the System
        system (System): system to be saved
    """

    with open(out_file, "w") as fo:
        fo.write(XmlSerializer.serialize(system))
    return

def save_simulation(simulation, out_file: str):

    simulation.saveCheckpoint(f"{out_file}.chk")
    simulation.saveState(f"{out_file}.xml")

    return

def load_system(system_path: str) -> System:
    """Loads the desired system.

    Args:
        system_path (str): path to the system file

    Returns:
        System: system
    """
    with open(system_path) as fi:
        system = XmlSerializer.deserialize(fi.read())
    return system

def save_amber_topology(topology: app.Topology, positions: list, system: System, forcefield: app.ForceField, out_path: str):
    """Save the topology files necessary for MD simulations according to the simulation engine specified.

    Args:
        topology (app.Topology): openmm topology 
        positions (list): list of 3D coordinates of the topology
        system (System): openmm system
        forcefield (app.Forcefield): openmm forcefield
        out_path (str): output path to where to save the topology files
    """
    os.makedirs(out_path, exist_ok=True)
    new_system = forcefield.createSystem(topology,
                                            nonbondedMethod=app.PME,
                                            nonbondedCutoff=10*openmmunit.angstrom,
                                            removeCMMotion=False,
                                            rigidWater=False,
                                            hydrogenMass=3.0*openmmunit.amu)
    
    parmed_structure = parmed.openmm.topsystem.load_topology(topology, new_system, positions)   
    
    parmed_structure.save(f'{out_path}/system.prmtop', overwrite=True, format="amber")
    parmed_structure.save(f'{out_path}/system.rst7', overwrite=True, format="rst7")

    return

def select_platform(platform_name: str=None):

    if platform_name == None or platform_name == 'fastest':
        platform_name = get_fastest_platform().getName()

    try:
        platform = Platform.getPlatformByName(platform_name)
        logging.info(f'Using {platform_name} platform.')

        if platform_name in ['OpenCL']:
            platform.setPropertyDefaultValue('Precision', 'mixed')
            platform.setPropertyDefaultValue('DeviceIndex','0')
        if platform_name in ['CUDA']:
            platform.setPropertyDefaultValue('DeterministicForces', 'false')
            platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
            platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
    except:
        logging.error(f'Something went wrong trying to get {platform_name} platform.')

    return platform


def add_reporters(simulation,
                  out_dir:str='test', 
                  suffix:str='RestEq', 
                  total_steps:int=250000,
                  logperiod:int=1000,
                  ):
    
    """ Set up the reporters """
    
    simulation.reporters = []  # Delete all current reporters

    simulation.reporters.append(StateDataReporter(stdout, logperiod, step=True,
                                                time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                temperature=True, progress=True, volume=True, density=True,
                                                remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))

    simulation.reporters.append(StateDataReporter(f'{out_dir}/statistics_{suffix}.csv', 
                                                logperiod, step=True,time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                temperature=True, progress=True, volume=True, density=True,
                                                remainingTime=True, speed=True, totalSteps=total_steps))

    # Save coordinates every N logperiods
    simulation.reporters.append(DCDReporter(f"{out_dir}/trajectory_{suffix}.dcd",
                                            reportInterval=logperiod, enforcePeriodicBox=None))
    return 

def print_current_forces(system):
    for index, fc in enumerate(system.getForces()):
        print(f'Force Index:{index} | Name: {fc.getName()} | Group: {fc.getForceGroup()}')
    return None

def get_ligand_ha(topology, lig_name: str = "UNK"):
    """get names for all non-hydrogen ligand atoms"""

    residues = topology.residues()
    lig_ha_idx = []
    lig_ha_names = []
    for r in residues:
        if r.name == lig_name:
            lig_ha_names = [a.name for a in r.atoms() if not a.name.startswith("H")]
            lig_ha_idx = [a.index for a in r.atoms() if not a.name.startswith("H")]

    return lig_ha_idx, lig_ha_names

# def get_pocket_ha(topology, pocket_resid:list[int]=None):
#     """get names for all non-hydrogen ligand atoms"""

#     residues = topology.residues()
#     pocket_ha_idx = []

#     for r in residues:
#         if r.index in pocket_resid:
#             print(f'match for {r.index} {r.name} {r.id}')
#             res_ha_idx = [a.index for a in r.atoms() if not a.name.startswith('H')]
#             pocket_ha_idx.extend(res_ha_idx)

#     return pocket_ha_idx

def get_COG_dist(simulation, groupA, groupB):

    # Get COM distance between two groups of atoms
    positions = simulation.context.getState(getPositions=True).getPositions()
    g1_positions = [positions[index]/openmmunit.nanometers for index in groupA]
    g2_positions = [positions[index]/openmmunit.nanometers for index in groupB]
    dist = np.linalg.norm(np.mean(np.asarray(g1_positions), axis=0) - np.mean(np.asarray(g2_positions), axis=0))
    
    return dist # This is unitless but its nm because of OpenMM

def calculate_com_distance(u, lig_name, pocket_atoms):

    ligand_atoms = u.select_atoms(f'resname {lig_name} and (not name H*)')

    com_distance = []
    for ts in u.trajectory:
        lig_com = ligand_atoms.center_of_mass()
        prot_com = pocket_atoms.center_of_mass()
        com_distance.append(np.linalg.norm(prot_com-lig_com))

    return pd.DataFrame(com_distance, columns=['com_d'], index=range(len(com_distance)))

def calculate_cog_distance(u, lig_name, pocket_atoms):

    ligand_atoms = u.select_atoms(f'resname {lig_name} and (not name H*)')

    cog_distance = []
    for ts in u.trajectory:
        lig_cog = ligand_atoms.center_of_geometry(pbc=True)
        prot_cog = pocket_atoms.center_of_geometry(pbc=True)
        dist = np.linalg.norm(prot_cog-lig_cog) / 10 #to nm
        cog_distance.append(dist)

    return pd.DataFrame(cog_distance, columns=['cog_d'], index=range(len(cog_distance)))

def get_ligand_rmsd(u, lig_resname, alig_select):
    """A function to calculate the ligand RMSD from a trajectory.

    Parameters
    ----------
    'u : Universe
        MDAnalysis Universe
    lig_resname : str
        Residue name of the ligand that was biased.
    alig_select : str
        Selection to be considered in the alignment.
    Returns
    -------
    rmsds : np.array 
        ligand rmsd for every frame of the trajectory.
    """
    if alig_select == 'ligand':
        alig_select = f'resname {lig_resname} and not name H*'
        
    # Align each frame using the backbone as reference
    # Calculate the RMSD of ligand heavy atoms
    r = RMSD(u, 
            select=alig_select,
            groupselections=[f'resname {lig_resname} and not name H*'],
            ref_frame=0).run()
    # Get the PoseScores as np.array
    rmsds = r.rmsd[1:, -1]

    return pd.DataFrame(rmsds, columns=['rmsd'], index=range(len(rmsds)))

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

def find_inflexion_points(X, Y):

        # Fit a cubic spline to the data
        cs = CubicSpline(X, Y)

        # Define the second derivative of the spline
        def second_derivative(x):
            return cs(x, 2)  # 2 indicates the second derivative

        # Find potential inflection points by solving second_derivative(x) = 0
        initial_guesses = np.linspace(X.min(), X.max(), num=3)
        inflexion_points = fsolve(second_derivative, initial_guesses)

        return inflexion_points

def extract_sMD_statistics(files):
    data=[]
    for f in files:
        run_n = os.path.splitext(os.path.basename(f))[0].split('_')[2]

        df = pd.read_csv(f,  names=['r0', 'COMDist', 'force', 'work'])#[:500]
        df['replica'] = f'rep_{run_n}'
        df.reset_index(inplace=True, drop=False)
        data.append(df)

    data = pd.concat(data, axis=0)
    data['time'] = data['index'] / 1000
    data.reset_index(inplace=True, drop=True)

    return data

from sklearn.cluster import KMeans

# Find the closest points to the centroids
def find_closest_points(X, centroids):
    closest_points = []
    for centroid in centroids:
        distances = np.linalg.norm(X - centroid, axis=1)
        closest_point_index = np.argmin(distances)
        closest_points.append(closest_point_index)
    return closest_points

def cluster_data(data, var_names, n_clust):

    X = data[var_names].values
    kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init="auto").fit(X)
    data['cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # This is to order cluster centroids or milestones by distance
    cluster_means = data.groupby('cluster')[var_names].mean().reset_index()
    sorted_clusters = cluster_means.sort_values(by='cog_d').reset_index(drop=True)
    sorted_clusters['new_cluster'] = range(1, len(sorted_clusters) + 1)
    cluster_mapping = sorted_clusters.set_index('cluster')['new_cluster'].to_dict()
    data['milestone'] = data['cluster'].map(cluster_mapping)

    closest_points_indices = find_closest_points(X, centroids)
    closest_points_df = data.iloc[closest_points_indices]
    
    return data, closest_points_df

def cluster_pulling_MD(traj_files, equilibrated_system, prmtop_file, pocket_selection, n_clusters):
    distances = []
    for traj in traj_files:

        u_ref = mda.Universe(equilibrated_system)
        
        run_n = os.path.splitext(os.path.basename(traj))[0].split('_')[2]

        universe = mda.Universe(prmtop_file, traj, in_memory=True)

        reference = u_ref.select_atoms('protein and name CA')
        aligner = align.AlignTraj(universe, reference=reference, select="protein and name CA", in_memory=True).run()

        pocket_select = u_ref.select_atoms(pocket_selection)

        cog_d = calculate_cog_distance(universe, 'UNK', pocket_select)
        rmsd = get_ligand_rmsd(universe, lig_resname='UNK', alig_select='ligand')
                
        dat = pd.concat([cog_d, rmsd], axis=1)
        # dat['sysname'] = sys_name
        dat['replica'] = f'rep_{run_n}'
        distances.append(dat)
        
    df = pd.concat(distances, axis=0)
    df.reset_index(inplace=True, drop=False)
    df.dropna(inplace=True)

    df_clustered, closest_points = cluster_data(df, ['rmsd','cog_d'], n_clusters)
    
    return df_clustered, closest_points

def cluster_milestones_pdbs(files, lig_resname, pocket_select, n_clust):
    distances = []
    for f in files:
        u = mda.Universe(f, in_memory=True)
        cog_dist = calculate_cog_distance(u, lig_resname, pocket_select)
        cog_dist['fname'] = f
        distances.append(cog_dist)
    df_dist = pd.concat(distances, axis=0)
    clustered_data, milestones = cluster_data(df_dist, ['cog_d'], n_clust)
    return clustered_data, milestones

def write_centroids_pdb(closest_points_df, prmtop_file, sys_name):

    os.makedirs(f"{sys_name}/milestones", exist_ok=True)

    for idx, row in closest_points_df.iterrows():
        
        replica = row['replica'].split('_')[1]
        milestone = row['milestone']
        frame = row['index']

        traj_file = f'{sys_name}/sMD/trajectory_sMD_{replica}.dcd'
        u = mda.Universe(prmtop_file, traj_file, in_memory=True)

        # Get the frame and write a pdb
        u.trajectory[frame]
        u.atoms.write(f'{sys_name}/milestones/milestone_{milestone}.pdb')

    # Include the equilibrated initial pose as milestone 0
    shutil.copyfile(f'{sys_name}/system_equilibrated.pdb', f'{sys_name}/milestones/milestone_0.pdb')

    return 