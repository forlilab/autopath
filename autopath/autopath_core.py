
import os
import time
import logging
import pandas as pd
from glob import glob
import MDAnalysis as mda

from autopath.utils import *
from autopath.analysis import *
from autopath import SystemPreparation, Equilibration, SteeredMD, RelaxMD, MetadynamicsMD
from autopath.config import Config

# OpenMM imports
from openmm import *
from openmm.app import *
from openmm.unit import *

class AutoPath:
    def __init__(self, 
                VS_mode: bool = False,
                pdb_path: str = None,
                do_fix_pdb: bool = True,
                pocket_selection: str = 'protein and (around 3 resname UNK) and (not name H*)',
                temperature: float = 300,
                random_state: int = 42,
                run_preparation: bool = True,
                forcefield: list = None,
                lig_ff: str = 'espaloma',
                boxShape: str = 'dodecahedron',
                padding: float = 1.0,
                ionicStrength: float = 0.0,
                run_equilibration: bool = True,
                run_sMDpulling: bool = True,
                sMD_pulling_dist: float = 0.5, # nm
                sMD_time: int = 1, # ns
                sMD_steps_per_move: int = 250, # 1 ps
                sMD_pulling_force: float = 20000, # KJ/mol/nm2
                sMD_replicas: int = 5,
                sMD_autostop: bool = False,
                extract_milestones: bool = True,
                n_milestones: int = 10,
                run_relax: bool = True,
                relax_steps: int = 25000,
                run_metadynamics: bool = True,
                mMD_walkers: int = 10,
                mMD_bias_factor: int = 3,
                mMD_hill_height: float = 0.3, # Kcal/mol
                mMD_time: int = 1, # ns 
    ):
        # General
        self.pocket_selection = pocket_selection
        self.temperature = temperature
        self.random_state = random_state
        # Preparation
        self.run_preparation = run_preparation
        self.forcefield = forcefield
        self.lig_ff = lig_ff
        self.boxShape = boxShape
        self.padding = padding
        self.ionicStrength = ionicStrength
        # Equilibration
        self.run_equilibration = run_equilibration
        # Steered MD
        self.run_sMDpulling = run_sMDpulling
        self.sMD_pulling_dist = sMD_pulling_dist
        self.sMD_time = sMD_time
        self.sMD_replicas = sMD_replicas
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_pulling_force = sMD_pulling_force
        self.sMD_autostop = sMD_autostop
        # Milestones
        self.extract_milestones = extract_milestones
        self.n_milestones = n_milestones
        self.run_relax = run_relax
        self.relax_steps = relax_steps
        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_walkers = mMD_walkers
        self.mMD_bias_factor = mMD_bias_factor
        self.mMD_hill_height = mMD_hill_height
        self.mMD_time = mMD_time
        # VS mode
        self.equilibration_checkpoint = False
        self.pulling_checkpoint = False
        if VS_mode:
            self.equilibration_checkpoint = True
            self.eq_checkpoint_cutoff = 0.3 # nm 
            self.pulling_checkpoint = True

        # Process the input PDB
        if do_fix_pdb:
            protein_pdb = fix_pdb(pdbfile=pdb_path, keep_heterogens=True, pH=7.4)
            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            self.protein_file = f'input/{pdb_name}_fixed.pdb'
            save_pdb(protein_pdb.topology, protein_pdb.positions, self.protein_file)
        else:
            self.protein_file = pdb_path

    def run(self,
            ligand_file: str = None,
            lig_resname: str = 'UNK'
            ):

        start_time = time.monotonic()

        sys_name = os.path.splitext(os.path.basename(ligand_file))[0]
        os.makedirs(sys_name, exist_ok=True)
        os.makedirs(f'{sys_name}/plots', exist_ok=True)

        logging.basicConfig(
            level="INFO",
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
                logging.StreamHandler(),
            ],
        )

        logging.info(f'Processing system {sys_name}')

        if self.run_preparation:
            prepare_system = SystemPreparation(lig_ff='espaloma',
                                               boxShape='dodecahedron',
                                               )
            
            prepare_system.run(self.protein_file, ligand_file)

        prmtop_file = f'{sys_name}/system.prmtop'
        system_file = f'{sys_name}/system.xml'
        solvated_system = f'{sys_name}/system.pdb'
        equilibrated_pdb = f'{sys_name}/system_equilibrated.pdb'
        equilibrated_traj = f'{sys_name}/trajectory_equilibration.dcd'
        equilibrated_chk = f'{sys_name}/equilibration_checkpoint.chk'

        # Run restrained equilibration
        if self.run_equilibration:
            equilibration = Equilibration(system_file,
                                          prmtop_file, 
                                          sys_name)
            
            equilibration.run(solvated_system)

        # Equilibration VS checkpoint
        u_eq = mda.Universe(prmtop_file, equilibrated_traj, in_memory=True)
        eq_rmsd = get_ligand_rmsd(u_eq, alig_select='backbone', lig_resname=lig_resname)
        plot_rmsd(eq_rmsd, sys_name, 'equilibration')

        if self.equilibration_checkpoint:
            final_rmsd = eq_rmsd[-1:].values
            if final_rmsd > self.eq_checkpoint_cutoff * 10: #to Angs
                logging.error(f'Simulation for ligand {sys_name} terminated because ligand RMSD={final_rmsd:.2f} > {self.eq_checkpoint_cutoff}')
                exit(1)

        # Get pocket atoms
        u_eq.trajectory[-1] # set pointer to last frame
        pocket_select = u_eq.select_atoms(self.pocket_selection)
        pocket_atoms = [atom.index for atom in pocket_select]
        pocket_residues = [f'{atom.resname}_{atom.resid}' for atom in pocket_select]
        pocket_full_names = [f'{atom.resname}_{atom.resid}_{atom.index}' for atom in pocket_select]
        
        # u_eq.trajectory[0] # set pointer to first frame
        eq_cog = calculate_cog_distance(u_eq, lig_resname, pocket_select)
        final_cog = eq_cog.values[-1][0]

        logging.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")
        logging.info(f"Pocket atoms are: {', '.join(set(pocket_full_names))}")

        logging.info(f'COG distance after equilibration is: {final_cog:.2f} nm')

        # Run pulling simulations
        if self.run_sMDpulling :

            steered_MD = SteeredMD(equilibrated_chk,
                                system_file,
                                prmtop_file,
                                pocket_atoms=pocket_atoms,
                                sys_name=sys_name)

            steered_MD.run(
                        sMD_time=self.sMD_time,
                        displacement=self.sMD_pulling_dist,
                        steps_per_move=self.sMD_steps_per_move,
                        pulling_force=self.sMD_pulling_force,
                        replicas=self.sMD_replicas)
            
        if self.extract_milestones:

            sMD_trajs = glob(f'{sys_name}/sMD/trajectory_sMD*')
            clustered_data, closest_points = cluster_pulling_MD(sMD_trajs, 
                                                                equilibrated_pdb, 
                                                                prmtop_file, 
                                                                self.pocket_selection, 
                                                                n_clusters=self.n_milestones)
            closest_points.to_csv(f'{sys_name}/closest_points.csv')
            print(closest_points)
            #TODO move this insed clustrring method
            write_centroids_pdb(closest_points, prmtop_file, sys_name)
            plot_clusters(clustered_data, closest_points, sys_name)

            initial_cluster_centroids = sorted(glob(f'{sys_name}/milestones/milestone_*.pdb', recursive=True))

            relaxMD = RelaxMD(system_file=system_file,
                              prmtop_file=prmtop_file, 
                              lig_name=lig_resname, 
                              sys_name=sys_name,
                              pocket_atoms=pocket_atoms)
            
            for milestone in initial_cluster_centroids:
                try:
                    milestone_idx = os.path.splitext(os.path.basename(milestone))[0]
                    logging.info(f'Relaxing {milestone_idx}')
                    relaxMD.run(pdb_file=milestone,
                                run_id=milestone_idx,
                                md_steps=self.relax_steps
                                )
                except:
                    logging.error(f'Relaxing failed for {milestone_idx}')
                    pass

        milestones = glob(f'{sys_name}/milestones/milestone_*_relax.pdb')

        if self.mMD_walkers < self.n_milestones:
            # Cluster the milestones to get one milestones per walker    
            clustered_data, milestones_df = cluster_milestones_pdbs(milestones, 
                                                                    lig_resname, 
                                                                    pocket_select, 
                                                                    n_clust=self.mMD_walkers)
            milestones = milestones_df['fname'].values

        if self.run_metadynamics:

            min_cog = final_cog * 0.5
            max_cog = final_cog + self.sMD_pulling_dist

            metadynamics_MD = MetadynamicsMD(sys_name, 
                                            prmtop_file, 
                                            lig_name=lig_resname, 
                                            pocket_atoms=pocket_atoms)

            for milestone in milestones:
                basename = os.path.splitext(os.path.basename(milestone))[0]
                system_file = f'{sys_name}/milestones/{basename}_system.xml'
                checkpoint_file = f'{sys_name}/milestones/{basename}_checkpoint.chk'
                
                logging.info(f'Running metadynamics for {basename}/{len(milestones)}')
                metadynamics_MD.run(system_file=system_file,
                                    checkpoint_file=checkpoint_file,
                                    run_id=basename,
                                    bias_factor=self.mMD_bias_factor,
                                    hill_height=self.mMD_hill_height,
                                    mMD_time=self.mMD_time,
                                    grid_dimensions=(min_cog, max_cog))
                
        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished AutoPath simulation in {simulation_time/60:.2f} min.')