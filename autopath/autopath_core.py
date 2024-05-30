
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
                pocket_selection:str="protein and (byres around 3 resname UNK) and backbone",
                temperature: float = 300,
                random_state: int = 42,
                run_preparation: bool = True,
                forcefield: list = None,
                lig_ff: str = 'espaloma',
                boxShape: str = 'dodecahedron',
                padding: float = 1.5,
                ionicStrength: float = 0.0,
                run_equilibration: bool = True,
                run_sMDpulling: bool = True,
                sMD_pulling_dist: float = 1.0, # nm
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
                mMD_CV: str = 'nc',
                mMD_walkers: int = 5,
                mMD_bias_factor: int = 5,
                mMD_hill_height: float = 0.3, # Kcal/mol approx 0.5KT
                mMD_time: int = 2, # ns 
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
        self.mMD_CV = mMD_CV
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
        assert pdb_path is not None, 'Please provide a valid protein PDB file'

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
        equilibrated_system = f'{sys_name}/system_equilibrated.xml'
        equilibrated_pdb = f'{sys_name}/system_equilibrated.pdb'
        equilibrated_traj = f'{sys_name}/trajectory_equilibration.dcd'
        equilibrated_chk = f'{sys_name}/equilibration_checkpoint.chk'

        # Run restrained equilibration
        if self.run_equilibration:
            equilibration = Equilibration(system_file,
                                          prmtop_file, 
                                          sys_name)
            
            equilibration.run(solvated_system)
            
            # Wrap, align and save the clean trajectory
            align_trajectory(prmtop_file, equilibrated_traj, f'{sys_name}/{sys_name}_equilibration', strip_mask=':HOH,NA,CL,K,POP')

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
        pocket_atoms = u_eq.select_atoms(self.pocket_selection)
        ligand_atoms = u_eq.select_atoms(f'resname {lig_resname} and (not name H*)')

        pocket_atom_indexes = [atom.index for atom in pocket_atoms]
        pocket_residues = [f'{atom.resname}_{atom.resid}' for atom in pocket_atoms]
        pocket_full_names = [f'{atom.resname}_{atom.resid}_{atom.index}' for atom in pocket_atoms]
        
        # u_eq.trajectory[0] # set pointer to first frame
        eq_cog = calculate_cog_distance(u_eq, ligand_atoms, pocket_atoms)
        final_cog = eq_cog.values[-1][0]

        logging.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")
        logging.info(f"Pocket atoms are: {', '.join(set(pocket_full_names))}")
        logging.info(f'COG distance after equilibration is: {final_cog:.2f} nm')

        # Run pulling simulations
        if self.run_sMDpulling :

            steered_MD = SteeredMD(equilibrated_chk,
                                equilibrated_system,
                                prmtop_file,
                                pocket_atoms=pocket_atom_indexes,
                                sys_name=sys_name)

            steered_MD.run(
                        sMD_time=self.sMD_time,
                        displacement=self.sMD_pulling_dist,
                        steps_per_move=self.sMD_steps_per_move,
                        pulling_force=self.sMD_pulling_force,
                        replicas=self.sMD_replicas)
            
            # Wrap, align and save the clean trajectory
            sMD_trajs = glob(f'{sys_name}/sMD/trajectory_sMD*')
            align_trajectory(prmtop_file, sMD_trajs, f'{sys_name}/sMD/{sys_name}_sMD_all', strip_mask=':HOH,NA,CL,K,POP')
            
        if self.extract_milestones:

            sMD_trajs = glob(f'{sys_name}/sMD/trajectory_sMD*')
            clustered_data, closest_points = cluster_pulling_MD(sMD_trajs, 
                                                                equilibrated_pdb, 
                                                                prmtop_file, 
                                                                lig_resname,
                                                                self.pocket_selection, 
                                                                n_clusters=self.n_milestones)
            #TODO move this insed clustrring method
            write_centroids_pdb(closest_points, prmtop_file, sys_name)
            plot_clusters(clustered_data, closest_points, sys_name)

            initial_cluster_centroids = sorted(glob(f'{sys_name}/milestones/milestone_*.pdb', recursive=True))

            relaxMD = RelaxMD(system_file=system_file,
                              prmtop_file=prmtop_file, 
                              lig_name=lig_resname, 
                              sys_name=sys_name,
                              pocket_atoms=pocket_atom_indexes)
            
            milestones_data = []
            for centroid_fname in initial_cluster_centroids:
                try:
                    centroid_idx = os.path.splitext(os.path.basename(centroid_fname))[0]
                    logging.info(f'Relaxing {centroid_idx}')
                    initial_dist, final_dist = relaxMD.run(pdb_file=centroid_fname,
                                                        run_id=centroid_idx,
                                                        md_steps=self.relax_steps
                                                        )
                    milestones_data.append([centroid_fname,initial_dist,final_dist])
                except:
                    logging.error(f'Relaxing failed for {centroid_idx}')
                    pass

            milestones_df = pd.DataFrame(milestones_data, columns=['milestone_fname','inital_dist','final_dist'])
            milestones_df.to_csv(f'{sys_name}/milestones/{sys_name}_milestones.csv')

        ##### METADYNAMICS #####
        if self.run_metadynamics:
            milestones_df = pd.read_csv(f'{sys_name}/milestones/{sys_name}_milestones.csv', index_col=0)
            if self.mMD_walkers < self.n_milestones:
                # Get the most diverse set of milestones based on cog distance
                walkers_df = get_most_diverse_points(milestones_df, n_points=self.mMD_walkers)
                walkers_df.sort_values(by='final_dist', ascending=True, inplace=True)
                walkers_df.to_csv(f'{sys_name}/milestones/{sys_name}_walkers.csv')
            else:
                walkers_df = milestones_df

            min_cog = final_cog * 0.5
            max_cog = final_cog + self.sMD_pulling_dist

            metadynamics_MD = MetadynamicsMD(sys_name, 
                                            prmtop_file, 
                                            lig_name=lig_resname, 
                                            pocket_atoms=pocket_atom_indexes)
            
            for walker_fname in walkers_df['milestone_fname']:
                basename = os.path.splitext(os.path.basename(walker_fname))[0]
                system_file = f'{sys_name}/milestones/{basename}_relax_system.xml'
                checkpoint_file = f'{sys_name}/milestones/{basename}_relax_checkpoint.chk'
                
                logging.info(f'Running metadynamics for {basename}/{len(walkers_df)}')

                metadynamics_MD.run(system_file=system_file,
                                    checkpoint_file=checkpoint_file,
                                    run_id=basename,
                                    bias_factor=self.mMD_bias_factor,
                                    hill_height=self.mMD_hill_height,
                                    mMD_time=self.mMD_time,
                                    mMD_CV='nc',
                                    hill_width=1,
                                    grid_dimensions=(0, 100))
                
                # metadynamics_MD.run2D(system_file=system_file,
                #                     checkpoint_file=checkpoint_file,
                #                     run_id=basename,
                #                     bias_factor=self.mMD_bias_factor,
                #                     hill_height=self.mMD_hill_height,
                #                     mMD_time=self.mMD_time,
                #                     grid_dimensions=(min_cog, max_cog))
                
            # Wrap, align and save the clean trajectory
            mMD_trajs = glob(f'{sys_name}/metadynamics/*.dcd')
            align_trajectory(prmtop_file, mMD_trajs, f'{sys_name}/metadynamics/{sys_name}_mMD_all', strip_mask=':HOH,NA,CL,K,POP')

        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished AutoPath simulation in {simulation_time/60:.2f} min.')