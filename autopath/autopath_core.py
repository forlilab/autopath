# General imports
import os
import time
import logging
import pandas as pd
import warnings
import shutil
from sys import exit
from glob import glob
import MDAnalysis as mda
import mdtraj as md

# OpenMM imports
from openmm import *
from openmm.app import *
from openmm.unit import *

# AutoPath imports
from autopath.utils import *
from autopath.analysis import *
from autopath import (
    SystemPreparation,
    Equilibration,
    SteeredMD,
    RelaxMD,
    MetadynamicsMD,
)

class AutoPath:
    def __init__(
        self,
        VS_mode: bool = False,
        pdb_path: str = None,
        do_fix_pdb: bool = True,
        pocket_selection: str = "same residue as protein and (around 4 resname UNK) and (not name H*)",
        use_murcko_scaffold: bool = True,
        temperature: float = 300,
        random_state: int = 42,
        run_preparation: bool = True,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3p_HFE_multivalent.xml",
        ],
        lig_ff: str = "espaloma",
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        ionicStrength: float = 0.15,
        variants: dict = None,
        is_membrane: bool = False,
        lipid_type: str = None,
        run_equilibration: bool = True,
        protocol_fname: str = None,
        run_sMDpulling: bool = True,
        sMD_pulling_speeds: dict = {0.001:2, 0.002:2, 0.003:2},  # nm/ps
        sMD_max_pulling_dist: float = None,  # nm
        sMD_time: int = None,  # ns
        sMD_steps_per_move: int = 250,  # 1 ps
        sMD_pulling_force: float = 50,  # KJ/mol/nm2
        sMD_autostop_freq: int = 10, #moves
        extract_milestones: bool = True,
        n_milestones: int = 5,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_bias_factor: int = 12,
        mMD_bias_frequency: int = 2,  # ps
        mMD_hill_height: float = 0.3,  # Kcal/mol approx 0.5KT
        mMD_hill_width: float = 0.02,
        mMD_time: int = 3,  # ns
    ):
        # General
        self.pocket_selection = pocket_selection
        self.use_murcko_scaffold = use_murcko_scaffold
        self.temperature = temperature
        self.random_state = random_state
        # Preparation
        self.run_preparation = run_preparation
        self.forcefield = forcefield
        self.lig_ff = lig_ff
        self.boxShape = boxShape
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.variants = variants
        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        # Equilibration
        self.run_equilibration = run_equilibration
        self.protocol_fname = protocol_fname
        # Steered MD
        self.run_sMDpulling = run_sMDpulling
        self.sMD_max_pulling_dist = sMD_max_pulling_dist
        self.sMD_time = sMD_time
        self.sMD_pulling_speeds = sMD_pulling_speeds
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_pulling_force = sMD_pulling_force
        self.sMD_autostop = sMD_autostop_freq
        # Milestones
        self.extract_milestones = extract_milestones
        self.n_milestones = n_milestones
        self.relax_steps = relax_steps
        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_bias_factor = mMD_bias_factor
        self.mMD_bias_frequency = mMD_bias_frequency
        self.mMD_hill_height = mMD_hill_height
        self.mMD_hill_width = mMD_hill_width
        self.mMD_time = mMD_time
        # VS mode
        self.equilibration_checkpoint = False
        self.pulling_checkpoint = False
        if VS_mode:
            self.equilibration_checkpoint = True
            self.eq_checkpoint_cutoff = 0.3  # nm
            self.pulling_checkpoint = True

        # Process the input PDB
        if do_fix_pdb:
            protein_pdb = fix_pdb(pdbfile=pdb_path, keep_heterogens=True, pH=7.4)
            self.protein_file = pdb_path.replace(".pdb", "_fixed.pdb")
            save_pdb(protein_pdb.topology, protein_pdb.positions, self.protein_file)
        else:
            self.protein_file = pdb_path

    def run(self, ligand_file: str = None, lig_resname: str = "UNK"):

        start_time = time.monotonic()

        if ligand_file is not None:
            sys_name = os.path.splitext(os.path.basename(ligand_file))[0]
        else:
            sys_name = os.path.splitext(os.path.basename(self.protein_file))[0]

        os.makedirs(sys_name, exist_ok=True)

        logging.basicConfig(
            level="INFO",
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
                logging.StreamHandler(),
            ],
        )

        # logger = logging.getLogger('autopath_core')

        logging.info(f"Processing system {sys_name}")

        ##############################################################################################
        ####################################### System preparation ###################################
        ##############################################################################################

        prmtop_file = f"{sys_name}/system.prmtop"
        solvated_system_pdb = f"{sys_name}/system.pdb"

        if self.run_preparation:
            prepare_system = SystemPreparation(
                out_dir=sys_name,
                forcefield=self.forcefield,
                lig_ff=self.lig_ff,
                boxShape=self.boxShape,
                padding=self.padding,
                ionicStrength=self.ionicStrength,
                is_membrane=self.is_membrane,
                lipid_type=self.lipid_type,
            )
            system, topology = prepare_system.run(self.protein_file, self.variants, ligand_file)

        system = load_system(f"{sys_name}/system.xml")
        topology = AmberPrmtopFile(prmtop_file).topology

        ##############################################################################################
        ##################################### System equilibration ###################################
        ##############################################################################################

        equilibrated_traj = f"{sys_name}/equilibration/trajectory_equilibration_{sys_name}.dcd"
        equilibrated_chk = f"{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk"

        if self.run_equilibration:
            equilibration = Equilibration(
                                topology=topology,
                                system=system,
                                out_dir=f"{sys_name}/equilibration",
                                restrained_minimization=True,
                                is_membrane=self.is_membrane,
                                protocol_fname=self.protocol_fname,
                                )
            equilibrated_system = equilibration.run(solvated_system_pdb, run_id=sys_name)
        
            #Wrap, align and save the clean trajectory
            traj = md.load(traj_file, top=prmtop_file)
            traj = traj.center_coordinates()
            traj = traj.image_molecules()
            backbone = traj.topology.select("backbone")
            traj = traj.superpose(traj[0], atom_indices=backbone)
            traj.save(traj_file.replace(".dcd", "_aligned.dcd"))
            os.remove(equilibrated_traj)

        ##############################################################################################
        ############################# Post-equilibration Analysis ####################################
        ##############################################################################################

        equilibrated_traj = equilibrated_traj.replace(".dcd", "_aligned.dcd")
        u_eq = mda.Universe(prmtop_file, equilibrated_traj, in_memory=True)
        rmsd = compute_rmsd(u_eq, u_eq,
                            alig_select="backbone", 
                            groupselections={"ligand":f"resname {lig_resname} and not name H*", 
                                            "protein":'protein and not name H*'},
                            out_dir=f"{sys_name}/equilibration"
                            )
        rmsd.to_csv(f"{sys_name}/equilibration/{sys_name}_ligand_rmsd.csv", index=False)
        plot_atomic_rmsf(u_eq, outname=f"{sys_name}/equilibration/{sys_name}_RMSF.png", log_rmsf=True)
        
        # Equilibration VS checkpoint
        # if self.equilibration_checkpoint:
        #     final_rmsd = lig_rmsd_equilibration[-1:].values
        #     if final_rmsd > self.eq_checkpoint_cutoff * 10:  # to Angs
        #         logging.warning(f"Simulation for ligand {sys_name} terminated because ligand RMSD={final_rmsd:.2f} > {self.eq_checkpoint_cutoff}")
        #         exit(1)

        pocket_atoms = get_pocket_atoms(u_eq, self.pocket_selection, f"resname {lig_resname}")
        pocket_atom_indices = [atom.index for atom in pocket_atoms]
        pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
        logging.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")

        # write out the pocket atoms to a pdb
        with mda.Writer(f"{sys_name}/pocket_atoms.pdb", u_eq.atoms.n_atoms) as W:
            W.write(pocket_atoms)
    
        ligand_atoms_indices = get_ligand_anchor_atoms(u_eq, lig_resname, 
                                                       mode='murcko', n_atoms=5,
                                                       out_dir=sys_name)
        ligand_atoms = u_eq.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')

        final_com = calculate_com_distance(u_eq, ligand_atoms, pocket_atoms, wrap=False)[-1] /10 # convert to nm
        logging.info(f"COM distance after equilibration is: {final_com:.2f} nm")

        u_eq.trajectory[-1]  # set pointer to last frame
        restrained_atoms = u_eq.select_atoms("group pocket_atoms and name CA", pocket_atoms=pocket_atoms)
        restrained_atoms_indices = [atom.index for atom in restrained_atoms]
        restrained_atoms_full_names = [f"{atom.resname}_{atom.resid}_{atom.index}" for atom in restrained_atoms]
        logging.info(f"Restrained atoms are: {', '.join(set(restrained_atoms_full_names))}")

        ##############################################################################################
        ##################################### Steered MD simulations #################################
        ##############################################################################################
        
        sMD_outdir = f"{sys_name}/sMD_pulling-w"

        if self.run_sMDpulling:
            
            equilibrated_system = load_system(f"{sys_name}/equilibration/system_equil_{sys_name}.xml")
            lig_ha_idx, lig_ha_names = get_ligand_ha(topology, lig_resname)
            # pulling_force = self.sMD_pulling_force * len(ligand_atoms_indices)  # Normalize by ligand size
            pulling_force = self.sMD_pulling_force * len(lig_ha_idx)  # Normalize by ligand size
            logging.info(f"Pulling force is set to {pulling_force} KJ/mol/nm2 for {len(lig_ha_idx)} atoms.")

            sMD = SteeredMD(
                system=equilibrated_system,
                topology=topology,
                groupA_atoms=ligand_atoms_indices,
                groupB_atoms=pocket_atom_indices,
                restrained_atoms=restrained_atoms_indices,
                restart_velocities=True,
                # timestep=0.002,  # 2 fs
                out_dir=sMD_outdir,
            )
            for speed, reps in self.sMD_pulling_speeds.items():
                for i in range(reps):
                    # sMD = SteeredMD(
                    #     system=equilibrated_system,
                    #     topology=topology,
                    #     groupA_atoms=ligand_atoms_indices,
                    #     groupB_atoms=pocket_atom_indices,
                    #     restrained_atoms=restrained_atoms_indices,
                    #     restart_velocities=True,
                    #     timestep=0.004,  # 2 fs
                    #     out_dir=sMD_outdir,
                    # )
                    sMD.run(
                        checkpoint_file=equilibrated_chk,
                        sMD_time=self.sMD_time,
                        max_displacement=self.sMD_max_pulling_dist,
                        pulling_speed=speed,  # nm/ps
                        steps_per_move=self.sMD_steps_per_move,
                        pulling_force=pulling_force,
                        rep_suffix=f'replica-{i+1}_v{speed}',
                        do_backwards=False,     #CAREFUL: this will run the pulling in both directions
                    )

            # Load and align the sMD trajectories
            sMD_trajs = glob(f"{sMD_outdir}/trajectory_sMD_replica_*_*.dcd")
            for traj_file in sMD_trajs:
                traj = md.load(traj_file, top=prmtop_file)
                traj = traj.center_coordinates()
                traj = traj.image_molecules()
                backbone = traj.topology.select("backbone")
                traj = traj.superpose(traj[0], atom_indices=backbone)
                traj.save(traj_file.replace(".dcd", "_aligned.dcd"))
                # os.remove(traj_file) # remove the dcd

        ##############################################################################################
        ###################################### Extract Milestones ####################################
        ##############################################################################################

        milestones_outdir = f"{sys_name}/milestones"
        min_dist = 6 # minimum distance between clusters of milestones
        
        if self.extract_milestones:
            
            os.makedirs(milestones_outdir, exist_ok=True)

            sMD_trajs = glob(f"{sMD_outdir}/trajectory_sMD_replica_*_*.dcd")
            sMD_trajs = [t for t in sMD_trajs if not t.endswith("_aligned.dcd")]

            if len(sMD_trajs) == 0:
                logging.error("No sMD trajectories found. Please check the sMD pulling step.")
                exit(1)
            
            #TODO move outside 
            # use the same pocket selection as in the equilibration, but create a new atomgroup for this Universe
            u_sMD = mda.Universe(prmtop_file, sMD_trajs)
            pocket_atoms = u_sMD.select_atoms(f'index {" ".join(map(str, pocket_atom_indices))}')
            ligand_atoms = u_sMD.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')
            ligand_atoms_full = u_sMD.select_atoms(f'resname {lig_resname} and not name H*')
            ligand_atoms_full_indices = [atom.index for atom in ligand_atoms_full]
            
            # calculate some features for clustering
            coms = calculate_com_distance(u_sMD, ligand_atoms_full, pocket_atoms, wrap=False)
            rmsd = compute_rmsd(u_sMD, u_sMD, 
                                alig_select=f"resname {lig_resname} and not name H*",
                                groupselections={'ligand': f"resname {lig_resname} and not name H*"},
                                out_dir=milestones_outdir)
            rmsd['COM'] = coms
            X = rmsd[['RMSD_ligand', 'COM']].values

            # I didn't use the wrapped trajs for COM distances to avoid imaging artifacts
            sMD_trajs_aligned = [traj.replace(".dcd", "_aligned.dcd") for traj in sMD_trajs]
            u_sMD_aligned = mda.Universe(prmtop_file, sMD_trajs_aligned)

            labels, sorted_cluster_centers = cluster_sMD_trajectories(u_sMD_aligned, X, 
                                                                      n_clusters=self.n_milestones,
                                                                      min_dist=min_dist,
                                                                      out_dir=milestones_outdir)

            # Plot the clustering results
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=rmsd['RMSD_ligand'], y=rmsd['COM'], hue=labels, palette='viridis')
            plt.scatter(sorted_cluster_centers[:, 0], sorted_cluster_centers[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
            plt.xlabel('RMSD (A)'); plt.ylabel('COM Distance (A)')
            plt.title(f"{sys_name} sMD clustering")
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"{milestones_outdir}/milestones_clustering_plot.png")
            plt.close()

        ##############################################################################################
        ##################################### Metadynamics simulations ###############################
        ##############################################################################################
        use_biasing_scheme = False
        biasing_scheme = {
                            1:{'height': 1.2, 'width': 0.04}, #KJ/mol and nm
                            2:{'height': 1.0, 'width': 0.05},
                            3:{'height': 0.8, 'width': 0.06},
                            # 4:{'height': 0.2, 'width': 0.07},
                            # 5:{'height': 0.1, 'width': 0.08}
                            }

        if self.run_metadynamics:

            min_com = final_com * 0.75
            max_com = final_com + self.sMD_pulling_dist

            milestones = glob(f'{milestones_outdir}/milestone_*.pdb')           
            if len(milestones) == 0:
                logging.error("No milestones found. Please check the milestone extraction step.")
                exit(1)

            #sort the milestones by their index
            milestones.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

            milestone_relax = RelaxMD(
                topology=topology,
                ligand_atoms=ligand_atoms_full_indices, # use all atoms
                pocket_atoms=pocket_atom_indices,
                out_dir=milestones_outdir,
                temp=self.temperature,
            )

            WTMetaD = MetadynamicsMD(
                topology=topology,
                ligand_atoms=ligand_atoms_indices,
                pocket_atoms=pocket_atom_indices,
                restrained_atoms=restrained_atoms_indices,
                out_dir=f"{sys_name}/metadynamics",
            )

            for milestone in milestones:
                milestone_name = os.path.basename(milestone).split('.')[0]
                milestone_number = int(milestone_name.split('_')[1])
                milestone_system = f"{milestones_outdir}/{milestone_name}_relax_system.xml"
                milestone_chk = f"{milestones_outdir}/{milestone_name}_relax_checkpoint.chk"

                if os.path.exists(milestone_system):
                    milestone_system = load_system(milestone_system)
                    logging.info(f"Loading relaxed milestone {milestone_name}")
                else:
                    logging.info(f'Relaxing milestone {milestone_name}')
                    try:
                        system = load_system(f"{sys_name}/system.xml")
                        milestone_system = milestone_relax.run(pdb_file=milestone, system=system, run_id=milestone_name)
                    except Exception as e:
                        logging.error(f"Error relaxing {milestone_name}: {e}")
                        continue
                
                logging.info(f"Running WTMetaD for milestone {milestone_name}")
                try:
                    WTMetaD.run(
                        checkpoint_file=milestone_chk,
                        system=milestone_system,
                        run_id=milestone_name,
                        mMD_CV='com',
                        mMD_time=self.mMD_time, #ns
                        bias_factor=self.mMD_bias_factor,
                        hill_height=biasing_scheme[milestone_number]['height'] if use_biasing_scheme else self.mMD_hill_height, #kcal/mol
                        hill_width=biasing_scheme[milestone_number]['width'] if use_biasing_scheme else self.mMD_hill_width, #nm
                        biasFrequency=self.mMD_bias_frequency, #ps
                        grid_dimensions=(min_com, max_com)
                    )
                except Exception as e:
                    logging.error(f"Error during WTMetaD for {milestone_name}: {e}")
                    continue

        # Load and align the WTMetaD trajectories
        WTMetaD_trajs = glob(f"{sys_name}/metadynamics/trajectory_metadynamics_milestone_*_frame_*.dcd")
        for traj_file in WTMetaD_trajs:
            traj = md.load(traj_file, top=prmtop_file)
            traj = traj.center_coordinates()
            traj = traj.image_molecules()
            backbone = traj.topology.select("backbone")
            traj = traj.superpose(traj[0], atom_indices=backbone)
            traj.save(traj_file.replace(".dcd", "_aligned.dcd"))
            # os.remove(traj_file) # remove the dcd

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished AutoPath simulation in {simulation_time/60:.2f} min.")

        return