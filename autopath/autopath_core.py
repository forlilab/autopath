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
from MDAnalysis.analysis.distances import distance_array

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
        pocket_selection: str = "protein and (around 5 resname UNK) and name CA",
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
        sMD_pulling_dist: float = 1.5,  # nm
        sMD_time: int = 1,  # ns
        sMD_steps_per_move: int = 250,  # 1 ps
        sMD_pulling_force: float = 1000,  # KJ/mol/nm2
        sMD_replicas: int = 3,
        sMD_autostop: bool = False,
        extract_milestones: bool = True,
        n_milestones: int = 5,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_bias_factor: int = 12,
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
        self.sMD_pulling_dist = sMD_pulling_dist
        self.sMD_time = sMD_time
        self.sMD_replicas = sMD_replicas
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_pulling_force = sMD_pulling_force
        self.sMD_autostop = sMD_autostop
        # Milestones
        self.extract_milestones = extract_milestones
        self.n_milestones = n_milestones
        self.relax_steps = relax_steps
        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_bias_factor = mMD_bias_factor
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
                                restrained_minimization=False,
                                is_membrane=self.is_membrane,
                                protocol_fname=self.protocol_fname,
                                )
            equilibrated_system = equilibration.run(solvated_system_pdb, run_id=sys_name)
        
            # Wrap, align and save the clean trajectory
            align_trajectory(prmtop_file=prmtop_file,
                            traj_file=equilibrated_traj,
                            out_fname=equilibrated_traj.replace(".dcd", "_aligned.dcd"))
            os.remove(equilibrated_traj)
            
        # Equilibration VS checkpoint
        equilibrated_traj = equilibrated_traj.replace(".dcd", "_aligned.dcd")
        u_eq = mda.Universe(prmtop_file, equilibrated_traj, in_memory=True)
        lig_rmsd_equilibration = compute_rmsd(u_eq, u_eq,
                                            alig_select="backbone", 
                                            groupselections={"ligand":f"resname {lig_resname} and not name H*", 
                                                            "protein":'protein and not name H*'},
                                            out_dir=f"{sys_name}/equilibration"
                                            )
        lig_rmsd_equilibration.to_csv(f"{sys_name}/equilibration/{sys_name}_ligand_rmsd.csv", index=False)
        plot_atomic_rmsf(u_eq, outname=f"{sys_name}/equilibration/{sys_name}_RMSF.png", log_rmsf=True)

        # if self.equilibration_checkpoint:
        #     final_rmsd = lig_rmsd_equilibration[-1:].values
        #     if final_rmsd > self.eq_checkpoint_cutoff * 10:  # to Angs
        #         logging.warning(f"Simulation for ligand {sys_name} terminated because ligand RMSD={final_rmsd:.2f} > {self.eq_checkpoint_cutoff}")
        #         exit(1)


        u_eq.trajectory[-1]  # set pointer to last frame

        # Get pocket atoms
        pocket_atoms = u_eq.select_atoms(self.pocket_selection)
        pocket_atom_indices = [atom.index for atom in pocket_atoms]
        pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
        # pocket_full_names = [f"{atom.resname}_{atom.resid}_{atom.index}" for atom in pocket_atoms]
        logging.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")
        
        #write out the pocket atoms to a pdb
        with mda.Writer(f"{sys_name}/pocket_atoms.pdb", u_eq.atoms.n_atoms) as W:
            W.write(pocket_atoms)
        #write out the pocket atoms to a pdb
        with mda.Writer(f"{sys_name}/equilib.pdb", u_eq.atoms.n_atoms) as W:
            W.write(u_eq.atoms)

        if self.use_murcko_scaffold:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            from rdkit.Chem import rdDepictor, Draw
            try:
                ligand_selection = u_eq.select_atoms(f'resname {lig_resname}')
                mol = ligand_selection.convert_to('RDKIT')
                sel_atoms = mol.GetAtoms()
                # Get Murcko scaffold and match it to parent molecule
                murcko = MurckoScaffold.GetScaffoldForMol(mol)
                murcko_match = mol.GetSubstructMatch(murcko)
                murcko_atom_names = [sel_atoms[i].GetProp('_MDAnalysis_name') for i in murcko_match]
                # select the Murcko scaffold atoms in the MDAnalysis universe
                ligand_atoms = u_eq.select_atoms(f'name {" ".join(map(str, murcko_atom_names))}')
                
                # save the Murcko scaffold image
                rdDepictor.Compute2DCoords(mol)
                mol = Chem.RemoveHs(mol)
                img = Draw.MolToImage(mol, size=(300, 300), highlightAtoms=murcko_match)
                img.save(f'{sys_name}/ligand_{lig_resname}_murcko.png')
            except Exception as e:
                logging.error(f"Error processing Murcko scaffold: {e}")
                logging.warning("Falling back to using all non-hydrogen atoms in the ligand.")
                ligand_atoms = u_eq.select_atoms(f'resname {lig_resname} and not name H*')
        else:
            # If not using Murcko scaffold, select all non-hydrogen atoms in the ligand
            logging.warning("Using all non-hydrogen atoms in the ligand as ligand_atoms.")
            ligand_atoms = u_eq.select_atoms(f"resname {lig_resname} and not name H*")

        ligand_atoms_indices = [atom.index for atom in ligand_atoms]

        eq_com = calculate_com_distance(u_eq, ligand_atoms, pocket_atoms)[-1] /10 # convert to nm
        logging.info(f"COM distance after equilibration is: {eq_com:.2f} nm")

        u_eq.trajectory[-1]  # set pointer to last frame
        restrained_atoms = u_eq.select_atoms("group pocket_atoms and name CA", pocket_atoms=pocket_atoms)
        restrained_atoms_indices = [atom.index for atom in restrained_atoms]
        restrained_atoms_full_names = [f"{atom.resname}_{atom.resid}_{atom.index}" for atom in restrained_atoms]
        logging.info(f"Restrained atoms are: {', '.join(set(restrained_atoms_full_names))}")

        ##############################################################################################
        ##################################### Steered MD simulations #################################
        ##############################################################################################

        if self.run_sMDpulling:
            
            equilibrated_system = load_system(f"{sys_name}/equilibration/system_equil_{sys_name}.xml")

            sMD = SteeredMD(
                system=equilibrated_system,
                topology=topology,
                ligand_atoms=ligand_atoms_indices,
                pocket_atoms=pocket_atom_indices,
                restrained_atoms=restrained_atoms_indices,
                restart_velocities=True,
                out_dir=f"{sys_name}/sMD",
            )

            sMD.run(
                checkpoint_file=equilibrated_chk,
                sMD_time=self.sMD_time,
                displacement=self.sMD_pulling_dist,
                steps_per_move=self.sMD_steps_per_move,
                pulling_force=self.sMD_pulling_force,
                replicas=self.sMD_replicas,
            )

            # Load and align the sMD trajectories
            sMD_trajs = glob(f"{sys_name}/sMD/trajectory_sMD*")
            for traj_file in sMD_trajs:
                align_trajectory(
                                prmtop_file=prmtop_file,
                                traj_file=traj_file,
                                out_fname=traj_file.replace(".dcd", "_aligned.dcd"),
                            )
                os.remove(traj_file) # remove the dcd

        ##############################################################################################
        ###################################### Extract Milestones ####################################
        ##############################################################################################
        
        if self.extract_milestones:
            from deeptime.clustering import RegularSpace

            os.makedirs(f"{sys_name}/milestones/pdbs", exist_ok=True)

            sMD_trajs = glob(f"{sys_name}/sMD/*_aligned.dcd")
            u_all = mda.Universe(prmtop_file, sMD_trajs)

            # use the same pocket selection as in the equilibration, but create a new atomgroup for this Universe
            pocket_atoms = u_all.select_atoms(f'index {" ".join(map(str, pocket_atom_indices))}')
            ligand_atoms = u_all.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')

            # pocket_atom_indices = [atom.index for atom in pocket_atoms]
            pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
            logging.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")
            pocket_full_names = [f"{atom.resname}_{atom.resid}_{atom.index}" for atom in pocket_atoms]
            # logging.info(f"Pocket atoms are: {', '.join(set(pocket_full_names))}")

            coms = calculate_com_distance(u_all, ligand_atoms, pocket_atoms, wrap=False)
            rmsd = compute_rmsd(u_all, u_all, alig_select=f"resname {lig_resname} and (not name H*)",
                                groupselections={'ligand': f"resname {lig_resname} and (not name H*)"},
                                                out_dir=f"{sys_name}/milestones/pdbs")
            rmsd['COM'] = coms
            X = rmsd[['RMSD_ligand', 'COM']].values

            # cluster_estimator = KMeans(n_clusters=5)
            cluster_estimator = RegularSpace(dmin=3, max_centers=self.n_milestones)
            fitted_model = cluster_estimator.fit(X).fetch_model()
            cluster_centers = fitted_model.cluster_centers
            labels = fitted_model.transform(X)

            closest_frames = match_cluster_centroids(X, cluster_centers) 

            # for i, idx in enumerate(closest_frames):
            #     dist = np.linalg.norm(X[idx] - cluster_centers[i])
            #     logging.info(f"Cluster {i}: Closest frame is {idx} (distance = {dist:.3f})")

            # Plot the clustering results
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=rmsd['RMSD_ligand'], y=rmsd['coms'], hue=labels, palette='viridis')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
            plt.xlabel('RMSD (A)'); plt.ylabel('COM Distance (A)')
            plt.title(f"{sys_name} sMD clustering")
            plt.legend()
            plt.savefig(f"{sys_name}/milestones/pdbs/milestones_clustering_plot.png")
            plt.close()

            # Write each representative frame to a PDB
            u_all.trajectory[0]  # reset
            for i, frame_index in enumerate(closest_frames):
                u_all.trajectory[frame_index]
                with mda.Writer(os.path.join(f"{sys_name}/milestones/pdbs", f"milestone_{i+1}_frame_{frame_index}.pdb"), u_all.atoms.n_atoms) as W:
                    W.write(u_all.atoms)

            # Add the equilibrated system is in the milestones folder
            # u_eq.trajectory[-1] 
            # with mda.Writer(os.path.join(f"{sys_name}/milestones/pdbs", f"milestone_0_frame_0.pdb"), u_eq.atoms.n_atoms) as W:
            #     W.write(u_eq.atoms)

        ##############################################################################################
        ##################################### Metadynamics simulations ###############################
        ##############################################################################################

        if self.run_metadynamics:

            min_com = eq_com * 0.75
            max_com = self.sMD_pulling_dist

            milestones = glob(f'{sys_name}/milestones/pdbs/milestone_*.pdb')           
            if len(milestones) == 0:
                logging.error("No milestones found. Please check the milestone extraction step.")
                exit(1)

            #sort the milestones by their index
            milestones.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

            milestone_relax = RelaxMD(
                topology=topology,
                ligand_atoms=ligand_atoms_indices,
                pocket_atoms=pocket_atom_indices,
                out_dir=f"{sys_name}/milestones",
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
                milestone_system = f"{sys_name}/milestones/{milestone_name}_relax_system.xml"
                milestone_chk = f"{sys_name}/milestones/{milestone_name}_relax_checkpoint.chk"

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
                        hill_height=self.mMD_hill_height, #kJ/mol
                        hill_width=self.mMD_hill_width, #nm
                        bias_frequency=2, #ps
                        grid_dimensions=(min_com, max_com),
                        # grid_dimensions=(0, 1),
                    )
                except Exception as e:
                    logging.error(f"Error during WTMetaD for {milestone_name}: {e}")
                    continue

        # Load and align the WTMetaD trajectories
        WTMetaD_trajs = glob(f"{sys_name}/metadynamics/trajectory_metadynamics*")
        for traj_file in WTMetaD_trajs:
            align_trajectory(prmtop_file=prmtop_file,
                            traj_file=traj_file,
                            out_fname=traj_file.replace(".dcd", "_aligned.dcd"),
                            )
            os.remove(traj_file) # remove the dcd

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished AutoPath simulation in {simulation_time/60:.2f} min.")

        return