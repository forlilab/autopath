# General imports
import os
import json
import time
import pandas as pd
import warnings
# import shutil
from sys import exit
from glob import glob
import MDAnalysis as mda
import mdtraj as md
from collections import defaultdict

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
from autopath.customForces import (
    generate_funnel_parameters_from_trajectory,
    create_funnel_force_from_trajectory_analysis,
)
from autopath.sMDAnalysis import SMDData, SMDAnalysis
from autopath.sMDAnalysis.PathModel import DTWPathModel
from autopath.sMDAnalysis.Diagnostics import plot_convergence_traces, plot_convergence_metrics

import logging
logger = logging.getLogger('autopath.core')

class AutoPath:
    def __init__(
        self,
        VS_mode: bool = False,
        pdb_path: str = None,
        do_fix_pdb: bool = True,
        pocket_selection: str = "same residue as protein and (around 4 resname UNK) and (not name H*)",
        temperature: float = 300,
        random_state: int = 42,
        platform: str = 'fastest',  # or 'CUDA', 'OpenCL', 'CPU'
        run_preparation: bool = True,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3pfb_HFE_multivalent.xml",
        ],
        hydrogenMass: float = 1.5,  # amu
        timestep: float = 0.004,  # ps
        lig_ff: str = "espaloma",
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        ionicStrength: float = 0.15,
        ions: dict = ('Na+', 'Cl-'),
        variants: dict = None,
        is_membrane: bool = False,
        lipid_type: str = None,
        run_equilibration: bool = True,
        protocol_fname: str = None,
        run_sMDpulling: bool = True,
        sMD_outdir: str = "sMD",
        sMD_pulling_dir: str = "forward",  # "forward" or "backward"
        sMD_pulling_speeds: dict = {0.005:None, 0.0025:None, 0.001:None},  # nm/ps
        sMD_max_pulling_dist: float = 3.0,  # nm
        sMD_time: int = None,  # ns
        sMD_steps_per_move: int = None,
        sMD_dx_per_move: float = 0.001,  # nm, this is the displacement per move
        sMD_spring_cte: float = None,  # KJ/mol/nm2
        sMD_ligand_anchor_mode: str = 'lig_ha',
        sMD_autostop_freq: int = 50, #moves
        sMD_run_analysis: bool = True,
        sMD_clust_selection:str = None,
        extract_milestones: bool = True,
        milestone_mode: str = "per_path",  # "per_path" or "all_medoids"
        milestone_min_frame_separation: int = 0,
        n_milestones: int = 5,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_use_funnel_potential: bool = True,
        mMD_bias_factor: int = 10,
        mMD_bias_frequency: int = 2,  # ps
        mMD_hill_height: float = 1.2,  # kJ/mol approx 0.5KT
        mMD_hill_width: float = 0.05,
        mMD_time: int = 10,  # ns
    ):
        # General
        self.pocket_selection = pocket_selection
        self.temperature = temperature
        self.random_state = random_state
        self.platform = platform
        # Preparation
        self.run_preparation = run_preparation
        self.forcefield = forcefield
        self.hydrogenMass = hydrogenMass
        self.timestep = timestep
        self.lig_ff = lig_ff
        self.boxShape = boxShape
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.ions = ions
        self.variants = variants
        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        # Equilibration
        self.run_equilibration = run_equilibration
        self.protocol_fname = protocol_fname
        # Steered MD
        self.run_sMDpulling = run_sMDpulling
        self.sMD_outdir = sMD_outdir
        self.sMD_pulling_dir = sMD_pulling_dir
        self.sMD_max_pulling_dist = sMD_max_pulling_dist
        self.sMD_time = sMD_time
        self.sMD_pulling_speeds = sMD_pulling_speeds
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_dx_per_move = sMD_dx_per_move
        self.sMD_spring_cte = sMD_spring_cte
        self.sMD_ligand_anchor_mode = sMD_ligand_anchor_mode
        self.sMD_autostop_freq = sMD_autostop_freq
        self.sMD_run_analysis = sMD_run_analysis
        self.sMD_clust_selection = sMD_clust_selection
        # Milestones
        self.extract_milestones = extract_milestones
        self.milestone_mode = milestone_mode
        self.milestone_min_frame_separation = milestone_min_frame_separation
        self.n_milestones = n_milestones
        self.relax_steps = relax_steps
        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_use_funnel_potential = mMD_use_funnel_potential
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

    def run(self, ligand_file: str = None, ligand_resname: str = "UNK"):

        start_time = time.monotonic()

        if ligand_file is not None:
            sys_name = os.path.splitext(os.path.basename(ligand_file))[0]
        else:
            sys_name = os.path.splitext(os.path.basename(self.protein_file))[0]

        self.sMD_outdir = f"{sys_name}/{self.sMD_outdir}"

        os.makedirs(sys_name, exist_ok=True)
        
        logger.info(f"Processing system {sys_name}")

        ##############################################################################################
        ####################################### System preparation ###################################
        ##############################################################################################

        prmtop_file = f"{sys_name}/system.prmtop"
        solvated_system_pdb = f"{sys_name}/system.pdb"

        if self.run_preparation:
            prepare_system = SystemPreparation(
                out_dir=sys_name,
                forcefield=self.forcefield,
                hydrogenMass=self.hydrogenMass,
                lig_ff=self.lig_ff,
                boxShape=self.boxShape,
                padding=self.padding,
                ionicStrength=self.ionicStrength,
                ions=self.ions,
                is_membrane=self.is_membrane,
                lipid_type=self.lipid_type,
            )
            system, topology = prepare_system.run(self.protein_file, self.variants, ligand_file)

        system = load_system(f"{sys_name}/system.xml")
        topology = PDBFile(solvated_system_pdb).topology

        # prmtop_file = f"{sys_name}/system.pdb" 
        # try:
        #     topology = AmberPrmtopFile(prmtop_file).topology
        # except Exception as e:
        #     logger.error(f"Error loading topology from {prmtop_file}: {e}")

        ##############################################################################################
        ##################################### System equilibration ###################################
        ##############################################################################################

        equilibrated_traj = f"{sys_name}/equilibration/equilibration_{sys_name}.dcd"
        equilibrated_chk = f"{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk"
        equilibrated_pdb = f"{sys_name}/equilibration/{sys_name}_equilibrated.pdb"
        equilibrated_system = f"{sys_name}/equilibration/system_equil_{sys_name}.xml"

        if self.run_equilibration:
            equilibration = Equilibration(
                                topology=topology,
                                system=system,
                                out_dir=f"{sys_name}/equilibration",
                                restrained_minimization=False,
                                is_membrane=self.is_membrane,
                                protocol_fname=self.protocol_fname,
                                platform=self.platform
                                )
            
            equilibrated_system = equilibration.run(solvated_system_pdb, run_id=sys_name)
        
            #Wrap, align and save the clean trajectory
            traj = md.load(equilibrated_traj, top=solvated_system_pdb)
            traj = traj.center_coordinates()
            traj = traj.image_molecules()
            try: # if there's no protein
                backbone = traj.topology.select("backbone")
                traj = traj.superpose(traj[0], atom_indices=backbone)
            except Exception as e:
                logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
            traj.save(equilibrated_traj.replace(".dcd", "_aligned.dcd"))
            os.remove(equilibrated_traj)

        ##############################################################################################
        ############################# Post-equilibration Analysis ####################################
        ##############################################################################################
        
        equilibrated_traj = equilibrated_traj.replace(".dcd", "_aligned.dcd")
        if os.path.exists(equilibrated_traj):
            u_eq = mda.Universe(equilibrated_pdb, equilibrated_traj, in_memory=True)
            try:
                rmsd = compute_rmsd(u_eq, u_eq,
                                    alig_select="backbone", 
                                    groupselections={"ligand":f"resname {ligand_resname} and not name H*", 
                                                    "protein":'protein and not name H*'},
                                    plots_outdir=f"{sys_name}/equilibration"
                                    )
                rmsd.to_csv(f"{sys_name}/equilibration/{sys_name}_rmsd.csv", index=False)
                plot_atomic_rmsf(u_eq, outname=f"{sys_name}/equilibration/{sys_name}_RMSF.png", log_rmsf=True)
            except Exception as e:
                logger.error(f"Error computing RMSD/RMSF: {e}")
                pass
            
            # Equilibration VS checkpoint
            # if self.equilibration_checkpoint:
            #     final_rmsd = lig_rmsd_equilibration[-1:].values
            #     if final_rmsd > self.eq_checkpoint_cutoff * 10:  # to Angs
            #         logger.warning(f"Simulation for ligand {sys_name} terminated because ligand RMSD={final_rmsd:.2f} > {self.eq_checkpoint_cutoff}")
            #         exit(1)
            # ligand_atoms = u_eq.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')
            # final_com = calculate_com_distance(u_eq, ligand_atoms, pocket_atoms, wrap=False)[-1] /10 # convert to nm
            # logger.info(f"COM distance after equilibration is: {final_com:.2f} nm")
            
            pocket_atom_indices = get_pocket_atoms_idxs(u_eq, self.pocket_selection)
            pocket_atoms = u_eq.select_atoms(f'index {" ".join(map(str, pocket_atom_indices))}')
            pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
            logger.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")
            
            ligand_total_hatoms = [a.index for a in u_eq.select_atoms(f'resname {ligand_resname} and not name H*')]
            ligand_atoms_indices = get_ligand_anchor_atoms(u_eq, ligand_resname, 
                                                        mode=self.sMD_ligand_anchor_mode,
                                                        n_atoms=5,
                                                        out_dir=sys_name)
            logger.info(f"Ligand anchor atom indices are: {', '.join(map(str, ligand_atoms_indices))}")
            # write out the protein/ligand/pocket PDBs and PyMOL session
            try:
                write_pocket_pymol_pml(
                    u=u_eq,
                    out_dir=sys_name,
                    protein_selection="protein",
                    ligand_selection=f"resname {ligand_resname}",
                    pocket_selection=self.pocket_selection,
                )
            except Exception as e:
                logger.error(f"Error writing pocket/ligand/protein pdbs: {e}")
                pass

        ##############################################################################################
        ##################################### Steered MD simulations #################################
        ##############################################################################################
        
        # sMD_collision_frequency = 1  # ps^-1
        # sMD_outdir = f"{sys_name}/sMD_{lig_anchor_mode}_{sMD_timestep}ps_{sMD_collision_frequency}ps_200stm"
        # in this paper they used 80 kcal·mol−1? units don match tho. Ziada et al 2022.
        sMD_spring_cte_per_atom = 100 * 4.184  # KJ/mol/nm2, converted from kcal. This affects thermal fluctuations
        sMD_traj_outdir = f"{self.sMD_outdir}/trajectories"
        sMD_analysis_outdir = f"{self.sMD_outdir}/analysis"
        
        if self.run_sMDpulling:
            equilibrated_system = load_system(f"{sys_name}/equilibration/system_equil_{sys_name}.xml")

            if self.sMD_spring_cte is None:
                sMD_spring_cte = sMD_spring_cte_per_atom * len(ligand_total_hatoms)  # Normalize by ligand size
                logger.info(f"Spring constant set to {sMD_spring_cte} KJ/mol/nm2 for {len(ligand_atoms_indices)} atoms.")
            else:
                sMD_spring_cte = self.sMD_spring_cte

            sMD = SteeredMD(
                system=equilibrated_system,
                topology=topology,
                groupA_atoms=ligand_atoms_indices,
                groupB_atoms=pocket_atom_indices,
                restart_velocities=True,
                autostop_freq=self.sMD_autostop_freq,
                timestep=self.timestep,
                temperature=self.temperature,
                out_dir=sMD_traj_outdir,
                platform=self.platform
            )

            for speed, reps in self.sMD_pulling_speeds.items():
                if reps is not None:
                    logger.info(f"Running sMD for speed {speed} nm/ps with {reps} replicas.")
                    for i in range(reps):
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                pulling_speed=speed,  # nm/ps
                                dx_per_move=self.sMD_dx_per_move,  # nm, this is the displacement per move
                                sMD_spring_cte=sMD_spring_cte,
                                pulling_direction=self.sMD_pulling_dir,
                                save_freq=None, # will use steps_per_move
                            )
                        except Exception as e:  
                            logger.error(f"Error during sMD pulling for speed {speed} nm/ps, replica {i+1}: {e}")
                            continue
                else:
                    logger.info(f"Running sMD for speed {speed} nm/ps until convergence.")
                    CONVERGED = False
                    while not CONVERGED:
                        log_files = glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
                        current_replica = len(log_files) + 1
                        logger.info(f"Starting replica {current_replica} for speed {speed} nm/ps.")
                        
                        # # cap the number of replicas to avoid infinite loops
                        # if current_replica > 50:
                        #     logger.warning(f"Reached maximum number of replicas (50) for speed {speed} nm/ps without convergence. Stopping.")
                        #     break
                        
                        if len(log_files) >= 5:  # need at least 5 replicas to assess convergence
                            
                            # loads the sMD data
                            smdanalysis = SMDAnalysis(sysname=sys_name, path_model='dtw', estimators=['cumulant'],
                                                    do_plots=False, seed=self.random_state,
                                                    temperature=self.temperature,
                                                    outdir=sMD_analysis_outdir,
                                                    ligand_select=f"resname {ligand_resname} and not name H*",
                                                    )
                            
                            # check convergence for this speed
                            conv_df, traces_df = smdanalysis.check_convergence(
                                logs=log_files, speeds=[speed],
                                # group_A=f"resname {ligand_resname} and not name H*",
                                # group_B=self.sMD_clust_selection
                            )
                            
                            conv_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_v{speed}_metrics.csv", index=False)
                            traces_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_v{speed}_traces.csv", index=False)

                            # Plot convergence results
                            conv_traces = glob(f"{sMD_analysis_outdir}/sMD_conv_*_traces.csv")
                            conv_metrics = glob(f"{sMD_analysis_outdir}/sMD_conv_*_metrics.csv")

                            plot_convergence_traces(conv_traces, outdir=sMD_analysis_outdir)
                            plot_convergence_metrics(conv_metrics, outdir=sMD_analysis_outdir)
                            
                            # Check convergence. Two last replicas must be converged
                            CONVERGED = conv_df['converged'].iloc[-2] and conv_df['converged'].iloc[-1]
                            if CONVERGED:
                                logger.warning(f"sMD pulling for speed {speed} nm/ps CONVERGED after {current_replica} replicas.")
                                continue # continue to next speed

                        # Run the next replica if not converged
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                pulling_speed=speed,  # nm/ps
                                dx_per_move=self.sMD_dx_per_move,  # nm, this is the displacement per move
                                sMD_spring_cte=sMD_spring_cte,
                                pulling_direction=self.sMD_pulling_dir,
                            )
                        except Exception as e:
                            logger.error(f"Error during sMD pulling for speed {speed} nm/ps, replica {current_replica}: {e}")
                            continue
                        
        # Load and align sMD trajectories
        sMD_trajs = glob(f"{sMD_traj_outdir}/sMD_replica-*_*_*.dcd")
        sMD_trajs = [f for f in sMD_trajs if "aligned" not in f]  # only process unaligned trajectories
        logger.info(f"Found {len(sMD_trajs)} sMD trajectories to align.")        
        for traj_file in sMD_trajs:
            traj = md.load(traj_file, top=solvated_system_pdb)
            traj = traj.center_coordinates()
            traj = traj.image_molecules()
            try:
                backbone = traj.topology.select("backbone")
                traj = traj.superpose(traj[0], atom_indices=backbone)
            except Exception as e:
                logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
            traj.save(traj_file.replace(".dcd", "_aligned.dcd")) #overwrite
            os.remove(traj_file) #remove original        
                                
        ##############################################################################################
        ######################################### sMD Analysis #######################################
        ##############################################################################################        
        
        sMD_trajs = glob(f"{sMD_traj_outdir}/sMD_replica-*_*_*_aligned.dcd")

        if self.sMD_run_analysis:
            
            if self.sMD_clust_selection is not None:
                # check that the selection is valid
                u_clust = mda.Universe(equilibrated_pdb, equilibrated_pdb)
                pocket_atoms = u_clust.select_atoms(self.sMD_clust_selection)
                pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
                logger.info(f"Auto-selected clustering selection: {', '.join(set(pocket_residues))}")
            
            # loads the sMD data
            logs = glob(f"{sMD_traj_outdir}/sMD_*_*_{self.sMD_pulling_dir}.dat")
            logger.info(f"Found {len(logs)} sMD logs for analysis.")
            smd_data = SMDData(logs, sys_name, temperature=self.temperature,
                               reference_pdb=equilibrated_pdb)
            
            # cluster trajectories into pathways
            cluster_model = DTWPathModel(seed=self.random_state, do_plots=True,
                                        outdir=sMD_analysis_outdir)

            smdanalysis = SMDAnalysis(sys_name, cluster_model,
                                    estimators=['cumulant', 'jarzynski'],
                                    do_plots=True, seed=self.random_state,
                                    temperature=self.temperature,
                                    ligand_select=f"resname {ligand_resname} and not name H*",
                                    outdir=sMD_analysis_outdir,
                                    )

            smd_data = smdanalysis.run(smd_data,
                                       group_A=f"resname {ligand_resname} and not name H*",
                                       group_B=self.sMD_clust_selection,
                                       merge_features=True,
                                    #    r_range=(0, 1.75)
                                       )

            # Store for downstream milestone extraction
            self._smdanalysis = smdanalysis

            # check convergence regardless of speed and autopstop
            conv_df, traces_df = smdanalysis.check_convergence(logs=logs,
                group_A=f"resname {ligand_resname} and not name H*",
                group_B=self.sMD_clust_selection
            )
            conv_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_vALL_metrics.csv", index=False)
            traces_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_vALL_traces.csv", index=False)

            smd_conv_traces = glob(f"{sMD_analysis_outdir}/sMD_conv_vALL_traces.csv")
            plot_convergence_traces(smd_conv_traces, outdir=sMD_analysis_outdir)
            smd_conv_metrics = glob(f"{sMD_analysis_outdir}/sMD_conv_vALL_metrics.csv")
            plot_convergence_metrics(smd_conv_metrics, outdir=sMD_analysis_outdir)

        ##############################################################################################
        ###################################### Extract Milestones ####################################
        ##############################################################################################

        milestones_outdir = f"{sys_name}/milestones"
        min_dist = 1.0 # minimum distance between clusters of milestones
        
        # Resolve ligand_atoms_full_indices for downstream use (metadynamics, relax)
        u_sMD = mda.Universe(solvated_system_pdb, sMD_trajs)
        ligand_atoms_full = u_sMD.select_atoms(f'resname {ligand_resname} and not name H*')
        ligand_atoms_full_indices = [atom.index for atom in ligand_atoms_full]

        ligand_sel = f"resname {ligand_resname} and not name H*"
        pocket_sel = f'index {" ".join(map(str, pocket_atom_indices))}'

        if self.extract_milestones:
            
            os.makedirs(milestones_outdir, exist_ok=True)

            logger.info(f"Found {len(sMD_trajs)} sMD trajectories for milestone extraction.")

            if len(sMD_trajs) == 0:
                logger.error("No sMD trajectories found. Please check the sMD pulling step.")
                exit(1)

            # Load medoid info (from analysis or from saved file)
            medoid_info_path = f"{sMD_analysis_outdir}/medoid_info.json"
            medoid_to_path = {}
            medoid_names = []

            if hasattr(self, '_smdanalysis') and hasattr(self._smdanalysis.path_model, 'all_medoid_names'):
                medoid_names = self._smdanalysis.path_model.all_medoid_names
                medoid_to_path = self._smdanalysis.path_model.medoid_to_path
            elif os.path.exists(medoid_info_path):
                with open(medoid_info_path, "r") as f:
                    medoid_info = json.load(f)
                medoid_names = medoid_info.get("medoid_names", [])
                medoid_to_path = medoid_info.get("medoid_to_path", {})
                logger.info(f"Loaded medoid info from {medoid_info_path}: {len(medoid_names)} medoids.")
            else:
                logger.warning(
                    "No medoid info found (run sMD analysis first, or provide medoid_info.json). "
                    "Falling back to using all sMD trajectories."
                )

            if medoid_names:
                medoid_dcds = [self._trajname_to_dcd(name) for name in medoid_names]
                medoid_dcds = [f for f in medoid_dcds if os.path.exists(f)]
                if not medoid_dcds:
                    logger.warning("Could not locate medoid DCD files. Falling back to all trajectories.")
                    medoid_dcds = sMD_trajs
                    medoid_to_path = {}
            else:
                medoid_dcds = sMD_trajs
                medoid_to_path = {}

            logger.info(f"Using {len(medoid_dcds)} trajectories for milestone extraction (mode={self.milestone_mode}).")
            
            u_milestone = mda.Universe(solvated_system_pdb, medoid_dcds)

            if self.milestone_mode == "per_path":
                # Mode: extract milestones from each path's medoid independently
                all_milestone_files = []
                
                if not medoid_to_path:
                    logger.warning(
                        "No path information available for per_path mode. "
                        "Falling back to all_medoids mode."
                    )
                    self.milestone_mode = "all_medoids"
                else:
                    # Group medoids by path
                    path_to_medoids = defaultdict(list)
                    for mname, pid in medoid_to_path.items():
                        path_to_medoids[pid].append(mname)

                    for path_id, med_names in sorted(path_to_medoids.items()):
                        path_dcds = [self._trajname_to_dcd(n) for n in med_names]
                        path_dcds = [f for f in path_dcds if os.path.exists(f)]
                        if not path_dcds:
                            logger.warning(f"No DCD files found for path {path_id}. Skipping.")
                            continue
                        
                        path_outdir = os.path.join(milestones_outdir, str(path_id))
                        u_path = mda.Universe(solvated_system_pdb, path_dcds)
                        
                        logger.info(f"Computing distance features for path {path_id} ({len(path_dcds)} trajs, {len(u_path.trajectory)} frames)...")
                        X_path = compute_distance_features(u_path, ligand_sel, pocket_sel)

                        labels, centers, ms_files = extract_milestones(
                            u_path, X_path,
                            n_milestones=self.n_milestones,
                            min_dist=min_dist,
                            out_dir=path_outdir,
                            prefix=f"milestone_{path_id}",
                            min_frame_separation=self.milestone_min_frame_separation,
                        )
                        all_milestone_files.extend(ms_files)

                        logger.info(f"Path {path_id}: extracted {len(ms_files)} milestones.")

            if self.milestone_mode == "all_medoids":
                # ---- Mode: pool all medoid trajectories, cluster together ----
                logger.info(f"Computing distance features for all medoids ({len(u_milestone.trajectory)} frames)...")
                X_all = compute_distance_features(u_milestone, ligand_sel, pocket_sel)

                labels, sorted_cluster_centers, milestone_files = extract_milestones(
                    u_milestone, X_all,
                    n_milestones=self.n_milestones,
                    min_dist=min_dist,
                    out_dir=milestones_outdir,
                    prefix="milestone",
                    min_frame_separation=self.milestone_min_frame_separation,
                )

                logger.info(f"Extracted {len(milestone_files)} milestones from {len(medoid_dcds)} medoid trajectories.")

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
            min_com = 0.0
            max_com = 3.0
            try:
                smd_raw = pd.read_csv(f'{sMD_analysis_outdir}/sMD_processed_data.csv')
                min_com = smd_raw['r_before'].min() * 0.75  # nm
                max_com = smd_raw['r_before'].max() * 1.1 # nm
            except Exception as e:
                logger.error(f"Error loading sMD raw data: {e}")

            logger.info(f"Using COM distance range for metadynamics: [{min_com}, {max_com}] nm")

            milestones = glob(f'{milestones_outdir}/milestone_*_*_*.pdb') + \
                        glob(f'{milestones_outdir}/**/milestone_*_*_*.pdb', recursive=True)
            milestones = list(set(milestones))  # deduplicate
            if len(milestones) == 0:
                logger.error("No milestones found. Please check the milestone extraction step.")
                exit(1)

            #sort the milestones by their index (milestone number is the last token before 'frame')
            milestones.sort(key=lambda x: int(os.path.basename(x).split('_frame_')[0].rsplit('_', 1)[-1]))

            milestone_relax = RelaxMD(
                topology=topology,
                ligand_atoms=ligand_atoms_full_indices, # use all atoms
                pocket_atoms=pocket_atom_indices,
                out_dir=milestones_outdir,
                is_membrane=self.is_membrane,
                temp=self.temperature,
                timestep=self.timestep,
                platform=self.platform
            )

            WTMetaD = MetadynamicsMD(
                topology=topology,
                ligand_atoms=ligand_atoms_indices,
                pocket_atoms=pocket_atom_indices,
                restrained_atoms=None,
                is_membrane=self.is_membrane,
                timestep=self.timestep,
                temp=self.temperature,
                out_dir=f"{sys_name}/metadynamics",
                platform=self.platform
            )

            # Generate funnel potential if requested
            funnel_force = None
            if self.mMD_use_funnel_potential:
                logger.info("Generating funnel potential from sMD trajectories...")
                try:
                    # Collect all sMD trajectories
                    smd_trajs = glob(f"{sMD_traj_outdir}/*.dcd")
                    
                    if not smd_trajs:
                        logger.warning("No sMD trajectories found. Skipping funnel potential generation.")
                    else:                       
                        # Create MDAnalysis Universe with all trajectories
                        universe = mda.Universe(solvated_system_pdb, smd_trajs)
                        logger.info(f"Loaded {len(universe.trajectory)} frames total")
                        
                        # Generate funnel parameters from trajectories
                        funnel_params = generate_funnel_parameters_from_trajectory(
                            universe,
                            host_selection="protein",
                            guest_selection="resname UNK",
                            use_pca=True,
                            percentile_z=95.0,
                            percentile_r_cyl=90.0,
                            percentile_r_funnel=85.0,
                            alpha_cone_degrees=35.0,
                            verbose=True,
                        )
                        
                        # Create funnel force from parameters
                        funnel_force = create_funnel_force_from_trajectory_analysis(funnel_params)
                        logger.info("Funnel potential successfully generated from trajectories")
                except Exception as e:
                    logger.error(f"Error generating funnel potential: {e}")
                    logger.warning("Continuing without funnel potential")
                    funnel_force = None

            for milestone in milestones:
                milestone_name = os.path.basename(milestone).split('.')[0]
                # milestone_number = int(milestone_name.split('_')[-3])
                milestone_system = f"{milestones_outdir}/{milestone_name}_relax_system.xml"
                milestone_chk = f"{milestones_outdir}/{milestone_name}_relax_checkpoint.chk"

                if os.path.exists(milestone_system):
                    milestone_system = load_system(milestone_system)
                    logger.info(f"Loading relaxed milestone {milestone_name}")
                else:
                    logger.info(f'Relaxing milestone {milestone_name}')
                    try:
                        system = load_system(f"{sys_name}/system.xml")
                        milestone_system = milestone_relax.run(system=system, pdb_file=milestone, run_id=milestone_name)
                    except Exception as e:
                        logger.error(f"Error relaxing {milestone_name}: {e}")
                        continue
                
                logger.info(f"Running WTMetaD for milestone {milestone_name}")
                try:
                    WTMetaD.run(
                        checkpoint_file=milestone_chk,
                        system=milestone_system,
                        run_id=milestone_name,
                        mMD_CV='com',
                        mMD_time=self.mMD_time, #ns
                        bias_factor=self.mMD_bias_factor,
                        # hill_height=biasing_scheme[milestone_number]['height'] if use_biasing_scheme else self.mMD_hill_height, #kcal/mol
                        # hill_width=biasing_scheme[milestone_number]['width'] if use_biasing_scheme else self.mMD_hill_width, #nm
                        biasFrequency=self.mMD_bias_frequency, #ps
                        grid_dimensions=(min_com, max_com),
                        funnel_force=funnel_force
                    )
                except Exception as e:
                    logger.error(f"Error during WTMetaD for {milestone_name}: {e}")
                    continue
                 
        # Load and align the WTMetaD trajectories
        WTMetaD_trajs = glob(f"{sys_name}/metadynamics/trajectory_metadynamics_milestone_*_frame_*.dcd")
        WTMetaD_trajs = [f for f in WTMetaD_trajs if "aligned" not in f]  # only process unaligned trajectories

        for traj_file in WTMetaD_trajs:
            traj = md.load(traj_file, top=solvated_system_pdb)
            traj = traj.center_coordinates()
            traj = traj.image_molecules()
            try: # if there's no protein
                backbone = traj.topology.select("backbone")
                traj = traj.superpose(traj[0], atom_indices=backbone)
            except Exception as e:
                logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
            traj.save(traj_file.replace(".dcd", "_aligned.dcd")) #overwrite
            os.remove(traj_file) #remove original

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished AutoPath simulation in {simulation_time/60:.2f} min.")

        return
    
    @staticmethod
    def _replica_idx_from_log(fn):
        base = os.path.basename(fn)[:-4]
        rep = base.split("_")[-3]
        return int(rep.split("-")[1])
    
    
    # Map medoid names to DCD trajectory files
    def _trajname_to_dcd(self, trajname: str) -> str:
        """Convert a trajname (log basename without ext) to the corresponding .dcd path."""
        dcd_name = trajname.replace("log", "traj") + "_aligned.dcd"
        return os.path.join(self.sMD_outdir, "trajectories", dcd_name)