# General imports
import os
import json
import time
import numpy as np
import pandas as pd
import warnings
import shutil
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
from autopath.pdb_preprocessor import PDBPreprocessor
from autopath.ap_PLIP import plot_atomic_property, calculate_ligand_rmsf, calculate_contact_frequency
from autopath import (
    SystemPreparation,
    Equilibration,
    SteeredMD,
)
from autopath.pulling import SMDData, SMDAnalysis
from autopath.pulling.Convergence import (tail_converged, validate_autostop_options,
                                          round_robin_order)
from autopath.pulling.PathModel import DTWPathModel, NullPathModel
from autopath.pulling.Estimators import validate_path_combination
from autopath.pulling.Diagnostics import plot_convergence_traces, plot_convergence_metrics

import logging
logger = logging.getLogger('autopath.core')

class AutoPath:
    """Main orchestrator for the AutoPath binding-free-energy pipeline.

    AutoPath runs a four-stage protocol that takes a protein-ligand PDB as input
    and produces sMD unbinding paths and free-energy profiles:

    1. **System preparation** — parametrises the protein + ligand, solvates,
       adds ions, and writes ``system.xml`` / ``system.pdb`` via
       :class:`~autopath.preparation.SystemPreparation`.
    2. **Equilibration** — multi-stage NPT/NVT equilibration via
       :class:`~autopath.equilibration.Equilibration`.  Saves a checkpoint and
       an equilibrated PDB used by all downstream stages.
    3. **Steered MD (sMD)** — pulls the ligand out of the pocket along the COM
       distance at multiple speeds via :class:`~autopath.pulling.SteeredMD`.
       Replicas can be run until convergence (``sMD_converge_speeds=True``) or
       to a fixed count.  A hard cap (``sMD_max_replicas``) prevents infinite
       loops.
    4. **Path clustering & PMF estimation** — :class:`~autopath.pulling.SMDAnalysis`
       clusters trajectories with DTW-based path models, estimates the PMF via
       cumulant and Jarzynski estimators, and identifies medoid trajectories.

    Subcomponents
    -------------
    :class:`~autopath.preparation.SystemPreparation`,
    :class:`~autopath.equilibration.Equilibration`,
    :class:`~autopath.pulling.SteeredMD`,
    :class:`~autopath.pulling.SMDAnalysis`

    Parameter groups
    ----------------
    General
        ``VS_mode``, ``pdb_path``, ``do_fix_pdb``, ``pocket_selection``,
        ``temperature``, ``random_state``, ``platform``
    System preparation
        ``run_preparation``, ``forcefield``, ``hydrogenMass``, ``timestep``,
        ``lig_ff``, ``boxShape``, ``padding``, ``ionicStrength``, ``ions``,
        ``variants``, ``is_membrane``, ``lipid_type``
    Equilibration
        ``run_equilibration``, ``protocol_fname``
    Steered MD
        ``run_sMDpulling``, ``sMD_outdir``, ``sMD_pulling_dir``,
        ``sMD_pulling_speeds``, ``sMD_max_pulling_dist``, ``sMD_max_r_offset``,
        ``sMD_autostop_nc``, ``sMD_autostop_nc_window``,
        ``sMD_autostop_min_displacement``, ``sMD_converge_speeds``,
        ``sMD_time``, ``sMD_steps_per_move``, ``sMD_dx_per_move``,
        ``sMD_spring_cte``, ``sMD_ligand_anchor_mode``, ``sMD_max_replicas``,
        ``sMD_run_analysis``, ``sMD_align_trajs``, ``sMD_clust_selection``, ``sMD_features``,
        ``sMD_log_geom_features``, ``sMD_plateau_frac``,
        ``sMD_cluster_to_boundary``, ``sMD_boundary_buffer_frac``,
        ``sMD_kramers_abs``, ``sMD_kramers_bound``,
        ``cluster_across_speeds``, ``sMD_min_replicas_per_path``
    """

    def __init__(
        self,
        VS_mode: bool = False,
        pdb_path: str = None,
        do_fix_pdb: bool = True,
        pocket_selection: str | list = None,
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
        lig_ff: str = "openff-2.3.0",
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        ionicStrength: float = 0.15,
        ions: tuple = ('Na+', 'Cl-'),
        variants: dict = None,
        is_membrane: bool = False,
        lipid_type: str = None,
        run_equilibration: bool = True,
        protocol_fname: str = None,
        run_sMDpulling: bool = True,
        sMD_outdir: str = "sMD",
        sMD_pulling_dir: str = "forward",  # "forward" or "backward"
        sMD_pulling_speeds: dict = {0.005:5, 0.0025:5, 0.001:5},  # nm/ps; value = min reps (floor) if sMD_converge_speeds, else total reps
        sMD_max_pulling_dist: float = 3.5,  # nm
        sMD_max_r_offset: float = 3.0,    # max displacement offset (nm): cap pull at r0 + offset nm (also capped at half-box - 0.5 nm)
        sMD_autostop_nc: float | None = 0.01,  # fraction of NC_initial; None disables
        sMD_autostop_nc_window: int = 5,
        sMD_autostop_min_displacement: float = 0.5,
        sMD_converge_speeds: bool = True,
        sMD_conv_window: int = 5,
        sMD_conv_streak: int = 3,
        sMD_autostop_estimator: str = "cumulant",
        sMD_alternate_speeds: bool = False,
        sMD_time: int = None,  # ns
        sMD_steps_per_move: int = None,
        sMD_dx_per_move: float = 0.001,  # nm, this is the displacement per move
        sMD_spring_cte: float = None,  # KJ/mol/nm2
        sMD_force_n_samples: int = 10,  # cap on restraint-force samples time-averaged per move (>=1); 1 = legacy single pre-step sample
        sMD_force_sample_stride: int = 5,  # MD steps between force samples; samples-per-move adapts to move length (capped by sMD_force_n_samples)
        sMD_ligand_anchor_mode: str = 'murcko',
        sMD_max_replicas: int = 50,  # max replicas per speed in convergence mode
        sMD_run_analysis: bool = True,
        sMD_align_trajs: bool = True,  # False: keep raw sMD DCDs (<stem>.dcd), skip wrap/align
        sMD_clust_selection:str = None,
        sMD_path_model: str = 'dtw',   # 'dtw' | 'null' ('null' = no clustering: one path)
        sMD_n_paths: int | None = None,  # fixed number of paths; None = silhouette-selected
        sMD_max_frac_neg_dG_first_half: float = 0.25,  # pass-3 binding-well filter; <=0 disables
        sMD_path_combination: str = 'eq21',  # 'eq21' (Wolf, Post & Stock 2023, Eq. 21) | 'linear' (p_eq mean)
        sMD_features: list | None = None,
        sMD_log_geom_features: bool | list = True,  # hybrid: log geom during pulling + merge into clustering
        sMD_plateau_frac: float = 0.4,  # force-plateau TS boundary: fraction of peak |force| (higher -> boundary nearer rupture, off the tail)
        sMD_cluster_to_boundary: bool = True,  # cluster + RMSD-converge only up to the force-plateau boundary (exclude bulk-solvent tail)
        sMD_boundary_buffer_frac: float = 0.1,  # extend the boundary cap by this fraction of r_ts (a little buffer past rupture)
        sMD_kramers_abs: str = "end",  # Kramers absorbing boundary: 'end' (pulled range) | 'force_plateau' (legacy)
        sMD_kramers_bound: str = "mirror",  # bound population below pull start: 'mirror' | 'none'
        cluster_across_speeds: bool = True,
        sMD_min_replicas_per_path: int = 5,
    ):
        # General
        self.pocket_selection = pocket_selection
        self.original_pdb_path = pdb_path
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
        self._sMD_outdir_base = sMD_outdir   # run() prefixes the ligand name per call
        self.sMD_outdir = sMD_outdir
        self.sMD_pulling_dir = sMD_pulling_dir
        self.sMD_max_pulling_dist = sMD_max_pulling_dist
        self.sMD_max_r_offset = sMD_max_r_offset
        self.sMD_converge_speeds = sMD_converge_speeds
        self.sMD_conv_window = sMD_conv_window
        self.sMD_conv_streak = sMD_conv_streak
        self.sMD_autostop_nc = sMD_autostop_nc
        self.sMD_autostop_nc_window = sMD_autostop_nc_window
        self.sMD_autostop_min_displacement = sMD_autostop_min_displacement
        for _name, _val in (("sMD_time", sMD_time), ("sMD_steps_per_move", sMD_steps_per_move)):
            if _val is not None:
                raise ValueError(f"{_name} is not supported; set sMD_pulling_speeds and sMD_dx_per_move instead.")
        self.sMD_time = sMD_time
        self.sMD_pulling_speeds = sMD_pulling_speeds
        self.sMD_autostop_estimator = sMD_autostop_estimator
        self.sMD_alternate_speeds = sMD_alternate_speeds
        validate_autostop_options(sMD_autostop_estimator, sMD_alternate_speeds,
                                  list(sMD_pulling_speeds.keys()),
                                  conv_window=sMD_conv_window,
                                  conv_streak=sMD_conv_streak)
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_dx_per_move = sMD_dx_per_move
        self.sMD_spring_cte = sMD_spring_cte
        self.sMD_force_n_samples = sMD_force_n_samples
        self.sMD_force_sample_stride = sMD_force_sample_stride
        self.sMD_ligand_anchor_mode = sMD_ligand_anchor_mode
        self.sMD_max_replicas = sMD_max_replicas
        self.sMD_run_analysis = sMD_run_analysis
        self.sMD_align_trajs = sMD_align_trajs
        self.sMD_clust_selection = sMD_clust_selection
        self.sMD_path_model = str(sMD_path_model).lower()
        self.sMD_n_paths = sMD_n_paths
        self.sMD_max_frac_neg_dG_first_half = float(sMD_max_frac_neg_dG_first_half)
        self.sMD_path_combination = validate_path_combination(sMD_path_combination)
        if self.sMD_path_model not in ('dtw', 'null'):
            raise ValueError(f"sMD_path_model must be 'dtw' or 'null', got {sMD_path_model!r}")
        self.sMD_features = sMD_features
        self.sMD_log_geom_features = sMD_log_geom_features
        self.sMD_plateau_frac = sMD_plateau_frac
        self.sMD_cluster_to_boundary = sMD_cluster_to_boundary
        self.sMD_boundary_buffer_frac = sMD_boundary_buffer_frac
        self.sMD_kramers_abs = sMD_kramers_abs
        self.sMD_kramers_bound = sMD_kramers_bound
        self.cluster_across_speeds = cluster_across_speeds
        self.sMD_min_replicas_per_path = sMD_min_replicas_per_path
        # VS mode
        if VS_mode:
            raise ValueError("VS_mode is not supported.")

        # Process the input PDB
        if do_fix_pdb:
            protein_pdb = PDBPreprocessor(pdb_path).fix(
                                  cap_termini=True,
                                  keep_heterogens=True, 
                                  pH=7.4)
            root, ext = os.path.splitext(pdb_path)
            self.protein_file = f"{root}_fixed{ext}"
            # Parallel jobs share the receptor: write privately, then swap atomically
            tmp = f"{root}_fixed.{os.getpid()}.tmp{ext}"
            save_pdb(protein_pdb.topology, protein_pdb.positions, tmp)
            os.replace(tmp, self.protein_file)
        else:
            self.protein_file = pdb_path

    def run(self, ligand_file: str = None, ligand_selection: str = "resname UNK"):
        """Execute the full AutoPath pipeline for one ligand.

        Stages are run in order: system preparation → equilibration →
        steered MD → path clustering.
        Each stage can be skipped by setting the corresponding ``run_*`` flag
        to ``False`` in the constructor; the stage will read outputs written by
        a previous run instead.

        Parameters
        ----------
        ligand_file : str, optional
            Path to the ligand SDF file.  The stem of this filename is used as
            the system name (output subdirectory).  If ``None``, the protein PDB
            stem is used instead.
        ligand_selection : str, optional
            MDAnalysis / OpenMM selection string identifying the ligand residue.
            Default ``"resname UNK"``.

        Returns
        -------
        None
            All results are written to disk under the system-name subdirectory.
        """
        start_time = time.monotonic()

        if ligand_file is not None:
            sys_name = os.path.splitext(os.path.basename(ligand_file))[0]
        else:
            sys_name = os.path.splitext(os.path.basename(self.protein_file))[0]

        self.sMD_outdir = f"{sys_name}/{self._sMD_outdir_base}"

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

        if isinstance(self.pocket_selection, list):
            mapping_path = f"{sys_name}/residue_mapping.json"
            if self.run_preparation or not os.path.exists(mapping_path):
                mapping = build_residue_mapping(self.original_pdb_path, solvated_system_pdb)
                with open(mapping_path, "w") as _f:
                    json.dump({str(k): v for k, v in mapping.items()}, _f, indent=2)
                logger.info(f"Residue mapping written to {mapping_path}")
            else:
                with open(mapping_path) as _f:
                    mapping = {int(k): v for k, v in json.load(_f).items()}
                logger.info(f"Loaded existing residue mapping from {mapping_path}")

            translated, missing = [], []
            for r in self.pocket_selection:
                (translated if r in mapping else missing).append(r)

            if missing:
                logger.warning(
                    f"Pocket residues absent from original PDB (crystallographic gaps?): {missing}"
                )
            self.pocket_selection = f'resid {" ".join(map(str, [mapping[r] for r in translated]))} and name CA'
            logger.info(f"pocket_selection (system numbering): {self.pocket_selection}")

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
        equilibrated_state_xml = f"{sys_name}/equilibration/checkpoint_equil_{sys_name}.xml"
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
            wrap_align_save_traj(equilibrated_traj, solvated_system_pdb, is_membrane=self.is_membrane)

        ##############################################################################################
        ############################# Post-equilibration Analysis ####################################
        ##############################################################################################
        
        _lig_sel_ha = f"({ligand_selection}) and not name H*"

        equilibrated_traj = equilibrated_traj.replace(".dcd", "_aligned.dcd")
        if self.run_sMDpulling and not os.path.exists(equilibrated_traj):
            raise FileNotFoundError(
                f"{equilibrated_traj} is missing; rerun equilibration (run_equilibration=True).")
        if os.path.exists(equilibrated_traj):
            u_eq = mda.Universe(equilibrated_pdb, equilibrated_traj, in_memory=True)
            is_peptide = u_eq.select_atoms(ligand_selection).n_residues > 1
            lig_mol = None
            try:
                rmsd = compute_rmsd(u_eq, u_eq,
                                    alig_select="backbone",
                                    groupselections={"ligand": _lig_sel_ha,
                                                    "protein":'protein and not name H*'},
                                    plots_outdir=f"{sys_name}/equilibration"
                                    )
                rmsd.to_csv(f"{sys_name}/equilibration/{sys_name}_rmsd.csv", index=False)
                _contact_freq = calculate_contact_frequency(u_eq, ligand_selection)
                _rmsf = calculate_ligand_rmsf(u_eq, ligand_selection)
                if not is_peptide and ligand_file is not None:
                    lig_mol = Chem.SDMolSupplier(ligand_file)[0] #Avoid chemiperception problems
                    plot_atomic_property(u_eq, _rmsf, lig_resname=ligand_selection,
                                         outname=f"{sys_name}/equilibration/{sys_name}_RMSF.svg",
                                         ref_mol=lig_mol)
                    plot_atomic_property(u_eq, _contact_freq, lig_resname=ligand_selection,
                                         outname=f"{sys_name}/equilibration/{sys_name}_contact_freq.svg",
                                         ref_mol=lig_mol, color='r')
                lig_ha_eq = u_eq.select_atoms(_lig_sel_ha)
                pd.DataFrame({
                    'atom_name':    [a.name for a in lig_ha_eq.atoms],
                    'element':      [a.element for a in lig_ha_eq.atoms],
                    'rmsf':         _rmsf,
                    'contact_freq': _contact_freq,
                }).to_csv(f"{sys_name}/equilibration/{sys_name}_RMSF.csv", index=False)
            except Exception as e:
                logger.error(f"Error computing RMSD/RMSF: {e}")
                pass

            # ligand_atoms = u_eq.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')
            # logger.info(f"COM distance after equilibration is: {final_com:.2f} nm")

            if self.pocket_selection is not None:
                pocket_atom_indices = get_pocket_atoms_idxs(u_eq, self.pocket_selection)
            else:
                pocket_atom_indices = get_pocket_atoms_idxs(u_eq, ligand_selection=_lig_sel_ha)
            pocket_atoms = u_eq.select_atoms(f'index {" ".join(map(str, pocket_atom_indices))}')
            pocket_residues = [f"{atom.resname}_{atom.resid}" for atom in pocket_atoms]
            logger.info(f"Pocket residues are: {', '.join(set(pocket_residues))}")

            ligand_total_hatoms = [a.index for a in u_eq.select_atoms(_lig_sel_ha)]
            ligand_atoms_indices = get_ligand_anchor_atoms(u_eq, ligand_selection,
                                                        mode=self.sMD_ligand_anchor_mode,
                                                        n_atoms=5,
                                                        out_dir=sys_name,
                                                        ref_mol=lig_mol)
            logger.info(f"Ligand anchor atom indices are: {', '.join(map(str, ligand_atoms_indices))}")
            # write out the protein/ligand/pocket PDBs and PyMOL session.
            if self.pocket_selection is not None:
                pocket_view_selection = self.pocket_selection
            else:
                pocket_view_selection = (
                    f"(same residue as index {' '.join(map(str, pocket_atom_indices))})"
                    " and name CA"
                )
            try:
                write_pocket_pymol(
                    u=u_eq,
                    out_dir=sys_name,
                    protein_selection="protein",
                    ligand_selection=ligand_selection,
                    pocket_selection=pocket_view_selection,
                )
            except Exception as e:
                logger.error(f"Error writing pocket/ligand/protein pdbs: {e}")
                pass

        ##############################################################################################
        ##################################### Steered MD simulations #################################
        ##############################################################################################
        MERGE_CLUSTERING_FEATURES = False
        # Autostop and the kept vALL record must cluster exactly like the final analysis
        self._cluster_kwargs = dict(
            group_A=_lig_sel_ha, group_B=self.sMD_clust_selection,
            features=self.sMD_features, merge_features=MERGE_CLUSTERING_FEATURES,
            ligand_sdf=ligand_file, geom_features=bool(self.sMD_log_geom_features),
            recompute_distances=False,
        )
        CONVERGENCE_TOLERANCES = {
            "dG_weighted-rmsd": 4.0,  # kJ/mol — must match tol_rmsd in check_convergence()
            "barrier_delta":    3.0,  # kJ/mol — must match tol_barrier
            "r_ts_delta":       0.1,  # nm     — must match tol_r_ts
        }
        
        # sMD_collision_frequency = 1  # ps^-1
        # sMD_outdir = f"{sys_name}/sMD_{lig_anchor_mode}_{sMD_timestep}ps_{sMD_collision_frequency}ps_200stm"
        # in this paper they used 80 kcal·mol−1? units don match tho. Ziada et al 2022.
        sMD_spring_cte_per_atom = 100 * 4.184  # KJ/mol/nm2, converted from kcal. This affects thermal fluctuations
        sMD_traj_outdir = f"{self.sMD_outdir}/trajectories"
        
        sMD_analysis_outdir = f"{self.sMD_outdir}/analysis_traces"
        if self.sMD_clust_selection is not None:
            logger.info(f"sMD clustering selection: {self.sMD_clust_selection}")
            if MERGE_CLUSTERING_FEATURES:
                logger.info("Merging clustering features for sMD analysis.")
                sMD_analysis_outdir = f"{self.sMD_outdir}/analysis_merged"
            else:
                sMD_analysis_outdir = f"{self.sMD_outdir}/analysis_features"
        
        if self.run_sMDpulling:
            equilibrated_system = load_system(f"{sys_name}/equilibration/system_equil_{sys_name}.xml")

            if self.sMD_spring_cte is None:
                sMD_spring_cte = sMD_spring_cte_per_atom * len(ligand_total_hatoms)  # Normalize by ligand size
                logger.info(f"Spring constant set to {sMD_spring_cte} KJ/mol/nm2 for {len(ligand_total_hatoms)} atoms.")
            else:
                sMD_spring_cte = self.sMD_spring_cte

            sMD = SteeredMD(
                system=equilibrated_system,
                topology=topology,
                groupA_atoms=ligand_atoms_indices,
                groupB_atoms=pocket_atom_indices,
                ligand_atoms=ligand_total_hatoms,  # NC autostop + geom features: whole ligand
                restart_velocities=True,
                timestep=self.timestep,
                temperature=self.temperature,
                out_dir=sMD_traj_outdir,
                platform=self.platform,
                dx_per_move=self.sMD_dx_per_move,
                max_displacement=self.sMD_max_pulling_dist,
                sMD_spring_cte=sMD_spring_cte,
                sMD_max_r_offset=self.sMD_max_r_offset,
                autostop_nc=self.sMD_autostop_nc,
                autostop_nc_window=self.sMD_autostop_nc_window,
                autostop_min_displacement=self.sMD_autostop_min_displacement,
                save_freq=5,
                force_n_samples=self.sMD_force_n_samples,
                force_sample_stride=self.sMD_force_sample_stride,
                log_geom_features=self.sMD_log_geom_features,
            )

            if self.sMD_alternate_speeds and self.sMD_converge_speeds:
                # Round-robin: run one replica per live speed in turn so that
                # the 'force' estimator always has >=2 speeds advancing
                # together (it needs a joint v->0 extrapolation, not a single
                # speed's running PMF). Per-speed estimators still retire
                # each speed independently; 'force' retires all of them at
                # once when the joint ladder converges.
                live = {s: True for s in self.sMD_pulling_speeds}
                failures = {s: 0 for s in live}
                force_ladder_dead = False   # log the "<2 live speeds" ERROR once
                while any(live.values()):
                    for speed in round_robin_order(live):
                        reps = self.sMD_pulling_speeds[speed]
                        log_files = glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
                        if len(log_files) >= self.sMD_max_replicas:
                            logger.warning(
                                f"Reached maximum number of replicas ({self.sMD_max_replicas}) "
                                f"for speed {speed} nm/ps without convergence. Stopping."
                            )
                            live[speed] = False
                            continue
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                state_xml_file=equilibrated_state_xml,
                                pulling_speed=speed,
                                pulling_direction=self.sMD_pulling_dir,
                            )
                            failures[speed] = 0
                        except Exception as e:
                            failures[speed] += 1
                            logger.error(
                                f"Error during sMD pulling for speed {speed} nm/ps: {e} "
                                f"(consecutive failure {failures[speed]}/5)"
                            )
                            if failures[speed] >= 5:
                                logger.error(
                                    f"Aborting speed {speed} nm/ps after 5 consecutive failures."
                                )
                                live[speed] = False
                            continue
                        if len(log_files) + 1 < reps:
                            continue
                        # decision: per-speed estimators check that speed; force
                        # checks the joint ladder and retires every speed at once
                        if self.sMD_autostop_estimator == "force":
                            # Only *live* speeds may feed the ladder. Its depth K is
                            # min(replicas) over the speeds it is given, so a retired
                            # speed — whose .dat files stay on disk — would freeze K
                            # at the replica count it died on and the force criterion
                            # could never fire again.
                            live_speeds = [
                                s for s in self.sMD_pulling_speeds
                                if live[s] and glob(
                                    f"{sMD_traj_outdir}/sMD_*_v{s}_{self.sMD_pulling_dir}.dat"
                                )
                            ]
                            if len(live_speeds) < 2:
                                if not force_ladder_dead:
                                    force_ladder_dead = True
                                    logger.error(
                                        f"Only {len(live_speeds)} live pulling speed(s) remain "
                                        f"({live_speeds}); the 'force' convergence criterion needs "
                                        f">=2 speeds for its v->0 extrapolation and can no longer "
                                        f"be evaluated. The remaining speed(s) will run to the "
                                        f"replica cap ({self.sMD_max_replicas}) without an autostop "
                                        f"decision."
                                    )
                                continue
                            conv_df, _ = self._check_speed(
                                sMD_analysis_outdir, sys_name, _lig_sel_ha, reps,
                                speeds=live_speeds)
                        else:
                            conv_df, _ = self._check_speed(
                                sMD_analysis_outdir, sys_name, _lig_sel_ha, reps,
                                speed=speed)
                        if conv_df is None or conv_df.empty:
                            continue
                        # Tail-anchored, not first_streak: the question here is
                        # "are we converged right now", and conv_df is rebuilt from
                        # scratch each call (not append-only).
                        if tail_converged(conv_df, k_consec=self.sMD_conv_streak):
                            n_conv = float(conv_df["n_replicas"].max())
                            if self.sMD_autostop_estimator == "force":
                                logger.warning(
                                    f"sMD pulling CONVERGED (force ladder, all speeds; "
                                    f"trailing streak of {self.sMD_conv_streak} through rung "
                                    f"{n_conv:.0f}, window {self.sMD_conv_window})."
                                )
                                live = {s: False for s in live}
                            else:
                                logger.warning(
                                    f"sMD pulling for speed {speed} nm/ps CONVERGED after "
                                    f"{len(log_files) + 1} replicas (trailing streak of "
                                    f"{self.sMD_conv_streak} through rung {n_conv:.0f}, "
                                    f"window {self.sMD_conv_window})."
                                )
                                live[speed] = False
            else:
              for speed, reps in self.sMD_pulling_speeds.items():
                if self.sMD_converge_speeds:
                    logger.info(f"Running sMD for speed {speed} nm/ps until convergence (min {reps} replicas).")
                    CONVERGED = False
                    # Guard against livelock: a failing sMD.run() writes no .dat file, so the
                    # replica count never advances and the sMD_max_replicas cap never trips.
                    # Abort after too many consecutive failures that make no progress.
                    consecutive_failures = 0
                    MAX_CONSECUTIVE_FAILURES = 5
                    while not CONVERGED:
                        log_files = glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
                        current_replica = len(log_files) + 1
                        logger.info(f"Starting replica {current_replica} for speed {speed} nm/ps.")

                        # cap the number of replicas to avoid infinite loops
                        if len(log_files) >= self.sMD_max_replicas:
                            logger.warning(f"Reached maximum number of replicas ({self.sMD_max_replicas}) for speed {speed} nm/ps without convergence. Stopping.")
                            break

                        if len(log_files) >= reps:
                            # check convergence for this speed
                            conv_df, traces_df = self._check_speed(
                                sMD_analysis_outdir, sys_name, _lig_sel_ha, reps, speed=speed)

                            # conv_df is empty when replicas == reps (first PMF comparison
                            # needs one more replica); skip writing/plotting until data is available
                            if conv_df.empty:
                                logger.info(f"Not enough replicas yet for convergence comparison at speed {speed} nm/ps.")
                            else:
                                conv_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_v{speed}_metrics.csv", index=False)
                                traces_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_v{speed}_traces.csv", index=False)

                                # Plot convergence results. Use only the per-speed files
                                # written during this loop
                                conv_traces = [
                                    f for f in glob(f"{sMD_analysis_outdir}/sMD_conv_v*_traces.csv")
                                    if not f.endswith("vALL_traces.csv")
                                ]
                                conv_metrics = [
                                    f for f in glob(f"{sMD_analysis_outdir}/sMD_conv_v*_metrics.csv")
                                    if not f.endswith("vALL_metrics.csv")
                                ]

                                plot_convergence_traces(conv_traces, outdir=sMD_analysis_outdir)
                                plot_convergence_metrics(conv_metrics, outdir=sMD_analysis_outdir, tolerances=CONVERGENCE_TOLERANCES)

                                # Convergence = the readouts hold a plateau for
                                # sMD_conv_streak consecutive rungs. Two adjacent
                                # passes are not evidence: on WDR5 the per-rung flag
                                # flickers with a pass rate around 0.13, so a pair
                                # arises by chance.
                                # The streak must be the TAIL of the history, not any
                                # streak in it (first_streak): conv_df is rebuilt from
                                # scratch on every call, so an old streak followed by a
                                # failing latest rung — a restarted campaign, or a
                                # re-clustering that flipped an earlier rung — must not
                                # stop the run.
                                if len(conv_df) >= self.sMD_conv_streak:
                                    CONVERGED = tail_converged(conv_df, k_consec=self.sMD_conv_streak)
                                    n_conv = float(conv_df["n_replicas"].max())
                                else:
                                    logger.info(
                                        f"Only {len(conv_df)} convergence comparisons available for "
                                        f"speed {speed} nm/ps; need {self.sMD_conv_streak}."
                                    )
                                if CONVERGED:
                                    logger.warning(
                                        f"sMD pulling for speed {speed} nm/ps CONVERGED after "
                                        f"{current_replica} replicas (trailing streak of "
                                        f"{self.sMD_conv_streak} through rung {n_conv:.0f}, "
                                        f"window {self.sMD_conv_window})."
                                    )
                                    continue

                        # Run the next replica
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                state_xml_file=equilibrated_state_xml,
                                pulling_speed=speed,  # nm/ps
                                pulling_direction=self.sMD_pulling_dir,
                            )
                            consecutive_failures = 0  # progress made; reset the failure counter
                        except Exception as e:
                            consecutive_failures += 1
                            logger.error(
                                f"Error during sMD pulling for speed {speed} nm/ps, replica {current_replica}: {e} "
                                f"(consecutive failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                            )
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                logger.error(
                                    f"Aborting sMD for speed {speed} nm/ps after {consecutive_failures} "
                                    f"consecutive failures with no progress (e.g. CUDA device could not be "
                                    f"loaded). Check GPU/driver and CUDA_CACHE_PATH."
                                )
                                break
                            time.sleep(min(60, 10 * consecutive_failures))  # linear backoff
                            continue
                else:
                    n_done = len(glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat"))
                    logger.info(f"Running sMD for speed {speed} nm/ps: {reps} replicas requested, {n_done} on disk.")
                    for i in range(max(0, reps - n_done)):
                        existing = glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
                        if len(existing) >= self.sMD_max_replicas:
                            logger.warning(
                                f"Reached maximum replicas ({self.sMD_max_replicas}) for speed {speed} nm/ps "
                                f"(folder already has {len(existing)}). Stopping."
                            )
                            break
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                state_xml_file=equilibrated_state_xml,
                                pulling_speed=speed,  # nm/ps
                                pulling_direction=self.sMD_pulling_dir,
                            )
                        except Exception as e:
                            logger.error(f"Error during sMD pulling for speed {speed} nm/ps, replica {i+1}: {e}")
                            continue

        # Load and align sMD trajectories
        sMD_trajs = glob(f"{sMD_traj_outdir}/sMD_replica-*_*_*.dcd")
        sMD_trajs = [f for f in sMD_trajs if "aligned" not in f]  # only process unaligned trajectories
        if self.sMD_align_trajs:
            logger.info(f"Found {len(sMD_trajs)} sMD trajectories to align.")
            wrap_align_save_traj(sMD_trajs, solvated_system_pdb, is_membrane=self.is_membrane)
        else:
            logger.info(f"sMD_align_trajs=False: keeping {len(sMD_trajs)} raw sMD trajectories (<stem>.dcd).")
                                
        ##############################################################################################
        ######################################### sMD Analysis #######################################
        ##############################################################################################        
        
        sMD_trajs = glob(f"{sMD_traj_outdir}/sMD_replica-*_*_*_aligned.dcd")
        if not self.sMD_align_trajs:
            sMD_trajs = [f for f in glob(f"{sMD_traj_outdir}/sMD_replica-*_*_*.dcd") if "aligned" not in f]

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
            # 'null' bypasses clustering entirely (all trajectories -> one path); it is the
            # no-clustering reference for isolating what the path split actually buys.
            if self.sMD_path_model == 'null':
                cluster_model = NullPathModel()
                logger.info("sMD_path_model='null': clustering bypassed, all trajectories "
                            "assigned to a single path.")
            else:
                cluster_model = DTWPathModel(seed=self.random_state, do_plots=True,
                                            outdir=sMD_analysis_outdir)

            smdanalysis = SMDAnalysis(sys_name, cluster_model,
                                    estimators=['cumulant', 'jarzynski', 'force'],
                                    do_plots=True, seed=self.random_state,
                                    temperature=self.temperature,
                                    ligand_select=_lig_sel_ha,
                                    pocket_select=self.pocket_selection,
                                    outdir=sMD_analysis_outdir,
                                    filter_low_support=True,
                                    min_samples_per_step=5,
                                    min_support_ratio=1.0,
                                    min_replicas_per_path=self.sMD_min_replicas_per_path,
                                    n_paths=self.sMD_n_paths,
                                    min_path_steps_ratio=0.6,
                                    max_frac_neg_dG_first_half=self.sMD_max_frac_neg_dG_first_half,
                                    min_speeds_for_extrapolation=2,
                                    path_combination=self.sMD_path_combination,
                                    )

            smd_data = smdanalysis.run(smd_data,
                                       group_A=_lig_sel_ha,
                                       group_B=self.sMD_clust_selection,
                                       merge_features=MERGE_CLUSTERING_FEATURES,
                                       cluster_across_speeds=self.cluster_across_speeds,
                                       features=self.sMD_features,
                                       ligand_sdf=ligand_file,
                                       geom_features=bool(self.sMD_log_geom_features),
                                       plateau_frac=self.sMD_plateau_frac,
                                       cluster_to_boundary=self.sMD_cluster_to_boundary,
                                       boundary_buffer_frac=self.sMD_boundary_buffer_frac,
                                       recompute_distances=False,  # load precomputed pocket-distance caches
                                       kramers_abs=self.sMD_kramers_abs,
                                       kramers_bound=self.sMD_kramers_bound,
                                    #    r_range=(0, 1.75)
                                       )

            # check convergence regardless of speed and autopstop.
            # Cluster with the same feature config as the main analysis run() above.
            # conv_window/estimator_name must mirror the deployment loop's autostop
            # settings: this vALL frame (plus its plots) is the only surviving
            # convergence artefact, so it has to be the criterion that actually
            # made the stopping decision, not the function defaults.
            # Same first comparison rung as the autostop loop (its per-speed `reps`)
            _reps = [r for r in self.sMD_pulling_speeds.values() if r is not None]
            conv_df, traces_df = smdanalysis.check_convergence(logs=logs,
                estimator_name=self.sMD_autostop_estimator,
                conv_window=self.sMD_conv_window,
                **({"min_replicas": min(_reps)} if _reps else {}),
                **self._cluster_kwargs,
                plateau_frac=self.sMD_plateau_frac,
                cluster_to_boundary=self.sMD_cluster_to_boundary,
                restrict_rmsd_to_boundary=self.sMD_cluster_to_boundary,
                boundary_buffer_frac=self.sMD_boundary_buffer_frac,
            )
            conv_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_vALL_metrics.csv", index=False)
            if not traces_df.empty:
                traces_df.to_csv(f"{sMD_analysis_outdir}/sMD_conv_vALL_traces.csv", index=False)

            # delete per-speed convergence files written during the convergence loop;
            # vALL files contain all the same data
            for _f in glob(f"{sMD_analysis_outdir}/sMD_conv_v*_metrics.csv"):
                if not _f.endswith("vALL_metrics.csv"):
                    os.remove(_f)
            for _f in glob(f"{sMD_analysis_outdir}/sMD_conv_v*_traces.csv"):
                if not _f.endswith("vALL_traces.csv"):
                    os.remove(_f)

            if not traces_df.empty:
                smd_conv_traces = glob(f"{sMD_analysis_outdir}/sMD_conv_vALL_traces.csv")
                plot_convergence_traces(smd_conv_traces, outdir=sMD_analysis_outdir)
            smd_conv_metrics = glob(f"{sMD_analysis_outdir}/sMD_conv_vALL_metrics.csv")
            plot_convergence_metrics(smd_conv_metrics, outdir=sMD_analysis_outdir, tolerances=CONVERGENCE_TOLERANCES)

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished AutoPath simulation in {simulation_time/60:.2f} min.")

        return

    def _check_speed(self, outdir, sys_name, lig_sel, min_replicas, speed=None,
                     speeds=None):
        """Build the analysis and run the convergence check for one speed.

        ``speed=None`` and ``speeds=None`` mean "every speed on disk jointly",
        which is what the force ladder needs. Pass ``speeds=[...]`` to restrict
        the joint check to a subset (the still-live speeds): the ladder's depth
        is min(replicas) over the speeds it is handed, so a retired speed left
        on disk would otherwise cap it forever.
        `min_replicas` is passed explicitly rather than derived from
        sMD_pulling_speeds, whose values are None in the Config default
        (`config.py:46`) and would raise on min().
        Returns (convergence_df, traces_df); convergence_df is empty when it
        cannot be computed yet.
        """
        if speeds is not None:
            speeds = list(speeds)
        elif speed is not None:
            speeds = [speed]
        if speeds is None:
            logs = glob(f"{self.sMD_outdir}/trajectories/sMD_*_{self.sMD_pulling_dir}.dat")
        else:
            logs = sorted({fn for s in speeds for fn in
                           glob(f"{self.sMD_outdir}/trajectories/sMD_*_v{s}_{self.sMD_pulling_dir}.dat")})
        smdanalysis = SMDAnalysis(sysname=sys_name, path_model=self.sMD_path_model,
                                  estimators=[self.sMD_autostop_estimator],
                                  do_plots=False, seed=self.random_state,
                                  temperature=self.temperature, outdir=outdir,
                                  ligand_select=lig_sel,
                                  pocket_select=self.pocket_selection,
                                  n_paths=self.sMD_n_paths,
                                  path_combination=self.sMD_path_combination)
        conv_df, traces_df = smdanalysis.check_convergence(
            logs=logs, speeds=speeds,
            min_replicas=min_replicas,
            estimator_name=self.sMD_autostop_estimator,
            conv_window=self.sMD_conv_window,
            **self._cluster_kwargs,
            plateau_frac=self.sMD_plateau_frac,
            cluster_to_boundary=self.sMD_cluster_to_boundary,
            restrict_rmsd_to_boundary=self.sMD_cluster_to_boundary,
            boundary_buffer_frac=self.sMD_boundary_buffer_frac,
        )
        return conv_df, traces_df

