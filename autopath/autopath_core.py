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
from autopath.clustering import (Clustering, PairwiseDistances, read_positions,
                                 order_milestones_bound_to_unbound)
from autopath.pdb_preprocessor import PDBPreprocessor
from autopath.ap_PLIP import plot_atomic_property, calculate_ligand_rmsf, calculate_contact_frequency
from autopath import (
    SystemPreparation,
    Equilibration,
    SteeredMD,
    RelaxMD,
    MetadynamicsMD,
)
from autopath.metadynamics import (
    MetadynamicsAnalysis,
    com_cv,
    pathCV_cv,
    funnel_z_cv,
    write_metad_preseed,
    generate_funnel_parameters_from_trajectory,
    create_funnel_force_from_trajectory_analysis,
    com_distance_at_z,
    save_funnel_params,
    write_funnel_pymol,
)
from autopath.pulling import SMDData, SMDAnalysis
from autopath.pulling.Convergence import (tail_converged, validate_autostop_options,
                                          round_robin_order)
from autopath.pulling.PathModel import DTWPathModel, NullPathModel
from autopath.pulling.Diagnostics import plot_convergence_traces, plot_convergence_metrics

import logging
logger = logging.getLogger('autopath.core')


def group_milestones_by_path(milestones: list[str],
                             milestones_outdir: str,
                             ) -> list[tuple[str | None, list[str]]]:
    """Group milestone state files by the unbinding path they came from.

    Per-path milestones live in a subdirectory named after the path label;
    files directly in ``milestones_outdir`` form one unlabelled group (``None``).

    Returns ``[(path_label_or_None, files)]`` sorted by key, ``None`` first.
    """
    groups = defaultdict(list)
    root = os.path.abspath(milestones_outdir)
    for m in milestones:
        rel = os.path.relpath(os.path.abspath(m), root)
        head = os.path.dirname(rel)
        if head and not head.startswith(os.pardir):
            key = head.split(os.sep)[0]
        else:
            key = None
        groups[key].append(m)
    return [(k, groups[k]) for k in sorted(groups, key=lambda k: (k is not None, k or ""))]


def walker_seed_order(path_milestones: list, order: str) -> list:
    """Order the seeds for sequential shared-bias walkers.

    `path_milestones` runs bound -> unbound. 'unbound_first' makes the most-bound
    walker inherit every earlier walker's bias; 'bound_first' lets accumulated
    bias push each later walker along the path instead of filling its basin.
    """
    if order == "unbound_first":
        return list(reversed(path_milestones))
    if order == "bound_first":
        return list(path_milestones)
    raise ValueError("order must be 'unbound_first' or 'bound_first', "
                     f"got {order!r}")


class AutoPath:
    """Main orchestrator for the AutoPath binding-free-energy pipeline.

    AutoPath runs a six-stage protocol that takes a protein-ligand PDB as input
    and produces a converged binding free energy estimate:

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
       cumulant and Jarzynski estimators, and identifies medoid trajectories for
       downstream milestone extraction.
    5. **Metadynamics** — multi-walker well-tempered funnel metadynamics via
       :class:`~autopath.metadynamics.MetadynamicsMD`.  Each milestone from
       stage 4 seeds a separate walker.  Walkers that remain stuck in the bound
       state (max PathCV < 0.5) are automatically retried with the accumulated
       bias surface.  Optionally, the sMD PMF is used to pre-seed the
       metadynamics bias (``mMD_preseed_bias=True``).
    6. **Analysis & funnel correction** — :class:`~autopath.metadynamics.MetadynamicsAnalysis`
       computes the FES, applies the funnel volume correction, and reports
       ΔG°_b and pK_d.

    Subcomponents
    -------------
    :class:`~autopath.preparation.SystemPreparation`,
    :class:`~autopath.equilibration.Equilibration`,
    :class:`~autopath.pulling.SteeredMD`,
    :class:`~autopath.pulling.SMDAnalysis`,
    :class:`~autopath.relax_md.RelaxMD`,
    :class:`~autopath.metadynamics.MetadynamicsMD`,
    :class:`~autopath.metadynamics.MetadynamicsAnalysis`

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
        ``sMD_run_analysis``, ``sMD_clust_selection``, ``sMD_features``,
        ``sMD_log_geom_features``, ``sMD_plateau_frac``,
        ``sMD_cluster_to_boundary``, ``sMD_boundary_buffer_frac``,
        ``cluster_across_speeds``, ``sMD_min_replicas_per_path``
    Milestone extraction
        ``extract_milestones``, ``milestone_mode``,
        ``milestone_min_frame_separation``, ``n_milestones``
    Metadynamics
        ``run_metadynamics``, ``mMD_use_funnel_potential``,
        ``mMD_bias_factor``, ``mMD_bias_frequency``, ``mMD_hill_height``,
        ``mMD_hill_width``, ``mMD_time``, ``mMD_milestone_seeding``,
        ``mMD_multiple_walkers``, ``mMD_preseed_bias``, ``mMD_preseed_speed``,
        ``mMD_funnel_host_selection``

    Notes
    -----
    * ``mMD_hill_height=1.2`` kJ/mol is ~0.5 kBT at 300 K; larger values
      over-fill barriers and give noisy FES.
    * CVs are :class:`~autopath.metadynamics.CVSpec` objects instantiated inside
      the metadynamics loop, once the OpenMM ``System`` is available.
    * ``mMD_milestone_seeding``: each walker starts from its own relaxed
      milestone (True) or all from the most-bound one (False).
    * ``mMD_multiple_walkers``: one shared bias directory (True) or one per
      walker (False).
    * Walkers whose CV never passes the grid midpoint are re-run once.
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
        sMD_force_n_samples: int = 10,  # max restraint-force samples averaged per move (>=1)
        sMD_force_sample_stride: int = 5,  # MD steps between force samples
        sMD_ligand_anchor_mode: str = 'murcko',
        sMD_max_replicas: int = 50,  # max replicas per speed in convergence mode
        sMD_run_analysis: bool = True,
        sMD_clust_selection:str = None,
        sMD_path_model: str = 'dtw',   # 'dtw' | 'null' ('null' = no clustering: one path)
        sMD_n_paths: int | None = None,  # fixed number of paths; None = silhouette-selected
        sMD_max_frac_neg_dG_first_half: float = 0.25,  # pass-3 binding-well filter; <=0 disables
        sMD_features: list | None = None,
        sMD_log_geom_features: bool | list = True,  # hybrid: log geom during pulling + merge into clustering
        sMD_plateau_frac: float = 0.4,  # force-plateau TS boundary as a fraction of peak |force|
        sMD_cluster_to_boundary: bool = True,  # cluster/converge only up to the force-plateau boundary
        sMD_boundary_buffer_frac: float = 0.1,  # boundary buffer as a fraction of r_ts
        cluster_across_speeds: bool = False,
        sMD_min_replicas_per_path: int = 5,
        extract_milestones: bool = True,
        milestone_mode: str = "all_medoids",  # "per_path" or "all_medoids"
        milestone_min_frame_separation: int = 0,
        n_milestones: int = 5,
        run_metadynamics: bool = True,
        mMD_use_funnel_potential: bool = False,
        mMD_bias_factor: int = 15,
        mMD_bias_frequency: int = 2,  # ps
        mMD_hill_height: float = 1.2,  # kJ/mol approx 0.5KT
        mMD_hill_width: float = 0.05,
        mMD_time: int = 5,  # ns
        mMD_milestone_seeding: bool = True,
        mMD_multiple_walkers: bool = False,
        mMD_seed_order: str = "unbound_first",
        mMD_bulk_cutoff_ang: float = 10.0,
        mMD_cv: str = "path",
        mMD_outdir_suffix: str = "",
        mMD_preseed_bias: bool = False,
        mMD_preseed_speed: float = None,
        mMD_funnel_host_selection: str | None = None,
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
        self.sMD_clust_selection = sMD_clust_selection
        self.sMD_path_model = str(sMD_path_model).lower()
        self.sMD_n_paths = sMD_n_paths
        self.sMD_max_frac_neg_dG_first_half = float(sMD_max_frac_neg_dG_first_half)
        if self.sMD_path_model not in ('dtw', 'null'):
            raise ValueError(f"sMD_path_model must be 'dtw' or 'null', got {sMD_path_model!r}")
        self.sMD_features = sMD_features
        self.sMD_log_geom_features = sMD_log_geom_features
        self.sMD_plateau_frac = sMD_plateau_frac
        self.sMD_cluster_to_boundary = sMD_cluster_to_boundary
        self.sMD_boundary_buffer_frac = sMD_boundary_buffer_frac
        self.cluster_across_speeds = cluster_across_speeds
        self.sMD_min_replicas_per_path = sMD_min_replicas_per_path
        # Milestones
        self.extract_milestones = extract_milestones
        self.milestone_mode = milestone_mode
        self.milestone_min_frame_separation = milestone_min_frame_separation
        self.n_milestones = n_milestones
        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_use_funnel_potential = mMD_use_funnel_potential
        self.mMD_bias_factor = mMD_bias_factor
        self.mMD_bias_frequency = mMD_bias_frequency
        self.mMD_hill_height = mMD_hill_height
        self.mMD_hill_width = mMD_hill_width
        self.mMD_time = mMD_time
        self.mMD_milestone_seeding = mMD_milestone_seeding
        self.mMD_multiple_walkers = mMD_multiple_walkers
        if mMD_seed_order not in ("unbound_first", "bound_first"):
            raise ValueError("mMD_seed_order must be 'unbound_first' or "
                             f"'bound_first', got {mMD_seed_order!r}")
        self.mMD_seed_order = mMD_seed_order
        self.mMD_bulk_cutoff_ang = mMD_bulk_cutoff_ang
        if mMD_cv not in ("path", "funnel_z"):
            raise ValueError(f"mMD_cv must be 'path' or 'funnel_z', got {mMD_cv!r}")
        if mMD_cv == "funnel_z" and not mMD_use_funnel_potential:
            raise ValueError("mMD_cv='funnel_z' is the funnel axis coordinate; "
                             "it needs mMD_use_funnel_potential=True")
        if mMD_cv == "funnel_z" and mMD_preseed_bias:
            raise ValueError("mMD_preseed_bias builds its bias on the path-CV grid; "
                             "it cannot seed mMD_cv='funnel_z'")
        self.mMD_cv = mMD_cv
        self.mMD_outdir_suffix = mMD_outdir_suffix
        self.mMD_preseed_bias = mMD_preseed_bias
        self.mMD_preseed_speed = mMD_preseed_speed
        self.mMD_funnel_host_selection = mMD_funnel_host_selection
        # VS mode
        self.equilibration_checkpoint = False
        self.pulling_checkpoint = False
        if VS_mode:
            self.equilibration_checkpoint = True
            self.eq_checkpoint_cutoff = 0.3  # nm
            self.pulling_checkpoint = True

        # Process the input PDB
        if do_fix_pdb:
            protein_pdb = PDBPreprocessor(pdb_path).fix(
                                  cap_termini=True,
                                  keep_heterogens=True, 
                                  pH=7.4)
            self.protein_file = pdb_path.replace(".pdb", "_fixed.pdb")
            save_pdb(protein_pdb.topology, protein_pdb.positions, self.protein_file)
        else:
            self.protein_file = pdb_path

    def run(self, ligand_file: str = None, ligand_selection: str = "resname UNK"):
        """Execute the full six-stage AutoPath pipeline for one ligand.

        Stages are run in order: system preparation → equilibration →
        steered MD → path clustering → metadynamics → trajectory alignment.
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

            # Equilibration VS checkpoint
            # if self.equilibration_checkpoint:
            #     final_rmsd = lig_rmsd_equilibration[-1:].values
            #     if final_rmsd > self.eq_checkpoint_cutoff * 10:  # to Angs
            #         logger.warning(f"Simulation for ligand {sys_name} terminated because ligand RMSD={final_rmsd:.2f} > {self.eq_checkpoint_cutoff}")
            #         exit(1)
            # ligand_atoms = u_eq.select_atoms(f'index {" ".join(map(str, ligand_atoms_indices))}')
            # final_com = calculate_com_distance(u_eq, ligand_atoms, pocket_atoms, wrap=False)[-1] /10 # convert to nm
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
        CONVERGENCE_TOLERANCES = {
            "dG_weighted-rmsd": 4.0,  # kJ/mol — must match tol_rmsd in check_convergence()
            "barrier_delta":    3.0,  # kJ/mol — must match tol_barrier
            "r_ts_delta":       0.1,  # nm     — must match tol_r_ts
        }
        
        # sMD_collision_frequency = 1  # ps^-1
        # sMD_outdir = f"{sys_name}/sMD_{lig_anchor_mode}_{sMD_timestep}ps_{sMD_collision_frequency}ps_200stm"
        # cf. Ziada et al. 2022
        sMD_spring_cte_per_atom = 100 * 4.184  # KJ/mol/nm2, converted from kcal
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
                logger.info(f"Spring constant set to {sMD_spring_cte} KJ/mol/nm2 for {len(ligand_atoms_indices)} atoms.")
            else:
                sMD_spring_cte = self.sMD_spring_cte

            sMD = SteeredMD(
                system=equilibrated_system,
                topology=topology,
                groupA_atoms=ligand_atoms_indices,
                groupB_atoms=pocket_atom_indices,
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
                # Round-robin keeps >=2 speeds advancing for the 'force' v->0
                # extrapolation; 'force' retires all speeds at once.
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
                        if self.sMD_autostop_estimator == "force":
                            # Only live speeds: ladder depth is min(replicas), so a
                            # retired speed left on disk would freeze it.
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
                        # tail-anchored: conv_df is rebuilt from scratch each call
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
                    # a failed run writes no .dat, so the replica cap alone cannot stop a livelock
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

                                # Converged = the last sMD_conv_streak rungs all pass.
                                # Must be the tail streak: conv_df is rebuilt each call.
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
                    logger.info(f"Running sMD for speed {speed} nm/ps with {reps} replicas.")
                    for i in range(reps):
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
        logger.info(f"Found {len(sMD_trajs)} sMD trajectories to align.")        
        wrap_align_save_traj(sMD_trajs, solvated_system_pdb, is_membrane=self.is_membrane)
                                
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
            
            # cluster trajectories into pathways ('null' = one path, no clustering)
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
                                    #    r_range=(0, 1.75)
                                       )

            # Store for downstream milestone extraction
            self._smdanalysis = smdanalysis

            # Convergence over all speeds, with the same features as run() above and
            # the same estimator/window as the autostop loop (this is the kept record).
            conv_df, traces_df = smdanalysis.check_convergence(logs=logs,
                estimator_name=self.sMD_autostop_estimator,
                conv_window=self.sMD_conv_window,
                group_A=_lig_sel_ha,
                group_B=self.sMD_clust_selection,
                features=self.sMD_features,
                merge_features=MERGE_CLUSTERING_FEATURES,
                ligand_sdf=ligand_file,
                geom_features=bool(self.sMD_log_geom_features),
                plateau_frac=self.sMD_plateau_frac,
                cluster_to_boundary=self.sMD_cluster_to_boundary,
                restrict_rmsd_to_boundary=self.sMD_cluster_to_boundary,
                boundary_buffer_frac=self.sMD_boundary_buffer_frac,
                recompute_distances=False,  # load precomputed pocket-distance caches
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

        ##############################################################################################
        ###################################### Extract Milestones ####################################
        ##############################################################################################

        milestones_outdir = f"{sys_name}/milestones"
        
        if self.extract_milestones or self.run_metadynamics:
            # Resolve ligand_atoms_full_indices for downstream use (metadynamics, relax)
            u_sMD = mda.Universe(solvated_system_pdb, sMD_trajs)
            ligand_atoms_full = u_sMD.select_atoms(_lig_sel_ha)
            ligand_atoms_full_indices = [atom.index for atom in ligand_atoms_full]

            ligand_sel = _lig_sel_ha
            pocket_sel = f'index {" ".join(map(str, pocket_atom_indices))}'

        if self.extract_milestones:
            
            os.makedirs(milestones_outdir, exist_ok=True)

            logger.info(f"Found {len(sMD_trajs)} sMD trajectories for milestone extraction.")

            if len(sMD_trajs) == 0:
                raise RuntimeError("No sMD trajectories found. Please check the sMD pulling step.")

            # Load medoid info (from analysis or from saved file); flat path is the legacy layout
            medoid_info_path = f"{sMD_analysis_outdir}/path_analysis/medoid_info.json"
            if not os.path.exists(medoid_info_path):
                medoid_info_path = f"{sMD_analysis_outdir}/medoid_info.json"
            medoid_to_path = {}
            medoid_names = []

            if hasattr(self, '_smdanalysis') and hasattr(self._smdanalysis.path_model, 'all_medoid_names'):
                medoid_names = self._smdanalysis.path_model.all_medoid_names
                medoid_to_path = self._smdanalysis.path_model.medoid_to_path
            elif os.path.exists(medoid_info_path):
                with open(medoid_info_path, "r") as f:
                    medoid_info = json.load(f)
                # flatten per-speed "by_speed" maps; top-level "medoid_to_path" is legacy
                medoid_to_path = dict(medoid_info.get("medoid_to_path", {}))
                for mapping in medoid_info.get("by_speed", {}).values():
                    medoid_to_path.update(mapping)
                medoid_names = list(medoid_to_path)
                logger.info(f"Loaded medoid info from {medoid_info_path}: {len(medoid_names)} medoids.")
            else:
                logger.warning(
                    "No medoid info found (run sMD analysis first, or provide medoid_info.json). "
                    "Falling back to using all sMD trajectories."
                )

            # medoid_info.json predates the path filters and its labels can be stale;
            # filter and re-label per trajectory from the final analysis output.
            if medoid_to_path:
                surviving = read_trajname_to_path(sMD_analysis_outdir)
                if not surviving:
                    logger.warning(
                        "No analysis path table under %s; cannot tell which "
                        "trajectories survived, so every medoid is used.",
                        sMD_analysis_outdir)
                medoid_to_path, dropped, disagree = (
                    filter_medoids_by_surviving_trajectories(medoid_to_path, surviving))
                if dropped:
                    logger.warning(
                        "Dropping %d medoid(s) whose trajectory did not survive the "
                        "sMD quality filters: %s.", len(dropped), sorted(dropped))
                if disagree:
                    logger.warning(
                        "medoid_info.json disagrees with the analysis on %d medoid(s); "
                        "using the analysis label. %s", len(disagree),
                        {m: f"{a} -> {b}" for m, (a, b) in disagree.items()})
                medoid_names = [m for m in medoid_names if m in medoid_to_path]
                if surviving and not medoid_to_path:
                    raise RuntimeError(
                        "No medoid trajectory survived the sMD path quality filters, "
                        "so milestones would come from rejected data. Re-run the sMD "
                        "analysis or relax its filters.")

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
                        clu = Clustering(
                            PairwiseDistances(ligand_sel, pocket_sel)
                        ).fit(u_path)
                        frames = clu.select(self.n_milestones, strategy="centroid",
                                            min_frame_separation=self.milestone_min_frame_separation)
                        ms_files = clu.write_states(
                            u_path, frames, topology, system,
                            out_dir=path_outdir, prefix=f"milestone_{path_id}",
                            temperature=self.temperature, ligand_sel=ligand_sel)
                        clu.plot(path_outdir, f"milestone_{path_id}", frames=frames)
                        all_milestone_files.extend(ms_files)

                        logger.info(f"Path {path_id}: extracted {len(ms_files)} milestones.")

            if self.milestone_mode == "all_medoids":
                # Mode: pool all medoid trajectories, cluster together
                logger.info(f"Computing distance features for all medoids ({len(u_milestone.trajectory)} frames)...")
                clu = Clustering(
                    PairwiseDistances(ligand_sel, pocket_sel)
                ).fit(u_milestone)
                frames = clu.select(self.n_milestones, strategy="centroid",
                                    min_frame_separation=self.milestone_min_frame_separation)
                milestone_files = clu.write_states(
                    u_milestone, frames, topology, system,
                    out_dir=milestones_outdir, prefix="milestone",
                    temperature=self.temperature, ligand_sel=ligand_sel)
                clu.plot(milestones_outdir, "milestone", frames=frames)

                logger.info(f"Extracted {len(milestone_files)} milestones from {len(medoid_dcds)} medoid trajectories.")

        ##############################################################################################
        ##################################### Metadynamics simulations ###############################
        ##############################################################################################
        
        if self.mMD_use_funnel_potential:
            if self.mMD_preseed_bias:
                mMD_out_dir = f"{sys_name}/metadynamics_funnel_preseed"
            else:
                mMD_out_dir = f"{sys_name}/metadynamics_funnel"
        else:
            mMD_out_dir = f"{sys_name}/metadynamics"
        mMD_out_dir += self.mMD_outdir_suffix
        os.makedirs(mMD_out_dir, exist_ok=True)
            
        if self.run_metadynamics:
            min_com = 0.0
            max_com = 3.0
            try:
                smd_raw = pd.read_csv(f'{sMD_analysis_outdir}/sMD_processed_data.csv')
                min_com = smd_raw['r_before'].min() * 0.75  # nm
                max_com = smd_raw['r_before'].max() * 1.1 # nm
            except Exception as e:
                logger.error(f"Error loading sMD raw data: {e}")

            logger.info(f"sMD COM distance range: [{min_com:.3f}, {max_com:.3f}] nm (informational; path CV uses [0.0, 1.0])")

            milestones = glob(f'{milestones_outdir}/**/milestone_*_frame_*.xml', recursive=True)
            # drop RelaxMD outputs (_relax_*) written beside the milestones
            milestones = [m for m in milestones if '_relax' not in os.path.basename(m)]

            if len(milestones) == 0:
                raise RuntimeError("No milestones found. Please check the milestone extraction step.")

            # One metadynamics stage per path: pooling distinct routes would need one
            # path CV and a funnel wide enough for all of them.
            milestone_groups = group_milestones_by_path(milestones, milestones_outdir)
            has_path_labels = any(key is not None for key, _ in milestone_groups)

            if has_path_labels:
                kept_paths = read_surviving_paths(sMD_analysis_outdir)
                if kept_paths:
                    dropped = [k for k, _ in milestone_groups
                               if k is not None and k not in kept_paths]
                    if dropped:
                        logger.warning(
                            "Skipping milestone group(s) %s: not among the paths that "
                            "survived the sMD analysis (%s).",
                            sorted(dropped), sorted(kept_paths))
                    # the unlabelled group is never dropped: it has no path to check
                    milestone_groups = [(k, v) for k, v in milestone_groups
                                        if k is None or k in kept_paths]
                    if not milestone_groups:
                        raise RuntimeError(
                            "No milestone group corresponds to a surviving sMD path "
                            f"(surviving paths: {sorted(kept_paths)}). Re-run the sMD "
                            "analysis or relax its filters.")

            base_system = load_system(f"{sys_name}/system.xml")
            # {trajname: path}: per-group trajectory selection, and rejection filter for the unlabelled group
            trajname_to_path = (read_trajname_to_path(sMD_analysis_outdir)
                                if self.mMD_use_funnel_potential or has_path_labels
                                else {})
            # Collect aligned forward-direction sMD trajectories only
            all_smd_trajs = glob(f"{sMD_traj_outdir}/*_{self.sMD_pulling_dir}_aligned.dcd")

            for group_key, group_milestones in milestone_groups:
                group_name = group_key if group_key is not None else "all milestones"
                # unlabelled group keeps the flat layout untouched
                group_out_dir = (mMD_out_dir if group_key is None
                                 else os.path.join(mMD_out_dir, group_key))
                # relax outputs stay beside the milestones they relax
                group_milestones_dir = (milestones_outdir if group_key is None
                                        else os.path.join(milestones_outdir, group_key))
                os.makedirs(group_out_dir, exist_ok=True)
                logger.info(
                    f"Metadynamics for {group_name}: {len(group_milestones)} milestone(s) "
                    f"-> {group_out_dir}")

                # sort by milestone index (last token before '_frame_')
                path_milestones = sorted(
                    group_milestones,
                    key=lambda x: int(os.path.basename(x).split('_frame_')[0].rsplit('_', 1)[-1]))

                # Index order follows PC1 (arbitrary sign): orient bound -> unbound here,
                # before any consumer below reads the list.
                path_milestones = order_milestones_bound_to_unbound(
                    path_milestones, pocket_atom_indices, ligand_atoms_indices)

                path_cv = pathCV_cv(
                    topology=topology,
                    milestones=path_milestones,
                    pocket_atoms=pocket_atom_indices,
                    ligand_atoms=ligand_atoms_indices,
                    system=base_system,
                    grid_min=0.0,
                    grid_max=1.0,
                    hill_width=self.mMD_hill_width,
                    sigma='auto', # ~ milestone spacing
                )
                metad_cv = path_cv
                # sigma should be ~0.5-3x the logged adjacent-milestone COM spacing

                # Pre-seed the bias from the sMD PMF before any walker starts (loaded via _syncWithDisk()).
                if self.mMD_preseed_bias:
                    try:
                        # milestone COM distances (same formula as pathCV_cv)
                        _pocket_masses = np.array([
                            base_system.getParticleMass(i).value_in_unit(dalton)
                            for i in pocket_atom_indices
                        ])
                        _ligand_masses = np.array([
                            base_system.getParticleMass(i).value_in_unit(dalton)
                            for i in ligand_atoms_indices
                        ])
                        _milestone_com_dists = []
                        for _m_file in path_milestones:
                            _pos = read_positions(_m_file)
                            _p_com = np.average(_pos[pocket_atom_indices], axis=0, weights=_pocket_masses)
                            _l_com = np.average(_pos[ligand_atoms_indices], axis=0, weights=_ligand_masses)
                            _milestone_com_dists.append(float(np.linalg.norm(_l_com - _p_com)))

                        write_metad_preseed(
                            smd_analysis_outdir=sMD_analysis_outdir,
                            milestone_com_distances=_milestone_com_dists,
                            bias_dir=group_out_dir,
                            gamma=self.mMD_bias_factor,
                            temperature=self.temperature,
                            speed=self.mMD_preseed_speed,
                            alpha=0.05
                        )
                    except Exception as _preseed_err:
                        logger.error(f"Preseed bias failed (non-fatal, continuing without preseed): {_preseed_err}")

                milestone_relax = RelaxMD(
                    topology=topology,
                    ligand_atoms=ligand_total_hatoms, # use all HA
                    pocket_atoms=pocket_atom_indices,
                    out_dir=group_milestones_dir,
                    is_membrane=self.is_membrane,
                    temp=self.temperature,
                    timestep=self.timestep,
                    platform=self.platform
                )

                WTMetaD = MetadynamicsMD(
                    topology=topology,
                    ligand_atoms=ligand_atoms_indices, # use ligand anchor atoms for biasing
                    pocket_atoms=pocket_atom_indices,
                    restrained_atoms=None,
                    is_membrane=self.is_membrane,
                    timestep=self.timestep,
                    temp=self.temperature,
                    out_dir=group_out_dir,
                    platform=self.platform
                )

                # Generate funnel potential if requested
                funnel_force = None
                funnel_params = None
                if self.mMD_use_funnel_potential:
                    logger.info("Generating funnel potential from sMD trajectories...")
                    try:
                        # Build the funnel from this path's trajectories only; pooling widens the cone.
                        smd_trajs = all_smd_trajs
                        if group_key is None:
                            # no single path: just drop the trajectories the analysis rejected
                            _ok = {os.path.basename(self._trajname_to_dcd(n))
                                   for n in trajname_to_path}
                            _kept = [t for t in all_smd_trajs
                                     if os.path.basename(t) in _ok]
                            if _kept:
                                logger.info(
                                    "Building the funnel from the %d of %d sMD "
                                    "trajectories that passed the quality filters.",
                                    len(_kept), len(all_smd_trajs))
                                smd_trajs = _kept
                            else:
                                logger.info(
                                    "No trajectory-to-path map under %s; building the "
                                    "funnel from all sMD trajectories.",
                                    sMD_analysis_outdir)
                        elif not trajname_to_path:
                            logger.warning(
                                "No trajectory-to-path map under %s; building the funnel "
                                "for %s from all sMD trajectories.",
                                sMD_analysis_outdir, group_key)
                        else:
                            _wanted = {os.path.basename(self._trajname_to_dcd(n))
                                       for n, p in trajname_to_path.items() if p == group_key}
                            _group_trajs = [t for t in all_smd_trajs
                                            if os.path.basename(t) in _wanted]
                            if _group_trajs:
                                smd_trajs = _group_trajs
                                logger.info(
                                    f"Funnel for {group_key}: {len(smd_trajs)} of "
                                    f"{len(all_smd_trajs)} aligned sMD trajectories.")
                            else:
                                logger.warning(
                                    "No aligned sMD trajectory maps to %s; building the "
                                    "funnel from all sMD trajectories.", group_key)

                        if not smd_trajs:
                            logger.warning("No sMD trajectories found. Skipping funnel potential generation.")
                        else:
                            universe = mda.Universe(solvated_system_pdb, smd_trajs)
                            logger.info(f"Loaded {len(universe.trajectory)} frames total")

                            # Funnel host: the axis must pass through the site (bound r_xy ~ 0).
                            # pocket_selection may be wide with an off-site COM, so default to
                            # CAs of residues contacting the equilibrated ligand.
                            if self.mMD_funnel_host_selection is not None:
                                funnel_host_sel = self.mMD_funnel_host_selection
                                logger.info(f"Using user-specified funnel host selection: {funnel_host_sel}")
                            else:
                                u_ref = mda.Universe(equilibrated_pdb)
                                contact_ca = u_ref.select_atoms(
                                    f"protein and name CA and same residue as "
                                    f"(around 6 ({ligand_selection}))"
                                )
                                if len(contact_ca) == 0:
                                    logger.warning(
                                        "No protein CA found within 6 Å of ligand in the "
                                        "equilibrated structure. Falling back to pocket_selection "
                                        "for funnel host — the funnel may be misaligned."
                                    )
                                    funnel_host_sel = f"index {' '.join(map(str, pocket_atom_indices))}"
                                else:
                                    contact_resids = sorted(set(contact_ca.resids))
                                    funnel_host_sel = (
                                        f"protein and name CA and resid "
                                        f"{' '.join(map(str, contact_resids))}"
                                    )
                                    logger.info(
                                        f"Auto-derived funnel host: {len(contact_ca)} CA atoms of "
                                        f"residues within 6 Å of ligand (resids: "
                                        f"{contact_resids[:5]}{'...' if len(contact_resids) > 5 else ''}). "
                                        f"Pass mMD_funnel_host_selection= to override."
                                    )

                            funnel_params = generate_funnel_parameters_from_trajectory(
                                universe,
                                host_selection=funnel_host_sel,
                                guest_selection=ligand_selection,
                                # z_cc must lie beyond the binding site so the cone covers it;
                                # R_cylinder comes from the radial spread beyond z_cc.
                                # z_cc_ang=5, # overrides the automatic z_cc (floor 5 A)
                                alpha_cone_degrees=45.0,
                                bulk_cutoff_ang=self.mMD_bulk_cutoff_ang,
                                verbose=True,
                            )

                            _zmax = float(funnel_params["z_max"].value_in_unit(
                                openmmunit.angstrom))
                            _zcc = float(funnel_params["z_cc"].value_in_unit(
                                openmmunit.angstrom))
                            if self.mMD_cv == "path":
                                # derive the path's far end from the wall so the two caps agree
                                _target_nm = com_distance_at_z(
                                    read_positions(path_milestones[-1]) * 10.0,
                                    ligand_atoms_indices, pocket_atom_indices,
                                    funnel_params["unbinding_axis"], _zmax)
                                _pos = read_positions(path_milestones[-1])
                                _mass = lambda idx: np.array([
                                    base_system.getParticleMass(int(i)).value_in_unit(openmmunit.dalton)
                                    for i in idx])
                                _last_nm = float(np.linalg.norm(
                                    np.average(_pos[ligand_atoms_indices], axis=0,
                                               weights=_mass(ligand_atoms_indices))
                                    - np.average(_pos[pocket_atom_indices], axis=0,
                                                 weights=_mass(pocket_atom_indices))))
                                if _target_nm > _last_nm:
                                    logger.info(
                                        "Extending the path to the cylinder cap: z_max="
                                        "%.2f A -> %.3f nm COM separation.",
                                        _zmax, _target_nm)
                                    path_cv = pathCV_cv(
                                        topology=topology,
                                        milestones=path_milestones,
                                        pocket_atoms=pocket_atom_indices,
                                        ligand_atoms=ligand_atoms_indices,
                                        system=base_system,
                                        grid_min=0.0, grid_max=1.0,
                                        hill_width=self.mMD_hill_width,
                                        sigma='auto',
                                        extend_path_to_nm=_target_nm,
                                    )
                                else:
                                    logger.info(
                                        "Last milestone (%.3f nm) already reaches the "
                                        "cylinder cap (%.3f nm); path not extended.",
                                        _last_nm, _target_nm)
                                _span = path_cv.cv_length_per_unit_ang
                                if _span is not None:
                                    # Wall and CV range are independent caps; compare them in
                                    # funnel z (s=1 must lie in the cylinder, not past the wall).
                                    try:
                                        _ax = np.asarray(funnel_params["unbinding_axis"],
                                                         dtype=float)
                                        _ax = _ax / np.linalg.norm(_ax)
                                        _p0 = read_positions(path_milestones[0]) * 10.0
                                        _z_first = float(np.dot(
                                            _p0[ligand_atoms_indices].mean(axis=0)
                                            - _p0[pocket_atom_indices].mean(axis=0), _ax))
                                        _z_path_far = _z_first + _span
                                    except Exception as _geom_err:
                                        _z_path_far = None
                                        logger.debug("path far z unavailable: %s", _geom_err)
                                    logger.info(
                                        "Reference geometry: cylinder %.1f-%.1f A "
                                        "(length %.1f A); path spans %.1f A%s.",
                                        _zcc, _zmax, _zmax - _zcc, _span,
                                        f" reaching z={_z_path_far:.1f} A"
                                        if _z_path_far is not None else "")
                                    if _z_path_far is not None and _z_path_far > _zmax + 2.0:
                                        logger.warning(
                                            "The path reaches z=%.1f A but the wall stops "
                                            "the guest at %.1f A, so the far end of the CV "
                                            "is unreachable and s=1 will never be sampled. "
                                            "Extend the path to the wall, not past it.",
                                            _z_path_far, _zmax)
                                    elif _z_path_far is not None and _z_path_far < _zcc:
                                        logger.warning(
                                            "The path ends at z=%.1f A, inside the cone "
                                            "(z_cc=%.1f A), so s=1 is not bulk however long "
                                            "the cylinder is.", _z_path_far, _zcc)
                                metad_cv = path_cv
                            else:
                                # Bias the funnel axis coordinate itself: it keeps changing up to the
                                # cap and is the length Eq. 15 integrates over, so no path extension.
                                # The funnel is symmetric in z, so the grid spans the mirrored
                                # cone too: a clamp at z=0 would be a hard edge at the bound site.
                                _zlo = -_zcc
                                metad_cv = funnel_z_cv(
                                    funnel_params["host_index"], funnel_params["guest_index"],
                                    funnel_params["unbinding_axis"],
                                    grid_min=_zlo / 10.0, grid_max=_zmax / 10.0,
                                    hill_width=self.mMD_hill_width,
                                    grid_points=int(round(124 * (_zmax - _zlo) / _zmax)) + 1,
                                )
                                _pos0 = read_positions(path_milestones[0]) * 10.0
                                _hm = [int(i) for i in funnel_params["host_index"]]
                                _gm = [int(i) for i in funnel_params["guest_index"]]
                                _w = lambda idx: np.array([
                                    base_system.getParticleMass(i).value_in_unit(openmmunit.dalton) for i in idx])
                                _ax0 = np.asarray(funnel_params["unbinding_axis"], dtype=float)
                                _z_bound = float(np.dot(
                                    np.average(_pos0[_gm], axis=0, weights=_w(_gm))
                                    - np.average(_pos0[_hm], axis=0, weights=_w(_hm)), _ax0 / np.linalg.norm(_ax0)))
                                logger.info(
                                    "Biasing the funnel axis coordinate z on [%.2f, %.2f] A; bound milestone at "
                                    "z=%.2f A.", _zlo, _zmax, _z_bound)
                                if _z_bound > 0.2 * _zmax:
                                    logger.warning(
                                        "The bound milestone (z=%.2f A) lies beyond 20%% of the way to the cap, "
                                        "which the analysis treats as the bound basin.", _z_bound)
                            funnel_force = create_funnel_force_from_trajectory_analysis(funnel_params)
                            logger.info("Funnel potential successfully generated from trajectories")

                            # persist for post-hoc correction / visualisation
                            save_funnel_params(
                                funnel_params,
                                os.path.join(group_out_dir, "funnel_params"),
                            )

                            # Write debug PSE showing funnel geometry
                            try:
                                write_funnel_pymol(
                                    funnel_params=funnel_params,
                                    reference_pdb=solvated_system_pdb,
                                    milestone_files=path_milestones,
                                    outdir=group_out_dir,
                                    ligand_selection=ligand_selection,
                                )
                            except Exception as viz_e:
                                logger.warning(f"Funnel visualization failed (non-fatal): {viz_e}")

                    except Exception as e:
                        logger.error(f"Error generating funnel potential: {e}")
                        logger.warning("Continuing without funnel potential")
                        funnel_force = None

                # most-bound milestone, resolved before the walker order is applied
                first_milestone_name = os.path.basename(path_milestones[0]).split('.')[0]
                first_milestone_state = f"{group_milestones_dir}/{first_milestone_name}_relax_state.xml"
                walker_run_info = {}  # {run_id: (state_xml_path, system_xml_path)} for stuck-walker retry

                _walker_order = walker_seed_order(path_milestones,
                                                  self.mMD_seed_order)
                logger.info("Walker seed order (%s): %s", self.mMD_seed_order,
                            [os.path.basename(m).split('.')[0] for m in _walker_order])
                for milestone in _walker_order:
                    milestone_name = os.path.basename(milestone).split('.')[0]
                    milestone_system_xml = f"{group_milestones_dir}/{milestone_name}_relax_system.xml"
                    milestone_state = f"{group_milestones_dir}/{milestone_name}_relax_state.xml"

                    if os.path.exists(milestone_system_xml):
                        milestone_system = load_system(milestone_system_xml)
                        logger.info(f"Loading relaxed milestone {milestone_name}")
                    else:
                        logger.info(f'Relaxing milestone {milestone_name}')
                        try:
                            system = load_system(f"{sys_name}/system.xml")
                            milestone_system = milestone_relax.run(system=system, state_xml_file=milestone, run_id=milestone_name)
                        except Exception as e:
                            logger.error(f"Error relaxing {milestone_name}: {e}")
                            continue

                    # Use the serialized state, not the checkpoint: metadynamics adds forces,
                    # so the Context differs from the relax one.
                    state_to_use = milestone_state if self.mMD_milestone_seeding else first_milestone_state

                    # isolated walkers get their own bias subdir; None = shared group biasDir
                    walker_bias_dir = (
                        os.path.join(group_out_dir, f"bias_{milestone_name}")
                        if not self.mMD_multiple_walkers else None
                    )

                    # isolated mode: copy the preseed into the walker's own biasDir
                    if walker_bias_dir is not None and self.mMD_preseed_bias:
                        preseed_src = os.path.join(group_out_dir, "bias_0_0.npy")
                        if os.path.exists(preseed_src):
                            os.makedirs(walker_bias_dir, exist_ok=True)
                            shutil.copy2(preseed_src, os.path.join(walker_bias_dir, "bias_0_0.npy"))

                    logger.info(f"Running WTMetaD for milestone {milestone_name}")
                    try:
                        WTMetaD.run(
                            state_xml_file=state_to_use,
                            system=milestone_system,
                            run_id=milestone_name,
                            cv_specs=[metad_cv],
                            mMD_time=self.mMD_time,
                            bias_factor=self.mMD_bias_factor,
                            hill_height=self.mMD_hill_height,
                            biasFrequency=self.mMD_bias_frequency,
                            funnel_force=funnel_force,
                            funnel_params=funnel_params if funnel_force is not None else None,
                            bias_dir=walker_bias_dir,
                        )
                        walker_run_info[milestone_name] = (state_to_use, milestone_system_xml)
                    except Exception as e:
                        logger.error(f"Error during WTMetaD for {milestone_name}: {e}")
                        continue

                # Retry walkers that never got halfway from the bound end, on the accumulated
                # bias (shared, or the walker's own in isolated mode).
                stuck_walkers = [
                    (run_id, state_xml, xml)
                    for run_id, (state_xml, xml) in walker_run_info.items()
                    if os.path.exists(f"{group_out_dir}/COLVAR_{run_id}.npy")
                    and float(np.load(f"{group_out_dir}/COLVAR_{run_id}.npy")[:, 0].max())
                    < metad_cv.bound_end + 0.5 * (metad_cv.grid_max - metad_cv.bound_end)
                ]
                if stuck_walkers:
                    logger.info(
                        f"{len(stuck_walkers)} stuck walker(s) detected "
                        f"({[r for r, _, _ in stuck_walkers]}); retrying with accumulated bias."
                    )
                    for run_id, state_xml, xml in stuck_walkers:
                        retry_bias_dir = (
                            os.path.join(group_out_dir, f"bias_{run_id}")
                            if not self.mMD_multiple_walkers else None
                        )
                        try:
                            WTMetaD.run(
                                state_xml_file=state_xml,
                                system=load_system(xml),
                                run_id=f"{run_id}_retry",
                                cv_specs=[metad_cv],
                                mMD_time=self.mMD_time,
                                bias_factor=self.mMD_bias_factor,
                                hill_height=self.mMD_hill_height,
                                biasFrequency=self.mMD_bias_frequency,
                                funnel_force=funnel_force,
                                funnel_params=funnel_params if funnel_force is not None else None,
                                bias_dir=retry_bias_dir,
                            )
                        except Exception as e:
                            logger.error(f"Retry failed for {run_id}: {e}")

                # Post-run diagnosis: FES, walker stats, and ΔG°_b from the coverage-filtered FES.
                try:
                    colvar_files_diag = sorted(glob(f"{group_out_dir}/COLVAR_*.npy"))
                    if colvar_files_diag:
                        ma = MetadynamicsAnalysis(out_dir=group_out_dir)
                        diag = ma.diagnose(
                            colvar_files=colvar_files_diag,
                            temperature=float(self.temperature),
                            bias_factor=float(self.mMD_bias_factor),
                            grid_min=metad_cv.grid_min,
                            grid_max=metad_cv.grid_max,
                            cv_name=("PathCV progress (s)" if metad_cv.name == "pathCV_progress"
                                     else "funnel z (nm)"),
                            out_prefix=os.path.join(group_out_dir, sys_name),
                            cv_length_per_unit_ang=metad_cv.cv_length_per_unit_ang,
                            cv_is_funnel_z=metad_cv.name == "funnel_z",
                            hill_width=metad_cv.hill_width,
                        )
                        if diag.get('dG_bind_std_kcal') is not None:
                            logger.info(
                                f"Final ΔG°_b = {diag['dG_bind_std_kcal']:.2f} kcal/mol  "
                                f"pKd = {diag['pKd']:.2f}  "
                                f"({diag['n_converged_cv']}/{diag['n_walkers']} walkers converged)"
                            )
                except Exception as _diag_err:
                    logger.warning(f"Post-run metadynamics diagnosis failed (non-fatal): {_diag_err}")

        # Load and align the WTMetaD trajectories (per-path runs write into subdirs)
        WTMetaD_trajs = glob(f"{mMD_out_dir}/**/trajectory_metadynamics_milestone_*_frame_*.dcd",
                             recursive=True)
        WTMetaD_trajs = [f for f in WTMetaD_trajs if "aligned" not in f]  # only process unaligned trajectories

        wrap_align_save_traj(WTMetaD_trajs, solvated_system_pdb, is_membrane=self.is_membrane)

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished AutoPath simulation in {simulation_time/60:.2f} min.")

        return

    def _check_speed(self, outdir, sys_name, lig_sel, min_replicas, speed=None,
                     speeds=None):
        """Build the analysis and run the convergence check for one speed.

        ``speed=None`` and ``speeds=None`` check every speed on disk jointly (the
        force ladder); ``speeds=[...]`` restricts the joint check to a subset.
        `min_replicas` is explicit because sMD_pulling_speeds values may be None.
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
                                  ligand_select=lig_sel)
        conv_df, traces_df = smdanalysis.check_convergence(
            logs=logs, speeds=speeds,
            min_replicas=min_replicas,
            estimator_name=self.sMD_autostop_estimator,
            conv_window=self.sMD_conv_window,
            geom_features=bool(self.sMD_log_geom_features),
            plateau_frac=self.sMD_plateau_frac,
            cluster_to_boundary=self.sMD_cluster_to_boundary,
            restrict_rmsd_to_boundary=self.sMD_cluster_to_boundary,
            boundary_buffer_frac=self.sMD_boundary_buffer_frac,
        )
        return conv_df, traces_df

    @staticmethod
    def _replica_idx_from_log(fn):
        """Extract the integer replica index from an sMD log filename.

        Expected filename pattern:
        ``sMD_replica-<N>_<speed>_<direction>.dat``

        Parameters
        ----------
        fn : str
            Full or basename path to the sMD ``.dat`` log file.

        Returns
        -------
        int
            Zero-based replica index embedded in the filename.
        """
        base = os.path.basename(fn)[:-4]
        rep = base.split("_")[-3]
        return int(rep.split("-")[1])
    
    
    def _trajname_to_dcd(self, trajname: str) -> str:
        """Resolve the aligned DCD path for a medoid trajectory name.

        Medoid names are stored as log-file basenames (without extension) by
        :class:`~autopath.pulling.PathModel.DTWPathModel`.  This method maps
        them to the corresponding aligned DCD file under
        ``<sMD_outdir>/trajectories/``.

        Parameters
        ----------
        trajname : str
            Log-file basename without extension (e.g.
            ``"sMD_replica-3_v0.001_forward"``).

        Returns
        -------
        str
            Absolute or relative path to ``<sMD_outdir>/trajectories/<name>_aligned.dcd``.
        """
        dcd_name = trajname.replace("log", "traj") + "_aligned.dcd"
        return os.path.join(self.sMD_outdir, "trajectories", dcd_name)