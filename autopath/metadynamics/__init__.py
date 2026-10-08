"""
Well-tempered metadynamics (WT-MetaD) simulation and analysis pipeline.

This package provides tools for running, analysing, and post-processing
funnel metadynamics simulations:

- MetadynamicsMD          : run OpenMM WT-MetaD simulations
- BoundaryCorrectedMetadynamics : WT-MetaD with McGovern–de Pablo edge-corrected hills
- TransitionTemperedMetadynamics : transition-tempered MetaD (Dama 2014) on the same hills
- load_hills / bias_from_hills / fes_history : rebuild the bias and FES(t) from HILLS v2
- MetadynamicsAnalysis    : post-run diagnostics and convergence analysis
- correct_fe_for_funnel   : Limongelli 2013 standard-state correction
- write_metad_preseed     : pre-seed metadynamics bias from sMD PMF
- CVSpec / cv factories   : collective variable specifications
- Funnel utilities        : funnel parameter generation and visualization
"""

# --- Simulation ---
from .MetadynamicsMD import MetadynamicsMD
from .BoundaryCorrection import BoundaryCorrectedMetadynamics
from .TransitionTempered import TransitionTemperedMetadynamics, vstar
from .Hills import load_hills, bias_from_hills, fes_history, hills_record

# --- Analysis ---
from .Analysis import (
    MetadynamicsAnalysis,
    correct_fe_for_funnel,
    rct_weights,
    dcd_frame_of_rows,
    reweighted_fes,
    write_metad_preseed,
    plot_colvar,
    plot_bias,
    plot_FE,
    plot_FE_rw,
    plot_FE_2D,
    plot_colvar_2D,
)

# --- CV factories ---
from .CV import (
    CVSpec,
    com_cv,
    rmsd_cv,
    rmsd_states_cv,
    path_rmsd_cv,
    pathCV_cv,
    funnel_z_cv,
    contacts_cv,
)

# --- Funnel ---
from .Funnel import (
    add_funnel_restraints,
    generate_funnel_parameters_from_trajectory,
    save_funnel_params,
    load_funnel_params,
    create_funnel_force_from_trajectory_analysis,
    funnel_lower_wall_ang,
    funnel_z_grid_min_ang,
    funnel_z_bounds_nm,
    com_distance_at_z,
    z_max_for_bulk,
    write_funnel_pymol,
)

__all__ = [
    "MetadynamicsMD",
    "BoundaryCorrectedMetadynamics",
    "TransitionTemperedMetadynamics",
    "vstar",
    "load_hills",
    "bias_from_hills",
    "fes_history",
    "hills_record",
    "MetadynamicsAnalysis",
    "correct_fe_for_funnel",
    "rct_weights",
    "dcd_frame_of_rows",
    "reweighted_fes",
    "write_metad_preseed",
    "plot_colvar",
    "plot_bias",
    "plot_FE",
    "plot_FE_rw",
    "plot_FE_2D",
    "plot_colvar_2D",
    "CVSpec",
    "com_cv",
    "rmsd_cv",
    "rmsd_states_cv",
    "path_rmsd_cv",
    "pathCV_cv",
    "funnel_z_cv",
    "contacts_cv",
    "add_funnel_restraints",
    "generate_funnel_parameters_from_trajectory",
    "save_funnel_params",
    "load_funnel_params",
    "create_funnel_force_from_trajectory_analysis",
    "funnel_lower_wall_ang",
    "funnel_z_grid_min_ang",
    "funnel_z_bounds_nm",
    "com_distance_at_z",
    "z_max_for_bulk",
    "write_funnel_pymol",
]
