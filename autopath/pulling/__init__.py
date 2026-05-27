"""
Steered molecular dynamics (sMD) analysis pipeline.

This package provides tools for running, loading, and analysing steered MD
simulations:

- SteeredMD      : run OpenMM sMD simulations (single replica)
- SMDData        : load and pre-process sMD log files
- SMDAnalysis    : end-to-end analysis pipeline (clustering, PMF, k_off)
- DTWPathModel   : cluster trajectories into unbinding paths via DTW + k-medoids
- LigandTrajectoryFeatures : per-frame ligand shape descriptors
- JarzynskiEstimator / CumulantEstimator : free-energy estimators
- FrictionEstimator : friction profile Γ(r) from work dissipation
- KramersEstimator  : unbinding rate k_off via Kramers/Pontryagin MFPT
"""

# --- Simulation ---
from .steered_md import SteeredMD

# --- Data & Analysis ---
from .SMDData import SMDData
from .AnalysisSMD import SMDAnalysis

# --- Path clustering ---
from .PathModel import DTWPathModel, NullPathModel

# --- Features ---
from .LigandFeatures import LigandTrajectoryFeatures

# --- Estimators ---
from .Estimators import (
    JarzynskiEstimator,
    CumulantEstimator,
)
__all__ = [
            "SteeredMD",
            "SMDData",
            "SMDAnalysis",
            "DTWPathModel",
            "NullPathModel",
            "LigandTrajectoryFeatures",
            "JarzynskiEstimator",
            "CumulantEstimator",
        ]