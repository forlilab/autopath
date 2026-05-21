import os
import math
import time
import logging
import numpy as np
from datetime import datetime

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import cvpack
from autopath.cv import CVSpec
from autopath.utils import *
from autopath.customForces import *
from autopath.analysis import (
    plot_bias,
    plot_colvar,
    plot_FE,
    plot_FE_rw,
    plot_FE_2D,
    plot_colvar_2D,
    correct_fe_for_funnel,
)

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    girsanov = False
    logging.warning("Please install openmmtools to use Girsanov reweighting.")

class MetadynamicsMD:

    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        is_membrane: bool = False,
        timestep: float = 0.004, #  4 fs timestep
        temp: float = 300,
        use_GReweighting: bool = False,
        ligand_resname: str = "UNK",
        platform: str = "fastest",
        out_dir: str = "metadynamics",
        verbose: bool = True,
    ) -> None:
        
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.topology = topology
        self.n_atoms = self.topology.getNumAtoms()
        self.is_membrane = is_membrane

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temp * openmmunit.kelvin

        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        self.ligand_resname = ligand_resname
        
        self.restrained_atoms = restrained_atoms

        self.platform = select_platform(platform)

        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            if not girsanov:
                logging.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logging.info("Using Girsanov reweighting for steered MD.")
            
        # These are for debugging purposes if one wants to check the CVs over the time of the simulation
        self.verbose = verbose
        self.record_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 10)  # record the CVs every 10 ps
        self.store_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 100)  # log the stored COLVAR every 100ps

        return

    def run(
        self,
        pdb_file: str = None,
        system: str = None,
        checkpoint_file: str = None,
        run_id: str = None,
        cv_specs: list[CVSpec] = None,
        mMD_time: int = 10,
        bias_factor: float = 10,
        hill_height: float = 1.2,  # kJ/mol approx 0.5 KbT
        biasFrequency: int = 2,
        saveFrequency: int = 50,
        funnel_force: Force = None,
        funnel_params: dict = None,
    ) -> str:

        start_time = time.monotonic()

        assert cv_specs is not None and len(cv_specs) > 0, \
            "cv_specs must be a non-empty list of CVSpec objects. " \
            "Use autopath.cv factory functions (e.g. com_cv, rmsd_cv) to build them."

        # ensure proper formatting of the file name
        if run_id is None:
            run_id = f'W-{datetime.now().strftime("%H%M%S")}'

        # Calculate the number of steps required
        mMD_steps = math.ceil(mMD_time / self.timestep.value_in_unit(openmmunit.picoseconds) * 1000.0)
        biasFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * biasFrequency)
        saveFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * saveFrequency)

        hill_height = hill_height * openmmunit.kilojoules_per_mole

        cv_names = ", ".join(s.name for s in cv_specs)
        logging.info(f"Running metadynamics with CV(s): {cv_names}")

        logging.debug("Setting up the integrator..")
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = biasFrequency,   # 500 is 2ps at 4fs timestep
                temperature = self.temperature,
                collision_rate = 1.0/openmmunit.picoseconds,
                timestep = self.timestep,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            integrator = LangevinMiddleIntegrator(self.temperature, 
                                                  1.0/openmmunit.picoseconds, 
                                                  self.timestep)

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is None and pdb_file is None:
            logging.error("Either pdb_file or checkpoint_file must be provided to set initial positions.")
            exit(1)
        elif checkpoint_file is None and pdb_file is not None:
            logging.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)

        elif checkpoint_file is not None and pdb_file is None:
            logging.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator  # Replace the integrator with the new one
        else:
            # if both are provided, use the checkpoint file but warn the user
            logging.warning("Both checkpoint_file and pdb_file are provided. Using checkpoint_file.")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator

        # Add harmonic positional restraints to protein CA
        input_positions = simulation.context.getState(getPositions=True).getPositions()

        if self.restrained_atoms is not None:
            add_harmonic_restraints(
                system,
                input_positions,
                self.topology,
                self.restrained_atoms,
                10,
                "k_CA",
                14,
            )

        # Add funnel potential if provided.
        # system.addForce() transfers C++ ownership of the force, so reusing the same
        # funnel_force object across multiple run() calls would leave it non-owning.
        # Serializing and deserializing gives a fresh owned copy each time.
        if funnel_force is not None:
            fresh_funnel = XmlSerializer.deserialize(XmlSerializer.serialize(funnel_force))
            system.addForce(fresh_funnel)
            logging.info(f"Added funnel potential with force group {funnel_force.getForceGroup()}")

        logging.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"WTMetaD_{run_id}",
            mMD_steps,
            biasFrequency,
        )
        if self.use_GReweighting:
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/GR_WTMetaD_{run_id}.dat", 
                                                            biasFrequency, 
                                                            integrator, 
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))
            
        # Resolve deferred CVs (those that need input_positions to build)
        resolved = [spec.resolve(input_positions, self.n_atoms, self.topology) for spec in cv_specs]

        bias_variables = [
            BiasVariable(
                s.cv,
                minValue=s.grid_min,
                maxValue=s.grid_max,
                biasWidth=s.hill_width,
                gridWidth=s.grid_points,
                periodic=s.periodic,
            )
            for s in resolved
        ]

        # Set up the metadynamics object
        meta = Metadynamics(
            system,
            bias_variables,
            self.temperature,
            bias_factor,
            hill_height,
            frequency=biasFrequency,
            saveFrequency=saveFrequency,
            biasDir=self.out_dir,
        )

        # meta._force.setForceGroup(1)  # force group 1 for reweighting girsanov
        
        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setStepCount(0)  # reset step count

        simulation.context.reinitialize(preserveState=True)

        colvar_array = np.array([meta.getCollectiveVariables(simulation)])
        for i in range(0, int(mMD_steps), self.record_CV):
            if self.verbose and i % self.store_CV == 0:
                np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)

            meta.step(simulation, self.record_CV)
            current_cvs = meta.getCollectiveVariables(simulation)
            colvar_array = np.append(colvar_array, [current_cvs], axis=0)

        np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)

        max_cv = float(colvar_array[:, 0].max())
        if max_cv < 0.5:
            logging.warning(
                f"Walker {run_id}: CV never exceeded 0.5 (max={max_cv:.3f}). "
                "The ligand did not reach the unbound state — this walker will distort the FE profile. "
                "Consider increasing mMD_time or mMD_hill_height."
            )

        fe_path = os.path.join(self.out_dir, f"FE_{run_id}.npy")
        np.save(fe_path, meta.getFreeEnergy())

        # Apply funnel standard-state correction if funnel was used
        if funnel_params is not None:
            try:
                temp_K = self.temperature.value_in_unit(openmmunit.kelvin)
                result = correct_fe_for_funnel(
                    fe_path,
                    funnel_params,
                    temperature=temp_K,
                )
                rw_path = os.path.join(self.out_dir, f"FE_{run_id}_rw.npy")
                np.save(rw_path, result["fe_corrected"])
                logging.info(
                    f"[{run_id}] ΔG_sim={result['dG_bind_sim_kj_mol']:.2f} kJ/mol  "
                    f"correction={result['correction_kj_mol']:.2f} kJ/mol  "
                    f"ΔG°_b={result['dG_bind_std_kj_mol']:.2f} kJ/mol  "
                    f"pKd={result['pKd']:.2f}"
                )
            except Exception as e:
                logging.warning(f"Funnel standard-state correction failed for {run_id}: {e}")

        # Create plots
        if len(resolved) == 1:
            s = resolved[0]
            plot_colvar(self.out_dir, s.name)
            plot_bias(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name)
            plot_FE(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name)
            if funnel_params is not None:
                plot_FE_rw(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name)
        else:
            a, b = resolved[0], resolved[1]
            plot_colvar_2D(self.out_dir, a.name, b.name)
            plot_FE_2D(
                self.out_dir,
                a.grid_min, a.grid_max, a.grid_points, a.name,
                b.grid_min, b.grid_max, b.grid_points, b.name,
            )

        # Save everything
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_system(system, f"{self.out_dir}/WTMetaD_system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/WTMetaD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/WTMetaD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return run_id

    def run2D(self, *args, **kwargs):
        raise NotImplementedError(
            "run2D() has been removed. Pass two CVSpec objects via cv_specs to run() instead. "
            "Example:\n"
            "  from autopath.cv import rmsd_cv\n"
            "  cv_a = rmsd_cv(...)\n"
            "  cv_b = rmsd_cv(...)\n"
            "  metad.run(cv_specs=[cv_a, cv_b], ...)"
        )

    # ── Diagnosis helpers ──────────────────────────────────────────────────────
    #
    # OpenMM's Metadynamics saves each walker's OWN deposited Gaussians
    # (selfBias) to bias_<id>_<gridSize>.npy.  The FE estimate from
    # getFreeEnergy() uses totalBias = selfBias + sum(all loaded files).
    # Because each file contains only one walker's contribution, you can
    # reconstruct any subset FES with:
    #   FE(s) = -(gamma / (gamma-1)) * sum(selected_bias_arrays)
    #
    # NOTE: PLUMED HILLS files (individual Gaussians with timestamps) cannot
    # be reconstructed from these gridded sums.  All diagnostics below work
    # directly with OpenMM's format.

    @staticmethod
    def load_bias_files(bias_dir: str) -> dict:
        """Load all walker self-bias arrays from bias_dir.

        Returns {walker_id (int): bias_array (np.ndarray, kJ/mol)}.
        Only the latest snapshot for each walker is loaded (OpenMM
        overwrites with a higher save-index at each sync interval).
        """
        import re
        pattern = re.compile(r'bias_(\d+)_(\d+)\.npy')
        latest = {}  # walker_id → (save_index, filepath)
        for fname in sorted(os.listdir(bias_dir)):
            m = pattern.match(fname)
            if m:
                wid, idx = int(m.group(1)), int(m.group(2))
                if wid not in latest or idx > latest[wid][0]:
                    latest[wid] = (idx, os.path.join(bias_dir, fname))
        return {wid: np.load(fp) for wid, (_, fp) in latest.items()}

    @staticmethod
    def compute_fes(bias_arrays, temperature: float, bias_factor: float,
                    grid_min: float = 0.0, grid_max: float = 1.0) -> tuple:
        """Compute the WT-MetaD FES from a collection of bias arrays.

        Uses the standard estimator:
            FE(s) = -(gamma / (gamma − 1)) × V_total(s)
        where V_total = sum of all provided arrays.  Pass a subset of
        walkers' bias arrays to get a FES that excludes stuck walkers.

        Parameters
        ----------
        bias_arrays : list/array of np.ndarray
            Self-bias arrays in kJ/mol from load_bias_files().
        temperature : float
            Simulation temperature in K (not used in the formula but
            kept for API completeness / future extensions).
        bias_factor : float
            WT-MetaD bias factor γ.
        grid_min, grid_max : float
            CV grid bounds — used only to build the returned axis.

        Returns
        -------
        axis : np.ndarray   — CV values at each grid point
        fes  : np.ndarray   — Free energy in kJ/mol
        """
        V = np.sum(np.stack(bias_arrays), axis=0)
        gamma = float(bias_factor)
        fes = -(gamma / (gamma - 1.0)) * V
        axis = np.linspace(grid_min, grid_max, fes.shape[0])
        return axis, fes

    @staticmethod
    def analyze_colvar_transitions(colvar_array: np.ndarray,
                                    bound_threshold: float = 0.1,
                                    unbound_threshold: float = 0.9) -> dict:
        """Detect basin transitions in a 1D CV time series.

        A bound→unbound transition is counted each time the CV rises above
        unbound_threshold after having been below bound_threshold.  A round
        trip additionally requires a return below bound_threshold.

        Returns
        -------
        dict with keys:
            n_transitions        — bound→unbound crossings
            n_round_trips        — complete B→U→B cycles
            first_transition_frame — frame index of first B→U crossing (-1 = never)
            max_cv               — maximum CV value reached
            frac_bound           — fraction of frames in bound basin
            frac_unbound         — fraction of frames in unbound basin
            reached_unbound      — bool: crossed unbound_threshold at least once
        """
        cv = colvar_array[:, 0] if np.ndim(colvar_array) > 1 else np.asarray(colvar_array)
        n_trans, n_rt, first_frame = 0, 0, -1

        # Determine starting basin.  A walker seeded from an unbound sMD
        # milestone starts with CV≈1; we set in_unbound=True so that the
        # first return to the bound state is correctly counted as a round trip.
        in_unbound = cv[0] > unbound_threshold
        in_bound   = cv[0] < bound_threshold

        for i, v in enumerate(cv):
            if not in_unbound and v > unbound_threshold:
                n_trans += 1
                if first_frame < 0:
                    first_frame = i
                in_bound   = False
                in_unbound = True
            elif in_unbound and v < bound_threshold:
                n_rt += 1
                in_unbound = False
                in_bound   = True

        # A walker that starts unbound (seeded from a late sMD milestone) counts
        # as having "reached" the unbound state even with zero explicit transitions.
        started_unbound = bool(cv[0] > unbound_threshold)
        return {
            'n_transitions':          n_trans,
            'n_round_trips':          n_rt,
            'first_transition_frame': first_frame,
            'max_cv':                 float(cv.max()),
            'frac_bound':             float(np.mean(cv < bound_threshold)),
            'frac_unbound':           float(np.mean(cv > unbound_threshold)),
            'reached_unbound':        bool(cv.max() > unbound_threshold),
            'started_unbound':        started_unbound,
        }

    @staticmethod
    def bias_coverage(bias_array: np.ndarray, threshold_frac: float = 1e-3) -> float:
        """Fraction of CV grid bins with non-negligible bias.

        A stuck walker deposits hills only near CV=0, leaving most of the
        grid at zero.  This metric is a proxy for convergence that can be
        computed from bias files alone, without needing COLVAR data.
        """
        if bias_array.max() <= 0:
            return 0.0
        return float(np.mean(bias_array > threshold_frac * bias_array.max()))

    def select_converged_biases(self, min_coverage: float = 0.6) -> list:
        """Return bias arrays from walkers that covered ≥ min_coverage of the CV grid.

        Stuck walkers (deposited hills only in the bound basin) have low
        grid coverage and are excluded.  This is the bias-file proxy for
        the transition criterion when COLVAR data is not available.

        Returns
        -------
        list of (walker_id, bias_array) tuples that passed the threshold.
        """
        biases = self.load_bias_files(self.out_dir)
        selected = []
        for wid, barr in sorted(biases.items()):
            cov = self.bias_coverage(barr)
            if cov >= min_coverage:
                selected.append((wid, barr))
            else:
                logging.warning(
                    f"Walker id={wid}: bias coverage {100*cov:.1f}% < "
                    f"{100*min_coverage:.0f}% — likely stuck, excluded from converged set."
                )
        if not selected:
            logging.warning("No walkers passed the coverage threshold — returning all biases.")
            selected = list(sorted(biases.items()))
        return selected

    def diagnose(
        self,
        colvar_files: list,
        temperature: float,
        bias_factor: float,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        cv_name: str = "PathCV_progress",
        bound_threshold: float = 0.1,
        unbound_threshold: float = 0.9,
        coverage_threshold: float = 0.6,
        out_prefix: str = None,
    ) -> dict:
        """Comprehensive multi-walker WT-MetaD diagnostic.

        Produces a four-panel PNG figure (PLUMED-community style) and returns
        a summary dict.  The four panels are:

        1. CV time traces — each walker, annotated with transition counts.
           Solid lines = reached unbound state; dashed = stuck.
        2. Self-bias per walker — what each walker actually deposited on the
           CV grid (only its own Gaussians, not the total).
        3. Per-walker FES — FES computed from each walker's self-bias alone
           (normalized min=0).  Reveals whether a single walker gives a
           physically reasonable shape.
        4. Combined vs converged-only FES — FES from all biases summed vs
           FES from walkers that passed the coverage threshold, both
           normalized to zero at the unbound plateau.  ΔG estimates are
           annotated directly on the plot.

        Parameters
        ----------
        colvar_files : list of str
            Paths to COLVAR_*.npy files (one per walker).  Not matched 1:1
            to bias files (OpenMM uses random IDs); used only for CV traces
            and transition statistics.
        temperature : float
            Simulation temperature in K.
        bias_factor : float
            WT-MetaD bias factor γ.
        grid_min, grid_max : float
            CV grid bounds.
        cv_name : str
            Label for the CV axis.
        bound_threshold, unbound_threshold : float
            CV values defining bound and unbound basins.
        coverage_threshold : float
            Minimum grid coverage fraction to classify a walker as converged
            (used to select biases for the converged-only FES estimate).
        out_prefix : str, optional
            Output filename prefix.  Defaults to <out_dir>/diagnosis.

        Returns
        -------
        dict with keys:
            walker_stats    — {run_id: analyze_colvar_transitions result}
            fes_all_kcal    — FES from all walkers (kcal/mol, unbound=0)
            fes_conv_kcal   — FES from coverage-filtered walkers (kcal/mol, unbound=0)
            cv_axis         — grid axis
            n_converged_cv  — walkers that reached unbound_threshold
            n_walkers       — total walkers in colvar_files
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if out_prefix is None:
            out_prefix = os.path.join(self.out_dir, "diagnosis")

        # ── Load & analyse ─────────────────────────────────────────────────
        all_biases = self.load_bias_files(self.out_dir)

        walker_data = []
        for colvar_path in sorted(colvar_files):
            colvar = np.load(colvar_path)
            run_id = os.path.splitext(os.path.basename(colvar_path))[0].replace('COLVAR_', '')
            ts = self.analyze_colvar_transitions(colvar, bound_threshold, unbound_threshold)
            walker_data.append((run_id, colvar, ts))

        walker_stats = {run_id: ts for run_id, _, ts in walker_data}
        n_converged_cv = sum(1 for ts in walker_stats.values() if ts['reached_unbound'])
        converged_biases = self.select_converged_biases(coverage_threshold)

        # ── Compute FES variants ───────────────────────────────────────────
        all_bias_list = list(all_biases.values())
        axis, fes_all_kj = self.compute_fes(all_bias_list, temperature, bias_factor,
                                              grid_min, grid_max)
        n = len(axis)

        def _norm_plateau(fes_kj):
            plateau = fes_kj[int(0.85 * n):].mean()
            return (fes_kj - plateau) * 0.239006  # kcal/mol, unbound≈0

        fes_all_kcal = _norm_plateau(fes_all_kj)

        fes_conv_kcal = None
        if len(converged_biases) < len(all_biases):
            _, fes_conv_kj = self.compute_fes(
                [b for _, b in converged_biases], temperature, bias_factor, grid_min, grid_max
            )
            fes_conv_kcal = _norm_plateau(fes_conv_kj)

        # ── Figure ─────────────────────────────────────────────────────────
        n_w = len(walker_data)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_w, 1)))

        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)

        # Panel 1 — CV traces
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axhspan(0, bound_threshold, alpha=0.07, color='royalblue')
        ax1.axhspan(unbound_threshold, 1.0, alpha=0.07, color='tomato')
        ax1.axhline(bound_threshold,   color='royalblue', lw=0.7, ls='--', alpha=0.5)
        ax1.axhline(unbound_threshold, color='tomato',    lw=0.7, ls='--', alpha=0.5)
        for k, (run_id, colvar, ts) in enumerate(walker_data):
            cv = colvar[:, 0] if np.ndim(colvar) > 1 else colvar
            converged = ts['reached_unbound']
            label = (f"{run_id}  T={ts['n_transitions']} RT={ts['n_round_trips']}"
                     f"  max={ts['max_cv']:.3f}")
            ax1.plot(np.arange(len(cv)), cv, color=colors[k],
                      lw=1.8 if converged else 0.9,
                      ls='-' if converged else '--',
                      alpha=0.9 if converged else 0.6,
                      label=label)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Frame #")
        ax1.set_ylabel(cv_name)
        ax1.set_title("CV traces  (T=transitions  RT=round-trips  |  solid=reached unbound  dashed=stuck)")
        ax1.legend(fontsize=8, loc='upper left', framealpha=0.85)

        # Panel 2 — Self-bias per walker
        ax2 = fig.add_subplot(gs[1, 0])
        for k, (wid, barr) in enumerate(sorted(all_biases.items())):
            cov = self.bias_coverage(barr)
            ax2.plot(axis, barr * 0.239006, color=colors[k % len(colors)], lw=1.5,
                      label=f"id={wid}  cov={100*cov:.0f}%", alpha=0.85)
        ax2.set_xlabel(cv_name)
        ax2.set_ylabel("Self-bias (kcal/mol)")
        ax2.set_title("Self-bias per walker\n(only this walker's Gaussians)")
        ax2.legend(fontsize=8)

        # Panel 3 — Per-walker FES from self-bias only
        ax3 = fig.add_subplot(gs[1, 1])
        for k, (wid, barr) in enumerate(sorted(all_biases.items())):
            cov = self.bias_coverage(barr)
            _, fes_s_kj = self.compute_fes([barr], temperature, bias_factor, grid_min, grid_max)
            fes_s = fes_s_kj * 0.239006
            fes_s -= fes_s.min()  # normalize min→0
            ax3.plot(axis, fes_s, color=colors[k % len(colors)], lw=1.5,
                      label=f"id={wid}  cov={100*cov:.0f}%", alpha=0.85)
        ax3.set_xlabel(cv_name)
        ax3.set_ylabel("FE from self-bias (kcal/mol, min=0)")
        ax3.set_title("Per-walker FES\n(self-bias only — shape reflects individual sampling)")
        ax3.legend(fontsize=8)

        # Panel 4 — Combined vs converged-only FES
        ax4 = fig.add_subplot(gs[2, :])
        dG_all = float(fes_all_kcal.min())
        ax4.plot(axis, fes_all_kcal, 'k-', lw=2.0,
                  label=f"All walkers (n={len(all_biases)})  ΔG={dG_all:.1f} kcal/mol")
        if fes_conv_kcal is not None:
            dG_conv = float(fes_conv_kcal.min())
            ax4.plot(axis, fes_conv_kcal, 'g--', lw=2.0,
                      label=f"Coverage-filtered (n={len(converged_biases)})  ΔG={dG_conv:.1f} kcal/mol")
        ax4.axhline(0, color='gray', lw=0.7, ls=':')
        ax4.axvline(bound_threshold,   color='royalblue', lw=0.7, ls='--', alpha=0.4)
        ax4.axvline(unbound_threshold, color='tomato',    lw=0.7, ls='--', alpha=0.4)
        # annotate minimum
        idx_min = int(np.argmin(fes_all_kcal))
        ax4.annotate(f"ΔG={dG_all:.1f}", xy=(axis[idx_min], dG_all),
                      xytext=(axis[idx_min] + 0.1, dG_all + 1.5), fontsize=9,
                      arrowprops=dict(arrowstyle='->', lw=0.8))
        ax4.set_xlabel(cv_name)
        ax4.set_ylabel("ΔG (kcal/mol, unbound plateau = 0)")
        ax4.set_title(
            f"FES comparison — {n_converged_cv}/{n_w} walkers reached unbound state\n"
            f"Use coverage-filtered FES when walkers are compartmentalized (Raiteri 2006)"
        )
        ax4.legend(fontsize=9)

        plt.suptitle(
            f"WT-MetaD Diagnosis — {os.path.basename(self.out_dir)}",
            fontsize=12, fontweight='bold'
        )
        out_path = f"{out_prefix}_diagnosis.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Diagnostic plot saved to {out_path}")

        # ── Text summary ───────────────────────────────────────────────────
        logging.info("── Walker summary ──────────────────────────────────────")
        for run_id, _, ts in walker_data:
            status = "✓ converged" if ts['reached_unbound'] else "✗ stuck"
            logging.info(
                f"  {run_id}: {status}  T={ts['n_transitions']}  RT={ts['n_round_trips']}"
                f"  max_CV={ts['max_cv']:.3f}"
                f"  bound={100*ts['frac_bound']:.0f}%  unbound={100*ts['frac_unbound']:.0f}%"
            )
        logging.info(f"  ΔG(all walkers)     = {dG_all:.2f} kcal/mol")
        if fes_conv_kcal is not None:
            logging.info(f"  ΔG(coverage-filter) = {float(fes_conv_kcal.min()):.2f} kcal/mol")
        logging.info("────────────────────────────────────────────────────────")

        return {
            'walker_stats':   walker_stats,
            'fes_all_kcal':   fes_all_kcal,
            'fes_conv_kcal':  fes_conv_kcal,
            'cv_axis':        axis,
            'n_converged_cv': n_converged_cv,
            'n_walkers':      n_w,
        }
