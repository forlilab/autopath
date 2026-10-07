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

from autopath.metadynamics.CV import CVSpec
from autopath.metadynamics.BoundaryCorrection import BoundaryCorrectedMetadynamics
from autopath.metadynamics.Analysis import (
    correct_fe_for_funnel,
    plot_bias,
    plot_colvar,
    plot_FE,
    plot_FE_rw,
    plot_FE_2D,
    plot_colvar_2D,
)
from autopath.utils import (
    add_reporters,
    rebuild_simulation,
    save_system,
    save_simulation,
    save_pdb,
    select_platform,
)
from autopath.customForces import add_harmonic_restraints

logger = logging.getLogger(__name__)

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    girsanov = False
    logger.warning("Please install openmmtools to use Girsanov reweighting.")


class MetadynamicsMD:
    """OpenMM well-tempered metadynamics (WT-MetaD) driver supporting 1D and 2D CVs
    with optional funnel restraints.

    Parameters
    ----------
    topology : openmm.app.Topology
        System topology.
    ligand_atoms : list[int]
        Indices of ligand heavy atoms.
    pocket_atoms : list[int]
        Indices of binding-pocket reference atoms.
    restrained_atoms : list[int], optional
        Atom indices to harmonically restrain during the simulation.
    is_membrane : bool, optional
        Whether the system contains a lipid membrane (default False).
    timestep : float, optional
        Integration timestep in picoseconds (default 0.004, i.e. 4 fs).
    temp : float, optional
        Simulation temperature in Kelvin (default 300).
    use_GReweighting : bool, optional
        Enable Girsanov reweighting via openmmtools (default False).
    ligand_resname : str, optional
        Residue name of the ligand in the topology (default "UNK").
    platform : str, optional
        OpenMM platform name or "fastest" for automatic selection (default "fastest").
    out_dir : str, optional
        Directory for all output files (default "metadynamics").
    verbose : bool, optional
        Periodically save COLVAR arrays during the run (default True).

    Attributes
    ----------
    out_dir : str
        Output directory created (or confirmed) on construction.
    topology : openmm.app.Topology
        Stored topology reference.
    n_atoms : int
        Total number of atoms, cached from topology.
    timestep : openmm.unit.Quantity
        Timestep as an OpenMM Quantity (picoseconds).
    temperature : openmm.unit.Quantity
        Temperature as an OpenMM Quantity (kelvin).
    platform : openmm.Platform
        Selected OpenMM platform.
    store_CV : int
        Steps between on-disk COLVAR saves (~every 100 ps at default timestep).
    """

    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        is_membrane: bool = False,
        timestep: float = 0.004,  # 4 fs timestep
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
                logger.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logger.info("Using Girsanov reweighting for steered MD.")

        self.verbose = verbose
        self.store_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 100)

        return

    def run(
        self,
        pdb_file: str = None,
        system: str = None,
        checkpoint_file: str = None,
        state_xml_file: str = None,
        run_id: str = None,
        cv_specs: list[CVSpec] = None,
        mMD_time: int = 10,
        bias_factor: float = 10,
        hill_height: float = 1.2,  # kJ/mol approx 0.5 KbT
        biasFrequency: int = 2,
        saveFrequency: int = 50,
        funnel_force: Force = None,
        funnel_params: dict = None,
        bias_dir: str = None,
        boundary_correction: bool = True,
        reweight: bool = True,
        rct_ustride: int = 5,
        dcd_interval: float = 10.0,
    ) -> str:
        """Run a single WT-MetaD trajectory and return the run identifier.

        Parameters
        ----------
        pdb_file : str, optional
            Path to a PDB file used to set initial positions and box vectors.
        system : openmm.System
            Prepared OpenMM System object (forces already added before calling).
        checkpoint_file : str, optional
            Path to an OpenMM binary checkpoint to resume from.
        state_xml_file : str, optional
            Path to a portable OpenMM ``State`` XML. Only box, positions and
            velocities are applied (the state may carry stale global parameters).
            Precedence: ``checkpoint_file`` > ``state_xml_file`` > ``pdb_file``.
        run_id : str, optional
            Label for all output files.  Auto-generated from wall-clock time
            when ``None`` (format ``W-HHMMSS``).
        cv_specs : list[CVSpec]
            One or two ``CVSpec`` objects (e.g. from ``com_cv``); deferred
            specs are resolved once input positions are available.
        mMD_time : int, optional
            Total simulation time in nanoseconds (default 10).
        bias_factor : float, optional
            Well-tempered bias factor γ (default 10).
        hill_height : float, optional
            Initial Gaussian hill height in kJ/mol (default 1.2, ≈ 0.5 k_B T at
            300 K).
        biasFrequency : int, optional
            Gaussian hill deposition interval in picoseconds (default 2 ps).
        saveFrequency : int, optional
            Frequency of on-disk bias saves in picoseconds (default 50 ps).
        funnel_force : openmm.Force, optional
            Pre-built funnel restraint force; a serialized copy is added, since
            ``addForce()`` transfers C++ ownership.
        funnel_params : dict, optional
            Funnel parameters for the Limongelli 2013 standard-state correction
            and the funnel-axis trace.  Skipped when ``None``.
        bias_dir : str, optional
            Hill-file directory.  ``None`` uses ``self.out_dir``, shared by all
            walkers (multi-walker); a per-walker directory makes it independent.
        boundary_correction : bool, optional
            Divide each hill by its fraction inside ``CVSpec.bounds`` (default the
            grid), the McGovern–de Pablo edge correction (default True). Walkers
            sharing a bias must all use the same setting.
        reweight : bool, optional
            Record the bias at the walker and c(t) per frame in
            ``REWEIGHT_<run_id>.npz`` (Tiwary & Parrinello 2015; default True).
        rct_ustride : int, optional
            Hills between per-frame records (default 5, i.e. 10 ps at a 2 ps
            pace). COLVAR and REWEIGHT rows use this interval.
        dcd_interval : float, optional
            Trajectory (and state CSV) interval in ps (default 10).

        Returns
        -------
        str
            The ``run_id`` string used for all output file names produced by
            this call (COLVAR, FE, checkpoint, PDB, system XML). ``FRAMES_<run_id>.npz``
            gives each COLVAR row's time and DCD frame (-1 if none),
            ``REWEIGHT_<run_id>.npz`` V(s_t, t) and c(t) per row, ``HILLS_<run_id>.npz`` every hill.

        Warns
        -----
        UserWarning (via logging.warning)
            If the first CV never passes the middle of its grid (ligand
            probably never unbound).

        Raises
        ------
        ValueError
            If none of ``pdb_file``, ``checkpoint_file``, or ``state_xml_file``
            is supplied.
        AssertionError
            If ``cv_specs`` is ``None`` or empty.
        """

        start_time = time.monotonic()

        assert cv_specs is not None and len(cv_specs) > 0, \
            "cv_specs must be a non-empty list of CVSpec objects. " \
            "Use autopath.metadynamics CV factory functions (e.g. com_cv, rmsd_cv) to build them."

        if run_id is None:
            run_id = f'W-{datetime.now().strftime("%H%M%S%f")}'

        mMD_steps = math.ceil(mMD_time / self.timestep.value_in_unit(openmmunit.picoseconds) * 1000.0)
        biasFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * biasFrequency)
        saveFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * saveFrequency)
        record_steps = int(rct_ustride) * biasFrequency
        dcd_steps = int(round(dcd_interval / self.timestep.value_in_unit(openmmunit.picoseconds)))
        if record_steps < 1 or dcd_steps < 1:
            raise ValueError("rct_ustride and dcd_interval must give at least one step")

        hill_height = hill_height * openmmunit.kilojoules_per_mole

        cv_names = ", ".join(s.name for s in cv_specs)
        logger.info(f"Running metadynamics with CV(s): {cv_names}")

        logger.debug("Setting up the integrator..")
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = biasFrequency,
                temperature = self.temperature,
                collision_rate = 1.0/openmmunit.picoseconds,
                timestep = self.timestep,
                splitting = "R V O V R",
                constraint_tolerance = 1.0e-6,
            )
        else:
            integrator = LangevinMiddleIntegrator(self.temperature,
                                                  1.0/openmmunit.picoseconds,
                                                  self.timestep)

        logger.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)

        # Precedence: checkpoint_file > state_xml_file > pdb_file.
        if checkpoint_file is not None:
            logger.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator
        elif state_xml_file is not None:
            logger.info(f"Setting state from portable XML state file {state_xml_file}")
            with open(state_xml_file) as f:
                state = XmlSerializer.deserialize(f.read())
            simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            simulation.context.setPositions(state.getPositions())
            simulation.context.setVelocities(state.getVelocities())
        elif pdb_file is not None:
            logger.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            raise ValueError(
                "One of pdb_file, checkpoint_file, or state_xml_file must be "
                "provided to set initial positions."
            )

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

        # Serialize/deserialize to get a fresh C++-owned copy (addForce transfers ownership).
        if funnel_force is not None:
            fresh_funnel = XmlSerializer.deserialize(XmlSerializer.serialize(funnel_force))
            system.addForce(fresh_funnel)
            logger.info(f"Added funnel potential with force group {funnel_force.getForceGroup()}")

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

        # None → shared self.out_dir (multi-walker); a per-walker path makes the walker independent.
        effective_bias_dir = bias_dir if bias_dir is not None else self.out_dir
        os.makedirs(effective_bias_dir, exist_ok=True)

        meta_args = (system, bias_variables, self.temperature, bias_factor, hill_height)
        meta_kwargs = dict(frequency=biasFrequency, saveFrequency=saveFrequency,
                           biasDir=effective_bias_dir)
        if boundary_correction:
            bounds = [s.bounds if s.bounds is not None else (s.grid_min, s.grid_max)
                      for s in resolved]
        else:
            bounds = [None] * len(resolved)
        meta = BoundaryCorrectedMetadynamics(*meta_args, bounds=bounds, **meta_kwargs)
        if boundary_correction:
            logger.info("Boundary-corrected hills on %s (max factor %.2f)",
                        ", ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in bounds),
                        float(meta.boundaryFactor.max()))

        # Rebuild rather than reinitialize(preserveState=True): the latter restores
        # global parameters by name, so stale values could override new forces' defaults.
        simulation = rebuild_simulation(simulation, system)
        simulation.context.setTime(0)
        simulation.context.setStepCount(0)
        simulation.currentStep = 0

        # After the rebuild: it returns a new Simulation.
        logger.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"WTMetaD_{run_id}",
            mMD_steps,
            dcd_steps,
        )
        if self.use_GReweighting:
            # The rebuilt simulation has its own integrator copy.
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/GR_WTMetaD_{run_id}.dat",
                                                            biasFrequency,
                                                            simulation.integrator,
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        colvar_list = [meta.getCollectiveVariables(simulation)]
        # Bias at the walker and c(t) per COLVAR row, for reweighting.
        bias_list = [meta.bias_energy(simulation)] if reweight else None
        rct_list = [meta.rct()] if reweight else None
        _ps_per_step = self.timestep.value_in_unit(openmmunit.picoseconds)

        def _save_reweight():
            steps = np.arange(len(colvar_list)) * record_steps
            # DCDReporter writes at steps D, 2D, ... so row k maps to frame k*R/D - 1
            dcd_frame = np.where((steps > 0) & (steps % dcd_steps == 0), steps // dcd_steps - 1, -1)
            time_ns = steps * _ps_per_step / 1000.0
            np.savez(os.path.join(self.out_dir, f"FRAMES_{run_id}.npz"),
                     time_ns=time_ns, dcd_frame=dcd_frame,
                     record_dt_ns=record_steps * _ps_per_step / 1000.0,
                     dcd_interval_ns=dcd_steps * _ps_per_step / 1000.0)
            if reweight:
                np.savez(os.path.join(self.out_dir, f"REWEIGHT_{run_id}.npz"),
                         time_ns=time_ns[:len(bias_list)],
                         bias_kj=np.array(bias_list), rct_kj=np.array(rct_list),
                         temperature=self.temperature.value_in_unit(openmmunit.kelvin),
                         bias_factor=float(bias_factor), rct_ustride=int(rct_ustride))
            h = np.array(meta.hills, dtype=float).reshape(-1, len(resolved) + 2)
            np.savez(os.path.join(self.out_dir, f"HILLS_{run_id}.npz"),
                     time_ns=h[:, 0] * _ps_per_step / 1000.0, cv=h[:, 1:-1], height_kj=h[:, -1])
        # ~40 snapshots over the run, so the curve is readable at any length
        fe_hist, fe_hist_steps = [], []
        # Start time orders sequential shared-bias walkers.
        _started_at = time.time()
        _fe_hist_every = max(record_steps,
                             (int(mMD_steps) // 40 // record_steps or 1) * record_steps)
        _store_every = max(1, round(self.store_CV / record_steps))
        for k, i in enumerate(range(0, int(mMD_steps), record_steps)):
            if self.verbose and k % _store_every == 0:
                np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), np.array(colvar_list))
                _save_reweight()

            meta.step(simulation, record_steps)
            colvar_list.append(meta.getCollectiveVariables(simulation))
            if reweight:
                bias_list.append(meta.bias_energy(simulation))
                rct_list.append(meta.rct())

            # FES history for convergence; OpenMM does not keep past bias states.
            if i % _fe_hist_every == 0:
                fe_hist.append(np.asarray(
                    meta.getFreeEnergy().value_in_unit(
                        openmmunit.kilojoule_per_mole), dtype=float))
                fe_hist_steps.append(i + record_steps)
        colvar_array = np.array(colvar_list)

        _save_reweight()
        if fe_hist:
            np.savez(
                os.path.join(self.out_dir, f"FEhist_{run_id}.npz"),
                fes_kj=np.array(fe_hist),
                time_ns=np.array(fe_hist_steps, dtype=float) * _ps_per_step / 1000.0,
                started_at=np.array(_started_at, dtype=float),
            )
            logger.info("Saved %d FES snapshots for the convergence curve (%s).",
                        len(fe_hist), f"FEhist_{run_id}.npz")

        np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)

        max_cv = float(colvar_array[:, 0].max())
        _lo = resolved[0].bound_end
        _half = _lo + 0.5 * (resolved[0].grid_max - _lo)
        if max_cv < _half:
            logger.warning(
                f"Walker {run_id}: CV never got halfway from its bound end to grid_max "
                f"({_half:.3f}; max={max_cv:.3f}). "
                "The ligand did not reach the unbound state — this walker will distort the FE profile. "
                "Consider increasing mMD_time or mMD_hill_height."
            )

        fe_path = os.path.join(self.out_dir, f"FE_{run_id}.npy")
        np.save(fe_path, meta.getFreeEnergy())

        if funnel_params is not None:
            try:
                temp_K = self.temperature.value_in_unit(openmmunit.kelvin)
                result = correct_fe_for_funnel(
                    fe_path,
                    funnel_params,
                    temperature=temp_K,
                    grid_min=resolved[0].grid_min,
                    grid_max=resolved[0].grid_max,
                    cv_length_per_unit_ang=resolved[0].cv_length_per_unit_ang,
                    cv_origin=resolved[0].bound_end,
                )
                rw_path = os.path.join(self.out_dir, f"FE_{run_id}_rw.npy")
                np.save(rw_path, result["fe_corrected"])
                logger.info(
                    f"[{run_id}] ΔG_sim={result['dG_bind_sim_kj_mol']:.2f} kJ/mol  "
                    f"correction={result['correction_kj_mol']:.2f} kJ/mol  "
                    f"ΔG°_b={result['dG_bind_std_kj_mol']:.2f} kJ/mol  "
                    f"pKd={result['pKd']:.2f}"
                )
            except Exception as e:
                logger.warning(f"Funnel standard-state correction failed for {run_id}: {e}", exc_info=True)

        if len(resolved) == 1:
            s = resolved[0]
            plot_colvar(self.out_dir, s.name)
            plot_bias(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name)
            plot_FE(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name)
            if funnel_params is not None:
                plot_FE_rw(self.out_dir, s.grid_min, s.grid_max, s.grid_points, s.name,
                           temperature=self.temperature.value_in_unit(openmmunit.kelvin),
                           origin=s.bound_end)
        else:
            a, b = resolved[0], resolved[1]
            plot_colvar_2D(self.out_dir, a.name, b.name)
            plot_FE_2D(
                self.out_dir,
                a.grid_min, a.grid_max, a.grid_points, a.name,
                b.grid_min, b.grid_max, b.grid_points, b.name,
            )

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors())
        save_system(system, f"{self.out_dir}/WTMetaD_system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/WTMetaD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/WTMetaD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return run_id

    def run2D(self, *args, **kwargs):
        """Deprecated stub — use ``run()`` with two ``CVSpec`` objects instead.

        Always raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "run2D() has been removed. Pass two CVSpec objects via cv_specs to run() instead. "
            "Example:\n"
            "  from autopath.metadynamics import rmsd_cv\n"
            "  cv_a = rmsd_cv(...)\n"
            "  cv_b = rmsd_cv(...)\n"
            "  metad.run(cv_specs=[cv_a, cv_b], ...)"
        )
