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
from autopath.metadynamics.TransitionTempered import TransitionTemperedMetadynamics
from autopath.metadynamics.Hills import HILLS_KEYS, hills_record
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


def _atomic_save(path, array=None, **arrays):
    """np.save / np.savez via a temporary file, so readers never see a partial file."""
    tmp = f"{path[:-4]}.tmp{path[-4:]}"
    if arrays:
        np.savez(tmp, **arrays)
    else:
        np.save(tmp, array)
    os.replace(tmp, path)

MONITOR_GROUP = 31


def add_monitor(system, cv, name):
    """Unbiased CV: zero energy, value via getCollectiveVariableValues (like PLUMED PRINT)."""
    f = CustomCVForce("0*m")
    f.addCollectiveVariable("m", cv)
    f.setForceGroup(MONITOR_GROUP)
    f.setName(f"monitor_{name}")
    system.addForce(f)
    return f


def add_orientation_restraint(system, funnel_force, funnel_params, positions_nm):
    """Add the OrientationRestraintForce of orientation-restrained funnel params to ``system``.

    Refuses a start whose core is more than ``ORIENT_MAX_START_DEG`` from ``orient_ref``, a funnel
    force that does not match the params' mode, or a system that already has an orientation restraint.
    Returns the start θ (deg), or None for other funnel modes.
    """
    from autopath.metadynamics.Funnel import (ORIENT_OFFSET_PARAMS, check_orient_start, funnel_mode,
                                              orientation_restraint_force)
    has_offset = funnel_force is not None and any(
        funnel_force.getGlobalParameterName(i) == ORIENT_OFFSET_PARAMS[0]
        for i in range(getattr(funnel_force, "getNumGlobalParameters", lambda: 0)()))
    orient = funnel_params is not None and funnel_mode(funnel_params) == "orient"
    if has_offset != orient:
        raise ValueError("funnel force and funnel_params disagree on the orientation restraint: not supported")
    if not orient:
        return None
    if any(isinstance(f, OrientationRestraintForce) for f in system.getForces()):
        raise ValueError("system already has an OrientationRestraintForce: not supported")
    theta = check_orient_start(funnel_params, positions_nm)
    system.addForce(orientation_restraint_force(funnel_params, system.getNumParticles()))
    logger.info("Added the orientation restraint (k %.3g kJ/mol); start core %.3f deg from orient_ref",
                float(funnel_params["orient_k"]), theta)
    return theta


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
        monitor_cvs: dict = None,
        dcd_interval: float = 10.0,
        tt_basins: list = None,
        tt_threshold: float = 0.0,
        tt_bias_factor: float = None,
        run_meta: dict = None,
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
        run_meta : dict, optional
            Strings stored in the HILLS record, e.g. ``funnel_file``,
            ``learned_cv_file``. Items that are ``None`` or empty are skipped; keys of
            the HILLS record itself (``Hills.HILLS_KEYS``) raise ValueError.
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
        monitor_cvs : dict, optional
            ``{name: cvpack CV or CVSpec}`` logged unbiased as ``monitor_<name>``
            arrays in REWEIGHT (zero-energy force in group 31).
        rct_ustride : int, optional
            Hills between per-frame records (default 5, i.e. 10 ps at a 2 ps
            pace). COLVAR and REWEIGHT rows use this interval.
        dcd_interval : float, optional
            Trajectory (and state CSV) interval in ps (default 10).
        tt_basins : list, optional
            Basin points (CV units, one coordinate per CV) for transition-tempered
            metadynamics (Dama 2014); ``None`` (default) runs well-tempered.
            ``bias_factor`` then sets ΔT = (γ - 1) T after the basins connect.
        tt_threshold : float, optional
            V_th in kJ/mol: tempering starts once V* exceeds it (default 0).
        tt_bias_factor : float, optional
            With ``tt_basins``: keep well tempering with ``bias_factor`` and temper
            by V* with this factor (PLUMED BIASFACTOR + TTBIASFACTOR). None (default)
            is pure TT with ``bias_factor``.

        Returns
        -------
        str
            The ``run_id`` string used for all output file names produced by
            this call (COLVAR, FE, checkpoint, PDB, system XML). ``REWEIGHT_<run_id>.npz``
            holds V(s_t, t) and c(t) per COLVAR row, ``HILLS_<run_id>.npz`` every hill.
            HILLS and REWEIGHT store ``record_dt_ns`` and ``dcd_interval_ns``; see
            ``Analysis.dcd_frame_of_rows`` for the row -> trajectory frame map.

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

        clash = sorted(set(run_meta or {}) & HILLS_KEYS)
        if clash:
            raise ValueError(f"run_meta keys {clash} are HILLS record keys: not supported")

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

        add_orientation_restraint(system, funnel_force, funnel_params,
                                  np.asarray(input_positions.value_in_unit(openmmunit.nanometer)))

        # Serialize/deserialize to get a fresh C++-owned copy (addForce transfers ownership).
        if funnel_force is not None:
            fresh_funnel = XmlSerializer.deserialize(XmlSerializer.serialize(funnel_force))
            system.addForce(fresh_funnel)
            logger.info(f"Added funnel potential with force group {funnel_force.getForceGroup()}")

        resolved = [spec.resolve(input_positions, self.n_atoms, self.topology) for spec in cv_specs]

        monitors = {}
        for mname, mcv in (monitor_cvs or {}).items():
            if hasattr(mcv, "resolve"):
                mcv = mcv.resolve(input_positions, self.n_atoms, self.topology).cv
            monitors[mname] = add_monitor(system, mcv, mname)
        monitor_vals = {m: [] for m in monitors}

        def _record_monitors():
            for m, f in monitors.items():
                monitor_vals[m].append(float(f.getCollectiveVariableValues(simulation.context)[0]))

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
        if tt_basins is not None:
            meta = TransitionTemperedMetadynamics(*meta_args, bounds=bounds, basins=tt_basins,
                                                  threshold=tt_threshold,
                                                  ttBiasFactor=tt_bias_factor, **meta_kwargs)
            logger.info("Transition-tempered hills: basins %s, V_th %.2f kJ/mol%s",
                        tt_basins, tt_threshold,
                        "" if tt_bias_factor is None else
                        f", well-tempered γ {bias_factor:g} x TT γ {tt_bias_factor:g}")
        else:
            meta = BoundaryCorrectedMetadynamics(*meta_args, bounds=bounds, **meta_kwargs)
        tempering = "wt" if tt_basins is None else "tt" if tt_bias_factor is None else "wt+tt"
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
        if reweight:
            _record_monitors()
        _ps_per_step = self.timestep.value_in_unit(openmmunit.picoseconds)

        intervals = dict(record_dt_ns=record_steps * _ps_per_step / 1000.0,
                         dcd_interval_ns=dcd_steps * _ps_per_step / 1000.0)

        def _save_reweight(finished=False):
            if reweight:
                _atomic_save(os.path.join(self.out_dir, f"REWEIGHT_{run_id}.npz"),
                         time_ns=np.arange(len(bias_list)) * intervals["record_dt_ns"],
                         bias_kj=np.array(bias_list), rct_kj=np.array(rct_list),
                         **{f"monitor_{m}": np.array(v) for m, v in monitor_vals.items()},
                         **intervals,
                         temperature=self.temperature.value_in_unit(openmmunit.kelvin),
                         bias_factor=float(bias_factor), rct_ustride=int(rct_ustride),
                         tempering=tempering)
            _atomic_save(os.path.join(self.out_dir, f"HILLS_{run_id}.npz"),
                         **hills_record(meta, run_id, _started_at, _ps_per_step, resolved,
                                        tempering, finished=finished, **intervals),
                         **{k: str(v) for k, v in (run_meta or {}).items() if v not in (None, "")})
        _started_at = time.time()
        _store_every = max(1, round(self.store_CV / record_steps))
        for k, i in enumerate(range(0, int(mMD_steps), record_steps)):
            if self.verbose and k % _store_every == 0:
                _atomic_save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), np.array(colvar_list))
                _save_reweight()

            meta.step(simulation, record_steps)
            colvar_list.append(meta.getCollectiveVariables(simulation))
            if reweight:
                bias_list.append(meta.bias_energy(simulation))
                rct_list.append(meta.rct())
                _record_monitors()
        colvar_array = np.array(colvar_list)

        _save_reweight(finished=True)

        _atomic_save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)

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

        # Eq. 15 needs a 1D CV with a length scale (funnel z or path); others are reweighted later.
        if (funnel_params is not None and len(resolved) == 1
                and resolved[0].cv_length_per_unit_ang is not None):
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

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors())
        save_system(system, f"{self.out_dir}/WTMetaD_system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/WTMetaD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/WTMetaD_{run_id}.pdb")

        # Quick looks read every walker's files in out_dir; concurrent walkers rewrite
        # theirs meanwhile, so a failed plot must not kill a finished run.
        try:
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
        except Exception as e:
            logger.warning(f"Quick-look plots failed for {run_id} (non-fatal): {e}")

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
