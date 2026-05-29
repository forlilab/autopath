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
    logging.warning("Please install openmmtools to use Girsanov reweighting.")


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
    record_CV : int
        Steps between CV snapshots (~every 10 ps at default timestep).
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
                logging.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logging.info("Using Girsanov reweighting for steered MD.")

        self.verbose = verbose
        self.record_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 10)
        self.store_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 100)

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
        bias_dir: str = None,
    ) -> str:
        """Run a single WT-MetaD trajectory and return the run identifier.

        Parameters
        ----------
        pdb_file : str, optional
            Path to a PDB file used to set initial positions and box vectors.
            Mutually exclusive with ``checkpoint_file``; if both are given,
            the checkpoint takes precedence with a warning.
        system : openmm.System
            Prepared OpenMM System object (forces already added before calling).
        checkpoint_file : str, optional
            Path to an OpenMM binary checkpoint to resume from.
        run_id : str, optional
            Label for all output files.  Auto-generated from wall-clock time
            when ``None`` (format ``W-HHMMSS``).
        cv_specs : list[CVSpec]
            One or two ``CVSpec`` objects defining the collective variables.
            CVSpecs may be deferred — they call ``.resolve()`` internally once
            input positions are available.  Use CV factory functions such as
            ``com_cv`` or ``rmsd_cv`` to construct them.
        mMD_time : int, optional
            Total simulation time in nanoseconds (default 10).
        bias_factor : float, optional
            Well-tempered γ parameter controlling bias growth rate (default 10).
            Higher values allow larger biases and faster exploration at the cost
            of convergence noise.
        hill_height : float, optional
            Initial Gaussian hill height in kJ/mol (default 1.2, ≈ 0.5 k_B T at
            300 K).
        biasFrequency : int, optional
            Frequency of Gaussian hill deposition in picoseconds (default 2 ps).
            Converted internally to an integer step count.
        saveFrequency : int, optional
            Frequency of on-disk bias saves in picoseconds (default 50 ps).
        funnel_force : openmm.Force, optional
            Pre-built funnel restraint force.  Because ``system.addForce()``
            transfers C++ ownership, the force is serialized/deserialized inside
            ``run()`` to obtain a fresh owned copy.  Do not reuse the same
            ``Force`` object across multiple ``run()`` calls.
        funnel_params : dict, optional
            Parameter dict passed to ``correct_fe_for_funnel`` for the
            Limongelli 2013 standard-state correction.  Skipped when ``None``.
        bias_dir : str, optional
            Directory where OpenMM reads and writes hill files.  Defaults to
            ``None``, which maps to ``self.out_dir`` — all walkers writing to
            the same directory accumulate each other's hills (OpenMM multi-walker
            convention).  Pass a per-walker subdirectory to make a walker
            independent.

        Returns
        -------
        str
            The ``run_id`` string used for all output file names produced by
            this call (COLVAR, FE, checkpoint, PDB, system XML).

        Warns
        -----
        UserWarning (via logging.warning)
            Emitted after the simulation if the first CV never exceeded 0.5,
            indicating the ligand may not have reached the unbound state.

        Raises
        ------
        ValueError
            If neither ``pdb_file`` nor ``checkpoint_file`` is supplied.
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

        hill_height = hill_height * openmmunit.kilojoules_per_mole

        cv_names = ", ".join(s.name for s in cv_specs)
        logging.info(f"Running metadynamics with CV(s): {cv_names}")

        logging.debug("Setting up the integrator..")
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

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is None and pdb_file is None:
            raise ValueError("Either pdb_file or checkpoint_file must be provided to set initial positions.")
        elif checkpoint_file is None and pdb_file is not None:
            logging.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)
        elif checkpoint_file is not None and pdb_file is None:
            logging.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator
        else:
            logging.warning("Both checkpoint_file and pdb_file are provided. Using checkpoint_file.")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator

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

        meta = Metadynamics(
            system,
            bias_variables,
            self.temperature,
            bias_factor,
            hill_height,
            frequency=biasFrequency,
            saveFrequency=saveFrequency,
            biasDir=effective_bias_dir,
        )

        simulation.context.setTime(0)
        simulation.context.setStepCount(0)
        simulation.context.reinitialize(preserveState=True)

        colvar_list = [meta.getCollectiveVariables(simulation)]
        for i in range(0, int(mMD_steps), self.record_CV):
            if self.verbose and i % self.store_CV == 0:
                np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), np.array(colvar_list))

            meta.step(simulation, self.record_CV)
            colvar_list.append(meta.getCollectiveVariables(simulation))
        colvar_array = np.array(colvar_list)

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
                logger.warning(f"Funnel standard-state correction failed for {run_id}: {e}", exc_info=True)

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

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors())
        save_system(system, f"{self.out_dir}/WTMetaD_system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/WTMetaD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/WTMetaD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return run_id

    def run2D(self, *args, **kwargs):
        """Deprecated stub — use ``run()`` with two ``CVSpec`` objects instead.

        .. deprecated::
            Pass a two-element ``cv_specs`` list to :meth:`run` to run 2D
            metadynamics.  This method always raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "run2D() has been removed. Pass two CVSpec objects via cv_specs to run() instead. "
            "Example:\n"
            "  from autopath.metadynamics import rmsd_cv\n"
            "  cv_a = rmsd_cv(...)\n"
            "  cv_b = rmsd_cv(...)\n"
            "  metad.run(cv_specs=[cv_a, cv_b], ...)"
        )
