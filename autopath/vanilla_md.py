import time
import math

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *
from autopath.customForces import add_harmonic_restraints

import logging
logger = logging.getLogger("autopath")

class VanillaMD:
    """Unbiased (plain) MD runner for AutoPath production simulations.

    Runs a standard NVT Langevin MD simulation, optionally with harmonic
    positional restraints on a subset of atoms (e.g. protein Cα). Outputs
    trajectory, checkpoint, system XML, and a final PDB.

    Parameters
    ----------
    system : openmm.System
        OpenMM System object containing all force field terms.
    topology : openmm.app.Topology
        OpenMM topology of the full system.
    restrained_atoms : list of int, optional
        Atom indices to restrain with harmonic positional restraints during
        the simulation. Typically protein Cα atoms. ``None`` disables restraints.
    timestep : float, optional
        Integration timestep in picoseconds. Default is 0.004 ps (4 fs),
        enabled by hydrogen-mass repartitioning.
    temperature : float, optional
        Simulation temperature in Kelvin.
    save_freq : int, optional
        Interval (in steps) between checkpoint and trajectory saves.
        Default is 25 000 steps, which corresponds to every 0.1 ns at the
        default 4 fs timestep.
    out_dir : str, optional
        Directory where trajectory, checkpoint, system XML, and PDB are written.
    platform : str, optional
        OpenMM platform name or ``"fastest"`` to auto-select the best
        available platform.
    verbose : int, optional
        Verbosity level passed to the trajectory reporter (0 = silent,
        higher values add more output columns).
    """

    def __init__(
        self,
        system: str = None,
        topology: str = None,
        restrained_atoms: list[int] = None,
        timestep: float = 0.004,  # 4 fs timestep — safe with HMR
        temperature: float = 300,
        save_freq: int = 25000,
        out_dir: str = "MD",
        platform: str = "fastest",
        verbose: int = 2,
    ):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.system = system
        self.topology = topology

        self.restrained_atoms = restrained_atoms

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temperature * openmmunit.kelvin

        self.save_freq = save_freq

        self.verbose = verbose
        self.platform = select_platform(platform)

        return

    def run(
        self,
        checkpoint_file: str = None,
        pdb_file: str = None,
        run_id: str = None,
        MD_time: int = 10,
        restart_velocities: bool = False,
    ):
        """Run a production MD simulation.

        Loads initial state from a checkpoint or PDB, optionally applies
        harmonic positional restraints, and integrates for the requested
        simulation time.

        Parameters
        ----------
        checkpoint_file : str, optional
            Path to a binary OpenMM checkpoint file. When provided, the
            simulation continues directly from that state (positions,
            velocities, box vectors). Takes precedence over ``pdb_file``
            when both are supplied.
        pdb_file : str, optional
            Path to a PDB file used to set initial positions and box vectors
            when no checkpoint is available. Velocities are initialised at
            ``temperature``.
        run_id : str, optional
            Label used as a prefix for all output files written to
            ``self.out_dir``.
        MD_time : int, optional
            Total simulation time in nanoseconds.
        restart_velocities : bool, optional
            When True, reassign velocities from a Maxwell–Boltzmann
            distribution at ``temperature`` after loading the initial state.
            Useful when continuing from a checkpoint with stale velocities.

        Returns
        -------
        None
            Checkpoint, system XML, trajectory, and final PDB are written to
            ``self.out_dir``.

        Note
        ----
        The default ``save_freq=25000`` saves output every 0.1 ns at the
        default 4 fs timestep (25 000 × 4 fs = 100 ps = 0.1 ns).
        """

        start_time = time.monotonic()

        MD_steps = math.ceil(MD_time / self.timestep.value_in_unit(openmmunit.picoseconds) * 1000.0)

        logger.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )

        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        if checkpoint_file is None and pdb_file is None:
            raise ValueError("Either pdb_file or checkpoint_file must be provided to set initial positions.")
        elif checkpoint_file is None and pdb_file is not None:
            logger.info(f"Setting positions and box vectors from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)

        elif checkpoint_file is not None and pdb_file is None:
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            # Both provided: checkpoint takes precedence.
            logger.warning("Both checkpoint_file and pdb_file were provided. Using checkpoint_file.")
            simulation.loadCheckpoint(checkpoint_file)

        if restart_velocities:
            logger.info(f"Resetting velocities to temperature {self.temperature}..")
            simulation.context.setVelocitiesToTemperature(self.temperature)

        input_positions = simulation.context.getState(getPositions=True).getPositions()
        if self.restrained_atoms is not None:
            add_harmonic_restraints(
                self.system,
                input_positions,
                self.topology,
                self.restrained_atoms,
                10,
                "k_restraint_MD",
                14,
            )

        simulation.context.reinitialize(preserveState=True)

        add_reporters(
            simulation, self.out_dir, f"MD_{run_id}", MD_steps, self.save_freq, self.verbose
        )

        simulation.step(MD_steps)

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        # Persist correct box vectors to the PDB so downstream tools read the right unit cell.
        self.topology.setPeriodicBoxVectors(
            simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
        )
        save_simulation(simulation, f"{self.out_dir}/MD_{run_id}_checkpoint")
        save_system(self.system, f"{self.out_dir}/MD_{run_id}_system.xml")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/MD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished MD {run_id} in {simulation_time/60:.2f} min.")

        return
