# general imports
import os
import time
import logging
logger = logging.getLogger("autopath")

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# AutoPath imports
from autopath.utils import *
from autopath.customForces import add_flatbottom_COM_restraints, add_harmonic_restraints, print_current_forces
from autopath.equilibration import warm_up_system

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    logger.warning("Please install openmmtools to use Girsanov reweighting.")

class RelaxMD:
    """Short equilibration / relaxation MD runner.

    Performs energy minimization, temperature ramp-up, and a brief NPT run to
    relax the system before production MD. Harmonic positional restraints are
    applied to the ligand during relaxation and removed before saving, so that
    the returned system is unrestrained and ready for downstream stages.

    Parameters
    ----------
    topology : openmm.app.Topology
        OpenMM topology of the full system.
    ligand_atoms : list of int
        Atom indices belonging to the ligand; used for COM-distance tracking
        and positional restraints.
    pocket_atoms : list of int
        Atom indices of the binding-pocket residues; used for COM-distance
        tracking.
    out_dir : str, optional
        Directory where checkpoint, system XML, and PDB files are written.
    is_membrane : bool, optional
        When True, a membrane-compatible semi-isotropic barostat is used
        instead of the default isotropic one.
    timestep : float, optional
        Integration timestep in picoseconds. Default is 0.004 ps (4 fs),
        enabled by hydrogen-mass repartitioning.
    temp : float, optional
        Simulation temperature in Kelvin.
    platform : str, optional
        OpenMM platform name or ``"fastest"`` to auto-select the best
        available platform.
    use_GReweighting : bool, optional
        When True, the Girsanov-reweighted Langevin integrator
        (``LangevinSplittingGirsanov``) from openmmtools is used. Requires
        optional dependencies ``openmmtools`` and ``reweightingreporter``.

    Note
    ----
    Force-group 19 is reserved for the temporary harmonic ligand restraints
    added during this stage. This is offset by 15 relative to other AutoPath
    force groups to avoid collisions.
    """

    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        out_dir: str = "relax_md",
        is_membrane: bool = False,
        timestep: float = 0.004,  # 4 fs timestep — safe with HMR
        temp: float = 300,
        platform: str = "fastest",
        use_GReweighting: bool = False,
    ) -> None:

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.topology = topology

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temp * openmmunit.kelvin

        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        self.is_membrane = is_membrane

        self.platform = select_platform(platform)

        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            try:
                from openmmtools.integrators import LangevinSplittingGirsanov
                from reweightingreporter import ReweightingReporter
            except ImportError:
                raise ImportError("Please install openmmtools to use Girsanov reweighting.")

        return None

    def run(
        self,
        system: str = None,
        checkpoint_file: str = None,
        pdb_file: str = None,
        run_id: str = None,
        npt_steps: int = 25000,
    ) -> System:
        """Run relaxation MD: minimize, warm up, and perform a short NPT run.

        Applies temporary harmonic positional restraints to the ligand,
        runs energy minimization and a gradual temperature ramp, then
        integrates an NPT ensemble. Restraints are removed before writing
        output so the returned system is unrestrained.

        Parameters
        ----------
        system : openmm.System
            OpenMM System object containing all force field terms. Modified
            in-place (barostat added, temporary restraints removed).
        checkpoint_file : str, optional
            Path to a binary OpenMM checkpoint file from a previous run.
            Takes precedence over ``pdb_file`` when both are supplied.
        pdb_file : str, optional
            Path to a PDB file used to set initial positions when no
            checkpoint is available.
        run_id : str, optional
            Label used as a prefix for all output files written to
            ``self.out_dir``.
        npt_steps : int, optional
            Number of integration steps for both the warm-up phase and the
            NPT run. At the default 4 fs timestep, 25 000 steps ≈ 0.1 ns.

        Returns
        -------
        openmm.System
            The relaxed system with the barostat added and temporary
            restraint forces removed. Checkpoint, system XML, and final
            PDB are written to ``self.out_dir``.
        """

        start_time = time.monotonic()

        logger.debug("Setting up the integrator..")
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout=100000000,  # output frequency irrelevant for this short stage
                temperature=self.temperature,
                collision_rate=1.0 / openmmunit.picoseconds,
                timestep=self.timestep,
                splitting="R V O V R",       # ABOBA splitting — required for Girsanov reweighting
                constraint_tolerance=1.0e-6,
            )
        else:
            integrator = LangevinMiddleIntegrator(self.temperature,
                                                  1.0 / openmmunit.picoseconds,
                                                  self.timestep)

        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logger.debug("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)

        logger.debug("Adding harmonic ligand restraints..")
        add_harmonic_restraints(
            system,
            initial_positions,
            self.topology,
            self.ligand_atoms,
            restraint_force=5,
            force_name="k_harmonic_restrain",
            force_group=19,  # offset by 15 to avoid overlap with other AutoPath force groups
        )

        simulation.context.reinitialize(preserveState=True)

        logger.debug("Minimizing..")
        simulation.minimizeEnergy(maxIterations=1000)

        logger.debug("Warming up the system..")
        warm_up_system(simulation, integrator,
                       warming_steps=npt_steps,
                       timestep=0.004 * openmmunit.picoseconds,  # lower timestep for warming
                       Tend=self.temperature.value_in_unit(openmmunit.kelvin))

        logger.debug("Running short NPT..")
        system = add_barostat(system, self.temperature, is_membrane=self.is_membrane)

        # Warm-up may have altered the integrator step size; restore before production.
        if self.timestep != integrator.getStepSize():
            logger.debug(f"Adjusting timestep from {integrator.getStepSize()} to {self.timestep}.")
            integrator.setStepSize(self.timestep)

        simulation.context.reinitialize(preserveState=True)
        logger.debug(f"Stepsize set to {integrator.getStepSize()}")

        simulation.step(npt_steps)  # 0.1 ns at default settings

        # Remove temporary harmonic restraints before returning the system.
        forces_to_remove = []
        for f_idx in range(system.getNumForces()):
            force = system.getForce(f_idx)
            if force.getName().startswith("k_harmonic_restrain"):
                logger.info(f"Removing force {force.getName()} at index {f_idx}.")
                forces_to_remove.append(f_idx)

        for f_idx in sorted(forces_to_remove, reverse=True):
            system.removeForce(f_idx)

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        # Persist correct box vectors to the PDB so downstream tools read the right unit cell.
        self.topology.setPeriodicBoxVectors(
            simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
        )
        save_simulation(simulation, f"{self.out_dir}/{run_id}_relax_checkpoint")
        save_system(system, f"{self.out_dir}/{run_id}_relax_system.xml")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}_relax.pdb")

        finaldist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)

        logger.info(f"{run_id} - Initial:{startdist:.3f} nm - Final:{finaldist:.3f} nm")

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished {run_id} relaxation in {simulation_time/60:.2f} min.")
        return system
