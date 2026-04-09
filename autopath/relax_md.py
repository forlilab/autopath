# general imports
import os
import time
import logging
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
    girsanov = False
    logger.warning("Please install openmmtools to use Girsanov reweighting.")

class RelaxMD:
    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        out_dir: str = "relax_md",
        is_membrane: bool = False,
        timestep: float = 0.004, #  # 4 fs timestep
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
    ) -> Tuple[float, float]:

        start_time = time.monotonic()

        logger.debug("Setting up the integrator..")
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = 100000000,   # we dont care about this here
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

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logger.debug("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)
        # Add flat-bottom COM restraints to prevent ligand from drifting too far away
        logger.debug("Adding flat-bottom COM restraints..")
        # add_flatbottom_COM_restraints(system, self.ligand_atoms, self.pocket_atoms, 
        #                               r0=startdist,
        #                               upper_wall=0.001, # 0.1 nm upper wall
        #                               K_flat=10000, # 500 kJ/mol/nm^2
        #                               )
        add_harmonic_restraints(
                system,
                initial_positions,
                self.topology,
                self.ligand_atoms,
                restraint_force=5,
                force_name=f"k_harmonic_restrain",
                force_group=19, #Offset by 15 to avoid overlap with other forces
            )
                
        simulation.context.reinitialize(preserveState=True)
        # print_current_forces(system)

        logger.debug("Minimizing..")
        simulation.minimizeEnergy(maxIterations=1000)

        logger.debug("Warming up the system..")
        warm_up_system(simulation, integrator, 
                       warming_steps=npt_steps, 
                       timestep=0.004 * openmmunit.picoseconds, # lower timestep for warming
                       Tend=self.temperature.value_in_unit(openmmunit.kelvin))

        # logger.info("Minimizing..")
        # simulation.minimizeEnergy()

        logger.debug("Running short NPT..")
        # Add barostat to the system
        system = add_barostat(system, self.temperature, is_membrane=self.is_membrane)

        # adjust timestep if needed
        if self.timestep != integrator.getStepSize():
            logger.debug(f"Adjusting timestep from {integrator.getStepSize()} to {self.timestep}.")
            integrator.setStepSize(self.timestep)
    
        simulation.context.reinitialize(preserveState=True)
        logger.debug(f"Stepsize set to {integrator.getStepSize()}")

        # run npt simulation
        simulation.step(npt_steps) #0.1 ns

        # remove existing restraint forces
        forces_to_remove = []
        for f_idx in range(system.getNumForces()):
            force = system.getForce(f_idx)
            # if force.getName().startswith("k_flat_com"):
            if force.getName().startswith("k_harmonic_restrain"):
                logger.info(f"Removing force {force.getName()} at index {f_idx}.")
                forces_to_remove.append(f_idx)

        for f_idx in sorted(forces_to_remove, reverse=True):
            system.removeForce(f_idx)

        # print_current_forces(system)

        # save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_simulation(simulation, f"{self.out_dir}/{run_id}_relax_checkpoint")
        save_system(system, f"{self.out_dir}/{run_id}_relax_system.xml")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}_relax.pdb")

        # Get COM distance
        finaldist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)

        logger.info(f"{run_id} - Initial:{startdist:.3f} nm - Final:{finaldist:.3f} nm")

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished {run_id} relaxation in {simulation_time/60:.2f} min.")
        return system
