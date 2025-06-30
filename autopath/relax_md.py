# general imports
import os
import time
import logging

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# AutoPath imports
from autopath.utils import *
from autopath.customForces import add_flatbottom_COM_restraints
from autopath.equilibration import warm_up_system


class RelaxMD:
    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        out_dir: str = "relax_md",
        is_membrane: bool = False,
        HMR: bool = True,
        temp: float = 300,
        use_GReweighting: bool = False,
    ) -> None:

        self.topology = topology

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.timestep = 0.004 if HMR else 0.002
        self.temperature = temp * openmmunit.kelvin

        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        self.is_membrane = is_membrane
        self.use_GReweighting = use_GReweighting

        self.platform = select_platform("fastest")

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

        logging.debug("Setting up the integrator..")
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = 1000000,   # we dont care about this here
                temperature = self.temperature,
                collision_rate = 1.0/openmmunit.picoseconds,
                timestep = self.timestep * openmmunit.picoseconds,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            integrator = LangevinMiddleIntegrator(self.temperature, 1 / openmmunit.picoseconds, self.timestep)

        # integrator.setRandomNumberSeed(int(rep_idx))

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.debug("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)
        # Add flat-bottom COM restraints to prevent ligand from drifting too far away
        logging.debug("Adding flat-bottom COM restraints..")
        add_flatbottom_COM_restraints(system, self.ligand_atoms, self.pocket_atoms, r0=startdist)

        logging.debug("Minimizing..")
        simulation.minimizeEnergy()

        logging.debug("Warming up the system..")
        warm_up_system(simulation, integrator, 
                       warming_steps=npt_steps*2, 
                       timestep=0.002,# * openmmunit.picoseconds, # lower timestep for warming
                       Tend=self.temperature.value_in_unit(openmmunit.kelvin))

        # logging.info("Minimizing..")
        # simulation.minimizeEnergy()

        logging.debug("Running short NPT..")
        # Add barostat to the system
        system = add_barostat(system, self.temperature, is_membrane=self.is_membrane)

        # adjust timestep if needed
        if self.timestep != integrator.getStepSize():
            logging.debug(f"Adjusting timestep from {integrator.getStepSize()} to {self.timestep} ps.")
            integrator.setStepSize(self.timestep * openmmunit.picoseconds)
    
        simulation.context.reinitialize(preserveState=True)
        logging.debug(f"Stepsize set to {integrator.getStepSize()}")

        # run npt simulation
        simulation.step(npt_steps) #0.1 ns

        # remove existing restraint forces
        forces_to_remove = []
        for f_idx in range(system.getNumForces()):
            force = system.getForce(f_idx)
            if force.getName().startswith("k_flat_com"):
                logging.debug(f"Removing force {force.getName()} at index {f_idx}.")
                forces_to_remove.append(f_idx)

        for f_idx in sorted(forces_to_remove, reverse=True):
            system.removeForce(f_idx)

        # print_current_forces(system)

        # save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_simulation(simulation, f"{self.out_dir}/{run_id}_relax_checkpoint")
        save_system(system, f"{self.out_dir}/{run_id}_relax_system.xml")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}_relax.pdb")

        # Get COM distance
        finaldist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)

        logging.info(f"{run_id} - Initial:{startdist:.3f} nm - Final:{finaldist:.3f} nm")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} relaxation in {simulation_time/60:.2f} min.")
        return system
