import time
import math
import logging

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *


class VanillaMD:
    def __init__(
        self,
        system: str = None,
        topology: str = None,
        HMR: bool = True,
        temp: float = 300,
        out_dir: str = "MD",
    ):

        self.system = system
        self.topology = topology
<<<<<<< HEAD
        self.timestep = 0.004 if HMR else 0.002

=======

        self.timestep = 0.004 if HMR else 0.002
>>>>>>> main
        self.temperature = temp * openmmunit.kelvin

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.platform = select_platform("fastest")

        return

    def run(
        self,
        checkpoint_file: str = None,
        run_id: str = None,
        MD_time: int = 10,
    ):

        start_time = time.monotonic()

        # Calculate the number of steps required
        MD_steps = math.ceil(MD_time / self.timestep * 1000.0)  # 250.000 1ns at 4fs

        logging.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(int(rep_idx))

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        # If a checkpoint is provided, it will assume it comes from an equilibration simulation, so it will just continue
        if checkpoint_file is not None:
            logging.debug("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)

        add_reporters(
            simulation, self.out_dir, f"MD_{run_id}", MD_steps, 12500
        )  # save /0.1ns

        # Run the simulation
        simulation.step(MD_steps)

        # save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_simulation(simulation, f"{self.out_dir}/MD_{run_id}_checkpoint")
        save_system(self.system, f"{self.out_dir}/MD_{run_id}_system.xml")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/MD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished MD {run_id} in {simulation_time/60:.2f} min.")

        return
