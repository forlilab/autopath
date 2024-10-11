import os
import math
import time
import logging
from glob import glob

from autopath.utils import *
from autopath.analysis import plot_sMD_statistics

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

class SteeredMD:
    """
    A class to perform steered molecular dynamics, pulling a ligand out of its binding pocket
    """

    def __init__(
        self,
        system: str = None,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        HMR: bool = True,
        temp: float = 300,
        out_dir: str = None,
    ):
        self.system = system
        self.topology = topology
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.timestep = 0.004 if HMR else 0.002
        self.temperature = temp * openmmunit.kelvin
        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        self.restrained_atoms = restrained_atoms
        self.platform = select_platform("fastest")

    def run_single_direction(
        self, simulation, rep_idx, direction, dx_per_move, sMD_moves, steps_per_move, initial_r0, final_r0
    ):
        """Run the pulling process in a single direction (forward or backward) for a single replica."""
        logging.info(f"Replica {rep_idx} - {direction} pulling")

        add_reporters(
            simulation,
            self.out_dir,
            f"sMD_{rep_idx}_{direction}",
            sMD_moves,
            steps_per_move * 2,
        )

        logging.info(f"Initial COM distance: {initial_r0} nm")
        simulation.context.setParameter("r0", initial_r0)

        # Initialize work
        work_val_old = openmmunit.Quantity(value=0, unit=openmmunit.kilojoules_per_mole)
        f = open(f"{self.out_dir}/sMD_log_{rep_idx}_{direction}.dat", "a")

        for i in range(sMD_moves):
            current_dist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms) * openmmunit.nanometers

            # Get radius of starting point and end point
            if direction == "backward":
                r_current = final_r0 - float(i + 1) * abs(dx_per_move)
                r_start = final_r0 - float(i) * abs(dx_per_move)
            else:
                r_current = initial_r0 + float(i + 1) * dx_per_move
                r_start = initial_r0 + float(i) * dx_per_move

            simulation.context.setParameter("r0", r_current)
            force_val = -self.fc_pull * (current_dist - r_current)

            simulation.step(steps_per_move)

            # Calculate work for difference in potential energy in transition
            spr_energy_end = 0.5 * -self.fc_pull * (current_dist - r_current) ** 2
            spr_energy_start = 0.5 * -self.fc_pull * (current_dist - r_start) ** 2
            work_val = work_val_old + spr_energy_end - spr_energy_start
            work_val_old = work_val

            # Write log file
            f.write(
                f"{i},{r_current / openmmunit.nanometers},{current_dist / openmmunit.nanometers},{force_val / openmmunit.kilojoules_per_mole * openmmunit.nanometer},{work_val / openmmunit.kilojoules_per_mole}\n"
            )

        f.close()

        # Log final COM distance
        final_dist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)
        logging.info(f"Final COM distance: {final_dist:.2f} nm")

        # Save final positions
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_pdb(
            self.topology, final_positions, f"{self.out_dir}/steeredMD_{rep_idx}_{direction}.pdb"
        )

    def run(
        self,
        sMD_time: float = 1,  # 1ns
        displacement: float = 0.5,  # nm
        steps_per_move: int = 250,  # 1ps
        pulling_force: int = 1000,
        replicas: int = 5,
        checkpoint_file: str = None,
        do_backwards: bool = False,
    ):
        """Main method to run steered MD in both directions (forward and backward) for multiple replicas."""
        simulation_start_time = time.monotonic()
        sys_name = self.out_dir.split("/")[0]

        # Calculate the number of steps
        sMD_steps = math.ceil(sMD_time / self.timestep * 1000.0)
        sMD_moves = int(sMD_steps / steps_per_move)
        dx_per_move = (displacement / sMD_moves) * openmmunit.nanometers

        self.fc_pull = pulling_force * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2

        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        if checkpoint_file:
            simulation.loadCheckpoint(checkpoint_file)

        # Add harmonic positional restraints to protein CA
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        if self.restrained_atoms is not None:
            add_harmonic_restraints(
                self.system,
                input_positions,
                self.topology,
                self.restrained_atoms,
                10,
                "k_CA",
                14,
            )

        # Add COM force with arbitrary initial r0, then run_single_direction will set it properly
        add_COM_force(
            self.system, self.ligand_atoms, self.pocket_atoms, self.fc_pull, 0)
        simulation.context.setTime(0)  # reset simulation time
        simulation.context.reinitialize(preserveState=True)
        
        # Loop over replicas
        for rep_idx in range(1, replicas + 1):
            replica_start_time = time.monotonic()

            if checkpoint_file:
                simulation.loadCheckpoint(checkpoint_file)

            # simulation.context.setVelocitiesToTemperature(self.temperature)

            startdist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)
            initial_r0 = startdist * openmmunit.nanometers
            final_r0 = initial_r0 + abs(dx_per_move) * sMD_moves
            
            # Run forward direction
            self.run_single_direction(
                simulation, rep_idx, direction="forward", dx_per_move=dx_per_move, sMD_moves=sMD_moves,
                steps_per_move=steps_per_move, initial_r0=initial_r0, final_r0=final_r0
                )

            # Generate statistics and plots for the forward direction
            files_f = glob(f"{self.out_dir}/sMD_log_*_forward.dat")
            plot_sMD_statistics(files_f, f'{sys_name}_forward', self.out_dir)

            if do_backwards:
                # Run backward direction
                self.run_single_direction(
                    simulation, rep_idx, direction="backward", dx_per_move=-dx_per_move, sMD_moves=sMD_moves,
                    steps_per_move=steps_per_move, initial_r0=initial_r0, final_r0=final_r0
                )

                # Generate statistics and plots for the backward direction
                files_b = glob(f"{self.out_dir}/sMD_log_*_backward.dat")
                plot_sMD_statistics(files_b, f'{sys_name}_backward', self.out_dir)

            # Logging the time taken for each replica
            replica_time = time.monotonic() - replica_start_time
            logging.info(f"Finished replica {rep_idx}/{replicas} in {replica_time/60:.2f} min")

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logging.info(
            f"Finished {replicas} replicas of sMD {'with backwards pulling' if do_backwards else ''} in {simulation_time/60:.2f} min."
        )