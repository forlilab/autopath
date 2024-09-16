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

    def run(
        self,
        sMD_time: float = 1,  # 5ns
        displacement: float = 0.5,  # nm
        steps_per_move: int = 250,  # 1ps
        pulling_force: int = 1000,
        replicas: int = 5,
        checkpoint_file: str = None,
    ):

        start_time = time.monotonic()

        # Calculate the number of steps required
        sMD_steps = math.ceil(sMD_time / self.timestep * 1000.0)  # 250.000 1ns at 4fs
        sMD_moves = int(sMD_steps / steps_per_move)

        self.fc_pull = (
            pulling_force * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        )

        logging.info("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(int(rep_idx))

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.info("Loading simulation checkpoint..")
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

        startdist = get_COG_dist(simulation, self.ligand_atoms, self.pocket_atoms)
        initial_r0 = startdist * openmmunit.nanometers
        logging.info(f"Initial COM distance is {startdist:.2f} nm")

        velocity = displacement / sMD_time
        logging.info(f"Pulling rate i.e. velocity will be {velocity:.2f} nm/ns")

        dx_per_move = (displacement / sMD_moves) * openmmunit.nanometers

        logging.info(
            f"Running {sMD_time} ns in {sMD_moves} moves of {steps_per_move} steps each.."
        )

        add_COM_force(
            self.system, self.ligand_atoms, self.pocket_atoms, self.fc_pull, initial_r0
        )
        simulation.context.reinitialize(preserveState=True)
        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setParameter("r0", initial_r0)

        for rep_idx in range(1, replicas + 1):

            logging.info(f"Replica {rep_idx}/{replicas}")

            if checkpoint_file is not None:
                logging.info("Loading simulation checkpoint..")
                simulation.loadCheckpoint(checkpoint_file)

            simulation.context.setParameter("r0", initial_r0)

            add_reporters(
                simulation,
                self.out_dir,
                f"sMD_{rep_idx}",
                sMD_steps,
                steps_per_move * 2,
            )  # dont save too often

            # Initializing work
            work_val_old = openmmunit.Quantity(
                value=0, unit=openmmunit.kilojoules_per_mole
            )

            f = open(f"{self.out_dir}/sMD_log_{rep_idx}.dat", "a")
            for i in range(sMD_moves):

                # Get COM distance
                current_dist = get_COG_dist(
                    simulation, self.ligand_atoms, self.pocket_atoms
                )
                current_dist = current_dist * openmmunit.nanometers
                # logging.info(f'Current distance is {current_dist}')

                # Get radius of starting point and end point
                r_current = initial_r0 + float(i + 1) * dx_per_move
                r_start = initial_r0 + float(i) * dx_per_move

                simulation.context.setParameter("r0", r_current)

                # Calculate force F = -k * x
                force_val = -self.fc_pull * (current_dist - r_current)

                simulation.step(steps_per_move)

                # Calculate work for difference in potential energy in transition
                # W = EK_end - EK_start
                # EK = 0.5 * k * x^2
                spr_energy_end = 0.5 * -self.fc_pull * (current_dist - r_current) ** 2
                spr_energy_start = 0.5 * -self.fc_pull * (current_dist - r_start) ** 2
                work_val = work_val_old + spr_energy_end - spr_energy_start
                work_val_old = work_val

                # Write steered.dat log file
                f.write(
                    f"{i},{r_current / openmmunit.nanometers},{current_dist / openmmunit.nanometers},{force_val / openmmunit.kilojoules_per_mole * openmmunit.nanometer},{work_val / openmmunit.kilojoules_per_mole}\n"
                )
            f.close()

            # _print_current_forces(system)

            # Save state in PDB file
            final_positions = simulation.context.getState(
                getPositions=True
            ).getPositions()
            save_pdb(
                self.topology,
                final_positions,
                f"{self.out_dir}/steeredMD_{rep_idx}.pdb",
            )

            # Get statistics related to the pooling and plot them
            files = glob(f"{self.out_dir}/*.dat")
            data = extract_sMD_statistics(files)
            sys_name = self.out_dir.split("/")[0]
            plot_sMD_statistics(data, sys_name, self.out_dir)

        simulation_time = time.monotonic() - start_time
        logging.info(
            f"Finished {replicas} replicas of sMD in {simulation_time/60:.2f} min."
        )
