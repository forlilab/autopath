import os
import math
import time
import logging
from glob import glob

from autopath.utils import *
from autopath.customForces import add_COM_force, add_harmonic_restraints, print_current_forces
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
        restart_velocities: bool = False,
        HMR: bool = True,
        temperature: float = 300,
        use_GReweighting: bool = True,
        out_dir: str = None,
        verbose: int = 2,
    ):
        self.system = system
        self.topology = topology
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.restart_velocities = restart_velocities
        self.timestep = 0.004 if HMR else 0.002
        self.temperature = temperature * openmmunit.kelvin
        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        self.restrained_atoms = restrained_atoms
        self.verbose = verbose

        self.platform = select_platform("fastest")

        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            try:
                from openmmtools.integrators import LangevinSplittingGirsanov
                from reweightingreporter import ReweightingReporter
            except ImportError:
                raise ImportError("Please install openmmtools to use Girsanov reweighting.")
    
        return None

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
            steps_per_move,
            self.verbose
        )

        if self.use_GReweighting:
            from reweightingreporter import ReweightingReporter
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/GR_{rep_idx}_{direction}.dat", 
                                                            steps_per_move, 
                                                            self.integrator, 
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        logging.info(f"Initial COM distance: {initial_r0}")
        simulation.context.setParameter("r0", initial_r0)

        # Initialize work
        work_val_old = openmmunit.Quantity(value=0, unit=openmmunit.kilojoules_per_mole)
        f = open(f"{self.out_dir}/sMD_log_{rep_idx}_{direction}.dat", "a")

        for i in range(sMD_moves):
            current_dist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms) * openmmunit.nanometers

            # Get distance of starting point and end point
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
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_pdb(self.topology, final_positions, f"{self.out_dir}/steeredMD_{rep_idx}_{direction}.pdb")

    def run(
        self,
        sMD_time: float = 1,  # 1ns
        displacement: float = 2.5,  # nm
        steps_per_move: int = 250,  # 1ps
        pulling_force: int = 1000,
        replicas: int = 3,
        rep_suffix: str = None,
        checkpoint_file: str = None,
        do_backwards: bool = False,
    ):
        """Main method to run steered MD in both directions (forward and backward) for multiple replicas."""
        simulation_start_time = time.monotonic()

        # Calculate the number of steps
        sMD_steps = math.ceil(sMD_time / self.timestep * 1000.0)
        sMD_moves = int(sMD_steps / steps_per_move)
        dx_per_move = (displacement / sMD_moves) * openmmunit.nanometers

        self.fc_pull = pulling_force * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2

        if self.use_GReweighting:
            from openmmtools.integrators import LangevinSplittingGirsanov
            from reweightingreporter import ReweightingReporter
            self.integrator = LangevinSplittingGirsanov(
                nstxout = steps_per_move,   
                temperature = self.temperature,
                collision_rate = 1.0/openmmunit.picoseconds,
                timestep = self.timestep * openmmunit.picoseconds,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            self.integrator = LangevinMiddleIntegrator(self.temperature, 1 / openmmunit.picoseconds, self.timestep)

        simulation = Simulation(self.topology, self.system, self.integrator, self.platform)

        # Load checkpoint file
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
        add_COM_force(self.system, self.ligand_atoms, self.pocket_atoms, self.fc_pull, 0, 1) # groupd 1 for reweighting
        simulation.context.setTime(0)  # reset simulation time
        simulation.context.reinitialize(preserveState=True)
        
        # print_current_forces(self.system)

        # Loop over replicas
        for rep_idx in range(1, replicas + 1):
            rep_name = f"{rep_suffix}_{rep_idx}" if rep_suffix else f"replica_{rep_idx}"
            replica_start_time = time.monotonic()

            # Load checkpoint file
            simulation.loadCheckpoint(checkpoint_file)

            # Reset velocities to temperature
            if self.restart_velocities:
                simulation.context.setVelocitiesToTemperature(self.temperature)

            startdist = get_COM_dist(simulation, self.ligand_atoms, self.pocket_atoms)
            initial_r0 = startdist * openmmunit.nanometers
            final_r0 = initial_r0 + abs(dx_per_move) * sMD_moves
            
            # Run forward direction
            self.run_single_direction(
                simulation, rep_name, direction="forward", dx_per_move=dx_per_move, sMD_moves=sMD_moves,
                steps_per_move=steps_per_move, initial_r0=initial_r0, final_r0=final_r0
                )

            if self.verbose > 0:
                files_f = glob(f"{self.out_dir}/sMD_log_*_forward.dat")
                plot_sMD_statistics(files_f, f'{rep_name}_forward', self.out_dir)

            if do_backwards:
                # Run backward direction
                # Reset velocities to temperature after forward pulling
                simulation.context.setVelocitiesToTemperature(self.temperature)
                self.run_single_direction(simulation, rep_name, direction="backward", dx_per_move=-dx_per_move, sMD_moves=sMD_moves,
                                        steps_per_move=steps_per_move, initial_r0=initial_r0, final_r0=final_r0)

                if self.verbose > 0:
                    files_f = glob(f"{self.out_dir}/sMD_log_*_backward.dat")
                    plot_sMD_statistics(files_f, f'{rep_name}_backward', self.out_dir)

            # Logging the time taken for each replica
            replica_time = time.monotonic() - replica_start_time
            logging.info(f"Finished replica {rep_idx}/{replicas} in {replica_time/60:.2f} min")


        # Generate statistics and plots for the forward direction
        files_f = glob(f"{self.out_dir}/sMD_log_*_forward.dat")
        plot_sMD_statistics(files_f, f'{rep_name}_forward', self.out_dir)

        if do_backwards:
            # Generate statistics and plots for the backward direction
            files_b = glob(f"{self.out_dir}/sMD_log_*_backward.dat")
            plot_sMD_statistics(files_b, f'{rep_name}_backward', self.out_dir)

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logging.info(f"Finished {replicas} replicas of sMD {'with backwards pulling' if do_backwards else ''} in {simulation_time/60:.2f} min.")