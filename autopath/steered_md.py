import os
import math
import time
import logging
from glob import glob
from copy import deepcopy

from autopath.utils import *
from autopath.customForces import add_COM_force, add_harmonic_restraints, print_current_forces
from autopath.analysis import plot_sMD_statistics

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import cvpack

class SteeredMD:
    """
    A class to perform steered molecular dynamics, pulling a ligand out of its binding pocket
    """

    def __init__(
        self,
        system: str = None,
        topology: str = None,
        groupA_atoms: list[int] = None,
        groupB_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        restart_velocities: bool = True,
        timestep: float = 0.004, #  # 4 fs timestep
        temperature: float = 300,
        use_GReweighting: bool = False,
        out_dir: str = None,
        verbose: int = 2,
    ):
        self.system = system
        self.topology = topology
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.restart_velocities = restart_velocities
        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temperature * openmmunit.kelvin
        self.groupA_atoms = groupA_atoms # ligand atoms
        self.groupB_atoms = groupB_atoms # pocket atoms
        self.restrained_atoms = restrained_atoms
        self.verbose = verbose
        self.autostop_freq = 10  # In moves. Stop pulling if the ligand is unbound

        self.platform = select_platform("fastest")

        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            try:
                from openmmtools.integrators import LangevinSplittingGirsanov
                from reweightingreporter import ReweightingReporter
            except ImportError:
                raise ImportError("Please install openmmtools to use Girsanov reweighting.")
    
        return None

    def pull_single_direction(self, simulation, rep_idx, direction, 
                             dx_per_move, sMD_moves, steps_per_move, pulling_force,
                             initial_r0, final_r0):
        """Run the pulling process in a single direction (forward or backward) for a single replica."""
        
        add_reporters(
            simulation,
            self.out_dir,
            f"sMD_{rep_idx}_{direction}",
            sMD_moves*steps_per_move, # total steps
            steps_per_move,
            self.verbose
        )

        if self.use_GReweighting:
            from reweightingreporter import ReweightingReporter
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/GR_{rep_idx}_{direction}.dat", 
                                                            steps_per_move, 
                                                            simulation.integrator, 
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        logging.info(f"Initial COM distance: {initial_r0}")
        simulation.context.setParameter("r0", initial_r0)

        with open(f"{self.out_dir}/sMD_log_{rep_idx}_{direction}.dat","w") as f:
            # f.write("step,r_target(nm),r_before(nm),r_after(nm),force(kJ/mol/nm),work(kJ/mol),shadow_work(KJ/mol)\n")
            f.write("step,r_target(nm),r_before(nm),r_after(nm),force(kJ/mol/nm),work(kJ/mol),shadow_work(kJ/mol),protocol_work(kJ/mol)\n")

            work_val = 0.0

            for i in range(sMD_moves):
                #actual distance before
                dist_before = get_COM_dist(simulation, self.groupA_atoms, self.groupB_atoms)

                #compute your new target distance
                if direction == "backward":
                    r_end = final_r0 - (i+1)*abs(dx_per_move)
                else:
                    r_end = initial_r0 + (i+1)*dx_per_move

                simulation.context.setParameter("r0", r_end)

                r_end = r_end.value_in_unit(openmmunit.nanometers)

                #force before the step
                force_val = -pulling_force * (dist_before - r_end)
                force_val = force_val * openmmunit.nanometer**2/openmmunit.kilojoules_per_mole # remove units for logging

                simulation.step(steps_per_move)

                #actual distance after
                dist_after = get_COM_dist(simulation, self.groupA_atoms, self.groupB_atoms)

                # accumulate work
                work_val += force_val * (dist_after - dist_before)
                
                #log everything
                shadow_work = simulation.integrator.get_shadow_work().value_in_unit(openmmunit.kilojoules_per_mole)
                protocol_work = simulation.integrator.get_protocol_work().value_in_unit(openmmunit.kilojoules_per_mole)
                f.write(f"{i},{r_end},{dist_before},{dist_after},{force_val},{work_val},{shadow_work},{protocol_work}\n")

                if self.autostop_freq is not None and i%self.autostop_freq == 0:  # Check every 5 moves approx 5ps
                    # Check if the ligand is unbound
                    nc_now = (self.nc_cv.getValue(simulation.context, allowReinitialization=True)).value_in_unit(openmmunit.dimensionless)
                    logging.debug(f"Step {i+1}/{sMD_moves}: r_target={r_end:.2f} nm, r_before={dist_before:.2f} nm, r_after={dist_after:.2f} nm, nc={nc_now}")
                
                    if nc_now < 1:
                        logging.info(f"Stopping pulling at step {i} because the ligand unbound with n_contacts={nc_now}.")
                        break   


        #make sure to reset the integrator
        simulation.integrator.reset() #only openmmtools integrators have this method

        # Log final COM distance
        final_dist = get_COM_dist(simulation, self.groupA_atoms, self.groupB_atoms)
        logging.info(f"Final COM distance: {final_dist:.2f} nm")

        # Save final positions
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_pdb(self.topology, final_positions, f"{self.out_dir}/steeredMD_{rep_idx}_{direction}.pdb")

    def run(
        self,
        sMD_time: float = 1,  # 1ns
        max_displacement: float = 2.0,  # nm
        steps_per_move: int = 250,  # 1ps
        pulling_speed: float = 0.002,  # nm/ps
        pulling_force: int = 1000,
        rep_suffix: str = None,
        checkpoint_file: str = None,
        do_backwards: bool = False,
    ):
        """Main method to run steered MD in both directions (forward and backward)."""
        simulation_start_time = time.monotonic()

        # Calculate the number of steps
        if pulling_speed is not None:
            sMD_time = max_displacement / pulling_speed/1000  # in ns
            sMD_steps = math.ceil(sMD_time / self.timestep.value_in_unit(openmmunit.picoseconds) * 1000.0)
            sMD_moves = int(sMD_steps / steps_per_move)
            # dx_per_move = (max_displacement / sMD_moves) * openmmunit.nanometers
            time_per_move = steps_per_move * self.timestep.value_in_unit(openmmunit.picoseconds)
            dx_per_move = pulling_speed * time_per_move * openmmunit.nanometers
        else:
            # If pulling speed is not defined, use displacement and time to calculate dx_per_move
            if sMD_time is not None and max_displacement is not None:
                dx_per_move = (max_displacement * openmmunit.nanometers) / (sMD_time * 1000 / steps_per_move)
            else:
                raise ValueError("Either pulling_speed or both sMD_time and max_displacement must be provided.")

        pulling_force = pulling_force * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2

        if self.use_GReweighting:
            from openmmtools.integrators import LangevinSplittingGirsanov
            from reweightingreporter import ReweightingReporter
            integrator = LangevinSplittingGirsanov(
                nstxout = steps_per_move,   
                temperature = self.temperature,
                collision_rate = 1.0/openmmunit.picoseconds,
                timestep = self.timestep,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            # integrator = LangevinMiddleIntegrator(self.temperature, 
            #                                            1.0/openmmunit.picoseconds, 
            #                                            self.timestep)

            from openmmtools.integrators import LangevinIntegrator, ExternalPerturbationLangevinIntegrator
            # integrator = LangevinIntegrator(self.temperature, 
            #                                     1.0/openmmunit.picoseconds, 
            #                                     self.timestep,
            #                                     measure_shadow_work=True,
            #                                     # constraint_tolerance=1.0e-6,
            #                                     )
            integrator = ExternalPerturbationLangevinIntegrator(self.temperature, 
                                        1.0/openmmunit.picoseconds, 
                                        self.timestep,
                                        measure_shadow_work=True,
                                        # constraint_tolerance=1.0e-6,
                                        )    
        system = deepcopy(self.system)  # Create a copy of the system to avoid modifying the original
        simulation = Simulation(self.topology, system, integrator, self.platform)

        # Load checkpoint file
        simulation.loadCheckpoint(checkpoint_file)
        
        # simulation.context.setDefaultPeriodicBoxVectors()
        # self.system.setDefaultPeriodicBoxVectors(*simulation.context.getState(getPositions=True).getPeriodicBoxVectors())
        # simulation.context.reinitialize(preserveState=True)

        # Add harmonic positional restraints to protein CA
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

        if self.autostop_freq is not None:
            forces = {f.getName(): f for f in system.getForces()}
            self.nc_cv = cvpack.NumberOfContacts(
                self.groupA_atoms,
                self.groupB_atoms,
                forces["NonbondedForce"],
                # stepFunction="1/(1+x^6)",
                stepFunction="step(1-x)",
                thresholdDistance=0.5,
                # cutoffFactor=2.0,
                # switchFactor=1.5,
                # reference=simulation.context
            )
            self.nc_cv.setForceGroup(18)  # Use a separate force group for the CV
            system.addForce(self.nc_cv)
            # self.nc_cv.addToSystem(self.system)
        
        #restart here to have harmonic restraints and NOT COM force in the system
        simulation.context.reinitialize(preserveState=True)

        # Reset velocities to temperature
        if self.restart_velocities:
            simulation.context.setVelocitiesToTemperature(self.temperature)
            # run a super short simulation to ensure the system is stable after temp reset
            simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps
            simulation.integrator.reset()  # Reset the integrator to avoid issues with reinitialization
            
        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setStepCount(0)  # reset step count

        # Add COM force with arbitrary initial r0, then run_single_direction will set it properly
        add_COM_force(system, self.groupA_atoms, self.groupB_atoms, pulling_force, 0, 1) # keep it in group 1 for GR reweighting
        simulation.context.reinitialize(preserveState=True)

        startdist = get_COM_dist(simulation, self.groupA_atoms, self.groupB_atoms)
        initial_r0 = startdist * openmmunit.nanometers
        final_r0 = initial_r0 + abs(dx_per_move) * sMD_moves
        
        rep_name = rep_suffix if rep_suffix else f"replica-{np.random.randint(1000000)}_v{pulling_speed}"

        logging.info(f"Forward pulling of replica {rep_suffix} at {pulling_speed} nm/ps for {max_displacement} nm in {sMD_time} ns")
        logging.info(f"dx_per_move: {dx_per_move.value_in_unit(openmmunit.nanometers)} nm, time_per_move: {time_per_move} ps, steps_per_move: {steps_per_move}, total moves: {sMD_moves}")

        # Run forward direction
        self.pull_single_direction(simulation, rep_name, direction="forward", 
                                  dx_per_move=dx_per_move, sMD_moves=sMD_moves,
                                  steps_per_move=steps_per_move, pulling_force=pulling_force,
                                  initial_r0=initial_r0, final_r0=final_r0)

        # if self.verbose > 0:
            # files_f = glob(f"{self.out_dir}/sMD_log_*_forward.dat")
            # plot_sMD_statistics(files_f, f'{rep_name}_forward', self.out_dir)

        if do_backwards:
            logging.info(f"Running backward pulling of replica {rep_suffix} at speed {pulling_speed} nm/ps")
            # Reset velocities to temperature after forward pulling
            simulation.context.setVelocitiesToTemperature(self.temperature)
            
            # run a super short simulation to ensure the system is stable after temp reset
            simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps
            self.pull_single_direction(simulation, rep_name, direction="backward", 
                                      dx_per_move=-dx_per_move, sMD_moves=sMD_moves,
                                      steps_per_move=steps_per_move, pulling_force=pulling_force,
                                      initial_r0=initial_r0, final_r0=final_r0)

            # if self.verbose > 0:
                # files_f = glob(f"{self.out_dir}/sMD_log_*_backward.dat")
                # plot_sMD_statistics(files_f, f'{rep_name}_backward', self.out_dir)

            # Logging the time taken for each replica
            # replica_time = time.monotonic() - replica_start_time
            # logging.info(f"Finished replica {rep_idx}/{replicas} in {replica_time/60:.2f} min")


        # Generate statistics and plots for the forward direction
        # files_f = glob(f"{self.out_dir}/sMD_log_*_forward.dat")
        # plot_sMD_statistics(files_f, f'{rep_name}_forward', self.out_dir)

        # if do_backwards:
            # Generate statistics and plots for the backward direction
            # files_b = glob(f"{self.out_dir}/sMD_log_*_backward.dat")
            # plot_sMD_statistics(files_b, f'{rep_name}_backward', self.out_dir)

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logging.info(f"Finished sMD simulation {'with backwards pulling' if do_backwards else ''} in {simulation_time/60:.2f} min.")