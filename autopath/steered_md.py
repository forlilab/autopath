import os
import math
import time
import logging
from glob import glob
from copy import deepcopy

from autopath.utils import *
from autopath.customForces import (add_harmonic_restraints, 
                                   print_current_forces, 
                                   remove_openmm_force)

from openmm.app import *
import openmm.unit as openmmunit

import cvpack

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    girsanov = False
    logging.warning("Please install openmmtools to use Girsanov reweighting.")
    
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
        restart_velocities: bool = False,
        timestep: float = 0.004, #  # 4 fs timestep
        temperature: float = 300,
        use_NVT: bool = True,  # Use NVT ensemble
        use_GReweighting: bool = False,
        out_dir: str = None,
        verbose: int = 0,
    ):
        self.system = system
        self.topology = topology
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temperature * openmmunit.kelvin
        self.groupA_atoms = groupA_atoms # ligand atoms
        self.groupB_atoms = groupB_atoms # pocket atoms
        self.restrained_atoms = restrained_atoms
        self.restart_velocities = restart_velocities

        self.atoms = [atom for atom in self.topology.atoms()]
        protein_atoms = [atom for atom in self.topology.atoms() if atom.residue.name not in ["HOH", "WAT", "SOL", "NAC", "CL"]]
        self.protein_HA = [atom.index for atom in protein_atoms if atom.element.symbol != "H"]  # Exclude hydrogens
        
        self.verbose = verbose
        self.autostop_freq = 50  # In moves. Stop pulling if the ligand is unbound
        self.use_NVT = use_NVT

        self.integrator_friction = 2.0 / openmmunit.picoseconds  # Friction coefficient for Langevin integrator
        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            if not girsanov:
                logging.error("Girsanov reweighting is enabled but openmmtools is not installed.")
                self.use_GReweighting = False
            logging.info("Using Girsanov reweighting for steered MD.")

        self.platform = select_platform("fastest")

        return None

    def pull_single_direction(self, simulation, 
                              rep_idx: int, 
                              direction: str = "forward",
                             ):
        """Run the pulling process in a single direction (forward or backward) for a single replica."""
        
        add_reporters(simulation, self.out_dir, f"sMD_{rep_idx}_{direction}",
            self.sMD_moves*self.steps_per_move, # total steps 
            self.steps_per_move,
            0 #verbose level
        )

        if self.use_GReweighting:
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/GR_{rep_idx}_{direction}.dat", 
                                                            self.steps_per_move, 
                                                            simulation.integrator, 
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        # Set the initial r0 parameter
        initial_r0 = self.com_dist.getValue(simulation.context, allowReinitialization=False)
        logging.info(f"Initial COM distance: {initial_r0}")
        simulation.context.setParameter("r0", initial_r0)

        with open(f"{self.out_dir}/sMD_log_{rep_idx}_{direction}.dat","w") as f:

            f.write("step,r_target(nm),r_before(nm),r_after(nm),force(kJ/mol/nm),work(kJ/mol),m_eff(dalton)\n")

            work = 0.0 * openmmunit.kilojoules_per_mole
            dist_after_nm = 0.0 
            dist_before_nm = 0.0
            m_eff_dalton = 0.0

            # Loop over the number of moves
            for i in range(self.sMD_moves):

                dist_before = self.com_dist.getValue(simulation.context, allowReinitialization=False)
                m_eff_dalton = self.com_dist.getEffectiveMass(simulation.context).value_in_unit(openmmunit.dalton)

                # Compute new r_end
                if direction == "backward":
                    r_end = initial_r0 - (i+1)*abs(self.dx_per_move)
                else:
                    r_end = initial_r0 + (i+1)*self.dx_per_move

                simulation.context.setParameter("r0", r_end)

                delta = dist_before - r_end
                force = - self.sMD_spring_cte * delta
                # print("force", force) # force in kJ/mol/nm          

                # get the force from the COM CV
                # force_cvpack = self.com_force.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.kilojoules_per_mole)
                # print("force_cvpack", force_cvpack) # force in kJ/mol/nm          
                
                # increment the work -v do not use (dist_after - dist_before), use the expected displacement dx_per_move
                work += (force * self.dx_per_move)#.value_in_unit(openmmunit.kilojoules_per_mole)  # kJ/mol
                # work_cvpack += force_cvpack * dx_per_move.value_in_unit(openmmunit.nanometers) # kJ/mol

                # run for steps_per_move
                simulation.step(self.steps_per_move)

                # actual distance after
                if self.verbose > 0:
                    dist_after_nm = self.com_dist.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.nanometers)
                    dist_before_nm = dist_before.value_in_unit(openmmunit.nanometers)
                
                #log everything
                #shadow_work = simulation.integrator.get_shadow_work().value_in_unit(openmmunit.kilojoules_per_mole)
                #protocol_work = simulation.integrator.get_protocol_work().value_in_unit(openmmunit.kilojoules_per_mole)
                r_end_nm = r_end.value_in_unit(openmmunit.nanometers)
                force_kjmnm = force.value_in_unit(openmmunit.kilojoules_per_mole / openmmunit.nanometer)
                work_kjmol = work.value_in_unit(openmmunit.kilojoules_per_mole)

                f.write(f"{i},{r_end_nm},{dist_before_nm},{dist_after_nm},{force_kjmnm},{work_kjmol},{m_eff_dalton}\n")

                # Check if the ligand is unbound
                if self.autostop_freq is not None and i%self.autostop_freq == 0:
                    nc_now = self.nc_cv.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.dimensionless)
                    logging.debug(f"Step {i+1}/{self.sMD_moves}: r_target={r_end_nm:.2f} nm, r_before={dist_before_nm:.2f} nm, r_after={dist_after_nm:.2f} nm, nc={nc_now}")
                
                    if nc_now < 1:
                        logging.warning(f"Stopping pulling at step {i} because the ligand unbound with n_contacts={nc_now}.")
                        break   

        #make sure to reset the integrator
        try:
            simulation.integrator.reset() #only openmmtools integrators have this method
        except AttributeError:
            logging.warning("Integrator does not have reset method. This is expected for standard OpenMM integrators.")
            pass

        # Log final COM distance
        final_dist_nm = self.com_dist.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.nanometers)
        logging.info(f"Final COM distance: {final_dist_nm:.2f} nm")

        # Save final positions
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_pdb(self.topology, final_positions, f"{self.out_dir}/steeredMD_{rep_idx}_{direction}.pdb")
        return
    

    def run(self,
        max_displacement: float = 3.0,  # nm
        steps_per_move: int = 50,
        pulling_speed: float = 0.001,  # nm/ps equi 1 nm/ns
        sMD_spring_cte: int = 1000, # kJ/mol/nm^2
        rep_suffix: str = None,
        checkpoint_file: str = None,
        pdb_file: str = None,
        do_backwards: bool = False,
    ):
        """Main method to run steered MD in both directions (forward and backward)."""
        simulation_start_time = time.monotonic()

        timestep_ps = self.timestep.value_in_unit(openmmunit.picoseconds)
        params = self.compute_smd_params(
            pulling_speed=pulling_speed,
            steps_per_move=steps_per_move,
            timestep_ps=timestep_ps,
            max_displacement=max_displacement,
        )

        self.sMD_spring_cte = sMD_spring_cte * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        self.sMD_moves = params['sMD_moves']
        self.steps_per_move = steps_per_move
        self.dx_per_move = params['dx_per_move'] * openmmunit.nanometers   # Quantity with units

        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = self.steps_per_move,   
                temperature = self.temperature,
                collision_rate = self.integrator_friction,
                timestep = self.timestep,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            # This is the default openMM but do not track work
            # dicussion https://github.com/openmm/openmm/issues/2520
            integrator = LangevinMiddleIntegrator(self.temperature, 
                                                self.integrator_friction, 
                                                self.timestep
                                                # constraint_tolerance = 1.0e-6
                                                )
                    
        system = deepcopy(self.system)  # Create a copy of the system to avoid modifying the original
        simulation = Simulation(self.topology, system, integrator, self.platform)

        #If the systems was equilibrated with a different integrator I get NaNs (even with same splitting)
        # so Im using the PDB instead of the checkpoint file
        if checkpoint_file is None and pdb_file is not None:
            logging.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)

        elif checkpoint_file is not None:
            logging.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator  # Replace the integrator with the new one

        # Add harmonic positional restraints to protein CA
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        if self.restrained_atoms is not None:
            add_harmonic_restraints(system, input_positions, 
                                    self.topology, self.restrained_atoms,
                                    restraint_force=10,
                                    force_name="k_CA",
                                    force_group=14,
                                )
            
        # system.setDefaultPeriodicBoxVectors(*PDBFile(pdb_file).topology.getPeriodicBoxVectors())
        # simulation.context.reinitialize(preserveState=True)

        # Add COM force to the ligand and pocket groups
        if self.autostop_freq is not None:
            forces = {f.getName(): f for f in system.getForces()}
            self.nc_cv = cvpack.NumberOfContacts(
                self.groupA_atoms,
                self.groupB_atoms,
                forces["NonbondedForce"],
                # stepFunction="1/(1+x^6)",
                stepFunction="step(1-x)",
                # TODO This should be adaptaed based on the pocket definition or consider the whole protein but that may be too slow?
                thresholdDistance=1.1,  # nm
            )
            self.nc_cv.setForceGroup(18)  # Use a separate force group for the CV
            system.addForce(self.nc_cv)

        #Reset velocities to temperature. Check https://github.com/openmm/openmm/pull/259
        if self.restart_velocities:
            simulation.context.setVelocitiesToTemperature(self.temperature)
        
        if self.use_NVT:
            # Remove existing MonteCarloBarostat to run NVT
            system = remove_openmm_force(system, "MonteCarloBarostat")

        # Reinitialize the simulation context with the updated system
        simulation.context.reinitialize(preserveState=True)

        #run a super short simulation to ensure the system is stable after temp reset
        simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps
        
        try:
            simulation.integrator.reset()  # Reset the integrator. Only openmmtools integrators have this method    
        except AttributeError:
            logging.warning("Integrator does not have reset method. This is expected for standard OpenMM integrators.")

        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setStepCount(0)  # reset step count
        
        # Add COM force to the ligand and pocket groups with a harmonic potential shape
        groups = [self.groupA_atoms] + [self.groupB_atoms]
        self.com_force = cvpack.CentroidFunction(
            "0.5 * fc_pull * (distance(g1,g2)-r0)^2",
            openmmunit.kilojoules_per_mole,  # energy not force
            groups,
            weighByMass=True if len(self.groupB_atoms) > 1 else False, # avoid problems with single DUM massless atoms
            pbc=True,
        )
        
        self.com_force.addGlobalParameter("r0", 0)
        self.com_force.addGlobalParameter('fc_pull', self.sMD_spring_cte)
        self.com_force.setForceGroup(1)  # Use a separate force group for the CV GROUP 1
        system.addForce(self.com_force)

        # Add COM distance function to measure the distance between the two groups
        self.com_dist = cvpack.CentroidFunction(
            "distance(g1,g2)",
            openmmunit.nanometers,  # distance not energy
            groups,
            weighByMass=True if len(self.groupB_atoms) > 2 else False, # avoid problems with single DUM massless atoms
            pbc=True,
        )
        
        self.com_dist.setForceGroup(3)  # Use a separate force group for the CV GROUP 3
        system.addForce(self.com_dist)

        simulation.context.reinitialize(preserveState=True)

        rep_name = rep_suffix if rep_suffix else f"replica-{np.random.randint(1000000)}_v{pulling_speed}"

        # Run forward direction
        self.pull_single_direction(simulation, rep_name, direction="forward")

        if do_backwards:
            logging.info(f"Running backward pulling of replica {rep_suffix} at speed {pulling_speed} nm/ps")
            # Reset velocities to temperature after forward pulling
            simulation.context.setVelocitiesToTemperature(self.temperature)
            # run a super short simulation to ensure the system is stable after temp reset
            simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps

            try:
                simulation.integrator.reset()  # Reset the integrator. Only openmmtools integrators have this method    
            except AttributeError:
                logging.warning("Integrator does not have reset method. This is expected for standard OpenMM integrators.")

            simulation.context.setTime(0)  # reset simulation time
            simulation.context.setStepCount(0)  # reset step count

            self.pull_single_direction(simulation, rep_name, direction="backward")

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logging.info(f"Finished sMD simulation {'with backwards pulling' if do_backwards else ''} in {simulation_time/60:.2f} min.")

    @staticmethod
    def guess_steps_per_move(v_nm_per_ps,
                            dt_ps=0.004,
                            k_spring=1000,   # kJ/mol/nm**2
                            T_K=300,
                            Rmax=0.3,
                            verbose=True
                            ):
        """This funcion calculates the number of steps per move for sMD based on the velocity,
        It assumes a harmonic potential and calculates the thermal fluctuation of the CV.
        Then uses the Rmax relation with is the maximum displacement per move relative to that sigma.
        If steps_per_move jump is much smaller than sigma, the atoms cannot tell that 
        the restraint was moved in discrete steps, they see an effectively 
        continuous constant-velocity bias, which is the idea"""

        kB = 0.0083144621          # kJ/mol/K
        sigma = (kB*T_K/k_spring)**0.5
        t_move_ps = Rmax * sigma / v_nm_per_ps
        steps_per_move = max(1, int(t_move_ps / dt_ps))

        if verbose:
            print(f"Guessing steps_per_move for sMD with parameters:")
            print(f'Pulling speed = {v_nm_per_ps:.4f} nm/ps')
            print(f'Timestep = {dt_ps:.4f} ps')
            print(f"Thermal fluctuation σ = {sigma:.3f} nm")
            print(f"Max displacement per move = {Rmax*sigma:.6f} nm")
            print(f"Required time per move = {t_move_ps:.2f} ps")
            print(f"steps_per_move = {steps_per_move}")

        return steps_per_move

    @staticmethod
    def compute_smd_params(
        pulling_speed: float, # nm / ps
        steps_per_move: int, #50
        timestep_ps: float, # timestep in ps 0.004
        max_displacement: float = 2.5, # nm
    ) -> dict:
        """
        Returns:
            {
                'sMD_time': float (ps),
                'sMD_moves': int,
                'dx_per_move': Quantity (nm),
                'sMD_steps': int,
            }
        """

        time_per_move = steps_per_move * timestep_ps     # ps
        dx_per_move = pulling_speed * time_per_move
        sMD_moves = math.ceil(max_displacement / dx_per_move)  # number of moves
        sMD_steps = sMD_moves * steps_per_move
        sMD_time = sMD_steps * timestep_ps      # ps

        # print(f"Steered MD parameters: sMD_time={sMD_time:.2f} ps, sMD_moves={sMD_moves}, dx_per_move={dx_per_move:.4f} nm, sMD_steps={sMD_steps}")

        logging.info(f"Max displacement: {max_displacement} nm")
        logging.info(f"Pulling speed: {pulling_speed:.4f} nm/ps")
        logging.info(f"Steps per move: {steps_per_move} steps")
        logging.info(f"Time per move: {steps_per_move * timestep_ps:.3f} ps")
        logging.info(f"Displacement per move: {dx_per_move} nm")
        logging.info(f"Total sMD time: {sMD_time:.2f} ps, Moves: {sMD_moves}, Total steps: {sMD_steps}")

        print(f"Steered MD parameters: sMD_time={sMD_time:.2f} ps, sMD_moves={sMD_moves}, sMD_steps={sMD_steps}, dx_per_move={dx_per_move:.4f} nm, pulling_speed={pulling_speed:.4f} nm/ps, max_displacement={max_displacement:.2f} nm")

        return {
            'sMD_time': sMD_time,
            'sMD_moves': sMD_moves,
            'dx_per_move': dx_per_move,
            'sMD_steps': sMD_steps,
        }

    @staticmethod
    def adapt_speed_from_meff(m_eff_dalton, dx_target_nm=0.01, gammaL_ps=2.0,
                          v_min=0.0005, v_max=0.01):
        """Heuristically adapt the pulling speed based on effective mass"""
        v = dx_target_nm * gammaL_ps / m_eff_dalton  # in nm/ps
        return v
        # return min(max(v, v_min), v_max)