import os
import math
import time
import logging
from glob import glob
from copy import deepcopy
import numpy as np
from datetime import datetime

from autopath.utils import *
from autopath.customForces import (add_harmonic_restraints, remove_openmm_force)

from openmm.app import *
import openmm.unit as openmmunit

import cvpack
import logging
logger = logging.getLogger("autopath")

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    girsanov = False
    logger.warning("Please install openmmtools to use Girsanov reweighting.")
    
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
        autostop_freq: int = None,  # If None, do not stop pulling
        use_NVT: bool = False,  # Use NVT ensemble
        use_GReweighting: bool = False,
        out_dir: str = None,
        platform: str = "fastest",
        dx_per_move: float = 0.001,        # nm — RC grid spacing (held constant across speeds)
        max_displacement: float = 5.0,     # nm — total RC range
        sMD_spring_cte: float = 10000,     # kJ/mol/nm^2
        save_freq: int = 5,                # writes DCD every save_freq*steps_per_move
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

        self.verbose = verbose
        self.autostop_freq = autostop_freq  # In moves. Stop pulling if the ligand is unbound
        self.use_NVT = use_NVT

        self.integrator_friction = 1.0 / openmmunit.picoseconds  # Friction coefficient for Langevin integrator
        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            if not girsanov:
                logger.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logger.info("Using Girsanov reweighting for steered MD.")

        self.platform = select_platform(platform)

        # Sweep-fixed parameters: identical for every replica/speed in a sweep,
        # so they live on the instance rather than on .run().
        self.max_displacement = float(max_displacement)                                                 # nm
        self.dx_per_move = float(dx_per_move) * openmmunit.nanometers                                   # Quantity
        self.sMD_spring_cte = float(sMD_spring_cte) * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        self.save_freq = int(save_freq)
        self.sMD_moves = int(math.ceil(self.max_displacement / float(dx_per_move)))

        # Thermal fluctuation of the harmonic restraint: sigma = sqrt(kT/k).
        # If dx_per_move is comparable to sigma the protocol stops looking smooth.
        kB_kJ_per_mol_K = 0.0083144621
        T_K = self.temperature.value_in_unit(openmmunit.kelvin)
        k_spring = self.sMD_spring_cte.value_in_unit(
            openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        )
        self.sigma_thermal = math.sqrt(kB_kJ_per_mol_K * T_K / k_spring)  # nm
        if float(dx_per_move) > 0.5 * self.sigma_thermal:
            logger.warning(
                f"dx_per_move ({dx_per_move:.4g} nm) exceeds 0.5*sigma_thermal "
                f"({0.5*self.sigma_thermal:.4g} nm) for k={k_spring:.1f} kJ/mol/nm^2, T={T_K:.1f} K. "
                f"Consider a stiffer spring or smaller dx_per_move."
            )

        return None

    def pull_single_direction(self, 
                              simulation, 
                              run_id: int, 
                              direction: str = "forward",
                             ):
        """Run the pulling process in a single direction (forward or backward) for a single replica."""
                    
        add_reporters(simulation, self.out_dir, f"sMD_{run_id}",
            total_steps=self.sMD_moves*self.steps_per_move, # total steps 
            logperiod=self.save_freq*self.steps_per_move, # steps
            verbose=0 #verbose level
        )

        if self.use_GReweighting:
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/sMD_{run_id}.dat", 
                                                            self.steps_per_move, 
                                                            simulation.integrator, 
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        # Set the initial r0 parameter
        initial_r0 = self.com_dist.getValue(simulation.context, allowReinitialization=False)
        logger.info(f"Initial COM distance: {initial_r0}")
        simulation.context.setParameter("r0_smd", initial_r0)

        with open(f"{self.out_dir}/sMD_{run_id}.dat","w") as f:
            f.write("step,time,r_target,r_before,r_after,NC,force,U_cvpack,dW_protocol,m_eff\n")
            
            r_before_nm = 0.0
            r_after_nm = 0.0
            m_eff_dalton = 0.0
            nc_now = 0.0
            
            if self.autostop_freq is not None and direction == "forward":
                nc_now = self.nc_cv.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.dimensionless)
            
            # Loop over the number of moves
            for i in range(self.sMD_moves):

                r_before = self.com_dist.getValue(simulation.context, allowReinitialization=False)
                time_before = simulation.context.getState().getTime().value_in_unit(openmmunit.picoseconds)
                U_pre_old = self.com_force.getValue(simulation.context, allowReinitialization=False)

                if direction == "backward":
                    r_target = initial_r0 - (i+1)*self.dx_per_move
                    if r_target.value_in_unit(openmmunit.nanometers) <= 0.0:
                        logger.warning(f"Stopping backward pulling: r_target reached 0 at move {i}.")
                        break
                else:
                    r_target = initial_r0 + (i+1)*self.dx_per_move

                simulation.context.setParameter("r0_smd", r_target)

                delta = r_before - r_target
                force = - self.sMD_spring_cte * delta # F = -k(x - x0) Kj/mol/nm

                # get the potential energy of the spring from the COM CV
                U_cvpack = self.com_force.getValue(simulation.context, allowReinitialization=False) # kJ/mols
                dW_protocol = U_cvpack - U_pre_old

                simulation.step(self.steps_per_move)

                r_after_nm = self.com_dist.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.nanometers)

                #log everything
                r_target_nm = r_target.value_in_unit(openmmunit.nanometers)
                r_before_nm = r_before.value_in_unit(openmmunit.nanometers)
                force_kjmnm = force.value_in_unit(openmmunit.kilojoules_per_mole / openmmunit.nanometer)
                U_cvpack_kjm = U_cvpack.value_in_unit(openmmunit.kilojoules_per_mole)
                dW_protocol_kjm = dW_protocol.value_in_unit(openmmunit.kilojoules_per_mole)

                # Check if the ligand is unbound. Only for forward pulling
                # Check the distance is 0 for the backward pulling
                if self.autostop_freq is not None and i%self.autostop_freq == 0:
                    if direction == "forward":
                        nc_now = self.nc_cv.getValue(simulation.context, allowReinitialization=False).value_in_unit(openmmunit.dimensionless)
                        print(f"Step {i+1}/{self.sMD_moves}: r_target={r_target_nm:.2f} nm, r_before={r_before_nm:.2f} nm, r_after={r_after_nm:.2f} nm, nc={nc_now}")
                        if nc_now < 1:
                            f.write(f"{i},{time_before},{r_target_nm},{r_before_nm},{r_after_nm},{nc_now},{force_kjmnm},{U_cvpack_kjm},{dW_protocol_kjm},{m_eff_dalton}\n")
                            logger.warning(f"Stopping pulling at step {i} because n_contacts={nc_now}.")
                            break
                    else:
                        print(f"Step {i+1}/{self.sMD_moves}: r_target={r_target_nm:.2f} nm, r_before={r_before_nm:.2f} nm, r_after={r_after_nm:.2f} nm")
                        if r_before_nm < 0.05:
                            f.write(f"{i},{time_before},{r_target_nm},{r_before_nm},{r_after_nm},{nc_now},{force_kjmnm},{U_cvpack_kjm},{dW_protocol_kjm},{m_eff_dalton}\n")
                            logger.warning(f"Stopping backward pulling at step {i} with r_target={r_target_nm:.2f} nm and r_before={r_before_nm:.2f} nm.")
                            break
                f.write(f"{i},{time_before},{r_target_nm},{r_before_nm},{r_after_nm},{nc_now},{force_kjmnm},{U_cvpack_kjm},{dW_protocol_kjm},{m_eff_dalton}\n")
                
        # Save final positions
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_pdb(self.topology, final_positions, f"{self.out_dir}/sMD_{run_id}.pdb")
        return
    

    def run(self,
        pulling_speed: float = 0.001,           # nm/ps — varies per replica/call
        pulling_direction: str = "forward",
        run_id: str = None,
        checkpoint_file: str = None,
        pdb_file: str = None,
    ):
        """Run one steered MD replica at the requested speed.

        Sweep-fixed parameters (dx_per_move, max_displacement, sMD_spring_cte,
        save_freq) are set in __init__. Only the speed-dependent bookkeeping
        (steps_per_move, realized_speed) is computed here.
        """

        if pulling_direction not in ["forward", "backward"]:
            raise ValueError("pulling_direction must be either 'forward' or 'backward'.")

        # ensure proper formating of the file name
        if run_id is None:
            timestamp = datetime.now().strftime("%H%M%S")
            run_id = f"replica-{timestamp}_v{pulling_speed}_{pulling_direction}"
        else:
            run_id = f"replica-{run_id}_v{pulling_speed}_{pulling_direction}"

        simulation_start_time = time.monotonic()

        # Speed-dependent derivation. dx_per_move and sMD_moves are set in __init__
        # (RC grid is held constant across speeds so the analysis-side protocol
        # grid is identical between replicas of different speeds).
        dt_ps = self.timestep.value_in_unit(openmmunit.picoseconds)
        dx_nm = self.dx_per_move.value_in_unit(openmmunit.nanometers)
        pulling_speed = float(pulling_speed)
        steps_per_move_float = dx_nm / pulling_speed / dt_ps
        self.steps_per_move = max(1, int(round(steps_per_move_float)))
        realized_speed = dx_nm / (self.steps_per_move * dt_ps)

        if steps_per_move_float < 1.5:
            logger.warning(
                f"steps_per_move clipped to 1 (requested {steps_per_move_float:.3f}). "
                f"Speed {pulling_speed:g} nm/ps is too fast for dx_per_move={dx_nm:g} nm "
                f"and dt={dt_ps:g} ps; realized speed will be {realized_speed:g} nm/ps."
            )
        elif abs(realized_speed - pulling_speed) / pulling_speed > 0.05:
            logger.warning(
                f"Realized speed {realized_speed:.5g} nm/ps differs from "
                f"requested {pulling_speed:.5g} nm/ps by >5% due to rounding of "
                f"steps_per_move ({steps_per_move_float:.3f} -> {self.steps_per_move})."
            )

        ########################################################################################
        logger.info("#"*80)
        logger.info(f"Steered MD parameters for {run_id}:")
        logger.info(f"Pulling direction: {pulling_direction}")
        logger.info(f"Pulling speed: {pulling_speed} nm/ps (realized: {realized_speed:.5g} nm/ps)")
        logger.info(f"dx_per_move: {dx_nm:.4f} nm (sigma_thermal: {self.sigma_thermal:.4f} nm)")
        logger.info(f"steps_per_move: {self.steps_per_move} steps")
        logger.info(f"Time per move: {self.steps_per_move * dt_ps:.3f} ps")
        logger.info(f"Max displacement: {self.max_displacement:.3f} nm, total sMD moves: {self.sMD_moves}")
        logger.info(f'Saving DCD every {self.save_freq*self.steps_per_move} steps')
        logger.info("#"*80)
        ########################################################################################
        
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
        if checkpoint_file is None and pdb_file is None:
            logger.error("Either pdb_file or checkpoint_file must be provided to set initial positions.")
            exit(1)
        elif checkpoint_file is None and pdb_file is not None:
            logger.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)

        elif checkpoint_file is not None and pdb_file is None:
            logger.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator  # Replace the integrator with the new one
        else:
            # if both are provided, use the checkpoint file but warn the user
            logger.warning("Both checkpoint_file and pdb_file are provided. Using checkpoint_file.")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator

        # Add harmonic positional restraints to protein CA.
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        if self.restrained_atoms is not None:
            add_harmonic_restraints(system, input_positions, 
                                    self.topology, self.restrained_atoms,
                                    restraint_force=100,
                                    force_name="k_CA",
                                    force_group=14,
                                )
            
        # system.setDefaultPeriodicBoxVectors(*PDBFile(pdb_file).topology.getPeriodicBoxVectors())
        # simulation.context.reinitialize(preserveState=True)
        
        # Get the subset of protein atoms close to the ligand
        subset_protein_HA, subset_protein_residues = self._get_pocket_atoms(simulation, cutoff=0.5)
        if self.autostop_freq is not None and pulling_direction == "forward":
            forces = {f.getName(): f for f in system.getForces()}
            self.nc_cv = cvpack.NumberOfContacts(
                self.groupA_atoms,
                subset_protein_HA,
                forces["NonbondedForce"],
                stepFunction="1/(1+x^6)",
                # stepFunction="step(1-x)",
                thresholdDistance=0.4*openmmunit.nanometers,  # nm
            )
            self.nc_cv.setForceGroup(19)  # Use a separate force group for the CV
            system.addForce(self.nc_cv)

        # Reset velocities to temperature. Check https://github.com/openmm/openmm/pull/259
        if self.restart_velocities:
            simulation.context.setVelocitiesToTemperature(self.temperature)
        
        if self.use_NVT:
            # Remove existing MonteCarloBarostat to run NVT
            system = remove_openmm_force(system, "MonteCarloBarostat")

        # Reinitialize the simulation context with the updated system
        simulation.context.reinitialize(preserveState=True)

        #run a super short simulation to ensure the system is stable after temp reset
        simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps
        
        # try:
        #     simulation.integrator.reset()  # Reset the integrator. Only openmmtools integrators have this method    
        # except AttributeError:
        #     logger.warning("Integrator does not have reset method. This is expected for standard OpenMM integrators.")

        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setStepCount(0)  # reset step count
        
        weighByMass = True
        if len(self.groupA_atoms) == 1 or len(self.groupB_atoms) == 1:
            weighByMass = False  # avoid problems with single DUM massless atom
        
        # Add COM force to the ligand and pocket groups with a harmonic potential shape
        groups = [self.groupA_atoms] + [self.groupB_atoms]
        self.com_force = cvpack.CentroidFunction(
            "0.5 * fc_pull * (distance(g1,g2)-r0_smd)^2",
            openmmunit.kilojoules_per_mole,  # energy not force
            groups,
            weighByMass=weighByMass,
            pbc=True,
        )
        
        self.com_force.addGlobalParameter("r0_smd", 0)
        self.com_force.addGlobalParameter('fc_pull', self.sMD_spring_cte)
        self.com_force.setForceGroup(1)  # Use a separate force group for the CV GROUP 1
        system.addForce(self.com_force)

        # Add COM distance function to measure the distance between the two groups
        self.com_dist = cvpack.CentroidFunction(
            "distance(g1,g2)",
            openmmunit.nanometers,  # distance not energy
            groups,
            weighByMass=weighByMass,
            pbc=True,
        )
        
        self.com_dist.setForceGroup(3)  # Use a separate force group for the CV GROUP 3
        system.addForce(self.com_dist)

        simulation.context.reinitialize(preserveState=True)

        # Run the actual pulling
        self.pull_single_direction(simulation, run_id, direction=pulling_direction)

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logger.info(f"Finished {pulling_direction} sMD simulation in {simulation_time/60:.2f} min.")

        return run_id

    @staticmethod
    def adapt_speed_from_meff(m_eff_dalton, dx_target_nm=0.01, gammaL_ps=2.0,
                          v_min=0.0005, v_max=0.01):
        """Heuristically adapt the pulling speed based on effective mass"""
        v = dx_target_nm * gammaL_ps / m_eff_dalton  # in nm/ps
        return min(max(v, v_min), v_max)
    
    def _get_pocket_atoms(self, simulation, cutoff=0.6):
        """ Get a subset of protein heavy atoms within cutoff nm of the ligand heavy atoms.
        This is used to define the pocket for contact calculations and autostopping.

        Args:
            simulation (_type_): 
            cutoff (float, optional): Defaults to 0.6.

        Returns:
            _type_: subset_protein_HA, subset_protein_residues
        """

        protein_atoms = [atom for atom in self.topology.atoms() if atom.residue.name not in ["HOH", "WAT", "SOL", "NA", "CL", "K","MG", 'UNK']]
        protein_HA = [atom.index for atom in protein_atoms if atom.element.symbol != "H"]  # Exclude hydrogens
        
        # Find a subset of protein_HA that are cutoff nm away from groupA_atoms
        state = simulation.context.getState(getPositions=True, getVelocities=False)
        positions = state.getPositions(asNumpy=True) / openmmunit.nanometers
        ligand_pos = positions[self.groupA_atoms]
        pocket_pos = positions[protein_HA]

        distances = np.linalg.norm(ligand_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :], axis=2)
        close_indices = np.any(distances < cutoff, axis=0)
        subset_protein_HA = np.array(protein_HA)[close_indices]
        subset_protein_residues = [atom.residue for atom in protein_atoms if atom.index in subset_protein_HA]
        
        return subset_protein_HA, subset_protein_residues

