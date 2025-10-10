import time
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *
import datetime
import math

@dataclass
class EquilibrationStep:
    name: str
    forces: List[float]
    npt_flag: bool
    nsteps: int
    stepsize: float


def set_force_constants(
    simulation,
    components: List[str],
    forces: List[float],
    prev_constants: Dict[str, float],
) -> None:
    """Set the force constants for the given components."""
    for component, force_constant in zip(components, forces):
        if (
            prev_constants[component] is None
            or force_constant != prev_constants[component]
        ):
            simulation.context.setParameter(
                f"k_{component}",
                (
                    force_constant
                    * openmmunit.kilocalories_per_mole
                    / openmmunit.angstroms**2
                ),
            )
            prev_constants[component] = force_constant
    return


def warm_up_system(
    simulation,
    integrator,
    Tstart: int = 100,
    Tend: int = 300,
    Tstep: int = 5,
    timestep: float = 0.001,
    warming_steps: int = 100000,
) -> None:
    """
    Perform simulated annealing, slowly increasing the temperature
    to warm-up the system in the NVT ensemble.
    """

    integrator.setStepSize(timestep * openmmunit.picoseconds)
    logging.debug(f"Stepsize set to {integrator.getStepSize()}")
    simulation.context.reinitialize(preserveState=True)

    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)

    # Warm up the system gradually
    for i in range(nT + 1):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        logging.debug(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    return None

def update_force_constants(
    simulation,
    force_constants_dict:dict=None
    ) -> None:
    """Update the force constants for the given components."""

    for component, force_constant in force_constants_dict.items():
        try:
            simulation.context.setParameter(
                f"k_{component}",
                (
                    force_constant
                    * openmmunit.kilocalories_per_mole
                    / openmmunit.angstroms**2
                ),
            )
        except Exception as e:
            logging.error(f"Error updating force constant for {component}.\n{e}")
            pass

    return None

def run_restrained_minimization(
                    simulation,
                    components: List[str],
                    minim_scheme: Dict[str, Any]
                    ) -> None:
    """Perform restrained minimization, progressively releasing constraints."""
    
    for stage in minim_scheme:
        logging.info(f"Minimization stage {stage['name']}")
        force_constants = stage['forces']
        force_constants_dict = {k: v for k, v in zip(components, force_constants)}
        update_force_constants(simulation, force_constants_dict)
        simulation.minimizeEnergy(maxIterations=0) # default is 0 meaning until convergence
        logging.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
    return None

def run_restrained_md(
    simulation,
    system,
    integrator,
    components: List[str],
    equil_scheme: List[Dict[str, Any]],
    temp: int = 300,
    is_membrane: bool = False,
) -> None:
    """Perform restrained equilibration, adjusting force constants, timestep, and barostat as needed."""

    steps = [EquilibrationStep(**step) for step in equil_scheme]

    prev_constants = {component: None for component in components}
    npt_prev, stepsize_prev = None, None
    for step in steps:
        logging.info(
            f"Equilibration {step.name}: force_constants={step.forces} | NPT={step.npt_flag} | n_steps={step.nsteps} | stepsize={step.stepsize} | components={components}"
        )

        set_force_constants(
            simulation, components, step.forces, prev_constants
        )

        if step.npt_flag != npt_prev and step.npt_flag:
            add_barostat(system, temp, is_membrane)
            simulation.context.reinitialize(preserveState=True)

        if stepsize_prev is None or step.stepsize != stepsize_prev:
            integrator.setStepSize(step.stepsize)
            simulation.context.reinitialize(preserveState=True)
            logging.debug(f"Stepsize set to {integrator.getStepSize()}")

        simulation.step(step.nsteps)

        npt_prev = step.npt_flag
        stepsize_prev = step.stepsize

    return None

from openmm import CustomExternalForce

def add_fixed_exclusion_sphere_external(
    system,
    ion_indices,
    center,
    r_excl=0.6,
    K_flat=200.0,
    force_name="flat_excl_fixed",
    force_group=30,
):
    energy = "(k_flat/2)*max(r_excl - sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2), 0)^2"
    ext = CustomExternalForce(energy)
    ext.addGlobalParameter("k_flat", K_flat * openmmunit.kilojoules_per_mole / (openmmunit.nanometer**2))
    ext.addGlobalParameter("r_excl", r_excl * openmmunit.nanometer)
    x0, y0, z0 = center
    ext.addGlobalParameter("x0", x0 * openmmunit.nanometer)
    ext.addGlobalParameter("y0", y0 * openmmunit.nanometer)
    ext.addGlobalParameter("z0", z0 * openmmunit.nanometer)
    for idx in ion_indices:
        ext.addParticle(int(idx), [])
    # ext.setUsesPeriodicBoundaryConditions(True)
    ext.setForceGroup(force_group)
    ext.setName(force_name)
    system.addForce(ext)
    return ext


class GISTMD:
    def __init__(
        self,
        topology: str = None,
        system: str = None,
        out_dir: str = "gist_md",
        restrained_minimization: bool = False,
        protocol_fname: str = "autopath/data/equilibration.json",
        timestep: float = 0.002,
        pocket_selection: str = None,
        save_freq: int = 6250, # 12500 is 0.05ns at 4fs timestep
        MD_time: int = 100, 
        is_membrane: bool = False,
        verbose: int = 2,

    ) -> None:

        self.system = system
        self.topology = topology
        self.verbose = verbose

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.is_membrane = is_membrane

        self.timestep = timestep * openmmunit.picoseconds
        self.save_freq = save_freq
        self.MD_steps = math.ceil(MD_time / timestep * 1000.0)  # 250.000 1ns at 4fs

        self.pocket_selection = pocket_selection

        self.restrained_minimization = restrained_minimization

        self.platform = select_platform("fastest")

        # Load the equilibration protocol
        self.protocol = self.from_json(protocol_fname)

        return

    def from_json(self, fname: str = None) -> dict:
        try:
            logging.info(f"Loading equilibration protocol from {fname}.")
            with open(fname) as f:
                protocol = json.load(f)
        except FileNotFoundError:
            logging.error(f"{fname} not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"{fname} is not valid JSON.")
            raise

        # Initialize the variables
        self.components_lookup = protocol['components_lookup']  # Components lookup
        self.minimization_scheme = protocol['minimization']  # Minimization scheme
        self.equilibration_scheme = protocol['equilibration']  # Equilibration scheme
        self.equilibration_steps = sum([int(v["nsteps"]) for v in self.equilibration_scheme])

        self.warmup_scheme = protocol['warmup']
        self.temp_init = self.warmup_scheme["T_initial"]
        self.temperature = self.warmup_scheme["T_final"]
        self.temp_steps = self.warmup_scheme["T_step"]
        self.warm_up_steps = int(self.warmup_scheme["nsteps"])
        self.warm_up_timestep = self.warmup_scheme["stepsize"]

        self.total_steps = self.warm_up_steps + self.equilibration_steps

        self.simulation_time = self.warm_up_steps * self.warm_up_timestep # in picoseconds
        for stage in self.equilibration_scheme:
            stage_time = int(stage['nsteps']) * stage['stepsize']
            self.simulation_time += stage_time

        logging.info(f"Total equilibration time: {self.simulation_time:.2f} ps")

        return protocol

    def to_json(self, fname: str = None) -> None:
        self.protocol['datetime'] = str(datetime.datetime.now())
        self.protocol['time_elapsed'] = f"{self.simulation_time:.2f} min"

        try:
            logging.info("Saving equilibration protocol to JSON file.")
            with open(fname, "w") as f:
                json.dump(self.protocol, f, indent=4)
        except FileNotFoundError:
            logging.error(f"Could not save to {fname}.")
            raise

        return

    def run(self, pdb_file: str = None, run_id:str=None, checkpoint_file:str=None) -> None:

        start_time = time.monotonic()

        logging.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(seed)
        # integrator.setConstraintTolerance(0.00001)

        logging.info("Starting forces:")
        print_current_forces(self.system)

        pdb = PDBFile(pdb_file)
        initial_positions = pdb.positions
        u = mda.Universe(pdb_file)

        simulation = Simulation(self.topology, self.system, integrator, self.platform)
        # simulation.context.setPositions(initial_positions)
        
        # If a checkpoint is provided, it will assume it comes from an equilibration simulation, so it will just continue
        if checkpoint_file is not None:
            logging.info("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            logging.info("Setting initial positions..")
            simulation.context.setPositions(initial_positions)

        # simulation.context.reinitialize(preserveState=True)

        # # remove existing restraint forces and the barostat
        self.system = remove_openmm_force(self.system, "k_")
        self.system = remove_openmm_force(self.system, "MonteCarlo")
        simulation.context.reinitialize(preserveState=True)

        add_reporters(
            simulation,
            self.out_dir,
            f"gistmd_{run_id}",
            logperiod=self.save_freq, #25000 is 0.1 ns at 4 fs timestep
            total_steps=self.total_steps,
            verbose=self.verbose
        )

        # Add the required forces to the system
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            restrain_idxs = u.select_atoms(selection).indices
            restrain_names = [u.atoms[idx].name for idx in restrain_idxs]
            logging.info(f"Adding {len(restrain_idxs)} harmonic restraints to {name}..")
            logging.debug(f"The following {name} atoms will be restrained: {', '.join(restrain_names)}")

            add_harmonic_restraints(
                self.system,
                initial_positions,
                self.topology,
                restrain_idxs,
                restraint_force=100, # Some default value
                force_name=f"k_{name}",
                force_group=num+15, #Offset by 15 to avoid overlap with other forces
            )
            simulation.context.reinitialize(preserveState=True)

        ions_idxs = u.select_atoms("resname NA CL K").indices
        pocket_idxs = u.select_atoms(self.pocket_selection).indices
        sel = [i for i,a in enumerate(self.topology.atoms()) if a.index in pocket_idxs]
        logging.info(f"Found {len(ions_idxs)} ions and {len(sel)} pocket atoms.")

        positions = simulation.context.getState(getPositions=True, getVelocities=False).getPositions(asNumpy=True) / openmmunit.nanometers
        group_positions = positions[pocket_idxs]  # Get positions for the group pocket_idxs
        center_nm = np.mean(group_positions, axis=0)  # Simple mean for COG
        logging.info(f"Center of geometry for the pocket: {center_nm}")
        add_fixed_exclusion_sphere_external(self.system, ion_indices=ions_idxs, center=center_nm)

        simulation.context.reinitialize(preserveState=True)
        logging.info("Current forces before minimization A:")
        print_current_forces(self.system)

        # minim_scheme = [{ "name": "Water", "forces": [5.0, 5.0]},
        #                 { "name": "Water_sidechain", "forces": [2.5, 0]},
        #                 { "name": "Water_sidechain_backbone", "forces": [0.0, 0.0]}
        #                 ]
        run_restrained_minimization(simulation, list(self.components_lookup.keys()), self.minimization_scheme)
        
        # Save the minimized structure
        minimized_positions = simulation.context.getState(getPositions=True).getPositions()
        save_pdb(self.topology, minimized_positions, f"{self.out_dir}/{run_id}_minim-A.pdb")

        # After restrained minimization remove and re-add restraints with updated reference positions
        logging.debug("Resetting harmonic restraints after minimization to update reference positions.")
        
        # remove existing restraint forces
        self.system = remove_openmm_force(self.system, "k_")
        simulation.context.reinitialize(preserveState=True)

        # Re-add the restraints with updated positions.
        # Because the forces exist this will update them, there's no need to remove them first (I think).
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            restrain_idxs = u.select_atoms(selection).indices
            logging.info(f"Re-adding {len(restrain_idxs)} harmonic restraints to {name} after minimization.")

            add_harmonic_restraints(
                self.system,
                minimized_positions,
                self.topology,
                restrain_idxs,
                restraint_force=100,  # some default value, will be updated during equilibration
                force_name=f"k_{name}",
                force_group=num + 15,
            )

            simulation.context.reinitialize(preserveState=True)

        logging.info("Current forces after minimization A:")
        print_current_forces(self.system)

        logging.info("Warming up the system to 600K..")
        warm_up_system(simulation, integrator, 
                       Tstart=self.temp_init, 
                       Tend=600, 
                       Tstep=self.temp_steps,
                       warming_steps=self.warm_up_steps
                       )

        logging.info("Running second minimization..")
        # minim_scheme = [{ "name": "Stage 1", "forces": [0.0, 0.0]}]
        run_restrained_minimization(simulation, list(self.components_lookup.keys()), self.minimization_scheme)
        logging.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")

        minimized_positions = simulation.context.getState(getPositions=True).getPositions()
        save_pdb(self.topology, minimized_positions, f"{self.out_dir}/{run_id}_minim-B.pdb")

        logging.info("Warming up the system to 300K..")
        warm_up_system(simulation, integrator, 
                       Tstart=self.temp_init, 
                       Tend=self.temperature, 
                       Tstep=self.temp_steps,
                       warming_steps=self.warm_up_steps
                       )
        
        # remove existing restraint forces
        self.system = remove_openmm_force(self.system, "k_")
        simulation.context.reinitialize(preserveState=True)

        positions = simulation.context.getState(getPositions=True).getPositions()
        save_pdb(self.topology, positions, f"{self.out_dir}/{run_id}_warm.pdb")

        # Re-add the restraints with updated positions.
        # Because the forces exist this will update them, there's no need to remove them first (I think).
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            restrain_idxs = u.select_atoms(selection).indices
            logging.info(f"Re-adding {len(restrain_idxs)} harmonic restraints to {name} after minimization.")

            add_harmonic_restraints(
                self.system,
                positions,
                self.topology,
                restrain_idxs,
                restraint_force=100,  # some default value, will be updated during equilibration
                force_name=f"k_{name}",
                force_group=num + 15,
            )

            simulation.context.reinitialize(preserveState=True)

        logging.info("Running NPT for 2ns..")
        run_restrained_md(
            simulation,
            self.system,
            integrator,
            list(self.components_lookup.keys()),
            self.equilibration_scheme,
            self.temperature,
            self.is_membrane,
        )

        # remove existing restraint forces and the barostat
        self.system = remove_openmm_force(self.system, "k_")
        self.system = remove_openmm_force(self.system, "MonteCarlo")
        simulation.context.reinitialize(preserveState=True)

        positions = simulation.context.getState(getPositions=True).getPositions()
        save_pdb(self.topology, positions, f"{self.out_dir}/{run_id}_npt.pdb")

        # Re-add the restraints with updated positions.
        # Because the forces exist this will update them, there's no need to remove them first (I think).
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            restrain_idxs = u.select_atoms(selection).indices
            logging.info(f"Re-adding {len(restrain_idxs)} harmonic restraints to {name} after minimization.")

            add_harmonic_restraints(
                self.system,
                positions,
                self.topology,
                restrain_idxs,
                restraint_force=100.0,  # GIST restrains, suggested > 2.5 kcal/mol/A^2. They used like 100 kcal/mol/A^2 in the paper
                force_name=f"k_{name}",
                force_group=num + 15,
            )

            simulation.context.reinitialize(preserveState=True)

        logging.info("Current forces after NPT:")
        print_current_forces(self.system)

        # The first 1200000 steps are equilibration, the rest is production
        logging.info("Running production NVT..")
        # simulation.step(50000000) #100ns at 2fs
        simulation.step(self.MD_steps) #1 at 2fs

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_system(self.system, f"{self.out_dir}/system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}.pdb")

        logging.info(f"GIST equilibration finished in {(time.monotonic() - start_time)/60:.2f} min.")

        return self.system
