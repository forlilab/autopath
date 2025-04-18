import time
import json
import logging

from dataclasses import dataclass
from typing import List, Dict, Any

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *
import datetime

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
    
    logging.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
    for stage in minim_scheme:
        logging.info(f"Minimization stage {stage['name']}")
        force_constants = stage['forces']
        force_constants_dict = {k: v for k, v in zip(components, force_constants)}
        update_force_constants(simulation, force_constants_dict)
        simulation.minimizeEnergy()
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


class Equilibration:
    def __init__(
        self,
        topology: str = None,
        system: str = None,
        out_dir: str = "equilibration",
        restrained_minimization: bool = True,
        protocol_fname: str = "autopath/data/equilibration.json",
        timestep: float = 0.004,
        save_freq: int = 12500, # 12500 is 0.05ns at 4fs timestep
        is_membrane: bool = False,
        verbose: int = 2,

    ) -> None:

        self.system = system
        self.topology = topology
        self.verbose = verbose

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.is_membrane = is_membrane

        # self.temperature = 100 * openmmunit.kelvin # a low value to initilize the integrator
        self.timestep = timestep * openmmunit.picoseconds
        self.save_freq = save_freq

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

        self.total_steps = self.warm_up_steps + self.equilibration_steps

        self.simulation_time = self.warm_up_steps * 0.001 # in picoseconds

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

    def run(self, pdb_file: str = None, run_id:str=None) -> None:

        start_time = time.monotonic()

        logging.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(seed)
        # integrator.setConstraintTolerance(0.00001)

        pdb = PDBFile(pdb_file)
        initial_positions = pdb.positions
        u = mda.Universe(pdb_file)

        simulation = Simulation(self.topology, self.system, integrator, self.platform)
        simulation.context.setPositions(initial_positions)

        add_reporters(
            simulation,
            self.out_dir,
            f"equilibration_{run_id}",
            logperiod=self.save_freq, #25000 is 0.1 ns at 4 fs timestep
            total_steps=self.total_steps,
            verbose=self.verbose
        )

       # Add the required forces to the system
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            logging.info(f"Adding harmonic restraints to {name}..")
            restrain_idxs = u.select_atoms(selection).indices
            restrain_names = [u.atoms[idx].name for idx in restrain_idxs]

            logging.debug(
                f"The following {name} atoms will be restrained: {', '.join(restrain_names)}")

            add_harmonic_restraints(
                self.system,
                initial_positions,
                self.topology,
                restrain_idxs,
                restraint_force=15, # Some default value
                force_name=f"k_{name}",
                force_group=num+15, #Offset by 15 to avoid overlap with other forces
            )
            simulation.context.reinitialize(preserveState=True)

        if not self.restrained_minimization:
            logging.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
            logging.info("Running standard minimization..")
            simulation.minimizeEnergy()
            logging.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
        else:
            logging.info("Running enhanced minimization..")
            run_restrained_minimization(simulation, list(self.components_lookup.keys()), self.minimization_scheme)
        
        if self.verbose > 0:
            # Save the minimized structure
            final_positions = simulation.context.getState(getPositions=True).getPositions()
            save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}_minim.pdb")



        # # Set the contraint forces to their initial values before running the equilibration
        # initial_force_constants = self.equilibration_scheme[0]['forces']
        # force_constants_dict = {k: v for k, v in zip(list(self.components_lookup.keys()), initial_force_constants)}
        # update_force_constants(simulation, force_constants_dict)

        logging.info("Warming up the system..")
        warm_up_system(simulation, integrator, 
                       Tstart=self.temp_init, 
                       Tend=self.temperature, 
                       Tstep=self.temp_steps,
                       warming_steps=self.warm_up_steps
                       )

        logging.info("Running restrained equilibration protocol..")
        run_restrained_md(
            simulation,
            self.system,
            integrator,
            list(self.components_lookup.keys()),
            self.equilibration_scheme,
            self.temperature,
            self.is_membrane,
        )

        # TODO forces should be removed by name. OpenMM behavior is weird with that
        # Remove protein and ligand force restraints
        simulation.context.getSystem().removeForce(
            simulation.context.getSystem().getNumForces() - 3
        )
        simulation.context.getSystem().removeForce(
            simulation.context.getSystem().getNumForces() - 2
        )
        # print_current_forces(self.system)

        final_positions = simulation.context.getState(getPositions=True).getPositions()

        save_system(self.system, f"{self.out_dir}/system_equil_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/checkpoint_equil_{run_id}")
        save_pdb(
            self.topology, final_positions, f"{self.out_dir}/{run_id}_equilibrated.pdb"
        )

        self.simulation_time = (time.monotonic() - start_time) / 60 
        logging.info(
            f"Restrained equilibration completed in {self.simulation_time:.2f} min."
        )

        self.to_json(f"{self.out_dir}/equilibration_protocol.json")

        return self.system
