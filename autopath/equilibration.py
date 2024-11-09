import time
import json
import logging

from dataclasses import dataclass
from typing import List, Dict, Any

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *


@dataclass
class EquilibrationStep:
    name: str
    force_constants: List[float]
    npt_flag: bool
    nsteps: int
    stepsize: float


def set_force_constants(
    simulation,
    components: List[str],
    force_constants: List[float],
    prev_constants: Dict[str, float],
) -> None:
    """Set the force constants for the given components."""
    for component, force_constant in zip(components, force_constants):
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
    Tstart: int = 5,
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

    return


def equilibrate_restrained_system(
    simulation,
    system,
    integrator,
    equil_scheme: Dict[str, Any],
    temp: int = 300,
    is_membrane: bool = False,
) -> None:
    """Perform restrained equilibration, adjusting force constants, timestep, and barostat as needed."""

    components = equil_scheme["components"]
    steps = [EquilibrationStep(**step) for step in equil_scheme["steps"]]

    prev_constants = {component: None for component in components}
    npt_prev, stepsize_prev = None, None

    for step in steps:
        logging.info(
            f"Equilibration {step.name}: force_constants={step.force_constants} | NPT={step.npt_flag} | n_steps={step.nsteps} | stepsize={step.stepsize} | components={components}"
        )

        set_force_constants(
            simulation, components, step.force_constants, prev_constants
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

    return


class Equilibration:
    def __init__(
        self,
        topology: str = None,
        system: str = None,
        out_dir: str = "equilibration",
        lig_name: str = "UNK",
        equilibration_fname: str = "autopath/data/equilibration.json",
        warm_up_steps: int = 100000,
        temperature: float = 300,
        timestep: float = 0.004,
        is_membrane: bool = False,
        lipid_type: str = "POPC",
        verbose: int = 2,

    ) -> None:

        self.system = system
        self.topology = topology
        self.lig_name = lig_name
        self.verbose = verbose

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        # TODO lipid type and residue name is not the same i.e. POPC/POP. We need a mapping dict

        self.temperature = temperature * openmmunit.kelvin
        self.timestep = timestep * openmmunit.picoseconds
        self.warm_up_steps = warm_up_steps

        self.platform = select_platform("fastest")

        self.equilibration_scheme = self.from_json(equilibration_fname)

        equilibration_steps = sum(
            [v["nsteps"] for v in self.equilibration_scheme["steps"]]
        )
        self.total_steps = warm_up_steps + equilibration_steps

        return

    def from_json(self, equilibration_fname: str = None) -> dict:
        try:
            logging.info("Loading equilibration protocol from JSON file.")
            with open(equilibration_fname) as f:
                equilibration_scheme = json.load(f)
        except FileNotFoundError:
            logging.error(f"{equilibration_fname} not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"{equilibration_fname} is not valid JSON.")
            raise

        return equilibration_scheme

    def run(self, pdb_file: str = None, run_id:str=None) -> None:

        start_time = time.monotonic()

        logging.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(seed)
        integrator.setConstraintTolerance(0.00001)

        pdb = PDBFile(pdb_file)
        initial_positions = pdb.positions

        simulation = Simulation(self.topology, self.system, integrator, self.platform)
        simulation.context.setPositions(initial_positions)

        add_reporters(
            simulation,
            self.out_dir,
            f"equil_{run_id}",
            logperiod=1250,
            total_steps=self.total_steps,
            verbose=self.verbose
        )

        logging.info("Adding harmonic restraints to the protein..")
        prot_ha_idx, prot_ha_names = get_protein_ha(self.topology, self.lig_name)
        logging.debug(
            f"The following protein heavy atoms will be restrained: {', '.join(prot_ha_names)}"
        )
        add_harmonic_restraints(
            self.system,
            initial_positions,
            self.topology,
            prot_ha_idx,
            restraint_force=5,
            force_name="k_protein",
            force_group=12,
        )

        logging.info("Adding harmonic restraints to the ligand..")
        lig_ha_idx, lig_ha_names = get_ligand_ha(self.topology, self.lig_name)
        logging.info(
            f"The following ligand heavy atoms will be restrained: {', '.join(lig_ha_names)}"
        )
        add_harmonic_restraints(
            self.system,
            initial_positions,
            self.topology,
            lig_ha_idx,
            restraint_force=5,
            force_name="k_ligand",
            force_group=13,
        )

        if self.is_membrane:
            logging.info("Adding harmonic restraints to the membrane..")
            lipid_ha_idx, lipid_ha_names = get_ligand_ha(self.topology, "POP")
            add_harmonic_restraints(
                self.system,
                initial_positions,
                self.topology,
                lipid_ha_idx,
                restraint_force=10,
                force_name="k_membrane",
                force_group=14,
            )

        logging.info("Minimizing..")
        simulation.minimizeEnergy()

        logging.info("Warming up the system..")
        warm_up_system(simulation, integrator, warming_steps=self.warm_up_steps)

        logging.info("Running restrained equilibration protocol..")
        equilibrate_restrained_system(
            simulation,
            self.system,
            integrator,
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

        simulation_time = time.monotonic() - start_time
        logging.info(
            f"Restrained equilibration completed in {simulation_time/60:.2f} min."
        )

        return self.system
