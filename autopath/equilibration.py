import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

from autopath.utils import *
from autopath.customForces import *
import datetime

import logging
logger = logging.getLogger("autopath")

@dataclass
class EquilibrationStep:
    """Single stage in the equilibration protocol as parsed from JSON.

    Attributes
    ----------
    name : str
        Human-readable stage label (e.g. ``"NVT_heavy"``).
    forces : list of float
        Force constants in kcal/mol/Å² for each restrained component, in the
        same order as ``components_lookup``.
    npt_flag : bool
        If True, a barostat is added for this stage (NPT); otherwise NVT.
    nsteps : int
        Number of integration steps to run for this stage.
    stepsize : float
        Integration timestep in picoseconds.
    """
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
    """Update simulation context parameters for each restrained component.

    Only parameters whose value has changed (or that have never been set) are
    pushed to the context, avoiding redundant reinitializations.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The active simulation whose context is updated.
    components : list of str
        Restrained component names (e.g. ``["backbone", "sidechain"]``).
    forces : list of float
        New force constants in kcal/mol/Å², parallel to *components*.
    prev_constants : dict
        Mutable mapping of component name → last-applied force constant,
        used to skip redundant context updates. Updated in-place.
    """
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

    integrator.setStepSize(timestep)
    logger.debug(f"Timestep set to {integrator.getStepSize()}")
    simulation.context.reinitialize(preserveState=True)

    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)

    # Warm up the system gradually
    for i in range(nT + 1):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        logger.debug(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    return None

def update_force_constants(
    simulation,
    force_constants_dict: dict = None
) -> None:
    """Push a mapping of component → force-constant values to the simulation context.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The active simulation whose context parameters are updated.
    force_constants_dict : dict
        Mapping of component name (str) → force constant (float) in
        kcal/mol/Å². Each entry is set as ``k_<name>`` in the context.
        Errors for individual components are logged and skipped.
    """

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
            logger.error(f"Error updating force constant for {component}.\n{e}")
            pass

    return None

def run_restrained_minimization(
    simulation,
    components: List[str],
    minim_scheme: Dict[str, Any]
) -> None:
    """Perform energy minimization in multiple stages with progressively relaxed restraints.

    Each stage sets new force constants and runs minimization to convergence
    (``maxIterations=0``), allowing heavy atoms to be released before lighter
    atoms in subsequent stages to avoid clashes.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The simulation to minimize.
    components : list of str
        Restrained component names, parallel to ``forces`` in each stage dict.
    minim_scheme : list of dict
        Ordered list of minimization stage dicts, each with keys ``"name"``
        (str) and ``"forces"`` (list of float in kcal/mol/Å²).
    """
    
    for stage in minim_scheme:
        logger.info(f"Minimization stage {stage['name']}")
        force_constants = stage['forces']
        force_constants_dict = {k: v for k, v in zip(components, force_constants)}
        update_force_constants(simulation, force_constants_dict)
        simulation.minimizeEnergy(maxIterations=0) # default is 0 meaning until convergence
        logger.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
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
    """Run a multi-stage restrained MD equilibration protocol.

    Iterates through *equil_scheme* stages, updating force constants, the
    integration timestep, and the barostat ensemble (NVT ↔ NPT) as required.
    A barostat is added on the first NPT stage encountered; ensemble switches
    from NVT to NPT are detected by comparing ``npt_flag`` to the previous stage.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The active simulation.
    system : openmm.System
        The OpenMM system (modified in-place when a barostat is added).
    integrator : openmm.Integrator
        The integrator whose step size may be updated between stages.
    components : list of str
        Restrained component names, parallel to the ``forces`` lists in
        *equil_scheme*.
    equil_scheme : list of dict
        Ordered equilibration stages, each deserializable as an
        :class:`EquilibrationStep`.
    temp : int
        Target temperature in Kelvin (passed to the barostat if added).
    is_membrane : bool
        If True, use a membrane-appropriate semi-isotropic barostat.
    """

    steps = [EquilibrationStep(**step) for step in equil_scheme]

    prev_constants = {component: None for component in components}
    npt_prev, stepsize_prev = None, None
    for step in steps:
        force_comp_dict = {k: v for k, v in zip(components, step.forces)}
        logger.info(
            f"Equilibration {step.name}: force_constants={force_comp_dict} | NPT={step.npt_flag} | n_steps={step.nsteps} | timestep={step.stepsize}"
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
            logger.debug(f"Stepsize set to {integrator.getStepSize()}")

        simulation.step(step.nsteps)

        npt_prev = step.npt_flag
        stepsize_prev = step.stepsize

    return None


class Equilibration:
    """Orchestrate the multi-stage equilibration/relaxation protocol before production MD.

    Loads a JSON protocol describing minimization and equilibration stages,
    applies progressively weakened positional restraints, warms the system from
    a low temperature to the target, and saves the final equilibrated structure
    and system XML.

    Parameters
    ----------
    topology : openmm.app.Topology
        OpenMM topology of the system to equilibrate.
    system : openmm.System
        OpenMM system (force field parameters, bonds, etc.).
    out_dir : str
        Directory where equilibration outputs are written.
    restrained_minimization : bool
        If True, run a staged restrained minimization (recommended); otherwise
        run a single unconstrained minimization.
    restrained_minimization_only : bool
        If True, stop after minimization and return the minimized system without
        running the warm-up or equilibration MD.
    protocol_fname : str
        Path to the JSON file specifying the minimization and equilibration stages.
    save_freq : int
        Reporter output interval in steps (default 6250 ≈ 0.025 ns at 4 fs).
    is_membrane : bool
        If True, use a semi-isotropic membrane barostat for NPT stages.
    platform : str
        OpenMM platform name (``"CUDA"``, ``"OpenCL"``, ``"CPU"``, or
        ``"fastest"`` to auto-select).
    verbose : int
        Reporter verbosity: 0 = no CSV, 1 = CSV without energies,
        2 = full thermodynamic CSV output.
    """

    def __init__(
        self,
        topology: str = None,
        system: str = None,
        out_dir: str = "equilibration",
        restrained_minimization: bool = True,
        restrained_minimization_only: bool = False,
        protocol_fname: str = "autopath/data/equilibration.json",
        save_freq: int = 6250,  # 12500 is 0.05 ns at 4 fs timestep
        is_membrane: bool = False,
        platform: str = "fastest",
        verbose: int = 2,
    ) -> None:

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.system = system
        self.topology = topology

        self.is_membrane = is_membrane

        # Default timestep; overridden by the first equilibration stage in the protocol
        self.timestep = 0.004 * openmmunit.picoseconds
        self.save_freq = save_freq

        self.restrained_minimization = restrained_minimization
        self.restrained_minimization_only = restrained_minimization_only

        self.platform = select_platform(platform)
        self.verbose = verbose

        # Load the equilibration protocol
        self.protocol = self.from_json(protocol_fname)

        return

    def from_json(self, fname: str = None) -> dict:
        """Load and parse the equilibration protocol from a JSON file.

        Populates all protocol-derived attributes on this instance
        (``components_lookup``, ``minimization_scheme``, ``equilibration_scheme``,
        ``warmup_scheme``, temperatures, step counts, and total simulation time).

        Parameters
        ----------
        fname : str
            Path to the JSON protocol file.

        Returns
        -------
        dict
            The full parsed protocol dictionary.

        Raises
        ------
        FileNotFoundError
            If *fname* does not exist.
        json.JSONDecodeError
            If *fname* is not valid JSON.
        """
        try:
            logger.info(f"Loading equilibration protocol from {fname}.")
            with open(fname) as f:
                protocol = json.load(f)
        except FileNotFoundError:
            logger.error(f"{fname} not found.")
            raise
        except json.JSONDecodeError:
            logger.error(f"{fname} is not valid JSON.")
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

        logger.info(f"Total equilibration time: {self.simulation_time:.2f} ps")

        return protocol

    def to_json(self, fname: str = None) -> None:
        """Write the current protocol (with run metadata) to a JSON file.

        Adds ``datetime`` and ``time_elapsed`` keys to the protocol dict before
        serialization so the output captures when and how long the run took.

        Parameters
        ----------
        fname : str
            Destination JSON file path.

        Raises
        ------
        FileNotFoundError
            If the directory containing *fname* does not exist.
        """
        self.protocol['datetime'] = str(datetime.datetime.now())
        self.protocol['time_elapsed'] = f"{self.simulation_time:.2f} min"

        try:
            logger.info("Saving equilibration protocol to JSON file.")
            with open(fname, "w") as f:
                json.dump(self.protocol, f, indent=4)
        except FileNotFoundError:
            logger.error(f"Could not save to {fname}.")
            raise

        return

    def run(self, pdb_file: str = None, run_id: str = None) -> openmm.System:
        """Execute the full equilibration pipeline and return the final system.

        Steps performed:

        1. Build the simulation with a Langevin integrator.
        2. Add harmonic positional restraints for each component in
           ``components_lookup``.
        3. Run restrained energy minimization (staged or single-pass).
        4. Remove and re-add restraints anchored to the minimized geometry.
        5. Warm up from ``T_initial`` to ``T_final`` in NVT.
        6. Run the staged restrained equilibration (NVT → NPT).
        7. Remove all restraint forces and save outputs.

        Parameters
        ----------
        pdb_file : str
            Path to the starting structure PDB file.
        run_id : str
            String identifier appended to output filenames (e.g. walker index).

        Returns
        -------
        openmm.System
            The equilibrated system with all restraint forces removed.
        """
        start_time = time.monotonic()

        logger.debug("Setting up the integrator..")
        # The native OpenMM LangevinMiddleIntegrator is faster but does not
        # expose the splitting string. By default it uses "V V R O R".
        # If using openmmtools integrators downstream, set the splitting
        # explicitly to match. See https://github.com/openmm/openmm/issues/2532
        integrator = LangevinMiddleIntegrator(self.temperature, 
                                        1 / openmmunit.picoseconds, 
                                        self.timestep)
        
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
            restrain_idxs = u.select_atoms(selection).indices
            if len(restrain_idxs) > 0:
                restrain_names = [u.atoms[idx].name for idx in restrain_idxs]
                logger.info(f"Adding {len(restrain_idxs)} harmonic restraints to {name}..")
                logger.debug(f"The following {name} atoms will be restrained: {', '.join(restrain_names)}")
            else:
                logger.warning(f"Skipping harmonic restraints for {name}: No atoms found for selection '{selection}'")

            add_harmonic_restraints(
                self.system,
                initial_positions,
                self.topology,
                restrain_idxs,
                restraint_force=15,  # default; overridden by minimization/equilibration stages
                force_name=f"k_{name}",
                force_group=num + 15,  # offset of 15 keeps each component in a distinct group; see customForces._FG_FUNNEL
            )
        
        simulation.context.reinitialize(preserveState=True)
        
        logger.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
        if not self.restrained_minimization:
            logger.info("Running standard minimization..")
            simulation.minimizeEnergy()
            logger.info(f"Current system's energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
        else:
            logger.info("Running enhanced minimization..")
            run_restrained_minimization(simulation, list(self.components_lookup.keys()), self.minimization_scheme)
        
        minimized_positions = simulation.context.getState(getPositions=True).getPositions()

        if self.verbose > 0: # Save the minimized structure
            self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
            save_pdb(self.topology, minimized_positions, f"{self.out_dir}/{run_id}_minim.pdb")

        # After restrained minimization remove and re-add restraints with updated reference positions
        logger.debug("Resetting harmonic restraints after minimization to update reference positions.")

        # remove existing restraint forces
        self.system = remove_openmm_force(self.system, "k_")

        if self.restrained_minimization_only:
            return self.system 

        # Re-add restraints anchored to minimized positions so the warm-up and
        # equilibration stages pull toward the post-minimization geometry rather
        # than the original input coordinates. The old forces were removed above.
        for num, (name, selection) in enumerate(self.components_lookup.items()):
            restrain_idxs = u.select_atoms(selection).indices
            logger.info(f"Re-adding {len(restrain_idxs)} harmonic restraints to {name} after minimization.")

            add_harmonic_restraints(
                self.system,
                minimized_positions,
                self.topology,
                restrain_idxs,
                restraint_force=15,  # default; overridden by equilibration stages
                force_name=f"k_{name}",
                force_group=num + 15,  # offset of 15 keeps each component in a distinct group; see customForces._FG_FUNNEL
            )

        simulation.context.reinitialize(preserveState=True)
        # print_current_forces(self.system)

        logger.info(f"Warming up the system from {self.temp_init} K to {self.temperature} K..")
        warm_up_system(simulation, integrator, 
                       Tstart=self.temp_init, 
                       Tend=self.temperature, 
                       Tstep=self.temp_steps,
                       warming_steps=self.warm_up_steps
                       )
        
        logger.info("Running restrained equilibration protocol..")
        run_restrained_md(
            simulation,
            self.system,
            integrator,
            list(self.components_lookup.keys()),
            self.equilibration_scheme,
            self.temperature,
            self.is_membrane,
        )

        # remove the restraint forces after equilibration
        self.system = remove_openmm_force(self.system, "k_")
        simulation.context.reinitialize(preserveState=True)

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_system(self.system, f"{self.out_dir}/system_equil_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/checkpoint_equil_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/{run_id}_equilibrated.pdb")

        self.simulation_time = (time.monotonic() - start_time) / 60 
        logger.info(f"Autopath equilibration completed in {self.simulation_time:.2f} min.")
        self.to_json(f"{self.out_dir}/equilibration_protocol.json")

        return self.system
