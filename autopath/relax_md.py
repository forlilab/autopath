# general imports
import os
import time
import logging

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# AutoPath imports
from autopath.utils import *
from autopath.equilibration import warm_up_system


class RelaxMD:
    def __init__(
        self,
        system: str = None,
        topology: str = None,
        lig_name: str = "UNK",
        out_dir: str = "relax_md",
        pocket_atoms: list[int] = None,
        use_flat_bottom_rest: bool = False,
        HMR: bool = True,
        temp: float = 300,
    ) -> None:

        self.system = system
        self.topology = topology

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.timestep = 0.004 if HMR else 0.002
        self.temperature = temp * openmmunit.kelvin

        self.ligand_ha_idx, self.lig_ha_names = get_ligand_ha(self.topology, lig_name)
        self.pocket_atoms = pocket_atoms
        self.use_flat_bottom_rest = use_flat_bottom_rest

        self.platform = select_platform("fastest")

        return None

    def run(
        self,
        checkpoint_file: str = None,
        pdb_file: str = None,
        run_id: str = None,
        md_steps: int = 25000,
    ) -> Tuple[float, float]:

        start_time = time.monotonic()

        logging.debug("Setting up the integrator..")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(int(rep_idx))

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.debug("Loading simulation checkpoint..")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)

        if self.use_flat_bottom_rest:
            add_flatbottom_COM_restraints(
                self.system, self.ligand_ha_idx, self.pocket_atoms, startdist
            )

        logging.debug("Minimizing..")
        simulation.minimizeEnergy()

        logging.debug("Warming up the system..")
        warm_up_system(
            simulation, integrator, warming_steps=md_steps, timestep=self.timestep
        )

        logging.debug("Minimizing..")
        simulation.minimizeEnergy()

        if self.use_flat_bottom_rest:
            # Remove the force before saving
            simulation.context.getSystem().removeForce(
                simulation.context.getSystem().getNumForces() - 1
            )

        # save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_simulation(simulation, f"{self.out_dir}/{run_id}_relax_checkpoint")
        save_system(self.system, f"{self.out_dir}/{run_id}_relax_system.xml")
        save_pdb(
            self.topology,
            final_positions,
            f"{self.out_dir}/{run_id}_relax.pdb",
        )

        # Get COM distance
        finaldist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)

        logging.info(
            f"{run_id} - Initial:{startdist:.3f} nm - Final:{finaldist:.3f} nm"
        )

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} relaxation in {simulation_time/60:.2f} min.")

        return startdist, finaldist
