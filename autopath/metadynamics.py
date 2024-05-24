import os
import time
import numpy as np
import logging

import cvpack

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit
from openmm.app.amberprmtopfile import AmberPrmtopFile

from autopath.utils import *
from autopath.analysis import plot_bias, plot_colvar, plot_FE

class MetadynamicsMD:

    def __init__(
        self,
        sys_name: str = None,
        prmtop_file:str = None,
        lig_name: str = "UNK",
        pocket_atoms: list[int] = None,
        HMR: bool = True,
        temp: float = 300,
        verbose: bool = True,
        ) -> None:


        logging.basicConfig(
            level="INFO",
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"metadynamics.log", mode="w"),
                logging.StreamHandler(),
            ],
        )

        if HMR:
            self.timestep = 0.004
        else:
            self.timestep = 0.002

        self.warming_steps = 25000 # for temp annealing
        self.temperature = temp * openmmunit.kelvin

        # These are for debugging purposes if one wants to check the CVs over the time of the simulation
        self.verbose = verbose
        self.record_CV = 2500  # record the CVs every 10 ps this would give 1000 points for 10ns
        self.store_CV = 25000  # log the stored COLVAR every 100ps
        
        self.sys_name = sys_name
        self.write_dir = f"{self.sys_name}/metadynamics"
        os.makedirs(self.write_dir, exist_ok=True)

        prmtop = AmberPrmtopFile(prmtop_file)
        self.topology = prmtop.topology

        self.ligand_ha_idx, self.lig_ha_names = get_ligand_ha(self.topology, lig_name)
        self.pocket_atoms = pocket_atoms
        
        self.platform = select_platform("fastest")

    def run(self,
            pdb_file: str = None,
            system_file:str = None,
            checkpoint_file: str = None,
            run_id: str = None,
            mMD_time: int = 10,
            bias_factor: float = 10,
            hill_height: float = 0.3,
            hill_width: float = 0.01, # also known as sigma
            grid_dimensions: tuple = (0.0, 1.0),
            bias_frequency: int = 2,
            saveFrequency: int = 50,
        ):

        start_time = time.monotonic()

        # Metadynamics time in ns
        mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
        total_steps = self.warming_steps + mMD_steps
        bias_frequency = 250 * bias_frequency # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps

        hill_height = hill_height * openmmunit.kilocalories_per_mole

        grid_width = hill_width / 5
        grid_min, grid_max = grid_dimensions  # nm
        # 'grid' here refers to the number of grid points (2500 points)
        grid = int(abs(grid_min - grid_max) / grid_width)
        logging.info(f'COM boundaries are min={grid_min:.3f} nM - max={grid_max:.3f} nM')
        logging.info(f'Sigma is {hill_width} nm and there are {grid} grid points ')

        groups = [self.pocket_atoms] + [self.ligand_ha_idx]

        # logging.info('Setting up the integrator..')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(int(rep_idx))

        system = load_system(system_file)

        logging.info(f'Creating the simulation for {run_id}..')
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.info('Loading simulation checkpoint..')
            simulation.loadCheckpoint(checkpoint_file)

        # fb_eq = f'sqrt((distance(g1,g2))^2)-{initial_COM_dist}' # Offset for initial COM dist
        fb_eq = f"sqrt(distance(g1,g2)^2)"
        COM = cvpack.CentroidFunction(fb_eq, openmmunit.nanometers, groups, weighByMass=False, pbc=True)

        com_cv = BiasVariable(
            COM,
            minValue=grid_min,
            maxValue=grid_max,
            biasWidth=hill_width,
            periodic=False,
            gridWidth=grid,
        )

        meta = Metadynamics(
            system,
            [com_cv],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=bias_frequency,
            saveFrequency=saveFrequency,
            biasDir=self.write_dir,
        )

        simulation.context.reinitialize(preserveState=True)

        logging.info(f"Setting up reporters for {run_id}..")
        add_reporters(simulation,self.write_dir, f'metadynamics_{run_id}', total_steps, bias_frequency)

        if not self.verbose:
            # # Advance all steps at once do not record CVs
            meta.step(simulation, mMD_steps)
        else:
            # Record CVs along the way, might be usefull for debugging
            colvar_array = np.array([meta.getCollectiveVariables(simulation)])
            for i in range(0, int(mMD_steps), self.record_CV):
                if i % self.store_CV == 0:
                    np.save(os.path.join(self.write_dir, f"COLVAR_{run_id}.npy"), colvar_array)

                meta.step(simulation, self.record_CV)
                current_cvs = meta.getCollectiveVariables(simulation)
                colvar_array = np.append(colvar_array, [current_cvs], axis=0)

        np.save(os.path.join(self.write_dir, f"COLVAR_{run_id}.npy"), colvar_array)
        np.save(os.path.join(self.write_dir, f"FE_{run_id}.npy"), meta.getFreeEnergy())

        # Create plots for all current runs
        plot_colvar(self.write_dir, 'COM_dist')
        plot_bias(self.write_dir, grid_min, grid_max, grid)
        plot_FE(self.write_dir, grid_min, grid_max, grid)

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_system(system, f'{self.write_dir}/system_mMD_{run_id}.xml')
        save_simulation(simulation, f'{self.write_dir}/mMD_checkpoint_{run_id}')
        save_pdb(self.topology, final_positions, f'{self.write_dir}/mMD_{run_id}.pdb')

        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished {run_id} metadynamics in {simulation_time/60:.2f} min.')