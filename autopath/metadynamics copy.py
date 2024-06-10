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
        NPT: bool = True,
        verbose: bool = True,
        ) -> None:

        if HMR:
            self.timestep = 0.004
        else:
            self.timestep = 0.002

        self.warming_steps = 25000 # for temp annealing
        self.temperature = temp * openmmunit.kelvin
        self.NPT = NPT

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

        return

    def run(self,
            pdb_file: str = None,
            system_file:str = None,
            checkpoint_file: str = None,
            run_id: str = None,
            mMD_time: int = 10,
            bias_factor: float = 10,
            hill_height: float = 0.3,
            bias_frequency: int = 2,
            saveFrequency: int = 50,
            mMD_CV: str = 'cog',
            hill_width: float = 0.01, # also known as sigma
            grid_dimensions: tuple = (0.0, 1.0),

        ) -> None:

        start_time = time.monotonic()

        assert mMD_CV in ['cog', 'rmsd', 'nc'], f"The selectec colective variable {mMD_CV} is not implemented"

        mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
        total_steps = self.warming_steps + mMD_steps
        bias_frequency = 250 * bias_frequency # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps

        hill_height = hill_height * openmmunit.kilocalories_per_mole

        grid_width = hill_width / 5
        grid_min, grid_max = grid_dimensions  # 'grid' here refers to the number of grid points (2500 points)
        grid = int(abs(grid_min - grid_max) / grid_width)

        logging.info(f'Running metadynamics with Colective Variable {mMD_CV}')
        logging.info(f'Grid boundaries are min={grid_min:.3f} - max={grid_max:.3f}')
        logging.info(f'Sigma is {hill_width} nm and there are {grid} grid points ')

        logging.debug(f'Loading a simulation file')
        system = load_system(system_file)

        if mMD_CV == 'cog':
            
            groups = [self.pocket_atoms] + [self.ligand_ha_idx]

            cv = cvpack.CentroidFunction(f"sqrt(distance(g1,g2)^2)", 
                                        openmmunit.nanometers, 
                                        groups, 
                                        weighByMass=False, 
                                        pbc=True)

        elif mMD_CV == 'rmsd':

            input_positions = simulation.context.getState(getPositions=True).getPositions()
            num_atoms = self.topology.getNumAtoms()
            cv = cvpack.RMSD(input_positions, 
                               self.ligand_ha_idx, 
                               num_atoms)
            
        elif mMD_CV == 'nc':

            forces = {f.getName(): f for f in system.getForces()}

            cv = cvpack.NumberOfContacts(
                self.pocket_atoms,
                self.ligand_ha_idx,
                forces["NonbondedForce"],
                stepFunction="1/(1+x^6)",
                thresholdDistance=0.35,
                cutoffFactor=2.0,
                switchFactor=1.5
            )

        bias_variable = BiasVariable(cv, 
                        minValue=grid_min, 
                        maxValue=grid_max, 
                        biasWidth=hill_width,
                        periodic=False, 
                        gridWidth=grid)
            
        logging.debug('Setting up the integrator')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(int(rep_idx))
        
        if self.NPT:
            logging.debug(f'Adding a Montecarlo Barostat to the system')
            system.addForce(MonteCarloBarostat(1 * openmmunit.atmosphere, self.temperature))

        logging.debug(f'Creating the simulation for {run_id}')
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if pdb_file is not None:
            logging.debug('Setting positions from PDB filet')
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        if checkpoint_file is not None:
            logging.debug('Loading simulation checkpoint')
            simulation.loadCheckpoint(checkpoint_file)

        # Set up the metadynamics object
        meta = Metadynamics(
            system,
            [bias_variable],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=bias_frequency,
            saveFrequency=saveFrequency,
            biasDir=self.write_dir,
        )

        simulation.context.reinitialize(preserveState=True)

        logging.debug(f"Setting up reporters for {run_id}..")
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
        plot_colvar(self.write_dir, mMD_CV)
        plot_bias(self.write_dir, grid_min, grid_max, grid)
        plot_FE(self.write_dir, grid_min, grid_max, grid)

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_system(system, f'{self.write_dir}/system_mMD_{run_id}.xml')
        save_simulation(simulation, f'{self.write_dir}/mMD_checkpoint_{run_id}')
        save_pdb(self.topology, final_positions, f'{self.write_dir}/mMD_{run_id}.pdb')

        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished {run_id} metadynamics in {simulation_time/60:.2f} min.')

        return
    

    def run2D(self,
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
            ) -> None:

            start_time = time.monotonic()

            logging.debug(f'Loading a simulation file')
            system = load_system(system_file)

            # Metadynamics time in ns
            mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
            total_steps = self.warming_steps + mMD_steps
            bias_frequency = 250 * bias_frequency # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
            saveFrequency = 250 * saveFrequency  # write bias every 50ps
            hill_height = hill_height * openmmunit.kilocalories_per_mole

            # add sensor loop beta sheet CV
            #residues = list(it.islice(modeller.topology.residues(), 346, 360))
            residues = [r for r in self.topology.residues() if int(r.id) >= 283 and int(r.id) <= 297 and r.name != "HOH"]
            print("Sensor loop residues:", *[(r.name, r.id) for r in residues])
            sheet_content = cvpack.SheetRMSDContent(residues, 
                                                    system.getNumParticles(), 
                                                    normalize=True)

            hill_width = 0.1
            grid_width = hill_width / 5
            grid_min, grid_max = 0, 1
            grid = int(abs(grid_min - grid_max) / grid_width)

            sc_cv = BiasVariable(
                sheet_content,
                minValue=grid_min,
                maxValue=grid_max,
                biasWidth=hill_width,
                periodic=False,
                gridWidth=grid,
            )

            hill_width = 0.1
            grid_width = hill_width / 5
            grid_min, grid_max = 0, 1
            grid = int(abs(grid_min - grid_max) / grid_width)

            # add TBD-Lon interaction cv
            lon_group = [r for r in self.topology.residues() if int(r.id) >= 27 and int(r.id) <= 29 and r.name != "HOH"] + [r for r in self.topology.residues() if int(r.id) >= 34 and int(r.id) <= 39 and r.name != "HOH"]
            tbd_group = [r for r in self.topology.residues() if int(r.id) >= 285 and int(r.id) <= 295 and r.name != "HOH"]

            print("Lon residues:", *[(r.name, r.id) for r in lon_group])
            print("TBD residues:", *[(r.name, r.id) for r in tbd_group])

            residue_coordination = cvpack.ResidueCoordination(
                lon_group,
                tbd_group,
                stepFunction='1/(1+x^6)',
                thresholdDistance=0.6,
                normalize=True,
            )

            rc_cv = BiasVariable(
                residue_coordination,
                minValue=grid_min,
                maxValue=grid_max,
                biasWidth=hill_width,
                periodic=False,
                gridWidth=grid,
            )
    

            logging.debug('Setting up the integrator')
            integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
            # integrator.setRandomNumberSeed(int(rep_idx))
            
            if self.NPT:
                logging.debug(f'Adding a Montecarlo Barostat to the system')
                system.addForce(MonteCarloBarostat(1 * openmmunit.atmosphere, self.temperature))

            logging.info(f'Creating the simulation for {run_id}')
            simulation = Simulation(self.topology, system, integrator, self.platform)

            if pdb_file is not None:
                logging.debug('Setting positions from PDB filet')
                initial_positions = PDBFile(pdb_file).positions
                simulation.context.setPositions(initial_positions)

            if checkpoint_file is not None:
                logging.debug('Loading simulation checkpoint')
                simulation.loadCheckpoint(checkpoint_file)

            meta = Metadynamics(
                system,
                [sc_cv, rc_cv],
                self.temperature,
                bias_factor,
                hill_height,
                frequency=bias_frequency,
                saveFrequency=saveFrequency,
                biasDir=self.write_dir,
            )

            simulation.context.reinitialize(preserveState=True)

            logging.debug(f"Setting up reporters for {run_id}..")
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
            # plot_colvar(self.write_dir, 'COM_dist')
            # plot_bias(self.write_dir, grid_min, grid_max, grid)
            # plot_FE(self.write_dir, grid_min, grid_max, grid)

            final_positions = simulation.context.getState(getPositions=True).getPositions()
            save_system(system, f'{self.write_dir}/system_mMD_{run_id}.xml')
            save_simulation(simulation, f'{self.write_dir}/mMD_checkpoint_{run_id}')
            save_pdb(self.topology, final_positions, f'{self.write_dir}/mMD_{run_id}.pdb')

            simulation_time = time.monotonic() - start_time
            logging.info(f'Finished {run_id} metadynamics in {simulation_time/60:.2f} min.')

            return




    
    # def run2D(self,
    #         pdb_file: str = None,
    #         system_file:str = None,
    #         checkpoint_file: str = None,
    #         run_id: str = None,
    #         mMD_time: int = 10,
    #         bias_factor: float = 10,
    #         hill_height: float = 0.3,
    #         hill_width: float = 0.01, # also known as sigma
    #         grid_dimensions: tuple = (0.0, 1.0),
    #         bias_frequency: int = 2,
    #         saveFrequency: int = 50,
    #     ) -> None:

    #     start_time = time.monotonic()

    #     # Metadynamics time in ns
    #     mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
    #     total_steps = self.warming_steps + mMD_steps
    #     bias_frequency = 250 * bias_frequency # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
    #     saveFrequency = 250 * saveFrequency  # write bias every 50ps

    #     hill_height = hill_height * openmmunit.kilocalories_per_mole

    #     grid_width = hill_width / 5
    #     grid_min, grid_max = grid_dimensions  # nm
    #     # 'grid' here refers to the number of grid points (2500 points)
    #     grid = int(abs(grid_min - grid_max) / grid_width)
    #     logging.info(f'COM boundaries are min={grid_min:.3f} nM - max={grid_max:.3f} nM')
    #     logging.info(f'Sigma is {hill_width} nm and there are {grid} grid points ')

    #     groups = [self.pocket_atoms] + [self.ligand_ha_idx]

    #     logging.debug('Setting up the integrator')
    #     integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
    #     # integrator.setRandomNumberSeed(int(rep_idx))

    #     logging.debug(f'Loading a simulation file')
    #     system = load_system(system_file)
        
    #     if self.NPT:
    #         logging.debug(f'Adding a Montecarlo Barostat to the system')
    #         system.addForce(MonteCarloBarostat(1 * openmmunit.atmosphere, self.temperature))

    #     logging.info(f'Creating the simulation for {run_id}')
    #     simulation = Simulation(self.topology, system, integrator, self.platform)

    #     if pdb_file is not None:
    #         logging.debug('Setting positions from PDB filet')
    #         initial_positions = PDBFile(pdb_file).positions
    #         simulation.context.setPositions(initial_positions)

    #     if checkpoint_file is not None:
    #         logging.debug('Loading simulation checkpoint')
    #         simulation.loadCheckpoint(checkpoint_file)

    #     fb_eq = f"sqrt(distance(g1,g2)^2)"
    #     COM = cvpack.CentroidFunction(fb_eq, 
    #                                   openmmunit.nanometers, 
    #                                   groups, 
    #                                   weighByMass=False, 
    #                                   pbc=True)

    #     com_cv = BiasVariable(
    #         COM,
    #         minValue=grid_min,
    #         maxValue=grid_max,
    #         biasWidth=hill_width,
    #         periodic=False,
    #         gridWidth=grid,
    #     )

    #     input_positions = simulation.context.getState(getPositions=True).getPositions()
    #     num_atoms = self.topology.getNumAtoms()
    #     rmsd = cvpack.RMSD(input_positions, self.ligand_ha_idx, num_atoms)
        
    #     hill_width = 0.01
    #     grid_width = hill_width / 5
    #     grid_min, grid_max = 0.0, 0.3 # # nm
    #     grid = int(abs(grid_min - grid_max) / grid_width) # 2500 points
    #     rmsd_cv = BiasVariable(rmsd, 
    #                            minValue=grid_min, 
    #                            maxValue=grid_max, 
    #                            biasWidth=hill_width,
    #                            periodic=False, 
    #                            gridWidth=grid)

    #     meta = Metadynamics(
    #         system,
    #         [rmsd_cv, com_cv],
    #         self.temperature,
    #         bias_factor,
    #         hill_height,
    #         frequency=bias_frequency,
    #         saveFrequency=saveFrequency,
    #         biasDir=self.write_dir,
    #     )

    #     simulation.context.reinitialize(preserveState=True)

    #     logging.debug(f"Setting up reporters for {run_id}..")
    #     add_reporters(simulation,self.write_dir, f'metadynamics_{run_id}', total_steps, bias_frequency)

    #     if not self.verbose:
    #         # # Advance all steps at once do not record CVs
    #         meta.step(simulation, mMD_steps)
    #     else:
    #         # Record CVs along the way, might be usefull for debugging
    #         colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    #         for i in range(0, int(mMD_steps), self.record_CV):
    #             if i % self.store_CV == 0:
    #                 np.save(os.path.join(self.write_dir, f"COLVAR_{run_id}.npy"), colvar_array)

    #             meta.step(simulation, self.record_CV)
    #             current_cvs = meta.getCollectiveVariables(simulation)
    #             colvar_array = np.append(colvar_array, [current_cvs], axis=0)

    #     np.save(os.path.join(self.write_dir, f"COLVAR_{run_id}.npy"), colvar_array)
    #     np.save(os.path.join(self.write_dir, f"FE_{run_id}.npy"), meta.getFreeEnergy())

    #     # Create plots for all current runs
    #     # plot_colvar(self.write_dir, 'COM_dist')
    #     # plot_bias(self.write_dir, grid_min, grid_max, grid)
    #     # plot_FE(self.write_dir, grid_min, grid_max, grid)

    #     final_positions = simulation.context.getState(getPositions=True).getPositions()
    #     save_system(system, f'{self.write_dir}/system_mMD_{run_id}.xml')
    #     save_simulation(simulation, f'{self.write_dir}/mMD_checkpoint_{run_id}')
    #     save_pdb(self.topology, final_positions, f'{self.write_dir}/mMD_{run_id}.pdb')

    #     simulation_time = time.monotonic() - start_time
    #     logging.info(f'Finished {run_id} metadynamics in {simulation_time/60:.2f} min.')

    #     return