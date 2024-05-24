import time
import numpy as np
import logging

from autopath.utils import *
from autopath.equilibration import warm_up_system

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit
from openmm.app.amberprmtopfile import AmberPrmtopFile

class RelaxMD:
    def __init__(self,
                    checkpoint_file:str=None,
                    system_file:str=None,
                    prmtop_file:str=None,
                    sys_name:str= 'test',
                    lig_name:str = 'UNK',
                    pocket_atoms:list[int]=None,
                    HMR:bool= True,
                    temp:float= 300,
                    ):

        self.system_file = system_file

        prmtop = AmberPrmtopFile(prmtop_file)
        self.topology = prmtop.topology
        
        self.sys_name = sys_name

        if HMR:
            self.timestep = 0.004
        else:
            self.timestep = 0.002

        self.temperature = temp * openmmunit.kelvin

        self.ligand_ha_idx, self.lig_ha_names  = get_ligand_ha(self.topology, lig_name)
        self.pocket_atoms = pocket_atoms

        # Select MD platform
        self.platform = select_platform('fastest')

        return
    
    def run(self,
            checkpoint_file:str = None,
            pdb_file: str = None,
            run_id: str = None,
            md_steps:int = 25000,
        ):
        
        start_time = time.monotonic()
        
        # logging.info('Setting up the integrator..')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(int(rep_idx))

        system = load_system(self.system_file)

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.info('Loading simulation checkpoint..')
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        logging.info(f'Initial COG distance is {startdist:.2f} nm')
        
        # logging.info(f'Running {vMD_time} ns..')
        
        logging.info('Minimizing..')
        simulation.minimizeEnergy()

        logging.info('Warming up the system..')
        warm_up_system(simulation, integrator, warming_steps=md_steps, timestep=self.timestep)
        
        #save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_simulation(simulation, f'{self.sys_name}/milestones/{run_id}_relax_checkpoint')
        save_system(system, f'{self.sys_name}/milestones/{run_id}_relax_system.xml')
        save_pdb(self.topology, final_positions, f'{self.sys_name}/milestones/{run_id}_relax.pdb')

        # Get COM distance
        current_dist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        logging.info(f'Current COG distance is {current_dist:.2f} nm')
        
        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished pose relaxation in {simulation_time/60:.2f} min.')

        return