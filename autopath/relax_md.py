
import os
import time
import math
import numpy as np
import logging
from sys import stdout
from glob import glob

from utils import select_platform, add_reporters, load_system, save_pdb, save_system, save_simulation
from utils import get_ligand_ha, get_COG_dist
from equilibration import warm_up_system

from openmm.app.amberprmtopfile import AmberPrmtopFile
import openmm.unit as openmmunit
from openmm import *
from openmm.app import *
   
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

        self.checkpoint_file = checkpoint_file
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

    def run(self,
            pdb_file: str = None,
            run_id: str = None,
            vMD_time:float = 1,
        ):
        
        start_time = time.monotonic()

        initial_positions = PDBFile(pdb_file).positions

        # Calculate the number of steps required
        vMD_steps = math.ceil(vMD_time / self.timestep * 1000.0) #250.000 1ns at 4fs

        # logging.info('Setting up the integrator..')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(int(rep_idx))

        system = load_system(self.system_file)

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, system, integrator, self.platform)

        simulation.context.setPositions(initial_positions)

        # if self.checkpoint_file is not None:
        #     logging.info('Loading simulation checkpoint..')
        #     simulation.loadCheckpoint(self.checkpoint_file)

        startdist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        logging.info(f'Initial COM distance is {startdist:.2f} nm')
        
        # logging.info(f'Running {vMD_time} ns..')
        
        logging.info('Minimizing..')
        simulation.minimizeEnergy()

        if vMD_steps == 0: #just do minim and temp annealing
            logging.info('Warming up the system..')
            warm_up_system(simulation, integrator, warming_steps=25000, timestep=0.004)
            final_positions = simulation.context.getState(getPositions=True).getPositions()
            save_simulation(simulation, f'{self.sys_name}/{run_id}_relax_checkpoint')
            save_system(system, f'{self.sys_name}/{run_id}_relax_system.xml')
            save_pdb(self.topology, final_positions, f'{self.sys_name}/{run_id}_relax.pdb')

        else:
            add_reporters(simulation, self.sys_name, f'relax_{run_id}', vMD_steps, 250)
            # add a barostat
            simulation.step(vMD_steps)

            save_simulation(simulation, f'{self.sys_name}/{run_id}_checkpoint')
            save_pdb(self.topology, final_positions, f'{self.sys_name}/{run_id}.pdb')

        # Get COM distance
        current_dist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        logging.info(f'Current distance is {current_dist:.2f} nm')
        
        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished pose relaxation in {simulation_time/60:.2f} min.')