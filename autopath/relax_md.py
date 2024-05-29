import time
import numpy as np
import logging

from autopath.utils import *
from autopath.equilibration import warm_up_system

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit
from openmm.app.amberprmtopfile import AmberPrmtopFile


def add_flatbottom_restraint(system,
                             groupA, groupB, 
                             r0:float=None,
                             upper_wall:int=0.1, 
                             K_flat:float=200):

    fb_eq = '(k_flat/2)*max(distance(g1,g2) - upper_wall, r0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(groupA)
    upper_wall_rest.addGroup(groupB)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k_flat', K_flat*openmmunit.kilojoules_per_mole)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall*openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter('r0', r0*openmmunit.nanometer)

    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    
    upper_wall_rest.setForceGroup(30)

    system.addForce(upper_wall_rest)

    return None


class RelaxMD:
    def __init__(self,
                    checkpoint_file:str=None,
                    system_file:str=None,
                    prmtop_file:str=None,
                    sys_name:str= 'test',
                    lig_name:str = 'UNK',
                    pocket_atoms:list[int]=None,
                    use_flat_bottom_rest:bool=False,
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
        self.use_flat_bottom_rest = use_flat_bottom_rest

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
        
        logging.debug('Setting up the integrator..')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(int(rep_idx))

        system = load_system(self.system_file)

        # Setting Simulation object and loading the checkpoint
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.debug('Loading simulation checkpoint..')
            simulation.loadCheckpoint(checkpoint_file)
        else:
            initial_positions = PDBFile(pdb_file).positions
            simulation.context.setPositions(initial_positions)

        startdist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        
        if self.use_flat_bottom_rest:
            add_flatbottom_restraint(system, self.ligand_ha_idx, self.pocket_atoms, startdist)
                
        logging.debug('Minimizing..')
        simulation.minimizeEnergy()

        logging.debug('Warming up the system..')
        warm_up_system(simulation, integrator, warming_steps=md_steps, timestep=self.timestep)
        
        logging.debug('Minimizing..')
        simulation.minimizeEnergy()
        
        if self.use_flat_bottom_rest:
            # Remove the force before saving
            simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces()-1)

        #save stuff
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_simulation(simulation, f'{self.sys_name}/milestones/{run_id}_relax_checkpoint')
        save_system(system, f'{self.sys_name}/milestones/{run_id}_relax_system.xml')
        save_pdb(self.topology, final_positions, f'{self.sys_name}/milestones/{run_id}_relax.pdb')

        # Get COM distance
        finaldist = get_COG_dist(simulation, self.ligand_ha_idx, self.pocket_atoms)
        
        logging.info(f'{run_id} - Initial:{startdist:.3f} nm - Final:{finaldist:.3f} nm')

        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished {run_id} relaxation in {simulation_time/60:.2f} min.')

        return startdist, finaldist