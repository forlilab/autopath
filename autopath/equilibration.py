import time
import logging
import numpy as np
from sys import stdout

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit
from openmm.app.amberprmtopfile import AmberPrmtopFile

from autopath.utils import *

def warm_up_system(simulation, integrator,
                   Tstart: int=5, Tend: int=300, Tstep: int=5,
                   timestep: float=0.001,
                   warming_steps: int=100000):

    """
    Run simulated annealing equilibration. 
    WarmUp with in NVT ensemble, slowly increasing the temperature
    """

    integrator.setStepSize(timestep * openmmunit.picoseconds)
    logging.debug(f'Stepsize set to {integrator.getStepSize()}')
    simulation.context.reinitialize(preserveState=True)

    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)
    
    # Warm up the system gradually
    for i in range(nT):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        logging.debug(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    return

def equilibrate_restrained_system(simulation, system, integrator, temp) -> None:
    """ Do restrained equilibration, releasing constraints 
    on protein and ligands and increasing timestep
    """

    equil_scheme={    
    'step1': {'k_prot': 5.0, 'k_lig': 5.0, 'npt_flag': False, 'nsteps': 50000, 'stepsize': 0.002},
    'step2': {'k_prot': 4.5, 'k_lig': 5.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step3': {'k_prot': 4.0, 'k_lig': 5.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step4': {'k_prot': 3.5, 'k_lig': 5.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step5': {'k_prot': 3.0, 'k_lig': 4.5, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step6': {'k_prot': 2.5, 'k_lig': 4.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step7': {'k_prot': 2.0, 'k_lig': 3.5, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step8': {'k_prot': 1.5, 'k_lig': 3.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step9': {'k_prot': 1.0, 'k_lig': 3.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step10': {'k_prot': 0.5, 'k_lig': 2.5, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step11': {'k_prot': 0.0, 'k_lig': 2.0, 'npt_flag': True, 'nsteps': 25000, 'stepsize': 0.004},
    'step12': {'k_prot': 0.0, 'k_lig': 1.5, 'npt_flag': True, 'nsteps': 50000, 'stepsize': 0.004},
    'step13': {'k_prot': 0.0, 'k_lig': 1.0, 'npt_flag': True, 'nsteps': 50000, 'stepsize': 0.004},
    'step14': {'k_prot': 0.0, 'k_lig': 0.5, 'npt_flag': True, 'nsteps': 50000, 'stepsize': 0.004},
    'step15': {'k_prot': 0.0, 'k_lig': 0.0, 'npt_flag': True, 'nsteps': 50000, 'stepsize': 0.004},
    }
    
    # Initialize variables as None
    k_lig_prev, k_prot_prev, npt_prev, stepsize_prev = None, None, None, None

    for step_name, params in equil_scheme.items():
        k_lig = params['k_lig']
        k_prot = params['k_prot']
        npt_flag = params['npt_flag']
        nsteps = params['nsteps']
        stepsize = params['stepsize']

        logging.info(f"Equilibration {step_name} with K_prot={k_prot} - K_lig={k_lig}")

        # Adjust force constant for the ligand if it has changed
        if k_lig_prev is None or k_lig != k_lig_prev: 
            simulation.context.setParameter('k_lig', (k_lig * openmmunit.kilocalories_per_mole / openmmunit.angstroms**2))

        # Adjust force constant for the protein if it has changed
        if k_prot_prev is None or k_prot != k_prot_prev: 
            simulation.context.setParameter('k_prot', (k_prot * openmmunit.kilocalories_per_mole / openmmunit.angstroms**2))

        # Enable NPT if needed
        # if npt_prev is None or npt_flag != npt_prev and npt_flag:
        if npt_flag != npt_prev and npt_flag:
            logging.info(f'Adding a Montecarlo Barostat to the system')
            system.addForce(MonteCarloBarostat(1 * openmmunit.atmosphere, temp))
            simulation.context.reinitialize(preserveState=True)

        # Adjust the timestep if it has changed
        if stepsize_prev is None or stepsize != stepsize_prev:
            integrator.setStepSize(stepsize)
            simulation.context.reinitialize(preserveState=True)
            logging.info(f'Stepsize set to {integrator.getStepSize()}')

        # Run the simulation for the specified number of steps
        simulation.step(nsteps)

        # Update previous values
        k_lig_prev = k_lig
        k_prot_prev = k_prot
        npt_prev = npt_flag
        stepsize_prev = stepsize

    return
           
class Equilibration:
    def __init__(self,
                 system_file:str = 'system.xml',
                 prmtop_file:str = 'system.prmtop',
                 sys_name:str = None,
                 lig_name:str = 'UNK',
                 temperature: float = 300,
                 timestep:float = 0.004,
                 ) -> None:

        self.system = load_system(system_file)
        prmtop = AmberPrmtopFile(prmtop_file)
        self.topology = prmtop.topology
        self.sys_name = sys_name
        self.lig_name = lig_name

        self.temperature = temperature * openmmunit.kelvin
        self.timestep = timestep * openmmunit.picoseconds
 
        self.platform = select_platform('fastest')

        return

    def run(self, pdb_file):
        
        start_time = time.monotonic()

        logging.info('Setting up the integrator..')
        integrator = LangevinMiddleIntegrator(self.temperature, 1/openmmunit.picoseconds, self.timestep)
        # integrator.setRandomNumberSeed(seed)
        integrator.setConstraintTolerance(0.00001)

        logging.info(f'Creating the simulation for {self.sys_name}..')
        simulation = Simulation(self.topology, self.system, integrator, self.platform)

        initial_positions = PDBFile(pdb_file).positions

        simulation.context.setPositions(initial_positions)

        logging.info(f'Setting up reporters for {self.sys_name}..')
        add_reporters(simulation, self.sys_name, 'equilibration', logperiod=2000, total_steps=600000)

        logging.info('Adding harmonic restraints to the protein..')
        prot_ha_idx, prot_ha_names = get_protein_ha(self.topology, self.lig_name)
        logging.debug(f"The following protein heavy atoms will be restrained: {', '.join(prot_ha_names)}")
        add_harmonic_restraints(self.system, initial_positions,
                                self.topology, prot_ha_idx, restraint_force=5,
                                force_name='k_prot', force_group=12)
        
        logging.info('Adding harmonic restraints to the ligand..')
        lig_ha_idx, lig_ha_names = get_ligand_ha(self.topology, self.lig_name)
        logging.debug(f"The following ligand heavy atoms will be restrained: {', '.join(lig_ha_names)}")
        add_harmonic_restraints(self.system, initial_positions,
                                self.topology, lig_ha_idx, restraint_force=5,
                                force_name='k_lig', force_group=13)

        logging.info('Minimizing..')
        simulation.minimizeEnergy()

        logging.info('Warming up the system..')
        warm_up_system(simulation, integrator, warming_steps=100000)
        
        logging.info('Running restrained equilibration protocol..')
        equilibrate_restrained_system(simulation, self.system, integrator, self.temperature)

        # Remove both protein and ligand force restraints
        simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces()-3)
        simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces()-2)
        print_current_forces(self.system)

        final_positions = simulation.context.getState(getPositions=True).getPositions()

        save_system(self.system, f'{self.sys_name}/system_equilibrated.xml')
        save_simulation(simulation, f'{self.sys_name}/equilibration_checkpoint')
        save_pdb(self.topology, final_positions, f'{self.sys_name}/system_equilibrated.pdb')
        
        simulation_time = time.monotonic() - start_time
        logging.info(f'Restrained equilibration completed in {simulation_time/60:.2f} min.')

        return None