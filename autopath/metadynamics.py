import os
import math
import time
import logging
import numpy as np
from datetime import datetime

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import cvpack
from glob import glob
from autopath.utils import *
from autopath.customForces import *
from autopath.analysis import (
    plot_bias,
    plot_colvar,
    plot_FE,
    plot_FE_2D,
    plot_colvar_2D,
)

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDMolSupplier


class MetadynamicsMD:

    def __init__(
        self,
        topology: str = None,
        ligand_atoms: list[int] = None,
        pocket_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        out_dir: str = "metadynamics",
        is_membrane: bool = False,
        timestep: float = 0.004, #  # 4 fs timestep
        temp: float = 300,
        use_GReweighting: bool = False,
        platform: str = "fastest",
        verbose: bool = True,
    ) -> None:
        
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.topology = topology
        self.n_atoms = self.topology.getNumAtoms()
        self.is_membrane = is_membrane

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temp * openmmunit.kelvin

        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        
        self.restrained_atoms = restrained_atoms

        self.platform = select_platform(platform)

        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            if not girsanov:
                logging.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logging.info("Using Girsanov reweighting for steered MD.")
            
        # These are for debugging purposes if one wants to check the CVs over the time of the simulation
        self.verbose = verbose
        self.record_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 10)  # record the CVs every 10 ps
        self.store_CV = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * 100)  # log the stored COLVAR every 100ps

        return

    def run(
        self,
        pdb_file: str = None,
        system: str = None,
        state_xml: str = None,
        checkpoint_file: str = None,
        run_id: str = None,
        mMD_CV: str = "com",
        mMD_time: int = 10,
        bias_factor: float = 10,
        hill_height: float = 1.2, #  # 1.2 kJ/mol approx 0.5 KbT
        hill_width: float = 0.05,
        grid_dimensions: tuple = (0.0, 1.0),
        grid_points: int = 125,
        biasFrequency: int = 2,
        saveFrequency: int = 50,
    ) -> None:

        start_time = time.monotonic()

        assert mMD_CV in [
            "com",
            "com_z",
            "com_x_y_distance",            
            "rmsd",
            "rmsd_states",
            "path_rmsd",
            "path_cv",
            "nc",
        ], f"The selected colective variable {mMD_CV} is not implemented"

        # ensure proper formating of the file name
        if run_id is None:
            run_id = f'W-{datetime.now().strftime("%H%M%S")}'

        # Calculate the number of steps required
        mMD_steps = math.ceil(mMD_time / self.timestep.value_in_unit(openmmunit.picoseconds) * 1000.0)  # 250.000 1ns at 4fs
        biasFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * biasFrequency)  # deposit bias every 2 ps (250 is 1ps at 4fs timestep)
        saveFrequency = int((1/self.timestep.value_in_unit(openmmunit.picoseconds)) * saveFrequency)  # write bias every 50ps

        hill_height = hill_height * openmmunit.kilojoules_per_mole
        grid_min, grid_max = grid_dimensions

        logging.info(f"Running metadynamics with Colective Variable {mMD_CV}")
        logging.info(f"Grid boundaries are min={grid_min:.3f} - max={grid_max:.3f}")
        logging.info(f"Sigma is {hill_width} nm and there are {grid} grid points ")

        logging.debug("Setting up the integrator")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(int(rep_idx))

        if self.topology is None:
            if pdb_file is None:
                logging.error(
                    f"Either a PDB or a prmtop file must be provided to get the topology from"
                )
                exit(1)
            else:
                self.topology = PDBFile(pdb_file).topology

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)
        simulation.context.setPeriodicBoxVectors(*self.topology.getPeriodicBoxVectors())
        box_vec=self.topology.getPeriodicBoxVectors()
        print(f'setting box vectors: {box_vec} ')

        if checkpoint_file is not None:
            logging.debug(f"Loading simulation checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            if state_xml is not None:
                with open(state_xml) as f:
                    state = XmlSerializer.deserialize(f.read())
                    simulation.context.setState(state)
            else:
                if pdb_file is not None:
                    logging.debug(f"Setting positions from PDB file {pdb_file}")
                    simulation.context.setPositions(PDBFile(pdb_file).positions)
                else:
                    logging.error(
                        f"Either a PDB or a checkpoint file or state xml must be provided to get coordinates from"
                    )
                    exit(1)

        # Add harmonic positional restraints to protein CA
        input_positions = simulation.context.getState(getPositions=True).getPositions()

        if self.restrained_atoms is not None:
            add_harmonic_restraints(
                system,
                input_positions,
                self.topology,
                self.restrained_atoms,
                10,
                "k_CA",
                14,
            )
            
        # lig_name = "UNK"
        # ligand_atoms = [a.index for a in self.topology.atoms() if a.residue.name == lig_name]
        # add_cylindrical_restraints(system, host_index=self.pocket_atoms, guest_index=ligand_atoms, R_cylinder=1.5 * openmmunit.nanometers, force_group=31)

        logging.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"metadynamics_{run_id}",
            total_steps,
            saveFrequency,
        )
        if mMD_CV == "com":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            cv = cvpack.CentroidFunction(
                f"sqrt(distance(g1,g2)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=True,
                pbc=False,
            )
        if mMD_CV == "com_z":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            cv = cvpack.CentroidFunction(
                f"sqrt(pointdistance(0,0,z1,0,0,z2)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
            )
        if mMD_CV == "com_x_y_distance":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            cv = cvpack.CentroidFunction(
                f"sqrt(pointdistance(x1,y1,0,x2,y2,0)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
            )

        elif mMD_CV == "rmsd":
            cv = cvpack.RMSD(input_positions, self.ligand_atoms, self.n_atoms)

        elif mMD_CV == "rmsd_states":

            atom_names_to_match = ["CA"]
            system_residues = [r for r in self.topology.residues() if r.name not in ["UNK", "HOH", "NA", "CL", "K"]]

            atom_indexes_to_match = []
            for residue in system_residues:
                for atom in residue.atoms():
                    if atom.name in atom_names_to_match:
                        atom_indexes_to_match.append(atom.index)

            states_pdbs = glob("input/milestone_*.pdb")
            milestones_dicts = [self._get_reference_dict(pdb, atom_names_to_match, atom_indexes_to_match) for pdb in states_pdbs]

            cv = cvpack.PathInRMSDSpace(
                metric=cvpack.path.progress,
                milestones=milestones_dicts,
                sigma=0.01 * openmmunit.nanometers,
                numAtoms=self.n_atoms,
            )
        elif mMD_CV == "path_rmsd":

            # atom_names_to_match = ["CA"]
            ligand_residue = [r for r in self.topology.residues() if r.name == "UNK"]
            ligand_atoms_to_match = [a.index for a in ligand_residue[0].atoms() if not a.name.startswith("H")]
            ligand_names_to_match = [a.name for a in ligand_residue[0].atoms() if not a.name.startswith("H")]
            logging.info(f"Matched {len(ligand_atoms_to_match)} heavy atoms from the ligand")
            sys_name =self.out_dir.split('/')[0] 
            milestones = glob(f'{sys_name}/milestones/pdbs/milestone_*.pdb')
            milestones.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
       
            milestones_dicts = [self._get_reference_dict(pdb, ligand_names_to_match, ligand_atoms_to_match) for pdb in milestones]
            cv = cvpack.PathInRMSDSpace(
                metric=cvpack.path.progress,
                milestones=milestones_dicts,
                sigma=0.001 * openmmunit.nanometers,
                numAtoms=self.n_atoms,
            )
        elif mMD_CV == "path_cv":

            from copy import deepcopy

            ligand_residue = [r for r in self.topology.residues() if r.name == "UNK"]
            ligand_atoms_to_match = [a.index for a in ligand_residue[0].atoms() if not a.name.startswith("H")]
            ligand_names_to_match = [a.name for a in ligand_residue[0].atoms() if not a.name.startswith("H")]
            logging.info(f"Matched {len(ligand_atoms_to_match)} heavy atoms from the ligand")
            print(ligand_atoms_to_match)
            
            sys_name = '6dy7_A'
            milestones = glob(f'{sys_name}/milestones/pdbs/milestone_*.pdb')           

            print(f"Found {len(milestones)} milestones")
            
            milestones_array = np.zeros((len(milestones), 2))
            for i, milestone in enumerate(milestones):
                milestone_name = os.path.basename(milestone)
                _system = deepcopy(system)
                _context = Context(_system, VerletIntegrator(1.0), self.platform)
                milestone_positions = PDBFile(milestone).positions
                _context.setPositions(milestone_positions)
                logging.info(f"Calculating CVs for {milestone_name}..")
                cv1 = cvpack.RMSD(milestone_positions, self.ligand_atoms, self.n_atoms)
                cv2 = cvpack.CentroidFunction(
                                                f"sqrt(distance(g1,g2)^2)",
                                                openmmunit.nanometers,
                                                groups=[self.pocket_atoms] + [self.ligand_atoms],
                                                weighByMass=True,
                                                pbc=False,
                                            )
                cv1.addToSystem(_system)
                cv2.addToSystem(_system)
                _context.reinitialize(preserveState=True)
                _context.setPositions(milestone_positions)
                logging.info(f"Milestone {milestone_name} CV1: {cv1.getValue(_context)}, CV2: {cv2.getValue(_context)}")
                milestones_array[i, 0] = round(cv1.getValue(_context).value_in_unit(openmmunit.nanometers),2)
                milestones_array[i, 1] = round(cv2.getValue(_context).value_in_unit(openmmunit.nanometers),2)    
            
            print(milestones_array)

            cv1 = cvpack.RMSD(input_positions, self.ligand_atoms, self.n_atoms)
            cv2 = cvpack.CentroidFunction(
                                        f"sqrt(distance(g1,g2)^2)",
                                        openmmunit.nanometers,
                                        groups=[self.pocket_atoms] + [self.ligand_atoms],
                                        weighByMass=True,
                                        pbc=False,
                                    )
 
            cv = cvpack.PathInCVSpace(
                metric=cvpack.path.progress,
                variables=[cv1, cv2],
                milestones=milestones_array,
                sigma=0.001 * openmmunit.nanometers,
            )

        elif mMD_CV == "nc":

            forces = {f.getName(): f for f in system.getForces()}

            cv = cvpack.NumberOfContacts(
                self.pocket_atoms,
                self.ligand_atoms,
                forces["NonbondedForce"],
                stepFunction="1/(1+x^6)",
                thresholdDistance=0.35,
                cutoffFactor=2.0,
                switchFactor=1.5,
                reference=50,
            )

        bias_variable = BiasVariable(
            cv,
            minValue=grid_min,
            maxValue=grid_max,
            biasWidth=hill_width,
            gridWidth=grid_points,
            periodic=False,
        )

        # Set up the metadynamics object
        meta = Metadynamics(
            system,
            [bias_variable],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=biasFrequency,
            saveFrequency=saveFrequency,
            biasDir=self.out_dir,
        )

        meta._force.setForceGroup(1)            # force group 1 for reweighting

        simulation.context.reinitialize(preserveState=True)  
        
        # print_current_forces(system)

        if not self.verbose:
            # # Advance all steps at once do not record CVs
            meta.step(simulation, mMD_steps)
        else:
            # Record CVs along the way, might be usefull for debugging
            colvar_array = np.array([meta.getCollectiveVariables(simulation)])
            for i in range(0, int(mMD_steps), self.record_CV):
                if i % self.store_CV == 0:
                    np.save(
                        os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"),
                        colvar_array,
                    )
                print(f'current CV: {meta.getCollectiveVariables(simulation)}')
                meta.step(simulation, self.record_CV)
                current_cvs = meta.getCollectiveVariables(simulation)
                colvar_array = np.append(colvar_array, [current_cvs], axis=0)

        np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)
        np.save(os.path.join(self.out_dir, f"FE_{run_id}.npy"), meta.getFreeEnergy())

        # Create plots for all current runs
        plot_colvar(self.out_dir, mMD_CV)
        plot_bias(self.out_dir, grid_min, grid_max, grid_points, mMD_CV)
        plot_FE(self.out_dir, grid_min, grid_max, grid_points, mMD_CV)

        # Save everything
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_system(system, f"{self.out_dir}/WTMetaD_system_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/WTMetaD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/WTMetaD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return run_id

    def run2D(
        self,
        pdb_file: str = None,
        ligand_sdf: str = None,
        system: str = None,
        checkpoint_file: str = None,
        state_xml: str = None,
        run_id: str = None,
        mMD_time: int = 10,
        bias_factor: float = 10,
        hill_height: float = 0.3,
        hill_width_A: float = 0.01,
        grid_dimensions_A: tuple = (0.0, 1.0),
        hill_width_B: float = 0.01,
        grid_dimensions_B: tuple = (0.0, 1.0),
        biasFrequency: int = 2,
        saveFrequency: int = 50,
        cv1: str = "com_z",
        cv2: str = "lmz",
        atom1: list = None, #these are the two atoms in the delta z calculation....make this more clear somehow?
        atom2: list = None, #these are the two atoms in the delta z calculation....make this more clear somehow?
        reference_points: list = None, 
        # reference_points defines the vector/axis for com_distance_along_vector
        # Example: [[x_start, y_start, z_start], [x_end, y_end, z_end]]
        rmsd_1_reference_positions: list = None,
        rmsd_1_group: list = None,
        rmsd_2_reference_positions: list = None,
        rmsd_2_group: list = None,
    ) -> None:

        # Validate cv1
        if cv1 not in ["com_z", 'com', "rmsd", "com_distance_along_vector", "com_z_distance"]:
            logging.error(f"Invalid value for cv1: '{cv2}'. Must be 'com_z', 'com' or 'rmsd' or 'com_distance_along_vector'.")

        # Validate cv2
        if cv2 not in ["lmz", "dz", "rmsd", 'nc', 'proj_vec', 'com_x_y_distance']:
            logging.error(f"Invalid value for cv2: '{cv2}'. Must be 'lmz', 'dz', 'rmsd', or  'nc' or 'proj_vec.")

        # Require atom1 and atom2 if using distance-based CV
            if cv2 == "dz":
                if not atom1 or not atom2:
                    raise ValueError("When cv2 is 'dz', atom1 and atom2 must be provided as lists of atom indices.")

        start_time = time.monotonic()

        # Metadynamics time in ns
        mMD_steps = math.ceil(mMD_time / self.timestep * 1000.0)  # 250.000 1ns at 4fs
        biasFrequency = 250 * biasFrequency  # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps
        hill_height = hill_height * openmmunit.kilocalories_per_mole
        
        logging.debug("Setting up the integrator")
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / openmmunit.picoseconds, self.timestep
        )
        # integrator.setRandomNumberSeed(int(rep_idx))

        if self.topology is None:
            if pdb_file is None:
                logging.error(f"Either a PDB or a prmtop file must be provided to get the topology from")
                exit(1)
            else:
                self.topology = PDBFile(pdb_file).topology
                self.topology = PDBFile(pdb_file).topology

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)
        simulation.context.setPeriodicBoxVectors(*self.topology.getPeriodicBoxVectors()) #added this in metadynmaics.py

        if checkpoint_file is not None:
            logging.debug(f"Loading simulation checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            if state_xml is not None:
                with open(state_xml) as f:
                    logging.debug(f"Simulation state restored from {state_xml}")
                    state = XmlSerializer.deserialize(f.read())
                    simulation.context.setState(state)
            else:
                if pdb_file is not None:
                    logging.debug(f"Setting positions from PDB file {pdb_file}")
                    simulation.context.setPositions(PDBFile(pdb_file).positions)
                else:
                    logging.error(
                        f"Either a PDB or a checkpoint file or state xml must be provided to get coordinates from"
                    )
                    exit(1)
        # Add harmonic positional restraints to protein CA
        input_positions = simulation.context.getState(getPositions=True).getPositions()

        if self.restrained_atoms is not None:
            add_harmonic_restraints(
                system,
                input_positions,
                self.topology,
                self.restrained_atoms,
                10,
                "k_CA",
                14,
            )

        grid_width_A = hill_width_A / 5
        grid_min_A, grid_max_A = grid_dimensions_A
        grid_A = round(abs(grid_min_A - grid_max_A) / grid_width_A)
        grid_width_B = hill_width_B / 5
        grid_min_B, grid_max_B = grid_dimensions_B
        grid_B = round(abs(grid_min_B - grid_max_B) / grid_width_B)



        ##################### com distance along vector CV (cv1) #################################

        if cv1 == "com_distance_along_vector":
            groups = [self.ligand_atoms]
            print(f'Ligand group: {groups}')

            a, b, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
            box_x = a[0].value_in_unit(unit.nanometer)  # x-length
            box_y = b[1].value_in_unit(unit.nanometer)  # y-length
            box_z = c[2].value_in_unit(unit.nanometer)  # z-length

            for i in range(system.getNumForces()):
                f = system.getForce(i)
                if hasattr(f, "setGlobalParameterDefaultValue"):
                    for j in range(f.getNumGlobalParameters()):
                        name = f.getGlobalParameterName(j)
                        if name == "box_x":
                            f.setGlobalParameterDefaultValue(j, box_x)
                        elif name == "box_y":
                            f.setGlobalParameterDefaultValue(j, box_y)
                        elif name == "box_z":
                            f.setGlobalParameterDefaultValue(j, box_z)

            # reference points (in nm)
            p1 = np.array(reference_points[0])
            p2 = np.array(reference_points[1])

            # displacement vector with PBC
            dp = p2 - p1
            dp_norm = np.linalg.norm(dp)
            ux, uy, uz = dp / dp_norm

            # Expression for displacement between COM (x1,y1,z1) and reference p1 with PBC
            dx = f"(x1 - {p1[0]} - box_x*floor((x1 - {p1[0]})/box_x + 0.5))"
            dy = f"(y1 - {p1[1]} - box_y*floor((y1 - {p1[1]})/box_y + 0.5))"
            dz = f"(z1 - {p1[2]} - box_z*floor((z1 - {p1[2]})/box_z + 0.5))"

            # projection along reference unit vector
            proj_expr = f"({ux})*{dx} + ({uy})*{dy} + ({uz})*{dz}"

            # Wrap as CV
            com_along_vec_cv = cvpack.CentroidFunction(
                proj_expr,
                openmmunit.nanometer,
                groups,
                weighByMass=True,
                pbc=True,
                box_x = box_x,
                box_y = box_y,
                box_z = box_z,
            )

            # Bias variable for metadynamics
            grid_width_A = hill_width_A / 5
            grid_min_A, grid_max_A = grid_dimensions_A
            grid_A = round(abs(grid_min_A - grid_max_A) / grid_width_A)

            cv1_BiasVariable = BiasVariable(
                com_along_vec_cv,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False,
                gridWidth=grid_A,
            )


        ##################### Z depth CV (cv1) #################################

        if cv1 == "com_z":
            dummy_atom = [atom.index for atom in self.topology.atoms() if atom.residue.name == "DUM"]
            if not dummy_atom:
                raise ValueError("Could not find ligand (UNK) or dummy atom (DUM) in the topology.")

            #add z size parameter
            _, _, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
            box_z = c[2].value_in_unit(openmmunit.nanometers)


            groups = [self.ligand_atoms] + [dummy_atom]
            print(f'groups: {groups}')
            COM_Z = cvpack.CentroidFunction(
                # f"z1-z2",
                # f"z1",
                "select(step((z1 - z2)/box_z - floor((z1 - z2)/box_z) - 0.5), -1, 1) * pointdistance(0, 0, z1, 0, 0, z2)",
                # "select(step((z1 - 0)/box_z - floor((z1 - 0)/box_z) - 0.5), -1, 1) * pointdistance(0, 0, z1, 0, 0, 0)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
                box_z = box_z
            )



            cv1_BiasVariable = BiasVariable(
                COM_Z,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
        )
        
        ##################### RMSD CV (cv1) #################################
        if cv1 == "rmsd":
            
            num_atoms = system.getNumParticles()

            # Instantiate the RMSD collective variable
            if rmsd_1_reference_positions is not None:
                rmsd_cv = cvpack.RMSD(
                    referencePositions=rmsd_1_reference_positions,
                    group=rmsd_1_group,
                    numAtoms=num_atoms,  
                    name='rmsd_1'
                )
            else:
                print('biasing RMSD from input position')
                rmsd_cv = cvpack.RMSD(input_positions, rmsd_1_group, num_atoms)

            cv1_BiasVariable = BiasVariable(
                rmsd_cv,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
        )        

        ##################### COM CV (cv1) #################################
        if cv1 == "com":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            com_cv = cvpack.CentroidFunction(
                f"sqrt(distance(g1,g2)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=True,
                pbc=False,
            )
            cv1_BiasVariable = BiasVariable(
                com_cv,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
        )  

        ##################### com_x_y_distance CV (cv2) #################################
        if cv1 == "com_z_distance":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            com_x_y_distance_cv = cvpack.CentroidFunction(
                f"sqrt(pointdistance(0,0,z1,0,0,z2)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
            )


            cv1_BiasVariable = BiasVariable(
                com_x_y_distance_cv,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
        ) 
        ##################### Lipophilicity moment CV (cv2) #################################

        # get crippen contribution list
        if cv2 == "lmz":
            logging.info(f"Calculating Crippen contributions for the ligand")
            ligand_mol = SDMolSupplier(ligand_sdf, removeHs=False)[0]
            atom_contribs = rdMolDescriptors._CalcCrippenContribs(ligand_mol)
            clogp_contributions = [contrib[0] for contrib in atom_contribs]

            # Get the centroid (x, y, z) of the molecule "
            
            for i in self.ligand_atoms:
                if i == 0:
                    centroid_x = f"(x{i+1})"
                    centroid_y = f"(y{i+1})"
                    centroid_z = f"(z{i+1})"

                else:
                    centroid_x += f" + (x{i+1})"
                    centroid_y += f" + (y{i+1})"
                    centroid_z += f" + (z{i+1})"

            N = len(self.ligand_atoms)
            centroid_x = f"(({centroid_x}) / {N})"
            centroid_y = f"(({centroid_y}) / {N})"
            centroid_z = f"(({centroid_z}) / {N})"

            #get:
            #delta_lipophilicity_x = sum ((atom_i_x-centroid_x)*crippen_contribution_i),
            #delta_lipophilicity_y  sum ((atom_i_y-centroid_y)*crippen_contribution_i),
            #delta_lipophilicity_z  sum ((atom_i_z-centroid_z)*crippen_contribution_i)
            
            for i in self.ligand_atoms:
                if i == 0:
                    delta_lipophilicity_x = f"(((x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y = f"(((y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z = f"(((z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"                 
                else:
                    delta_lipophilicity_x += f" + (((x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y += f" + (((y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z += f" + (((z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"   


            #now we want the -magnitude of the LM in the z direction (delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z)
            #((delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z) dot (0, 0, -1)/|(delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z)||(0, 0, -1)|)
            #(0*delta_lipophilicity_x + 0*delta_lipophilicity_y+ -1*delta_lipophilicity_z) / ((sqrt(0^2+0^2+(-1)^2)) * sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2)))
            #((-delta_lipophilicity_z) / (sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2))
            
            lm_num = f"({delta_lipophilicity_z})"
            lm_denom = f"sqrt((({delta_lipophilicity_x})^2) + (({delta_lipophilicity_y})^2) + (({delta_lipophilicity_z})^2))"
            lipophilicity_moment_z = f"({lm_num})/({lm_denom})"

            print(f'lipophilicity_moment z component: {lipophilicity_moment_z}')

            LM = cvpack.AtomicFunction(lipophilicity_moment_z,
                                        openmmunit.nanometer,
                                        self.ligand_atoms)

        


            cv2_BiasVariable = BiasVariable(
                LM,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False,
                gridWidth=grid_B,
            )



        ##################### Delta Z CV (cv2) #################################
        if cv2 == "dz":
            groups = [atom1] + [atom2]
            print(f'groups: {groups}')
            dz = "(z2-z1)/(sqrt((x1-x2)^2+(y1-y2)^2+(z2-z1)^2))"
            dz =  cvpack.CentroidFunction(dz,
                                        openmmunit.nanometer,
                                        groups,
                                        periodicBounds = None,
                                        pbc = True)

        
            grid_width_B = hill_width_B / 5
            grid_min_B, grid_max_B = grid_dimensions_B
            grid_B = round(abs(grid_min_B - grid_max_B) / grid_width_B)

            cv2_BiasVariable = BiasVariable(
                dz,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False,
                gridWidth=grid_B,
            )

        ##################### end-to-end distance projection along vector(cv2) #################################
        if cv2 == "proj_vec":
            # Groups: two atoms defining molecular vector
            groups = [[atom1], [atom2]]
            print(f'groups: {groups}')

            #get box vectors
            a, b, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
            box_x = a[0].value_in_unit(unit.nanometer)  # x-length
            box_y = b[1].value_in_unit(unit.nanometer)  # y-length
            box_z = c[2].value_in_unit(unit.nanometer)  # z-length
            
            for i in range(system.getNumForces()):
                f = system.getForce(i)
                if hasattr(f, "setGlobalParameterDefaultValue"):
                    for j in range(f.getNumGlobalParameters()):
                        name = f.getGlobalParameterName(j)
                        if name == "box_x":
                            f.setGlobalParameterDefaultValue(j, box_x)
                        elif name == "box_y":
                            f.setGlobalParameterDefaultValue(j, box_y)
                        elif name == "box_z":
                            f.setGlobalParameterDefaultValue(j, box_z)

            # Reference vector endpoints (in nm)
            p1 = np.array(reference_points[0])
            p2 = np.array(reference_points[1])
            dp = p2 - p1
            dp /= np.linalg.norm(dp)   # normalize
            ux, uy, uz = dp

            # PBC-aware displacement for atom2 - atom1
            dx = "(x2 - x1)"
            dy = "(y2 - y1)"
            dz = "(z2 - z1)"

            # Dot product projection, normalized by molecular vector length
            proj_expr = (
                f"(({ux})*{dx} + ({uy})*{dy} + ({uz})*{dz})"
                f" / sqrt({dx}*{dx} + {dy}*{dy} + {dz}*{dz})"
            )

            # Wrap as CV
            proj_cv = cvpack.CentroidFunction(
                proj_expr,
                openmmunit.dimensionless,
                groups,
                weighByMass=False,
                pbc=True,
                box_x = box_x,
                box_y = box_y,
                box_z = box_z,
            )

            # Bias setup
            grid_width_B = hill_width_B / 5
            grid_min_B, grid_max_B = grid_dimensions_B
            grid_B = round(abs(grid_min_B - grid_max_B) / grid_width_B)

            cv2_BiasVariable = BiasVariable(
                proj_cv,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False,
                gridWidth=grid_B,
            )

        ##################### RMSD CV (cv2) #################################
        if cv2 == "rmsd":
            
            num_atoms = system.getNumParticles()

            # Instantiate the RMSD collective variable
            if rmsd_2_reference_positions is not None:
                rmsd_cv = cvpack.RMSD(
                    referencePositions=rmsd_2_reference_positions,
                    group=rmsd_2_group,
                    numAtoms=num_atoms,  
                    name=name
                )

            else:
                print('biasing RMSD from input position')
                rmsd_cv = cvpack.RMSD(input_positions, rmsd_2_group, num_atoms)

            cv2_BiasVariable = BiasVariable(
                rmsd_cv,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False, 
                gridWidth=grid_B,
        )    


        ##################### NC CV (cv2) #################################
        if cv2 == "nc":

            forces = {f.getName(): f for f in system.getForces()}

            nc_cv = cvpack.NumberOfContacts(
                atom1,
                atom2,
                forces["NonbondedForce"],
                stepFunction="1/(1+x^6)",
                thresholdDistance=0.35,
                cutoffFactor=2.0,
                switchFactor=1.5,
                reference=50,
            )

            cv2_BiasVariable = BiasVariable(
                nc_cv,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False, 
                gridWidth=grid_B,
        )    


        ##################### com_x_y_distance CV (cv2) #################################
        if cv2 == "com_x_y_distance":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            com_x_y_distance_cv = cvpack.CentroidFunction(
                f"sqrt(pointdistance(x1,y1,0,x2,y2,0)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
            )


            cv2_BiasVariable = BiasVariable(
                com_x_y_distance_cv,
                minValue=grid_min_B,
                maxValue=grid_max_B,
                biasWidth=hill_width_B,
                periodic=False, 
                gridWidth=grid_B,
        ) 

     ##############################################################
        meta = Metadynamics(
            system,
            [cv1_BiasVariable, cv2_BiasVariable],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=biasFrequency,
            saveFrequency=saveFrequency,
            biasDir=self.out_dir,
        )

        simulation.context.reinitialize(preserveState=True)
        
        logging.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"metadynamics_{run_id}",
            mMD_steps,
            biasFrequency,
        )

        # if not self.verbose:
        #     # # Advance all steps at once do not record CVs
        #     meta.step(simulation, mMD_steps)
        # else:
            # Record CVs along the way, might be usefull for debugging
        colvar_array = np.array([meta.getCollectiveVariables(simulation)])
        for i in range(0, int(mMD_steps), self.record_CV):
            if i % self.store_CV == 0:
                np.save(
                    os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"),
                    colvar_array,
                )


            # Print cylindrical restraint energy (force group 10)
            state = simulation.context.getState(getEnergy=True, groups={10})
            cylinder_energy = state.getPotentialEnergy()
            print(f'cylindrical restraint energy: {cylinder_energy}') 
            print(f'current CV: {meta.getCollectiveVariables(simulation)}')
            meta.step(simulation, self.record_CV)
            if cv1 == "com_z":
                _, _, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
                box_z = c[2].value_in_unit(openmmunit.nanometers)
                simulation.context.setParameter('box_z', box_z)
            if cv1 == "com_distance_along_vector" or cv2 == "proj_vec":
                a, b, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
                box_x = a[0].value_in_unit(unit.nanometer)  # x-length
                box_y = b[1].value_in_unit(unit.nanometer)  # y-length
                box_z = c[2].value_in_unit(unit.nanometer)  # z-length

                simulation.context.setParameter('box_x', box_x)
                simulation.context.setParameter('box_y', box_y)
                simulation.context.setParameter('box_z', box_z)

            current_cvs = meta.getCollectiveVariables(simulation)
            colvar_array = np.append(colvar_array, [current_cvs], axis=0)

        np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)
        np.save(os.path.join(self.out_dir, f"FE_{run_id}.npy"), meta.getFreeEnergy())

        # Create plots for all current runs
        plot_colvar_2D(self.out_dir, "com_z", "lm_cv")
        plot_FE_2D(
            self.out_dir,
            grid_min_A,
            grid_max_A,
            grid_A,
            cv1,
            grid_min_B,
            grid_max_B,
            grid_B,
            cv2,
        )

        # final_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions() #I added  enforcePeriodicBox=True
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_system(system, f"{self.out_dir}/system_mMD_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/mMD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/mMD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return run_id
