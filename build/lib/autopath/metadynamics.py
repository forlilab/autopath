import os
import math
import time
import logging
import numpy as np

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import cvpack
from glob import glob
from autopath.utils import *
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
        HMR: bool = True,
        temp: float = 300,
        verbose: bool = True,
    ) -> None:

        self.timestep = 0.004 if HMR else 0.002
        self.temperature = temp * openmmunit.kelvin

        self.topology = topology

        self.is_membrane = is_membrane

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.ligand_atoms = ligand_atoms
        self.pocket_atoms = pocket_atoms
        
        self.restrained_atoms = restrained_atoms

        self.platform = select_platform("fastest")

        # These are for debugging purposes if one wants to check the CVs over the time of the simulation
        self.verbose = verbose
        self.record_CV = 2500  # record the CVs every 10 ps
        self.store_CV = 25000  # log the stored COLVAR every 100ps

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
        hill_height: float = 0.3,
        hill_width: float = 0.01,
        grid_dimensions: tuple = (0.0, 1.0),
        bias_frequency: int = 2,
        saveFrequency: int = 50,
    ) -> None:

        start_time = time.monotonic()

        assert mMD_CV in [
            "com",
            "com_z",
            "com_x_y",            
            "rmsd",
            "rmsd_states",
            "nc",
            "min_dist",
            "dist",
        ], f"The selected colective variable {mMD_CV} is not implemented"

        # Calculate the number of steps required
        mMD_steps = math.ceil(mMD_time / self.timestep * 1000.0)  # 250.000 1ns at 4fs
        total_steps = 25000 + mMD_steps
        bias_frequency = (
            250 * bias_frequency
        )  # deposit bias every 2 ps (250 is 1ps at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps

        hill_height = hill_height * openmmunit.kilocalories_per_mole

        grid_width = hill_width / 5  # also known as sigma
        grid_min, grid_max = grid_dimensions
        grid = int(abs(grid_min - grid_max) / grid_width)

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
        num_atoms = self.topology.getNumAtoms()

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
        if mMD_CV == "com_x_y":

            groups = [self.pocket_atoms] + [self.ligand_atoms]

            cv = cvpack.CentroidFunction(
                f"sqrt(pointdistance(x1,y1,0,x2,y2,0)^2)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
            )

        elif mMD_CV == "rmsd":
            cv = cvpack.RMSD(input_positions, self.ligand_atoms, num_atoms)

        elif mMD_CV == "rmsd_states":

            atom_names_to_match = ["CA"]
            system_residues = [
                r
                for r in self.topology.residues()
                if r.name not in ["UNK", "HOH", "NA", "CL"]
            ]

            atom_indexes_to_match = []
            for residue in system_residues:
                for atom in residue.atoms():
                    if atom.name in atom_names_to_match:
                        atom_indexes_to_match.append(atom.index)

            states_pdbs = glob("input/milestone_*.pdb")
            milestones_dicts = [
                self._get_reference_dict(
                    pdb, atom_names_to_match, atom_indexes_to_match
                )
                for pdb in states_pdbs
            ]

            cv = cvpack.PathInRMSDSpace(
                metric=cvpack.path.progress,
                milestones=milestones_dicts,
                sigma=0.01 * openmmunit.nanometers,
                numAtoms=num_atoms,
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

        elif mMD_CV == "min_dist":

            num_atoms = system.getNumParticles()
            cv = cvpack.ShortestDistance(
                self.pocket_atoms,
                self.ligand_atoms,
                num_atoms,
                cutoffDistance=0.5,
            )

        elif mMD_CV == "dist":
            cv = cvpack.Distance(
                self.pocket_atoms[0], self.ligand_atoms[0], pbc=False, name="distance"
            )

        bias_variable = BiasVariable(
            cv,
            minValue=grid_min,
            maxValue=grid_max,
            biasWidth=hill_width,
            periodic=False,
            gridWidth=grid,
        )

        # Set up the metadynamics object
        meta = Metadynamics(
            system,
            [bias_variable],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=bias_frequency,
            saveFrequency=saveFrequency,
            biasDir=self.out_dir,
        )

        simulation.context.reinitialize(preserveState=True)

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
        plot_bias(self.out_dir, grid_min, grid_max, grid, mMD_CV)
        plot_FE(self.out_dir, grid_min, grid_max, grid, mMD_CV)

        # Save everything
        final_positions = simulation.context.getState(getPositions=True).getPositions()

        save_system(system, f"{self.out_dir}/system_mMD_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/mMD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/mMD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return

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
        bias_frequency: int = 2,
        saveFrequency: int = 50,
        cv1: str = "com_z",
        cv2: str = "lmz",
        atom1: list = None, #these are the two atoms in the delta z calculation....make this more clear somehow?
        atom2: list = None, #these are the two atoms in the delta z calculation....make this more clear somehow?
        rmsd_1_reference_positions: list = None,
        rmsd_1_group: list = None,
        rmsd_2_reference_positions: list = None,
        rmsd_2_group: list = None,
    ) -> None:

        # Validate cv1
        if cv1 not in ["com_z", 'com', "rmsd"]:
            logging.error(f"Invalid value for cv1: '{cv2}'. Must be 'com_z', 'com' or 'rmsd'.")

        # Validate cv2
        if cv2 not in ["lmz", "dz", "rmsd", 'nc']:
            logging.error(f"Invalid value for cv2: '{cv2}'. Must be 'lmz', 'dz', 'rmsd', or  'nc'.")

        # Require atom1 and atom2 if using distance-based CV
            if cv2 == "dz":
                if not atom1 or not atom2:
                    raise ValueError("When cv2 is 'dz', atom1 and atom2 must be provided as lists of atom indices.")

        start_time = time.monotonic()

        # Metadynamics time in ns
        mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
        total_steps = 25000 + mMD_steps
        bias_frequency = (
            250 * bias_frequency
        )  # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps
        hill_height = hill_height * openmmunit.kilocalories_per_mole
        
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

        ##################### Z depth CV (cv1) #################################

        if cv1 == "com_z":
            dummy_atom = [atom.index for atom in self.topology.atoms() if atom.residue.name == "DUM"]
            if not dummy_atom:
                raise ValueError("Could not find ligand (UNK) or dummy atom (DUM) in the topology.")

            #add z size parameter
            _, _, c = simulation.context.getState(getPositions=False, getVelocities=False, getEnergy=False).getPeriodicBoxVectors()
            zsize = c[2].value_in_unit(openmmunit.nanometers)


            groups = [self.ligand_atoms] + [dummy_atom]
            print(f'groups: {groups}')
            COM_Z = cvpack.CentroidFunction(
                # f"z1-z2",
                # f"z1",
                "select(step((z1 - z2)/zsize - floor((z1 - z2)/zsize) - 0.5), -1, 1) * pointdistance(0, 0, z1, 0, 0, z2)",
                # "select(step((z1 - 0)/zsize - floor((z1 - 0)/zsize) - 0.5), -1, 1) * pointdistance(0, 0, z1, 0, 0, 0)",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=True,
                zsize = zsize
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
            rmsd_cv = cvpack.RMSD(
                referencePositions=rmsd_1_reference_positions,
                group=rmsd_1_group,
                numAtoms=num_atoms,  
                name='rmsd_1'
            )
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


        ##################### RMSD CV (cv2) #################################
        if cv2 == "rmsd":
            
            num_atoms = system.getNumParticles()

            # Instantiate the RMSD collective variable
            rmsd_cv = cvpack.RMSD(
                referencePositions=rmsd_2_reference_positions,
                group=rmsd_2_group,
                numAtoms=num_atoms,  
                name=name
            )
            cv2_BiasVariable = BiasVariable(
                rmsd_cv,
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
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
                minValue=grid_min_A,
                maxValue=grid_max_A,
                biasWidth=hill_width_A,
                periodic=False, 
                gridWidth=grid_A,
        )    

     ##############################################################
        meta = Metadynamics(
            system,
            [cv1_BiasVariable, cv2_BiasVariable],
            self.temperature,
            bias_factor,
            hill_height,
            frequency=bias_frequency,
            saveFrequency=saveFrequency,
            biasDir=self.out_dir,
        )

        simulation.context.reinitialize(preserveState=True)
        
        logging.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"metadynamics_{run_id}",
            total_steps,
            bias_frequency,
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
                zsize = c[2].value_in_unit(openmmunit.nanometers)
                simulation.context.setParameter('zsize', zsize)
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
        save_system(system, f"{self.out_dir}/system_mMD_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/mMD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/mMD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return
