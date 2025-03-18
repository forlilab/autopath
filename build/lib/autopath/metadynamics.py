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

from rdkit.Chem import AllChem, rdMolDescriptors
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
        
        # Im not exposing all options here because I want to keep it simple
        self.restrained_atoms = restrained_atoms

        self.platform = select_platform("fastest")

        # These are for debugging purposes if one wants to check the CVs over the time of the simulation
        self.verbose = verbose
        self.record_CV = 2500  # record the CVs every 10 ps
        self.store_CV = 25000  # log the stored COLVAR every 100ps

        return

    def _get_reference_dict(
        self,
        pdb_file,
        atoms_names: list[str] = ["CA"],
        system_atom_indexes: list[int] = None,
    ) -> dict[int, tuple[float, float, float]]:
        """
        Get the reference dictionary for the RMSD CV
        """
        pdb = PDBFile(pdb_file)
        reference_positions = pdb.positions
        reference_atoms = pdb.topology.atoms()

        reference_dict = {}
        for atom in reference_atoms:
            if atom.name in atoms_names:
                reference_dict[atom.index] = (
                    reference_positions[atom.index] / openmmunit.nanometers
                )

        reference_dict = dict(zip(system_atom_indexes, list(reference_dict.values())))

        return reference_dict

    def run(
        self,
        pdb_file: str = None,
        system: str = None,
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
        ligand_sdf = None
    ) -> None:

        start_time = time.monotonic()

        assert mMD_CV in [
            "com",
            "rmsd",
            "rmsd_states",
            "nc",
            "min_dist",
            "dist",
            "z_distance",
            "LM_angle_with_z", "LM_z_component"
        ], f"The selected collective variable {mMD_CV} is not implemented"

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
                pdb = PDBFile(pdb_file)
                self.topology = pdb.topology

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.debug(f"Loading simulation checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            if pdb_file is not None:
                logging.debug(f"Setting positions from PDB file {pdb_file}")
                pdb = PDBFile(pdb_file)
                simulation.context.setPositions(pdb.positions)
            else:
                logging.error(
                    f"Either a PDB or a checkpoint file must be provided to get coordinates from"
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
                weighByMass=False,
                pbc=True,
            )

        elif mMD_CV == "z_distance":
            dummy_atom = [atom.index for atom in self.topology.atoms() if atom.residue.name == "DUM"]
            if not dummy_atom:
                raise ValueError("Could not find ligand (UNK) or dummy atom (DUM) in the topology.")

            groups = [self.ligand_atoms] + [dummy_atom]
            cv = cvpack.CentroidFunction(
                f"z1-z2",
                openmmunit.nanometers,
                groups,
                weighByMass=False,
                pbc=False,
            )
            # z_distance_force = openmm.CustomCentroidBondForce(2, "z1-z2")
            # z_distance_force.addGroup(self.ligand_atoms)  # COM of ligand
            # z_distance_force.addGroup(dummy_atom)  # Single dummy atom
            # z_distance_force.addBond([0, 1])


            # # Wrap the force as a collective variable
            # cv = cvpack.OpenMMForceWrapper(z_distance_force, unit.nanometer)

        elif mMD_CV == "LM_angle_with_z":
            
            #get crippen contribution list
            ligand_mol = SDMolSupplier(ligand_sdf, removeHs=False)[0]
            atom_contribs = rdMolDescriptors._CalcCrippenContribs(ligand_mol)
            clogp_contributions = [contrib[0] for contrib in atom_contribs]

            num_clogp_contributions = len(clogp_contributions)
            num_ligand_atoms = len(self.ligand_atoms)

            print(f"Number of clogP contributions: {num_clogp_contributions}")
            print(f"Number of ligand atoms: {len(self.ligand_atoms)}")
            # Get the centroid (x, y, z) of the molecule "
            
            for i in self.ligand_atoms:
                if i == 0:
                    centroid_x = f"(10 *x{i+1})"
                    centroid_y = f"(10 *y{i+1})"
                    centroid_z = f"(10 *z{i+1})"
  
                else:
                    centroid_x += f" + (10 *x{i+1})"
                    centroid_y += f" + (10 *y{i+1})"
                    centroid_z += f" + (10 *z{i+1})"

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
                    delta_lipophilicity_x = f"(((10 *x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y = f"(((10 *y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z = f"(((10 *z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"                 
                else:
                    delta_lipophilicity_x += f" + (((10 *x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y += f" + (((10 *y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z += f" + (((10 *z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"   


            #now we want the angle between (delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z) and the negative z-axis (0, 0, -1)
            #arccos((delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z) dot (0, 0, -1)/|(delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z)||(0, 0, -1)|)
            #arccos(0*delta_lipophilicity_x + 0*delta_lipophilicity_y+ -1*delta_lipophilicity_z) / ((sqrt(0^2+0^2+(-1)^2)) * sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2)))
            #arccos((-delta_lipophilicity_z) / (sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2))
            


            lm_num = f"-({delta_lipophilicity_z})"
            lm_denom = f"sqrt((({delta_lipophilicity_x})^2) + (({delta_lipophilicity_y})^2) + (({delta_lipophilicity_z})^2))"
            lipophilicity_moment = f"acos(({lm_num})/({lm_denom}))"
            print(f'lipophilicity_moment: {lipophilicity_moment}')



            cv = cvpack.AtomicFunction(lipophilicity_moment,
                                        openmmunit.radian,
                                        # openmmunit.nanometer,
                                    self.ligand_atoms)


        elif mMD_CV == "LM_z_component":
            
            #get crippen contribution list
            ligand_mol = SDMolSupplier(ligand_sdf, removeHs=False)[0]
            atom_contribs = rdMolDescriptors._CalcCrippenContribs(ligand_mol)
            clogp_contributions = [contrib[0] for contrib in atom_contribs]

            num_clogp_contributions = len(clogp_contributions)
            num_ligand_atoms = len(self.ligand_atoms)

            print(f"Number of clogP contributions: {num_clogp_contributions}")
            print(f"Number of ligand atoms: {len(self.ligand_atoms)}")
            # Get the centroid (x, y, z) of the molecule "
            
            for i in self.ligand_atoms:
                if i == 0:
                    centroid_x = f"(10 *x{i+1})"
                    centroid_y = f"(10 *y{i+1})"
                    centroid_z = f"(10 *z{i+1})"
  
                else:
                    centroid_x += f" + (10 *x{i+1})"
                    centroid_y += f" + (10 *y{i+1})"
                    centroid_z += f" + (10 *z{i+1})"

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
                    delta_lipophilicity_x = f"(((10 *x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y = f"(((10 *y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z = f"(((10 *z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"                 
                else:
                    delta_lipophilicity_x += f" + (((10 *x{i+1}) - ({centroid_x})) * {clogp_contributions[i]})"
                    delta_lipophilicity_y += f" + (((10 *y{i+1}) - ({centroid_y})) * {clogp_contributions[i]})"
                    delta_lipophilicity_z += f" + (((10 *z{i+1}) - ({centroid_z})) * {clogp_contributions[i]})"   


            #now we want the angle between (delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z) and the negative z-axis (0, 0, -1)
            #arccos((delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z) dot (0, 0, -1)/|(delta_lipophilicity_x, delta_lipophilicity_y, delta_lipophilicity_z)||(0, 0, -1)|)
            #arccos(0*delta_lipophilicity_x + 0*delta_lipophilicity_y+ -1*delta_lipophilicity_z) / ((sqrt(0^2+0^2+(-1)^2)) * sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2)))
            #arccos((-delta_lipophilicity_z) / (sqrt(delta_lipophilicity_x^2 + delta_lipophilicity_y^2 +delta_lipophilicity_z^2))
            


            lm_num = f"-({delta_lipophilicity_z})"
            lm_denom = f"sqrt((({delta_lipophilicity_x})^2) + (({delta_lipophilicity_y})^2) + (({delta_lipophilicity_z})^2))"
            lipophilicity_moment_z = f"({lm_num})/({lm_denom})"
            print(f'lipophilicity_moment z component: {lipophilicity_moment_z}')



            cv = cvpack.AtomicFunction(lipophilicity_moment_z,
                                        openmmunit.nanometer,
                                    self.ligand_atoms)



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
                print(f'current CV: {meta.getCollectiveVariables(simulation)}')
                if i % self.store_CV == 0:
                    np.save(
                        os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"),
                        colvar_array,
                    )

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
        # ref_ligand: str = None,
        system: str = None,
        checkpoint_file: str = None,
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
    ) -> None:

        start_time = time.monotonic()

        # Metadynamics time in ns
        mMD_steps = 250000 * mMD_time  # 250.000 1ns at 4fs
        total_steps = 25000 + mMD_steps
        bias_frequency = (
            250 * bias_frequency
        )  # deposit bias every 2 ps (250 is 1ns at 4fs timestep)
        saveFrequency = 250 * saveFrequency  # write bias every 50ps

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
                pdb = PDBFile(pdb_file)
                self.topology = pdb.topology

        logging.debug(f"Creating the simulation for {run_id}")
        simulation = Simulation(self.topology, system, integrator, self.platform)

        if checkpoint_file is not None:
            logging.debug(f"Loading simulation checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
        else:
            if pdb_file is not None:
                logging.debug(f"Setting positions from PDB file {pdb_file}")
                pdb = PDBFile(pdb_file)
                simulation.context.setPositions(pdb.positions)
            else:
                logging.error(
                    f"Either a PDB or a checkpoint file must be provided to get coordinates from"
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

        ##################### Number of contacts CV #################################

        # forces = {f.getName(): f for f in system.getForces()}
        # nc_cv = cvpack.NumberOfContacts(
        #     self.pocket_atoms,
        #     self.ligand_atoms,
        #     forces["NonbondedForce"],
        #     stepFunction="1/(1+x^6)",
        #     thresholdDistance=0.35,
        #     cutoffFactor=2.0,
        #     switchFactor=1.5,
        #     reference=50,
        # )

        # grid_width_A = hill_width_A / 5
        # grid_min_A, grid_max_A = grid_dimensions_A
        # grid_A = int(abs(grid_min_A - grid_max_A) / grid_width_A)
        # nc_variable = BiasVariable(
        #     nc_cv,
        #     minValue=grid_min_A,
        #     maxValue=grid_max_A,
        #     biasWidth=hill_width_A,
        #     periodic=False,
        #     gridWidth=grid_A,
        # )

        # logging.info(
        #     f"COM boundaries are min={grid_min_A:.3f} nM - max={grid_max_A:.3f} nM"
        # )
        # logging.info(f"Sigma is {hill_width_A} nm and there are {grid_A} grid points ")

        ##################### COM CV #################################

        # groups = [self.pocket_atoms] + [self.ligand_atoms]

        # fb_eq = f"sqrt(distance(g1,g2)^2)"

        # COM = cvpack.CentroidFunction(
        #     fb_eq, openmmunit.nanometers, groups, weighByMass=False, pbc=True
        # )

        # grid_width_A = hill_width_A / 5
        # grid_min_A, grid_max_A = grid_dimensions_A
        # grid_A = int(abs(grid_min_A - grid_max_A) / grid_width_A)

        # com_cv = BiasVariable(
        #     COM,
        #     minValue=grid_min_A,
        #     maxValue=grid_max_A,
        #     biasWidth=hill_width_A,
        #     periodic=False,
        #     gridWidth=grid_A,
        # )

        ##################### RMSD CV #################################

        atom_names_to_match = ["CA"]
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        n_atoms = self.topology.getNumAtoms()

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

        print(
            f"Matched {len(atom_indexes_to_match)} protein {atom_names_to_match} atoms from system"
        )

        reference_dict_6ydj = self._get_reference_dict(
            "cluster_4_idx_5_plddt_96_openmm_refinement_relaxed_wrt_6ydj_A_openmm_refinement.pdb",
            atom_names_to_match,
            atom_indexes_to_match,
        )
        reference_dict_6hdh = self._get_reference_dict(
            "cluster_7_idx_1_plddt_95_openmm_refinement_relaxed_wrt_6hdh_A_openmm_refinement.pdb",
            atom_names_to_match,
            atom_indexes_to_match,
        )

        print(f"Matched {len(reference_dict_6ydj)} heavy atoms from the reference 6ydj")
        print(f"Matched {len(reference_dict_6hdh)} heavy atoms from the reference 6hdh")

        rmsd_6ydj = cvpack.RMSD(
            referencePositions=reference_dict_6ydj,
            group=atom_indexes_to_match,
            numAtoms=n_atoms,
        )

        grid_width_A = hill_width_A / 5
        grid_min_A, grid_max_A = grid_dimensions_A
        grid_A = int(abs(grid_min_A - grid_max_A) / grid_width_A)

        rmsd_6ydj_cv = BiasVariable(
            rmsd_6ydj,
            minValue=grid_min_A,
            maxValue=grid_max_A,
            biasWidth=hill_width_A,
            periodic=False,
            gridWidth=grid_A,
        )

        rmsd_6hdh = cvpack.RMSD(
            referencePositions=reference_dict_6hdh,
            group=atom_indexes_to_match,
            numAtoms=n_atoms,
        )

        grid_width_B = hill_width_B / 5
        grid_min_B, grid_max_B = grid_dimensions_B
        grid_B = int(abs(grid_min_B - grid_max_B) / grid_width_B)

        rmsd_6hdh_cv = BiasVariable(
            rmsd_6hdh,
            minValue=grid_min_B,
            maxValue=grid_max_B,
            biasWidth=hill_width_B,
            periodic=False,
            gridWidth=grid_B,
        )

        ##################### RMSD STATES CV #################################

        # ref_pdb = PDBFile(ref_ligand)
        # reference_positions = ref_pdb.positions
        # reference_atoms = ref_pdb.topology.atoms()

        # reference_dict = {}
        # for atom in reference_atoms:
        #     if not atom.name.startswith("H"):
        #         reference_dict[atom.index] = (
        #             reference_positions[atom.index] / openmmunit.nanometers
        #         )

        # print(f"Matched {len(reference_dict)} heavy atoms from the reference")

        # n_atoms = self.topology.getNumAtoms()

        # system_residues = [r for r in self.topology.residues() if r.name == "UNK"]
        # logging.info([f"SYSTEM {r.name}_{r.index}" for r in system_residues])

        # system_atoms = []
        # for residue in system_residues:
        #     for atom in residue.atoms():
        #         if not atom.name.startswith("H"):
        #             system_atoms.append(atom.index)

        # logging.info(f"Matched {len(system_atoms)} heavy atoms from the system")

        # # changing keys of reference dictionary to match system's atom names
        # reference_dict = dict(zip(system_atoms, list(reference_dict.values())))

        # rmsd_cv = cvpack.PathInRMSDSpace(
        #     metric=cvpack.path.progress,
        #     milestones=[reference_dict],
        #     sigma=0.01 * openmmunit.nanometers,
        #     numAtoms=n_atoms,
        # )

        ##############################################################

        meta = Metadynamics(
            system,
            [rmsd_6ydj_cv, rmsd_6hdh_cv],
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

                meta.step(simulation, self.record_CV)
                current_cvs = meta.getCollectiveVariables(simulation)
                colvar_array = np.append(colvar_array, [current_cvs], axis=0)

        np.save(os.path.join(self.out_dir, f"COLVAR_{run_id}.npy"), colvar_array)
        np.save(os.path.join(self.out_dir, f"FE_{run_id}.npy"), meta.getFreeEnergy())

        # Create plots for all current runs
        plot_colvar_2D(self.out_dir, "RMSD_A", "RMSD_B")
        plot_FE_2D(
            self.out_dir,
            grid_min_A,
            grid_max_A,
            grid_A,
            "rmsd_6ydj",
            grid_min_B,
            grid_max_B,
            grid_B,
            "rmsd_6hdh",
        )

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_system(system, f"{self.out_dir}/system_mMD_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/mMD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/mMD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return
