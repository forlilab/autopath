import os
import math
import time
import logging
import numpy as np

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import cvpack

from autopath.utils import *
from autopath.analysis import plot_bias, plot_colvar, plot_FE, plot_FE_2D, plot_colvar_2D

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
                logging.error(f"Either a PDB or a prmtop file must be provided to get the topology from")
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
                simulation.context.setPositions(pdb.positions)
            else:
                logging.error(f"Either a PDB or a checkpoint file must be provided to get coordinates from")
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
            
        logging.debug(f"Setting up reporters for {run_id}..")
        add_reporters(
            simulation,
            self.out_dir,
            f"metadynamics_{run_id}",
            total_steps,
            bias_frequency,
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

        elif mMD_CV == "rmsd":

            input_positions = simulation.context.getState(
                getPositions=True
            ).getPositions()
            num_atoms = self.topology.getNumAtoms()
            cv = cvpack.RMSD(input_positions, self.ligand_atoms, num_atoms)

        elif mMD_CV == "rmsd_states":

            model_pdb = PDBFile(f"2hu4/system.pdb")
            reference_positions = model_pdb.positions
            reference_residues = model_pdb.topology.residues()
            ref_residues = [r for r in reference_residues if r.name == "UNK"]
            logging.info([f"REFERENCE {r.name}_{r.index}" for r in ref_residues])

            reference_dict = {}
            for residue in ref_residues:
                for atom in residue.atoms():
                    if not atom.name.startswith("H"):
                        reference_dict[atom.index] = (
                            reference_positions[atom.index] / openmmunit.nanometers
                        )

            logging.info(
                f"Matched {len(reference_dict)} heavy atoms from the reference"
            )

            n_atoms = self.topology.getNumAtoms()

            system_residues = [r for r in self.topology.residues() if r.name == "UNK"]
            logging.info([f"SYSTEM {r.name}_{r.index}" for r in system_residues])

            system_atoms = []
            for residue in system_residues:
                for atom in residue.atoms():
                    if not atom.name.startswith("H"):
                        system_atoms.append(atom.index)

            logging.info(f"Matched {len(system_atoms)} heavy atoms from the system")

            # changing keys of reference dictionary to match system's atom names
            reference_dict = dict(zip(system_atoms, list(reference_dict.values())))

            cv = cvpack.PathInRMSDSpace(
                metric=cvpack.path.progress,
                milestones=[reference_dict, reference_dict],
                sigma=0.01 * openmmunit.nanometers,
                numAtoms=n_atoms,
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
        ref_ligand: str = None,
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

        groups = [self.pocket_atoms] + [self.ligand_atoms]

        fb_eq = f"sqrt(distance(g1,g2)^2)"

        COM = cvpack.CentroidFunction(
            fb_eq, openmmunit.nanometers, groups, weighByMass=False, pbc=True
        )

        grid_width_A = hill_width_A / 5
        grid_min_A, grid_max_A = grid_dimensions_A
        grid_A = int(abs(grid_min_A - grid_max_A) / grid_width_A)

        com_cv = BiasVariable(
            COM,
            minValue=grid_min_A,
            maxValue=grid_max_A,
            biasWidth=hill_width_A,
            periodic=False,
            gridWidth=grid_A,
        )

        ##################### RMSD CV #################################
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        n_atoms = self.topology.getNumAtoms()

        ref_pdb = PDBFile(ref_ligand)
        reference_positions = ref_pdb.positions
        reference_atoms = ref_pdb.topology.atoms()

        reference_dict_B = {}
        reference_dict = {}
        for atom in reference_atoms:
            if not atom.name.startswith("H"):
                reference_dict[atom.index] = (
                    reference_positions[atom.index] / openmmunit.nanometers
                )
                reference_dict_B[atom.index] = (atom.name, reference_positions[atom.index])
        print(f"Matched {len(reference_dict)} heavy atoms from the reference")
        
        print('reference_dict')
        print(reference_dict_B)

        system_residues = [r for r in self.topology.residues() if r.name == "UNK"]
        print([f"SYSTEM {r.name}_{r.index}" for r in system_residues])

        system_atoms = []
        system_dict_B = {}
        for residue in system_residues:
            for atom in residue.atoms():
                if not atom.name.startswith("H"):
                    system_atoms.append(atom.index)
                    system_dict_B[atom.index] = (atom.name, input_positions[atom.index])
        print(f"Matched {len(system_atoms)} heavy atoms from the system")

        print('system_dict')
        print(system_dict_B)

        # # changing keys of reference dictionary to match system's atom names
        reference_dict = dict(zip(system_atoms, list(reference_dict.values())))

        print('system_dict')
        print(reference_dict)

        rmsd = cvpack.RMSD(input_positions, self.ligand_atoms, n_atoms)

        grid_width_B = hill_width_B / 5
        grid_min_B, grid_max_B = grid_dimensions_B
        grid_B = int(abs(grid_min_B - grid_max_B) / grid_width_B)

        rmsd_cv = BiasVariable(
            rmsd,
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
            [com_cv, rmsd_cv],
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
        plot_colvar_2D(self.out_dir, "COM", "RMSD")
        plot_FE_2D(
            self.out_dir,
            grid_min_A,
            grid_max_A,
            grid_A,
            "COM",
            grid_min_B,
            grid_max_B,
            grid_B,
            "RMSD",
        )

        final_positions = simulation.context.getState(getPositions=True).getPositions()
        save_system(system, f"{self.out_dir}/system_mMD_{run_id}.xml")
        save_simulation(simulation, f"{self.out_dir}/mMD_checkpoint_{run_id}")
        save_pdb(self.topology, final_positions, f"{self.out_dir}/mMD_{run_id}.pdb")

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished {run_id} metadynamics in {simulation_time/60:.2f} min.")

        return