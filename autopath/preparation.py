# general imports
import os
import time
import logging
import numpy as np
from sys import exit
from typing import Union, List

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# OpenFF-toolkit imports
from openff.toolkit import Molecule
from openff.toolkit import Topology as offTopology
from openff.units.openmm import to_openmm as offquantity_to_openmm
from openmmforcefields.generators import (
    EspalomaTemplateGenerator,
    SMIRNOFFTemplateGenerator,
    GAFFTemplateGenerator,
)

# RDKit imports
from rdkit import Chem

# AutoPath imports
from autopath.utils import assign_bondOrders, add_variants, save_pdb, save_system, save_amber_files


class SystemPreparation:
    def __init__(
        self,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3p_HFE_multivalent.xml",
        ],
        lig_ff: str = "espaloma",
        hydrogenMass: float = 1.5, # # in amu, 1.5 is the default in OpenMM
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        num_solvent: int = None,
        ionicStrength: float = 0.15,
        is_membrane: bool = False,
        lipid_type: str = None,
        out_dir: str = "system_preparation",
    ) -> None:

        if lig_ff.upper() in ["ESPALOMA", "SMIRNOFF", "GAFF"]:
            self.lig_ff = lig_ff.upper()
        else:
            logging.error(
                f"Ligand forcefield must be one of Espaloma, SMIRNOFF or GAFF"
            )
            exit(1)

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.forcefield = ForceField(*forcefield)

        self.hydrogenMass = (hydrogenMass * openmmunit.amu if hydrogenMass is not None else None)
        self.boxShape = boxShape  # cube, dodecahedron

        self.padding = padding
        self.num_solvent = num_solvent
        if self.padding is not None:
            self.padding = self.padding * openmmunit.nanometers
            if num_solvent is not None:
                logging.warning("Both 'num_solvent' and 'padding' were specified. 'padding' will be ignored.")
                self.num_solvent = num_solvent
                self.padding = None
        elif self.num_solvent is not None:
            self.num_solvent = num_solvent
            self.padding = None
        else:
            logging.error("Either 'num_solvent' or 'padding' must be specified.")
            exit(1)
            
        self.ionicStrength = ionicStrength * openmmunit.molar

        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        self._available_lipids = [
            "POPC",
            "POPE",
            "DLPC",
            "DLPE",
            "DMPC",
            "DOPC",
            "DPPC",
        ]

        if is_membrane and self.lipid_type is not None:
            assert self.lipid_type in self._available_lipids, logging.error(
                f"{self.lipid_type} lipid is not supported. Available lipids are:\n\t{self._available_lipids}"
            )
        elif is_membrane and self.lipid_type is None:
            logging.error(
                f"For building a membrane system a lipid type must be specified"
            )
            exit(1)

        # you proabably dont want to change this
        self.nb_cutoff = 1.0 * openmmunit.nanometers
        self.switchDistance = 0.9 * openmmunit.nanometers

    def _ligand_to_mol(self, lig_fname: str = None, lig_smiles: str = None):
        """Load ligand SDF/PDB and transform to OpenMM molecule"""

        try:
            if lig_fname.endswith(".pdb"):
                rdkit_mol = Chem.MolFromPDBFile(lig_fname, removeHs=False)
            elif lig_fname.endswith(".sdf") or lig_fname.endswith(".mol2"): # SDMolSupplier also works for mol2 files
                rdkit_mol = Chem.SDMolSupplier(lig_fname, removeHs=False)[0]
            else:
                logging.error(f"Ligand file format not recognized. Please provide a .sdf or .pdb file.")
                exit(1)
        except Exception as e:
            logging.error(f"Something went wrong loading {lig_fname}..\n{e}")
            exit(1)
            
        # assign bond orders from SMILES if provided
        if lig_smiles is not None:
            rdkit_mol = assign_bondOrders(rdkit_mol, lig_smiles)
            # save the fixed ligand
            fixed_ligfname = os.path.join(self.out_dir, os.path.basename(lig_fname), "_fixed.sdf")
            writer = Chem.SDWriter(fixed_ligfname)
            for cid in range(rdkit_mol.GetNumConformers()):
                writer.write(rdkit_mol, confId=-1)
                    
        # Convert to OpenMM molecule
        ligand = Molecule.from_rdkit(rdkit_mol, True)

        return ligand

    def _parametrize_ligand(self, ligand):

        if self.lig_ff == "ESPALOMA":
            template_generator = EspalomaTemplateGenerator(
                molecules=ligand, 
                # template_generator_kwargs = {"reference_forcefield": "openff_unconstrained-2.2.1"}
                # forcefield="espaloma-0.3.2"
            )
        elif self.lig_ff == "SMIRNOFF":
            template_generator = SMIRNOFFTemplateGenerator(
                molecules=ligand, 
                # forcefield="openff-1.2.0"
            )
        elif self.lig_ff == "GAFF":
            template_generator = GAFFTemplateGenerator(
                molecules=ligand,
            )

        # add the template generator to the ff
        self.forcefield.registerTemplateGenerator(template_generator.generator)

        # make an OpenFF Topology of the ligand
        ligand_off_topology = offTopology.from_molecules(molecules=[ligand])

        # convert it to an OpenMM Topology
        ligand_omm_topology = ligand_off_topology.to_openmm()

        # get the positions of the ligand
        ligand_positions = offquantity_to_openmm(ligand.conformers[0])

        return ligand_omm_topology, ligand_positions

    def run(self, 
            protein: str = None, 
            variants: dict = None, 
            ligands: Union[str, List[tuple[str, str, str]]] = None
            ) -> tuple[System, Topology]:

        start_time = time.monotonic()

        # if ligands is not None and protein is None:
        #     if isinstance(ligands, str):
        #         logging.info(f"Parametrizing ligand {os.path.basename(ligands)}..")
        #         lig = self._ligand_to_mol(ligands)
        #         ligand_topology, ligand_positions = self._parametrize_ligand(lig)
        #         for res in ligand_topology.residues():
        #             res.name ='UNK'
        #         modeller = Modeller(ligand_topology, ligand_positions)

        #     elif isinstance(ligands, dict):
        #         for lig_name, lig_path in ligands.items():
        #             logging.info(f"Parametrizing ligand {lig_name}..")
        #             lig = self._ligand_to_mol(lig_path)
        #             ligand_topology, ligand_positions = self._parametrize_ligand(lig)
        #             for res in ligand_topology.residues():
        #                 res.name = lig_name
        #             modeller = Modeller(ligand_topology, ligand_positions)

        if protein is not None:
            rec_name = os.path.splitext(os.path.basename(protein))[0]
            try:
                protein_pdb = PDBFile(protein)
                logging.info(f"Loaded {rec_name} PDB..")
            except Exception as e:
                logging.error(f"Something went wrong loading {rec_name} PDB..\n{e}")
                raise

            # make an OpenMM Modeller object with the protein
            modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

            # Add residue variants, like protonation states for HIS, CYS, etc.
            if variants is not None:
                modeller = add_variants(modeller, variants)

            # Add the ligand to the Modeller built from the protein structure
            if ligands is not None:
                if isinstance(ligands, str):
                    logging.info(f"Parametrizing ligand {os.path.basename(ligands)}..")
                    lig = self._ligand_to_mol(ligands)
                    ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                    for res in ligand_topology.residues():
                        res.name ='UNK'
                    modeller.add(ligand_topology, ligand_positions)

                elif isinstance(ligands, list):
                    used_chains = set(c.id for c in modeller.topology.chains()) if modeller else set()
                    chain_id = ord('A')
                    for lig_name, lig_path, lig_smiles in ligands:
                        while chr(chain_id) in used_chains:
                            chain_id += 1
                        logging.info(f"Parametrizing ligand {lig_name}..")
                        lig = self._ligand_to_mol(lig_path, lig_smiles)
                        ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                        for chain in ligand_topology.chains():
                            chain.id = chr(chain_id)
                        for res in ligand_topology.residues():
                            res.name = lig_name
                        modeller.add(ligand_topology, ligand_positions)
                        used_chains.add(chr(chain_id))
                        chain_id += 1

        # CASE: Ligand and membrane only
        max_length = 0.0 * openmmunit.angstroms
        if protein is None and self.is_membrane:

            # Center ligand at 0,0,0
            lig_com = np.mean(ligand_positions, axis=0)
            for i, xyz_i in enumerate(ligand_positions):
                ligand_positions[i] = xyz_i - lig_com

            # Calculate the maximum distance between any two atoms in the molecule
            pairwise_distances = np.linalg.norm(
                ligand_positions[:, None] - ligand_positions, axis=2
            )
            max_length = np.max(pairwise_distances) * openmmunit.angstroms

            translation_distance = 3.0

            translation_vector = np.array([0, 0, translation_distance])

            # Apply translation to coordinates and update positions
            ligand_positions += translation_vector * openmmunit.nanometers

            # create a new modeller from the ligand structure
            modeller = Modeller(ligand_topology, ligand_positions)

        if self.is_membrane:

            logging.info(f"Adding a {self.lipid_type} membrane to the system..")
            try:
                modeller.addMembrane(
                    forcefield=self.forcefield,
                    lipidType=self.lipid_type,
                    neutralize=True,
                    ionicStrength=self.ionicStrength,
                    minimumPadding=self.padding + max_length,
                )

            except OpenMMException as e:
                logging.error(f"Something went wrong while building the membrane.\n{e}")
                exit(1)

        else:
            logging.info(f"Solvating the system..")
            modeller.addSolvent(
                self.forcefield,
                neutralize=True,
                numAdded=self.num_solvent,
                ionicStrength=self.ionicStrength,
                boxShape=self.boxShape,
                padding=self.padding,
            )

        logging.info(f"Creating the an OpenMM system..")
        system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nb_cutoff,
            switchDistance=self.switchDistance,
            removeCMMotion=True,
            rigidWater=True,
            hydrogenMass=self.hydrogenMass,
            constraints=HBonds,
        )

        save_system(system, f"{self.out_dir}/system.xml")
        save_pdb(modeller.topology, modeller.positions, f"{self.out_dir}/system.pdb")
        # This is why: https://parmed.github.io/ParmEd/html/openmm.html
        openmm_system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nb_cutoff,
            switchDistance=self.switchDistance,
            removeCMMotion=True,
            rigidWater=False, # DO NOT USE THIS, it will not work with parmed
            hydrogenMass=self.hydrogenMass,
            # constraints=app.HBonds, # DO NOT USE THIS, it will not work with parmed
        )

        save_amber_files(modeller.topology, modeller.positions, openmm_system, self.out_dir)

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished system preparation in {simulation_time:.2f} seconds.")

        return system, modeller.topology
