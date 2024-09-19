# general imports
import os
import time
import logging
import numpy as np
from sys import exit

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
from rdkit.Chem import SDMolSupplier

# AutoPath imports
from autopath.utils import add_variants, save_pdb, save_system, save_amber_topology


class SystemPreparation:
    def __init__(
        self,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3p_HFE_multivalent.xml",
        ],
        lig_ff: str = "espaloma",
        allow_undefined_stereo: bool = True,
        hydrogenMass: float = 3,
        boxShape: str = "dodecahedron",
        padding: float = 1.0,
        ionicStrength: float = 0.0,
        is_membrane: bool = False,
        lipid_type: str = None,
    ) -> None:

        if lig_ff.upper() in ["ESPALOMA", "SMIRNOFF", "GAFF"]:
            self.lig_ff = lig_ff.upper()
        else:
            logging.error(
                f"Ligand forcefield must be one of Espaloma, SMIRNOFF or GAFF"
            )
            exit(1)

        self.forcefield = ForceField(*forcefield)
        self.allow_undefined_stereo = allow_undefined_stereo

        self.hydrogenMass = hydrogenMass * openmmunit.amu  # Use HMR
        self.boxShape = boxShape  # cube, dodecahedron
        self.padding = padding * openmmunit.nanometers
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

    def _sdf_to_mol(self, lig_sdf: str = None):
        """Load ligand SDF and transform to OpenMM molecule"""
        try:
            rdkit_mol = SDMolSupplier(lig_sdf)[0]
        except Exception as e:
            logging.error(f"Something went wrong loading {lig_sdf}..\n{e}")
            raise

        # Convert to OpenMM molecule
        ligand = Molecule.from_rdkit(rdkit_mol, self.allow_undefined_stereo)
        return ligand

    def _parametrize_ligand(self, ligand):

        if self.lig_ff == "ESPALOMA":
            template_generator = EspalomaTemplateGenerator(
                molecules=ligand, forcefield="espaloma-0.3.2"
            )
        elif self.lig_ff == "SMIRNOFF":
            template_generator = SMIRNOFFTemplateGenerator(
                molecules=ligand, forcefield="openff-1.2.0"
            )
        elif self.lig_ff == "GAFF":
            template_generator = GAFFTemplateGenerator(
                molecules=ligand, forcefield="gaff-2.11"
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

    def run(
        self, prot_path: str = None, variants: dict = None, lig_path: str = None
    ) -> tuple[System, Topology]:

        start_time = time.monotonic()

        if lig_path is not None:
            lig_name = os.path.splitext(os.path.basename(lig_path))[0]
            out_dir = lig_name

            logging.info(f"Parametrizing ligand {lig_name}..")

            lig = self._sdf_to_mol(lig_path)
            ligand_topology, ligand_positions = self._parametrize_ligand(lig)

            modeller = Modeller(ligand_topology, ligand_positions)

        if prot_path is not None:
            rec_name = os.path.splitext(os.path.basename(prot_path))[0]
            if lig_path is None:
                out_dir = rec_name

            try:
                protein_pdb = PDBFile(prot_path)
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
            if lig_path is not None:
                modeller.add(ligand_topology, ligand_positions)

        # CASE: Ligand and membrane only
        if prot_path is None and self.is_membrane:

            # Center ligand at 0,0,0
            lig_com = np.mean(ligand_positions, axis=0)
            for i, xyz_i in enumerate(ligand_positions):
                ligand_positions[i] = xyz_i - lig_com

            lig_com = lig_com.value_in_unit(openmmunit.nanometer)

            dummy_position = openmm.Vec3(
                lig_com[0],
                lig_com[1],
                0,
            )

            # Ensure dummy_position is a Quantity with units
            dummy_position_quantity = openmmunit.Quantity(
                dummy_position, openmmunit.angstrom
            )

            # Create a new topology for the dummy atom
            dummy_topology = app.Topology()
            dummy_chain = dummy_topology.addChain()
            dummy_residue = dummy_topology.addResidue("DUM", dummy_chain)
            dummy_atom = dummy_topology.addAtom("C", element.carbon, dummy_residue)

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
               
                # Get the periodic box vectors
                vectors = modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
                print(vectors)

                # Define the box vector with an increased z dimension
                box_vector = (
                    openmm.Vec3(vectors[0][0], 0, 0),
                    openmm.Vec3(0, vectors[1][1], 0),
                    openmm.Vec3(0, 0, vectors[2][2] * 1.5)
                )
                print(box_vector)
                
                # Add solvent to the system
                modeller.addSolvent(
                        self.forcefield,
                        neutralize=True,
                        boxVectors=box_vector
                    )

            except OpenMMException as e:
                logging.error(f"Something went wrong while building the membrane.\n{e}")
                exit(1)

        else:
            logging.info(f"Solvating the system..")
            modeller.addSolvent(
                self.forcefield,
                neutralize=True,
                ionicStrength=self.ionicStrength,
                boxShape=self.boxShape,
                padding=self.padding,
            )

        logging.info(f"Creating an OpenMM system..")
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
       
        # Add the dummy atom to the modeller
        modeller.add(dummy_topology, [dummy_position_quantity])

        unmatched_residues = self.forcefield.getUnmatchedResidues(modeller.topology)
        print(
            f"unmatched residues:{[unmatched_residue.name for unmatched_residue in unmatched_residues]}"
        )
        [templates, residues] = self.forcefield.generateTemplatesForUnmatchedResidues(
            modeller.topology
        )

        # reduce residues to uniquely named
        residues = list(dict.values({r.name: r for r in residues}))
        templates = {t.name: t for t in templates}
        for residue in residues:
            print(f"creating template for residue {residue.name}")
            template = templates[residue.name]
            for atom in template.atoms:
                atom.type = "protein-C"
            self.forcefield.registerResidueTemplate(template)

        unmatched_residues = self.forcefield.getUnmatchedResidues(modeller.topology)
        print(
            f"unmatched residues:{[unmatched_residue.name for unmatched_residue in unmatched_residues]}"
        )

        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        # Add a single dummy particle
        dummyIndex = system.addParticle(0)  # 0 mass
        nonbonded.addParticle(
            0, 0, 0
        )  # 0 charge, 0 sigma (VDWR), 0 epsilon (interaction strength)

        os.makedirs(out_dir, exist_ok=True)
        save_system(system, f"{out_dir}/system.xml")
        save_pdb(modeller.topology, modeller.positions, f"{out_dir}/system.pdb")
        # save_amber_topology(
        #     modeller.topology, modeller.positions, self.forcefield, out_dir
        # )

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished system preparation in {simulation_time:.2f} seconds.")

        return system, modeller.topology
