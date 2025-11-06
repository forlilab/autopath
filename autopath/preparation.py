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
from autopath.utils import *

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
            '/gpfs/home/abarkdull/Forli/Manually_Prepared_Dataset/LILAC_DB_Non_Redundant/4zdy/charmm_gui/heme_params.xml'
        ],
        lig_ff: str = "espaloma",
        allow_undefined_stereo: bool = True,
        hydrogenMass: float = 3,
        boxShape: str = "dodecahedron",
        padding: float = 1.0,
        num_solvent: int = None,
        ionicStrength: float = 0.0,
        is_membrane: bool = False,
        lipid_type: str = None,
        ligand_com_z: float = 2.5, #nm
        add_cylindrical_restraint: bool = False,
        dummy_atom_position: Optional[List[float]] = None,
        out_dir: str = ".",
    ) -> None:
        print(f'lig_ff: {lig_ff}')
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
        self.allow_undefined_stereo = allow_undefined_stereo

        self.hydrogenMass = (
    hydrogenMass * openmmunit.amu if hydrogenMass is not None else None
)
        self.boxShape = boxShape  # cube, dodecahedron

        if num_solvent is not None and padding is not None:
            logging.error(
                "The arguments 'num_solvent' and 'padding' are incompatible. Please specify only one."
            )
            exit(1)
        if padding is not None:
            self.padding = padding * openmmunit.nanometers
        self.num_solvent = num_solvent

        self.ionicStrength = ionicStrength * openmmunit.molar

        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        self.ligand_com_z = ligand_com_z
        self.add_cylindrical_restraint = add_cylindrical_restraint
        self.dummy_atom_position = dummy_atom_position
        self._available_lipids = [
            # Original lipids
            "POPC",
            "POPE",
            "DLPC",
            "DLPE",
            "DMPC",
            "DOPC",
            "DPPC",

            # Complex membrane descriptions, i made these with charmm gui, maybe a better way to handle this
            "120_lip_per_leaf_100_percent_POPC_20A_wat",
            "120_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
            "15A_wat",
            "50_lip_per_leaf_100_percent_POPC_20A_wat",
            "50_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
            "80_lip_per_leaf_100_percent_POPC_20A_wat",
            "80_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
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

    def run(self, 
            protein: str = None, 
            variants: dict = None, 
            ligands: Union[str, List[Tuple[str, str]]] = None
            ) -> tuple[System, Topology]:

        start_time = time.monotonic()

        if ligands is not None and protein is None:
            if isinstance(ligands, str):
                logging.info(f"Parametrizing ligand {os.path.basename(ligands)}..")
                lig = self._sdf_to_mol(ligands)
                ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                for res in ligand_topology.residues():
                    res.name ='UNK'
                modeller = Modeller(ligand_topology, ligand_positions)

            elif isinstance(ligands, list):
                # Start with an empty modeller (no protein, no ligand yet)
                modeller = Modeller(Topology(), [])
                used_chains = set()

                chain_id = ord('A')
                for lig_name, lig_path in ligands:
                    while chr(chain_id) in used_chains:
                        chain_id += 1
                    logging.info(f"Parametrizing ligand {lig_name}..")
                    lig = self._sdf_to_mol(lig_path)
                    ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                    for chain in ligand_topology.chains():
                        chain.id = chr(chain_id)
                    for res in ligand_topology.residues():
                        res.name = lig_name
                    modeller.add(ligand_topology, ligand_positions)
                    used_chains.add(chr(chain_id))
                    chain_id += 1

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
                    lig = self._sdf_to_mol(ligands)
                    ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                    for res in ligand_topology.residues():
                        res.name ='UNK'
                    modeller.add(ligand_topology, ligand_positions)

                elif isinstance(ligands, list):
                    used_chains = set(c.id for c in modeller.topology.chains()) if modeller else set()
                    chain_id = ord('A')
                    for lig_name, lig_path in ligands:
                        while chr(chain_id) in used_chains:
                            chain_id += 1
                        logging.info(f"Parametrizing ligand {lig_name}..")
                        lig = self._sdf_to_mol(lig_path)
                        ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                        for chain in ligand_topology.chains():
                            chain.id = chr(chain_id)
                        for res in ligand_topology.residues():
                            res.name = lig_name
                        modeller.add(ligand_topology, ligand_positions)
                        used_chains.add(chr(chain_id))
                        chain_id += 1

        # CASE: Ligand and membrane only
        if protein is None and self.is_membrane:

            # Center ligand at 0,0,0
            lig_com = np.mean(ligand_positions, axis=0)
            for i, xyz_i in enumerate(ligand_positions):
                ligand_positions[i] = xyz_i - lig_com

            lig_com = lig_com.value_in_unit(openmmunit.nanometer)


            # Calculate the maximum distance between any two atoms in the molecule
            # pairwise_distances = np.linalg.norm(
            #     ligand_positions[:, None] - ligand_positions, axis=2
            # )
            # max_length = np.max(pairwise_distances) * openmmunit.angstroms


            translation_vector = np.array([0, 0, self.ligand_com_z])

            # Apply translation to coordinates and update positions
            ligand_positions += translation_vector * openmmunit.nanometers

            # create a new modeller from the ligand structure
            modeller = Modeller(ligand_topology, ligand_positions)


        if self.is_membrane:

            
            complex_membranes = {
                    "120_lip_per_leaf_100_percent_POPC_20A_wat",
                    "120_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
                    "15A_wat",
                    "50_lip_per_leaf_100_percent_POPC_20A_wat",
                    "50_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
                    "80_lip_per_leaf_100_percent_POPC_20A_wat",
                    "80_lip_per_leaf_70_percent_POPC_30_percent_CLR_20A_wat",
                }
            try:
                if self.lipid_type in complex_membranes:
                    logging.info(f"Adding a {self.lipid_type} (complex lipid) membrane to the system..")
                    lipid_patch_path = f"/mnt/forli/group/abarkdull/Ligand_Bilayer_MD/lipid_patches/{self.lipid_type}/{self.lipid_type}.pdb"
                    if not os.path.isfile(lipid_patch_path):
                        raise FileNotFoundError(f"Lipid patch PDB file not found at: {lipid_patch_path}")

                    lipid_patch = PDBFile(lipid_patch_path)
                    modeller.addMembrane(
                        forcefield=self.forcefield,
                        lipidType=lipid_patch,
                        neutralize=True,
                        ionicStrength=self.ionicStrength,
                        minimumPadding=self.padding,
                    )               
                else:
                    logging.info(f"Adding a {self.lipid_type} membrane to the system..")
                    modeller.addMembrane(
                        forcefield=self.forcefield,
                        lipidType=self.lipid_type,
                        neutralize=True,
                        ionicStrength=self.ionicStrength,
                        minimumPadding=self.padding,
                    )
               
                # # Get the periodic box vectors
                # vectors = modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
                # print(vectors)

                # # Define the box vector with an increased z dimension
                # box_vector = (
                #     openmm.Vec3(vectors[0][0], 0, 0),
                #     openmm.Vec3(0, vectors[1][1], 0),
                #     openmm.Vec3(0, 0, vectors[2][2] * 1.5)
                # )
                # print(box_vector)
                
                # # Add solvent to the system
                # modeller.addSolvent(
                #         self.forcefield,
                #         neutralize=True,
                #         boxVectors=box_vector
                #     )

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
         # CASE: Ligand and membrane only
        if protein is None and self.is_membrane: #so better allison
            # # Add the dummy atom to the modeller
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
                print(
                    "creating template for residue",
                    residue.name,
                    "(MDSimulationProcess::172)",
                )
                template = templates[residue.name]
                forcefield.registerResidueTemplate(template)

            nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
            # Add a single dummy particle
            dummyIndex = system.addParticle(0)  # 0 mass
            nonbonded.addParticle(
                0, 0, 0
            )  # 0 charge, 0 sigma (VDWR), 0 epsilon (interaction strength)

        #translating the system up so that all coords are positve in the z-dimension to avoid some weird metadynamics behavior 
        # box_vectors = modeller.topology.getPeriodicBoxVectors()
        # dimensions = modeller.topology.getUnitCellDimensions()
        # z_dimension = dimensions[2]
        # z_dimension_value = z_dimension.value_in_unit(openmm.unit.nanometer)
        # half_z_nm = z_dimension_value/2
        # print(f'translating the system up by {half_z_nm}')
        # positions_np = modeller.positions.value_in_unit(unit.nanometer)
        # translated_positions_vec3 = [Vec3(pos[0], pos[1], pos[2] + half_z_nm) for pos in positions_np]
        # modeller.positions = translated_positions_vec3 * unit.nanometer
    
        if self.dummy_atom_position is not None:
            #add dummy atom
            scaled_position = [coord * 10 for coord in self.dummy_atom_position]  # multiply each by 10
            dummy_position_vec3 = Vec3(*scaled_position)

            # Ensure dummy_position is a Quantity with units
            dummy_position_quantity = unit.Quantity(dummy_position_vec3, unit.angstrom)
            nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
            positions = modeller.getPositions()
            positions.append(dummy_position_quantity)
            dummyIndex = system.addParticle(0.0) #0 mass
            nonbonded.addParticle(0.0, 1.0, 0.0) #0 charge, 0 sigma (VDWR), 0 epsilon (interaction strength)
            dummy_topology = Topology()
            dummy_chain = dummy_topology.addChain()
            dummy_residue = dummy_topology.addResidue('DUM', dummy_chain)
            dummy_atom = dummy_topology.addAtom('DUM', element.sodium, dummy_residue) # this has to be an ion otherwise i get an issue when tring to greate a prmtop file bc there is no typename that matches an unbonded carbon
            modeller.add(dummy_topology, [dummy_position_quantity])

            #pin the dummy atom in place
            dummy_position_nm = modeller.getPositions()[-1].value_in_unit(unit.nanometer)
            print(f'dummy_position_nm: {dummy_position_nm}')
            x0_dum, y0_dum, z0_dum = dummy_position_nm.x, dummy_position_nm.y, dummy_position_nm.z

            # Define a harmonic potential centered at the dummy's initial position
            pin_force = CustomExternalForce(
                # "100000000000.0 * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
                "100000000000.0 * periodicdistance(x, y, z, x0_dum, y0_dum, z0_dum)"
            )
            pin_force.addGlobalParameter("x0_dum", x0_dum)
            pin_force.addGlobalParameter("y0_dum", y0_dum)
            pin_force.addGlobalParameter("z0_dum", z0_dum)
            pin_force.addParticle(dummyIndex, [])  # Apply only to dummy
            system.addForce(pin_force)

        if self.add_cylindrical_restraint:
            print("Applying cylindrical restraint to ligand...")
            add_cylindrical_restraints(
                system=system,
                guest_index=[atom.index for atom in modeller.topology.atoms() if atom.residue.name == "UNK"],
                host_index=[atom.index for atom in modeller.topology.atoms() if atom.residue.name == "DUM"],
                k_xy=10 * openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2,
                R_cylinder=10 * openmmunit.angstrom #to do: pick this better
            )
               

        os.makedirs(self.out_dir, exist_ok=True)
        save_system(system, f"{self.out_dir}/system.xml")
        save_pdb(modeller.topology, modeller.positions, f"{self.out_dir}/system.pdb")
        save_amber_topology(
            modeller.topology, modeller.positions, self.forcefield, self.out_dir
        )

        simulation_time = time.monotonic() - start_time
        logging.info(f"Finished system preparation in {simulation_time:.2f} seconds.")

        return system, modeller.topology
