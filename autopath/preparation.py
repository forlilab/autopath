# general imports
import os
import time
import numpy as np
from typing import Union, List

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# OpenFF-toolkit imports
from openff import toolkit
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

import logging
logger = logging.getLogger("autopath.preparation")

class SystemPreparation:
    """Prepare a solvated or membrane-embedded simulation system for OpenMM MD.

    Orchestrates ligand parametrization (via OpenFF, GAFF, or Espaloma),
    protein loading, residue-variant assignment, solvation or membrane
    embedding, and OpenMM system creation. Writes a ``system.xml``,
    ``system.pdb``, and AMBER-compatible topology/coordinate files to
    ``out_dir``.

    Parameters
    ----------
    forcefield : list of str, optional
        OpenMM ForceField XML files to load. Defaults to AMBER14 with
        TIP3P-FB water.
    lig_ff : str, optional
        Ligand force field including version, e.g. ``"openff-2.3.0"``,
        ``"espaloma-0.3.2"``, ``"gaff-2.11"``. The family prefix selects
        the openmmforcefields template generator.
    hydrogenMass : float, optional
        Hydrogen mass in amu for mass repartitioning. 1.5 amu allows a
        4 fs timestep. Set to ``None`` to disable.
    boxShape : str, optional
        Solvent box shape: ``"cube"`` or ``"dodecahedron"``.
    padding : float, optional
        Minimum distance in nm between the solute and the box edge.
        Mutually exclusive with ``num_solvent``.
    num_solvent : int, optional
        Explicit number of solvent molecules to add. Mutually exclusive
        with ``padding``.
    ionicStrength : float, optional
        Salt concentration in molar.
    ions : tuple of str, optional
        ``(positiveIon, negativeIon)`` residue names, e.g.
        ``("Na+", "Cl-")``.
    is_membrane : bool, optional
        When True, add a lipid bilayer instead of bulk solvent.
    lipid_type : str, optional
        Lipid residue name (e.g. ``"POPC"``) or path to a custom PDB patch.
        Required when ``is_membrane=True``.
    out_dir : str, optional
        Directory where all output files are written.
    """

    def __init__(
        self,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3pfb_HFE_multivalent.xml",
        ],
        lig_ff: str = "openff-2.3.0",
        hydrogenMass: float = 1.5,  # in amu; 1.5 enables 4 fs timestep via HMR
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        # addSolvent previously used its default (tip3p), so a 4-site water model could not be
        # requested: the forcefield would carry e.g. amber19/opc.xml while the solvent added
        # was 3-site, and createSystem then has no template for it. OPC has the same topology
        # as TIP4P-Ew (O, H1, H2, M + one virtual site), so pass water_model="tip4pew" to
        # build the right topology and let opc.xml supply the parameters. ff19SB is
        # parameterised for OPC, so the two go together.
        water_model: str = "tip3p",
        num_solvent: int = None,
        ionicStrength: float = 0.15,
        ions: tuple[str] = ("Na+", "Cl-"),  # positiveIon, negativeIon
        is_membrane: bool = False,
        lipid_type: str = 'POPC',
        out_dir: str = "system",
    ) -> None:

        self.lig_ff, self._lig_ff_family = self._parse_lig_ff(lig_ff)

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.forcefield = ForceField(*forcefield)

        self.hydrogenMass = (hydrogenMass * openmmunit.amu if hydrogenMass is not None else None)
        self.boxShape = boxShape  # cube, dodecahedron

        self.padding = padding
        self.water_model = water_model
        self.num_solvent = num_solvent
        if self.padding is not None:
            self.padding = self.padding * openmmunit.nanometers
            if num_solvent is not None:
                logger.warning("Both 'num_solvent' and 'padding' were specified. 'padding' will be ignored.")
                self.num_solvent = num_solvent
                self.padding = None
        elif self.num_solvent is not None:
            self.num_solvent = num_solvent
            self.padding = None
        else:
            raise ValueError("Either 'num_solvent' or 'padding' must be specified.")

        self.ionicStrength = ionicStrength * openmmunit.molar
        self.ions = ions  # positiveIon, negativeIon

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
            if not os.path.exists(self.lipid_type):
                if self.lipid_type not in self._available_lipids:
                    raise ValueError(
                        f"{self.lipid_type!r} lipid is not supported. "
                        f"Available lipids are: {self._available_lipids}"
                    )
            else:
                logger.info(f"Using custom lipid patch from {self.lipid_type}")
        elif is_membrane and self.lipid_type is None:
            raise ValueError("For building a membrane system a lipid type must be specified.")

        # Non-bonded parameters — probably don't want to change these defaults.
        self.nb_cutoff = 1.0 * openmmunit.nanometers
        self.switchDistance = 0.9 * openmmunit.nanometers

    @staticmethod
    def _parse_lig_ff(lig_ff: str) -> tuple[str, str]:
        """Parse ``lig_ff`` into ``(full_name, family)``.

        Accepts strings of the form ``<family>-<version>`` (e.g.
        ``openff-2.3.0``, ``espaloma-0.3.2``, ``gaff-2.11``). The family
        prefix selects the openmmforcefields template generator; the full
        name (including version) is passed through to that generator.

        Parameters
        ----------
        lig_ff : str
            Ligand force field string including version suffix.

        Returns
        -------
        lig_ff : str
            The original string, passed through unchanged.
        family : str
            Normalised family key: ``"OPENFF"``, ``"ESPALOMA"``, or
            ``"GAFF"``.

        Raises
        ------
        ValueError
            If ``lig_ff`` does not include a version (no ``"-"``) or if
            the family prefix is not recognised.
        """
        family_aliases = {
            "OPENFF": "OPENFF",
            "SMIRNOFF": "OPENFF",
            "ESPALOMA": "ESPALOMA",
            "GAFF": "GAFF",
        }
        if not isinstance(lig_ff, str) or "-" not in lig_ff:
            raise ValueError(
                f"Ligand forcefield must include a version, e.g. 'openff-2.3.0', "
                f"'espaloma-0.3.2', 'gaff-2.11'. Got: {lig_ff!r}"
            )
        family_prefix = lig_ff.split("-", 1)[0].upper()
        if family_prefix not in family_aliases:
            raise ValueError(
                f"Unknown ligand forcefield family {family_prefix!r}. "
                f"Supported: openff-*, espaloma-*, gaff-*."
            )
        return lig_ff, family_aliases[family_prefix]

    def _ligand_to_mol(self, lig_fname: str = None, lig_smiles: str = None, lig_from_xray: bool = False):
        """Load a ligand file and return an OpenFF ``Molecule``.

        Parameters
        ----------
        lig_fname : str
            Path to the ligand file (.sdf, .mol2, or .pdb).
        lig_smiles : str, optional
            Reference SMILES string used to assign correct bond orders after
            reading the file. Required when ``lig_from_xray=True`` or when
            the file lacks bond-order information.
        lig_from_xray : bool, optional
            When True, all bonds in the RDKit molecule are reset to single
            before SMILES-based bond-order assignment. This avoids
            kekulization errors common in ligands extracted from X-ray PDBs.

        Returns
        -------
        openff.toolkit.Molecule
            OpenFF molecule with 3-D coordinates.

        Raises
        ------
        RuntimeError
            If the file cannot be loaded or the format is not recognised.
        """

        if lig_smiles is not None:
            sanitize_mol_upon_reading = False
            removeHs_upon_reading = True
        else:
            sanitize_mol_upon_reading = True
            removeHs_upon_reading = False

        try:
            if lig_fname.endswith(".pdb"):
                rdkit_mol = Chem.MolFromPDBFile(lig_fname, sanitize=sanitize_mol_upon_reading, removeHs=removeHs_upon_reading)
            elif lig_fname.endswith(".sdf") or lig_fname.endswith(".mol2"):  # SDMolSupplier also works for mol2 files
                rdkit_mol = Chem.SDMolSupplier(lig_fname, sanitize=sanitize_mol_upon_reading, removeHs=removeHs_upon_reading)[0]
            else:
                raise ValueError(f"Ligand file format not recognised. Please provide a .sdf or .pdb file.")
        except (ValueError, AttributeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Something went wrong loading {lig_fname}..\n{e}") from e

        if lig_smiles is not None:
            if lig_from_xray:
                # Kekulization errors often arise when reading a ligand from an X-ray structure;
                # resetting bond info before SMILES assignment avoids them.
                for bond in rdkit_mol.GetBonds():
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)
            rdkit_mol = assign_bondOrders(rdkit_mol, lig_smiles)
            Chem.SanitizeMol(rdkit_mol)
            # save the bond-order-corrected ligand for traceability
            basename = os.path.basename(lig_fname)
            lig_fname = os.path.splitext(basename)[0]
            fixed_ligfname = os.path.join(self.out_dir, f"{lig_fname}_fixed.sdf")
            writer = Chem.SDWriter(fixed_ligfname)
            for cid in range(rdkit_mol.GetNumConformers()):
                writer.write(rdkit_mol, confId=-1)

        ligand = Molecule.from_rdkit(rdkit_mol, True)

        return ligand

    def _parametrize_ligand(self, ligand):
        """Parametrize a ligand and return its OpenMM topology and positions.

        Parameters
        ----------
        ligand : openff.toolkit.Molecule
            OpenFF molecule with 3-D coordinates.

        Returns
        -------
        ligand_omm_topology : openmm.app.Topology
            OpenMM topology containing only the ligand.
        ligand_positions : openmm.unit.Quantity
            Atom positions in nanometres.
        """

        if self._lig_ff_family == "ESPALOMA":
            template_generator = EspalomaTemplateGenerator(
                molecules=ligand,
                forcefield=self.lig_ff,
            )

        elif self._lig_ff_family == "OPENFF":
            template_generator = SMIRNOFFTemplateGenerator(
                molecules=ligand,
                forcefield=self.lig_ff,
            )

        elif self._lig_ff_family == "GAFF":
            template_generator = GAFFTemplateGenerator(
                molecules=ligand,
                forcefield=self.lig_ff,
            )

        self.forcefield.registerTemplateGenerator(template_generator.generator)

        ligand_off_topology = offTopology.from_molecules(molecules=[ligand])
        ligand_omm_topology = ligand_off_topology.to_openmm()
        ligand_positions = offquantity_to_openmm(ligand.conformers[0])

        return ligand_omm_topology, ligand_positions

    def run(self,
            protein: str = None,
            variants: dict = None,
            ligands: Union[str, List[tuple[str, str, str]]] = None
            ) -> tuple[System, Topology]:
        """Build, solvate, and parametrize the full simulation system.

        Stages:

        1. Load protein PDB and optionally apply residue variants
           (protonation states, disulfide bonds, etc.).
        2. Parametrize and add each ligand to the OpenMM Modeller.
        3. Solvate with explicit water/ions or embed in a lipid bilayer.
        4. Create the OpenMM System with PME electrostatics and HBond
           constraints.
        5. Write ``system.xml``, ``system.pdb``, and AMBER topology /
           coordinate files to ``self.out_dir``.

        Parameters
        ----------
        protein : str, optional
            Path to the prepared protein PDB file. If ``None``, a
            ligand-only or membrane-only system is built.
        variants : dict, optional
            Residue variant map passed to ``add_variants()`` (e.g. to
            specify HIS protonation states or CYS oxidation).
        ligands : str or list of tuple, optional
            Either a single SDF/PDB path (str) or a list of
            ``(name, path, smiles, from_xray)`` tuples for multiple
            ligands.

        Returns
        -------
        system : openmm.System
            Fully parametrized OpenMM System object.
        topology : openmm.app.Topology
            Topology of the solvated/membrane system.

        Raises
        ------
        RuntimeError
            If membrane building fails.
        Exception
            Re-raised if the protein PDB cannot be loaded.
        """

        start_time = time.monotonic()

        if protein is not None:
            rec_name = os.path.splitext(os.path.basename(protein))[0]
            try:
                protein_pdb = PDBFile(protein)
                logger.info(f"Loaded {rec_name} PDB..")
            except Exception as e:
                logger.error(f"Something went wrong loading {rec_name} PDB..\n{e}")
                raise

            modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

            if variants is not None:
                modeller = add_variants(modeller, variants)

            if ligands is not None:
                if isinstance(ligands, str):
                    logger.info(f"Parametrizing ligand {os.path.basename(ligands)}..")
                    lig = self._ligand_to_mol(ligands)
                    ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                    for res in ligand_topology.residues():
                        res.name = 'UNK'
                    modeller.add(ligand_topology, ligand_positions)

                elif isinstance(ligands, list):
                    used_chains = set(c.id for c in modeller.topology.chains()) if modeller else set()
                    chain_id = ord('A')
                    for lig_name, lig_path, lig_smiles, lig_from_xray in ligands:
                        while chr(chain_id) in used_chains:
                            chain_id += 1
                        logger.info(f"Parametrizing ligand {lig_name}..")
                        lig = self._ligand_to_mol(lig_path, lig_smiles, lig_from_xray)
                        ligand_topology, ligand_positions = self._parametrize_ligand(lig)
                        for chain in ligand_topology.chains():
                            chain.id = chr(chain_id)
                        for res in ligand_topology.residues():
                            res.name = lig_name
                        modeller.add(ligand_topology, ligand_positions)
                        used_chains.add(chr(chain_id))
                        chain_id += 1

        # CASE: Ligand and membrane only (no protein)
        max_length = 0.0 * openmmunit.angstroms
        if protein is None and self.is_membrane:

            # Centre ligand at the origin before translating it into the membrane.
            lig_com = np.mean(ligand_positions, axis=0)
            for i, xyz_i in enumerate(ligand_positions):
                ligand_positions[i] = xyz_i - lig_com

            pairwise_distances = np.linalg.norm(
                ligand_positions[:, None] - ligand_positions, axis=2
            )
            max_length = np.max(pairwise_distances) * openmmunit.angstroms

            translation_distance = 3.0

            translation_vector = np.array([0, 0, translation_distance])
            ligand_positions += translation_vector * openmmunit.nanometers

            modeller = Modeller(ligand_topology, ligand_positions)

        # A 4-site water model needs its virtual site on EVERY water, including the
        # crystallographic ones carried in from the input, and this has to happen BEFORE
        # solvation: addSolvent computes the neutralising charge and therefore requires every
        # residue to match a template, so 3-site input waters abort it with "matches HOH, but
        # the residue is missing 1 extra site".
        if self.water_model in ("tip4pew", "tip5p", "swm4ndp"):
            logger.info(f"Adding extra particles for the {self.water_model} water model..")
            modeller.addExtraParticles(self.forcefield)

        if self.is_membrane:
            logger.info(f"Adding a {os.path.basename(self.lipid_type)} membrane to the system..")
            if os.path.exists(self.lipid_type):
                lipid_patch = PDBFile(self.lipid_type)
            else:
                lipid_patch = self.lipid_type
            try:
                modeller.addMembrane(
                    forcefield=self.forcefield,
                    lipidType=lipid_patch,
                    neutralize=True,
                    ionicStrength=self.ionicStrength,
                    positiveIon=self.ions[0],
                    negativeIon=self.ions[1],
                    minimumPadding=self.padding + max_length,
                )

            except OpenMMException as e:
                raise RuntimeError(f"Something went wrong while building the membrane.\n{e}") from e

        else:
            logger.info(f"Solvating the system..")
            modeller.addSolvent(
                self.forcefield,
                model=self.water_model,
                neutralize=True,
                numAdded=self.num_solvent,
                ionicStrength=self.ionicStrength,
                positiveIon=self.ions[0],
                negativeIon=self.ions[1],
                boxShape=self.boxShape,
                padding=self.padding,
            )

        logger.info(f"Creating the OpenMM system..")
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

        # A second system is created with rigidWater=False and without HBond
        # constraints so that ParmEd can convert it to AMBER topology files.
        # See: https://parmed.github.io/ParmEd/html/openmm.html
        openmm_system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nb_cutoff,
            switchDistance=self.switchDistance,
            removeCMMotion=True,
            rigidWater=False,   # required for ParmEd compatibility
            hydrogenMass=self.hydrogenMass,
            # constraints intentionally omitted for ParmEd compatibility
        )

        save_amber_files(modeller.topology, modeller.positions, openmm_system, self.out_dir)

        simulation_time = time.monotonic() - start_time
        logger.info(f"Finished system preparation in {simulation_time:.2f} seconds.")

        return system, modeller.topology
