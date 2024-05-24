# general imports
import os
import time
import logging
import numpy as np

# OpenMM imports
from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

# OpenFF-toolkit imports
from openff.toolkit import Molecule
from openff.toolkit import Topology as offTopology
from openff.units.openmm import to_openmm as offquantity_to_openmm
from openmmforcefields.generators import EspalomaTemplateGenerator, SMIRNOFFTemplateGenerator, GAFFTemplateGenerator

# RDKit imports
from rdkit.Chem import SDMolSupplier

# AutoPath imports
from autopath.utils import fix_pdb, save_pdb, save_system, save_amber_topology

class SystemPreparation:
    def __init__(self,
                 forcefield:list = ['amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml', 'amber/tip3p_HFE_multivalent.xml'],
                 lig_ff:str = 'espaloma',
                 allow_undefined_stereo:bool = True,
                 hydrogenMass:float=3,
                 boxShape:str='dodecahedron',
                 padding:float=1.0,
                 ionicStrength:float=0.0,
                 ) -> None:

        if lig_ff.upper() in ['ESPALOMA', 'SMIRNOFF', 'GAFF']:
            self.lig_ff = lig_ff.upper()
        else:
            logging.error(f'Ligand forcefield must be one of Espaloma, SMIRNOFF or GAFF')
            exit(0)

        self.forcefield = ForceField(*forcefield)
        self.allow_undefined_stereo = allow_undefined_stereo

        self.hydrogenMass = hydrogenMass * openmmunit.amu # Use HMR 
        self.boxShape = boxShape #cube, dodecahedron
        self.padding = padding * openmmunit.nanometers
        
        self.ionicStrength = ionicStrength * openmmunit.molar

        # you proabably dont want to change this
        self.nb_cutoff = 1.0 * openmmunit.nanometers 
        self.switchDistance = 0.9 * openmmunit.nanometers

    def _sdf_to_mol(self, lig_sdf):

        # Load ligand SDF
        rdkit_mol = SDMolSupplier(lig_sdf)[0]

        # Convert to OpenMM molecule
        ligand = Molecule.from_rdkit(rdkit_mol,
                                    allow_undefined_stereo = self.allow_undefined_stereo
                                    )
        return ligand

    def _parametrize_ligand(self, ligand):
        
        if self.lig_ff=='ESPALOMA':
            template_generator =  EspalomaTemplateGenerator(molecules=ligand, forcefield='espaloma-0.3.2')
        elif self.lig_ff=='SMIRNOFF':
            template_generator =  SMIRNOFFTemplateGenerator(molecules=ligand, forcefield='openff-1.2.0')
        elif self.lig_ff == 'GAFF':
            template_generator = GAFFTemplateGenerator(molecules=ligand, forcefield='gaff-2.11')
        
        # add the template generator to the ff
        self.forcefield.registerTemplateGenerator(template_generator.generator)
        
        # make an OpenFF Topology of the ligand
        ligand_off_topology = offTopology.from_molecules(molecules=[ligand])

        # convert it to an OpenMM Topology
        ligand_omm_topology = ligand_off_topology.to_openmm()

        # get the positions of the ligand
        ligand_positions = offquantity_to_openmm(ligand.conformers[0])

        return ligand_omm_topology, ligand_positions

    def run(self, prot_path, lig_path):

        start_time = time.monotonic()

        rec_name = os.path.splitext(os.path.basename(prot_path))[0]
        lig_name = os.path.splitext(os.path.basename(lig_path))[0]

        # process ligand
        logging.info(f'Parametrizing ligand {lig_name}..')
        lig = self._sdf_to_mol(lig_path)
        ligand_topology, ligand_positions = self._parametrize_ligand(lig)

        # process protein
        logging.info(f'Loading {rec_name} PDB..')
        protein_pdb = PDBFile(prot_path)

        # make an OpenMM Modeller object with the protein
        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

        # add the ligand to the Modeller
        modeller.add(ligand_topology, ligand_positions)
        
        logging.info(f'Adding solvent..')
        modeller.addSolvent(self.forcefield, neutralize=True, 
                            ionicStrength=self.ionicStrength,
                            boxShape=self.boxShape, padding=self.padding)
        
        logging.info(f'Creating the system..')
        system = self.forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=self.nb_cutoff,
                                switchDistance=self.switchDistance,removeCMMotion=True, rigidWater=True, 
                                hydrogenMass=self.hydrogenMass, constraints=HBonds)

        
        save_system(system, f'{lig_name}/system.xml')
        save_pdb(modeller.topology, modeller.positions, f'{lig_name}/system.pdb')
        save_amber_topology(modeller.topology, modeller.positions, system, self.forcefield, lig_name)

        simulation_time = time.monotonic() - start_time
        logging.info(f'Finished system preparation in {simulation_time:.2f} seconds.')

        return