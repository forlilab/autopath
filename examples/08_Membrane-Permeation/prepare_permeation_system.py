#!/usr/bin/env python
"""Build a ligand + membrane system for metadynamics permeation runs (a thin driver over SystemPreparation).

Ligand from SMILES (3D by molscrub; acid/base, tautomers and rings left as given) or an SDF,
centred in x/y with its nearest atom |z| nm from the membrane centre, on the side of the sign
of z. Inner depths |z| <= 2.6 nm are built at +/-2.6 nm: reach them by steered MD.
Membrane: an OpenMM lipid name (POPC, ...) or a pre-equilibrated patch, PATCH_DIR/NAME/NAME.pdb.
Adds the membrane-centre reference DUM (``add_membrane_center_atom``) and a cylinder restraint
on the ligand centroid (free along z, flat bottom of radius R in x/y).

Writes OUT/UNK.sdf, OUT/system.xml, OUT/system.pdb (no barostat: equilibration adds it).
With --both-sides (z > 0) also OUT/system_minus.pdb: the same system rotated 180 deg about x,
(x, y, z) -> (x, -y, -z), so the ligand sits at -z. A rotation, not a mirror (chirality kept);
same atoms in the same order, so both sides share system.xml and their topologies match.
Also OUT/system.prmtop/.rst7 (with DUM) for Equilibration's atom selections only: DUM matches the
Na+ template there; the simulation uses system.xml, where DUM has no charge and no interactions.
All starts share this one prmtop (same topology).

Example:
    python prepare_permeation_system.py --smiles "CCO" --membrane 50_lip_per_leaf_100_percent_POPC_20A_wat \\
        --z 2.7 --both-sides --out ethanol
"""
import argparse
import math
import os
import sys

DEFAULT_PATCH_DIR = "/mnt/forli/group/abarkdull/Ligand_Bilayer_MD/lipid_patches"
SMD_DEPTH = 2.6                                   # nm; inner starts are built here and pulled in
LIGAND_FF = {"espaloma": "espaloma-0.3.2", "openff": "openff-1.2.0", "gaff": "gaff-2.11"}


def ligand_mol(smiles=None, sdf=None):
    """RDKit Mol with explicit H and 3D coordinates."""
    from rdkit import Chem
    if sdf:
        mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
        if mol is None:
            sys.exit(f"could not read {sdf}")
        return mol
    try:
        from molscrub import Scrub
    except ImportError:                           # older molscrub releases install it as `scrubber`
        from scrubber import Scrub
    scrub = Scrub(skip_acidbase=True, skip_gen3d=False, skip_tautomers=True, skip_ringfix=True, etkdg_rng_seed=1)
    mols = list(scrub(Chem.MolFromSmiles(smiles)))
    if not mols:
        sys.exit(f"molscrub returned no molecule for {smiles}")
    return Chem.AddHs(mols[-1], addCoords=True)


def place(mol, z_nm):
    """Centre the ligand in x/y; put its nearest atom |z_nm| from z = 0, on the side of the sign."""
    from rdkit.Geometry import Point3D
    conf = mol.GetConformer()
    xyz = conf.GetPositions()
    xyz -= xyz.mean(axis=0)
    z_ang = 10.0 * z_nm
    xyz[:, 2] += (z_ang - xyz[:, 2].min()) if z_ang >= 0 else (z_ang - xyz[:, 2].max())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*p))
    return mol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    lig = ap.add_mutually_exclusive_group(required=True)
    lig.add_argument("--smiles")
    lig.add_argument("--sdf", help="ligand with explicit H and 3D coordinates")
    ap.add_argument("--membrane", required=True, help="OpenMM lipid name or a patch folder name in --patch-dir")
    ap.add_argument("--z", type=float, required=True, help="starting depth (nm) of the ligand's nearest atom")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--both-sides", action="store_true",
                    help="also write system_minus.pdb, the build rotated 180 deg about x (ligand at -z)")
    ap.add_argument("--patch-dir", default=DEFAULT_PATCH_DIR)
    ap.add_argument("--ligand-ff", choices=sorted(LIGAND_FF), default="espaloma")
    ap.add_argument("--hmass", type=float, default=3.0, help="hydrogen mass (amu) for the 4 fs timestep")
    ap.add_argument("--ionic-strength", type=float, default=0.0, help="molar, on top of neutralising")
    ap.add_argument("--cyl-k", type=float, default=5.0, help="cylinder restraint k (kcal/mol/A^2)")
    ap.add_argument("--cyl-r", type=float, default=5.0, help="cylinder radius (A)")
    opt = ap.add_argument_group("opt-in restraints")
    opt.add_argument("--z-wall", action="store_true",
                     help="flat-bottom wall on |z| of the ligand centroid at 95%% of half the box height")
    a = ap.parse_args(argv)
    if a.both_sides and a.z <= 0:
        ap.error("--both-sides builds at +z and rotates to -z: give z > 0")

    from openmm import CustomCentroidBondForce, unit
    from openmm.app import PME, Modeller, PDBFile
    from rdkit import Chem
    from autopath.customForces import add_cylindrical_restraints, add_membrane_center_atom
    from autopath.preparation import SystemPreparation
    from autopath.utils import save_amber_files, save_pdb, save_system

    # ligand at its build depth, written as the SystemPreparation input
    z_build = a.z if abs(a.z) > SMD_DEPTH else math.copysign(SMD_DEPTH, a.z)
    if z_build != a.z:
        print(f"building at z = {z_build:+.1f} nm; reach z = {a.z:+.1f} nm by steered MD")
    mol = place(ligand_mol(a.smiles, a.sdf), z_build)
    os.makedirs(a.out, exist_ok=True)
    sdf = os.path.join(a.out, "UNK.sdf")
    with Chem.SDWriter(sdf) as w:
        w.write(mol)

    patch = os.path.join(a.patch_dir, a.membrane, f"{a.membrane}.pdb")
    prep = SystemPreparation(forcefield=["amber14-all.xml", "amber14/tip3pfb.xml"], lig_ff=LIGAND_FF[a.ligand_ff],
                             hydrogenMass=a.hmass, padding=1.0, ionicStrength=a.ionic_strength, is_membrane=True,
                             lipid_type=patch if os.path.exists(patch) else a.membrane,
                             membrane_ligand_z=None, out_dir=a.out)
    system, _ = prep.run(ligands=sdf)

    # DUM, cylinder and opt-in restraints on top of SystemPreparation's system
    pdb = PDBFile(os.path.join(a.out, "system.pdb"))
    modeller = Modeller(pdb.topology, pdb.positions)
    solute = [x.index for x in modeller.topology.atoms() if x.residue.name == "UNK"]
    if len(solute) != mol.GetNumAtoms():
        sys.exit(f"found {len(solute)} UNK atoms for a {mol.GetNumAtoms()}-atom ligand")
    dum = add_membrane_center_atom(system, modeller)
    add_cylindrical_restraints(system, [dum], solute,
                               k_xy=a.cyl_k * unit.kilocalorie_per_mole / unit.angstrom**2,
                               R_cylinder=a.cyl_r * unit.angstrom)

    if a.z_wall:
        lz = modeller.topology.getUnitCellDimensions()[2].value_in_unit(unit.nanometer)
        wall = CustomCentroidBondForce(2, "step(d - upper_wall) * 0.5 * k_flat * (d - upper_wall)^2;"
                                          "d = pointdistance(0, 0, z1, 0, 0, z2)")
        wall.addGlobalParameter("upper_wall", 0.95 * 0.5 * lz)
        wall.addGlobalParameter("k_flat", (10.0 * unit.kilocalorie_per_mole / unit.angstrom**2)
                                .value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
        wall.addBond([wall.addGroup(solute), wall.addGroup([dum])], [])
        wall.setUsesPeriodicBoundaryConditions(False)
        wall.setForceGroup(12)                    # 14-31 are reserved (customForces._FG_*)
        system.addForce(wall)


    save_system(system, os.path.join(a.out, "system.xml"))
    save_pdb(modeller.topology, modeller.positions, os.path.join(a.out, "system.pdb"))
    # prmtop for Equilibration's MDAnalysis selections; ParmEd needs flexible water and no constraints
    amber_system = prep.forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
                                                nonbondedCutoff=1.0 * unit.nanometer, rigidWater=False,
                                                hydrogenMass=a.hmass * unit.amu)
    save_amber_files(modeller.topology, modeller.positions, amber_system, a.out)
    if a.both_sides:
        import numpy as np
        xyz = np.array(modeller.positions.value_in_unit(unit.nanometer)) * [1.0, -1.0, -1.0]   # about x, through DUM
        save_pdb(modeller.topology, xyz * unit.nanometer, os.path.join(a.out, "system_minus.pdb"))
    box =modeller.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    print(f"wrote {a.out}/UNK.sdf, system.xml, system.pdb: {system.getNumParticles()} atoms, "
          f"box {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} nm")


if __name__ == "__main__":
    main()
