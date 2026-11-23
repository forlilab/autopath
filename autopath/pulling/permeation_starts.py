"""Helpers for building permeation start sets: end-to-end ligand atoms and topology checks."""
import numpy as np


def check_same_topology(pdb_files, system_files=None) -> None:
    """Raise ValueError unless every start has the same atoms, in the same order.

    Walkers sharing one bias must simulate the same system. Compares (residue name, atom name,
    element) of every atom across ``pdb_files`` and, if given, the particle count of every
    serialized System in ``system_files``. The first file of each list is the reference.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    def atoms(path):
        return [(a.residue.name, a.name, a.element.symbol if a.element else None)
                for a in PDBFile(path).topology.atoms()]

    ref = atoms(pdb_files[0])
    for p in pdb_files[1:]:
        other = atoms(p)
        if len(other) != len(ref):
            raise ValueError(f"{p} has {len(other)} atoms, {pdb_files[0]} has {len(ref)}")
        bad = next((i for i, (x, y) in enumerate(zip(ref, other)) if x != y), None)
        if bad is not None:
            raise ValueError(f"atom {bad}: {other[bad]} in {p}, {ref[bad]} in {pdb_files[0]}")
    for s in system_files or []:
        with open(s) as f:
            n = XmlSerializer.deserialize(f.read()).getNumParticles()
        if n != len(ref):
            raise ValueError(f"{s} has {n} particles, {pdb_files[0]} has {len(ref)} atoms")


def _load_mol(mol_or_sdf):
    """RDKit Mol with explicit H from a Mol or an SDF path (atom order kept)."""
    from rdkit import Chem
    if isinstance(mol_or_sdf, str):
        mol = Chem.MolFromMolFile(mol_or_sdf, removeHs=False)
        if mol is None:
            raise ValueError(f"could not read {mol_or_sdf}")
        return mol
    return mol_or_sdf


def end_to_end_atoms(mol_or_sdf, num_confs: int = 10, seed: int = 42) -> tuple[int, int]:
    """Indices (in the molecule's atom order) of the two heavy atoms farthest apart.

    Per ETKDG conformer: atom 1 is the heavy atom farthest from the centroid, atom 2 the
    heavy atom farthest from atom 1; the most common pair over the conformers wins.
    Hydrogens are skipped (their fast motion only adds noise to an orientation CV).
    Map to system indices with ``solute_atoms[i]``; this assumes the solute's topology
    follows the SDF atom order, which holds when the topology was built from that SDF.
    """
    from collections import Counter
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.Mol(_load_mol(mol_or_sdf))
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    if heavy.sum() < 2:
        raise ValueError("need at least two heavy atoms to define an end-to-end axis")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not ids:
        raise RuntimeError("conformer generation failed")
    pairs = []
    for cid in ids:
        xyz = mol.GetConformer(cid).GetPositions()
        d_c = np.where(heavy, np.linalg.norm(xyz - xyz.mean(axis=0), axis=1), -1.0)
        a1 = int(np.argmax(d_c))
        d_1 = np.where(heavy, np.linalg.norm(xyz - xyz[a1], axis=1), -1.0)
        pairs.append((a1, int(np.argmax(d_1))))
    return Counter(pairs).most_common(1)[0][0]
