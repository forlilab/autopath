"""F1–F8 feature definitions for one complex, from its structure and its sMD frames."""
import os

import MDAnalysis as mda
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from .features import Feature

NATIVE_CUTOFF_NM, NATIVE_FRACTION = 0.45, 0.5
R0 = dict(F1=0.35, F2=0.45, F4=0.32, F6=0.40)
EXIT_CONTACT_NM = 0.45
_FACTORY = None


def _factory():
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    return _FACTORY


def ligand_polar_atoms(ligand_sdf, ligand_idx, topology_pdb) -> list:
    """Topology indices of ligand donors/acceptors/ionisable atoms (RDKit BaseFeatures)."""
    mol = Chem.MolFromMolFile(ligand_sdf, removeHs=False)
    u = mda.Universe(topology_pdb)
    lig = u.atoms[list(ligand_idx)]
    heavy_top = [a.index for a in lig if a.element != "H"]
    heavy_sdf = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    el_top = [u.atoms[i].element.capitalize() for i in heavy_top]
    el_sdf = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_sdf]
    if el_top != el_sdf:
        raise ValueError(f"SDF and topology ligand heavy atoms differ in element order: {el_sdf} vs {el_top}")
    sdf_to_top = dict(zip(heavy_sdf, heavy_top))
    fams = ("Donor", "Acceptor", "PosIonizable", "NegIonizable")
    polar = {sdf_to_top[i] for f in _factory().GetFeaturesForMol(mol) if f.GetFamily() in fams
             for i in f.GetAtomIds() if i in sdf_to_top}
    return sorted(polar)


def _dist(X, i, j):
    return np.linalg.norm(X[:, j] - X[:, i], axis=-1)


def define_features(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, bound_X, exit_X) -> list:
    u = mda.Universe(topology_pdb)
    lig_heavy = [int(i) for i in ligand_idx if u.atoms[int(i)].element != "H"]
    pocket = u.atoms[sorted(set(int(i) for i in pocket_idx))]
    pocket_heavy = [int(a.index) for a in pocket if a.element != "H"]
    pocket_polar = [int(a.index) for a in pocket if a.element in ("N", "O")]
    wat = [int(i) for i in u.select_atoms("resname HOH WAT SOL and name O OW").indices]
    feats = []

    polar = ligand_polar_atoms(ligand_sdf, ligand_idx, topology_pdb)
    partners = set()
    for i in polar:
        for j in pocket_polar:
            if np.mean(_dist(bound_X, i, j) < NATIVE_CUTOFF_NM) >= NATIVE_FRACTION:
                a, b = u.atoms[i], u.atoms[j]
                feats.append(Feature(f"F1_{a.name}-{b.name}_{b.resid}", "pair", "F1",
                                     pairs=[(i, j)], r0=R0["F1"]))
                partners.add(j)

    for res in pocket.residues:
        heavy = [int(a.index) for a in res.atoms if a.element != "H"]
        feats.append(Feature(f"F2_res{res.resid}", "contacts", "F2", group_a=lig_heavy,
                             group_b=heavy, r0=R0["F2"]))

    feats.append(Feature("F3_com", "centroid_distance", "F3", groups=[pocket_heavy, lig_heavy]))

    for j in sorted(partners):
        a = u.atoms[j]
        feats.append(Feature(f"F4_{a.name}_{a.resid}", "contacts", "F4", group_a=[j], group_b=wat,
                             r0=R0["F4"]))

    pc = bound_X[:, pocket_heavy].mean(1)
    rg = float(np.sqrt(np.mean(np.sum((bound_X[:, pocket_heavy] - pc[:, None]) ** 2, -1))))
    feats.append(Feature("F5_pocket_water", "centroid_count", "F5", groups=[pocket_heavy],
                         group_b=wat, r0=max(rg, 0.45)))

    feats.append(Feature("F6_ligand_water", "contacts", "F6", group_a=lig_heavy, group_b=wat,
                         r0=R0["F6"]))

    feats.extend(_exit_gates(u, pocket, lig_heavy, pocket_heavy, exit_X))
    feats.append(_orientation(u, pocket, lig_heavy, pocket_heavy, bound_X, exit_X))
    return feats


def _exit_residues(pocket, lig_heavy, exit_X):
    out = []
    for res in pocket.residues:
        heavy = [int(a.index) for a in res.atoms if a.element != "H"]
        r = np.linalg.norm(exit_X[:, lig_heavy][:, :, None] - exit_X[:, heavy][:, None], axis=-1)
        if np.any(r.min(axis=(1, 2)) < EXIT_CONTACT_NM):
            out.append(res)
    return out


def _exit_gates(u, pocket, lig_heavy, pocket_heavy, exit_X):
    res = _exit_residues(pocket, lig_heavy, exit_X)
    if len(res) < 2:
        return []
    axis = exit_X[:, lig_heavy].mean((0, 1)) - exit_X[:, pocket_heavy].mean((0, 1))
    axis /= np.linalg.norm(axis)
    c0 = exit_X[:, pocket_heavy].mean((0, 1))
    side = {}
    for r in res:
        ca = r.atoms.select_atoms("name CA")
        if len(ca):
            v = exit_X[:, ca.indices[0]].mean(0) - c0
            side[r.resid] = v - (v @ axis) * axis          # radial vector off the exit axis
    feats = []
    for k, r1 in enumerate(res):
        for r2 in res[k + 1:]:
            if r1.resid in side and r2.resid in side and side[r1.resid] @ side[r2.resid] < 0:
                ca1, ca2 = (int(r.atoms.select_atoms("name CA").indices[0]) for r in (r1, r2))
                feats.append(Feature(f"F7_ca{r1.resid}-{r2.resid}", "centroid_distance", "F7",
                                     groups=[[ca1], [ca2]]))
                sc1, sc2 = ([int(a.index) for a in r.atoms.select_atoms(
                    "not backbone and not name H* and not name CB")] for r in (r1, r2))
                if sc1 and sc2:
                    feats.append(Feature(f"F7_sc{r1.resid}-{r2.resid}", "centroid_distance", "F7",
                                         groups=[sc1, sc2]))
    return feats


def _orientation(u, pocket, lig_heavy, pocket_heavy, bound_X, exit_X):
    m = bound_X[:, lig_heavy].mean(0)
    d = np.linalg.norm(m[:, None] - m[None], axis=-1)
    a, b = np.unravel_index(np.argmax(d), d.shape)
    exit_res = _exit_residues(pocket, lig_heavy, exit_X) or list(pocket.residues)
    exit_heavy = [int(x.index) for r in exit_res for x in r.atoms if x.element != "H"]
    return Feature("F8_ligand_tilt", "cos", "F8",
                   groups=[[lig_heavy[a]], [lig_heavy[b]], pocket_heavy, exit_heavy])
