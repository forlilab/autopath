# autopath/metadynamics/learned/water.py
"""W1 (ligand-atom hydration) and W3 (non-ligand fill of apolar pocket patches) feature definitions."""
import logging

import numpy as np
from openmm import app, unit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from .features import Feature

log = logging.getLogger(__name__)

WATER_RES = ("HOH", "WAT", "SOL")
WATER_O = ("O", "OW")
AMINO = frozenset("""ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS HID HIE HIP ILE LEU LYS LYN MET PHE PRO SER THR TRP TYR
VAL ASH GLH CYM HYP ACE NME NMA""".split())
BULK_NM = 1.0
GRID_NM = 0.1


def _load(topology_pdb):
    pdb = app.PDBFile(topology_pdb)
    return pdb.topology, np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def _is_protein(res):
    return res.name in AMINO or {"N", "CA", "C"} <= {a.name for a in res.atoms()}


def _heavy(atom):
    return atom.element is not None and atom.element.symbol != "H"


def _lig_residues(atoms, ligand_idx):
    return {atoms[int(i)].residue.index for i in ligand_idx}


def _is_water_o(atom):
    return atom.residue.name in WATER_RES and atom.name in WATER_O


def apolar_pocket_atoms(topology_pdb, ligand_idx, shell_nm=0.6) -> list:
    """Apolar protein heavy atoms in the ligand pocket shell.

    Parameters
    ----------
    topology_pdb : str
        PDB with coordinates (nm after loading) and bonds.
    ligand_idx : sequence of int
        Ligand atom indices; their residues are never protein.
    shell_nm : float
        Shell radius around ligand heavy atoms.

    Returns
    -------
    list of int
        Sorted atom indices of sulphur, or carbon with no N/O bonded neighbour.
    """
    top, X = _load(topology_pdb)
    atoms = list(top.atoms())
    lres = _lig_residues(atoms, ligand_idx)
    lig = [int(i) for i in ligand_idx if _heavy(atoms[int(i)])]
    cand = [a.index for a in atoms if a.residue.index not in lres and _is_protein(a.residue) and _heavy(a)]
    if not lig or not cand:
        return []
    d, _ = cKDTree(X[lig]).query(X[cand])
    shell = [i for i, di in zip(cand, d) if di <= shell_nm]
    nbrs = {i: [] for i in shell}
    for b in top.bonds():
        for u, v in ((b[0], b[1]), (b[1], b[0])):
            if u.index in nbrs:
                nbrs[u.index].append(v.element.symbol if v.element is not None else "")
    for res in {atoms[i].residue for i in shell}:
        heavy = [a for a in res.atoms() if _heavy(a)]
        if len(heavy) > 1 and not any(nbrs.get(a.index) is not None and nbrs[a.index] for a in heavy):
            raise ValueError(f"residue {res.name}{res.id} has no bonds in {topology_pdb}; "
                             "non-standard residue not supported")
    out = []
    for i in shell:
        el = atoms[i].element.symbol
        if el == "S" or (el == "C" and not {"N", "O"} & set(nbrs[i])):
            out.append(i)
    return sorted(out)


def patches(positions_nm, atoms, cut_nm=0.6, min_size=3) -> list:
    """Average-linkage patches of atoms.

    Parameters
    ----------
    positions_nm : array (n_atoms, 3)
        Coordinates indexed by atom id.
    atoms : sequence of int
        Atom ids to cluster.
    cut_nm : float
        Distance at which the dendrogram is cut.
    min_size : int
        Clusters smaller than this are merged into the cluster with the nearest centroid.

    Returns
    -------
    list of list of int
        Patches sorted by smallest atom id.
    """
    atoms = sorted(int(i) for i in atoms)
    if len(atoms) < 2:
        return [atoms] if atoms else []
    P = np.asarray(positions_nm, dtype=float)[atoms]
    lab = fcluster(linkage(P, method="average"), t=cut_nm, criterion="distance")
    groups = [[k for k in range(len(atoms)) if lab[k] == c] for c in sorted(set(lab))]
    while len(groups) > 1:
        small = [g for g in groups if len(g) < min_size]
        if not small:
            break
        g = min(small, key=lambda g: (len(g), g[0]))
        cen = P[g].mean(0)
        rest = [h for h in groups if h is not g]
        h = min(rest, key=lambda h: (np.linalg.norm(P[h].mean(0) - cen), h[0]))
        h.extend(g)
        groups = rest
    return sorted((sorted(atoms[k] for k in g) for g in groups), key=lambda g: g[0])


def protein_weight(topology_pdb, ligand_idx=()) -> float:
    """Ratio of bulk-water oxygen density to protein-core heavy-atom density.

    Parameters
    ----------
    topology_pdb : str
        PDB with a periodic box.
    ligand_idx : sequence of int
        Ligand atom indices; their residues are solute but not protein.

    Returns
    -------
    float
        rho_O (water O farther than 1 nm from every solute atom) divided by the density of
        protein heavy atoms within 0.5 R_g of the protein centre.
    """
    top, X = _load(topology_pdb)
    bv = top.getPeriodicBoxVectors()
    if bv is None:
        raise ValueError(f"{topology_pdb} has no box (CRYST1)")
    box = np.array([[v.x, v.y, v.z] for v in bv.value_in_unit(unit.nanometer)])
    atoms = list(top.atoms())
    lres = _lig_residues(atoms, ligand_idx)
    solute = [a.index for a in atoms if a.residue.name not in WATER_RES and (
        a.residue.index in lres or _is_protein(a.residue) or len(list(a.residue.atoms())) > 1)]
    prot = [a.index for a in atoms if _heavy(a) and a.residue.index not in lres and _is_protein(a.residue)]
    wat = [a.index for a in atoms if _is_water_o(a)]
    if not prot or not solute:
        raise ValueError(f"{topology_pdb} has no protein heavy atoms")
    inv = np.linalg.inv(box)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ box
    S = X[solute]
    S = (S - np.floor(S @ inv) @ box)[None] + shifts[:, None]
    tree = cKDTree(S.reshape(-1, 3))
    W = X[wat]
    W = W - np.floor(W @ inv) @ box
    n_bulk = int(np.sum(tree.query(W)[0] > BULK_NM)) if len(W) else 0
    n = np.maximum(np.round(np.linalg.norm(box, axis=1) / GRID_NM).astype(int), 1)
    f = [(np.arange(k) + 0.5) / k for k in n]
    G = np.array(np.meshgrid(*f, indexing="ij")).reshape(3, -1).T @ box
    frac_bulk = np.mean(tree.query(G)[0] > BULK_NM)
    if n_bulk == 0 or frac_bulk == 0:
        raise ValueError(f"no bulk water more than {BULK_NM} nm from the solute in {topology_pdb}")
    rho_o = n_bulk / (frac_bulk * abs(np.linalg.det(box)))
    P = X[prot]
    com = P.mean(0)
    rg = np.sqrt(np.mean(np.sum((P - com) ** 2, axis=1)))
    r = 0.5 * rg
    n_in = int(np.sum(np.linalg.norm(P - com, axis=1) <= r))
    if n_in == 0 or r == 0:
        raise ValueError(f"no protein heavy atoms within 0.5 Rg of the protein centre in {topology_pdb}")
    return float(rho_o / (n_in / (4.0 / 3.0 * np.pi * r ** 3)))


def linked_exceptions(feature, system) -> list:
    """NonbondedForce exception pairs linked by the terms of a feature.

    Parameters
    ----------
    feature : Feature
        cn feature with terms.
    system : openmm.System
        System holding a NonbondedForce.

    Returns
    -------
    list of list of int
        Sorted [i, j] pairs with i and j on opposite sides of some term.
    """
    from .forces import _nonbonded
    nb = _nonbonded(system)
    sides = {}
    for t, (a, b, _) in enumerate(feature.terms):
        for i in a:
            sides.setdefault(i, set()).add((t, 0))
        for i in b:
            sides.setdefault(i, set()).add((t, 1))
    out = set()
    for e in range(nb.getNumExceptions()):
        i, j = nb.getExceptionParameters(e)[:2]
        if i in sides and j in sides and any((t, 1 - s) in sides[j] for t, s in sides[i]):
            out.add((min(i, j), max(i, j)))
    return sorted([list(p) for p in out])


def define_water_features(topology_pdb, ligand_idx, system=None):
    """W1 and W3 cn features of a complex.

    Parameters
    ----------
    topology_pdb : str
        PDB with coordinates, bonds and a periodic box.
    ligand_idx : sequence of int
        Ligand atom indices.
    system : openmm.System, optional
        Source of the NonbondedForce exceptions for ``Feature.exclude``; required when W3 is non-empty.

    Returns
    -------
    features : list of Feature
        ``W1_<atom>`` per ligand heavy atom, then ``W3_p<k>`` per apolar patch.
    info : dict
        ``w_p``, ``patches`` (``resid:name`` members) and ``n_apolar``.
    """
    top, X = _load(topology_pdb)
    atoms = list(top.atoms())
    if system is not None and system.getNumParticles() != len(atoms):
        raise ValueError(f"system has {system.getNumParticles()} particles but {topology_pdb} has "
                         f"{len(atoms)} atoms; not supported")
    lres = _lig_residues(atoms, ligand_idx)
    wat = [a.index for a in atoms if _is_water_o(a)]
    lig = [int(i) for i in ligand_idx if _heavy(atoms[int(i)])]
    names = [atoms[i].name for i in lig]
    feats = [Feature(f"W1_{n}" if names.count(n) == 1 else f"W1_{n}_{i}", "cn", "W1", terms=[[[i], wat, 1.0]])
             for i, n in zip(lig, names)]
    apolar = apolar_pocket_atoms(topology_pdb, ligand_idx)
    groups = [g for g in patches(X, apolar) if len(g) >= 3]
    info = {"w_p": None, "patches": [], "n_apolar": len(apolar)}
    if not groups:
        log.warning("no apolar pocket patch of >= 3 atoms near the ligand; W3 features are empty")
    else:
        if system is None:
            raise ValueError("W3 exclusions need the openmm.System; define_water_features(system=None) is "
                             "not supported when W3 is non-empty")
        w_p = protein_weight(topology_pdb, ligand_idx)
        info["w_p"] = w_p
        prot = [a.index for a in atoms if _heavy(a) and a.residue.index not in lres and _is_protein(a.residue)]
        for k, patch in enumerate(groups):
            by_res = {}
            for i in patch:
                by_res.setdefault(atoms[i].residue.index, []).append(i)
            terms = []
            for r, grp in sorted(by_res.items(), key=lambda kv: kv[1][0]):
                other = [i for i in prot if atoms[i].residue.index != r]
                terms.append([grp, wat, 1.0])
                terms.append([grp, other, w_p])
            feats.append(Feature(f"W3_p{k}", "cn", "W3", terms=terms))
            info["patches"].append([f"{atoms[i].residue.id}:{atoms[i].name}" for i in patch])
    if system is not None:
        for f in feats:
            f.exclude = linked_exceptions(f, system) or None
    return feats, info
