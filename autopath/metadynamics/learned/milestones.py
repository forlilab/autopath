# autopath/metadynamics/learned/milestones.py
"""Class-medoid milestone frames of the learned CV and relaxed simulation-ready states for them."""
import logging
import os

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger("autopath")
MAX_PER_CLASS = 2000
MIN_CONTACT_NM = 0.15


def class_medoid_frames(features_npz, learned_npz, order=("B", "TS", "U")) -> list:
    """Medoid frame of each class in the standardised kept-feature + z/sigma_z space.

    Only rows with ``valid`` set are used. The medoid minimises the summed Euclidean distance to the
    class frames. A class of more than 2000 frames is subsampled to 2000 (seeded per class, so independent
    of the class order) and the medoid is chosen from that subsample.

    Parameters
    ----------
    features_npz : path of features.npz (``V``, ``labels``, ``z``, ``pull``, ``frame``, ``dcd``, ``valid``)
    learned_npz : path of learned_cv.npz (``kept``, ``mu``, ``sd``)
    order : class labels, in output order

    Returns
    -------
    list of dict with ``class``, ``pull``, ``frame`` (DCD frame index) and ``dcd`` (DCD basename)
    """
    with np.load(features_npz, allow_pickle=False) as f:
        V, labels, z = f["V"], f["labels"].astype(str), f["z"]
        pull, frame, dcd = f["pull"].astype(str), f["frame"].astype(int), f["dcd"].astype(str)
        valid = f["valid"].astype(bool)
    with np.load(learned_npz, allow_pickle=False) as g:
        kept, mu, sd = g["kept"].astype(int), g["mu"], g["sd"]
    if z.size != len(V):
        raise ValueError("features.npz has no per-row funnel z; milestones are not supported")
    in_class = valid & np.isin(labels, list(order))
    if not in_class.any():
        raise ValueError("no valid frames in the requested classes")
    sig_z = float(z[in_class].std())
    if not sig_z > 0:
        raise ValueError("funnel z has zero spread over the class frames; not supported")
    X = np.column_stack([(V[:, kept] - mu) / sd, z / sig_z])
    out = []
    for c in order:
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(valid & (labels == c))
        if not len(idx):
            raise ValueError(f"class {c!r} has no valid frames")
        if len(idx) > MAX_PER_CLASS:
            idx = np.sort(rng.choice(idx, MAX_PER_CLASS, replace=False))
        Xc = X[idx]
        if not np.isfinite(Xc).all():
            raise ValueError(f"non-finite features among the valid frames of class {c!r}")
        d = cdist(Xc, Xc).sum(1)
        i = idx[int(np.argmin(d))]
        out.append({"class": c, "pull": str(pull[i]), "frame": int(frame[i]), "dcd": str(dcd[i])})
    return out


def _graph(topology):
    """(molecule id per atom, edge array): edges are bonds plus consecutive atoms of a residue."""
    n = topology.getNumAtoms()
    edges = [(a.index, b.index) for a, b in topology.bonds()]
    for r in topology.residues():
        ids = [a.index for a in r.atoms()]
        edges += list(zip(ids[:-1], ids[1:]))
    edges = np.array(edges, dtype=int).reshape(-1, 2)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[find(i)] = find(j)
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1], edges


BRICK_FIT_MAX_NM = 0.05


def brick_fit(X, half, centre_idx, water_mols):
    """Rotation R (X @ R) that puts the water centres of an imaged frame back inside its axis-aligned brick.

    Minimises the summed squared excess of the water centroids outside the brick of half widths ``half``
    about the unweighted centroid of centre_idx (Nelder-Mead on a rotation vector, started at identity).

    Parameters
    ----------
    X : (N, 3) coordinates, nm
    half : (3,) half widths of the frame's own box, nm
    centre_idx : anchor atoms whose centroid is the imaging centre
    water_mols : (n_waters, k) atom indices of each water

    Returns
    -------
    R : (3, 3) rotation matrix
    residual : largest water excess outside the brick after the fit, nm
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    W = X[water_mols].mean(1)
    C = X[list(centre_idx)].mean(0)

    def excess(rv):
        R = Rotation.from_rotvec(rv).as_matrix()
        return np.maximum(np.abs((W - C) @ R) - half, 0.0)

    simplex = np.vstack([np.zeros(3), 0.1 * np.eye(3)])
    r = minimize(lambda rv: float((excess(rv) ** 2).sum()), np.zeros(3), method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, xatol=1e-9, fatol=1e-14, maxiter=5000, maxfev=5000))
    return Rotation.from_rotvec(r.x).as_matrix(), float(excess(r.x).max())


def restore_frame(X, box, centre_idx, water_mols, mol, edges):
    """Frame X (aligned, nm) rotated back to its lab orientation, bonds made whole, molecules imaged into box.

    Parameters
    ----------
    X : (N, 3) coordinates of the aligned frame, nm
    box : (3, 3) the frame's own reduced box vectors (rows), nm
    centre_idx : anchor atoms whose centroid is the imaging centre
    water_mols : (n_waters, k) atom indices of each water
    mol : (N,) molecule id per atom
    edges : (E, 2) atom pairs the molecules are made whole along

    Returns
    -------
    Y : (N, 3) coordinates, nm
    info : dict with ``angle_deg`` (rotation applied) and ``residual_nm`` (brick-fit residual)
    """
    from scipy.spatial.transform import Rotation

    R, resid = brick_fit(X, np.diag(box) / 2, centre_idx, water_mols)
    if resid > BRICK_FIT_MAX_NM:
        raise ValueError(f"water centres still lie {resid:.3f} nm outside the frame's box after the best "
                         f"rotation; trajectory not imaged into the frame's own brick; not supported")
    Y = X @ R
    Y += box.sum(0) / 2 - Y[list(centre_idx)].mean(0)
    inv = np.linalg.inv(box)

    def minimg(d):
        return d - np.round(d @ inv) @ box

    # unwrap each molecule along its edges, then wrap its centre into the cell
    nbr = [[] for _ in range(len(Y))]
    for i, j in edges:
        nbr[i].append(j)
        nbr[j].append(i)
    done = np.zeros(len(Y), bool)
    for s in range(len(Y)):
        if done[s]:
            continue
        done[s] = True
        stack = [s]
        while stack:
            i = stack.pop()
            for j in nbr[i]:
                if not done[j]:
                    Y[j] = Y[i] + minimg(Y[j] - Y[i])
                    done[j] = True
                    stack.append(j)
    for m in np.unique(mol):
        ids = mol == m
        Y[ids] -= np.floor(Y[ids].mean(0) @ inv) @ box
    return Y, {"angle_deg": float(np.degrees(Rotation.from_matrix(R).magnitude())), "residual_nm": resid}


def min_intermolecular_nm(Y, box, mol, heavy, cutoff=0.5):
    """Smallest heavy-atom distance between different molecules (inf when none within cutoff), nm."""
    from MDAnalysis.lib.distances import capped_distance
    from MDAnalysis.lib.mdamath import triclinic_box
    dims = triclinic_box(*(box * 10.0))
    idx = np.flatnonzero(heavy)
    pairs, d = capped_distance(Y[idx] * 10.0, Y[idx] * 10.0, cutoff * 10.0, box=dims, return_distances=True)
    keep = (pairs[:, 0] < pairs[:, 1]) & (mol[idx][pairs[:, 0]] != mol[idx][pairs[:, 1]])
    return float(d[keep].min() / 10.0) if keep.any() else float("inf")


def write_milestones(medoids, topology_pdb, out_dir, relax=True, platform=None, traj_dir=None, system=None,
                     ligand_atoms=None, pocket_atoms=None, return_info=False):
    """Write ``milestone_<class>_frame_<frame>.pdb`` (and relaxed states) for each medoid.

    The aligned DCD frame is rotated back to its lab orientation by a brick fit of its waters, and every
    molecule is imaged whole into the frame's own box, which is also the PDB box and the System box. The
    smallest inter-molecular heavy-atom distance must exceed 0.15 nm. With relax=True RelaxMD then writes
    ``<name>_relax_state.xml`` and ``<name>_relax_system.xml`` and the potential energy of the relaxed
    state must be finite.

    Parameters
    ----------
    medoids : output of class_medoid_frames
    topology_pdb : PDB of the simulated system (atoms, residues, bonds)
    out_dir : output directory
    relax : run RelaxMD
    platform : OpenMM platform name for relaxation (None = fastest)
    traj_dir : directory of the DCDs named in the medoids
    system : openmm.System or system.xml path matching topology_pdb; required when relax
    ligand_atoms, pocket_atoms : atom indices for RelaxMD; required when relax
    return_info : also return one dict per milestone (angle_deg, residual_nm, dmin_nm, box)

    Returns
    -------
    list of milestone names (``milestone_<class>_frame_<frame>``); (names, infos) with return_info
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_vectors
    from openmm import app, unit

    from .imaging import anchor_atoms
    from .water import WATER_RES

    if relax and (system is None or ligand_atoms is None or pocket_atoms is None):
        raise ValueError("relax=True needs system, ligand_atoms and pocket_atoms")
    top = app.PDBFile(topology_pdb).topology
    mol, edges = _graph(top)
    heavy = np.array([a.element is not None and a.element.symbol != "H" for a in top.atoms()])
    u0 = mda.Universe(topology_pdb)
    wres = [r.atoms.indices for r in u0.residues if r.resname in WATER_RES]
    if not wres or len({len(w) for w in wres}) > 1:
        raise ValueError("topology needs water residues of one atom count; not supported")
    water_mols = np.array(wres, dtype=int)
    centre_idx = anchor_atoms(topology_pdb)
    os.makedirs(out_dir, exist_ok=True)
    names, infos = [], []
    for m in medoids:
        name = f"milestone_{m['class']}_frame_{m['frame']}"
        dcd = m["dcd"] if os.path.isabs(m["dcd"]) or traj_dir is None else os.path.join(traj_dir, m["dcd"])
        u = mda.Universe(topology_pdb, dcd)
        ts = u.trajectory[m["frame"]]
        if ts.dimensions is None or not np.all(ts.dimensions[:3] > 0):
            raise ValueError(f"{name}: DCD frame has no box; not supported")
        box = triclinic_vectors(ts.dimensions).astype(float) / 10.0
        Y, info = restore_frame(np.array(ts.positions, dtype=float) / 10.0, box, centre_idx, water_mols, mol, edges)
        dmin = min_intermolecular_nm(Y, box, mol, heavy)
        if dmin <= MIN_CONTACT_NM:
            raise ValueError(f"{name}: minimum inter-molecular heavy-atom distance {dmin:.3f} nm <= "
                             f"{MIN_CONTACT_NM} nm after re-imaging into the frame's box")
        info.update(dmin_nm=dmin, box=box.tolist(), name=name)
        logger.info("%s: brick fit %.2f deg, residual %.4f nm, dmin %.3f nm", name, info["angle_deg"],
                    info["residual_nm"], dmin)
        top.setPeriodicBoxVectors(box)
        pdb = os.path.join(out_dir, name + ".pdb")
        with open(pdb, "w") as fh:
            app.PDBFile.writeFile(top, Y * unit.nanometer, fh, keepIds=True)
        if relax:
            _relax(top, pdb, name, out_dir, platform, system, ligand_atoms, pocket_atoms, box)
        names.append(name)
        infos.append(info)
    return (names, infos) if return_info else names


def _relax(top, pdb, name, out_dir, platform, system, ligand_atoms, pocket_atoms, box):
    import openmm as mm
    from openmm import app, unit

    from ...relax_md import RelaxMD
    from ...utils import select_platform

    if isinstance(system, (str, os.PathLike)):
        with open(system) as fh:
            system = mm.XmlSerializer.deserialize(fh.read())
    else:
        system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in box])
    RelaxMD(topology=top, ligand_atoms=list(ligand_atoms), pocket_atoms=list(pocket_atoms), out_dir=out_dir,
            platform=platform or "fastest").run(system=system, pdb_file=pdb, run_id=name)
    with open(os.path.join(out_dir, name + "_relax_system.xml")) as fh:
        rsys = mm.XmlSerializer.deserialize(fh.read())
    with open(os.path.join(out_dir, name + "_relax_state.xml")) as fh:
        st = mm.XmlSerializer.deserialize(fh.read())
    ctx = mm.Context(rsys, mm.VerletIntegrator(0.001), select_platform(platform or "fastest"))
    ctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    ctx.setPositions(st.getPositions())
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    if not np.isfinite(e):
        raise ValueError(f"{name}: potential energy after relaxation is not finite ({e})")
