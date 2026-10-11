# autopath/metadynamics/learned/milestones.py
"""Class-medoid milestone frames of the learned CV and relaxed simulation-ready states for them."""
import os

import numpy as np
from scipy.spatial.distance import cdist

MAX_PER_CLASS = 2000
MIN_CONTACT_NM = 0.15


def class_medoid_frames(features_npz, learned_npz, order=("B", "TS", "U")) -> list:
    """Medoid frame of each class in the standardised kept-feature + z/sigma_z space.

    Only rows with ``valid`` set are used. The medoid minimises the summed Euclidean distance to the
    class frames (at most 2000 per class, subsampled with seed 0).

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
    rng = np.random.default_rng(0)
    for c in order:
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


def kabsch(P, Q):
    """Rotation R and translation so that ``(x - P.mean(0)) @ R + Q.mean(0)`` best maps P onto Q."""
    p0, q0 = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - p0).T @ (Q - q0))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R, p0, q0


def restore_frame(X, ref, ca, box, mol, edges):
    """Frame X (aligned, nm) rotated back onto ref, bonds made whole, molecules imaged into the box.

    Parameters
    ----------
    X : (N, 3) coordinates of the aligned frame, nm
    ref : (N, 3) topology_pdb coordinates, nm
    ca : protein CA atom indices used for the fit
    box : (3, 3) box vectors of topology_pdb (rows), nm
    mol : (N,) molecule id per atom
    edges : (E, 2) atom pairs the molecules are made whole along

    Returns
    -------
    (N, 3) coordinates, nm
    """
    R, p0, q0 = kabsch(X[ca], ref[ca])
    Y = (X - p0) @ R + q0
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
    return Y


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
                     ligand_atoms=None, pocket_atoms=None) -> list:
    """Write ``milestone_<class>_frame_<frame>.pdb`` (and relaxed states) for each medoid.

    The aligned DCD frame is rotated back onto topology_pdb (protein CA fit), given the topology_pdb box,
    and every molecule is imaged whole into that box. The smallest inter-molecular heavy-atom distance
    must exceed 0.15 nm. With relax=True RelaxMD then writes ``<name>_relax_state.xml`` and
    ``<name>_relax_system.xml`` and the potential energy of the relaxed state must be finite.

    Parameters
    ----------
    medoids : output of class_medoid_frames
    topology_pdb : PDB of the simulated system (topology, reference orientation and box)
    out_dir : output directory
    relax : run RelaxMD
    platform : OpenMM platform name for relaxation (None = fastest)
    traj_dir : directory of the DCDs named in the medoids
    system : openmm.System or system.xml path matching topology_pdb; required when relax
    ligand_atoms, pocket_atoms : atom indices for RelaxMD; required when relax

    Returns
    -------
    list of milestone names (``milestone_<class>_frame_<frame>``)
    """
    import MDAnalysis as mda
    from openmm import app, unit

    pdbf = app.PDBFile(topology_pdb)
    top, ref = pdbf.topology, np.array(pdbf.positions.value_in_unit(unit.nanometer))
    box = top.getPeriodicBoxVectors()
    if box is None:
        raise ValueError("topology_pdb has no CRYST1 box; not supported")
    box = np.array(box.value_in_unit(unit.nanometer))
    if relax and (system is None or ligand_atoms is None or pocket_atoms is None):
        raise ValueError("relax=True needs system, ligand_atoms and pocket_atoms")
    mol, edges = _graph(top)
    heavy = np.array([a.element is not None and a.element.symbol != "H" for a in top.atoms()])
    u0 = mda.Universe(topology_pdb)
    ca = u0.select_atoms("protein and name CA").indices
    if len(ca) < 3:
        raise ValueError("fewer than 3 protein CA atoms to rotate the frame back; not supported")
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for m in medoids:
        name = f"milestone_{m['class']}_frame_{m['frame']}"
        dcd = m["dcd"] if os.path.isabs(m["dcd"]) or traj_dir is None else os.path.join(traj_dir, m["dcd"])
        u = mda.Universe(topology_pdb, dcd)
        X = np.array(u.trajectory[m["frame"]].positions, dtype=float) / 10.0
        Y = restore_frame(X, ref, ca, box, mol, edges)
        dmin = min_intermolecular_nm(Y, box, mol, heavy)
        if dmin <= MIN_CONTACT_NM:
            raise ValueError(f"{name}: minimum inter-molecular heavy-atom distance {dmin:.3f} nm <= "
                             f"{MIN_CONTACT_NM} nm after re-imaging into the topology box")
        top.setPeriodicBoxVectors(box)
        pdb = os.path.join(out_dir, name + ".pdb")
        with open(pdb, "w") as fh:
            app.PDBFile.writeFile(top, Y * unit.nanometer, fh, keepIds=True)
        if relax:
            _relax(top, pdb, name, out_dir, platform, system, ligand_atoms, pocket_atoms, box)
        names.append(name)
    return names


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
