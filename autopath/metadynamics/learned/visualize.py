"""Headless PyMOL session showing which learned-CV features were kept."""
import re
from collections import OrderedDict

import numpy as np

from .features import CN_DMAX, CN_R0, features_from_dict

WATER_RESN = "HOH+WAT+SOL+TIP3+TIP4+T3P"
POCKET_CA_NM = 0.6
KEPT_RGB = {"pos": (0.9, 0.1, 0.1), "neg": (0.1, 0.2, 0.9), "other": (0.95, 0.6, 0.1)}
GREY = (0.6, 0.6, 0.6)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def _sel(ids, obj):
    ids = sorted({int(i) for i in ids})
    return f"({obj} and rank {'+'.join(map(str, ids))})"


def _centres(f):
    """Atoms of f that are drawn (cn: the group_a atoms of every term)."""
    if f.kind == "cn":
        return sorted({i for a, _, _ in f.terms for i in a})
    if f.kind == "pair":
        return sorted({i for p in f.pairs for i in p})
    if f.kind == "contacts":
        return sorted(set(f.group_a) | set(f.group_b))
    if f.kind in ("centroid_distance", "cos"):
        return sorted({i for g in f.groups for i in g})
    if f.kind == "centroid_count":
        return sorted(set(f.groups[0]) | set(f.group_b))
    raise ValueError(f"feature {f.name!r}: kind {f.kind!r} is not supported by the viewer")


def _arrow(p, q, rgb, radius=0.15):
    from pymol.cgo import COLOR, CONE, CYLINDER
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    n = np.linalg.norm(d)
    tip = p + d * max(n - 0.8, 0.0) / n
    rgb = [float(x) for x in rgb]
    return [COLOR, *rgb, CYLINDER, *map(float, p), *map(float, tip), radius, *rgb, *rgb,
            CONE, *map(float, tip), *map(float, q), 3 * radius, 0.0, *rgb, *rgb, 1.0, 1.0]


def _draw(cmd, f, weight, wmax, ref, X, rgb, kept):
    """Draw feature f on the reference object; returns the created object names."""
    nm = _safe(f.name if kept else f"{f.name}_dropped")
    col = f"col_{nm}"
    cmd.set_color(col, [float(x) for x in rgb])
    objs = []
    if f.kind == "cn":
        cmd.create(nm, _sel(_centres(f), ref))
        cmd.hide("everything", nm)
        cmd.show("spheres", nm)
        scale = 0.6 if wmax == 0 else 0.25 + 0.75 * abs(weight) / wmax
        cmd.set("sphere_scale", float(scale if kept else 0.25), nm)
        cmd.color(col, nm)
        if kept:
            cmd.label(f"first {nm}", f'"{f.name} {weight:+.2f}"')
        objs.append(nm)
    elif f.kind == "pair":
        for k, (i, j) in enumerate(f.pairs):
            cmd.distance(nm, _sel([i], ref), _sel([j], ref))
        cmd.hide("labels", nm)
        cmd.color(col, nm)
        objs.append(nm)
    elif f.kind == "contacts":
        cmd.create(nm, _sel(_centres(f), ref))
        cmd.hide("everything", nm)
        cmd.show("sticks", nm)
        cmd.color(col, nm)
        objs.append(nm)
    elif f.kind in ("centroid_distance", "centroid_count"):
        grp = f.groups if f.kind == "centroid_distance" else f.groups[:1]
        names = []
        for k, g in enumerate(grp):
            cmd.pseudoatom(f"{nm}_c{k}", pos=[float(x) for x in X[g].mean(0)])
            cmd.show("spheres", f"{nm}_c{k}")
            cmd.color(col, f"{nm}_c{k}")
            names.append(f"{nm}_c{k}")
        if f.kind == "centroid_distance":
            cmd.distance(f"{nm}_d", names[0], names[1])
            cmd.hide("labels", f"{nm}_d")
            cmd.color(col, f"{nm}_d")
            names.append(f"{nm}_d")
        else:
            cmd.create(f"{nm}_b", _sel(f.group_b, ref))
            cmd.hide("everything", f"{nm}_b")
            cmd.show("sticks", f"{nm}_b")
            cmd.color(col, f"{nm}_b")
            names.append(f"{nm}_b")
        objs += names
    elif f.kind == "cos":
        c = [X[g].mean(0) for g in f.groups]
        cgo = _arrow(c[0], c[1], rgb) + _arrow(c[2], c[3], [0.5 * x for x in rgb])
        cmd.load_cgo(cgo, nm)
        objs.append(nm)
    else:
        raise ValueError(f"feature {f.name!r}: kind {f.kind!r} is not supported by the viewer")
    return objs


def _funnel_cgo_obj(cmd, funnel, X, ref):
    from openmm import unit as openmmunit
    from ..Funnel import _funnel_cgo, funnel_lower_wall_ang

    axis = np.asarray(funnel["unbinding_axis"], float)
    host = list(funnel.get("host_index", []))
    if host:
        cmd.create("funnel_host_tmp", _sel(host, ref))
        com = np.asarray(cmd.centerofmass("funnel_host_tmp"), float)
        cmd.delete("funnel_host_tmp")
    else:
        com = np.asarray(cmd.centerofmass(f"{ref} and polymer.protein"), float)
    ang = openmmunit.angstrom
    cgo = _funnel_cgo(com, axis, funnel_lower_wall_ang(funnel), funnel["z_cc"].value_in_unit(ang),
                      funnel["z_max"].value_in_unit(ang), funnel["R_cylinder"].value_in_unit(ang),
                      funnel["alpha"].value_in_unit(openmmunit.degrees))
    cmd.load_cgo(cgo, "funnel")


def write_feature_pse(learned_npz, structures, out_pse, funnel=None, waters=True, water_cut_nm=None) -> str:
    """Write a PyMOL session of the kept learned-CV features.

    Parameters
    ----------
    learned_npz : str
        learned_cv.npz (features_json, kept, names, sw_weights or weights).
    structures : dict
        Ordered {name: pdb path}; the first is the reference, the rest are aligned on its pocket C-alpha
        atoms. All structures must share the atom order of the reference.
    out_pse : str
        Output session path.
    funnel : dict, optional
        Funnel parameters as used by ``write_funnel_pymol``; drawn in the reference frame.
    waters : bool
        Add per-structure water oxygens within ``water_cut_nm`` of any kept cn centre.
    water_cut_nm : float, optional
        Water cutoff (nm), default CN_DMAX; no periodic imaging is applied.

    Returns
    -------
    str
        out_pse.
    """
    import pymol2

    structures = OrderedDict(structures)
    if not structures:
        raise ValueError("structures is empty")
    cut = CN_DMAX if water_cut_nm is None else float(water_cut_nm)
    with np.load(learned_npz, allow_pickle=False) as d:
        feats = features_from_dict(d)
        kept = [int(i) for i in d["kept"]]
        w = np.asarray(d["sw_weights"] if "sw_weights" in d.files else d["weights"][:, 0], float)
    if len(w) != len(kept):
        raise ValueError(f"{len(w)} weights for {len(kept)} kept features")
    kf = [feats[i] for i in kept]
    wmax = float(np.abs(w).max()) if len(w) else 0.0
    fams = list(OrderedDict.fromkeys(f.family for f in kf))
    centres = sorted({i for f in kf if f.kind == "cn" for i in _centres(f)})
    pocket_ids = sorted({i for f in kf for i in _centres(f)})

    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.feedback("disable", "all", "actions results")
        names = list(structures)
        for n in names:
            cmd.load(structures[n], f"state_{_safe(n)}")
        ref = f"state_{_safe(names[0])}"
        nat = cmd.count_atoms(ref)
        for n in names[1:]:
            if cmd.count_atoms(f"state_{_safe(n)}") != nat:
                raise ValueError(f"structure {n!r} has a different atom count from {names[0]!r}")
        if feats and max(i for f in feats for i in _centres(f)) >= nat:
            raise ValueError("feature atom index beyond the reference structure")
        X = cmd.get_coords(ref)

        # ligand: residues of the kept W1 centres, else resname UNK
        lig_ids = sorted({i for f in kf if f.family == "W1" for i in _centres(f)})
        lig_sel = f"byres {_sel(lig_ids, ref)}" if lig_ids else f"({ref} and resname UNK)"
        if not cmd.count_atoms(lig_sel):
            raise ValueError("no ligand found in the reference structure")
        pocket = f"({ref} and name CA and polymer.protein and (({lig_sel}) around {POCKET_CA_NM * 10}))"
        if len(names) > 1:
            rank_ids = []
            cmd.iterate(pocket, "rank_ids.append(rank)", space={"rank_ids": rank_ids})
            if len(rank_ids) < 3:
                raise ValueError("fewer than 3 pocket C-alpha atoms within the alignment cutoff of the ligand")
            for n in names[1:]:
                o = f"state_{_safe(n)}"
                cmd.pair_fit(_sel(rank_ids, o), _sel(rank_ids, ref))

        for n in names:
            o = f"state_{_safe(n)}"
            cmd.hide("everything", o)
            cmd.show("cartoon", f"{o} and polymer.protein")
            cmd.color("grey80", f"{o} and polymer.protein")
            cmd.show("sticks", f"byres {_sel(lig_ids, o)}" if lig_ids else f"{o} and resname UNK")
            if pocket_ids:
                cmd.show("lines", f"byres {_sel(pocket_ids, o)} and not resn {WATER_RESN}")

        # kept features by family; dropped candidates of the same families in grey
        kept_set = set(kept)
        for fam in fams:
            members = []
            for f, wt in zip(kf, w):
                if f.family != fam:
                    continue
                rgb = KEPT_RGB["other"] if f.kind != "cn" else KEPT_RGB["pos" if wt > 0 else "neg"]
                members += _draw(cmd, f, float(wt), wmax, ref, X, rgb, True)
            cmd.group(fam, " ".join(members))
            dropped = []
            for i, f in enumerate(feats):
                if f.family == fam and i not in kept_set:
                    dropped += _draw(cmd, f, 0.0, wmax, ref, X, GREY, False)
            if dropped:
                cmd.group(f"{fam}_dropped", " ".join(dropped))
                cmd.disable(" ".join(dropped + [f"{fam}_dropped"]))

        if centres:
            from pymol.cgo import ALPHA, COLOR, SPHERE
            cgo = [ALPHA, 0.15]
            for i in centres:
                cgo += [COLOR, 0.9, 0.9, 0.2, SPHERE, *map(float, X[i]), CN_R0 * 10.0]
            cmd.load_cgo(cgo, "shell_cgo")
            cmd.group("shells", "shell_cgo")
            cmd.disable("shells shell_cgo")

        wat = []
        if waters and centres:
            for n in names:
                o = f"state_{_safe(n)}"
                ids = []
                cmd.iterate(f"({o} and resn {WATER_RESN} and name O+OW)", "ids.append(rank)", space={"ids": ids})
                Y = cmd.get_coords(o)
                ids = np.asarray(ids, int)
                if len(ids):
                    dist = np.linalg.norm(Y[ids][:, None] - Y[centres][None], axis=-1).min(1)
                    ids = ids[dist <= cut * 10.0]
                wo = f"waters_{_safe(n)}"
                if len(ids):
                    cmd.create(wo, _sel(ids, o))
                else:
                    cmd.create(wo, f"({o} and none)")
                cmd.hide("everything", wo)
                cmd.show("spheres", wo)
                cmd.set("sphere_scale", 0.3, wo)
                cmd.color("tv_blue", wo)
                wat.append(wo)

        if funnel is not None:
            _funnel_cgo_obj(cmd, funnel, X, ref)

        states = [f"state_{_safe(n)}" for n in names]
        for k, n in enumerate(names):
            for o in states + wat:
                cmd.disable(o)
            cmd.enable(states[k])
            if wat:
                cmd.enable(f"waters_{_safe(n)}")
            cmd.zoom(ref, 5)
            cmd.scene(f"scene_{_safe(n)}", "store")
        cmd.save(out_pse)
    return out_pse
