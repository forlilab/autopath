#!/usr/bin/env python
"""One metadynamics walker for a solute permeating a membrane.

CVs: the signed z of the solute centroid from the membrane centre (``membrane_z_cv``) and,
with ``--axis-atoms``, the cosine of a solute bond to the membrane normal (``axis_cos_cv``).
Tempering: ``wt`` (well-tempered), ``tt`` (transition-tempered) or ``ttwt`` (both).
Layout: walkers of one run directory share one bias (``shared``) or each has its own
(``isolated``). Outputs are MetadynamicsMD's (HILLS/COLVAR/REWEIGHT per walker) plus
``walker_<k>.json`` with every setting.

The system must already hold whatever keeps the solute in its column (e.g. a cylinder
restraint) and a membrane-centre reference (e.g. a fixed dummy atom).

Example (walker 1 of a shared WT run):
    python run_permeation.py --system system.xml --pdb system.pdb --run-dir runs/wt_shared/r1 \\
        --walker 1 --method wt --layout shared --axis-atoms O1x C2x --ns 100
"""
import argparse
import json
import os
import sys

BASIN_PRESETS = {                       # |z| points (nm) of the TT basins; cos θ = 0 is added with --axis-atoms
    "symmetric": [0.0, 1.3, -1.3],      # centre + both outer basins
    "paper": [0.0, 1.3],                # Sun et al. 2016 "good basins" (one side)
    "leaflet": [1.4, -1.4],
    "bulk": [3.0, -3.0],
}
DEFAULT_GAMMA = {"wt": 11.0, "tt": 5.0, "ttwt": 11.0}     # Sun et al. 2016 (ethanol / POPC)


def parse_basins(text, with_axis):
    """Preset name or space-separated points, e.g. "0,0 1.3,0" (z[,cos])."""
    if text in BASIN_PRESETS:
        return [[z, 0.0] if with_axis else [z] for z in BASIN_PRESETS[text]]
    pts = [[float(x) for x in p.split(",")] for p in text.split()]
    if any(len(p) != (2 if with_axis else 1) for p in pts):
        sys.exit(f"basins {text!r}: need {'z,cos' if with_axis else 'z'} per point")
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("system")
    io.add_argument("--system", required=True, help="serialized OpenMM System (.xml), barostat included")
    io.add_argument("--pdb", required=True, help="topology + starting positions/box")
    io.add_argument("--state", help="OpenMM State XML to start from instead of the PDB positions")
    io.add_argument("--solute-resname", default="UNK")
    io.add_argument("--center-resname", default="DUM", help="membrane-centre reference residue")
    cv2 = io.add_mutually_exclusive_group()
    cv2.add_argument("--axis-atoms", nargs=2, metavar=("TAIL", "HEAD"),
                     help="second CV: cos of this solute atom pair to the normal (omit all three for z only)")
    cv2.add_argument("--end-to-end", action="store_true",
                     help="second CV: cos of the solute's two farthest heavy atoms to the normal (needs --ligand-sdf)")
    cv2.add_argument("--lmz", action="store_true",
                     help="second CV: z component of the unit lipophilicity moment (needs --ligand-sdf)")
    io.add_argument("--ligand-sdf", help="solute SDF in the topology's atom order (for --end-to-end / --lmz)")
    io.add_argument("--headgroup-atom",
                    help="lipid atom name (e.g. P) whose centroid z is logged as monitor_bilayer_z to catch "
                         "bilayer drift; off by default (an extra CV force, ~25%% slower)")
    io.add_argument("--temperature", type=float, default=310.0)
    io.add_argument("--timestep-fs", type=float, default=4.0)

    run = ap.add_argument_group("run")
    run.add_argument("--run-dir", required=True, help="one run: all its walkers write here")
    run.add_argument("--walker", type=int, required=True, help="walker number within the run (1, 2, ...)")
    run.add_argument("--layout", choices=["shared", "isolated"], default="shared")
    run.add_argument("--ns", type=float, default=100.0)
    run.add_argument("--dcd-ps", type=float, default=100.0, help="trajectory / log interval")

    md = ap.add_argument_group("metadynamics")
    md.add_argument("--method", choices=["wt", "tt", "ttwt"], required=True)
    md.add_argument("--gamma", type=float, help="WT bias factor (wt, ttwt) or TT factor (tt); default 11 / 5")
    md.add_argument("--tt-gamma", type=float, default=5.0, help="TT factor on top of WT (ttwt)")
    md.add_argument("--basins", default="symmetric",
                    help=f"TT basins: preset {sorted(BASIN_PRESETS)} or points 'z,cos z,cos' (tt, ttwt)")
    md.add_argument("--hill-kj", type=float, default=0.02)
    md.add_argument("--hill-ps", type=int, default=1, help="hill pace (whole ps)")
    md.add_argument("--z-sigma", type=float, default=0.2, help="z hill width (nm)")
    md.add_argument("--cos-sigma", type=float, default=0.1, help="second-CV hill width (cos θ or LMZ)")
    md.add_argument("--half-width", type=float, help="z grid half-width (nm); default half the starting box height")
    a = ap.parse_args(argv)
    if (a.end_to_end or a.lmz) and not a.ligand_sdf:
        ap.error("--end-to-end and --lmz need --ligand-sdf")
    with_cv2 = bool(a.axis_atoms or a.end_to_end or a.lmz)

    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile
    from autopath.metadynamics import (MetadynamicsMD, axis_cos_cv, crippen_contributions, end_to_end_atoms,
                                       lipophilicity_z_cv, membrane_z_cv)

    pdb = PDBFile(a.pdb)
    top = pdb.topology
    atoms = list(top.atoms())
    solute = [x.index for x in atoms if x.residue.name == a.solute_resname]
    center = [x.index for x in atoms if x.residue.name == a.center_resname]
    heads = [x.index for x in atoms if a.headgroup_atom and x.name == a.headgroup_atom
             and x.residue.name not in (a.solute_resname, a.center_resname)]
    if a.headgroup_atom and not heads:
        sys.exit(f"no {a.headgroup_atom} atoms outside the solute and centre")
    if not solute or not center:
        sys.exit(f"no {a.solute_resname} or {a.center_resname} atoms in {a.pdb}")
    hw = a.half_width or 0.5 * top.getUnitCellDimensions()[2].value_in_unit(unit.nanometer)

    cvs = [membrane_z_cv(solute, center, hw, hill_width=a.z_sigma)]
    if a.ligand_sdf:
        from rdkit import Chem
        n_sdf = Chem.MolFromMolFile(a.ligand_sdf, removeHs=False).GetNumAtoms()
        if n_sdf != len(solute):
            sys.exit(f"{a.ligand_sdf} has {n_sdf} atoms, {a.solute_resname} has {len(solute)}")
    if a.axis_atoms:
        names = {x.name: x.index for x in atoms if x.residue.name == a.solute_resname}
        missing = [n for n in a.axis_atoms if n not in names]
        if missing:
            sys.exit(f"axis atoms {missing} not in {a.solute_resname}")
        cvs.append(axis_cos_cv(names[a.axis_atoms[0]], names[a.axis_atoms[1]], hill_width=a.cos_sigma))
    elif a.end_to_end:
        i, j = end_to_end_atoms(a.ligand_sdf)
        a.axis_atoms = [atoms[solute[i]].name, atoms[solute[j]].name]       # recorded in walker_<k>.json
        cvs.append(axis_cos_cv(solute[i], solute[j], hill_width=a.cos_sigma))
    elif a.lmz:
        cvs.append(lipophilicity_z_cv(solute, crippen_contributions(a.ligand_sdf), hill_width=a.cos_sigma))
    monitors = {"bilayer_z": membrane_z_cv(heads, center, hw, name="bilayer_z")} if heads else None

    gamma = a.gamma if a.gamma is not None else DEFAULT_GAMMA[a.method]
    basins = parse_basins(a.basins, with_cv2) if a.method != "wt" else None
    tt_gamma = a.tt_gamma if a.method == "ttwt" else None

    os.makedirs(a.run_dir, exist_ok=True)
    bias_dir = os.path.join(a.run_dir, "bias" if a.layout == "shared" else f"bias_w{a.walker}")
    run_id = f"w{a.walker}"
    settings = dict(vars(a), gamma=gamma, tt_gamma=tt_gamma, basins=basins, half_width=hw, bias_dir=bias_dir,
                    run_id=run_id, n_solute=len(solute), n_headgroup=len(heads))
    with open(os.path.join(a.run_dir, f"walker_{a.walker}.json"), "w") as f:
        json.dump(settings, f, indent=1)

    with open(a.system) as f:
        system = XmlSerializer.deserialize(f.read())
    mmd = MetadynamicsMD(topology=top, ligand_atoms=solute, pocket_atoms=center, is_membrane=True,
                         timestep=a.timestep_fs / 1000.0, temp=a.temperature, out_dir=a.run_dir)
    mmd.run(pdb_file=a.pdb, state_xml_file=a.state, system=system, run_id=run_id, cv_specs=cvs,
            mMD_time=a.ns, bias_factor=gamma, hill_height=a.hill_kj, biasFrequency=a.hill_ps,
            bias_dir=bias_dir, boundary_correction=True, reweight=True, rct_ustride=10,
            monitor_cvs=monitors, dcd_interval=a.dcd_ps, tt_basins=basins, tt_bias_factor=tt_gamma,
            run_meta=dict(method=a.method, layout=a.layout))
    print(f"done: {a.run_dir} walker {a.walker} ({a.method}, {a.layout}, {a.ns} ns)")


if __name__ == "__main__":
    main()
