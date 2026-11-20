#!/usr/bin/env python
"""Equilibrated starts for ``run_permeation.py --start-dirs``, from ``prepare_permeation_system.py --both-sides``.

Per side (+: system.pdb, -: system_minus.pdb), on one GPU:
  1. equilibrate the build (equilibrate_ligand_membrane.json, 310 K)          -> equilibration_+-OUTER/
  2. steered MD: the ligand end atom farthest from its centroid pulled toward DUM
     (SteeredMD, backward, 1 nm/ns, k = 1000 kJ/mol/nm^2; NPT)               -> smd_output_+-INNER/
     The pull stops when the end atom reaches the centre, so long ligands may not bring their
     centroid to +-INNER: the closest frame is used, with a warning.
  3. the frame whose ligand centroid z is closest to +-INNER, box from that frame,
     re-equilibrated                                                          -> equilibration_+-INNER/
Folders sit next to BUILD (default; or in --out). Each equilibration folder holds Equilibration's
output plus the start files run_permeation reads: system.xml (with barostat), system.pdb, state.xml.
The sides are independent: --side plus / --side minus run them as two parallel jobs.
Once all four starts exist they are checked to share one topology.

Example:
    python make_starts.py ethanol_replica_1/md_input                # both sides
    python make_starts.py ethanol_replica_1/md_input --side plus    # equilibration_2.7, _1.0 only
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.join(HERE, "equilibrate_ligand_membrane.json")


def equilibrate(build, pdb_path, out_dir, protocol):
    """Equilibrate the prepared (barostat-free) system from pdb_path, then write the start files in out_dir."""
    from openmm import XmlSerializer
    from openmm.app import PDBFile
    from autopath import Equilibration
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(build, "system.prmtop"), os.path.splitext(pdb_path)[0] + ".prmtop")  # selections only
    pdb = PDBFile(pdb_path)
    with open(os.path.join(build, "system.xml")) as f:
        system = XmlSerializer.deserialize(f.read())
    system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())   # post-SMD frames: their box
    Equilibration(topology=pdb.topology, system=system, out_dir=out_dir, protocol_fname=protocol,
                  is_membrane=True, verbose=2).run(pdb_file=pdb_path, run_id="sys")
    for src, dst in [("system_equil_sys.xml", "system.xml"), ("sys_equilibrated.pdb", "system.pdb"),
                     ("checkpoint_equil_sys.xml", "state.xml")]:
        shutil.copy(os.path.join(out_dir, src), os.path.join(out_dir, dst))


def pull_to(eq_dir, sdf, target_z, smd_dir):
    """Steered MD from an equilibrated start; writes the frame closest to target_z, returns its path."""
    import MDAnalysis as mda
    import numpy as np
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile
    from autopath.metadynamics import end_to_end_atoms
    from autopath.pulling.steered_md import SteeredMD
    from autopath.utils import save_pdb

    pdb = PDBFile(os.path.join(eq_dir, "system.pdb"))
    top = pdb.topology
    solute = [x.index for x in top.atoms() if x.residue.name == "UNK"]
    dum = [x.index for x in top.atoms() if x.residue.name == "DUM"]
    end = solute[end_to_end_atoms(sdf)[0]]                     # farthest from the centroid
    z_end = abs(pdb.positions[end].z - pdb.positions[dum[0]].z)   # nm (DUM sits at the origin)
    with open(os.path.join(eq_dir, "system.xml")) as f:
        system = XmlSerializer.deserialize(f.read())
    smd = SteeredMD(system=system, topology=top, groupA_atoms=[end], groupB_atoms=dum, temperature=310,
                    autostop_nc=None, out_dir=smd_dir, max_displacement=z_end + 0.25, dx_per_move=0.001,
                    sMD_spring_cte=1000, verbose=0)
    run_id = smd.run(pulling_speed=0.001, pulling_direction="backward", run_id="pull",
                     state_xml_file=os.path.join(eq_dir, "state.xml"))

    u = mda.Universe(os.path.join(eq_dir, "system.pdb"), os.path.join(smd_dir, f"sMD_{run_id}.dcd"))
    lig, ref = u.atoms[solute], u.atoms[dum]
    best, best_dz = None, np.inf
    for ts in u.trajectory:
        lz = ts.dimensions[2]
        dz = lig.positions[:, 2].mean() - ref.positions[0, 2]
        dz = (dz + lz / 2) % lz - lz / 2                         # minimum image, A
        if abs(dz / 10 - target_z) < best_dz:
            best, best_dz, best_z = ts.frame, abs(dz / 10 - target_z), dz / 10
    if best_dz > 0.1:
        print(f"WARNING: SMD reached no closer than {best_dz:.2f} nm to z = {target_z:+.2f}; "
              f"using the closest frame (centroid z = {best_z:+.2f} nm)", file=sys.stderr)
    u.trajectory[best]
    a, b, c = (u.dimensions[:3] / 10.0)
    top.setPeriodicBoxVectors([[a, 0, 0], [0, b, 0], [0, 0, c]] * unit.nanometer)
    frame = os.path.join(smd_dir, "smd_frame.pdb")
    save_pdb(top, (u.atoms.positions / 10.0) * unit.nanometer, frame)
    print(f"SMD frame {best}: ligand centroid z = {best_z:+.2f} nm (target {target_z:+.2f})")
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build", help="output folder of prepare_permeation_system.py --both-sides")
    ap.add_argument("--outer", type=float, default=2.7, help="label of the as-built depth (nm)")
    ap.add_argument("--inner", type=float, default=1.0, help="SMD target depth of the ligand centroid (nm)")
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--out", help="where the equilibration_*/smd_output_* folders go; default next to BUILD")
    ap.add_argument("--side", choices=["plus", "minus", "both"], default="both",
                    help="which side to run (plus and minus can run as parallel jobs)")
    a = ap.parse_args(argv)

    from autopath.metadynamics.permeation import check_same_topology
    build = os.path.abspath(a.build)
    out = os.path.abspath(a.out or os.path.dirname(build))
    sdf = os.path.join(build, "UNK.sdf")
    for sign, pdb_name in ((+1, "system.pdb"), (-1, "system_minus.pdb")):
        side = "plus" if sign > 0 else "minus"
        if a.side not in (side, "both"):
            continue
        outer = os.path.join(out, f"equilibration_{sign * a.outer:.1f}")
        inner = os.path.join(out, f"equilibration_{sign * a.inner:.1f}")
        os.makedirs(outer, exist_ok=True)
        src = os.path.join(outer, "input.pdb")             # a copy, so the prmtop lands next to it
        shutil.copy(os.path.join(build, pdb_name), src)
        equilibrate(build, src, outer, a.protocol)
        frame = pull_to(outer, sdf, sign * a.inner, os.path.join(out, f"smd_output_{sign * a.inner:.1f}"))
        equilibrate(build, frame, inner, a.protocol)
        print(f"{side} side done: {outer} {inner}")

    starts = [os.path.join(out, f"equilibration_{z:.1f}") for z in (a.outer, a.inner, -a.inner, -a.outer)]
    done = [d for d in starts if os.path.exists(os.path.join(d, "state.xml"))]
    if len(done) < len(starts):
        print(f"{len(done)}/4 starts so far; the topology check runs when the other side finishes")
        return
    check_same_topology([os.path.join(d, "system.pdb") for d in starts],
                        [os.path.join(d, "system.xml") for d in starts])
    print("starts (same topology):", " ".join(starts))


if __name__ == "__main__":
    main()
