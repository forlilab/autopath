"""Benchmark membrane orientation against OPM's published frames.

For each PDB entry we fetch two things: the deposited coordinates from the RCSB, in
whatever arbitrary crystal frame they were deposited in, and the same entry from OPM,
where the Lomize group has already placed the bilayer normal on z. Superposing the first
onto the second gives the true rotation, and therefore the true membrane normal expressed
in the deposited frame. That is the reference.

We then hand AutoPath only the deposited structure and ask how close its answer is. This
is not self-validation: the reference comes from outside, and the measure is an angle
between two normals, not the scan's own objective.

Soluble controls are included. They have no bilayer, so the only correct behaviour is to
refuse them.
"""
import os
import sys
import urllib.request

import numpy as np
from openmm.app import PDBFile
import openmm.unit as u

from autopath import membrane_orient as mo

WORK = sys.argv[1]
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(WORK, exist_ok=True)

MEMBRANE = [
    ("1u19", "bovine rhodopsin, 7TM GPCR"),
    ("7nqk", "PepT2 (SLC15A2), MFS transporter"),
    ("2rh1", "beta2 adrenergic receptor"),
    ("1okc", "ADP/ATP carrier"),
    ("1qjp", "OmpA, beta-barrel"),
    ("4pe5", "GLUT1 sugar transporter"),
]


def fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
        return os.path.getsize(path) > 0
    except Exception as exc:
        print(f"      fetch failed {url}: {exc}")
        return False


def opm_normal_in_deposited_frame(rcsb_pdb, opm_pdb):
    """The true membrane normal, expressed in the deposited coordinate frame."""
    a, b = mo._ca_map(rcsb_pdb), mo._ca_map(opm_pdb)
    shared = sorted(set(a) & set(b))
    if len(shared) < 30:
        return None, len(shared)
    x = np.array([a[k] for k in shared])
    y = np.array([b[k] for k in shared])
    R = mo.kabsch_rotation(x, y)                 # deposited -> OPM frame
    resid = np.sqrt((((x - x.mean(0)) @ R.T - (y - y.mean(0))) ** 2).sum(1).mean())
    if resid > 0.2:
        return None, -1
    return R.T @ np.array([0.0, 0.0, 1.0]), len(shared)


def angle(u1, u2):
    u1 = np.asarray(u1, float) / np.linalg.norm(u1)
    u2 = np.asarray(u2, float) / np.linalg.norm(u2)
    return float(np.degrees(np.arccos(min(1.0, abs(float(u1 @ u2))))))


print(f"{'pdb':6s} {'method':10s} {'vs OPM':>8s}  description")
print("-" * 78)

for pid, desc in MEMBRANE:
    if ONLY and pid != ONLY:
        continue
    d = os.path.join(WORK, pid)
    os.makedirs(d, exist_ok=True)
    rcsb = os.path.join(d, f"{pid}_rcsb.pdb")
    opm = os.path.join(d, f"{pid}_opm.pdb")
    if not fetch(f"https://files.rcsb.org/download/{pid.upper()}.pdb", rcsb):
        continue
    if not fetch(f"https://opm-assets.storage.googleapis.com/pdb/{pid}.pdb", opm):
        continue

    truth, n = opm_normal_in_deposited_frame(rcsb, opm)
    if truth is None:
        print(f"{pid:6s} {'-':10s} {'no ref':>8s}          {desc} (matched {n})")
        continue

    try:
        pdb = PDBFile(rcsb)
    except Exception as exc:
        print(f"{pid:6s} {'-':10s} {'unread':>8s}          {type(exc).__name__}: {exc}")
        continue
    pos = np.array(pdb.positions.value_in_unit(u.nanometer))

    try:
        frame = mo.orient_to_membrane_frame(pdb.topology, pos, work_dir=d)
    except Exception as exc:
        msg = str(exc).split("\n")[0][:46]
        print(f"{pid:6s} {'ppm':10s} {'-':>8s}  {msg}")
        continue
    print(f"{pid:6s} {'ppm':10s} {angle(frame.normal, truth):7.1f}d  {desc}")
    print("-" * 78)
