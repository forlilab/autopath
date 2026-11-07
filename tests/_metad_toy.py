"""One particle in a 1D corrugated well, for short metadynamics runs."""
import numpy as np
import openmm as mm
import openmm.unit as u
from openmm.app import Element, PDBFile, Topology

from autopath.metadynamics import CVSpec


def toy(tmp_path):
    top = Topology()
    top.addAtom("C", Element.getBySymbol("C"), top.addResidue("UNK", top.addChain()))
    top.setPeriodicBoxVectors(np.eye(3) * 3.0 * u.nanometer)
    pdb = str(tmp_path / "one.pdb")
    with open(pdb, "w") as fh:
        PDBFile.writeFile(top, [[1.0, 1.5, 1.5]] * u.nanometer, fh)
    s = mm.System()
    s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0))
    f = mm.CustomExternalForce("6*cos(2*3.14159*x/0.45) + 0.5*41840*(step(0.6-x)*(0.6-x)^2"
                               "+ step(x-1.5)*(x-1.5)^2) + 1000*((y-1.5)^2+(z-1.5)^2)")
    f.addParticle(0, [])
    s.addForce(f)
    return top, s, pdb


def x_spec():
    cv = mm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return CVSpec(cv=cv, grid_min=0.5, grid_max=1.6, hill_width=0.03, grid_points=111, name="x")
