"""Inline RDKit ligand shape descriptors for the sMD pull loop.

RDKit's 3D shape descriptors (PMI-based NPR/asphericity/eccentricity/…, and the
plane-of-best-fit) are purely geometric + mass, so they need neither bond orders
nor an SDF. We build a bond-less, unsanitized RDKit mol from the ligand heavy-atom
element symbols **once** (masses come from the elements), attach one conformer, and
mutate its coordinates per frame. Results are byte-for-byte identical to a fully
bonded mol and to the post-processing :mod:`LigandFeatures` path.
"""
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdMolDescriptors
from rdkit.Geometry import Point3D

# name -> callable taking an RDKit mol with a 3D conformer.
SHAPE_FN = {
    "asphericity":    Descriptors3D.Asphericity,
    "eccentricity":   Descriptors3D.Eccentricity,
    "inertial_shape": Descriptors3D.InertialShapeFactor,
    "npr1":           Descriptors3D.NPR1,
    "npr2":           Descriptors3D.NPR2,
    "spherocity":     Descriptors3D.SpherocityIndex,
    "pbf":            rdMolDescriptors.CalcPBF,
    "rog":            lambda m: Descriptors3D.RadiusOfGyration(m) / 10.0,  # Å -> nm
}
SHAPE_FEATURES = frozenset(SHAPE_FN)


class ShapeDescriptorCalculator:
    """Per-frame RDKit shape descriptors for a fixed ligand atom set.

    Parameters
    ----------
    elements : sequence of str
        Heavy-atom element symbols of the ligand, in the order coordinates
        will be supplied to :meth:`compute`.
    which : sequence of str
        Descriptor names to evaluate; subset of :data:`SHAPE_FEATURES`.
    """

    def __init__(self, elements: Sequence[str], which: Sequence[str]):
        which = list(which)
        unknown = [w for w in which if w not in SHAPE_FEATURES]
        if unknown:
            raise ValueError(
                f"Unsupported shape features: {sorted(unknown)}. "
                f"Supported: {sorted(SHAPE_FEATURES)}"
            )
        self.which = which
        rw = Chem.RWMol()
        for sym in elements:
            rw.AddAtom(Chem.Atom(sym))
        self.mol = rw.GetMol()
        self.mol.AddConformer(Chem.Conformer(self.mol.GetNumAtoms()), assignId=True)
        self._conf = self.mol.GetConformer()
        self._fns = [(name, SHAPE_FN[name]) for name in which]

    def compute(self, pos_ang) -> dict[str, float]:
        """Evaluate the requested descriptors for one frame.

        pos_ang : (N, 3) heavy-atom coordinates in Angstrom, same order/length
        as ``elements``. RoG is returned in nm; PBF is a mean distance in Å.
        """
        for i, xyz in enumerate(pos_ang):
            self._conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        return {name: float(fn(self.mol)) for name, fn in self._fns}
