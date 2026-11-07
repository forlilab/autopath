import numpy as np
import pytest


def _nc_norm_reference(lig, pocket, thr=0.5):
    d = np.linalg.norm(lig[:, None, :] - pocket[None, :, :], axis=2)
    return float(np.sum(1.0 / (1.0 + (d / thr) ** 6)))


def test_compute_nc_squared_matches_norm():
    from autopath.pulling.steered_md import SteeredMD
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(40, 3))
    lig_idx = np.arange(0, 6)
    pocket_idx = np.arange(6, 25)
    smd = SteeredMD.__new__(SteeredMD)          # bypass __init__ (no OpenMM needed)
    smd.groupA_atoms = list(lig_idx)
    smd.subset_protein_CA = pocket_idx
    expected = _nc_norm_reference(positions[lig_idx], positions[pocket_idx], 0.5)
    got = smd._compute_nc(positions, threshold_nm=0.5)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-10)


SHAPE = ["rog", "npr1", "npr2", "eccentricity", "inertial_shape", "spherocity", "pbf", "asphericity"]


def test_shape_calculator_matches_rdkit():
    """Elements-only bond-less mol reproduces a fully bonded RDKit mol."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors as D
    from autopath.pulling.geom_kernel import ShapeDescriptorCalculator

    ref_vals = {
        "rog": lambda m: Descriptors3D.RadiusOfGyration(m) / 10.0,
        "npr1": Descriptors3D.NPR1, "npr2": Descriptors3D.NPR2,
        "eccentricity": Descriptors3D.Eccentricity,
        "inertial_shape": Descriptors3D.InertialShapeFactor,
        "spherocity": Descriptors3D.SpherocityIndex,
        "pbf": D.CalcPBF, "asphericity": Descriptors3D.Asphericity,
    }
    for smi in ["CCON(C)C(=O)c1ccccc1", "O=C(O)Cc1ccc(cc1)N", "c1ccccc1"]:
        for seed in (1, 7, 42):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
                continue
            m = Chem.RemoveHs(m)
            elements = [a.GetSymbol() for a in m.GetAtoms()]
            pos = m.GetConformer().GetPositions()
            calc = ShapeDescriptorCalculator(elements, SHAPE)
            got = calc.compute(pos)
            for k in SHAPE:
                ref = ref_vals[k](m)
                assert got[k] == pytest.approx(ref, abs=1e-5), f"{k} {smi} seed={seed}"


def test_shape_calculator_rog_in_nm():
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D
    from autopath.pulling.geom_kernel import ShapeDescriptorCalculator
    m = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    calc = ShapeDescriptorCalculator([a.GetSymbol() for a in m.GetAtoms()], ["rog"])
    got = calc.compute(m.GetConformer().GetPositions())
    assert got["rog"] == pytest.approx(Descriptors3D.RadiusOfGyration(m) / 10.0, abs=1e-6)


def test_shape_calculator_rejects_unknown():
    from autopath.pulling.geom_kernel import ShapeDescriptorCalculator
    with pytest.raises(ValueError):
        ShapeDescriptorCalculator(["C", "C"], ["not_a_descriptor"])
