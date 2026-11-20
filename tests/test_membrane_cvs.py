import numpy as np
import openmm as mm
import pytest
from openmm import unit

from autopath.metadynamics.CV import axis_cos_cv, membrane_z_cv

LZ = 7.0
SOLUTE, CENTER = [0, 1], [2]


def _context(spec, pos_nm, lz=LZ):
    s = mm.System()
    for _ in range(len(pos_nm)):
        s.addParticle(1.0)
    s.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, lz))
    cv = spec.resolve(pos_nm, len(pos_nm), None).cv
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm)
    return cv, ctx


def _positions(z_solute):
    return np.array([[0.1, 0.2, z_solute - 0.05], [0.3, 0.1, z_solute + 0.05], [0.0, 0.0, 0.0]])


def _z(cv, ctx):
    return cv.getValue(ctx).value_in_unit(unit.nanometers)


@pytest.mark.parametrize("z", [-3.2, -1.3, -0.4, 0.4, 1.3, 3.2])
def test_membrane_z_is_the_signed_offset_inside_the_box(z):
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=LZ / 2), _positions(z))
    assert _z(cv, ctx) == pytest.approx(z, abs=1e-6)


@pytest.mark.parametrize("z, wrapped", [(4.0, -3.0), (-4.5, 2.5), (9.0, 2.0), (-12.0, 2.0)])
def test_membrane_z_takes_the_minimum_image_across_the_box(z, wrapped):
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=LZ / 2), _positions(z))
    assert _z(cv, ctx) == pytest.approx(wrapped, abs=1e-6)


def test_membrane_z_follows_a_box_change_without_reparametrising():
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=4.0), _positions(3.2))
    assert _z(cv, ctx) == pytest.approx(3.2, abs=1e-6)
    ctx.setPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 6.0))
    assert _z(cv, ctx) == pytest.approx(-2.8, abs=1e-6)


def test_membrane_z_is_clamped_to_the_grid():
    spec = membrane_z_cv(SOLUTE, CENTER, half_width=2.0)
    cv, ctx = _context(spec, _positions(3.0))
    assert _z(cv, ctx) == pytest.approx(2.0, abs=1e-6)
    ctx.setPositions(_positions(-3.0))
    assert _z(cv, ctx) == pytest.approx(-2.0, abs=1e-6)


def test_membrane_z_spec_grid_spacing_is_a_fifth_of_the_hill_width():
    spec = membrane_z_cv(SOLUTE, CENTER, half_width=3.6, hill_width=0.2)
    assert (spec.grid_min, spec.grid_max, spec.grid_points) == (-3.6, 3.6, 181)
    assert spec.cv_length_per_unit_ang == 10.0


@pytest.mark.parametrize("head, cos", [((0, 0, 1), 1.0), ((0, 0, -1), -1.0), ((1, 0, 0), 0.0),
                                       ((1, 0, 1), np.sqrt(0.5))])
def test_axis_cos_is_the_cosine_to_z(head, cos):
    pos = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) + np.array([[0, 0, 0], head]) * 0.15
    cv, ctx = _context(axis_cos_cv(0, 1), pos)
    assert cv.getValue(ctx).value_in_unit(unit.dimensionless) == pytest.approx(cos, abs=1e-6)


# --- lipophilicity moment z (LMZ) and end-to-end axis -------------------------------------------

from autopath.metadynamics.CV import crippen_contributions, end_to_end_atoms, lipophilicity_z_cv


def _lmz_reference(xyz, c):
    """LM = sum_i (r_i - r_c) C_i, LMZ = LM_z / |LM| (the definition, in numpy)."""
    lm = ((xyz - xyz.mean(axis=0)) * np.asarray(c)[:, None]).sum(axis=0)
    return lm[2] / np.linalg.norm(lm)


def _lmz_value(spec, pos_nm):
    s = mm.System()
    for _ in range(len(pos_nm)):
        s.addParticle(1.0)
    cv = spec.resolve(pos_nm, len(pos_nm), None).cv
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm)
    return cv.getValue(ctx).value_in_unit(unit.dimensionless)


@pytest.mark.parametrize("offset", [0, 7])            # 7: the solute is not the first residue
def test_lmz_matches_the_definition_wherever_the_solute_sits(offset):
    rng = np.random.default_rng(1)
    c = rng.normal(size=6)
    xyz = rng.normal(scale=0.3, size=(6, 3)) + 2.0
    pos = np.vstack([rng.normal(size=(offset, 3)), xyz])
    spec = lipophilicity_z_cv(list(range(offset, offset + 6)), c)
    assert _lmz_value(spec, pos) == pytest.approx(_lmz_reference(xyz, c), abs=1e-5)
    assert (spec.grid_min, spec.grid_max, spec.name) == (-1.0, 1.0, "lm_z")


def test_lmz_is_translation_invariant_and_flips_with_z():
    rng = np.random.default_rng(2)
    c = rng.normal(size=5)
    xyz = rng.normal(scale=0.3, size=(5, 3))
    spec = lipophilicity_z_cv(list(range(5)), c)
    v = _lmz_value(spec, xyz)
    assert _lmz_value(spec, xyz + [1.0, -2.0, 3.0]) == pytest.approx(v, abs=1e-6)
    assert _lmz_value(spec, xyz * [1, 1, -1]) == pytest.approx(-v, abs=1e-6)


def test_lmz_rejects_mismatched_contributions():
    with pytest.raises(ValueError):
        lipophilicity_z_cv([0, 1, 2], [0.1, -0.1])


def test_crippen_contributions_are_rdkits_in_atom_order():
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    expected = [x[0] for x in rdMolDescriptors._CalcCrippenContribs(mol)]
    assert crippen_contributions(mol).tolist() == pytest.approx(expected)


def test_end_to_end_atoms_are_the_heavy_chain_ends():
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))     # heavy atoms 0..6; ends are C0 and O6
    assert set(end_to_end_atoms(mol)) == {0, 6}
