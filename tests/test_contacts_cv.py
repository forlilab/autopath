"""Contact-total CV: value against numpy, spec grid/bounds, 2D metad runs, control-script options."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import BoundaryCorrectedMetadynamics, MetadynamicsMD
from autopath.metadynamics.BoundaryCorrection import LinearComposite, boundary_factor
from autopath.metadynamics.CV import contacts_cv, funnel_z_cv, load_contacts_def
from autopath.metadynamics.learned.forces import learned_cv_spec

from .learned_toys import AXIS, GUEST, HOST, X0, Z0, s_plain, toy_pdb, toy_system, write_npz_r2, z_of

REF = mm.Platform.getPlatformByName("Reference")
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "04_Full-Pipeline" / "run_single_walker_control.py"
STEP = "1/(1+x^6)"
DEF = {"group1": [0, 1], "group2": [2, 3, 4], "threshold_nm": 0.4, "step_function": STEP,
       "cutoff_factor": 2.0, "switch_factor": 1.5, "bounds": [0.0, 6.0], "hill_width": 0.25}


def _np_contacts(X, g1, g2, r0, cf, sf, box=None, excl=()):
    """sum over g1 x g2 of 1/(1+(r/r0)^6) times OpenMM's switch on [sf r0, cf r0]."""
    rs, rc, tot = sf * r0, cf * r0, 0.0
    for i in g1:
        for j in g2:
            if (i, j) in excl or (j, i) in excl:
                continue
            d = X[i] - X[j]
            if box is not None:
                d -= box * np.round(d / box)
            r = np.linalg.norm(d)
            if r >= rc:
                continue
            x = min(1.0, max(0.0, (r - rs) / (rc - rs)))
            tot += (1 / (1 + (r / r0) ** 6)) * (1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5)
    return tot


def _gas_system(n, box=None, excl=()):
    s = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic if box else mm.NonbondedForce.CutoffNonPeriodic)
    for _ in range(n):
        s.addParticle(1.0)
        nb.addParticle(0.0, 0.3, 0.0)
    for i, j in excl:
        nb.addException(i, j, 0.0, 0.3, 0.0)
    if box:
        s.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    s.addForce(nb)
    return s


def _eval(cv, system, X):
    """(value, forces) of a cvpack CV added alone to ``system``."""
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    s.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(cv)))
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(X)
    st = ctx.getState(getEnergy=True, getForces=True)
    return st.getPotentialEnergy()._value, np.asarray(st.getForces(asNumpy=True)._value)


@pytest.mark.parametrize("box", [None, 2.0])
def test_contact_total_matches_numpy(box):
    rng = np.random.default_rng(3)
    n, g1, g2 = 12, [0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10]
    excl = [(0, 4), (2, 9)]
    system = _gas_system(n, box, excl)
    spec = contacts_cv(g1, g2, system, threshold_nm=0.3, step_function=STEP, cutoff_factor=3.0,
                       switch_factor=1.5, bounds=(0.0, 30.0), hill_width=0.5)
    comp = spec.resolve(None, n, None).cv
    for _ in range(4):
        # clusters around the origin and the box corner, so periodic images interact
        X = rng.uniform(-0.6, 0.6, size=(n, 3)) + (0 if box is None else rng.integers(0, 2, (n, 1)) * box)
        v, _ = _eval(comp.forces[0], system, X)
        assert v == pytest.approx(_np_contacts(X, g1, g2, 0.3, 3.0, 1.5, box, excl), abs=1e-8)
        assert _np_contacts(X, g1, g2, 0.3, 3.0, 1.5, box, excl) > 0.5


def test_factored_step_function_is_finite_at_threshold():
    system = _gas_system(2, 3.0)
    spec = contacts_cv([0], [1], system, threshold_nm=0.3, step_function=STEP, cutoff_factor=2.0,
                       switch_factor=1.5, bounds=(0.0, 1.0), hill_width=0.1)
    X = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])        # r = r0, x = 1
    v, f = _eval(spec.resolve(None, 2, None).cv.forces[0], system, X)
    assert v == pytest.approx(0.5, abs=1e-12)
    assert np.all(np.isfinite(f)) and f[1, 0] == pytest.approx(6 * 0.25 / 0.3, rel=1e-10)


def test_unfactored_step_function_not_supported():
    with pytest.raises(ValueError, match="not supported"):
        contacts_cv([0], [1], _gas_system(2, 3.0), threshold_nm=0.3, step_function="(1-x^6)/(1-x^12)",
                    cutoff_factor=2.0, switch_factor=1.5, bounds=(0.0, 1.0), hill_width=0.1)


def test_overlapping_groups_and_open_bounds_not_supported():
    s = _gas_system(3, 3.0)
    kw = dict(threshold_nm=0.3, step_function=STEP, cutoff_factor=2.0, switch_factor=1.5, hill_width=0.1)
    with pytest.raises(ValueError, match="not supported"):
        contacts_cv([0, 1], [1, 2], s, bounds=(0.0, 1.0), **kw)
    with pytest.raises(ValueError, match="bounds"):
        contacts_cv([0], [1, 2], s, bounds=(0.0, np.inf), **kw)


def test_spec_grid_bounds_and_clamp():
    spec = contacts_cv(**DEF, system=toy_system())
    assert spec.name == "contacts" and spec.is_deferred()
    assert spec.bounds == (0.0, 6.0) and spec.hill_width == 0.25
    assert spec.grid_min == pytest.approx(-0.75) and spec.grid_max == pytest.approx(6.75)
    assert (spec.grid_max - spec.grid_min) / (spec.grid_points - 1) == pytest.approx(0.25 / 5)
    comp = spec.resolve(None, 5, None).cv
    assert isinstance(comp, LinearComposite) and comp.clamp == (0.0, 6.0)
    assert comp.value([7.3]) == 6.0 and comp.value([2.5]) == 2.5
    v = BiasVariable(comp.forces[0], spec.grid_min, spec.grid_max, spec.hill_width, gridWidth=spec.grid_points)
    bf = boundary_factor([v], [spec.bounds])
    g = np.linspace(spec.grid_min, spec.grid_max, spec.grid_points)
    assert bf[np.argmin(abs(g - 0.0))] == pytest.approx(2.0, rel=1e-6)
    assert bf[np.argmin(abs(g - 6.0))] == pytest.approx(2.0, rel=1e-6)
    assert bf[np.argmin(abs(g - 3.0))] == pytest.approx(1.0, abs=1e-9)


def _run(tmp_path, specs, monkeypatch, **kw):
    made = []

    class Spy(BoundaryCorrectedMetadynamics):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(sys.modules[MetadynamicsMD.__module__], "BoundaryCorrectedMetadynamics", Spy)
    top, pdb = toy_pdb(tmp_path)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=toy_system(), run_id="w", cv_specs=specs, mMD_time=0.02, bias_factor=10,
           hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1, **kw)
    return made[0]


def _toy_contacts(X):
    return _np_contacts(X, DEF["group1"], DEF["group2"], 0.4, 2.0, 1.5, 10.0)


def test_funnel_z_contacts_2d_metad_records_both_variables(tmp_path, monkeypatch):
    z = funnel_z_cv(HOST, GUEST, AXIS, grid_min=Z0 - 1.0, grid_max=Z0 + 1.0, hill_width=0.05, grid_points=81)
    meta = _run(tmp_path, [z, contacts_cv(**DEF, system=toy_system())], monkeypatch)
    assert meta.bounds[1] == (0.0, 6.0)
    colvar = np.load(tmp_path / "out" / "COLVAR_w.npy")
    hills = np.load(tmp_path / "out" / "HILLS_w.npz")
    assert colvar.shape == (11, 2) and [str(n) for n in hills["cv_names"]] == ["funnel_z", "contacts"]
    Xp = np.round(X0 * 10, 3) / 10                       # PDB precision
    assert 0.1 < _toy_contacts(Xp) < 6.0
    assert colvar[0] == pytest.approx([z_of(Xp), _toy_contacts(Xp)], rel=1e-4)


def test_contacts_learned_2d_metad_with_z_monitor(tmp_path, monkeypatch):
    npz = write_npz_r2(str(tmp_path / "r2.npz"), residualised=False)
    specs = [contacts_cv(**DEF, system=toy_system()),
             *learned_cv_spec(npz, toy_system(), 1, n_plain_cvs=1)]
    zmon = funnel_z_cv(HOST, GUEST, AXIS, grid_min=-10.0, grid_max=10.0)
    meta = _run(tmp_path, specs, monkeypatch, monitor_cvs={"funnel_z": zmon}, reweight=True)
    assert meta.bounds[0] == (0.0, 6.0)
    colvar = np.load(tmp_path / "out" / "COLVAR_w.npy")
    hills = np.load(tmp_path / "out" / "HILLS_w.npz")
    assert [str(n) for n in hills["cv_names"]] == ["contacts", "learned_sw"]
    Xp = np.round(X0 * 10, 3) / 10
    assert colvar[0] == pytest.approx([_toy_contacts(Xp), s_plain(Xp)], rel=1e-4)
    rw = np.load(next((tmp_path / "out").glob("REWEIGHT_w.npz")))
    assert "monitor_funnel_z" in rw.files and rw["monitor_funnel_z"][0] == pytest.approx(z_of(Xp), rel=1e-4)


def test_load_contacts_def_validates(tmp_path):
    p = tmp_path / "def.json"
    p.write_text(json.dumps({"contacts": DEF, "learned_cv_file": "x.npz"}))
    d = load_contacts_def(str(p))
    assert d["contacts"]["bounds"] == (0.0, 6.0) and d["learned_cv_file"] == "x.npz"
    p.write_text(json.dumps({"contacts": DEF}))
    assert load_contacts_def(str(p))["learned_cv_file"] is None
    for key in DEF:
        p.write_text(json.dumps({"contacts": {k: v for k, v in DEF.items() if k != key}}))
        with pytest.raises(ValueError, match=key):
            load_contacts_def(str(p))
    p.write_text(json.dumps({"contacts": {**DEF, "extra": 1}}))
    with pytest.raises(ValueError, match="extra"):
        load_contacts_def(str(p))
    p.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="contacts"):
        load_contacts_def(str(p))


def _script(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stderr


def test_control_script_contacts_options_validation(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"contacts": DEF}))
    bad.write_text(json.dumps({"contacts": {k: v for k, v in DEF.items() if k != "bounds"}}))
    rc, err = _script("--cv", "funnel_z_contacts")
    assert rc == 2 and "--cv-def" in err
    rc, err = _script("--cv", "funnel_z_contacts", "--cv-def", str(bad))
    assert rc == 2 and "bounds" in err
    rc, err = _script("--cv", "funnel_z_contacts", "--cv-def", str(good), "--tt-basins", "1,2", "3,4")
    assert rc == 2 and "not supported" in err
    rc, err = _script("--cv", "contacts_learned", "--cv-def", str(good))
    assert rc == 2 and "learned_cv_file" in err
