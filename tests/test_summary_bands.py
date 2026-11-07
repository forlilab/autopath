import numpy as np
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.analysis import Band, bands, recrossings
from ._summary_fixtures import make_run
from openmm import unit


def _funnel():
    A = unit.angstrom
    return {"z_cc": 12.0 * A, "z_max": 24.0 * A, "z_low": -1.5 * A, "R_cylinder": 1.0 * A,
            "alpha": 31.5 * unit.degrees}


def test_funnel_z_bands(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=_funnel())
    b = bands(run)["funnel_z"]
    assert np.allclose(b.bound, (-0.15, 0.2 * 2.4)) and np.allclose(b.unbound, (1.2, 2.4 - 0.15)) and b.source == "funnel"


def test_monitor_only_funnel_z_without_z_low(tmp_path):
    fp = _funnel()
    del fp["z_low"]
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=("funnel_z",)), funnel=fp)
    lo = min(w.cvs["funnel_z"].min() for w in run.walkers)
    assert np.allclose(bands(run)["funnel_z"].bound[0], lo)


def test_grid_edge_bands_and_override(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=()))
    b = bands(run)["dist"]
    assert np.allclose(b.bound, (0.2, 0.2 + 0.34)) and np.allclose(b.unbound, (3.6 - 0.34, 3.6)) and b.source == "grid"
    b2 = bands(run, {"dist": {"bound_end": "high"}})["dist"]
    assert np.allclose(b2.bound, (3.6 - 0.34, 3.6)) and np.allclose(b2.unbound, (0.2, 0.2 + 0.34))
    b3 = bands(run, {"dist": {"bound": (0.2, 0.5)}})["dist"]
    assert np.allclose(b3.bound, (0.2, 0.5)) and b3.source == "override"


def test_learned_without_basins(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()), learned_cv=None)
    assert bands(run)["learned_sw"].source == "grid"


def test_learned_with_basins(tmp_path):
    lcv = {"basin_B": np.array([0.1, -0.5]), "basin_U": np.array([1.4, 0.8]), "sw_hill_width": 0.2}
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()), learned_cv=lcv)
    b = bands(run)["learned_sw"]
    assert np.allclose(b.bound, (-0.9, -0.1)) and np.allclose(b.unbound, (0.4, 1.2)) and b.source == "learned_cv"


def test_periodic_cv_bands(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("torsion",), monitors=()))
    b = bands(run)["torsion"]
    assert b.bound[0] == -np.pi and b.unbound[1] == np.pi


def test_recrossings_counts_full_cycles():
    class W:  # minimal walker
        cvs = {"x": np.array([0.0, 0.05, 0.5, 0.95, 0.5, 0.05, 0.08, 0.06, 0.95, 0.04])}
    r = recrossings(W, "x", Band((0.0, 0.1), (0.9, 1.0), "low", "grid"))
    assert r["n_cycles"] == 2 and r["frac_bound"] == 0.6 and r["frac_unbound"] == 0.2


def test_clamped_cv_bands_inside_clamp(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=(), bounds={"dist": (0.5, 2.0)}))
    b = bands(run)["dist"]
    assert b.source == "clamp" and b.bound_end == "low"
    assert np.allclose(b.bound, (0.5, 0.65)) and np.allclose(b.unbound, (1.85, 2.0))


def test_half_open_clamp_uses_grid_for_open_edge(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=(), bounds={"dist": (-np.inf, 2.0)}))
    b = bands(run)["dist"]
    assert np.allclose(b.bound, (0.2, 0.38)) and np.allclose(b.unbound, (1.82, 2.0))


def test_contacts_bound_at_high_edge(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("contacts",), monitors=(), bounds={"contacts": (0.0, 6.0)}))
    b = bands(run)["contacts"]
    assert b.bound_end == "high" and b.source == "clamp"
    assert np.allclose(b.bound, (5.4, 6.0)) and np.allclose(b.unbound, (0.0, 0.6))
    low = bands(run, {"contacts": {"bound_end": "low"}})["contacts"]
    assert np.allclose(low.bound, (0.0, 0.6))


def test_recorded_cv_type_overrides_name(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=(),
                            extra_meta={"cv_types": np.array(["contacts"])}))
    assert bands(run)["dist"].bound_end == "high"
    run = load_run(make_run(str(tmp_path / "s"), cvs=("contacts",), monitors=(),
                            extra_meta={"cv_types": np.array(["distance"])}))
    assert bands(run)["contacts"].bound_end == "low"
