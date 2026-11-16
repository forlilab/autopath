"""Two-channel toy: model vs frozen inputs, exact PMF, pulls, escapes and validate()."""
import os

import numpy as np
import pandas as pd
import pytest
import openmm as mm
import openmm.unit as u

from autopath.pulling import toy

DATA = os.path.join(os.path.dirname(__file__), "data", "toy")


def _energy(ctx, x):
    ctx.setPositions(x * u.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


@pytest.mark.parametrize("scenario", list(toy.SCENARIOS))
def test_build_matches_frozen_system(scenario):
    _, system, pos = toy.build(scenario)
    frozen = mm.XmlSerializer.deserialize(open(f"{DATA}/{scenario}/system.xml").read())
    plat = mm.Platform.getPlatformByName("Reference")
    cn, cf = (mm.Context(s, mm.VerletIntegrator(0.001), plat) for s in (system, frozen))
    rng = np.random.default_rng(0)
    lig = toy.LIGAND_ATOMS
    for _ in range(50):
        x = pos.copy()
        x[lig] += toy.O + rng.uniform([-1.5, -0.5, -0.3], [1.5, 2.2, 0.3]) - pos[lig].mean(0)
        e = _energy(cn, x)
        assert e == pytest.approx(_energy(cf, x), abs=1e-6)
        # numpy twin used by the quadrature
        assert e == pytest.approx(float(toy._potential(scenario, x[lig].mean(0))), abs=1e-6)


@pytest.mark.parametrize("scenario", list(toy.SCENARIOS))
def test_exact_pmf_coarse_matches_frozen_barrier(scenario):
    pmf = toy.exact_pmf(scenario, h=0.02)
    ref = pd.read_csv(f"{DATA}/{scenario}/reference.csv")
    assert list(pmf.columns) == list(ref.columns)
    lo, hi = toy.params(scenario)["r_barrier_window"]

    def barrier(df):
        g = df.G_total.to_numpy() - np.interp(toy.OFF, df.r, df.G_total)
        return g[(df.r >= lo) & (df.r <= hi)].max()

    assert barrier(pmf) == pytest.approx(barrier(ref), abs=0.1)


def test_exact_friction():
    assert toy.exact_friction("Open") == pytest.approx(5 * 12.011 * 50.0)


def test_run_pulls_smoke(tmp_path):
    dats = toy.run_pulls("Closed", [10.0], 1, str(tmp_path), seed=1)
    assert len(dats) == 1 and dats[0].endswith("sMD_replica-000001_v0.01_forward.dat")
    d = pd.read_csv(dats[0], comment="#")
    assert d.r_target.iloc[-1] == pytest.approx(toy.R_PULL_END, abs=1e-3)
    assert abs(d.r_after.iloc[0] - toy.OFF) < 0.1
    assert os.path.exists(dats[0].replace(".dat", "_geom.dat"))
    routes = toy.pull_routes(str(tmp_path / "trajectories"), "Closed")
    assert routes.final_tunnel.iloc[0] in (0, 1) and not routes.switch.iloc[0]


def test_run_escapes_censored(tmp_path):
    e = toy.run_escapes("Closed", 2, seed=3, out_dir=str(tmp_path), t_max_ps=5.0)
    assert list(e.columns) == ["run", "time_ps", "exit_tunnel", "escaped"]
    assert (~e.escaped).all() and (e.exit_tunnel == -1).all() and np.allclose(e.time_ps, 5.0)
    com = np.load(tmp_path / "esc_3_0.npz")["com"]
    assert com.shape == (6, 3) and np.linalg.norm(com[0] - toy.O) < 1e-6


def _write_dcd(path, com_nm):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    n = toy.N_POCKET + toy.N_LIG
    x = np.zeros((len(com_nm), n, 3), np.float32)
    x[:, toy.LIGAND_ATOMS] = 10.0 * np.asarray(com_nm)[:, None, :]
    uni = mda.Universe.empty(n, trajectory=True)
    uni.load_new(x, format=MemoryReader)
    with mda.Writer(str(path), n) as w:
        for _ in uni.trajectory:
            w.write(uni.atoms)


def _synthetic_analysis(tmp_path, scenario, pmf):
    """Analysis CSVs with known errors: barrier +1 (cumulant), plateau -2 (jarzynski), force exact."""
    r = np.round(np.arange(0.401, 1.6505, 0.001), 6)
    ex = np.interp(r, pmf.r, pmf.G_total)
    lo, hi = toy.params(scenario)["r_barrier_window"]
    bump = np.where((r >= lo) & (r <= hi), 1.0, 0.0)   # flat +1 over the whole window
    tail = np.where(r >= r.max() - 0.1, -2.0, 0.0)
    rows = [dict(speed=0.001, estimator="cumulant", r_coord=r, dG=ex + bump),
            dict(speed=0.001, estimator="jarzynski", r_coord=r, dG=ex + tail),
            dict(speed=0.0, estimator="force", r_coord=r, dG=ex + 5.0)]
    mix = pd.concat([pd.DataFrame({**d, "step": np.arange(len(r))}) for d in rows])
    a = tmp_path / "ana"
    a.mkdir()
    mix.to_csv(a / "mixture_pmfs.csv", index=False)
    pd.DataFrame(dict(estimator=["cumulant", "jarzynski", "force", "cumulant"],
                      source=["per_speed(v=0.001)", "per_speed(v=0.001)", "v0_extrapolation", "v0_extrapolation"],
                      speed_nm_per_ps=[0.001, 0.001, 0.0, 0.0], tau_mfpt_ns=[10.0, 20.0, 40.0, 99.0])
                 ).to_csv(a / "koff_kramers.csv", index=False)
    g = toy.exact_friction(scenario)
    pd.DataFrame(dict(estimator="force", method="regression", speed=0.0, r_coord=r,
                      Gamma=np.where(np.arange(len(r)) % 2, g + 30.0, g - 30.0))).to_csv(a / "friction.csv", index=False)
    return str(a)


def test_validate_synthetic(tmp_path):
    scenario = "Open"
    pmf = toy.exact_pmf(scenario, h=0.02)
    adir = _synthetic_analysis(tmp_path, scenario, pmf)
    esc = pd.DataFrame(dict(run=range(4), time_ps=[10e3, 20e3, 30e3, 1e3],
                            exit_tunnel=[0, 0, 1, -1], escaped=[True, True, True, False]))
    traj = tmp_path / "traj"
    traj.mkdir()
    O, (e0, e1) = toy.O, np.asarray(toy.params(scenario)["tunnels"])
    s = np.linspace(0, 1.0, 20)[:, None]
    _write_dcd(traj / "sMD_replica-000001_v0.001_forward.dcd", O + s * e0)               # E, no switch
    _write_dcd(traj / "sMD_replica-000002_v0.001_forward.dcd",
               np.vstack([O + s[:15] * e1, O + 0.7 * e0 + 0 * s[:5]]))                  # H committed -> E
    v = toy.validate(adir, scenario, esc, pulls_dir=str(traj), pmf=pmf).set_index("estimator")

    assert v.loc["cumulant", "barrier_err"] == pytest.approx(1.0, abs=1e-9)
    assert v.loc["cumulant", "plateau_err"] == pytest.approx(0.0, abs=1e-9)
    assert v.loc["jarzynski", "plateau_err"] == pytest.approx(-2.0, abs=1e-9)
    assert v.loc["force", "barrier_err"] == pytest.approx(0.0, abs=1e-9)   # constant offset aligns away
    assert v.loc["force", "speed_m_per_s"] == 0.0
    assert np.allclose(v.gamma_rmse, 30.0)
    assert v.tau_exact_ns.iloc[0] == pytest.approx(20.0) and v.n_escapes.iloc[0] == 3
    assert v.loc["jarzynski", "tau_ratio"] == pytest.approx(1.0)
    assert v.esc_t1.iloc[0] == pytest.approx(1 / 3)
    assert v.loc["cumulant", "n_pulls"] == 2
    assert v.loc["cumulant", "exit_t0"] == pytest.approx(1.0)
    assert v.loc["cumulant", "switch"] == pytest.approx(0.5)
    assert 0 < v.loc["cumulant", "switch_lo"] < 0.5 < v.loc["cumulant", "switch_hi"] < 1
