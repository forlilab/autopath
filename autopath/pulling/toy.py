"""Two-channel unbinding toy model with exact references for validating the sMD analysis.

A ring of restrained pocket particles (name CA) and a stiff five-particle ligand (resname UNK)
whose centre of mass (COM) feels an analytic potential. Tunnel ``k`` has unit axis ``e_k``; with
``s_k = (R - O).e_k`` and ``rho_k^2 = |R - O|^2 - s_k^2``::

    U_k = f(s_k; B_k) + kr/2 rho_k^2

``f`` is a bound well at ``s = 0``, a barrier bump ``B_k``, a flat bulk plateau and back/outer
walls. Channels are joined by a smooth minimum ``U = -kT ln sum_k exp(-U_k/kT)``. The pocket
COM sits ``OFF`` nm behind the well along ``-y`` so the pulling coordinate ``r`` grows along
both tunnels (at +-45 deg from +y). Without solvent or nonbonded terms the Langevin bath is the
only friction, so the COM friction is exactly ``m * FRICTION`` and the COM equilibrium density is
exactly ``exp(-U/kT)``.

Scenarios: ``Closed`` (tunnel 0 easy, tunnel 1 hard), ``Open`` (adds a passage joining the
tunnels past their barriers), ``Open-HR`` / ``Open-HW`` (``Open`` plus ripples / a secondary
well on the hard tunnel beyond the passage).
"""
import math
import os
from glob import glob
from itertools import groupby

import numpy as np
import pandas as pd
import openmm as mm
import openmm.app as app
import openmm.unit as u

KB = 0.0083144626   # kJ/mol/K
T = 300.0
KT = KB * T
BOX = 6.0
O = np.array([3.0, 2.2, 3.0])   # bound-well centre (nm)
OFF = 0.4                        # pocket COM sits OFF nm behind the well along -y
FRICTION = 50.0                  # Langevin collision rate (1/ps)
TIMESTEP_PS = 0.004
R_PULL_END = 1.65                # nm, pull end in r
SPRING_PER_ATOM = 418.4          # kJ/mol/nm^2 per ligand heavy atom
S_ESCAPE = 1.0                   # nm down a tunnel axis that ends an escape run
S_LABEL, S_COMMIT, PERSIST = 0.3, 0.6, 2   # route labels: axial thresholds (nm), min run (frames)
BASE = dict(D=10.0, sw=0.15,     # bound well depth / width (kJ/mol, nm)
            xb=0.55, sb=0.12,    # barrier bump position / width
            P=2.0, xp=0.9, wp=0.08,   # bulk plateau level / onset / softness
            kr=500.0,            # transverse confinement (kJ/mol/nm^2)
            sback=-0.15, aw=2.4, kw=5000.0)  # back wall / outer wall on s
_S45 = float(np.sqrt(0.5))
_CLOSED = dict(tunnels=[[_S45, _S45, 0.0], [-_S45, _S45, 0.0]], B=[5.0, 12.0])
_PASSAGE = dict(sx=0.75, Bx=0.0, sbx=0.12)
SCENARIOS = {
    "Closed": _CLOSED,
    "Open": dict(_CLOSED, passage=_PASSAGE),
    "Open-HR": dict(_CLOSED, hmod="rough", passage=_PASSAGE),
    "Open-HW": dict(_CLOSED, hmod="well", passage=_PASSAGE),
}
LIG = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [-0.05, 0.14, 0.0],
                [-0.05, -0.07, 0.12], [-0.05, -0.07, -0.12]])   # nm
N_POCKET, R_POCKET = 8, 0.45
N_LIG = len(LIG)
LIGAND_ATOMS = list(range(N_POCKET, N_POCKET + N_LIG))
POCKET_ATOMS = list(range(N_POCKET))
MASS = 12.011
EPS_ATAN = 1e-6   # nm; keeps atan2's gradient finite at the well centre
_HMOD_S0 = 0.85
_TWO_PI = 2 * math.pi


def _scen(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    return {**BASE, **SCENARIOS[scenario]}


def params(scenario):
    """Model and protocol parameters of one scenario.

    Parameters
    ----------
    scenario : str
        Key of ``SCENARIOS``.

    Returns
    -------
    dict
        Landscape parameters plus ``T``, ``box``, ``well_centre``, ``pocket_offset_y``,
        ``friction_per_ps``, ``ligand_mass``, ``spring_k``, ``r_pull_end``,
        ``r_barrier_window`` (r range scored for the barrier) and ``s_escape``.
    """
    p = _scen(scenario)
    e0 = p["tunnels"][0]
    return dict(scenario=scenario, T=T, box=BOX, well_centre=O.tolist(), pocket_offset_y=-OFF,
                friction_per_ps=FRICTION, ligand_mass=MASS * N_LIG,
                spring_k=SPRING_PER_ATOM * N_LIG, r_pull_end=R_PULL_END,
                r_barrier_window=[r_of_s(0.2, e0), r_of_s(0.85, e0)], s_escape=S_ESCAPE, **p)


def r_of_s(s, e):
    """Pull coordinate r reached at axial distance ``s`` (nm) on the axis of tunnel direction ``e``."""
    return float(np.linalg.norm(np.array([0.0, OFF, 0.0]) + s * np.asarray(e, float)))


def exact_friction(scenario):
    """Exact COM friction ``m * gamma`` (kJ ps / mol / nm^2)."""
    p = params(scenario)
    return p["ligand_mass"] * p["friction_per_ps"]


# --- potential: Lepton expression (simulation) and numpy twin (quadrature) ---

def _hmod_np(kind, s):
    if kind == "rough":
        return 0.5 * (1 + np.tanh((s - _HMOD_S0) / 0.03)) * 1.5 * np.cos(_TWO_PI * (s - _HMOD_S0) / 0.05)
    if kind == "well":
        return -5.0 * np.exp(-(s - 1.05)**2 / (2 * 0.06**2))
    raise ValueError(kind)


def _hmod_expr(kind, s):
    if kind == "rough":
        return f"(0.5*(1+tanh(({s}-{_HMOD_S0})/0.03))*1.5*cos({_TWO_PI!r}*({s}-{_HMOD_S0})/0.05))"
    if kind == "well":
        return f"(-5.0*exp(-({s}-1.05)^2/(2*0.06^2)))"
    raise ValueError(kind)


def _f_expr(s, B, p):
    base = (f"(-{p['D']}*exp(-({s})^2/(2*{p['sw']}^2))"
            f" + {B}*exp(-(({s})-{p['xb']})^2/(2*{p['sb']}^2))"
            f" + {p['P']}*0.5*(1+tanh((({s})-{p['xp']})/{p['wp']}))"
            f" + 0.5*{p['kw']}*step({p['sback']}-({s}))*(({s})-{p['sback']})^2"
            f" + 0.5*{p['kw']}*step(({s})-{p['aw']})*(({s})-{p['aw']})^2)")
    # the hard-tunnel modification acts on tunnel 1 only
    return f"({base} + {_hmod_expr(p['hmod'], s)})" if p.get("hmod") and s == "s1" else base


def _f_np(s, B, p):
    return (-p["D"] * np.exp(-s**2 / (2 * p["sw"]**2))
            + B * np.exp(-(s - p["xb"])**2 / (2 * p["sb"]**2))
            + p["P"] * 0.5 * (1 + np.tanh((s - p["xp"]) / p["wp"]))
            + 0.5 * p["kw"] * (s < p["sback"]) * (s - p["sback"])**2
            + 0.5 * p["kw"] * (s > p["aw"]) * (s - p["aw"])**2)


def _energy_expr(p):
    """Lepton expression for U(COM); x1, y1, z1 = ligand centroid."""
    us = [f"({_f_expr(f's{k}', B, p)} + 0.5*{p['kr']}*(d2-s{k}^2))" for k, B in enumerate(p["B"])]
    defs = "; ".join([f"s{k}=dx*{e[0]}+dy*{e[1]}+dz*{e[2]}" for k, e in enumerate(p["tunnels"])]
                     + ["d2=dx^2+dy^2+dz^2", f"dx=x1-{O[0]}", f"dy=y1-{O[1]}", f"dz=z1-{O[2]}"])
    if "passage" in p:
        return _passage_expr(us, defs, p)
    # stable smooth minimum: min(U0,U1) - kT ln(1 + exp(-|U0-U1|/kT))
    return (f"min(U0,U1) - {KT}*log(1+exp(-abs(U0-U1)/{KT})); "
            f"U0={us[0]}; U1={us[1]}; {defs}")


def _passage_geom(p):
    """Passage floors at both ends and arc frame (bisector b, normal n); tunnel axes lie in xy."""
    e0, e1 = (np.asarray(e, float) for e in p["tunnels"][:2])
    fl = [float(_f_np(np.array(p["passage"]["sx"]), B, p)) for B in p["B"][:2]]
    th = [float(np.arctan2(e[1], e[0])) for e in (e0, e1)]
    b = (e0 + e1) / np.linalg.norm(e0 + e1)
    n = e1 - (e1 @ b) * b
    n /= np.linalg.norm(n)
    return fl, th, b, n


def _passage_expr(us, defs, p):
    """Circular arc of radius sx about O from tunnel 0 to tunnel 1, as a third soft-min channel."""
    q = p["passage"]
    fl, th, b, n = _passage_geom(p)
    sx = q["sx"]
    tdef = (f"tc=min(max(0.5+atan2(dx*{n[0]}+dy*{n[1]}+dz*{n[2]},dx*{b[0]}+dy*{b[1]}+dz*{b[2]}+{EPS_ATAN})"
            f"/({th[1] - th[0]}),0),1)")
    ang = f"({th[0]}+tc*({th[1] - th[0]}))"
    dist2 = f"(dx-{sx}*cos({ang}))^2+(dy-{sx}*sin({ang}))^2+dz^2"
    plen2 = (sx * (th[1] - th[0]))**2
    ux = (f"({fl[0]} + {fl[1] - fl[0]}*tc + {q['Bx']}*exp(-(tc-0.5)^2*{plen2}/(2*{q['sbx']}^2))"
          f" + 0.5*{p['kr']}*({dist2}))")
    return (f"m - {KT}*log(exp(-(U0-m)/{KT})+exp(-(U1-m)/{KT})+exp(-(U2-m)/{KT})); "
            f"m=min(min(U0,U1),U2); U0={us[0]}; U1={us[1]}; U2={ux}; {tdef}; {defs}")


def _passage_energy(D, p):
    """Passage energy and clamped arc parameter tc (0 at tunnel 0, 1 at tunnel 1)."""
    q = p["passage"]
    fl, th, b, n = _passage_geom(p)
    sx = q["sx"]
    tc = np.clip(0.5 + np.arctan2(D @ n, D @ b + EPS_ATAN) / (th[1] - th[0]), 0, 1)
    a = th[0] + tc * (th[1] - th[0])
    d2 = (D[..., 0] - sx * np.cos(a))**2 + (D[..., 1] - sx * np.sin(a))**2 + D[..., 2]**2
    plen2 = (sx * (th[1] - th[0]))**2
    return (fl[0] + (fl[1] - fl[0]) * tc + q["Bx"] * np.exp(-(tc - 0.5)**2 * plen2 / (2 * q["sbx"]**2))
            + 0.5 * p["kr"] * d2), tc


def _channel_energies(D, p):
    """Channel energies (tunnels, then passage) for displacements ``D`` (..., 3) from O."""
    d2 = (D**2).sum(-1)
    out = []
    for k, (e, B) in enumerate(zip(p["tunnels"], p["B"])):
        s = D @ np.asarray(e, float)
        U = _f_np(s, B, p) + 0.5 * p["kr"] * (d2 - s**2)
        out.append(U + _hmod_np(p["hmod"], s) if p.get("hmod") and k == 1 else U)
    if "passage" in p:
        out.append(_passage_energy(D, p)[0])
    return np.stack(out)


def _potential(scenario, com_nm):
    """COM potential energy (kJ/mol) at ligand COM positions ``com_nm`` (..., 3), box frame."""
    p = _scen(scenario)
    Uk = _channel_energies(np.asarray(com_nm, float) - O, p)
    m = Uk.min(0)
    return m - KT * np.log(np.exp(-(Uk - m) / KT).sum(0))


# --- system ---

def build(scenario):
    """OpenMM model of one scenario.

    Parameters
    ----------
    scenario : str
        Key of ``SCENARIOS``.

    Returns
    -------
    topology : openmm.app.Topology
        Pocket particles (chain A, name CA) then ligand (chain B, resname UNK).
    system : openmm.System
        Restrained pocket, stiff ligand bonds and the COM potential.
    positions : numpy.ndarray
        (13, 3) start positions in nm, ligand centred on the bound well.
    """
    p = _scen(scenario)
    top = app.Topology()
    chA = top.addChain("A")
    for _ in range(N_POCKET):
        top.addAtom("CA", app.element.carbon, top.addResidue("ALA", chA))
    res = top.addResidue("UNK", top.addChain("B"))
    for i in range(N_LIG):
        top.addAtom(f"C{i + 1}", app.element.carbon, res)
    top.setPeriodicBoxVectors(np.eye(3) * BOX * u.nanometer)
    ang = 2 * np.pi * np.arange(N_POCKET) / N_POCKET
    pc = O - np.array([0.0, OFF, 0.0])
    pocket = pc + np.c_[R_POCKET * np.cos(ang), np.zeros(N_POCKET), R_POCKET * np.sin(ang)]
    pos = np.vstack([pocket, LIG - LIG.mean(0) + O])

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in np.eye(3) * BOX])
    for _ in range(len(pos)):
        system.addParticle(MASS)
    rest = mm.CustomExternalForce("0.5*kp*periodicdistance(x,y,z,x0,y0,z0)^2")
    rest.addGlobalParameter("kp", 1e5)
    for k in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(k)
    for i in POCKET_ATOMS:
        rest.addParticle(i, list(pocket[i]))
    system.addForce(rest)
    bonds = mm.HarmonicBondForce()
    for a in range(N_LIG):
        for b in range(a + 1, N_LIG):
            bonds.addBond(LIGAND_ATOMS[a], LIGAND_ATOMS[b], float(np.linalg.norm(LIG[a] - LIG[b])), 2e4)
    system.addForce(bonds)
    land = mm.CustomCentroidBondForce(1, _energy_expr(p))
    land.addGroup(LIGAND_ATOMS)
    land.addBond([0], [])
    system.addForce(land)
    return top, system, pos


def exact_pmf(scenario, h=0.003, dr=0.01, rmax=2.3):
    """Exact PMF along r = |ligand COM - pocket COM| by 3D quadrature of exp(-U/kT).

    Parameters
    ----------
    scenario : str
        Key of ``SCENARIOS``.
    h : float
        Quadrature grid step (nm), midpoint rule; keep well below ``dr``.
    dr : float
        Shell width in r (nm).
    rmax : float
        Upper edge of the r grid (nm).

    Returns
    -------
    pandas.DataFrame
        ``r`` (shell centres), ``G_total`` and one column per channel (``G_t0``, ``G_t1``,
        ``G_pass``), kJ/mol, all shifted so that min ``G_total`` = 0; empty shells are inf.
    """
    p = _scen(scenario)
    beta = 1 / KT
    gx = np.arange(-2.0, 2.0, h) + h / 2
    gy = np.arange(-0.6, 2.6, h) + h / 2
    gz = np.arange(-0.35, 0.35, h) + h / 2
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    edges = np.arange(0, rmax + dr, dr)
    zs = None
    for x in gx:   # x slices bound memory
        D = np.stack([np.full_like(Y, x), Y, Z], -1)
        w_k = np.exp(-beta * _channel_energies(D, p))
        r = np.sqrt(x**2 + (Y + OFF)**2 + Z**2).ravel()
        h_k = np.stack([np.histogram(r, bins=edges, weights=w.ravel())[0] for w in w_k])
        zs = h_k if zs is None else zs + h_k
    out = {"r": 0.5 * (edges[1:] + edges[:-1])}
    with np.errstate(divide="ignore"):
        out["G_total"] = -np.log(zs.sum(0) / dr) / beta
        for k in range(len(zs)):
            out[f"G_t{k}" if k < len(p["tunnels"]) else "G_pass"] = -np.log(zs[k] / dr) / beta
    df = pd.DataFrame(out)
    ref = df["G_total"][np.isfinite(df["G_total"])].min()
    df.iloc[:, 1:] -= ref
    return df


# --- simulations ---

def _run_id(i):
    """HHMMSS-valid, int-parseable replica id."""
    return f"{i // 3600:02d}{(i // 60) % 60:02d}{i % 60:02d}"


def run_pulls(scenario, speeds_m_per_s, n, out_dir, seed, platform="Reference", first=1):
    """Forward sMD pulls of the toy with :class:`SteeredMD` at the benchmark settings.

    4 fs, 300 K, Langevin friction ``FRICTION``, 0.001 nm restraint moves, spring 418.4 kJ/mol/nm^2
    per ligand atom, contact autostop off, pulled from the start r to ``R_PULL_END``, geometric
    features logged.

    Parameters
    ----------
    scenario : str
        Key of ``SCENARIOS``.
    speeds_m_per_s : sequence of float
        Pulling speeds in m/s (1 m/s = 0.001 nm/ps).
    n : int
        Pulls per speed.
    out_dir : str
        Writes ``out_dir/start.pdb`` and ``out_dir/trajectories/sMD_replica-*``.
    seed : int
        Base seed of :class:`SteeredMD`; each pull is seeded from (seed, replica name).
    platform : str
        OpenMM platform name.
    first : int
        Index of the first replica (to split a campaign across jobs).

    Returns
    -------
    list of str
        Paths of the written ``.dat`` logs.
    """
    from autopath.pulling.steered_md import SteeredMD
    p = params(scenario)
    top, system, pos = build(scenario)
    os.makedirs(f"{out_dir}/trajectories", exist_ok=True)
    pdb = f"{out_dir}/start.pdb"
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(top, pos * u.nanometer, fh)
    smd = SteeredMD(system=system, topology=top, groupA_atoms=LIGAND_ATOMS, groupB_atoms=POCKET_ATOMS,
                    restart_velocities=True, timestep=TIMESTEP_PS, temperature=T,
                    out_dir=f"{out_dir}/trajectories", platform=platform, dx_per_move=0.001,
                    max_displacement=R_PULL_END - OFF, sMD_spring_cte=p["spring_k"],
                    autostop_nc=None, save_freq=5, log_geom_features=True, seed=seed)
    smd.integrator_friction = p["friction_per_ps"] / u.picosecond
    dats = []
    for v in speeds_m_per_s:
        for i in range(first, first + n):
            rid = smd.run(pulling_speed=v / 1000.0, pulling_direction="forward", run_id=_run_id(i),
                          pdb_file=pdb)
            dats.append(f"{smd.out_dir}/sMD_{rid}.dat")
    return dats


def run_escapes(scenario, n, seed, out_dir=None, stride_ps=1.0, t_max_ps=2e6, platform="Reference"):
    """Unbiased escapes from the bound well: exact MFPT and exit-tunnel split.

    Each run starts from the build positions with fresh velocities and ends when the ligand COM
    is more than ``s_escape`` nm down a tunnel axis (checked every ``stride_ps``).

    Parameters
    ----------
    scenario : str
        Key of ``SCENARIOS``.
    n : int
        Number of runs.
    seed : int
        Seeds the integrator and the per-run velocities.
    out_dir : str, optional
        If given, writes the COM trace of run ``i`` (nm, every ``stride_ps``, from t = 0) to
        ``out_dir/esc_<seed>_<i>.npz``.
    stride_ps : float
        Escape check / COM save interval (ps).
    t_max_ps : float
        Censoring time (ps); such runs have ``escaped`` = False and ``exit_tunnel`` = -1.
    platform : str
        OpenMM platform name.

    Returns
    -------
    pandas.DataFrame
        ``run``, ``time_ps``, ``exit_tunnel`` (index into ``params(scenario)['tunnels']``), ``escaped``.
    """
    p = params(scenario)
    _, system, x0 = build(scenario)
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, p["friction_per_ps"] / u.picosecond,
                                        TIMESTEP_PS * u.picosecond)
    integ.setRandomNumberSeed(int(seed))
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    E = np.asarray(p["tunnels"], float)
    chunk = int(round(stride_ps / TIMESTEP_PS))
    rng = np.random.default_rng(seed)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    for run in range(n):
        ctx.setPositions(x0 * u.nanometer)
        ctx.setVelocitiesToTemperature(T * u.kelvin, int(rng.integers(1, 2**31 - 1)))
        com, t, tun = [x0[LIGAND_ATOMS].mean(0)], 0.0, -1
        while t < t_max_ps:
            integ.step(chunk)
            t += chunk * TIMESTEP_PS
            c = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
            com.append(c[LIGAND_ATOMS].mean(0))
            s = E @ (com[-1] - O)
            if s.max() > p["s_escape"]:
                tun = int(np.argmax(s))
                break
        if out_dir:
            np.savez_compressed(f"{out_dir}/esc_{seed}_{run}.npz", com=np.asarray(com, np.float32),
                                t_ps=t, exit_tunnel=tun, stride_ps=stride_ps, seed=seed)
        rows.append(dict(run=run, time_ps=t, exit_tunnel=tun, escaped=tun >= 0))
    return pd.DataFrame(rows)


# --- validation ---

def _persistent(seq, k):
    """Labels of runs of at least ``k`` equal consecutive items."""
    return [lab for lab, g in groupby(seq) if len(list(g)) >= k]


def pull_routes(traj_dir, scenario):
    """Exit tunnel and committed route switch of each pull, from its DCD.

    A frame is labelled with the tunnel the COM is furthest down once that axial distance exceeds
    ``S_LABEL`` (exit) or ``S_COMMIT`` (switch); label runs shorter than ``PERSIST`` frames are
    ignored. ``final_tunnel`` is the last persistent label (-1 if none); ``switch`` is True when
    the persistent committed labels include more than one tunnel.

    Parameters
    ----------
    traj_dir : str
        Folder with ``sMD_*_forward.dcd`` written by :func:`run_pulls`.
    scenario : str
        Key of ``SCENARIOS``.

    Returns
    -------
    pandas.DataFrame
        ``trajname``, ``speed`` (nm/ps), ``final_tunnel``, ``switch``.
    """
    from MDAnalysis.coordinates.DCD import DCDReader
    E = np.asarray(params(scenario)["tunnels"], float)
    rows = []
    for f in sorted(glob(f"{traj_dir}/sMD_*_forward.dcd")):
        with DCDReader(f) as rd:
            com = np.array([ts.positions[LIGAND_ATOMS].mean(0) for ts in rd]) / 10.0
        s = (com - O) @ E.T
        lab = _persistent(s.argmax(1)[s.max(1) > S_LABEL].tolist(), PERSIST)
        labc = _persistent(s.argmax(1)[s.max(1) > S_COMMIT].tolist(), PERSIST)
        name = os.path.basename(f)[:-4]
        rows.append(dict(trajname=name, speed=float(name.split("_v")[1].split("_")[0]),
                         final_tunnel=lab[-1] if lab else -1, switch=len(set(labc)) > 1))
    return pd.DataFrame(rows)


def _wilson(k, n, z=1.959964):
    """Fraction k/n and its Wilson score interval."""
    if n == 0:
        return np.nan, np.nan, np.nan
    f = k / n
    c = (f + z**2 / (2 * n)) / (1 + z**2 / n)
    hw = z * np.sqrt(f * (1 - f) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return f, c - hw, c + hw


def _frac_cols(prefix, k, n):
    f, lo, hi = _wilson(k, n)
    return {prefix: f, f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def validate(analysis_dir, scenario, escapes, pulls_dir=None, force_dir=None, pmf=None,
             n_boot=2000, seed=0):
    """Score an sMD analysis of the toy against its exact answers.

    Reads the CSVs written by :class:`SMDAnalysis` (no estimator is re-run). Profiles are aligned
    at their first grid point; the barrier error is max G over ``r_barrier_window`` minus the
    exact max there, the plateau error is the mean error over the last 0.1 nm of the profile.

    Parameters
    ----------
    analysis_dir : str
        Analysis folder for the cumulant and Jarzynski rows (per speed).
    scenario : str
        Key of ``SCENARIOS``.
    escapes : pandas.DataFrame
        Output of :func:`run_escapes` (``time_ps``, ``exit_tunnel``, ``escaped``).
    pulls_dir : str, optional
        Folder with the pull DCDs; enables the exit-fraction and switching columns.
    force_dir : str, optional
        Analysis folder for the force (v->0) row and the friction profile; defaults to
        ``analysis_dir`` (use the cross-speed clustering run here).
    pmf : pandas.DataFrame, optional
        Precomputed :func:`exact_pmf`; computed if None.
    n_boot : int
        Bootstrap resamples of the escape times for the exact-MFPT interval.
    seed : int
        Bootstrap seed.

    Returns
    -------
    pandas.DataFrame
        One row per estimator and speed (force at speed 0): ``barrier_err``, ``plateau_err``
        (kJ/mol), ``gamma_rmse`` (force-regression friction vs exact over the pulled range, same on
        every row), ``tau_ns``, ``tau_exact_ns`` with ``_lo``/``_hi`` (95% bootstrap), ``tau_ratio``;
        with ``pulls_dir``: ``n_pulls``, ``exit_t<k>`` (fraction of pulls whose final tunnel is k),
        ``switch`` (committed route switch), each with a 95% Wilson ``_lo``/``_hi``, the force row
        pooling all speeds; and the escape exit fractions ``esc_t<k>`` on every row.
    """
    p = params(scenario)
    force_dir = force_dir or analysis_dir
    pmf = exact_pmf(scenario) if pmf is None else pmf
    lo, hi = p["r_barrier_window"]

    def score(r, dG):
        ex = np.interp(r, pmf.r, pmf.G_total)
        err = (dG - dG[0]) - (ex - ex[0])
        mb, mp = (r >= lo) & (r <= hi), r >= r.max() - 0.1
        return (dG[mb] - dG[0]).max() - (ex[mb] - ex[0]).max(), err[mp].mean()

    mix = pd.concat([pd.read_csv(f"{analysis_dir}/mixture_pmfs.csv").query(
                         "estimator in ['cumulant', 'jarzynski'] and speed > 0"),
                     pd.read_csv(f"{force_dir}/mixture_pmfs.csv").query("estimator == 'force' and speed == 0")])
    kk = pd.concat([pd.read_csv(f"{analysis_dir}/koff_kramers.csv").query(
                        "estimator in ['cumulant', 'jarzynski'] and source.str.startswith('per_speed')",
                        engine="python"),
                    pd.read_csv(f"{force_dir}/koff_kramers.csv").query(
                        "estimator == 'force' and source == 'v0_extrapolation'")])
    fr = pd.read_csv(f"{force_dir}/friction.csv").query("estimator == 'force' and method == 'regression'")
    gamma_rmse = float(np.sqrt(np.mean((fr.Gamma.to_numpy() - exact_friction(scenario))**2)))

    esc = escapes[escapes.escaped.astype(bool)]
    t = esc.time_ps.to_numpy() / 1000.0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(t, len(t)).mean() for _ in range(n_boot)]
    tau_ex = dict(tau_exact_ns=t.mean(), tau_exact_ns_lo=np.quantile(bs, 0.025),
                  tau_exact_ns_hi=np.quantile(bs, 0.975), n_escapes=len(t))
    n_t = len(p["tunnels"])
    esc_cols = {}
    for k in range(n_t):
        esc_cols.update(_frac_cols(f"esc_t{k}", int((esc.exit_tunnel == k).sum()), len(esc)))
    routes = pull_routes(pulls_dir, scenario) if pulls_dir else None

    rows = []
    for (est, sp), d in mix.groupby(["estimator", "speed"]):
        d = d.sort_values("r_coord")
        b, pl = score(d.r_coord.to_numpy(), d.dG.to_numpy())
        k = kk[(kk.estimator == est) & np.isclose(kk.speed_nm_per_ps, sp)]
        tau = float(k.tau_mfpt_ns.iloc[0]) if len(k) else np.nan
        row = dict(estimator=est, speed_m_per_s=round(sp * 1000, 6), barrier_err=b, plateau_err=pl,
                   gamma_rmse=gamma_rmse, tau_ns=tau, **tau_ex, tau_ratio=tau / tau_ex["tau_exact_ns"])
        if routes is not None:
            g = routes if sp == 0 else routes[np.isclose(routes.speed, sp)]
            row["n_pulls"] = len(g)
            for kt in range(n_t):
                row.update(_frac_cols(f"exit_t{kt}", int((g.final_tunnel == kt).sum()), len(g)))
            row.update(_frac_cols("switch", int(g.switch.sum()), len(g)))
        rows.append({**row, **esc_cols})
    return pd.DataFrame(rows)
