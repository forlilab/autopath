# SLC15A4 Minibinder sMD Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run steered-MD unbinding of 7 SLC15A4 minibinders from their 25 equilibrated end points each at 3 speeds, and produce a v→0 barrier ranking with between-seed error bars.

**Architecture:** Pure campaign logic (selections, run-id encoding, array decoding, preflight geometry, gate evaluation, bootstrap) lives in two importable modules with pytest coverage. Thin executable drivers wrap them and call `autopath.pulling.steered_md.SteeredMD` — autopath is used as a library and is not modified. A 15-replica pilot gates the 525-replica campaign.

**Tech Stack:** Python 3.12, OpenMM 8.x, MDAnalysis 2.10, mdtraj, numpy/pandas, pytest, SLURM. Run everything in the `autopath` micromamba env: `micromamba run -n autopath <cmd>`.

**Spec:** `docs/superpowers/specs/2026-09-24-slc15a4-smd-campaign-design.md` (in the autopath repo). Read it before Task 2.

## Global Constraints

- Target project root: `/gpfs/group/forli/mllanos/peter/minibinders/` — referred to below as `$PROJ`. All new code goes there, **not** into the autopath package.
- Do **not** modify anything under `autopath/`. The campaign uses it as a library.
- Env: `micromamba run -n autopath`. Never conda/miniconda.
- Systems: `slc_m1` … `slc_m7`.
- Speeds, in this exact order: `SPEEDS = (0.001, 0.005, 0.010)` nm/ps.
- `MAX_DISPLACEMENT = 2.7` nm, `DX_PER_MOVE = 0.001` nm → `EXPECTED_MOVES = 2700`.
- `TIMESTEP = 0.004` ps, `TEMPERATURE = 300.0` K, `SAVE_FREQ = 5`.
- `SPRING_PER_ATOM = 100 * 4.184` kJ/mol/nm² per binder heavy atom.
- `AUTOSTOP_NC = 0.01`, `AUTOSTOP_NC_WINDOW = 5`, `AUTOSTOP_MIN_DISPLACEMENT = 0.5` nm.
- Binder residues start at **451** in prmtop global numbering. `inputs/residue_map.csv` is per-chain and must never be used as a selection source.
- run_id format is exactly `f"00{seed:02d}{rep:02d}"`; seed and rep each in 1..59.
- Slow GPUs — always excluded in the sbatch template, never the submit line:
  `nodea0110,nodea0111,nodeb0417,nodeb0419,nodeb0423,nodeb0425,nodeb0427,nodeb0429,nodeb0431,nodeb0433,nodeb1201,nodeb1203,nodeb1205,nodeb1207,nodeb1209,nodeb1215,nodeb1217`
- Terse code comments. Never put project names or benchmark numbers in any file that lands under `autopath/`.
- Commit messages: short and concise. No `Co-Authored-By` trailer.

## Review Focus

1. **A seed State whose NPT box has breathed smaller than the reference** → the PBC cap clips below 2700 moves and the replica silently pulls a shorter distance than its peers, biasing that seed's work low. Preflight must check **every seed**, not one representative box. (Task 4)
2. **`extrapolate_to_v0` R² = 1.0 on two-speed rows** — its own docstring says edge rows with only `min_speeds` points are trivially perfect with zero degrees of freedom. A naive R² ≥ 0.9 gate passes on meaningless rows. The gate must require all 3 speeds contributed. (Task 9)
3. **A truncated `.dat` from a crashed or preempted replica** → a partial pull is indistinguishable from a complete one by row count alone and enters the ensemble as a spuriously low barrier. Must be detected and excluded, not silently averaged. (Task 9)
4. **`binder_range` detecting the wrong boundary** — this exact bug invalidated every prior RMSD number on this project. Detection must be verified against `inputs/binder.pdb` resnames, and must raise rather than fall back. (Task 2)
5. **autostop never firing, i.e. hitting the 2700-move cap with contacts still present** → the barrier is a lower bound, not a barrier, and averaging it with detached replicas is meaningless. Must be reported as a distinct outcome from "fired". (Task 9)

---

### Task 1: Version the target project

**Files:**
- Create: `$PROJ/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a git repo at `$PROJ` so later tasks can commit.

> **Veto point.** This runs `git init` in a directory holding ~500 GB of simulation data. The `.gitignore` excludes all of it; only scripts, JSON and small CSV are tracked. If the user declines, skip this task and drop every `git commit` step in later tasks — nothing else changes.

- [ ] **Step 1: Confirm it is not already a repo**

Run: `cd /gpfs/group/forli/mllanos/peter/minibinders && git rev-parse --show-toplevel`
Expected: `fatal: not a git repository`

- [ ] **Step 2: Write the .gitignore**

```gitignore
systems/
mmpbsa/
trimmed/
logs/
__pycache__/
*.pyc
*.dcd
*.nc
*.xml
*.prmtop
*.rst7
*.chk
*.png
*.svg
*.pdf
analysis/*.csv
!prep/pull_selections.json
!analysis/membrane_params.json
```

- [ ] **Step 3: Init and verify nothing large is staged**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git init -q && git add -A
git status --porcelain | wc -l          # expect a small number (<40)
du -ch $(git diff --cached --name-only) | tail -1   # expect < 5 MB
```

If the size check exceeds 5 MB, stop and widen `.gitignore` before committing.

- [ ] **Step 4: Commit**

```bash
git commit -q -m "track scripts and configs, ignore simulation data"
```

---

### Task 2: `smd_campaign.py` — binder range and selection loading

**Files:**
- Create: `$PROJ/prep/smd_campaign.py`
- Test: `$PROJ/prep/test_smd_campaign.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SYSTEMS: tuple[str, ...]` — the 7 system names.
  - `SPEEDS: tuple[float, float, float]` — `(0.001, 0.005, 0.010)`.
  - `binder_range(prmtop: str, binder_pdb: str) -> tuple[int, int]` — 1-based inclusive global resid range of the binder; raises `ValueError` on mismatch.
  - `load_selections(path: str) -> dict` — parsed `pull_selections.json`.
  - `group_indices(prmtop: str, sel: dict, sysname: str) -> tuple[list[int], list[int]]` — `(groupA_atoms, groupB_atoms)` as 0-based atom indices.

- [ ] **Step 1: Write the failing tests**

```python
# $PROJ/prep/test_smd_campaign.py
import json, pytest
from pathlib import Path
import smd_campaign as sc

PROJ = Path("/gpfs/group/forli/mllanos/peter/minibinders")

def _paths(s):
    return (str(PROJ / f"systems/{s}/system.prmtop"),
            str(PROJ / f"systems/{s}/inputs/binder.pdb"))

@pytest.mark.parametrize("sysname,expected", [
    ("slc_m1", (451, 529)), ("slc_m2", (451, 530)), ("slc_m3", (451, 533)),
    ("slc_m4", (451, 533)), ("slc_m5", (451, 533)), ("slc_m6", (451, 530)),
    ("slc_m7", (451, 509)),
])
def test_binder_range_matches_measured(sysname, expected):
    assert sc.binder_range(*_paths(sysname)) == expected

def test_binder_range_raises_on_resname_mismatch(tmp_path):
    prmtop, _ = _paths("slc_m1")
    bogus = tmp_path / "bogus.pdb"
    bogus.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n")
    with pytest.raises(ValueError, match="does not match"):
        sc.binder_range(prmtop, str(bogus))

def test_systems_and_speeds_are_frozen():
    assert sc.SYSTEMS == ("slc_m1","slc_m2","slc_m3","slc_m4","slc_m5","slc_m6","slc_m7")
    assert sc.SPEEDS == (0.001, 0.005, 0.010)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smd_campaign'`

- [ ] **Step 3: Implement**

```python
# $PROJ/prep/smd_campaign.py
"""Pure campaign logic for the minibinder sMD run. No OpenMM execution here."""
import json
import warnings

import numpy as np

SYSTEMS = ("slc_m1", "slc_m2", "slc_m3", "slc_m4", "slc_m5", "slc_m6", "slc_m7")
SPEEDS = (0.001, 0.005, 0.010)          # nm/ps == 1, 5, 10 m/s
MAX_DISPLACEMENT = 2.7                  # nm
DX_PER_MOVE = 0.001                     # nm
EXPECTED_MOVES = int(round(MAX_DISPLACEMENT / DX_PER_MOVE))
SPRING_PER_ATOM = 100 * 4.184           # kJ/mol/nm^2 per binder heavy atom
TIMESTEP = 0.004                        # ps
TEMPERATURE = 300.0                     # K
SAVE_FREQ = 5
_CAPS = ("ACE", "NME", "NMA")


def _universe(prmtop, coords=None):
    import MDAnalysis as mda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(prmtop, coords) if coords else mda.Universe(prmtop)


def binder_range(prmtop, binder_pdb):
    """Global 1-based resid range of the binder, verified against binder.pdb.

    The prmtop numbers residues globally; residue_map.csv numbers them per
    chain and is NOT a valid selection source. Raises rather than falling back.
    """
    u = _universe(prmtop)
    names = list(u.select_atoms("protein").residues.resnames)
    caps = [k for k, r in enumerate(names) if r in _CAPS]
    if not caps:
        raise ValueError(f"{prmtop}: no ACE/NME/NMA cap found, cannot locate binder")
    lo, hi = caps[-1] + 2, len(names)
    target = list(_universe(binder_pdb).residues.resnames)
    if names[lo - 1:hi] != target:
        raise ValueError(
            f"{prmtop}: residues {lo}-{hi} does not match {binder_pdb} "
            f"({len(names[lo-1:hi])} vs {len(target)} residues)")
    return lo, hi


def load_selections(path):
    with open(path) as f:
        return json.load(f)


def group_indices(prmtop, sel, sysname):
    """(groupA, groupB) 0-based atom indices: binder CA, shared receptor CA."""
    lo, hi = sel["systems"][sysname]["binder_resid_range"]
    u = _universe(prmtop)
    a = u.select_atoms(f"resid {lo}:{hi} and name CA")
    pocket = " ".join(map(str, sel["shared_pocket_resids"]))
    b = u.select_atoms(f"protein and name CA and resid {pocket}")
    if a.n_atoms != hi - lo + 1:
        raise ValueError(f"{sysname}: expected {hi-lo+1} binder CA, got {a.n_atoms}")
    if b.n_atoms != len(sel["shared_pocket_resids"]):
        raise ValueError(f"{sysname}: expected {len(sel['shared_pocket_resids'])} "
                         f"pocket CA, got {b.n_atoms}")
    return list(map(int, a.indices)), list(map(int, b.indices))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add prep/smd_campaign.py prep/test_smd_campaign.py
git commit -q -m "add binder range detection and selection loading"
```

---

### Task 3: run-id encoding and SLURM array decoding

**Files:**
- Modify: `$PROJ/prep/smd_campaign.py` (append)
- Modify: `$PROJ/prep/test_smd_campaign.py` (append)

**Interfaces:**
- Consumes: `SPEEDS` from Task 2.
- Produces:
  - `run_id(seed: int, rep: int = 1) -> str` — the 6-char id.
  - `decode_task(task_id: int) -> tuple[int, float]` — `(seed, speed)` for a 1-based SLURM array index.
  - `N_SEEDS: int = 25`, `N_TASKS: int = 75`.

- [ ] **Step 1: Write the failing tests**

The round-trip test imports autopath's real parsers, so the encoding is pinned against the code that actually consumes it rather than a copy of its logic.

```python
# append to $PROJ/prep/test_smd_campaign.py
from autopath.pulling.SMDData import SMDData

def test_run_id_format():
    assert sc.run_id(1) == "000101"
    assert sc.run_id(25) == "002501"
    assert sc.run_id(7, 3) == "000703"

@pytest.mark.parametrize("bad", [0, 60, -1])
def test_run_id_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        sc.run_id(bad)

def test_run_id_parses_with_autopath_and_sorts_seed_major(tmp_path):
    """Pins the encoding against autopath's real filename parsers."""
    import os, time
    logs = []
    for seed in range(1, 26):
        for v in sc.SPEEDS:
            p = tmp_path / f"sMD_replica-{sc.run_id(seed)}_v{v}_forward.dat"
            p.write_text("# x=1\nstep,time\n")
            logs.append(str(p))
    # same mtime for all, so ordering comes purely from the encoded id
    for p in logs:
        os.utime(p, (1_800_000_000, 1_800_000_000))
    slow = [f for f in logs if SMDData._speed_from_log(f) == 0.001]
    order = sorted(slow, key=SMDData._replica_start_datetime)
    seeds = [SMDData._replica_idx_from_log(f) // 100 for f in order]
    assert seeds == list(range(1, 26))

def test_decode_task_covers_grid_exactly_once():
    seen = [sc.decode_task(t) for t in range(1, sc.N_TASKS + 1)]
    assert len(set(seen)) == sc.N_TASKS
    assert sorted(set(s for s, _ in seen)) == list(range(1, sc.N_SEEDS + 1))
    assert sc.decode_task(1) == (1, 0.001)
    assert sc.decode_task(3) == (1, 0.010)
    assert sc.decode_task(75) == (25, 0.010)

@pytest.mark.parametrize("bad", [0, 76])
def test_decode_task_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        sc.decode_task(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -k "run_id or decode" -v`
Expected: FAIL with `AttributeError: module 'smd_campaign' has no attribute 'run_id'`

- [ ] **Step 3: Implement**

```python
# append to $PROJ/prep/smd_campaign.py
N_SEEDS = 25
N_TASKS = N_SEEDS * len(SPEEDS)


def run_id(seed, rep=1):
    """Encode provenance in a 6-digit id.

    Must be int-parseable (SMDData._replica_idx_from_log calls int()) AND a
    valid HHMMSS (_replica_start_datetime feeds it to datetime.replace), so
    hh=00, mm=seed, ss=rep. Sorting by that datetime is then seed-major,
    which makes force-ladder rung k mean "seeds 1..k".
    """
    if not 1 <= seed <= 59:
        raise ValueError(f"seed must be in 1..59, got {seed}")
    if not 1 <= rep <= 59:
        raise ValueError(f"rep must be in 1..59, got {rep}")
    return f"00{seed:02d}{rep:02d}"


def decode_task(task_id):
    """1-based SLURM array index -> (seed, speed). Speed varies fastest."""
    if not 1 <= task_id <= N_TASKS:
        raise ValueError(f"task_id must be in 1..{N_TASKS}, got {task_id}")
    i = task_id - 1
    return i // len(SPEEDS) + 1, SPEEDS[i % len(SPEEDS)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add prep/smd_campaign.py prep/test_smd_campaign.py
git commit -q -m "add seed-major run id encoding and array task decoding"
```

---

### Task 4: preflight geometry — per-seed PBC cap check

Covers Review Focus #1.

**Files:**
- Modify: `$PROJ/prep/smd_campaign.py` (append)
- Modify: `$PROJ/prep/test_smd_campaign.py` (append)

**Interfaces:**
- Consumes: `group_indices`, `MAX_DISPLACEMENT`, `EXPECTED_MOVES`.
- Produces:
  - `pbc_cap(com_a, com_b, box_lengths, safety=0.5) -> tuple[float, np.ndarray, float]` — `(r0, unit_vector, r_max)`; pure numpy, no I/O.
  - `preflight(prmtop, state_xml, sel, sysname) -> dict` — keys `r0`, `dir_z`, `r_max`, `moves_available`, `ok`.

- [ ] **Step 1: Write the failing tests**

```python
# append to $PROJ/prep/test_smd_campaign.py
import numpy as np

def test_pbc_cap_pure_geometry_along_z():
    # pull straight down z in a 10x10x8 nm box: cap = 8/2 - 0.5 = 3.5 nm
    r0, u, rmax = sc.pbc_cap(np.array([0., 0., -1.5]), np.zeros(3),
                             np.array([10., 10., 8.]), safety=0.5)
    assert r0 == pytest.approx(1.5)
    assert u[2] == pytest.approx(-1.0)
    assert rmax == pytest.approx(3.5)

def test_pbc_cap_applies_minimum_image():
    # separated by 9 nm in a 10 nm box -> minimum image is 1 nm, not 9
    r0, _, _ = sc.pbc_cap(np.array([9., 0., 0.]), np.zeros(3),
                          np.array([10., 10., 10.]))
    assert r0 == pytest.approx(1.0)

def test_pbc_cap_diagonal_pull_uses_tightest_axis():
    # equal x/z components, z box half as big -> z binds
    r0, _, rmax = sc.pbc_cap(np.array([1., 0., 1.]), np.zeros(3),
                             np.array([20., 20., 10.]), safety=0.0)
    assert rmax == pytest.approx((10. / 2) * r0 / 1.0)

@pytest.mark.parametrize("sysname", sc.SYSTEMS)
def test_preflight_every_seed_has_room(sysname):
    """Review Focus #1: check every seed's own box, not one representative."""
    sel = sc.load_selections(str(PROJ / "prep/pull_selections.json"))
    prmtop = str(PROJ / f"systems/{sysname}/system.prmtop")
    bad = []
    for seed in range(1, sc.N_SEEDS + 1):
        sx = PROJ / f"systems/{sysname}/replicas/rep_{seed:02d}/{sysname}_rep{seed:02d}_final.xml"
        r = sc.preflight(prmtop, str(sx), sel, sysname)
        assert 1.0 < r["r0"] < 2.2, (sysname, seed, r["r0"])
        assert r["dir_z"] < -0.85, (sysname, seed, r["dir_z"])
        if not r["ok"]:
            bad.append((seed, r["moves_available"]))
    assert not bad, f"{sysname}: seeds with < {sc.EXPECTED_MOVES} moves: {bad}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -k "pbc_cap or preflight" -v`
Expected: FAIL — `pbc_cap` undefined. (The `preflight` tests will also fail on the missing `pull_selections.json`; Task 5 creates it. Re-run them at the end of Task 5.)

- [ ] **Step 3: Implement**

```python
# append to $PROJ/prep/smd_campaign.py
def pbc_cap(com_a, com_b, box_lengths, safety=0.5):
    """Mirror SteeredMD's cap: r_max = min_i (L_i/2)*|v|/|v_i| - safety.

    Orthorhombic only, which these systems are.
    """
    L = np.asarray(box_lengths, dtype=float)
    v = np.asarray(com_a, dtype=float) - np.asarray(com_b, dtype=float)
    v -= np.round(v / L) * L                       # minimum image
    r0 = float(np.linalg.norm(v))
    av = np.abs(v)
    per_axis = np.where(av > 1e-6, (L / 2.0) * r0 / av, np.inf)
    return r0, v / r0, float(np.min(per_axis)) - safety


def _state_box_and_positions(state_xml):
    """Box lengths (nm) and positions (nm) from a serialized OpenMM State."""
    from openmm import XmlSerializer, unit
    with open(state_xml) as f:
        st = XmlSerializer.deserialize(f.read())
    box = np.array([v.value_in_unit(unit.nanometer)[i]
                    for i, v in enumerate(st.getPeriodicBoxVectors())])
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return box, np.asarray(pos)


def preflight(prmtop, state_xml, sel, sysname):
    """Geometry check for one seed. Run before burning GPU hours."""
    a_idx, b_idx = group_indices(prmtop, sel, sysname)
    box, pos = _state_box_and_positions(state_xml)
    # CA atoms: equal masses, so the mass-weighted centroid is the mean
    r0, u, r_max = pbc_cap(pos[a_idx].mean(axis=0), pos[b_idx].mean(axis=0), box)
    # min(r_max - r0, MAX_DISPLACEMENT), never (r0 + MAX) - r0: the latter loses a
    # ULP for some r0 (e.g. 1.355 -> 2.6999999999999997) and int() then truncates a
    # full move. The 1e-6 tolerance is 1e-9 nm, far below any real shortfall.
    avail = min(r_max - r0, MAX_DISPLACEMENT)
    moves = max(0, int(avail / DX_PER_MOVE + 1e-6))
    return {"r0": r0, "dir_z": float(u[2]), "r_max": r_max,
            "moves_available": moves, "ok": moves >= EXPECTED_MOVES}
```

- [ ] **Step 4: Run the pure-geometry tests**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -k "pbc_cap" -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add prep/smd_campaign.py prep/test_smd_campaign.py
git commit -q -m "add per-seed preflight PBC cap check"
```

---

### Task 5: derive and freeze `pull_selections.json`

**Files:**
- Create: `$PROJ/prep/make_selections.py`
- Create: `$PROJ/prep/pull_selections.json` (generated)

**Interfaces:**
- Consumes: `binder_range`, `SYSTEMS` from Task 2.
- Produces: `pull_selections.json` with keys `shared_pocket_resids` (list[int], length 98) and `systems.<name>.binder_resid_range` ([int, int]).

- [ ] **Step 1: Write the generator**

```python
# $PROJ/prep/make_selections.py
"""One-off: derive the shared helix-helix pulling selection. Output is frozen
into pull_selections.json and reviewed; the campaign reads the JSON, not this.

groupB = helical receptor residues in persistent (>=50% of sampled frames)
heavy-atom contact within 6 A of a helical binder residue, unioned over all
systems so every design shares one reaction coordinate.
"""
import json, sys, warnings
import numpy as np, MDAnalysis as mda, mdtraj as md
from MDAnalysis.lib.distances import distance_array
import smd_campaign as sc

warnings.filterwarnings("ignore")
PROJ = "/gpfs/group/forli/mllanos/peter/minibinders"
N_FRAMES, CUT, MIN_FREQ = 40, 6.0, 0.5

per_system, pockets = {}, []
for s in sc.SYSTEMS:
    prm = f"{PROJ}/systems/{s}/system.prmtop"
    dcd = f"{PROJ}/systems/{s}/continuation/cont_{s}.dcd"
    lo, hi = sc.binder_range(prm, f"{PROJ}/systems/{s}/inputs/binder.pdb")
    u = mda.Universe(prm, dcd)
    binder_ha = u.select_atoms(f"resid {lo}:{hi} and not name H*")
    rec_ha = u.select_atoms(f"protein and resid 1:{lo-1} and not name H*")
    rec_rids = np.array([a.resid for a in rec_ha])
    bnd_rids = np.array([a.resid for a in binder_ha])
    frames = np.unique(np.linspace(0, len(u.trajectory) - 1, N_FRAMES).astype(int))

    top = md.load_prmtop(prm)
    t = md.load_dcd(dcd, top=top, atom_indices=top.select("protein"))[frames]
    ss = md.compute_dssp(t, simplified=False)
    rseq = np.array([r.resSeq for r in t.topology.residues])
    helix = set(int(r) for r in rseq[np.mean(np.isin(ss, ["H", "G", "I"]), axis=0) >= 0.5])
    bmask = np.isin(bnd_rids, list(helix))
    if not bmask.any():
        raise SystemExit(f"{s}: no helical binder residues")

    freq = {}
    for ts in u.trajectory[frames]:
        dm = distance_array(rec_ha.positions, binder_ha.positions[bmask], box=ts.dimensions)
        for r in set(rec_rids[(dm < CUT).any(axis=1)]):
            freq[int(r)] = freq.get(int(r), 0) + 1
    pocket = {r for r, c in freq.items() if c / len(frames) >= MIN_FREQ} & helix
    pockets.append(pocket)
    per_system[s] = {"binder_resid_range": [lo, hi],
                     "pocket_resids_this_system": sorted(pocket)}
    print(f"{s}: binder {lo}-{hi}, pocket {len(pocket)}", flush=True)

shared = sorted(set().union(*pockets))
out = {
    "_derivation": (f"helical receptor residues within {CUT} A (heavy atom) of a "
                    f"helical binder residue in >= {MIN_FREQ:.0%} of {N_FRAMES} "
                    "continuation frames, unioned over all systems"),
    "shared_pocket_resids": shared,
    "systems": per_system,
}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print(f"shared pocket: {len(shared)} residues -> {sys.argv[1]}")
```

- [ ] **Step 2: Generate**

Run: `cd $PROJ/prep && micromamba run -n autopath python make_selections.py pull_selections.json`
Expected: 7 per-system lines, then `shared pocket: 98 residues`. Takes ~5 min (DSSP over 7 trajectories).

- [ ] **Step 3: Verify it reproduces the reviewed selection**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/prep
micromamba run -n autopath python -c "
import json; d=json.load(open('pull_selections.json'))
p=d['shared_pocket_resids']
assert len(p)==98, len(p)
assert p[0]==12 and p[-1]==444
assert d['systems']['slc_m7']['binder_resid_range']==[451,509]
print('OK', len(p), 'residues')"
```
Expected: `OK 98 residues`. If the count differs, stop — the derivation has drifted from the reviewed spec and must be reconciled before anything runs.

- [ ] **Step 4: Run the preflight tests deferred from Task 4**

Run: `cd $PROJ/prep && micromamba run -n autopath python -m pytest test_smd_campaign.py -v`
Expected: all passed, including the 7 `test_preflight_every_seed_has_room` cases. **If any seed reports `ok=False`, stop and lower `MAX_DISPLACEMENT` until every seed passes** — a mixed protocol grid invalidates the multi-speed analysis.

- [ ] **Step 5: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add prep/make_selections.py prep/pull_selections.json
git commit -q -m "derive and freeze shared helix-helix pulling selection"
```

---

### Task 6: `run_smd.py` driver and DCD stripping

**Files:**
- Create: `$PROJ/prep/run_smd.py`
- Create: `$PROJ/prep/strip_dcd.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: `systems/<sys>/sMD/trajectories/sMD_replica-00NN01_v<speed>_forward.{dat,dcd,csv}` plus `_geom.dat`.

- [ ] **Step 1: Write the stripper**

```python
# $PROJ/prep/strip_dcd.py
"""Strip an sMD DCD to protein only (21x smaller) and replace the original."""
import os, sys, warnings
import MDAnalysis as mda

warnings.filterwarnings("ignore")


def strip(prmtop, dcd, out=None):
    u = mda.Universe(prmtop, dcd)
    prot = u.select_atoms("protein")
    out = out or dcd.replace(".dcd", "_prot.dcd")
    with mda.Writer(out, prot.n_atoms) as w:
        for _ in u.trajectory:
            w.write(prot)
    return out


if __name__ == "__main__":
    prmtop, dcd = sys.argv[1], sys.argv[2]
    out = strip(prmtop, dcd)
    if os.path.getsize(out) < os.path.getsize(dcd):
        os.replace(out, dcd)            # keep the original filename
        print(f"stripped {dcd}")
    else:
        os.remove(out)
        raise SystemExit(f"refusing to replace {dcd}: stripped file is not smaller")
```

- [ ] **Step 2: Write the driver**

```python
# $PROJ/prep/run_smd.py
"""Run one sMD replica from one equilibrated end point.

    python run_smd.py -n slc_m2 -k 7 -v 0.001
"""
import argparse, os, sys, warnings

import smd_campaign as sc
from autopath.pulling.steered_md import SteeredMD
from autopath.utils import load_system
from openmm.app import PDBFile

warnings.filterwarnings("ignore")
PROJ = "/gpfs/group/forli/mllanos/peter/minibinders"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--name", required=True, choices=list(sc.SYSTEMS))
    p.add_argument("-k", "--seed", type=int, required=True)
    p.add_argument("-v", "--speed", type=float, required=True)
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--no-strip", action="store_true")
    a = p.parse_args()
    if a.speed not in sc.SPEEDS:
        sys.exit(f"speed {a.speed} not in {sc.SPEEDS}")

    sysdir = f"{PROJ}/systems/{a.name}"
    prmtop = f"{sysdir}/system.prmtop"
    state_xml = (f"{sysdir}/replicas/rep_{a.seed:02d}/"
                 f"{a.name}_rep{a.seed:02d}_final.xml")
    sel = sc.load_selections(f"{PROJ}/prep/pull_selections.json")

    pre = sc.preflight(prmtop, state_xml, sel, a.name)
    print(f"preflight: r0={pre['r0']:.3f} nm  dir_z={pre['dir_z']:+.3f}  "
          f"moves_available={pre['moves_available']} (need {sc.EXPECTED_MOVES})")
    if not pre["ok"]:
        sys.exit(f"preflight FAILED: only {pre['moves_available']} moves available")

    groupA, groupB = sc.group_indices(prmtop, sel, a.name)
    lo, hi = sel["systems"][a.name]["binder_resid_range"]
    import MDAnalysis as mda
    n_heavy = mda.Universe(prmtop).select_atoms(
        f"resid {lo}:{hi} and not name H*").n_atoms
    k = sc.SPRING_PER_ATOM * n_heavy

    out_dir = f"{sysdir}/sMD/trajectories"
    topology = PDBFile(f"{sysdir}/system.pdb").topology
    system = load_system(f"{sysdir}/equilibration/system_equil_{a.name}.xml")

    smd = SteeredMD(
        system=system, topology=topology,
        groupA_atoms=groupA, groupB_atoms=groupB,
        restart_velocities=True,
        timestep=sc.TIMESTEP, temperature=sc.TEMPERATURE,
        out_dir=out_dir, platform="fastest",   # SteeredMD calls select_platform itself
        dx_per_move=sc.DX_PER_MOVE,
        max_displacement=sc.MAX_DISPLACEMENT,
        sMD_spring_cte=k,
        sMD_max_r_offset=sc.MAX_DISPLACEMENT,
        autostop_nc=0.01, autostop_nc_window=5,
        autostop_min_displacement=0.5,
        save_freq=sc.SAVE_FREQ,
        log_geom_features=True,
    )
    print(f"{a.name} seed {a.seed:02d} v={a.speed} nm/ps  k={k:.4g} kJ/mol/nm2  "
          f"groupA={len(groupA)} CA  groupB={len(groupB)} CA")
    rid = smd.run(pulling_speed=a.speed, pulling_direction="forward",
                  run_id=sc.run_id(a.seed, a.rep), state_xml_file=state_xml)

    if not a.no_strip:
        from strip_dcd import strip
        dcd = f"{out_dir}/sMD_{rid}.dcd"
        if os.path.exists(dcd):
            out = strip(prmtop, dcd)
            if os.path.getsize(out) < os.path.getsize(dcd):
                os.replace(out, dcd)
            else:
                os.remove(out)
    print("done", rid)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the preflight path without running MD**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/prep
micromamba run -n autopath python -c "
import smd_campaign as sc
sel = sc.load_selections('pull_selections.json')
for s in ('slc_m2','slc_m7'):
    r = sc.preflight(f'/gpfs/group/forli/mllanos/peter/minibinders/systems/{s}/system.prmtop',
                     f'/gpfs/group/forli/mllanos/peter/minibinders/systems/{s}/replicas/rep_01/{s}_rep01_final.xml',
                     sel, s)
    print(s, {k: (round(v,3) if isinstance(v,float) else v) for k,v in r.items()})"
```
Expected: both print `ok: True` with `moves_available` ≥ 2700 and `dir_z` ≈ −0.95.

- [ ] **Step 4: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add prep/run_smd.py prep/strip_dcd.py
git commit -q -m "add sMD replica driver and DCD stripping"
```

---

### Task 7: SLURM array template

**Files:**
- Create: `$PROJ/qfiles/smd_ARRAY.q`

**Interfaces:**
- Consumes: `run_smd.py`, `decode_task`.
- Produces: one replica per array task.

- [ ] **Step 1: Confirm the slow-node list is still current**

Run: `sinfo -N -h -p forli,alphafold,gpu,forli-pro -o "%.12N %.12P %.9T %.22G" | sort -u -k1,1`
Compare against the Global Constraints list; node names drift. Update the template's `--exclude` if the gtx1080/gtx1080ti/titanv set has changed.

- [ ] **Step 2: Write the template**

```bash
#!/bin/bash
# Mirrors the project's proven qfiles/rep_ARRAY.q: same partition, exclude list,
# cpus, absolute log paths, nvcc export and bashrc-based env activation.
#SBATCH -e /gpfs/group/forli/mllanos/peter/minibinders/logs/smd_%x_%a.err
#SBATCH -o /gpfs/group/forli/mllanos/peter/minibinders/logs/smd_%x_%a.out
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --partition=forli,alphafold,gpu,forli-pro
#SBATCH --exclude=nodea0110,nodea0111,nodeb0417,nodeb0419,nodeb0423,nodeb0425,nodeb0427,nodeb0429,nodeb0431,nodeb0433,nodeb1201,nodeb1203,nodeb1205,nodeb1207,nodeb1209,nodeb1215,nodeb1217
#SBATCH --cpus-per-task=8
#SBATCH --array=1-75

# submit with:  sbatch --job-name=slc_m2 qfiles/smd_ARRAY.q
export OPENMM_CUDA_COMPILER=$(which nvcc)

source ~/.bashrc
micromamba activate autopath

PROJ=/gpfs/group/forli/mllanos/peter/minibinders
SYS="${SLURM_JOB_NAME}"
cd "$PROJ/prep"

read -r SEED SPEED < <(python -c "
import smd_campaign as sc
s, v = sc.decode_task(${SLURM_ARRAY_TASK_ID})
print(s, v)")

echo "task ${SLURM_ARRAY_TASK_ID}: $SYS seed $SEED speed $SPEED on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "start $(date)"
python run_smd.py -n "$SYS" -k "$SEED" -v "$SPEED"
echo "end $(date)"
```

The throughput guard lives in `SteeredMD` itself only for the continuation
script, so verify speed from the emitted `.csv` after the pilot (Task 9,
criterion 6) rather than aborting mid-run here — an sMD replica is short
enough that a slow card wastes hours, not days, and the `--exclude` list is
the primary protection.

- [ ] **Step 3: Verify the decoder is callable exactly as the template calls it**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/prep
for t in 1 2 3 74 75; do
  micromamba run -n autopath python -c "
import smd_campaign as sc; s,v = sc.decode_task($t); print($t, s, v)"
done
```
Expected: `1 1 0.001`, `2 1 0.005`, `3 1 0.01`, `74 25 0.005`, `75 25 0.01`.

- [ ] **Step 4: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add qfiles/smd_ARRAY.q
git commit -q -m "add sMD slurm array template"
```

---

### Task 8: Launch the pilot — slc_m2, seeds 1–5

**Files:** none created; produces data under `$PROJ/systems/slc_m2/sMD/trajectories/`.

**Interfaces:**
- Consumes: Tasks 5–7.
- Produces: 15 `.dat` + 15 `_geom.dat` + 15 stripped `.dcd`.

slc_m2 is the PBC-tightest system, so if 2.7 nm works there it works everywhere.

- [ ] **Step 1: Dry-run one replica interactively at the fastest speed**

The 10 m/s replica is only 270 ps, so it finishes in minutes and exercises the whole path end to end before 15 jobs are queued.

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/prep
micromamba run -n autopath python run_smd.py -n slc_m2 -k 1 -v 0.01
```
Expected: preflight prints `ok`, the run logs `Initial COM distance`, **no** `Capping sMD_moves` warning, and `sMD_replica-000101_v0.01_forward.dat` appears with ~2700 data rows.

- [ ] **Step 2: Confirm the log is well formed**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/systems/slc_m2/sMD/trajectories
head -7 sMD_replica-000101_v0.01_forward.dat
micromamba run -n autopath python -c "
from autopath.pulling.SMDData import SMDData
m = SMDData.parse_log_metadata('sMD_replica-000101_v0.01_forward.dat')
print(m)
assert abs(m['realized_speed_nm_per_ps']/m['requested_speed_nm_per_ps']-1) < 0.05"
```
Expected: metadata header parses; realized speed within 5% of requested.

- [ ] **Step 3: Submit the remaining pilot tasks**

Seeds 1–5 × 3 speeds = array indices 1–15, minus index 3 (seed 1, 0.01) already done.

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
sbatch --job-name=slc_m2 --array=1-2,4-15 qfiles/smd_ARRAY.q
```

- [ ] **Step 4: Wait for completion and count outputs**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/systems/slc_m2/sMD/trajectories
ls sMD_replica-*_forward.dat | wc -l     # expect 15
ls sMD_replica-*_geom.dat | wc -l        # expect 15
du -sh .
```

- [ ] **Step 5: No commit** (data is gitignored). Record the wall-times from `logs/` for Task 9 criterion 6.

---

### Task 9: `smd_gate.py` — evaluate the six pilot criteria

Covers Review Focus #2, #3 and #5.

**Files:**
- Create: `$PROJ/analysis/smd_gate.py`
- Test: `$PROJ/analysis/test_smd_gate.py`

**Interfaces:**
- Consumes: pilot `.dat` logs.
- Produces:
  - `read_log(path) -> tuple[pd.DataFrame, dict]` — rows and metadata.
  - `classify_replica(df, meta, expected_moves) -> dict` — keys `outcome` (`"detached"` | `"hit_cap"` | `"truncated"`), `n_rows`, `speed_error`, `final_displacement`.
  - `r2_all_speeds(extrap_df, n_speeds) -> pd.DataFrame` — extrapolation rows restricted to those where **all** speeds contributed. The R² column is named `R2` (verified in `Estimators.py`); the `mixedlm` branch writes `R2 = NaN`, so callers must check for NaN before gating.
  - `gate_report(traj_dir, expected_moves, n_speeds) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Synthetic logs, so the classifier is pinned without needing simulation data.

```python
# $PROJ/analysis/test_smd_gate.py
import numpy as np, pandas as pd, pytest
import smd_gate as g

HDR = ("# spring_constant_kJ_mol_nm2=250000\n"
       "# requested_speed_nm_per_ps={req}\n"
       "# realized_speed_nm_per_ps={real}\n"
       "# steps_per_move=250\n# force_n_samples=10\n")
COLS = "step,time,r_target,r_before,r_after,force,force_sem,U_cvpack,dW_protocol\n"

def _write(tmp_path, n_rows, req=0.001, real=0.001, name="sMD_replica-000101_v0.001_forward.dat"):
    p = tmp_path / name
    rows = "".join(f"{i},{i*1.0},{1.5+i*0.001},{1.5+i*0.001},{1.5+i*0.001},"
                   f"10.0,0.1,1.0,0.01\n" for i in range(n_rows))
    p.write_text(HDR.format(req=req, real=real) + COLS + rows)
    return str(p)

def test_full_length_replica_is_hit_cap(tmp_path):
    df, meta = g.read_log(_write(tmp_path, 2700))
    assert g.classify_replica(df, meta, 2700)["outcome"] == "hit_cap"

def test_short_replica_past_min_displacement_is_detached(tmp_path):
    # autostop fired at 1200 moves = 1.2 nm, past the 0.5 nm guard
    df, meta = g.read_log(_write(tmp_path, 1200))
    r = g.classify_replica(df, meta, 2700)
    assert r["outcome"] == "detached"
    assert r["final_displacement"] == pytest.approx(1.199, abs=1e-2)

def test_truncated_replica_is_flagged_not_counted_as_detached(tmp_path):
    """Review Focus #3: a crash before the min-displacement guard is not a detachment."""
    df, meta = g.read_log(_write(tmp_path, 300))   # 0.3 nm < 0.5 nm guard
    assert g.classify_replica(df, meta, 2700)["outcome"] == "truncated"

def test_speed_error_is_reported(tmp_path):
    df, meta = g.read_log(_write(tmp_path, 2700, req=0.001, real=0.00109))
    assert g.classify_replica(df, meta, 2700)["speed_error"] == pytest.approx(0.09, abs=1e-3)

def test_r2_gate_drops_rows_missing_a_speed(tmp_path):
    """Review Focus #2: two-speed rows have R2=1.0 trivially and must not count."""
    df = pd.DataFrame({"step": [1, 2, 3], "n_speeds": [3, 2, 3],
                       "R2": [0.95, 1.0, 0.92]})
    out = g.r2_all_speeds(df, n_speeds=3)
    assert list(out["step"]) == [1, 3]
    assert out["R2"].min() == pytest.approx(0.92)

def test_r2_gate_flags_mixedlm_rows_as_unusable(tmp_path):
    """extrapolate_to_v0's mixedlm branch writes R2=NaN; it cannot be gated on."""
    df = pd.DataFrame({"step": [1, 2], "n_speeds": [3, 3],
                       "R2": [np.nan, 0.95], "model": ["mixedlm", "linear"]})
    out = g.r2_all_speeds(df, n_speeds=3)
    assert out["R2"].isna().sum() == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd $PROJ/analysis && micromamba run -n autopath python -m pytest test_smd_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smd_gate'`

- [ ] **Step 3: Implement**

```python
# $PROJ/analysis/smd_gate.py
"""Evaluate the six pilot go/no-go criteria from the campaign spec."""
import glob, os

import pandas as pd

from autopath.pulling.SMDData import SMDData

MIN_DISPLACEMENT = 0.5      # nm — autostop's own guard
SPEED_TOL = 0.05


def read_log(path):
    df = pd.read_csv(path, comment="#")
    return df, SMDData.parse_log_metadata(path)


def classify_replica(df, meta, expected_moves):
    """detached | hit_cap | truncated, plus the numbers behind the call."""
    n = len(df)
    disp = float(df["r_target"].iloc[-1] - df["r_target"].iloc[0]) if n else 0.0
    if n >= expected_moves:
        outcome = "hit_cap"                  # ran the full protocol, never detached
    elif disp >= MIN_DISPLACEMENT:
        outcome = "detached"                 # autostop fired legitimately
    else:
        outcome = "truncated"                # crashed before it could detach
    req = meta.get("requested_speed_nm_per_ps", float("nan"))
    real = meta.get("realized_speed_nm_per_ps", float("nan"))
    return {"outcome": outcome, "n_rows": n, "final_displacement": disp,
            "speed_error": abs(real / req - 1.0) if req else float("nan")}


def r2_all_speeds(extrap_df, n_speeds):
    """Keep only rows where every speed contributed.

    extrapolate_to_v0's docstring: rows with only `min_speeds` points have
    R2=1.0 trivially (zero degrees of freedom), so an R2 gate that includes
    them passes on nothing. Column is `R2`; the mixedlm branch writes NaN.
    """
    return extrap_df[extrap_df["n_speeds"] >= n_speeds].copy()


def gate_report(traj_dir, expected_moves, n_speeds=3):
    rows = []
    for f in sorted(glob.glob(os.path.join(traj_dir, "sMD_replica-*_forward.dat"))):
        df, meta = read_log(f)
        rec = classify_replica(df, meta, expected_moves)
        rec["log"] = os.path.basename(f)
        rec["speed"] = SMDData._speed_from_log(f)
        rec["seed"] = SMDData._replica_idx_from_log(f) // 100
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    rep = gate_report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2700)
    print(rep.to_string(index=False))
    print("\noutcome counts:\n", rep["outcome"].value_counts().to_string())
    print(f"\nmax speed error: {rep['speed_error'].max():.3f} (tol {SPEED_TOL})")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd $PROJ/analysis && micromamba run -n autopath python -m pytest test_smd_gate.py -v`
Expected: 5 passed

- [ ] **Step 5: Evaluate the pilot**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/analysis
micromamba run -n autopath python smd_gate.py \
  ../systems/slc_m2/sMD/trajectories 2700
grep -c "Capping sMD_moves" ../logs/smd_slc_m2_*.out || echo "criterion 3 OK: no capping"
```

Record against the spec's criteria:
1. ≥4/5 slow replicas `detached` (not `hit_cap`, not `truncated`)
2. receptor drift — measured in Step 6
3. zero `Capping sMD_moves` occurrences
4. R² — measured in Task 12 once the analysis arm exists; for the pilot, confirm ≥3 speeds present per step
5. `max speed error` < 0.05
6. wall-times from `logs/`

- [ ] **Step 6: Measure receptor drift (criterion 2)**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/analysis
micromamba run -n autopath python -c "
import warnings, numpy as np, MDAnalysis as mda; warnings.filterwarnings('ignore')
P='/gpfs/group/forli/mllanos/peter/minibinders'
u=mda.Universe(f'{P}/systems/slc_m2/system.prmtop',
               f'{P}/systems/slc_m2/sMD/trajectories/sMD_replica-000101_v0.001_forward.dcd')
rec=u.select_atoms('protein and resid 1:450 and name CA')
c=np.array([rec.center_of_mass() for _ in u.trajectory])/10.0
print('receptor COM drift (nm):', np.linalg.norm(c-c[0],axis=1).max())"
```
Expected: < 0.5 nm. If larger, add `restrained_atoms` (TM-core CA) to `run_smd.py` and re-run the pilot before scaling.

- [ ] **Step 7: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add analysis/smd_gate.py analysis/test_smd_gate.py
git commit -q -m "add pilot gate evaluation"
```

---

### Task 10: GATE — human decision

**Files:** none.

- [ ] **Step 1: Present the gate report**

Show the user the outcome table, the six criteria with pass/fail, the measured receptor drift, and the projected campaign wall-time from the pilot timings.

- [ ] **Step 2: Stop and wait**

Do not proceed to Task 11 without an explicit decision. If any criterion fails, the remedy is named in the spec (§8): lengthen the pull, add restraints, or change the speed ladder — each of which sends work back to Task 5 or 6, not forward.

---

### Task 11: Launch the full campaign

**Files:** none created; produces data under `$PROJ/systems/*/sMD/trajectories/`.

- [ ] **Step 1: Submit the 6 remaining systems**

slc_m2 needs only its remaining 20 seeds (array 16–75); the rest run the full grid.

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
sbatch --job-name=slc_m2 --array=16-75 qfiles/smd_ARRAY.q
for s in slc_m1 slc_m3 slc_m4 slc_m5 slc_m6 slc_m7; do
  sbatch --job-name=$s --array=1-75 qfiles/smd_ARRAY.q
done
squeue -u "$USER" -h | wc -l
```

- [ ] **Step 2: Verify completeness before analysing**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
for s in slc_m1 slc_m2 slc_m3 slc_m4 slc_m5 slc_m6 slc_m7; do
  n=$(ls systems/$s/sMD/trajectories/sMD_replica-*_forward.dat 2>/dev/null | wc -l)
  echo "$s: $n/75"
done
```
Expected: 75 for every system. Re-submit any missing array indices before continuing — a missing seed silently unbalances the ensemble.

- [ ] **Step 3: Re-run the gate across all systems**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/analysis
for s in slc_m1 slc_m2 slc_m3 slc_m4 slc_m5 slc_m6 slc_m7; do
  echo "=== $s ==="
  micromamba run -n autopath python smd_gate.py ../systems/$s/sMD/trajectories 2700 | tail -6
done
```
Any `truncated` replica must be re-run or excluded explicitly in Task 12 — never silently averaged.

---

### Task 12: `smd_barriers.py` — PMF, v→0 barrier, seed bootstrap

**Files:**
- Create: `$PROJ/analysis/smd_barriers.py`
- Test: `$PROJ/analysis/test_smd_barriers.py`

**Interfaces:**
- Consumes: `smd_gate.classify_replica`, `smd_gate.r2_all_speeds`.
- Produces:
  - `seed_bootstrap(per_seed_fn, seeds, n_boot=2000, seed=42) -> tuple[float, float, float]` — `(median, lo, hi)` percentile interval.
  - `usable_logs(traj_dir, expected_moves=2700) -> list[str]` — logs minus truncated replicas.
  - `work_pivot(raw_data, speeds) -> tuple[np.ndarray, list[int], np.ndarray]` — `(W[n_seed, n_speed, n_step], seeds, steps)`, NaN where a replica autostopped early.
  - `barrier_from_seeds(W, speeds, sample) -> float` — mean work over the sampled seed indices, per-step OLS intercept at v=0, then `max(pmf) - pmf[0]`.
  - `analyse_system(sysname) -> dict` — keys `system`, `barrier`, `barrier_lo`, `barrier_hi`, `r_ts`, `min_r2`, `n_seeds`, `n_excluded`.
  - `$PROJ/analysis/smd_barriers.csv` — one row per system per estimator.

- [ ] **Step 1: Write the failing test for the bootstrap**

The bootstrap is the only piece with its own arithmetic; the rest is orchestration of autopath calls.

```python
# $PROJ/analysis/test_smd_barriers.py
import numpy as np, pytest
import smd_barriers as sb

def test_seed_bootstrap_recovers_a_known_mean():
    vals = {s: 10.0 + 0.1 * s for s in range(1, 26)}   # mean 11.3
    med, lo, hi = sb.seed_bootstrap(lambda ss: np.mean([vals[s] for s in ss]),
                                    list(vals), n_boot=2000, seed=0)
    assert med == pytest.approx(11.3, abs=0.15)
    assert lo < med < hi

def test_seed_bootstrap_is_deterministic_for_a_given_seed():
    vals = {s: float(s) for s in range(1, 26)}
    f = lambda ss: np.mean([vals[s] for s in ss])
    assert sb.seed_bootstrap(f, list(vals), n_boot=500, seed=7) == \
           sb.seed_bootstrap(f, list(vals), n_boot=500, seed=7)

def test_seed_bootstrap_interval_widens_with_spread():
    tight = {s: 10.0 for s in range(1, 26)}
    wide = {s: 10.0 + (-1) ** s * 5.0 for s in range(1, 26)}
    _, l1, h1 = sb.seed_bootstrap(lambda ss: np.mean([tight[s] for s in ss]), list(tight), 500, 0)
    _, l2, h2 = sb.seed_bootstrap(lambda ss: np.mean([wide[s] for s in ss]), list(wide), 500, 0)
    assert (h2 - l2) > (h1 - l1)

def test_seed_bootstrap_rejects_too_few_seeds():
    with pytest.raises(ValueError):
        sb.seed_bootstrap(lambda ss: 0.0, [1, 2], n_boot=10, seed=0)

def test_barrier_from_seeds_recovers_a_known_intercept():
    """Work linear in v with a known v=0 PMF must give back that PMF's barrier."""
    speeds = [0.001, 0.005, 0.010]
    steps = np.arange(100)
    pmf = 20.0 * np.sin(np.pi * steps / 99)          # peak 20, pmf[0] = 0
    W = np.empty((5, 3, 100))
    for s_i in range(5):
        for v_i, v in enumerate(speeds):
            W[s_i, v_i] = pmf + 1000.0 * v           # pure dissipation, linear in v
    got = sb.barrier_from_seeds(W, speeds, np.arange(5))
    assert got == pytest.approx(pmf.max() - pmf[0], abs=1e-9)   # 19.9975, not 20

def test_barrier_from_seeds_ignores_steps_missing_a_speed():
    speeds = [0.001, 0.005, 0.010]
    W = np.zeros((3, 3, 50))
    W[:, :, :] = np.arange(50)[None, None, :]
    W[:, 2, 40:] = np.nan                            # fastest speed stopped early
    # barrier must come from steps 0..39 only, i.e. 39, not 49
    assert sb.barrier_from_seeds(W, speeds, np.arange(3)) == pytest.approx(39.0)

def test_barrier_from_seeds_returns_nan_on_too_little_overlap():
    W = np.full((3, 3, 50), np.nan)
    W[:, :, :2] = 1.0
    assert np.isnan(sb.barrier_from_seeds(W, [0.001, 0.005, 0.010], np.arange(3)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd $PROJ/analysis && micromamba run -n autopath python -m pytest test_smd_barriers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smd_barriers'`

- [ ] **Step 3: Implement**

```python
# $PROJ/analysis/smd_barriers.py
"""v->0 barriers per design with between-seed bootstrap error bars."""
import glob, os, sys, warnings

import numpy as np
import pandas as pd

from autopath.pulling.AnalysisSMD import SMDAnalysis
from autopath.pulling.ConvergenceLadder import force_convergence_ladder
from autopath.pulling.Estimators import extrapolate_to_v0
from autopath.pulling.SMDData import SMDData
import smd_gate as gate

warnings.filterwarnings("ignore")
PROJ = "/gpfs/group/forli/mllanos/peter/minibinders"
SPEEDS = [0.001, 0.005, 0.010]
MIN_SEEDS = 3


def seed_bootstrap(per_seed_fn, seeds, n_boot=2000, seed=42):
    """Percentile CI from resampling SEEDS, not frames.

    Between-seed variance is the relevant scale: the MM-PBSA campaign on
    these systems found 73-93% of variance sits between productions.
    """
    seeds = list(seeds)
    if len(seeds) < MIN_SEEDS:
        raise ValueError(f"need >= {MIN_SEEDS} seeds to bootstrap, got {len(seeds)}")
    rng = np.random.default_rng(seed)
    vals = [per_seed_fn(rng.choice(seeds, len(seeds), replace=True))
            for _ in range(n_boot)]
    return (float(np.median(vals)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def usable_logs(traj_dir, expected_moves=2700):
    """Logs that are not truncated. Truncated replicas bias the barrier low."""
    keep, dropped = [], []
    for f in sorted(glob.glob(os.path.join(traj_dir, "sMD_replica-*_forward.dat"))):
        df, meta = gate.read_log(f)
        (dropped if gate.classify_replica(df, meta, expected_moves)["outcome"]
         == "truncated" else keep).append(f)
    if dropped:
        print(f"  excluded {len(dropped)} truncated replica(s)")
    return keep


def work_pivot(raw_data, speeds):
    """(W[n_seed, n_speed, n_step], seeds, steps). NaN where a replica stopped early."""
    df = raw_data.copy()
    df["seed"] = df["trajname"].map(
        lambda t: int(t.split("_")[-3].split("-")[1]) // 100)
    seeds = sorted(df["seed"].unique())
    steps = np.sort(df["step"].unique())
    si = {s: i for i, s in enumerate(seeds)}
    ti = {t: i for i, t in enumerate(steps)}
    W = np.full((len(seeds), len(speeds), len(steps)), np.nan)
    for (sd, sp), g in df.groupby(["seed", "speed"]):
        if sp not in speeds:
            continue
        W[si[sd], speeds.index(sp), [ti[t] for t in g["step"]]] = g["work"].to_numpy()
    return W, seeds, steps


def barrier_from_seeds(W, speeds, sample):
    """Barrier of the v=0 PMF for one (possibly repeated) sample of seed indices.

    Per-step OLS of mean work against speed; the intercept is the reversible
    work. Only steps where every speed contributed are used -- two-speed steps
    fit perfectly with zero degrees of freedom and are meaningless.
    """
    Wm = np.nanmean(W[np.asarray(sample)], axis=0)        # (n_speed, n_step)
    ok = np.isfinite(Wm).all(axis=0)
    if ok.sum() < 5:
        return float("nan")
    v = np.asarray(speeds, dtype=float)
    y = Wm[:, ok]
    dv = v - v.mean()
    slope = (dv[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (dv ** 2).sum()
    pmf = y.mean(axis=0) - slope * v.mean()               # intercept at v=0
    return float(np.nanmax(pmf) - pmf[0])


def analyse_system(sysname, outroot=None, n_boot=2000):
    traj = f"{PROJ}/systems/{sysname}/sMD/trajectories"
    outdir = outroot or f"{PROJ}/systems/{sysname}/sMD/analysis"
    os.makedirs(outdir, exist_ok=True)
    all_logs = sorted(glob.glob(os.path.join(traj, "sMD_replica-*_forward.dat")))
    logs = usable_logs(traj)
    ref = f"{PROJ}/systems/{sysname}/system.pdb"

    smd = SMDData(logs, sysname=sysname, temperature=300.0, reference_pdb=ref)
    sa = SMDAnalysis(sysname=sysname, path_model="dtw",
                     estimators=["force", "jarzynski", "cumulant"],
                     temperature=300.0, outdir=outdir, do_plots=True,
                     reference_pdb=ref)
    smd = sa.run(smd, geom_features=True, merge_features=False,
                 cluster_to_boundary=True)

    conv = force_convergence_ladder(sa, logs, speeds=SPEEDS, conv_window=5)
    conv.to_csv(f"{outdir}/force_ladder.csv", index=False)

    force_res = smd.results[smd.results["estimator"] == "force"]
    extrap = extrapolate_to_v0(force_res, param="dG",
                               speeds=list(SPEEDS), min_speeds=2)
    gated = gate.r2_all_speeds(extrap, n_speeds=len(SPEEDS))
    min_r2 = float(gated["R2"].min()) if len(gated) and gated["R2"].notna().any() \
        else float("nan")
    r_ts = float(gated.loc[gated["dG"].idxmax(), "r_coord"]) if len(gated) \
        else float("nan")

    W, seeds, _ = work_pivot(smd.raw_data, list(SPEEDS))
    idx = {s: i for i, s in enumerate(seeds)}
    med, lo, hi = seed_bootstrap(
        lambda ss: barrier_from_seeds(W, list(SPEEDS), [idx[s] for s in ss]),
        seeds, n_boot=n_boot)

    return {"system": sysname, "barrier": med, "barrier_lo": lo, "barrier_hi": hi,
            "r_ts": r_ts, "min_r2": min_r2, "n_seeds": len(seeds),
            "n_excluded": len(all_logs) - len(logs)}


if __name__ == "__main__":
    names = sys.argv[1:] or [f"slc_m{i}" for i in range(1, 8)]
    rows = []
    for n in names:
        print(f"=== {n} ===", flush=True)
        rows.append(analyse_system(n))
    out = pd.DataFrame(rows)
    dest = f"{PROJ}/analysis/smd_barriers.csv"
    if os.path.exists(dest) and len(names) < 7:
        prev = pd.read_csv(dest)
        out = pd.concat([prev[~prev["system"].isin(out["system"])], out])
    out.sort_values("system").to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {dest}")
```

- [ ] **Step 4: Run to verify the bootstrap tests pass**

Run: `cd $PROJ/analysis && micromamba run -n autopath python -m pytest test_smd_barriers.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the analysis on one system and check the R² gate**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/analysis
micromamba run -n autopath python smd_barriers.py slc_m2
```
Expected: the ladder CSV appears. Inspect the extrapolation R² **through `gate.r2_all_speeds`**, not raw — raw includes trivially-perfect two-speed rows. If the all-speeds minimum R² is < 0.9, spec criterion 8.4 has failed: report it and stop rather than shipping a barrier from a non-linear fit.

- [ ] **Step 6: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add analysis/smd_barriers.py analysis/test_smd_barriers.py
git commit -q -m "add v0 barrier analysis with seed bootstrap"
```

---

### Task 13: `smd_figs.py` — ranking figures

**Files:**
- Create: `$PROJ/analysis/smd_figs.py`

**Interfaces:**
- Consumes: `smd_barriers.csv`.
- Produces: `analysis/smd_barrier_ranking.png`, `analysis/smd_estimator_sensitivity.png`, `analysis/smd_convergence_ladder.png`.

- [ ] **Step 1: Read the dataviz skill before writing any plotting code**

Invoke the `dataviz` skill. Categorical colours for the 7 designs; a sequential ramp only where the quantity is ordered (the ladder rungs). Match the existing project figures' conventions.

- [ ] **Step 2: Write the figures**

Three panels:
1. **Ranking** — barrier per design with the seed-bootstrap 95% interval as error bars, sorted by barrier. One categorical colour per design.
2. **Estimator sensitivity** — the same ranking repeated for `force`, `jarzynski` and `cumulant` side by side, so a ranking that depends on the estimator is visible rather than hidden.
3. **Convergence ladder** — barrier vs rung k for each system, sequential ramp over k, with the spec's tolerance band (3.0 kJ/mol) drawn, answering "had it stopped moving by rung 25?"

- [ ] **Step 3: Generate and inspect**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders/analysis
micromamba run -n autopath python smd_figs.py
```
Read each PNG before claiming it is done.

- [ ] **Step 4: Commit**

```bash
cd /gpfs/group/forli/mllanos/peter/minibinders
git add analysis/smd_figs.py
git commit -q -m "add sMD barrier ranking figures"
```

- [ ] **Step 5: Report the open item**

The BLI Kd values exist only as `updated_BLI.png`. The Spearman-vs-BLI comparison in spec §7.5 cannot be computed until those numbers are machine-readable. Ask the user for them; do not transcribe them from the image.
