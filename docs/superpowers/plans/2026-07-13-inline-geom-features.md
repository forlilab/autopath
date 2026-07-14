# Inline Geometry Logging for sMD Clustering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log cheap per-frame geometric scalars (contacts, min-distance, ligand shape descriptors) during the sMD pull and reuse them as clustering features, avoiding a post-processing pass over the DCDs.

**Architecture:** A pure-numpy descriptor kernel computes RDKit-matching shape descriptors from positions already pulled to CPU. `SteeredMD` writes a per-run sidecar `sMD_{run_id}_geom.dat` at the DCD cadence when opted in. `SMDData`/`AnalysisSMD` merge the sidecar into the clustering feature matrix via the existing `merge_asof` path, with an `impute`/`aligned` alignment switch. All new behavior is opt-in and analysis tolerates missing sidecars.

**Tech Stack:** Python, numpy, pandas, MDAnalysis, RDKit (validation + opt-in backend), OpenMM/cvpack (pulling), pytest.

## Global Constraints

- Run tests with: `micromamba run -n autopath pytest <path> -v` (the `autopath` env has numpy/pandas/MDAnalysis/rdkit/openmm/cvpack/pytest; `base` does not).
- New pulling behavior is behind `SteeredMD(log_geom_features=...)`, default `False` → **zero behavior change and no error when autostop is unused**.
- Analysis must **never error** on absent `_geom.dat` sidecars — warn and skip.
- Sidecar columns are prefixed `geom_`. Numpy backend supports:
  `nc, mindist, rog, npr1, npr2, eccentricity, inertial_shape, spherocity, pbf`.
  `asphericity` is **excluded** from the numpy backend (no fallback) — requesting it under `backend="numpy"` raises `ValueError`.
- RoG returned in **nm** (Å/10), matching `LigandFeatures._compute_rog`.
- Default `geom_merge` mode in analysis is `"impute"` (nearest-fill, no NaN).
- Descriptor formulas are fixed and validated to match RDKit ≤1e-4 (see Task 2).

---

### Task 1: Squared-distance NC kernel (drop the sqrt)

**Files:**
- Modify: `autopath/pulling/steered_md.py` (`_compute_nc` ~594-602, `_get_pocket_atoms` distance block ~627-628)
- Test: `tests/test_geom_kernel.py`

**Interfaces:**
- Produces: `SteeredMD._compute_nc(self, positions_nm, threshold_nm=0.5) -> float` (unchanged signature; numerically identical result computed without `sqrt`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_geom_kernel.py`:

```python
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
    smd.subset_protein_HA = pocket_idx
    expected = _nc_norm_reference(positions[lig_idx], positions[pocket_idx], 0.5)
    got = smd._compute_nc(positions, threshold_nm=0.5)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n autopath pytest tests/test_geom_kernel.py::test_compute_nc_squared_matches_norm -v`
Expected: FAIL (current `_compute_nc` still passes numerically, so this may PASS immediately — that is acceptable; the test pins behavior before the refactor). If it errors on import, fix imports first.

- [ ] **Step 3: Rewrite `_compute_nc` to use squared distances**

Replace the body of `_compute_nc` in `autopath/pulling/steered_md.py`:

```python
    def _compute_nc(self, positions_nm, threshold_nm=0.5):
        """Soft contact count via switching function 1/(1+(d/r0)^6).

        Uses squared distances: 1/(1+(d/r0)^6) = 1/(1+(d^2/r0^2)^3),
        so the sqrt inside the norm is skipped (the result is immediately
        raised to the 6th power). Numerically identical to the norm form.
        """
        lig_pos = positions_nm[self.groupA_atoms]
        pocket_pos = positions_nm[self.subset_protein_HA]
        diff = lig_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        x2 = d2 / (threshold_nm * threshold_nm)
        return float(np.sum(1.0 / (1.0 + x2 ** 3)))
```

- [ ] **Step 4: Rewrite the `_get_pocket_atoms` distance block to compare squared distances**

In `_get_pocket_atoms`, replace:

```python
        distances = np.linalg.norm(ligand_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :], axis=2)
        close_indices = np.any(distances < cutoff, axis=0)
```

with:

```python
        diff = ligand_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        close_indices = np.any(d2 < cutoff * cutoff, axis=0)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n autopath pytest tests/test_geom_kernel.py::test_compute_nc_squared_matches_norm -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autopath/pulling/steered_md.py tests/test_geom_kernel.py
git commit -m "perf: compute NC/pocket distances via squared distances (drop sqrt)"
```

---

### Task 2: Numpy descriptor kernel + RDKit validation + benchmark

**Files:**
- Create: `autopath/pulling/geom_kernel.py`
- Create: `scripts/benchmark_geom_kernel.py`
- Test: `tests/test_geom_kernel.py` (append)

**Interfaces:**
- Produces:
  - `NUMPY_SHAPE_FEATURES: frozenset[str]` = `{"rog","npr1","npr2","eccentricity","inertial_shape","spherocity","pbf"}`
  - `compute_shape_descriptors(pos_ang: np.ndarray, mass: np.ndarray, which: Sequence[str]) -> dict[str, float]`
    — `pos_ang` heavy-atom coords in Å (N,3), `mass` (N,), returns `{name: value}`; `rog` in nm. Raises `ValueError` if `which` contains a name not in `NUMPY_SHAPE_FEATURES` (notably `"asphericity"`).
- Consumes (Task 3): `compute_shape_descriptors`, `NUMPY_SHAPE_FEATURES`.

- [ ] **Step 1: Write the failing validation test**

Append to `tests/test_geom_kernel.py`:

```python
SHAPE = ["rog", "npr1", "npr2", "eccentricity", "inertial_shape", "spherocity", "pbf"]


def _rdkit_ref(mol):
    from rdkit.Chem import Descriptors3D, rdMolDescriptors as D
    return {
        "rog": Descriptors3D.RadiusOfGyration(mol) / 10.0,   # Å -> nm
        "npr1": Descriptors3D.NPR1(mol),
        "npr2": Descriptors3D.NPR2(mol),
        "eccentricity": Descriptors3D.Eccentricity(mol),
        "inertial_shape": Descriptors3D.InertialShapeFactor(mol),
        "spherocity": Descriptors3D.SpherocityIndex(mol),
        "pbf": D.CalcPBF(mol),
    }


def test_shape_descriptors_match_rdkit():
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from autopath.pulling.geom_kernel import compute_shape_descriptors

    smis = ["CCON(C)C(=O)c1ccccc1", "O=C(O)Cc1ccc(cc1)N",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCCCCCCC", "c1ccccc1"]
    for smi in smis:
        for seed in (1, 7, 42):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
                continue
            m = Chem.RemoveHs(m)
            pos = m.GetConformer().GetPositions()
            mass = np.array([a.GetMass() for a in m.GetAtoms()])
            got = compute_shape_descriptors(pos, mass, SHAPE)
            ref = _rdkit_ref(m)
            for k in SHAPE:
                assert got[k] == pytest.approx(ref[k], abs=1e-4), f"{k} {smi} seed={seed}"


def test_shape_descriptors_reject_asphericity():
    import numpy as np
    from autopath.pulling.geom_kernel import compute_shape_descriptors
    pos = np.random.default_rng(0).normal(size=(10, 3))
    mass = np.ones(10)
    with pytest.raises(ValueError):
        compute_shape_descriptors(pos, mass, ["asphericity"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n autopath pytest tests/test_geom_kernel.py -k shape -v`
Expected: FAIL with "No module named 'autopath.pulling.geom_kernel'"

- [ ] **Step 3: Create the kernel**

Create `autopath/pulling/geom_kernel.py`:

```python
"""Pure-numpy ligand shape descriptors matching RDKit Descriptors3D.

Computed from a single mass-weighted inertia-tensor eigendecomposition plus
one unweighted-covariance eigendecomposition and one SVD, so all requested
descriptors share the same passes. Validated to match RDKit to <1e-4 for
every supported name. `asphericity` is intentionally unsupported (its RDKit
formula did not reproduce to tolerance); requesting it raises ValueError.
"""
from typing import Sequence
import numpy as np

NUMPY_SHAPE_FEATURES = frozenset({
    "rog", "npr1", "npr2", "eccentricity", "inertial_shape", "spherocity", "pbf",
})


def compute_shape_descriptors(pos_ang: np.ndarray,
                              mass: np.ndarray,
                              which: Sequence[str]) -> dict[str, float]:
    """Return {name: value} for each requested descriptor.

    Parameters
    ----------
    pos_ang : (N, 3) heavy-atom coordinates in Angstrom.
    mass    : (N,) atomic masses.
    which   : descriptor names, subset of NUMPY_SHAPE_FEATURES.

    RoG is returned in nm (Angstrom / 10); all others are dimensionless
    ratios or (pbf) a mean distance in Angstrom, matching RDKit.
    """
    which = list(which)
    unknown = [w for w in which if w not in NUMPY_SHAPE_FEATURES]
    if unknown:
        raise ValueError(
            f"Unsupported by numpy backend: {sorted(unknown)}. "
            f"Supported: {sorted(NUMPY_SHAPE_FEATURES)}. "
            f"(asphericity is excluded from the numpy backend.)"
        )

    pos = np.asarray(pos_ang, dtype=float)
    mass = np.asarray(mass, dtype=float)
    M = mass.sum()
    com = (mass[:, None] * pos).sum(0) / M
    X = pos - com
    r2 = (X * X).sum(1)

    out: dict[str, float] = {}
    need_inertia = any(k in which for k in
                       ("rog", "npr1", "npr2", "eccentricity", "inertial_shape"))
    if need_inertia:
        I = (np.einsum('n,ab->ab', mass * r2, np.eye(3))
             - np.einsum('n,na,nb->ab', mass, X, X))
        I1, I2, I3 = np.sort(np.linalg.eigvalsh(I))
        if "npr1" in which:
            out["npr1"] = float(I1 / I3)
        if "npr2" in which:
            out["npr2"] = float(I2 / I3)
        if "eccentricity" in which:
            out["eccentricity"] = float(np.sqrt(max(I3 * I3 - I1 * I1, 0.0)) / I3)
        if "inertial_shape" in which:
            out["inertial_shape"] = float(I2 / (I1 * I3))
        if "rog" in which:
            out["rog"] = float(np.sqrt((I1 + I2 + I3) / (2.0 * M)) / 10.0)

    if "spherocity" in which or "pbf" in which:
        Xg = pos - pos.mean(0)
        if "spherocity" in which:
            cov = (Xg.T @ Xg) / len(Xg)
            e = np.sort(np.linalg.eigvalsh(cov))
            out["spherocity"] = float(3.0 * e[0] / e.sum())
        if "pbf" in which:
            _, _, vt = np.linalg.svd(Xg)
            normal = vt[-1]
            out["pbf"] = float(np.abs(Xg @ normal).mean())

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n autopath pytest tests/test_geom_kernel.py -v`
Expected: PASS (all NC + shape + reject tests)

- [ ] **Step 5: Create the benchmark script**

Create `scripts/benchmark_geom_kernel.py`:

```python
"""Benchmark numpy shape kernel vs a slimmed RDKit path (us/frame).

Usage: micromamba run -n autopath python scripts/benchmark_geom_kernel.py
The RDKit path reuses one conformer (SetAtomPosition) with no re-sanitization,
mirroring the tightest reasonable per-frame RDKit usage.
"""
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors as D
from rdkit.Geometry import Point3D
from autopath.pulling.geom_kernel import compute_shape_descriptors

WHICH = ["rog", "npr1", "npr2", "eccentricity", "inertial_shape", "spherocity", "pbf"]
N_FRAMES = 2000


def main():
    m = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    base = conf.GetPositions()
    mass = np.array([a.GetMass() for a in m.GetAtoms()])
    rng = np.random.default_rng(0)
    frames = [base + 0.1 * rng.normal(size=base.shape) for _ in range(N_FRAMES)]

    t0 = time.perf_counter()
    for pos in frames:
        compute_shape_descriptors(pos, mass, WHICH)
    t_np = (time.perf_counter() - t0) / N_FRAMES * 1e6

    def rdkit_frame(pos):
        for i, xyz in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        return (Descriptors3D.RadiusOfGyration(m), Descriptors3D.NPR1(m),
                Descriptors3D.NPR2(m), Descriptors3D.Eccentricity(m),
                Descriptors3D.InertialShapeFactor(m), Descriptors3D.SpherocityIndex(m),
                D.CalcPBF(m))

    t0 = time.perf_counter()
    for pos in frames:
        rdkit_frame(pos)
    t_rd = (time.perf_counter() - t0) / N_FRAMES * 1e6

    print(f"numpy kernel : {t_np:8.2f} us/frame")
    print(f"rdkit (slim) : {t_rd:8.2f} us/frame")
    print(f"speedup      : {t_rd / t_np:6.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the benchmark and record the result**

Run: `micromamba run -n autopath python scripts/benchmark_geom_kernel.py`
Expected: numpy faster than RDKit (report the printed µs/frame + speedup in the commit body). Not asserted.

- [ ] **Step 7: Commit**

```bash
git add autopath/pulling/geom_kernel.py scripts/benchmark_geom_kernel.py tests/test_geom_kernel.py
git commit -m "feat: numpy ligand shape kernel (RDKit-validated) + benchmark"
```

---

### Task 3: Inline geometry logging in SteeredMD

**Files:**
- Modify: `autopath/pulling/steered_md.py` (`__init__`, `_get_pocket_atoms` gate in `run`, `pull_single_direction`)
- Test: `tests/test_steered_geom_logging.py`

**Interfaces:**
- Consumes: `compute_shape_descriptors`, `NUMPY_SHAPE_FEATURES` from `autopath.pulling.geom_kernel`.
- Produces:
  - `SteeredMD.__init__(..., log_geom_features: bool | list[str] = False, geom_backend: str = "numpy")`
  - `SteeredMD._resolve_geom_features(self) -> list[str]` — the resolved ordered list of geom feature names (empty when disabled).
  - `SteeredMD._compute_geom_row(self, positions_nm) -> dict[str, float]` — computes the requested geom scalars for one frame (keys are bare names: `nc`, `mindist`, `rog`, …).
  - Side effect: writes `{out_dir}/sMD_{run_id}_geom.dat` when enabled.

Design notes for the implementer:
- Default feature set when `log_geom_features is True`:
  `["nc", "mindist", "rog", "npr1", "npr2", "spherocity", "pbf"]`.
- `nc`/`mindist` need pocket atoms (`self.subset_protein_HA`); shape/`rog` need only ligand atoms (`self.groupA_atoms`).
- Ligand masses for the kernel: `np.array([system.getParticleMass(i).value_in_unit(dalton) for i in groupA_atoms])`; positions passed to the kernel must be in **Å** (positions in the loop are nm → multiply by 10 before the shape kernel; `mindist` stays in nm).
- Sidecar sampled at the same `i % self.save_freq == 0` cadence already used for NC.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_steered_geom_logging.py`:

```python
import numpy as np
import pytest
from autopath.pulling.steered_md import SteeredMD


def _make_stub():
    smd = SteeredMD.__new__(SteeredMD)
    smd.groupA_atoms = [0, 1, 2, 3]
    smd.subset_protein_HA = np.array([4, 5, 6, 7, 8])
    smd._ligand_masses = np.ones(4) * 12.0
    return smd


def test_resolve_geom_features_disabled():
    smd = _make_stub()
    smd.log_geom_features = False
    assert smd._resolve_geom_features() == []


def test_resolve_geom_features_default_true():
    smd = _make_stub()
    smd.log_geom_features = True
    feats = smd._resolve_geom_features()
    assert "nc" in feats and "rog" in feats and "npr1" in feats
    assert "asphericity" not in feats


def test_resolve_geom_features_rejects_asphericity_numpy():
    smd = _make_stub()
    smd.log_geom_features = ["rog", "asphericity"]
    smd.geom_backend = "numpy"
    with pytest.raises(ValueError):
        smd._resolve_geom_features()


def test_compute_geom_row_keys_and_values():
    smd = _make_stub()
    smd.geom_backend = "numpy"
    smd.log_geom_features = ["nc", "mindist", "rog", "npr1"]
    rng = np.random.default_rng(0)
    pos_nm = rng.normal(size=(9, 3)) * 0.3
    row = smd._compute_geom_row(pos_nm)
    assert set(row) == {"nc", "mindist", "rog", "npr1"}
    assert row["mindist"] >= 0.0
    assert 0.0 <= row["npr1"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n autopath pytest tests/test_steered_geom_logging.py -v`
Expected: FAIL (attributes/methods not defined)

- [ ] **Step 3: Add constructor args**

In `SteeredMD.__init__`, add parameters `log_geom_features: bool | list[str] = False` and `geom_backend: str = "numpy"` to the signature, and store:

```python
        self.log_geom_features = log_geom_features
        self.geom_backend = str(geom_backend)
        self._ligand_masses = None  # populated in run() once the system exists
```

- [ ] **Step 4: Implement `_resolve_geom_features` and `_compute_geom_row`**

Add these methods to `SteeredMD` (near `_compute_nc`):

```python
    # Bare feature names supported by the geom-logging sidecar.
    _GEOM_DEFAULT = ["nc", "mindist", "rog", "npr1", "npr2", "spherocity", "pbf"]
    _GEOM_POCKET = {"nc", "mindist"}

    def _resolve_geom_features(self) -> list[str]:
        """Ordered list of geom feature names to log (empty when disabled)."""
        from autopath.pulling.geom_kernel import NUMPY_SHAPE_FEATURES
        if not self.log_geom_features:
            return []
        if self.log_geom_features is True:
            feats = list(self._GEOM_DEFAULT)
        else:
            feats = list(self.log_geom_features)
        shape_names = set(feats) - self._GEOM_POCKET
        if getattr(self, "geom_backend", "numpy") == "numpy":
            bad = sorted(shape_names - set(NUMPY_SHAPE_FEATURES))
            if bad:
                raise ValueError(
                    f"geom features {bad} not supported by numpy backend "
                    f"(asphericity excluded). Use geom_backend='rdkit' or drop them."
                )
        return feats

    def _compute_geom_row(self, positions_nm) -> dict[str, float]:
        """Compute the requested geom scalars for one frame.

        positions_nm : (Natoms, 3) in nm. Shape descriptors need Angstrom, so
        ligand coords are scaled by 10 before the kernel; mindist stays in nm.
        """
        from autopath.pulling.geom_kernel import compute_shape_descriptors
        feats = self._resolve_geom_features()
        row: dict[str, float] = {}
        pocket_ok = self.subset_protein_HA is not None
        if "nc" in feats and pocket_ok:
            row["nc"] = self._compute_nc(positions_nm)
        if "mindist" in feats and pocket_ok:
            lig = positions_nm[self.groupA_atoms]
            poc = positions_nm[self.subset_protein_HA]
            diff = lig[:, None, :] - poc[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            row["mindist"] = float(np.sqrt(d2.min()))
        shape = [f for f in feats if f not in self._GEOM_POCKET]
        if shape:
            lig_ang = positions_nm[self.groupA_atoms] * 10.0
            row.update(compute_shape_descriptors(lig_ang, self._ligand_masses, shape))
        return row
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `micromamba run -n autopath pytest tests/test_steered_geom_logging.py -v`
Expected: PASS

- [ ] **Step 6: Wire pocket detection + ligand masses in `run()`**

In `run()`, change the pocket-atom gate (currently `if self.verbose > 0 or self.autostop_nc is not None:`) so geom logging that needs pocket atoms also triggers detection:

```python
        geom_feats = self._resolve_geom_features()
        needs_pocket = (self.verbose > 0 or self.autostop_nc is not None
                        or bool(set(geom_feats) & self._GEOM_POCKET))
        if needs_pocket:
            self.subset_protein_HA, _ = self._get_pocket_atoms(simulation, cutoff=0.5)
        else:
            self.subset_protein_HA = None
```

And after the system/simulation exist (anywhere before `pull_single_direction`), cache ligand masses for the kernel:

```python
        if geom_feats:
            self._ligand_masses = np.array([
                system.getParticleMass(i).value_in_unit(openmmunit.dalton)
                for i in self.groupA_atoms
            ])
```

- [ ] **Step 7: Write the sidecar in `pull_single_direction`**

At the start of `pull_single_direction`, before the main loop, resolve features and open the sidecar buffer:

```python
        geom_feats = self._resolve_geom_features()
        geom_cols = [f for f in geom_feats
                     if f not in self._GEOM_POCKET or self.subset_protein_HA is not None]
        geom_dropped = set(geom_feats) - set(geom_cols)
        if geom_dropped:
            logger.warning(
                f"[geom] pocket atoms unavailable; skipping {sorted(geom_dropped)}."
            )
        geom_path = f"{self.out_dir}/sMD_{run_id}_geom.dat"
        geom_buf: list[str] = []
        geom_fh = None
        if geom_cols:
            geom_fh = open(geom_path, "w")
            geom_fh.write("step,time," + ",".join(f"geom_{c}" for c in geom_cols) + "\n")
```

Inside the loop, at the existing `i % self.save_freq == 0` sampling point (reuse the `pos_nm` already fetched for NC when present; otherwise fetch once), append a geom row:

```python
                if geom_fh is not None and i % self.save_freq == 0:
                    if nc is not None and 'pos_nm' in dir():
                        _gpos = pos_nm
                    else:
                        _gpos = simulation.context.getState(getPositions=True)\
                            .getPositions(asNumpy=True) / openmmunit.nanometers
                    grow = self._compute_geom_row(_gpos)
                    geom_buf.append(
                        f"{i},{time_before}," +
                        ",".join(f"{grow[c]:.5f}" for c in geom_cols) + "\n")
                    if len(geom_buf) >= self.save_freq:
                        geom_fh.write(''.join(geom_buf)); geom_buf.clear()
```

After the main loop (near the trace `if _buf: f.write(...)`), flush and close:

```python
        if geom_fh is not None:
            if geom_buf:
                geom_fh.write(''.join(geom_buf))
            geom_fh.close()
```

Note for implementer: the `pos_nm` reuse above must reference the variable computed in the NC block (lines ~348-350). If cleaner, compute `_gpos` once per sampled move before both the NC and geom blocks and share it. Keep a single `getState` per sampled move.

- [ ] **Step 8: Add an integration-style test (off = no sidecar; on = sidecar)**

This needs a tiny real OpenMM system. If the repo has an existing sMD smoke-test fixture, reuse it; otherwise mark as `@pytest.mark.slow` and build a 2-particle system. Append to `tests/test_steered_geom_logging.py`:

```python
@pytest.mark.slow
def test_pull_writes_sidecar_when_enabled(tmp_path):
    pytest.importorskip("openmm")
    # Reuse the project's minimal sMD fixture if available; otherwise skip.
    fixture = pytest.importorskip(
        "tests.smd_fixture", reason="no minimal sMD fixture available")
    smd, kwargs = fixture.make_two_group_smd(out_dir=str(tmp_path),
                                             log_geom_features=True)
    run_id = smd.run(**kwargs)
    import os
    assert os.path.exists(f"{tmp_path}/sMD_{run_id}_geom.dat")
    import pandas as pd
    df = pd.read_csv(f"{tmp_path}/sMD_{run_id}_geom.dat")
    assert "geom_rog" in df.columns and "geom_npr1" in df.columns


@pytest.mark.slow
def test_pull_no_sidecar_when_disabled(tmp_path):
    pytest.importorskip("openmm")
    fixture = pytest.importorskip(
        "tests.smd_fixture", reason="no minimal sMD fixture available")
    smd, kwargs = fixture.make_two_group_smd(out_dir=str(tmp_path),
                                             log_geom_features=False)
    run_id = smd.run(**kwargs)
    import glob
    assert glob.glob(f"{tmp_path}/*_geom.dat") == []
```

If no sMD fixture exists in the repo, implement `tests/smd_fixture.py` with a
`make_two_group_smd(out_dir, **overrides)` helper building the smallest valid
system (two atom groups, a PDB written to `out_dir`), returning `(smd, run_kwargs)`.
If that is out of scope for this task, leave the two `@pytest.mark.slow` tests
`pytest.importorskip`-guarded so they skip cleanly, and rely on the Task-3 unit
tests for coverage.

- [ ] **Step 9: Run the fast unit tests**

Run: `micromamba run -n autopath pytest tests/test_steered_geom_logging.py -v -m "not slow"`
Expected: PASS (4 unit tests); slow tests skip if no fixture.

- [ ] **Step 10: Commit**

```bash
git add autopath/pulling/steered_md.py tests/test_steered_geom_logging.py tests/smd_fixture.py
git commit -m "feat: opt-in inline geom-feature logging to sMD sidecar (autostop-independent)"
```

---

### Task 4: Analysis-side consumption (load, merge, wiring)

**Files:**
- Modify: `autopath/pulling/SMDData.py` (add `load_geom_features`, `merge_geom_features`)
- Modify: `autopath/pulling/AnalysisSMD.py` (`_build_cluster_feature_df` + `run` signature)
- Test: `tests/test_geom_analysis.py`

**Interfaces:**
- Consumes: `SMDData.log_files`, `SMDData.merge_feature_sets` (existing).
- Produces:
  - `SMDData.load_geom_features(self) -> pd.DataFrame` — reads all `sMD_*_geom.dat` sidecars matching the loaded `log_files`; columns `trajname, speed, step, time, geom_*`. Returns an **empty** DataFrame (with no rows) when none exist. Never raises on missing files.
  - `SMDData.merge_geom_features(trace_df, geom_df, mode="impute") -> pd.DataFrame`
    — `mode="impute"`: `merge_feature_sets(trace_df, geom_df)` (nearest-fill, no NaN). `mode="aligned"`: inner merge on `(trajname, time)` exact matches only. Unknown mode → `ValueError`.
  - `AnalysisSMD.run(..., geom_features: bool = False, geom_merge: str = "impute")` and the same passthrough for `check_convergence` if present.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geom_analysis.py`:

```python
import numpy as np
import pandas as pd
import pytest
from autopath.pulling.SMDData import SMDData


def _trace():
    return pd.DataFrame({
        "trajname": ["t1"] * 6,
        "speed": [0.001] * 6,
        "step": [0, 1, 2, 3, 4, 5],
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "work": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _geom():
    return pd.DataFrame({
        "trajname": ["t1", "t1", "t1"],
        "speed": [0.001] * 3,
        "step": [0, 2, 4],
        "time": [0.0, 2.0, 4.0],
        "geom_rog": [0.5, 0.6, 0.7],
    })


def test_merge_geom_impute_no_nan_full_rows():
    out = SMDData.merge_geom_features(_trace(), _geom(), mode="impute")
    assert len(out) == 6
    assert "geom_rog" in out.columns
    assert not out["geom_rog"].isnull().any()


def test_merge_geom_aligned_exact_only():
    out = SMDData.merge_geom_features(_trace(), _geom(), mode="aligned")
    assert len(out) == 3
    assert sorted(out["time"].tolist()) == [0.0, 2.0, 4.0]
    assert not out["geom_rog"].isnull().any()


def test_merge_geom_bad_mode():
    with pytest.raises(ValueError):
        SMDData.merge_geom_features(_trace(), _geom(), mode="nope")


def test_load_geom_features_missing_returns_empty(tmp_path):
    # log file with no matching sidecar
    log = tmp_path / "sMD_replica-1_v0.001_forward.dat"
    log.write_text("step,time,r_target,r_before,r_after,force,U_cvpack,dW_protocol,lag_nm\n"
                   "0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0\n")
    smd = SMDData.__new__(SMDData)
    smd.log_files = [str(log)]
    df = smd.load_geom_features()
    assert df.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n autopath pytest tests/test_geom_analysis.py -v`
Expected: FAIL (`merge_geom_features` / `load_geom_features` not defined)

- [ ] **Step 3: Implement `merge_geom_features` (staticmethod on SMDData)**

Add to `SMDData`:

```python
    @staticmethod
    def merge_geom_features(trace_df: pd.DataFrame,
                            geom_df: pd.DataFrame,
                            mode: str = "impute") -> pd.DataFrame:
        """Merge geom sidecar features onto trace features.

        mode='impute'  : merge_feature_sets nearest-fill (no NaN, full rows).
        mode='aligned' : keep only exact (trajname, time) matches.
        """
        if mode == "impute":
            return SMDData.merge_feature_sets(trace_df, geom_df)
        if mode == "aligned":
            geom_cols = [c for c in geom_df.columns if c.startswith("geom_")]
            right = geom_df[["trajname", "time"] + geom_cols]
            return trace_df.merge(right, on=["trajname", "time"], how="inner")
        raise ValueError(f"geom_merge mode must be 'impute' or 'aligned', got {mode!r}")
```

- [ ] **Step 4: Implement `load_geom_features` (instance method on SMDData)**

Add to `SMDData`:

```python
    def load_geom_features(self) -> pd.DataFrame:
        """Load per-run geom sidecars (sMD_*_geom.dat) for the loaded logs.

        Returns an empty DataFrame when no sidecars are present; never raises
        on missing files. Columns: trajname, speed, step, time, geom_*.
        """
        rows = []
        for log_fn in self.log_files:
            geom_fn = log_fn[:-4] + "_geom.dat"      # replace trailing '.dat'
            if not os.path.exists(geom_fn):
                continue
            try:
                g = pd.read_csv(geom_fn, comment='#')
            except Exception as e:
                logger.warning(f"Could not read geom sidecar {geom_fn}: {e}")
                continue
            g["trajname"] = os.path.basename(log_fn)[:-4]
            g["speed"] = self._speed_from_log(log_fn)
            rows.append(g)
        if not rows:
            logger.info("No geom sidecars (_geom.dat) found; geom features unavailable.")
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `micromamba run -n autopath pytest tests/test_geom_analysis.py -v`
Expected: PASS

- [ ] **Step 6: Wire into `AnalysisSMD._build_cluster_feature_df` and `run`**

Add `geom_features: bool = False` and `geom_merge: str = "impute"` to `AnalysisSMD.run` (and store on `self` or thread through). In `_build_cluster_feature_df`, after the existing feature sources are assembled and just before returning `feat_df`, add:

```python
        if geom_features:
            geom_df = smd.load_geom_features()
            if geom_df.empty:
                logger.warning(
                    "geom_features requested but no _geom.dat sidecars found; "
                    "proceeding without inline geom features."
                )
            else:
                feat_df = SMDData.merge_geom_features(feat_df, geom_df, mode=geom_merge)
                _gcols = [c for c in feat_df.columns if c.startswith("geom_")]
                logger.info(f"Merged inline geom features ({geom_merge}): {_gcols}")
```

Pass `geom_features`/`geom_merge` from `run` into `_build_cluster_feature_df`
(add them to that method's signature with the same defaults).

- [ ] **Step 7: Run the full new test suite**

Run: `micromamba run -n autopath pytest tests/test_geom_kernel.py tests/test_geom_analysis.py tests/test_steered_geom_logging.py -v -m "not slow"`
Expected: PASS

- [ ] **Step 8: Run the existing suite to confirm no regressions**

Run: `micromamba run -n autopath pytest tests/ -v -m "not slow"`
Expected: PASS (existing estimator/support/convergence tests unaffected)

- [ ] **Step 9: Commit**

```bash
git add autopath/pulling/SMDData.py autopath/pulling/AnalysisSMD.py tests/test_geom_analysis.py
git commit -m "feat: consume inline geom sidecars in clustering (impute/aligned merge)"
```

---

## Self-Review

**Spec coverage:**
- NC/distance sqrt removal → Task 1. ✓
- Opt-in inline logging, autostop-independent, default off → Task 3 (Steps 3,6,7). ✓
- Numpy kernel + RDKit validation + benchmark; asphericity excluded (no fallback) → Task 2. ✓
- Sidecar at save_freq cadence, `geom_` prefix, RoG in nm → Task 3 (Step 7), Task 2 (kernel). ✓
- Analysis load + merge with `impute`/`aligned`, missing-sidecar skip → Task 4. ✓
- Never error when autostop/geom unused → Task 3 default `False`; Task 4 empty-DF path. ✓

**Placeholder scan:** All code steps contain full code. The Task 3 Step 8 integration tests are guarded with `importorskip` and describe the fixture fallback explicitly (not a placeholder — a documented skip path). ✓

**Type consistency:** `compute_shape_descriptors(pos_ang, mass, which)` and `NUMPY_SHAPE_FEATURES` used identically in Tasks 2 & 3. `_resolve_geom_features`/`_compute_geom_row`/`_GEOM_POCKET`/`_GEOM_DEFAULT` consistent across Task 3 steps. `merge_geom_features(trace_df, geom_df, mode)` and `load_geom_features()` consistent between Task 4 definition and tests. Bare feature names (`nc`,`rog`,…) vs prefixed columns (`geom_nc`,…) — kernel/row use bare, sidecar header and analysis use `geom_` prefix; conversion happens only at the sidecar write (Task 3 Step 7) and is consistent. ✓
