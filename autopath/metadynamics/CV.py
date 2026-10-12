from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
import os
import logging
import numpy as np
import cvpack
import openmm.unit as openmmunit
from openmm.app import PDBFile

from autopath.clustering import read_positions

logger = logging.getLogger("autopath.cv")


def _build_reference_dict(
    pdb_file: str,
    atom_names: list[str],
    system_atom_indexes: list[int],
) -> dict[int, tuple[float, float, float]]:
    """Build a {system_atom_index: (x, y, z)} reference dict from a PDB file."""
    pdb = PDBFile(pdb_file)
    reference_positions = pdb.positions

    coords = []
    for atom in pdb.topology.atoms():
        if atom.name in atom_names:
            coords.append(reference_positions[atom.index] / openmmunit.nanometers)

    return dict(zip(system_atom_indexes, coords))


@dataclass
class CVSpec:
    """A collective variable bundled with its BiasVariable grid parameters.

    Pass one or more CVSpec objects to MetadynamicsMD.run() via cv_specs.
    CVs that need input positions are deferred: they store a factory that
    materializes inside run() once the checkpoint is loaded.
    """
    cv: Any
    grid_min: float
    grid_max: float
    hill_width: float
    grid_points: int
    periodic: bool = False
    name: str = "cv"
    # Angstrom per CV unit; the funnel correction integrates in length units.
    cv_length_per_unit_ang: Optional[float] = None
    # Bound end of the CV, if the grid extends past it (default: grid_min).
    origin: Optional[float] = None
    # (lo, hi) the walker can reach, for the boundary correction (default: the grid).
    bounds: Optional[tuple] = None
    _factory: Optional[Callable] = field(default=None, repr=False)

    def is_deferred(self) -> bool:
        """Return True if this CV requires positions to materialize before use."""
        return self._factory is not None

    def resolve(self, input_positions, n_atoms: int, topology) -> "CVSpec":
        """Materialize a deferred CV using positions from the loaded simulation."""
        if not self.is_deferred():
            return self
        cv = self._factory(input_positions, n_atoms, topology)
        return CVSpec(
            cv=cv,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            hill_width=self.hill_width,
            grid_points=self.grid_points,
            periodic=self.periodic,
            name=self.name,
            cv_length_per_unit_ang=self.cv_length_per_unit_ang,
            origin=self.origin,
            bounds=self.bounds,
        )

    @property
    def bound_end(self) -> float:
        return self.grid_min if self.origin is None else self.origin


# Some example CVSpecs for common CV types

def com_cv(
    pocket_atoms: list[int],
    ligand_atoms: list[int],
    grid_min: float,
    grid_max: float,
    hill_width: float = 0.05,
    grid_points: int = 125,
) -> CVSpec:
    """Center-of-mass distance between pocket and ligand centroids (nm).

    Deferred so each run() gets a fresh force: addCollectiveVariable() takes
    C++ ownership, so reusing one object across walkers fails.
    """
    def factory(input_positions, n_atoms, topology):
        return cvpack.CentroidFunction(
            "sqrt(distance(g1,g2)^2)",
            openmmunit.nanometers,
            [pocket_atoms, ligand_atoms],
            weighByMass=True,
            pbc=False,
        )

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="com",
        _factory=factory,
    )


def funnel_z_cv(
    host_atoms: list[int],
    guest_atoms: list[int],
    axis,
    grid_min: float,
    grid_max: float,
    hill_width: float = 0.05,
    grid_points: int = 125,
    bounds: Optional[tuple] = None,
    frame: Optional[dict] = None,
) -> CVSpec:
    """Guest position along the funnel axis, r_z = n·(COM_guest - COM_host) (nm).

    Same coordinate as the funnel restraint, so the FES is in the length
    coordinate Limongelli 2013 Eq. 15 integrates over. Clamped to the grid
    because OpenMM's bias is zero outside it. z = 0 is the bound end; set
    grid_min < 0 so the bias also acts below the host COM. ``bounds`` are the
    boundary-correction edges (nm), see ``funnel_z_bounds_nm``. ``frame`` (from
    ``Funnel.funnel_frame``) rebuilds n from the protein frame; None uses ``axis``.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    host, guest = [int(i) for i in host_atoms], [int(i) for i in guest_atoms]

    def factory(input_positions, n_atoms, topology):
        if frame is not None:
            from .Funnel import _frame_definitions
            return cvpack.CentroidFunction(
                "min(max(r_z, fz_lo), fz_hi);" + _frame_definitions(1, 2, 1, 3, 4, frame["coeffs"], radial=False),
                openmmunit.nanometers,
                [host, guest, [int(i) for i in frame["b"]], [int(i) for i in frame["c"]]],
                weighByMass=True,
                pbc=False,
                fz_lo=[float(grid_min)], fz_hi=[float(grid_max)],
            )
        # 1-element arrays: scalar cvpack parameters become global Context parameters.
        return cvpack.CentroidFunction(
            "min(max(fz_nx*(x2-x1) + fz_ny*(y2-y1) + fz_nz*(z2-z1), fz_lo), fz_hi)",
            openmmunit.nanometers,
            [host, guest],
            weighByMass=True,
            pbc=False,
            fz_nx=[float(n[0])], fz_ny=[float(n[1])], fz_nz=[float(n[2])],
            fz_lo=[float(grid_min)], fz_hi=[float(grid_max)],
        )

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="funnel_z",
        cv_length_per_unit_ang=10.0,
        origin=0.0,
        bounds=bounds,
        _factory=factory,
    )


def rmsd_cv(
    ligand_atoms: list[int],
    grid_min: float,
    grid_max: float,
    hill_width: float = 0.05,
    grid_points: int = 125,
) -> CVSpec:
    """Ligand RMSD relative to the starting conformation (deferred: needs positions).

    Aligned RMSD: captures internal conformational change only.
    """
    def factory(input_positions, n_atoms, topology):
        return cvpack.RMSD(input_positions, ligand_atoms, n_atoms)

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="rmsd",
        _factory=factory,
    )


def rmsd_states_cv(
    topology,
    grid_min: float,
    grid_max: float,
    milestone_dir: str = "input",
    hill_width: float = 0.01,
    grid_points: int = 125,
    exclude_residues: tuple = ("UNK", "HOH", "NA", "CL", "K"),
    sigma: float = 0.01,
) -> CVSpec:
    """Path-in-RMSD-space CV through protein CA milestones (deferred: needs positions)."""
    from glob import glob

    def factory(input_positions, n_atoms, topology):
        atom_names = ["CA"]
        system_residues = [r for r in topology.residues() if r.name not in exclude_residues]
        atom_indexes = [
            atom.index
            for residue in system_residues
            for atom in residue.atoms()
            if atom.name in atom_names
        ]
        # Skip RelaxMD's <milestone>_relax.pdb outputs.
        milestone_pdbs = sorted(
            (p for p in glob(f"{milestone_dir}/milestone_*_frame_*.pdb")
             if "_relax" not in os.path.basename(p)),
            key=_milestone_sort_key,
        )
        if not milestone_pdbs:
            raise FileNotFoundError(
                f"no milestone_*_frame_*.pdb under {milestone_dir}. Milestones are now "
                "written as .xml states; rmsd_states_cv still requires PDBs "
                "(its _build_reference_dict reads PDB paths).")
        milestones_dicts = [
            _build_reference_dict(pdb, atom_names, atom_indexes)
            for pdb in milestone_pdbs
        ]
        return cvpack.PathInRMSDSpace(
            metric=cvpack.path.progress,
            milestones=milestones_dicts,
            sigma=sigma * openmmunit.nanometers,
            numAtoms=n_atoms,
        )

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="rmsd_states",
        _factory=factory,
    )


def _milestone_sort_key(path: str) -> int:
    """Milestone index from a `<prefix>_<index>_frame_<frame>` filename.

    Reads the token before `_frame_`, so it handles both `milestone_1_frame_12`
    and per-path `milestone_0_1_frame_12`.
    """
    base = os.path.basename(path)
    try:
        return int(base.split("_frame_")[0].rsplit("_", 1)[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse milestone index from '{base}'. "
            f"Expected format: <prefix>_<index>_frame_<frame>.<ext>"
        ) from exc


def path_rmsd_cv(
    topology,
    milestones: list[str],
    grid_min: float,
    grid_max: float,
    ligand_resname: str = "UNK",
    hill_width: float = 0.001,
    grid_points: int = 125,
    sigma: float = 0.001,
) -> CVSpec:
    """Path-in-RMSD-space CV through ligand heavy-atom milestones (deferred: needs positions)."""
    # PDBFile fails opaquely on serialized States; reject non-PDB paths up front.
    _bad = [m for m in milestones if not m.lower().endswith(".pdb")]
    if _bad:
        raise ValueError(
            f"path_rmsd_cv requires .pdb milestones, got {len(_bad)} with another "
            f"extension (e.g. {os.path.basename(_bad[0])}). Milestones are now written "
            "as serialized .xml states; path_rmsd_cv cannot read them because its "
            "_build_reference_dict needs a topology. Write PDB milestones "
            "(Clustering.write_pdbs) or use a CV that accepts states.")
    milestones = sorted(milestones, key=_milestone_sort_key)

    def factory(input_positions, n_atoms, topology):
        ligand_residue = [r for r in topology.residues() if r.name == ligand_resname]
        heavy_atoms = [a for a in ligand_residue[0].atoms() if not a.name.startswith("H")]
        atom_indexes = [a.index for a in heavy_atoms]
        atom_names = [a.name for a in heavy_atoms]
        milestones_dicts = [
            _build_reference_dict(pdb, atom_names, atom_indexes)
            for pdb in milestones
        ]
        return cvpack.PathInRMSDSpace(
            metric=cvpack.path.progress,
            milestones=milestones_dicts,
            sigma=sigma * openmmunit.nanometers,
            numAtoms=n_atoms,
        )

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="path_rmsd",
        _factory=factory,
    )


def _extend_milestones(milestones_array: np.ndarray,
                       extend_to_nm: float) -> np.ndarray:
    """Append milestones out to `extend_to_nm`, spaced like the existing ones.

    The far end defines "unbound", so it must clear host contact. Uniform spacing
    avoids a gap much wider than sigma, which would make progress a staircase.
    """
    d = milestones_array.flatten().astype(float)
    last = float(d[-1])
    if extend_to_nm <= last:
        raise ValueError(
            f"extend_path_to_nm ({extend_to_nm:.3f} nm) must be beyond the last "
            f"milestone ({last:.3f} nm); the path only runs bound -> unbound")
    gaps = np.abs(np.diff(d))
    spacing = float(np.median(gaps)) if gaps.size else (extend_to_nm - last)
    n_extra = max(1, int(round((extend_to_nm - last) / spacing)))
    extra = np.linspace(last, extend_to_nm, n_extra + 1)[1:]
    logger.info(
        "Extending the path to %.3f nm with %d milestone(s) at %.3f nm spacing "
        "(median of the existing spacings).",
        extend_to_nm, n_extra, (extend_to_nm - last) / n_extra)
    return np.concatenate([d, extra]).reshape(-1, 1)


def pathCV_cv(
    topology,
    milestones: list[str],
    pocket_atoms: list[int],
    ligand_atoms: list[int],
    system,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
    hill_width: float = 0.05,
    grid_points: int = 125,
    sigma: Union[float, str] = 'auto',
    extend_path_to_nm: float = None,
) -> CVSpec:
    """Path-in-CV-space CV through milestones evaluated with COM distance (deferred: needs positions).

    Progress is in [0, 1]. sigma (nm) is the kernel width; 'auto' uses half the
    minimum milestone spacing. Milestones may be ``.pdb`` or ``.xml`` (OpenMM State)
    and must share atom ordering with pocket_atoms/ligand_atoms.
    """
    def factory(input_positions, n_atoms, topology, sigma=sigma):
        pocket_masses = np.array([
            system.getParticleMass(i).value_in_unit(openmmunit.dalton)
            for i in pocket_atoms
        ])
        ligand_masses = np.array([
            system.getParticleMass(i).value_in_unit(openmmunit.dalton)
            for i in ligand_atoms
        ])

        milestones_array = []
        for milestone_path in milestones:
            positions = read_positions(milestone_path)
            pocket_com = np.average(positions[pocket_atoms], axis=0, weights=pocket_masses)
            ligand_com = np.average(positions[ligand_atoms], axis=0, weights=ligand_masses)
            com_dist = float(np.linalg.norm(ligand_com - pocket_com))
            milestones_array.append([round(com_dist, 4)])

        milestones_array = np.array(milestones_array)
        if extend_path_to_nm is not None:
            milestones_array = _extend_milestones(milestones_array,
                                                  float(extend_path_to_nm))
        logger.info(f"PathCV milestone COM distances (nm): {milestones_array.flatten().tolist()}")

        com_dists = milestones_array.flatten()
        steps = np.diff(com_dists)
        if np.any(steps < 0):
            logger.warning(
                "PathCV milestone COM distances are not monotonically increasing "
                "(%d of %d steps go backwards). The milestone list is meant to run "
                "bound -> unbound; a backwards step means the order is wrong, not "
                "that the path doubles back. Usual causes: milestone_mode='per_path' "
                "interleaves several paths into one CV, or PC1's sign was not "
                "anchored. The path CV will be built anyway, but its progress "
                "coordinate will not track unbinding.",
                int(np.sum(steps < 0)), steps.size)

        if sigma == 'auto':
            # |gaps|: cvpack silently accepts a negative sigma, which breaks the CV.
            gaps = np.abs(steps)
            usable = gaps[gaps > 1e-6]
            if usable.size == 0:
                raise ValueError(
                    f"cannot auto-tune sigma: the {com_dists.size} milestones are all "
                    f"at effectively the same pocket-ligand COM distance "
                    f"({com_dists.tolist()} nm), so there is no spacing to halve. "
                    "Pass an explicit sigma, or extract milestones that span the "
                    "unbinding coordinate.")
            sigma = 0.5 * float(usable.min())
            logger.info(f"Auto-tuned sigma for PathCV: {sigma:.4f} nm (0.5 × min milestone spacing)")

        cv_com_bias = cvpack.CentroidFunction(
            "sqrt(distance(g1,g2)^2)",
            openmmunit.nanometers,
            [pocket_atoms, ligand_atoms],
            weighByMass=True,
            pbc=False,
        )
        return cvpack.PathInCVSpace(
            metric=cvpack.path.progress,
            variables=[cv_com_bias],
            milestones=milestones_array,
            sigma=sigma,
        )

    # Angstrom per unit of progress, for the funnel correction; needs only the milestone files.
    _span_ang = None
    try:
        _ends = [read_positions(m) for m in (milestones[0], milestones[-1])]
        _d = []
        for _pos in _ends:
            _d.append(float(np.linalg.norm(_pos[ligand_atoms].mean(axis=0)
                                           - _pos[pocket_atoms].mean(axis=0))))
        _far = float(extend_path_to_nm) if extend_path_to_nm is not None else _d[-1]
        _span = abs(_far - _d[0]) * 10.0              # nm -> Angstrom
        _range = float(grid_max) - float(grid_min)
        if _span > 0 and _range > 0:
            _span_ang = _span / _range
            logger.info(
                "PathCV spans %.2f A of pocket-ligand separation over the CV range, "
                "so 1 unit of progress = %.2f A.", _span, _span_ang)
    except Exception as _e:
        logger.warning("Could not measure the PathCV length scale (%s); the funnel "
                       "standard-state correction will refuse to run.", _e)

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="pathCV_progress",
        cv_length_per_unit_ang=_span_ang,
        _factory=factory,
    )


CONTACTS_DEF_KEYS = ("group1", "group2", "threshold_nm", "step_function", "cutoff_factor",
                     "switch_factor", "bounds", "hill_width")


def _step_at_threshold(step_function: str) -> tuple:
    """(value, d/dx) of a cvpack step function at x = 1, evaluated by Lepton."""
    import openmm as mm
    s = mm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    f = mm.CustomBondForce(f"{step_function}; x=r")
    f.addBond(0, 1, [])
    s.addForce(f)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1, 0, 0)])
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(openmmunit.kilojoule_per_mole),
            -st.getForces(asNumpy=True)[1][0].value_in_unit(openmmunit.kilojoule_per_mole / openmmunit.nanometer))


def contacts_cv(
    group1: list[int],
    group2: list[int],
    system,
    threshold_nm: float,
    step_function: str,
    cutoff_factor: float,
    switch_factor: float,
    bounds: tuple,
    hill_width: float,
    grid_min: Optional[float] = None,
    grid_max: Optional[float] = None,
    name: str = "contacts",
) -> CVSpec:
    """Total group1-group2 contacts, sum of step(r/threshold) times OpenMM's switch (dimensionless).

    Built as ``cvpack.NumberOfContacts`` with reference 1, cutoff ``cutoff_factor * threshold_nm``
    and switching from ``switch_factor * threshold_nm``; PBC and exclusions follow the system's
    NonbondedForce. Clamped to ``bounds``, which are also the boundary-correction edges.

    Parameters
    ----------
    group1, group2 : disjoint system atom indices
    system : openmm.System with a NonbondedForce
    threshold_nm : contact distance r0 (nm)
    step_function : Lepton expression in x = r/r0, finite with finite derivative at x = 1
    cutoff_factor, switch_factor : cutoff and switching distance in units of r0
    bounds : finite (lo, hi) clamp and boundary-correction edges
    hill_width : hill width (contacts); grid spacing is hill_width / 5
    grid_min, grid_max : bias grid edges; default bounds -/+ 3 hill widths
    name : CV name
    """
    import openmm as mm
    from .BoundaryCorrection import LinearComposite
    g1, g2 = [int(i) for i in group1], [int(i) for i in group2]
    if set(g1) & set(g2):
        raise ValueError("contacts groups share atoms: not supported")
    lo, hi = (float(b) for b in bounds)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"contacts bounds {bounds} must be finite with hi > lo")
    v, dv = _step_at_threshold(step_function)
    if not (np.isfinite(v) and np.isfinite(dv)):
        raise ValueError(f"step function {step_function!r} is not finite at x = 1: not supported; "
                         "use its factored form")
    n_nb = sum(isinstance(f, mm.NonbondedForce) for f in system.getForces())
    if n_nb != 1:
        raise ValueError(f"contacts CV needs exactly one NonbondedForce, system has {n_nb}")
    hw = float(hill_width)
    gmin = lo - 3 * hw if grid_min is None else float(grid_min)
    gmax = hi + 3 * hw if grid_max is None else float(grid_max)

    def factory(input_positions, n_atoms, topology):
        # fetched here so the closure keeps the System (owner of the force) alive
        nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
        f = cvpack.NumberOfContacts(g1, g2, nb, reference=1.0, stepFunction=step_function,
                                    thresholdDistance=float(threshold_nm) * openmmunit.nanometer,
                                    cutoffFactor=float(cutoff_factor), switchFactor=float(switch_factor),
                                    name=name)
        return LinearComposite([f], [1.0], [0.0], [1.0], clamp=(lo, hi))

    return CVSpec(
        cv=None,
        grid_min=gmin,
        grid_max=gmax,
        hill_width=hw,
        grid_points=int(round((gmax - gmin) / (hw / 5))) + 1,
        name=name,
        bounds=(lo, hi),
        _factory=factory,
    )


def load_contacts_def(path: str) -> dict:
    """Read and validate a contact-CV JSON definition.

    Schema: ``{"contacts": {<CONTACTS_DEF_KEYS>}, "learned_cv_file": optional str}``.

    Returns
    -------
    dict with "contacts" (``contacts_cv`` keyword arguments, bounds as a tuple) and
    "learned_cv_file" (str or None). Raises ValueError on missing or unknown keys.
    """
    import json
    with open(path) as fh:
        d = json.load(fh)
    if not isinstance(d, dict) or not isinstance(d.get("contacts"), dict):
        raise ValueError(f"{path}: needs a 'contacts' object")
    unknown = set(d) - {"contacts", "learned_cv_file"}
    c = d["contacts"]
    missing = [k for k in CONTACTS_DEF_KEYS if k not in c]
    unknown |= {f"contacts.{k}" for k in set(c) - set(CONTACTS_DEF_KEYS)}
    if missing:
        raise ValueError(f"{path}: contacts is missing {missing}")
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if len(c["bounds"]) != 2:
        raise ValueError(f"{path}: contacts.bounds must be [lo, hi]")
    out = {k: c[k] for k in CONTACTS_DEF_KEYS}
    out["group1"], out["group2"] = [int(i) for i in c["group1"]], [int(i) for i in c["group2"]]
    out["bounds"] = (float(c["bounds"][0]), float(c["bounds"][1]))
    for k in ("threshold_nm", "cutoff_factor", "switch_factor", "hill_width"):
        out[k] = float(c[k])
    out["step_function"] = str(c["step_function"])
    lf = d.get("learned_cv_file")
    return {"contacts": out, "learned_cv_file": None if lf is None else str(lf)}
