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
    CVs that require input positions (RMSD-based) are deferred: they store a
    factory callable and materialize inside run() once positions are available.
    I had to do this because positions are available only after loading the checkpoint file,
    but CVs need to be defined before run() to set up the BiasVariables and grid.
    """
    cv: Any
    grid_min: float
    grid_max: float
    hill_width: float
    grid_points: int
    periodic: bool = False
    name: str = "cv"
    # Angstrom spanned per unit of this CV. The funnel correction integrates
    # exp(-beta W) over the bound site in LENGTH units; a progress coordinate is
    # dimensionless, so the integral needs this scale to mean anything.
    cv_length_per_unit_ang: Optional[float] = None
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
        )


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

    Deferred so that a fresh cvpack force is created on each run() call.
    OpenMM transfers C++ ownership of the force when addCollectiveVariable()
    is called, so reusing the same object across walkers causes an ownership error.
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
) -> CVSpec:
    """Guest position along the funnel axis, r_z = n·(COM_guest - COM_host) (nm).

    The coordinate the funnel restraint itself uses (mass-weighted centroids, no PBC),
    so the bias keeps acting up to the cylinder cap and the FES is in the length
    coordinate Limongelli Eq. 15 integrates over.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    host, guest = [int(i) for i in host_atoms], [int(i) for i in guest_atoms]

    def factory(input_positions, n_atoms, topology):
        # Per-collection parameters (1-element arrays): scalars would become global
        # Context parameters and collide with the funnel force's nx/ny/nz.
        return cvpack.CentroidFunction(
            "fz_nx*(x2-x1) + fz_ny*(y2-y1) + fz_nz*(z2-z1)",
            openmmunit.nanometers,
            [host, guest],
            weighByMass=True,
            pbc=False,
            fz_nx=[float(n[0])], fz_ny=[float(n[1])], fz_nz=[float(n[2])],
        )

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="funnel_z",
        cv_length_per_unit_ang=10.0,
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
    Note: This cv is aligned so only captures ligand internal conformational changes, not translation or rotation.
    """
    def factory(input_positions, n_atoms, topology):
        # input_positions in nm (OpenMM default); cvpack.RMSD expects nm
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
        # RelaxMD writes <milestone>_relax.pdb into the same directory; matching
        # those would build the path from relaxed structures instead of the
        # milestones themselves.
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

    Reads the token immediately before `_frame_`, so it is correct for both the
    pooled scheme (`milestone_1_frame_12`) and the per-path scheme
    (`milestone_0_1_frame_12`), where a positional split would return the path id.
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
    # _build_reference_dict reads each milestone with PDBFile, which fails with an
    # opaque IndexError on a serialized State; reject non-PDB paths up front.
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

    The path's far end sets what the progress coordinate calls unbound, so a path
    that stops while the guest still touches the host makes s=1 a contact state and
    no amount of sampling produces a flat reference. Extending with ONE distant
    milestone instead would leave a gap far wider than sigma, which degrades the
    progress coordinate into a staircase; keep the existing spacing.
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

    The path progress metric outputs values in [0, 1], so grid_min/grid_max should be 0.0/1.0.
    sigma controls the Gaussian kernel width (in nm); tune it so adjacent milestones overlap
    (a good rule of thumb: sigma ≈ half the minimum spacing between milestone COM distances).

    Milestones may be ``.pdb`` or ``.xml`` (serialized OpenMM State) files. Atom indices in
    pocket_atoms/ligand_atoms must match the atom ordering in the milestone files, which is
    guaranteed when milestones are extracted from the same solvated system.
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
            # Half the minimum spacing — guarantees kernels overlap at the closest pair
            # without over-smoothing at larger gaps.  mean-based sigma under-covers
            # the largest gap when milestone spacings are uneven.
            # Use |gaps|: a negative step (see the warning above) would otherwise make
            # sigma negative, which cvpack accepts and which silently breaks the CV.
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

    # How much physical distance one unit of progress spans. Computed eagerly:
    # it only needs the milestone files, not the deferred input positions. The
    # funnel standard-state correction integrates over the bound site in LENGTH
    # units, which a [0,1] progress coordinate cannot supply on its own.
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


def contacts_cv(
    pocket_atoms: list[int],
    ligand_atoms: list[int],
    system,
    grid_min: float,
    grid_max: float,
    hill_width: float = 5.0,
    grid_points: int = 100,
    step_function: str = "1/(1+x^6)",
    threshold_distance: float = 0.35,
    cutoff_factor: float = 2.0,
    switch_factor: float = 1.5,
    reference: int = 50,
) -> CVSpec:
    """Number of pocket-ligand contacts CV."""
    forces = {f.getName(): f for f in system.getForces()}
    cv = cvpack.NumberOfContacts(
        pocket_atoms,
        ligand_atoms,
        forces["NonbondedForce"],
        stepFunction=step_function,
        thresholdDistance=threshold_distance,
        cutoffFactor=cutoff_factor,
        switchFactor=switch_factor,
        reference=reference,
    )
    return CVSpec(
        cv=cv,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="nc",
    )
