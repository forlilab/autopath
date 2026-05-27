from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
import os
import logging
import numpy as np
import cvpack
import openmm.unit as openmmunit
from openmm.app import PDBFile

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
        milestone_pdbs = sorted(glob(f"{milestone_dir}/milestone_*.pdb"))
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
    """Extract the integer index from a milestone filename (<prefix>_<int>[_<suffix>].pdb)."""
    parts = os.path.basename(path).split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse milestone index from '{os.path.basename(path)}'. "
            f"Expected format: <prefix>_<integer>[_<suffix>].pdb"
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
) -> CVSpec:
    """Path-in-CV-space CV through milestones evaluated with COM distance (deferred: needs positions).

    The path progress metric outputs values in [0, 1], so grid_min/grid_max should be 0.0/1.0.
    sigma controls the Gaussian kernel width (in nm); tune it so adjacent milestones overlap
    (a good rule of thumb: sigma ≈ half the minimum spacing between milestone COM distances).

    Atom indices in pocket_atoms/ligand_atoms must match the PDB atom ordering in the milestone
    files, which is guaranteed when milestones are extracted from the same solvated system.
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
        for milestone_pdb in milestones:
            positions = np.array(
                PDBFile(milestone_pdb).positions.value_in_unit(openmmunit.nanometers)
            )
            pocket_com = np.average(positions[pocket_atoms], axis=0, weights=pocket_masses)
            ligand_com = np.average(positions[ligand_atoms], axis=0, weights=ligand_masses)
            com_dist = float(np.linalg.norm(ligand_com - pocket_com))
            milestones_array.append([round(com_dist, 4)])

        milestones_array = np.array(milestones_array)
        logger.info(f"PathCV milestone COM distances (nm): {milestones_array.flatten().tolist()}")

        if sigma == 'auto':
            # Half the minimum spacing — guarantees kernels overlap at the closest pair
            # without over-smoothing at larger gaps.  mean-based sigma under-covers
            # the largest gap when milestone spacings are uneven.
            sigma = 0.5 * np.min(np.diff(milestones_array.flatten()))
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

    return CVSpec(
        cv=None,
        grid_min=grid_min,
        grid_max=grid_max,
        hill_width=hill_width,
        grid_points=grid_points,
        name="pathCV_progress",
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
