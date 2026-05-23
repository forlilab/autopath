import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import tqdm
import MDAnalysis as mda

from .SMDData import SMDData

logger = logging.getLogger("autopath.pulling.LigandFeatures")


# Per-frame RDKit 3D shape descriptors. All are mass-weighted (PMI-based) or
# purely geometric, so they are invariant under permutation of same-element
# atoms — element-level matching between SDF and trajectory selection is
# sufficient to guarantee correctness. PMI1/2/3 are deliberately omitted:
# absolute-scale (Å^2·Da) and redundant with the mass-weighted
# RadiusOfGyration once the shape ratios above are kept.
def _rdkit_feature_fns() -> dict[str, Callable]:
    """Build the name → descriptor-callable map lazily to avoid top-level
    RDKit imports (the class only needs RDKit when an rdkit feature is
    actually requested)."""
    from rdkit.Chem import Descriptors3D, rdMolDescriptors
    return {
        "asphericity":    Descriptors3D.Asphericity,
        "eccentricity":   Descriptors3D.Eccentricity,
        "inertial_shape": Descriptors3D.InertialShapeFactor,
        "npr1":           Descriptors3D.NPR1,
        "npr2":           Descriptors3D.NPR2,
        "spherocity":     Descriptors3D.SpherocityIndex,
        "pbf":            rdMolDescriptors.CalcPBF,
    }


RDKIT_FEATURE_NAMES = frozenset({
    "asphericity", "eccentricity", "inertial_shape",
    "npr1", "npr2", "spherocity", "pbf",
})

SUPPORTED_FEATURES = frozenset({"rog"}) | RDKIT_FEATURE_NAMES


class LigandTrajectoryFeatures:
    """Per-frame ligand geometric features from sMD trajectories.

    Computes quantities (radius of gyration, …) at a given stride and
    returns a DataFrame that can be:
      - merged into the clustering feature matrix (alongside work, lag, pocket distances)
      - merged into sMD_processed_data.csv for downstream Bayesian analysis

    Both uses go through SMDData.merge_feature_sets(), which handles stride
    differences between log files and trajectory output via pd.merge_asof.

    Parameters
    ----------
    lig_resname : str
        MDAnalysis residue name for the ligand (default "UNK").
    sdf_file : str or None
        Path to an SDF with correct bond orders.  Required when
        ``"rdkit_3d"`` is in ``features``; used as the topology template
        whose conformer is mutated per frame.  Also used by
        :meth:`rdkit_descriptors_from_sdf` for topology-only descriptors.
    features : sequence of str
        Per-frame features to compute. Each name maps 1:1 to one output
        column ``lig_<name>``. Supported names:

        - ``"rog"``: radius of gyration (mass-weighted COM, mass-weighted
          spread). Matches the convention used by RDKit's
          ``Descriptors3D.RadiusOfGyration``.
        - RDKit 3D shape descriptors (any subset): ``"asphericity"``,
          ``"eccentricity"``, ``"inertial_shape"``, ``"npr1"``, ``"npr2"``,
          ``"spherocity"``, ``"pbf"``. Any of these requires ``sdf_file``.

        ``rog`` alone does not require an SDF.
    stride : int
        Sample every `stride`-th trajectory frame (default 2).

    Notes
    -----
    For RDKit descriptors the SDF and the MDAnalysis selection must agree
    on heavy-atom **element order**. The class validates this at the start
    of :meth:`compute` and raises on mismatch.
    """

    def __init__(
        self,
        lig_resname: str = "UNK",
        sdf_file: Optional[str] = None,
        features: list | tuple = ("rog",),
        stride: int = 2,
    ):
        features = list(features)
        unknown = set(features) - SUPPORTED_FEATURES
        if unknown:
            raise ValueError(
                f"Unsupported features: {sorted(unknown)}. "
                f"Supported: {sorted(SUPPORTED_FEATURES)}"
            )
        rdkit_requested = [f for f in features if f in RDKIT_FEATURE_NAMES]
        if rdkit_requested and sdf_file is None:
            raise ValueError(
                f"RDKit features {rdkit_requested} require sdf_file to be provided."
            )
        self.lig_resname = lig_resname
        self.sdf_file = sdf_file
        self.features = features
        self._rdkit_features = rdkit_requested
        self.stride = stride

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, traj_files: list, reference_pdb: str) -> pd.DataFrame:
        """Compute per-frame ligand features for all trajectories.

        Returns
        -------
        pd.DataFrame
            Columns: trajname, speed, time, [feature columns].
            One row per trajectory frame sampled at self.stride.
            Empty DataFrame if no trajectories could be processed.
        """
        sel = f"resname {self.lig_resname} and not name H*"
        all_rows = []

        # Build the RDKit template and descriptor-function map once per
        # compute() call (heavy-atoms only). Element-order validation happens
        # lazily against the first universe.
        rdkit_template = None
        rdkit_fns: dict[str, Callable] = {}
        rdkit_validated = False
        if self._rdkit_features:
            rdkit_template = self._load_rdkit_template(self.sdf_file)
            all_fns = _rdkit_feature_fns()
            rdkit_fns = {name: all_fns[name] for name in self._rdkit_features}

        compute_rog = "rog" in self.features

        for traj in tqdm.tqdm(traj_files, desc="Computing ligand features"):
            try:
                u = mda.Universe(reference_pdb, traj)
            except Exception as exc:
                logger.warning(f"Could not load {traj}: {exc}")
                continue

            lig = u.select_atoms(sel)
            if lig.n_atoms == 0:
                logger.warning(f"No ligand atoms found in {traj} with '{sel}'")
                continue

            if rdkit_template is not None and not rdkit_validated:
                self._validate_atom_order(rdkit_template, lig)
                rdkit_validated = True

            traj_name = SMDData._traj_to_log_name(traj)
            try:
                speed = float(traj_name.split("_")[-2].lstrip("v"))
            except (IndexError, ValueError):
                speed = float("nan")

            for ts in u.trajectory[:: self.stride]:
                row = {
                    "trajname": traj_name,
                    "speed": speed,
                    "time": float(getattr(ts, "time", ts.frame)),
                }
                if compute_rog:
                    row["lig_rog"] = self._compute_rog(lig)
                if rdkit_fns:
                    self._set_template_coords(rdkit_template, lig)
                    for name, fn in rdkit_fns.items():
                        row[f"lig_{name}"] = round(float(fn(rdkit_template)), 4)
                all_rows.append(row)

        if not all_rows:
            logger.warning("LigandTrajectoryFeatures: no features could be computed.")
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    @staticmethod
    def rdkit_descriptors_from_sdf(sdf_file: str) -> dict:
        """Compute fixed (topology-based) RDKit molecular descriptors.

        These do not vary per frame; use them as molecule-level metadata
        (e.g. to cross-check MW or LogP, or to set a Bayesian prior).

        Parameters
        ----------
        sdf_file : str
            Path to an SDF file with the ligand and correct bond orders.

        Returns
        -------
        dict
            Keys: mw, logp, hbd, hba, tpsa, rotatable_bonds, heavy_atom_count
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise ValueError(f"No valid molecule found in {sdf_file}")
        mol_noH = Chem.RemoveHs(mols[0])

        return {
            "mw": round(Descriptors.MolWt(mol_noH), 3),
            "logp": round(Descriptors.MolLogP(mol_noH), 3),
            "hbd": rdMolDescriptors.CalcNumHBD(mol_noH),
            "hba": rdMolDescriptors.CalcNumHBA(mol_noH),
            "tpsa": round(Descriptors.TPSA(mol_noH), 3),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol_noH),
            "heavy_atom_count": mol_noH.GetNumHeavyAtoms(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rog(lig_atoms) -> float:
        """Mass-weighted radius of gyration of heavy atoms, in nm.

        Matches the convention used by RDKit's ``Descriptors3D.RadiusOfGyration``
        and the PMI-based shape descriptors emitted by ``rdkit_3d``: both COM
        and spread are mass-weighted.
        """
        pos = lig_atoms.positions         # Angstrom, shape (N, 3)
        com = lig_atoms.center_of_mass()  # Angstrom, shape (3,) — mass-weighted
        d2 = np.sum((pos - com) ** 2, axis=1)
        rog_ang = np.sqrt(np.average(d2, weights=lig_atoms.masses))
        return round(rog_ang / 10.0, 4)  # Å → nm

    @staticmethod
    def _load_rdkit_template(sdf_file: str):
        """Load the first molecule from the SDF, strip Hs, ensure a conformer."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise ValueError(f"No valid molecule found in {sdf_file}")
        mol = Chem.RemoveHs(mols[0])
        if mol.GetNumConformers() == 0:
            raise ValueError(
                f"SDF molecule has no 3D conformer: {sdf_file}. "
                f"Descriptors3D needs an existing conformer to mutate."
            )
        return mol

    @staticmethod
    def _validate_atom_order(mol_template, lig_atoms) -> None:
        """Element-by-element check between the RDKit template and MDA selection.

        Mass-weighted/geometric shape descriptors are permutation-invariant
        within same-element atoms, so element-level matching is sufficient.
        Raises ValueError on count or element-sequence mismatch.
        """
        n_template = mol_template.GetNumHeavyAtoms()
        if n_template != lig_atoms.n_atoms:
            raise ValueError(
                f"Atom-count mismatch: SDF template has {n_template} heavy atoms, "
                f"MDA selection has {lig_atoms.n_atoms}."
            )

        rdkit_elements = [a.GetSymbol() for a in mol_template.GetAtoms()]
        try:
            mda_elements = [str(e) for e in lig_atoms.elements]
        except (mda.NoDataError, AttributeError):
            # Fall back to first letter of atom name (best-effort).
            mda_elements = [n[0].upper() for n in lig_atoms.names]

        mismatches = [
            (i, r, m)
            for i, (r, m) in enumerate(zip(rdkit_elements, mda_elements))
            if r.upper() != m.upper()
        ]
        if mismatches:
            preview = ", ".join(f"#{i}: SDF={r} vs MDA={m}" for i, r, m in mismatches[:5])
            raise ValueError(
                f"Atom-element order mismatch between SDF and trajectory "
                f"(first {len(mismatches)} of {len(rdkit_elements)} disagree): {preview}"
            )

    @staticmethod
    def _set_template_coords(mol_template, lig_atoms) -> None:
        """Inject Å coordinates from the MDA selection into the template conformer.

        Both MDAnalysis.positions and RDKit Point3D are in Å; no conversion.
        Called once per frame before evaluating any descriptor.
        """
        from rdkit.Geometry import Point3D

        conf = mol_template.GetConformer()
        for i, xyz in enumerate(lig_atoms.positions):
            conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
