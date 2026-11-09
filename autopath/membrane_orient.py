"""Rotate a membrane protein upright before the bilayer is built.

``Modeller.addMembrane`` builds the bilayer flat in the xy plane and never moves the
protein, so the protein must already be upright when it arrives. Two later steps assume
the same thing and neither checks it:
:func:`autopath.customForces.add_harmonic_z_restraints` keeps only the z component of each
lipid headgroup, and :func:`autopath.utils.add_barostat` returns a
``MonteCarloMembraneBarostat`` fixed to z. A tilted system still builds and still runs to
the end. It gives a wrong answer with no error.

:func:`orient_membrane` is the entry point. Give it the paths you were about to pass to
``SystemPreparation`` and use what it returns.

PPM 3.0 (``immers``) does the work. It is the program the OPM database is built from.
See ``examples/09_Membrane-Orientation/README.md`` for installation and accuracy.
"""
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict

import numpy as np
from openmm import Vec3
from openmm.app import Modeller, PDBFile
import openmm.unit as openmmunit

from autopath.metadynamics.Funnel import kabsch_rotation, rotation_angle_deg
from autopath.metadynamics.learned.water import AMINO

logger = logging.getLogger("autopath.membrane_orient")

#: Residue names treated as protein. AMINO holds the Amber set; the rest are CHARMM.
PROTEIN_RESIDUES = AMINO | frozenset("HSD HSE HSP TYM".split())

#: Fewest CA atoms that make a superposition worth trusting.
MIN_MATCHED_CA = 10

#: Superposition RMSD in nm above which PPM did not move the structure rigidly.
FIT_RMSD_TOL_NM = 0.01

#: Seconds to wait for PPM before giving up.
PPM_TIMEOUT_S = 900

_Z = np.array([0.0, 0.0, 1.0])


class OrientationError(RuntimeError):
    """PPM could not produce a transform worth applying."""


@dataclass(frozen=True)
class MembraneFrame:
    """A rigid-body transform into the membrane frame.

    Applied to row vectors as ``new = old @ rotation.T + translation``.

    Attributes
    ----------
    rotation : numpy.ndarray
        3x3 proper rotation.
    translation : numpy.ndarray
        Shift in nm, applied after the rotation.
    fit_rmsd : float
        Superposition residual in nm. Near zero means PPM moved the structure rigidly.
    """

    rotation: np.ndarray
    translation: np.ndarray
    fit_rmsd: float

    @property
    def normal(self) -> np.ndarray:
        """Membrane normal, in the input coordinate frame."""
        return self.rotation.T @ _Z

    def apply(self, positions_nm) -> np.ndarray:
        """Transform an ``(n, 3)`` array of positions in nm."""
        return np.asarray(positions_nm, float) @ self.rotation.T + self.translation

    def is_identity(self) -> bool:
        """True when applying this transform would not move anything."""
        return (
            rotation_angle_deg(self.rotation) < 0.01
            and float(np.abs(self.translation).max()) < 1e-4
        )

    def as_record(self) -> dict:
        """Return the transform as JSON-serialisable data, with its units."""
        rec = asdict(self)
        for key in ("rotation", "translation"):
            rec[key] = np.asarray(rec[key], float).tolist()
        rec["normal"] = self.normal.tolist()
        rec["method"] = "ppm"
        rec["units"] = "nanometer"
        rec["convention"] = "new = old @ rotation.T + translation"
        rec["normal_frame"] = "input"
        rec["rotation_angle_deg"] = rotation_angle_deg(self.rotation)
        return rec


def _packmol_memgen_dir():
    """Return the installed packmol_memgen directory, or None."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("packmol_memgen")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _find_ppm():
    """Return a usable ``immers`` binary, or None.

    ``AUTOPATH_PPM`` is checked first. The ``immers`` that packmol-memgen ships is
    broken on every membrane run, so most sites need their own build.
    """
    candidates = [os.environ.get("AUTOPATH_PPM"), shutil.which("immers")]
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        candidates.append(os.path.join(amberhome, "bin", "immers"))
    pkg = _packmol_memgen_dir()
    if pkg:
        candidates.append(os.path.join(pkg, "lib", "ppm3", "immers"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_res_lib(ppm_path):
    """Return a non-empty ``res.lib``, or None.

    The file must be non-empty. Fortran opens a missing file with ``status='unknown'``,
    which creates it empty. PPM then runs with no residue data, writes a believable
    output and exits 0, so a missing res.lib is otherwise silent.
    """
    candidates = [os.path.join(os.path.dirname(ppm_path), "res.lib")]
    pkg = _packmol_memgen_dir()
    if pkg:
        candidates.append(os.path.join(pkg, "lib", "ppm3", "res.lib"))
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        candidates.extend(
            glob.glob(
                os.path.join(
                    amberhome, "lib", "python*", "site-packages",
                    "packmol_memgen", "lib", "ppm3", "res.lib",
                )
            )
        )
    for candidate in candidates:
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _write_ppm_input(topology, positions_nm, path):
    """Write a protein-only, hydrogen-free PDB and return its sorted chain IDs.

    Water is removed because OpenMM writes ``HOH`` as ``ATOM`` records, so the chain ID
    harvest would treat water as protein. Ligands are removed because a bound ligand can
    pull PPM's hydrophobic belt off the real one. The recovered transform is applied to
    everything afterwards regardless.
    """
    modeller = Modeller(topology, np.asarray(positions_nm, float) * openmmunit.nanometer)
    drop = [r for r in modeller.topology.residues() if r.name not in PROTEIN_RESIDUES]
    if drop:
        modeller.delete(drop)
    hydrogens = [
        a for a in modeller.topology.atoms()
        if a.element is not None and a.element.symbol == "H"
    ]
    if hydrogens:
        modeller.delete(hydrogens)
    if modeller.topology.getNumAtoms() == 0:
        raise OrientationError("no protein atoms to send to PPM")
    PDBFile.writeFile(modeller.topology, modeller.positions, path, keepIds=True)
    return sorted({c.id.strip() for c in modeller.topology.chains() if c.id.strip()})


def _ca_map(pdb_path):
    """Return ``{(chain, resSeq, icode): xyz_nm}`` for CA atoms, read from the columns.

    Keyed rather than indexed because PPM does not return every record, so matching by
    position in the file would pair up different atoms. Alternate locations other than A
    are skipped so a residue is not matched against another conformer of itself.
    """
    out = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
                continue
            if line[17:20].strip() == "DUM":
                continue
            out[(line[21], line[22:26].strip(), line[26])] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            ) / 10.0
    return out


def _recover_transform(input_pdb, output_pdb):
    """Work out the rotation and shift PPM applied, and check it was rigid.

    Parameters
    ----------
    input_pdb : str
        The PDB given to PPM.
    output_pdb : str
        The oriented PDB PPM wrote.

    Returns
    -------
    tuple
        ``(rotation, translation, rmsd)``.

    Raises
    ------
    OrientationError
        PPM dropped or renumbered the structure, bent it, or mirrored it.
    """
    before, after = _ca_map(input_pdb), _ca_map(output_pdb)
    shared = sorted(set(before) & set(after))
    if len(shared) < MIN_MATCHED_CA:
        raise OrientationError(
            f"PPM returned {len(shared)} matchable CA atoms (need {MIN_MATCHED_CA}); it "
            "dropped, renamed or renumbered the structure"
        )
    x = np.array([before[k] for k in shared])
    y = np.array([after[k] for k in shared])

    rotation = kabsch_rotation(x, y)
    xc = (x - x.mean(axis=0)) @ rotation.T
    rmsd = float(np.sqrt(((xc - (y - y.mean(axis=0))) ** 2).sum(axis=1).mean()))
    if rmsd > FIT_RMSD_TOL_NM:
        raise OrientationError(
            f"PPM did not move the structure rigidly: superposition RMSD {rmsd:.3f} nm "
            f"> {FIT_RMSD_TOL_NM} nm"
        )
    if np.linalg.det(rotation) <= 0 or not np.allclose(
        rotation @ rotation.T, np.eye(3), atol=1e-6
    ):
        raise OrientationError("PPM implied a reflection rather than a rotation")

    return rotation, -x.mean(axis=0) @ rotation.T + y.mean(axis=0), rmsd


def orient_to_membrane_frame(topology, positions_nm, n_term="in", work_dir=None):
    """Run PPM and return the transform that puts the protein upright.

    PPM returns an oriented copy of the PDB it was given, so its own idea of the normal
    is never read. The transform is recovered by superposing that copy back onto the
    input, which also proves the structure did not bend or mirror.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the structure to orient.
    positions_nm : numpy.ndarray
        ``(n, 3)`` positions in nm.
    n_term : str
        ``"in"`` or ``"out"``, which side the N-terminus sits on. Passed to PPM. It
        fixes the sign of z, which geometry alone cannot determine.
    work_dir : str, optional
        Where ``ppm.log`` is written. Defaults to the current directory.

    Returns
    -------
    MembraneFrame

    Raises
    ------
    OrientationError
        PPM is missing, timed out, refused the structure, or returned something
        unusable. Nothing is guessed: a wrong orientation that looks plausible costs
        GPU-days before anyone sees it, an error costs a minute.
    """
    if n_term not in ("in", "out"):
        raise ValueError(f"n_term must be 'in' or 'out', got {n_term!r}")
    work_dir = work_dir or "."
    os.makedirs(work_dir, exist_ok=True)

    exe = _find_ppm()
    if exe is None:
        raise OrientationError(
            "no usable 'immers' binary found. Set AUTOPATH_PPM to a PPM 3.0 build; see "
            "examples/09_Membrane-Orientation/README.md."
        )
    res_lib = _find_res_lib(exe)
    if res_lib is None:
        raise OrientationError(
            "found 'immers' but no non-empty res.lib; PPM would run with an empty "
            "residue dictionary and still exit 0"
        )

    # PPM writes fixed filenames, so it gets a private directory. Never os.chdir: this
    # can be called concurrently.
    tmp = tempfile.mkdtemp(prefix="ppm_", dir=work_dir)
    try:
        pdb = os.path.join(tmp, "ppm3tmp.pdb")
        chains = _write_ppm_input(topology, positions_nm, pdb)
        shutil.copy(res_lib, os.path.join(tmp, "res.lib"))

        inp = os.path.join(tmp, "ppm.inp")
        with open(inp, "w") as handle:
            handle.write(
                "2\nno\nppm3tmp.pdb\n1\n    \nplanar\n{}\n{}\n".format(
                    n_term, ",".join(chains)
                )
            )

        log_path = os.path.join(work_dir, "ppm.log")
        try:
            with open(inp) as stdin, open(log_path, "w") as stdout:
                subprocess.run(
                    [exe], cwd=tmp, stdin=stdin, stdout=stdout,
                    stderr=subprocess.STDOUT, timeout=PPM_TIMEOUT_S, check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise OrientationError(f"PPM timed out after {PPM_TIMEOUT_S} s") from exc
        except OSError as exc:
            raise OrientationError(f"could not run {exe}: {exc}") from exc

        with open(log_path, errors="replace") as handle:
            log = handle.read()
        if "Peripheral protein" in log:
            raise OrientationError(
                "PPM says this is a peripheral protein; addMembrane cannot place a "
                "monotopic protein meaningfully"
            )
        out = os.path.join(tmp, "ppm3tmpout.pdb")
        if not os.path.isfile(out):
            raise OrientationError(
                f"PPM wrote no output. Log tail: {' '.join(log.split()[-40:])}"
            )

        rotation, translation, rmsd = _recover_transform(pdb, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # No re-centring in z. PPM writes its boundary planes symmetrically about z=0, the
    # same convention addMembrane builds to, so the recovered transform is already
    # centred.
    return MembraneFrame(rotation=rotation, translation=translation, fit_rmsd=rmsd)


def write_orientation_record(path, frame, extra=None):
    """Write the transform to ``path`` as JSON."""
    record = frame.as_record()
    if extra:
        record.update(extra)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)


def remark_lines(frame):
    """Return ``REMARK`` lines that make a written PDB self-describing."""
    return [
        "REMARK 100 AUTOPATH MEMBRANE ORIENTATION\n",
        f"REMARK 100   method=ppm rotated "
        f"{rotation_angle_deg(frame.rotation):.2f}deg onto the membrane normal\n",
        f"REMARK 100   fit_rmsd={frame.fit_rmsd:.4f}nm\n",
        "REMARK 100   transform recorded in membrane_orientation.json\n",
    ]


def _write_oriented_pdb(topology, positions_nm, path, frame):
    """Write a PDB with the REMARK record and no unit cell.

    The unit cell is dropped because a rotation invalidates it. addMembrane would
    otherwise size the box from the stale vectors.
    """
    topology.setPeriodicBoxVectors(None)
    quantity = [Vec3(*p) for p in positions_nm] * openmmunit.nanometer
    with open(path, "w") as handle:
        handle.writelines(remark_lines(frame))
        PDBFile.writeFile(topology, quantity, handle, keepIds=True)


def _orient_sdf_file(in_path, out_path, frame):
    """Apply ``frame`` to every conformer of every record, working in angstroms."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rotation_t = frame.rotation.T
    translation = np.asarray(frame.translation, float) * 10.0   # nm -> angstrom

    writer = Chem.SDWriter(out_path)
    try:
        # ForwardSDMolSupplier, not SDMolSupplier: the random-access reader silently
        # drops every record past the first 64 KiB.
        with open(in_path, "rb") as handle:
            for mol in Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False):
                if mol is None:
                    continue
                for conformer in mol.GetConformers():
                    moved = np.asarray(conformer.GetPositions()) @ rotation_t + translation
                    for index, point in enumerate(moved):
                        conformer.SetAtomPosition(index, Point3D(*map(float, point)))
                if mol.GetNumConformers() > 1:
                    # Docked poses are often conformers of one record; write them all.
                    for conformer in mol.GetConformers():
                        writer.write(mol, confId=conformer.GetId())
                else:
                    writer.write(mol)
    finally:
        writer.close()


def _orient_one_file(path, out_dir, frame):
    """Write an oriented copy of one PDB or SDF and return its path."""
    stem, extension = os.path.splitext(os.path.basename(path))
    out_path = os.path.join(out_dir, f"{stem}_oriented{extension}")
    if extension.lower() == ".sdf":
        _orient_sdf_file(path, out_path, frame)
    elif extension.lower() == ".pdb":
        pdb = PDBFile(path)
        positions = np.array(pdb.positions.value_in_unit(openmmunit.nanometer))
        _write_oriented_pdb(pdb.topology, frame.apply(positions), out_path, frame)
    else:
        raise OrientationError(
            f"cannot orient {path!r}: expected a .sdf or .pdb ligand, got {extension!r}"
        )
    return out_path


def orient_membrane(receptor, ligands=None, out_dir=".", n_term="in"):
    """Rotate the receptor upright and move every ligand with it.

    Call this on the paths you were about to pass to
    :class:`~autopath.preparation.SystemPreparation`, and use what it returns::

        receptor_path, ligands = orient_membrane(receptor_path, ligands, sys_name)

    One transform is used for every file, so docked poses stay in the pocket.

    Parameters
    ----------
    receptor : str
        Path to the receptor PDB.
    ligands : None or str or list
        ``None``, a path, or the list of ``(resname, path, smiles, from_xray)`` tuples
        the preparation scripts build. Returned in the shape it came in.
    out_dir : str
        Where the oriented copies and ``membrane_orientation.json`` are written.
    n_term : str
        ``"in"`` or ``"out"``, which side the N-terminus sits on.

    Returns
    -------
    tuple
        ``(receptor_path, ligands)``, pointing at the oriented copies. The inputs come
        back unchanged if the receptor is already upright.

    Raises
    ------
    OrientationError
        PPM failed, or a ligand was in a format this cannot move.
    """
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    pdb = PDBFile(receptor)
    positions = np.array(pdb.positions.value_in_unit(openmmunit.nanometer))
    frame = orient_to_membrane_frame(pdb.topology, positions, n_term=n_term,
                                     work_dir=out_dir)

    if frame.is_identity():
        logger.info("Receptor is already upright in the membrane frame; nothing written.")
        return receptor, ligands

    write_orientation_record(
        os.path.join(out_dir, "membrane_orientation.json"), frame,
        extra={"receptor": os.path.abspath(receptor)},
    )

    stem = os.path.splitext(os.path.basename(receptor))[0]
    oriented_receptor = os.path.join(out_dir, f"{stem}_oriented.pdb")
    _write_oriented_pdb(pdb.topology, frame.apply(positions), oriented_receptor, frame)
    logger.info(
        f"PPM rotated the receptor {rotation_angle_deg(frame.rotation):.1f} deg onto the "
        f"membrane normal. Wrote {oriented_receptor}."
    )

    if ligands is None:
        return oriented_receptor, None

    one = isinstance(ligands, str)
    oriented_ligands = []
    for item in ([ligands] if one else list(ligands)):
        path = item if isinstance(item, str) else item[1]
        if path is None:
            oriented_ligands.append(item)
            continue
        moved = _orient_one_file(path, out_dir, frame)
        oriented_ligands.append(moved if isinstance(item, str)
                                else (item[0], moved, *item[2:]))
        logger.info(f"Moved {path} with the receptor -> {moved}")

    return oriented_receptor, (oriented_ligands[0] if one else oriented_ligands)
