"""CHARMM36 systems: nonbonded defaults and restraint selections in equilibration.

``SystemPreparation`` writes a chamber-style prmtop for CHARMM force fields, which
MDAnalysis cannot read, so ``Equilibration`` selects restrained atoms on the PDB.
"""

import functools
import json
from pathlib import Path

import MDAnalysis as mda
import pytest
import openmm
from openmm.app import ForceField, PDBFile

import autopath.equilibration as equilibration
from autopath.equilibration import Equilibration
from autopath.preparation import SystemPreparation
from autopath.utils import load_system

PEPTIDE = str(Path(__file__).parent / "data" / "tripeptide.pdb")
CHARMM = ["charmm36.xml", "charmm36/water.xml"]
PROTOCOLS = sorted(str(p) for p in (Path(equilibration.__file__).parent / "data").glob("eq_*.json"))
BB = {"CA", "C", "N", "O"}


def _cutoffs(sp):
    return (sp.nb_cutoff.value_in_unit(openmm.unit.nanometer),
            sp.switchDistance.value_in_unit(openmm.unit.nanometer))


def _prepare(out_dir, forcefield=None):
    """Prepare the peptide in water, recording the kwargs of every ``createSystem`` call."""
    calls = []
    orig = ForceField.createSystem

    @functools.wraps(orig)  # OpenMM checks kwargs against the wrapped signature
    def spy(self, topology, **kw):
        if "switchDistance" in kw:  # skip Modeller's internal calls
            calls.append(kw)
        return orig(self, topology, **kw)

    kw = {} if forcefield is None else {"forcefield": forcefield}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ForceField, "createSystem", spy)
        SystemPreparation(padding=1.2, boxShape="cube", out_dir=str(out_dir), **kw).run(protein=PEPTIDE)
    return calls


@pytest.fixture(scope="module")
def charmm_system(tmp_path_factory):
    out = tmp_path_factory.mktemp("charmm")
    return out, _prepare(out, CHARMM)


def test_amber_default_cutoff(tmp_path):
    assert _cutoffs(SystemPreparation(out_dir=str(tmp_path))) == pytest.approx((1.0, 0.9))


def test_charmm_default_cutoff(tmp_path):
    assert _cutoffs(SystemPreparation(forcefield=CHARMM, out_dir=str(tmp_path))) == pytest.approx((1.2, 1.0))


def test_explicit_cutoff_is_honoured(tmp_path):
    sp = SystemPreparation(forcefield=CHARMM, nonbondedCutoff=1.1, switchDistance=0.8,
                           out_dir=str(tmp_path))
    assert _cutoffs(sp) == pytest.approx((1.1, 0.8))


def test_switch_beyond_cutoff_raises(tmp_path):
    with pytest.raises(ValueError, match="switchDistance"):
        SystemPreparation(nonbondedCutoff=1.0, switchDistance=1.0, out_dir=str(tmp_path))


def test_charmm_cutoff_reaches_both_systems(charmm_system):
    out, calls = charmm_system
    assert len(calls) == 2
    for kw in calls:
        assert kw["nonbondedCutoff"].value_in_unit(openmm.unit.nanometer) == pytest.approx(1.2)
        assert kw["switchDistance"].value_in_unit(openmm.unit.nanometer) == pytest.approx(1.0)
    system = load_system(str(out / "system.xml"))
    # CHARMM puts LJ in a CustomNonbondedForce (NBFIX), electrostatics in NonbondedForce
    nbs = [f for f in system.getForces()
           if isinstance(f, (openmm.NonbondedForce, openmm.CustomNonbondedForce))]
    assert any(isinstance(f, openmm.CustomNonbondedForce) for f in nbs)
    for nb in nbs:
        assert nb.getCutoffDistance().value_in_unit(openmm.unit.nanometer) == pytest.approx(1.2)
        assert nb.getUseSwitchingFunction()
        assert nb.getSwitchingDistance().value_in_unit(openmm.unit.nanometer) == pytest.approx(1.0)


def test_charmm_equilibration_restrains_expected_atoms(charmm_system, tmp_path, monkeypatch):
    out, _ = charmm_system
    assert "%FLAG CTITLE" in (out / "system.prmtop").read_text()[:2000]  # chamber prmtop

    protocol = {
        "components_lookup": {"protein_BB": "protein and name CA C N O",
                              "protein_SC": "protein and not name CA C N O H*"},
        "minimization": [{"name": "Stage 1", "forces": [10.0, 10.0]}],
        "warmup": {"forces": [10.0, 10.0], "T_initial": 250.0, "T_final": 300.0,
                   "T_step": 25.0, "npt_flag": False, "nsteps": 20, "stepsize": 0.001},
        "equilibration": [
            {"name": "Stage 1", "forces": [5.0, 5.0], "npt_flag": False, "nsteps": 20, "stepsize": 0.001},
            {"name": "Stage 2", "forces": [0.0, 0.0], "npt_flag": True, "nsteps": 20, "stepsize": 0.001},
        ],
    }
    fname = tmp_path / "protocol.json"
    fname.write_text(json.dumps(protocol))

    restrained = []
    orig = equilibration.add_component_restraint

    def spy(system, spec, positions, topology, idx, force_name, force_group):
        restrained.append((force_name, sorted(int(i) for i in idx)))
        return orig(system, spec, positions, topology, idx, force_name, force_group)

    monkeypatch.setattr(equilibration, "add_component_restraint", spy)

    pdb = str(out / "system.pdb")
    topology = PDBFile(pdb).topology
    eq = Equilibration(topology, load_system(str(out / "system.xml")), out_dir=str(tmp_path / "eq"),
                       protocol_fname=str(fname), save_freq=10, platform="CPU", verbose=0)
    eq.run(pdb_file=pdb, run_id="t")

    peptide = [a for a in topology.atoms() if a.residue.chain.index == 0]
    expected = {
        "k_protein_BB": sorted(a.index for a in peptide if a.name in BB),
        "k_protein_SC": sorted(a.index for a in peptide
                               if a.name not in BB and not a.name.startswith("H")),
    }
    assert expected["k_protein_BB"] and expected["k_protein_SC"]
    # restraints are added before minimization and re-added after it
    assert restrained == list(expected.items()) * 2
    assert (tmp_path / "eq" / "system_equil_t.xml").is_file()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_amber_selections_match_prmtop(tmp_path_factory, protocol):
    out = tmp_path_factory.getbasetemp() / "amber"
    if not (out / "system.prmtop").is_file():
        _prepare(out)
    from_prmtop = mda.Universe(str(out / "system.prmtop"))
    from_pdb = mda.Universe(PDBFile(str(out / "system.pdb")))
    for value in json.loads(Path(protocol).read_text())["components_lookup"].values():
        sel = value if isinstance(value, str) else value["selection"]
        assert (from_prmtop.select_atoms(sel).indices == from_pdb.select_atoms(sel).indices).all(), sel
