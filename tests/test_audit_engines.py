"""Engine side paths: VanillaMD keeps its system clean, prep rejects protein=None,
contact frequency is periodic."""
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest


def _two_atom_pdb(path, box_nm=3.0):
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    top.addAtom("CA", app.element.carbon, res)
    top.addAtom("CB", app.element.carbon, res)
    top.setPeriodicBoxVectors(np.eye(3) * box_nm * unit.nanometers)
    pos = [openmm.Vec3(1.0, 1.0, 1.0), openmm.Vec3(1.15, 1.0, 1.0)] * unit.nanometers
    with open(path, "w") as f:
        app.PDBFile.writeFile(top, pos, f)
    return top


def test_vanilla_md_does_not_restrain_its_own_system(tmp_path, monkeypatch):
    import autopath.vanilla_md as vm
    monkeypatch.setattr(vm, "add_reporters", lambda *a, **k: None)
    pdb = tmp_path / "s.pdb"
    top = _two_atom_pdb(pdb)
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0 * unit.nanometers))
    for _ in range(2):
        system.addParticle(12.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 1000.0)
    system.addForce(bond)
    md = vm.VanillaMD(system=system, topology=top, restrained_atoms=[0],
                      out_dir=str(tmp_path), platform="Reference", verbose=0)
    for run_id in ("a", "b"):
        md.run(pdb_file=str(pdb), run_id=run_id, MD_time=0.00001)
    assert system.getNumForces() == 1     # restraints never land on the shared system
    saved = openmm.XmlSerializer.deserialize((tmp_path / "MD_b_system.xml").read_text())
    assert saved.getNumForces() == 1


def test_preparation_rejects_missing_protein(tmp_path):
    from autopath.preparation import SystemPreparation
    with pytest.raises(ValueError, match="not supported"):
        SystemPreparation(out_dir=str(tmp_path)).run(protein=None)


def test_contact_frequency_across_box_edge():
    import MDAnalysis as mda
    from autopath.ap_PLIP import calculate_contact_frequency
    u = mda.Universe.empty(2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
    u.add_TopologyAttr("name", ["C1", "CA"])
    u.add_TopologyAttr("resname", ["UNK", "ALA"])
    u.add_TopologyAttr("resid", [1, 2])
    u.atoms.positions = np.array([[0.5, 50.0, 50.0], [99.0, 50.0, 50.0]])  # 1.5 Å apart via PBC
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    freq = calculate_contact_frequency(u, "resname UNK", pocket_cutoff=5.0, contact_cutoff=3.5)
    assert freq.tolist() == [1.0]
