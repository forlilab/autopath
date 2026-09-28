import os, numpy as np, pytest
import openmm as mm
from openmm import unit as ommunit
from openmm import app
from autopath.Clustering import read_positions


def _tiny_system(n=6):
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3, 0, 0) * ommunit.nanometer,
        mm.Vec3(0, 3, 0) * ommunit.nanometer,
        mm.Vec3(0, 0, 3) * ommunit.nanometer)
    return system


def test_written_state_roundtrips_positions_and_box(tmp_path):
    """Review Focus #3: box vectors must survive, else the restart silently
    adopts the System default box."""
    system = _tiny_system()
    integ = mm.LangevinMiddleIntegrator(300 * ommunit.kelvin,
                                        1 / ommunit.picosecond,
                                        0.002 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    pos = np.arange(18, dtype=float).reshape(6, 3) * 0.1
    box = [mm.Vec3(4.1, 0, 0), mm.Vec3(0, 4.2, 0), mm.Vec3(0, 0, 4.3)]
    ctx.setPeriodicBoxVectors(*[b * ommunit.nanometer for b in box])
    ctx.setPositions(pos * ommunit.nanometer)
    ctx.setVelocitiesToTemperature(300 * ommunit.kelvin)
    state = ctx.getState(getPositions=True, getVelocities=True)
    p = tmp_path / "s.xml"
    p.write_text(mm.XmlSerializer.serialize(state))

    back = mm.XmlSerializer.deserialize(p.read_text())
    np.testing.assert_allclose(
        back.getPositions(asNumpy=True).value_in_unit(ommunit.nanometer), pos,
        rtol=0, atol=1e-6)
    bv = back.getPeriodicBoxVectors()
    assert bv[0][0].value_in_unit(ommunit.nanometer) == pytest.approx(4.1)
    assert bv[2][2].value_in_unit(ommunit.nanometer) == pytest.approx(4.3)
    v = back.getVelocities(asNumpy=True).value_in_unit(
        ommunit.nanometer / ommunit.picosecond)
    assert np.abs(v).sum() > 0                      # velocities present


def test_write_states_roundtrips_a_real_trajectory(tmp_path):
    """write_states is this task's centrepiece; exercise it end-to-end against a
    real Universe rather than only testing the reader."""
    import MDAnalysis as mda
    from MDAnalysis.tests.datafiles import PSF, DCD
    from openmm import app as _app
    from autopath.Clustering import Clustering, PairwiseDistances

    from MDAnalysis.transformations.boxdimensions import set_dimensions
    u = mda.Universe(PSF, DCD)
    # PSF/DCD carries no unit cell; write_states refuses a box-less frame by
    # design, so give the fixture a deterministic box.
    u.trajectory.add_transformations(set_dimensions([60., 60., 60., 90., 90., 90.]))
    n = u.atoms.n_atoms
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    top = _app.Topology()
    ch = top.addChain()
    for _ in range(n):
        r = top.addResidue("UNK", ch)
        top.addAtom("C", _app.Element.getBySymbol("C"), r)

    c = Clustering(PairwiseDistances("resid 1-5 and name CA",
                                     "resid 20-30 and name CA")).fit(u)
    frames = c.select(2, strategy="time_spaced")
    out = c.write_states(u, frames, top, system, str(tmp_path), "ms",
                         temperature=300.0)
    assert len(out) == 2
    u.trajectory[int(frames[0])]
    expected = u.atoms.positions / 10.0
    st = mm.XmlSerializer.deserialize(open(out[0]).read())
    got = st.getPositions(asNumpy=True).value_in_unit(ommunit.nanometer)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
    bv = st.getPeriodicBoxVectors()
    assert bv[0][0].value_in_unit(ommunit.nanometer) == pytest.approx(6.0)
    assert bv[2][2].value_in_unit(ommunit.nanometer) == pytest.approx(6.0)
    assert np.abs(st.getVelocities(asNumpy=True).value_in_unit(
        ommunit.nanometer / ommunit.picosecond)).sum() > 0


def test_write_states_rejects_atom_count_mismatch(tmp_path):
    import MDAnalysis as mda
    from MDAnalysis.tests.datafiles import PSF, DCD
    from openmm import app as _app
    from autopath.Clustering import Clustering, PairwiseDistances
    u = mda.Universe(PSF, DCD)
    system = mm.System()
    for _ in range(5):                      # deliberately wrong
        system.addParticle(12.0)
    top = _app.Topology()
    c = Clustering(PairwiseDistances("resid 1-5 and name CA",
                                     "resid 20-30 and name CA")).fit(u)
    with pytest.raises(ValueError, match="atom-count mismatch"):
        c.write_states(u, [0], top, system, str(tmp_path), "ms", temperature=300.0)


def test_write_states_refuses_a_boxless_frame(tmp_path):
    """Substituting the System default box would restart at the wrong density."""
    import MDAnalysis as mda
    from MDAnalysis.tests.datafiles import PSF, DCD
    from openmm import app as _app
    from autopath.Clustering import Clustering, PairwiseDistances
    u = mda.Universe(PSF, DCD)                 # no unit cell
    n = u.atoms.n_atoms
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    top = _app.Topology()
    ch = top.addChain()
    for _ in range(n):
        r = top.addResidue("UNK", ch)
        top.addAtom("C", _app.Element.getBySymbol("C"), r)
    c = Clustering(PairwiseDistances("resid 1-5 and name CA",
                                     "resid 20-30 and name CA")).fit(u)
    with pytest.raises(ValueError, match="no unit cell"):
        c.write_states(u, [0], top, system, str(tmp_path), "ms", temperature=300.0)


def test_read_positions_accepts_xml(tmp_path):
    system = _tiny_system()
    integ = mm.VerletIntegrator(0.001 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    pos = np.arange(18, dtype=float).reshape(6, 3) * 0.1
    ctx.setPositions(pos * ommunit.nanometer)
    p = tmp_path / "s.xml"
    p.write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True)))
    np.testing.assert_allclose(read_positions(str(p)), pos, rtol=0, atol=1e-6)


def test_read_positions_rejects_unknown_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="extension"):
        read_positions(str(p))


def _six_atom_universe():
    """Universe whose 'ligand' sits one box away from the 'protein'."""
    import MDAnalysis as mda
    from MDAnalysis.transformations.boxdimensions import set_dimensions
    u = mda.Universe.empty(6, n_residues=2, atom_resindex=[0, 0, 0, 1, 1, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG", "C1", "C2", "C3"])
    u.add_TopologyAttr("type", ["C"] * 6)
    u.add_TopologyAttr("element", ["C"] * 6)
    u.add_TopologyAttr("resname", ["ALA", "UNK"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["A"])
    u.atoms.positions = np.array([[5., 5., 5.], [6., 5., 5.], [5., 6., 5.],
                                  [34., 5., 5.], [35., 5., 5.], [34., 6., 5.]])
    u.trajectory.add_transformations(set_dimensions([40., 40., 40., 90., 90., 90.]))
    return u


def _six_atom_system_and_topology():
    system = mm.System()
    for _ in range(6):
        system.addParticle(12.0)
    top = app.Topology()
    ch = top.addChain()
    for _ in range(6):
        r = top.addResidue("UNK", ch)
        top.addAtom("C", app.Element.getBySymbol("C"), r)
    return system, top


def test_write_states_reimages_the_ligand_when_asked(tmp_path):
    """I1: backbone superposition can leave the ligand a box away, and the
    State inherits the frame's coordinates verbatim, so write_states needs the
    same re-imaging write_pdbs does."""
    from autopath.Clustering import Clustering, PairwiseDistances
    u = _six_atom_universe()
    system, top = _six_atom_system_and_topology()
    # write_states needs no fit; a 1-frame Universe cannot be PCA'd anyway.
    c = Clustering(PairwiseDistances("resname UNK", "resname ALA"))

    plain = c.write_states(u, [0], top, system, str(tmp_path / "a"), "ms")
    got = mm.XmlSerializer.deserialize(open(plain[0]).read()).getPositions(
        asNumpy=True).value_in_unit(ommunit.nanometer)
    assert got[3][0] == pytest.approx(3.4, abs=1e-4)      # still a box away

    u = _six_atom_universe()                               # fresh coordinates
    fixed = c.write_states(u, [0], top, system, str(tmp_path / "b"), "ms",
                           ligand_sel="resname UNK")
    got = mm.XmlSerializer.deserialize(open(fixed[0]).read()).getPositions(
        asNumpy=True).value_in_unit(ommunit.nanometer)
    assert got[3][0] == pytest.approx(-0.6, abs=1e-4)      # minimum image
    np.testing.assert_allclose(got[0], [0.5, 0.5, 0.5], atol=1e-5)  # protein untouched


def test_write_pdbs_reindexes(tmp_path):
    """M4: extract_milestones wrote with reindex=True; the replacement must
    stay output-compatible."""
    import MDAnalysis as mda
    from autopath.Clustering import Clustering, PairwiseDistances
    u = _six_atom_universe()
    c = Clustering(PairwiseDistances("resname UNK", "resname ALA"))
    out = c.write_pdbs(u, [0], str(tmp_path), "ms")
    serials = [int(l[6:11]) for l in open(out[0]) if l.startswith(("ATOM", "HETATM"))]
    assert serials == list(range(1, 7))
