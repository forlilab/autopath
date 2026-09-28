import numpy as np, pytest
import openmm as mm
from openmm import unit as ommunit, app
from autopath.Clustering import read_positions


def test_pdb_and_xml_of_same_frame_give_same_coms(tmp_path):
    """A nm/Angstrom slip changes the CV silently; compare the derived COMs."""
    n = 8
    pos_nm = np.linspace(0.0, 1.4, n * 3).reshape(n, 3)

    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    integ = mm.VerletIntegrator(0.001 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm * ommunit.nanometer)
    xml = tmp_path / "m.xml"
    xml.write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True)))

    top = app.Topology()
    ch = top.addChain()
    for i in range(n):
        r = top.addResidue("UNK", ch)
        top.addAtom("C", app.Element.getBySymbol("C"), r)
    pdb = tmp_path / "m.pdb"
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(top, pos_nm * ommunit.nanometer, fh)

    a, b = read_positions(str(xml)), read_positions(str(pdb))
    grp = [0, 1, 2, 3]
    com_a = a[grp].mean(axis=0)
    com_b = b[grp].mean(axis=0)
    np.testing.assert_allclose(com_a, com_b, atol=2e-3)   # PDB is 3-decimal
