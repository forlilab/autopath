"""4-site water on the membrane path.

``Modeller.addMembrane`` takes no water-model argument and parametrizes its bundled
3-site lipid patch, and the solute, with whatever force field it is handed, so a 4-site
force field aborts the build before any virtual site could be added. ``SystemPreparation``
therefore places the membrane with a derived 3-site force field and promotes every water
afterwards, leaving ``createSystem`` on the real force field.
"""

import numpy as np
import pytest
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import Modeller, PDBFile, Simulation
from openmm.app.element import hydrogen
import openmm.unit as unit
from pdbfixer import PDBFixer

from autopath.preparation import SystemPreparation, substitute_three_site_water

FF_4SITE = ["amber19-all.xml", "amber19/opc.xml"]
FF_3SITE = ["amber19-all.xml", "amber19/tip3pfb.xml"]
SOURCE_PDB = "examples/data/3ptb_fixed.pdb"


class _Stop(Exception):
    """Abort run() once the ordering under test has been observed."""


@pytest.fixture(scope="module")
def peptide(tmp_path_factory):
    """A ~60-residue peptide centred on the origin, plus a few crystal waters.

    addMembrane centres the bilayer on z=0 but never translates the solute, so an
    off-centre solute lands outside the box it sizes and the packing run diverges.
    """
    out = tmp_path_factory.mktemp("peptide")
    src = PDBFile(SOURCE_PDB)
    modeller = Modeller(src.topology, src.positions)
    residues = list(modeller.topology.residues())
    protein = [r for r in residues if r.name != "HOH"]
    waters = [r for r in residues if r.name == "HOH"]
    keep = set(protein[:60]) | set(waters[:5])
    modeller.delete([r for r in residues if r not in keep])
    modeller.delete([a for a in modeller.topology.atoms() if a.element is hydrogen])
    sliced = out / "sliced.pdb"
    with open(sliced, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    fixer = PDBFixer(filename=str(sliced))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    xyz = np.array(fixer.positions.value_in_unit(unit.nanometer))
    xyz -= (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    fixer.topology.setPeriodicBoxVectors(None)
    path = out / "peptide.pdb"
    with open(path, "w") as f:
        PDBFile.writeFile(fixer.topology, xyz * unit.nanometer, f)
    return str(path)


def _prep(tmp_path, forcefield, water_model, is_membrane):
    return SystemPreparation(
        forcefield=forcefield,
        water_model=water_model,
        is_membrane=is_membrane,
        lipid_type="POPC",
        padding=1.0,
        hydrogenMass=4.0,
        out_dir=str(tmp_path),
    )


def test_only_the_water_xml_is_substituted():
    files, swapped = substitute_three_site_water(
        FF_4SITE + ["amber/tip3pfb_HFE_multivalent.xml"]
    )
    assert files == ["amber19-all.xml", "amber19/opc3.xml",
                     "amber/tip3pfb_HFE_multivalent.xml"]
    assert swapped == [("amber19/opc.xml", "amber19/opc3.xml")]


def test_substitution_keeps_the_release_directory():
    files, _ = substitute_three_site_water(["amber14-all.xml", "amber14/tip4pew.xml"])
    assert files == ["amber14-all.xml", "amber14/tip3p.xml"]


def test_three_site_files_are_left_alone():
    assert substitute_three_site_water(FF_3SITE) == (FF_3SITE, [])


def test_derived_forcefield_is_a_separate_object(tmp_path):
    prep = _prep(tmp_path, FF_4SITE, "tip4pew", True)
    assert prep._three_site_forcefield() is not prep.forcefield


def test_missing_substitution_raises(tmp_path):
    prep = _prep(tmp_path, FF_3SITE, "tip4pew", True)
    with pytest.raises(ValueError, match="extra-site water XML"):
        prep._three_site_forcefield()


def test_ligand_template_generators_carry_over(tmp_path):
    prep = _prep(tmp_path, FF_4SITE, "tip4pew", True)

    def generator(forcefield, residue):
        return False

    prep._template_generators.append(generator)
    assert generator in prep._three_site_forcefield()._templateGenerators


def test_solvated_path_promotes_water_before_solvation(monkeypatch, tmp_path, peptide):
    calls = []
    monkeypatch.setattr(Modeller, "addExtraParticles",
                        lambda self, forcefield, **kw: calls.append("extra"))

    def _solvent(self, *args, **kwargs):
        calls.append("solvent")
        raise _Stop

    monkeypatch.setattr(Modeller, "addSolvent", _solvent)
    with pytest.raises(_Stop):
        _prep(tmp_path, FF_4SITE, "tip4pew", False).run(protein=peptide)
    assert calls == ["extra", "solvent"]


def test_membrane_path_promotes_water_after_the_membrane(monkeypatch, tmp_path, peptide):
    calls = []
    prep = _prep(tmp_path, FF_4SITE, "tip4pew", True)

    def _membrane(self, forcefield, **kwargs):
        calls.append(("membrane", forcefield))

    def _extra(self, forcefield, **kwargs):
        calls.append(("extra", forcefield))
        raise _Stop

    monkeypatch.setattr(Modeller, "addMembrane", _membrane)
    monkeypatch.setattr(Modeller, "addExtraParticles", _extra)
    with pytest.raises(_Stop):
        prep.run(protein=peptide)

    assert [name for name, _ in calls] == ["membrane", "extra"]
    assert calls[0][1] is not prep.forcefield
    assert calls[1][1] is prep.forcefield


def test_three_site_water_needs_no_promotion(monkeypatch, tmp_path, peptide):
    calls = []
    monkeypatch.setattr(Modeller, "addExtraParticles",
                        lambda self, forcefield, **kw: calls.append("extra"))

    def _membrane(self, forcefield, **kwargs):
        calls.append(("membrane", forcefield))
        raise _Stop

    monkeypatch.setattr(Modeller, "addMembrane", _membrane)
    prep = _prep(tmp_path, FF_3SITE, "tip3p", True)
    with pytest.raises(_Stop):
        prep.run(protein=peptide)

    assert calls == [("membrane", prep.forcefield)]


@pytest.mark.slow
def test_membrane_build_gives_one_virtual_site_per_water(monkeypatch, tmp_path, peptide):
    counted = {}
    add_extra_particles = Modeller.addExtraParticles

    def _extra(self, forcefield, **kwargs):
        counted["before"] = self.topology.getNumAtoms()
        counted["waters"] = sum(1 for r in self.topology.residues() if r.name == "HOH")
        add_extra_particles(self, forcefield, **kwargs)
        counted["after"] = self.topology.getNumAtoms()

    monkeypatch.setattr(Modeller, "addExtraParticles", _extra)
    prep = _prep(tmp_path, FF_4SITE, "tip4pew", True)
    system, topology = prep.run(protein=peptide)

    assert counted["waters"] > 0
    assert counted["after"] - counted["before"] == counted["waters"]

    waters = [r for r in topology.residues() if r.name == "HOH"]
    assert len(waters) == counted["waters"]
    assert all(len(list(r.atoms())) == 4 for r in waters)

    virtual_sites = [i for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
    assert len(virtual_sites) == len(waters)
    assert system.getNumParticles() == topology.getNumAtoms()

    pdb = PDBFile(f"{tmp_path}/system.pdb")
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                          0.004 * unit.picoseconds)
    simulation = Simulation(topology, system, integrator, Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=200)
    forces = np.array(simulation.context.getState(getForces=True).getForces(asNumpy=True)
                      .value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    assert not np.isnan(forces).any()
