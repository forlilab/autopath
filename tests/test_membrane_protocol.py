"""Shape of the 10 ns membrane equilibration protocol.

The candidate protocol splits lipids into ``lipid_head`` and ``lipid_tail``. The
bilayer patch OpenMM builds is already equilibrated and minimized, so the thing that
needs to relax is the protein-lipid annulus left behind when ``addMembrane`` deletes
overlapping lipids and re-expands the solute. Tails are therefore freed early while
headgroups hold leaflet registry, and the ligand is released last so the pocket
adapts around the pose.
"""

import json
import os

import pytest

from autopath.customForces import _FG_COMPONENTS, _FG_COMPONENTS_LAST
from autopath.equilibration import Equilibration

MEMB_10NS = "autopath/data/eq_lig-prot-memb_10ns_4fs.json"
SOLUBLE_5NS = "autopath/data/eq_lig-prot_5ns_4fs.json"
ALL_PROTOCOLS = [
    SOLUBLE_5NS,
    MEMB_10NS,
]

# Real membrane system used as the reference for the old protocol.
REF_PRMTOP = "/gpfs/group/forli/mllanos/sodio/Nav1.6/system.prmtop"


def _equil(tmp_path, protocol):
    return Equilibration(out_dir=str(tmp_path), protocol_fname=protocol, platform="CPU")


def _ladder(raw, component, section):
    i = list(raw["components_lookup"]).index(component)
    return [stage["forces"][i] for stage in raw[section]]


def _load(protocol):
    with open(protocol) as f:
        return json.load(f)


def _selection(raw, name):
    value = raw["components_lookup"][name]
    return value["selection"] if isinstance(value, dict) else value


@pytest.mark.parametrize(
    "protocol,nanoseconds", [(SOLUBLE_5NS, 5.0), (MEMB_10NS, 10.0)]
)
def test_total_time(protocol, nanoseconds, tmp_path):
    equil = _equil(tmp_path, protocol)
    assert equil.simulation_time == pytest.approx(nanoseconds * 1000.0)


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
@pytest.mark.parametrize("section", ["minimization", "equilibration"])
def test_ladders_never_ramp_up(protocol, section):
    raw = _load(protocol)
    for component in raw["components_lookup"]:
        ladder = _ladder(raw, component, section)
        assert ladder == sorted(ladder, reverse=True), f"{component} {section} {ladder}"


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_ligand_is_released_no_earlier_than_anything_else(protocol):
    """Induced fit: the pocket must relax while the ligand is still held."""
    raw = _load(protocol)

    def first_zero(component):
        ladder = _ladder(raw, component, "equilibration")
        return next((i for i, k in enumerate(ladder) if k == 0), len(ladder))

    ligand = first_zero("ligand")
    for component in raw["components_lookup"]:
        if component == "ligand":
            continue
        assert first_zero(component) <= ligand, component


def test_tails_freed_after_the_first_minimization_stage():
    ladder = _ladder(_load(MEMB_10NS), "lipid_tail", "minimization")
    assert ladder[0] > 0 and all(k == 0 for k in ladder[1:]), ladder


def test_heads_outlive_tails_but_not_the_ligand():
    raw = _load(MEMB_10NS)
    head = _ladder(raw, "lipid_head", "equilibration")
    tail = _ladder(raw, "lipid_tail", "equilibration")
    ligand = _ladder(raw, "ligand", "equilibration")
    assert sum(k > 0 for k in head) > sum(k > 0 for k in tail)
    assert sum(k > 0 for k in ligand) > sum(k > 0 for k in head)


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_components_fit_the_reserved_force_groups(protocol, tmp_path):
    equil = _equil(tmp_path, protocol)
    last = _FG_COMPONENTS + len(equil.components_lookup) - 1
    assert last <= _FG_COMPONENTS_LAST


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(REF_PRMTOP), reason="reference system offline")
def test_lipid_selections_partition_the_bilayer():
    """head + tail must cover every lipid heavy atom exactly once."""
    mda = pytest.importorskip("MDAnalysis")
    raw = _load(MEMB_10NS)
    u = mda.Universe(REF_PRMTOP)

    head = u.select_atoms(_selection(raw, "lipid_head"))
    tail = u.select_atoms(_selection(raw, "lipid_tail"))
    every = u.select_atoms("resname POP and not name H*")

    assert len(head) > 0 and len(tail) > 0
    assert len(head.intersection(tail)) == 0
    assert len(head) + len(tail) == len(every)
    assert len(tail) > len(head)  # tails are the bulk of the bilayer


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(REF_PRMTOP), reason="reference system offline")
def test_every_component_selects_atoms_on_the_reference_system():
    """An empty selection would silently create a restraint with no particles."""
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe(REF_PRMTOP)
    raw = _load(MEMB_10NS)
    for name in raw["components_lookup"]:
        assert len(u.select_atoms(_selection(raw, name))) > 0, name


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_ligand_is_never_stiffer_than_the_backbone(protocol):
    """Both are absolute lab-frame restraints; a stiffer ligand makes the pocket
    walls breathe more than the ligand they enclose."""
    raw = _load(protocol)
    backbone = _ladder(raw, "protein_BB", "equilibration")
    ligand = _ladder(raw, "ligand", "equilibration")
    for stage, (bb, lig) in enumerate(zip(backbone, ligand), start=1):
        assert lig <= bb, f"Stage {stage}: ligand {lig} > backbone {bb}"

    warm = raw["warmup"]["forces"]
    comps = list(raw["components_lookup"])
    assert warm[comps.index("ligand")] <= warm[comps.index("protein_BB")]


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_ligand_is_held_through_minimization(protocol):
    """Minimization must not free the ligand before the pocket it sits in."""
    assert all(k > 0 for k in _ladder(_load(protocol), "ligand", "minimization"))


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_shipped_protocols_document_their_scope(protocol):
    assert _load(protocol).get("_comment", "").strip(), protocol


def _lipid21_templates():
    import xml.etree.ElementTree as ET
    import openmm.app
    path = os.path.join(os.path.dirname(openmm.app.__file__),
                        "data", "amber19", "lipid21.xml")
    if not os.path.exists(path):
        return None
    root = ET.parse(path).getroot()
    return {r.get("name"): [a.get("name") for a in r.findall("Atom")]
            for r in root.iter("Residue")}


def _template_universe(residues):
    """One residue per (resname, atom names) pair, no coordinates."""
    mda = pytest.importorskip("MDAnalysis")
    names, resindex, resnames = [], [], []
    for i, (resname, atoms) in enumerate(residues):
        names += atoms
        resindex += [i] * len(atoms)
        resnames.append(resname)
    u = mda.Universe.empty(len(names), n_residues=len(residues),
                           atom_resindex=resindex, trajectory=False)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", list(range(1, len(residues) + 1)))
    return u


@pytest.fixture(scope="module")
def lipid_universe():
    """Every lipid21 template, plus POPC under the truncated "POP" label."""
    templates = _lipid21_templates()
    if templates is None:
        pytest.skip("lipid21.xml not available")
    residues = sorted(templates.items()) + [("POP", templates["POPC"])]
    return _template_universe(residues)


def _split(u, raw, resname):
    res = u.select_atoms(f"resname {resname}")
    heavy = {a.name for a in res if not a.name.startswith("H")}
    head = {a.name for a in u.select_atoms(_selection(raw, "lipid_head")) if a.resname == resname}
    tail = {a.name for a in u.select_atoms(_selection(raw, "lipid_tail")) if a.resname == resname}
    return heavy, head, tail


@pytest.mark.parametrize("resname", ["POP", "POPC", "POPE", "POPS", "DOPC", "DOPE", "DPPC", "DMPC"])
def test_phospholipid_heads_and_tails_partition_the_template(lipid_universe, resname):
    """Lipid21 uses CHARMM-style atom names, so the split is checked against the
    force field's own templates. Any polar atom missed by the head list would be
    restrained as a tail, i.e. freed early."""
    import re

    heavy, head, tail = _split(lipid_universe, _load(MEMB_10NS), resname)
    assert head and tail
    assert not head & tail
    assert head | tail == heavy
    acyl = re.compile(r"^C[23]\d+$")          # sn-1 / sn-2 chain carbons
    assert all(acyl.match(a) for a in tail), sorted(a for a in tail if not acyl.match(a))


def test_cholesterol_is_anchored_by_its_hydroxyl_only(lipid_universe):
    """CHARMM sterol names reuse C1-C3/C11-C15/C21; only O3 may be z-restrained and
    the sterol body stays free like the acyl tails."""
    heavy, head, tail = _split(lipid_universe, _load(MEMB_10NS), "CHL1")
    assert head == {"O3"}
    assert not tail
    assert len(heavy) == 28
