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


def _lipid_resnames_and_head(raw):
    """Pull the resname list and head atom-name list out of the lipid_head selection."""
    parts = [p.strip() for p in _selection(raw, "lipid_head").split(" and ")]
    resnames = next(p[len("resname "):].split() for p in parts if p.startswith("resname "))
    head = next(p[len("name "):].split() for p in parts if p.startswith("name "))
    return resnames, head


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


def test_head_names_cover_every_non_acyl_atom_in_lipid21():
    """Anything not captured by the head list is restrained as a tail, i.e. freed early.

    Lipid21 uses CHARMM-style atom names, so the head list is checked against the
    force field's own templates rather than against a naming convention.
    """
    import re

    templates = _lipid21_templates()
    if templates is None:
        pytest.skip("lipid21.xml not available")

    resnames, head = _lipid_resnames_and_head(_load(MEMB_10NS))
    acyl = re.compile(r"^C[23]\d+$")          # sn-1 / sn-2 chain carbons
    checked = 0
    for resname in resnames:
        atoms = templates.get(resname)
        if atoms is None:                      # e.g. the truncated "POP" label
            continue
        checked += 1
        heavy = [a for a in atoms if not a.startswith("H")]
        missed = [a for a in heavy if a not in head and not acyl.match(a)]
        assert not missed, f"{resname}: {missed} would be restrained as tail, not head"
    assert checked > 0
