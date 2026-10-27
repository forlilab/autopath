"""Small rigid-protein + two-atom-ligand toy for projection-mode SteeredMD tests."""
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from autopath.metadynamics.Funnel import frame_basis

POCKET, LIGAND, FRAME_B, FRAME_C = [0, 1], [2, 3], [4, 5], [6, 7]
MASSES = np.array([12.0, 14.0, 16.0, 12.0, 12.0, 15.0, 13.0, 14.0])
AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
BOX_NM = 6.0
LIGAND_BOND_NM = 0.15
TETHER_K = 50.0   # kJ/mol/nm^2: keeps the free ligand near the pocket, weak next to the pull spring


def com(pos, idx):
    return np.average(pos[idx], axis=0, weights=MASSES[idx])


def positions():
    """Input positions (nm): ligand COM 0.4 nm up the axis from the pocket COM, slightly off-axis."""
    p = np.array([[3.0, 3.0, 3.0], [3.1, 2.9, 3.05],      # pocket
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],        # ligand, placed below
                  [4.0, 3.2, 2.9], [4.1, 3.3, 3.0],        # frame b
                  [3.1, 4.1, 3.2], [3.0, 4.0, 3.3]])       # frame c
    half = 0.5 * LIGAND_BOND_NM * np.array([1.0, 0.0, 0.0])
    centre = com(p, POCKET) + 0.4 * AXIS + np.array([0.02, 0.01, -0.01])
    w = MASSES[LIGAND] / MASSES[LIGAND].sum()
    p[2] = centre + 2 * w[1] * half
    p[3] = centre - 2 * w[0] * half
    return p


def params(p=None, R_cylinder_ang=2.0, z_cc_ang=6.0, alpha_deg=45.0, k_xy=10.0):
    """Protein-frame funnel params on the pull groups, with the axis = AXIS at ``p``."""
    p = positions() if p is None else p
    coeffs = frame_basis(com(p, POCKET), com(p, FRAME_B), com(p, FRAME_C)) @ AXIS
    return {"unbinding_axis": AXIS, "host_index": POCKET, "guest_index": LIGAND,
            "frame_b_index": FRAME_B, "frame_c_index": FRAME_C, "axis_coeffs": coeffs,
            "k_xy": k_xy * u.kilocalorie_per_mole / u.angstrom**2, "z_cc": z_cc_ang * u.angstrom,
            "z_max": 12.0 * u.angstrom, "alpha": alpha_deg * u.degrees,
            "R_cylinder": R_cylinder_ang * u.angstrom}


def z_input():
    return float(AXIS @ (com(positions(), LIGAND) - com(positions(), POCKET)))


def smd_system(tmp_path, p=None):
    """(System, Topology, pdb path): stiff protein bonds, one ligand bond, a weak ligand-pocket COM tether.

    All bonds use periodic images, so ``p`` may place particles one box vector away.
    """
    p = positions() if p is None else p
    s = mm.System()
    box = [mm.Vec3(*(BOX_NM * np.eye(3)[i])) for i in range(3)]
    s.setDefaultPeriodicBoxVectors(*box)
    for m in MASSES:
        s.addParticle(float(m))
    bonds = mm.HarmonicBondForce()
    protein = POCKET + FRAME_B + FRAME_C
    for i in protein:
        for j in protein:
            if i < j:
                bonds.addBond(i, j, float(np.linalg.norm(p[i] - p[j])), 1.0e4)
    bonds.addBond(*LIGAND, LIGAND_BOND_NM, 1.0e4)
    bonds.setUsesPeriodicBoundaryConditions(True)
    s.addForce(bonds)
    tether = mm.CustomCentroidBondForce(2, "0.5*k_tether*distance(g1,g2)^2")
    tether.addGlobalParameter("k_tether", TETHER_K)
    tether.addGroup(POCKET, [float(m) for m in MASSES[POCKET]])
    tether.addGroup(LIGAND, [float(m) for m in MASSES[LIGAND]])
    tether.addBond([0, 1], [])
    tether.setUsesPeriodicBoundaryConditions(True)
    s.addForce(tether)

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("PRT", chain)
    for i in range(len(MASSES)):
        if i in LIGAND and i == LIGAND[0]:
            res = top.addResidue("LIG", chain)
        elif i == FRAME_B[0]:
            res = top.addResidue("PRT", chain)
        top.addAtom(f"C{i}", app.element.carbon, res)
    top.setPeriodicBoxVectors(box)
    pdb = tmp_path / "start.pdb"
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(top, p * u.nanometers, fh)
    return s, top, str(pdb)
