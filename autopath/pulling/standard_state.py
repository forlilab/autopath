"""Standard-state binding free energy from a pulling PMF."""
import numpy as np

from autopath.metadynamics.Analysis import limongelli_dG

_KB = 8.314e-3  # kJ/mol/K


def dG_standard_from_profile(
    coord_nm: np.ndarray,
    G_kJ: np.ndarray,
    site_max_nm: float,
    bulk_range_nm: tuple,
    R_cyl_ang: float,
    temperature: float = 300.0,
    C0_per_A3: float = 1.0 / 1660.0,
) -> dict:
    """Eq. 15 standard-state ΔG° from a 1D PMF along the pull coordinate.

    Parameters
    ----------
    coord_nm : np.ndarray
        Uniform coordinate grid in nm.
    G_kJ : np.ndarray
        PMF in kJ/mol.
    site_max_nm : float
        Points with ``coord_nm <= site_max_nm`` form the bound site.
    bulk_range_nm : tuple
        (lo, hi) window in nm whose mean defines the unbound zero.
    R_cyl_ang : float
        Cylinder radius in Å.
    temperature : float
        Kelvin.
    C0_per_A3 : float
        Standard concentration in Å⁻³.

    Returns
    -------
    dict
        ``limongelli_dG`` outputs plus ``bulk_ref_kJ`` and
        ``bulk_slope_kJ_per_nm`` (linear fit over the bulk window).
    """
    z = np.asarray(coord_nm, float)
    G = np.asarray(G_kJ, float)
    if z.ndim != 1 or z.shape != G.shape or len(z) < 2:
        raise ValueError("not supported: coord and G must be equal-length 1D arrays")
    dz = np.diff(z)
    if dz[0] <= 0 or not np.allclose(dz, dz[0], rtol=1e-6, atol=0.0):
        raise ValueError("not supported: coordinate grid must be uniform and increasing")
    bulk = (z >= bulk_range_nm[0]) & (z <= bulk_range_nm[1])
    if bulk.sum() < 3:
        raise ValueError("not supported: fewer than 3 points in the bulk range")
    if not np.all(np.isfinite(G[bulk])):
        raise ValueError("not supported: non-finite PMF in the bulk range")
    site = z <= site_max_nm
    if site.sum() < 2 or not np.all(np.isfinite(G[site])):
        raise ValueError("not supported: site needs >= 2 points and no NaN")

    ref = float(G[bulk].mean())
    slope = float(np.polyfit(z[bulk], G[bulk], 1)[0])
    kT = _KB * temperature
    res = limongelli_dG(G[site] - ref, float(dz[0]) * 10.0, R_cyl_ang, kT, C0_per_A3)
    res["bulk_ref_kJ"] = ref
    res["bulk_slope_kJ_per_nm"] = slope
    return res
