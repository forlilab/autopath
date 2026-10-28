"""Periodic-lattice helpers for pulls in triclinic boxes."""
import numpy as np


def triclinic_minimum_image(d, box_nm) -> np.ndarray:
    """Periodic image of ``d`` as OpenMM reduces it: by box vector c, then b, then a.

    Parameters
    ----------
    d : array_like, shape (3,) or (M, 3)
        Displacement(s) (nm).
    box_nm : array_like, shape (3, 3)
        Box vectors a, b, c as rows (nm), in OpenMM's reduced form.

    Returns
    -------
    numpy.ndarray
        Same shape as ``d``.
    """
    box = np.asarray(box_nm, dtype=float)
    d = np.array(d, dtype=float)
    for k in (2, 1, 0):
        d = d - np.round(d[..., k] / box[k, k])[..., None] * box[k]
    return d


def reduction_identity_limit(n, box_nm):
    """Largest distance along ``n`` that :func:`triclinic_minimum_image` returns unchanged.

    The c, b, a reduction is the identity only inside the brick |d_i| < box[i, i] / 2, so the
    limit is min_i box[i, i] / (2 |n_i|).

    Parameters
    ----------
    n : array_like, shape (3,)
        Unit direction.
    box_nm : array_like, shape (3, 3)
        Box vectors a, b, c as rows (nm), in OpenMM's reduced form.

    Returns
    -------
    (float, int)
        The limit (nm) and the limiting axis (0, 1, 2).
    """
    n = np.abs(np.asarray(n, dtype=float))
    half = np.diag(np.asarray(box_nm, dtype=float)) / 2.0
    with np.errstate(divide="ignore"):
        per = np.where(n > 1e-12, half / n, np.inf)
    axis = int(np.argmin(per))
    return float(per[axis]), axis
