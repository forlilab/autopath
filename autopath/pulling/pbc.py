"""Periodic-lattice helpers for pulls in triclinic boxes."""
import itertools

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


def lattice_pbc_limit(n, box_nm):
    """Largest distance along ``n`` before a displacement leaves its periodic cell.

    min over the lattice vectors T = i·a + j·b + k·c (i, j, k in {-1, 0, 1}, not all 0, n·T ≠ 0)
    of |T|² / (2 |n·T|).

    Parameters
    ----------
    n : array_like, shape (3,)
        Unit direction.
    box_nm : array_like, shape (3, 3)
        Box vectors as rows (nm).

    Returns
    -------
    (float, tuple of int)
        The limit (nm) and the (i, j, k) of the limiting T.
    """
    n = np.asarray(n, dtype=float)
    box = np.asarray(box_nm, dtype=float)
    best, best_ijk = np.inf, None
    for ijk in itertools.product((-1, 0, 1), repeat=3):
        if ijk == (0, 0, 0):
            continue
        T = np.asarray(ijk, dtype=float) @ box
        nT = abs(float(n @ T))
        if nT < 1e-12:
            continue
        val = float(T @ T) / (2.0 * nT)
        if val < best:
            best, best_ijk = val, ijk
    return best, best_ijk
