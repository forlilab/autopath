"""CV names must be settable, and duplicates must be refused.

Per-walker CV values are stored in a dict keyed by name, so two CVs sharing one collapse into
a single entry; the mismatch then surfaces far away, as a shape error inside the histogram.
Most CV constructors default to a fixed name, so biasing two of the same kind collides unless
the caller names them.
"""
import numpy as np
import pytest

from autopath.metadynamics.CV import com_cv, rmsd_cv
from autopath.metadynamics.summary.data import _names


def test_com_cv_name_is_settable_and_defaults():
    assert com_cv([0, 1], [2, 3], 0.0, 2.0).name == "com"
    assert com_cv([0, 1], [2, 3], 0.0, 2.0, name="exit_dist").name == "exit_dist"


def test_rmsd_cv_name_is_settable_and_defaults():
    assert rmsd_cv([0, 1, 2], 0.0, 1.0).name == "rmsd"
    assert rmsd_cv([0, 1, 2], 0.0, 1.0, name="rmsd_siteB").name == "rmsd_siteB"


def test_two_of_the_same_cv_can_be_distinguished():
    """The case that collides in practice: two CVs from one constructor."""
    a = com_cv([0], [1], 0.0, 2.0, name="d_siteA")
    b = com_cv([0], [2], 0.0, 2.0, name="d_siteB")
    assert a.name != b.name


def test_loader_rejects_duplicate_cv_names():
    with pytest.raises(ValueError, match="duplicate CV names"):
        _names({"cv_names": np.array(["nc", "nc"])})


def test_loader_accepts_distinct_cv_names():
    assert _names({"cv_names": np.array(["react", "anchor"])}) == ["react", "anchor"]
