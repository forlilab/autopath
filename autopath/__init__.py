#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# AutoPath
#

__version__ = "0.1.0"

import importlib as _importlib


def __getattr__(name):
    """Lazily import heavy submodules on first attribute access.

    Importing OpenMM (and its CUDA/OpenCL runtime) at package-load time causes
    ``UnicodeDecodeError`` crashes in fork-based multiprocessing (used by
    ProLIF and ``multiprocess``).  By deferring the import until the name is
    first referenced, the package is safe to import in the parent process before
    ``fork()`` is called.

    Parameters
    ----------
    name : str
        Name of the attribute being accessed on the ``autopath`` module.

    Returns
    -------
    object
        The requested class or submodule.

    Raises
    ------
    AttributeError
        If *name* is not registered in the lazy-import table.
    """
    _lazy_imports = {
        "Config":               ".config",
        "Equilibration":        ".equilibration",
        "SystemPreparation":    ".preparation",
        "SteeredMD":            ".pulling",
        "MetadynamicsMD":       ".metadynamics",
        "MetadynamicsAnalysis": ".metadynamics",
        "RelaxMD":              ".relax_md",
        "VanillaMD":            ".vanilla_md",
        "CVSpec":               ".metadynamics",
        "pulling":              ".pulling",
        "metadynamics":         ".metadynamics",
    }
    if name in _lazy_imports:
        module = _importlib.import_module(_lazy_imports[name], __name__)
        if name in ("pulling", "metadynamics"):
            return module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Config",
    "Equilibration",
    "SystemPreparation",
    "SteeredMD",
    "pulling",
    "metadynamics",
    "MetadynamicsMD",
    "MetadynamicsAnalysis",
    "RelaxMD",
    "VanillaMD",
    "CVSpec",
]
