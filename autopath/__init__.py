#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# AutoPath
#

__version__ = "0.1.0"

import importlib as _importlib

# Lazy imports to avoid pulling in OpenMM (and its CUDA/OpenCL contexts)
# at package-load time.  This prevents fork-based multiprocessing
# (used by ProLIF / multiprocess) from breaking with UnicodeDecodeError.

def __getattr__(name):
    _lazy_imports = {
        "Equilibration":   ".equilibration",
        "SystemPreparation": ".preparation",
        "SteeredMD":       ".steered_md",
        "MetadynamicsMD":  ".metadynamics",
        "RelaxMD":         ".relax_md",
        "VanillaMD":       ".vanilla_md",
        "CVSpec":          ".cv",
        "sMDAnalysis":     ".sMDAnalysis",
    }
    if name in _lazy_imports:
        module = _importlib.import_module(_lazy_imports[name], __name__)
        if name == "sMDAnalysis":
            return module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Equilibration",
    "SystemPreparation",
    "SteeredMD",
    "sMDAnalysis",
    "MetadynamicsMD",
    "RelaxMD",
    "VanillaMD",
    "CVSpec",
]
