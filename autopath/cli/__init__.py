"""Command-line interface (CLI) for the AutoPath package.

This package contains the ``run_AutoPath`` script, which reads a JSON
configuration file and one or more ligand SDF files and runs the full AutoPath
MD simulation pipeline (preparation, equilibration, steered MD, milestone
extraction, and metadynamics).

The script is installed directly (see ``setup.py``) and can be invoked as::

    run_AutoPath.py -c config.json -l ligand.sdf

The primary entry point is :func:`autopath.cli.run_AutoPath.main`.
"""
