#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fnmatch
from setuptools import setup, find_packages


# Path to the directory that contains this setup.py file.
base_dir = os.path.abspath(os.path.dirname(__file__))

def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(
    name="AutoPath",
    version='0.1.0',
    author="Manuel A. Llanos",
    author_email="mllanos@scripps.edu",
    url="https://github.com/forlilab/autopath",
    description='A Python package for performing path metadynamics simulations',
    long_description=open(os.path.join(base_dir, 'README.md')).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['docs']),
    package_data={
        "autopath": ["data/*.json", "data/*.q"],
    },
    scripts=['autopath/cli/run_AutoPath.py'
            ],
    zip_safe=False,
    # openff-toolkit, openff-units, pymol-open-source and ambertools are conda-only: use environment.yml
    install_requires=[
        "numpy>=1.26,<2",
        "scipy>=1.11",
        "pandas>=2.0",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "scikit-learn>=1.4",
        "statsmodels>=0.14",
        "tqdm",
        "requests",
        "openmm>=8.2,<9",
        "cvpack>=0.12,<0.13",
        "openmmtools>=0.23",
        "openmmforcefields>=0.14",
        "pdbfixer>=1.9",
        "ParmEd>=4.2",
        "MDAnalysis>=2.7",
        "mdtraj>=1.10",
        "rdkit>=2024.3",
        "prolif>=2.0",
        "GridDataFormats",
        "deeptime>=0.4.4",
        "dtaidistance>=2.3",
        "kmedoids>=0.5",
    ],
    extras_require={
        "test": ["pytest", "MDAnalysisTests"],
        "scrub": ["molscrub"],
    },
    python_requires='>=3.10',
    license="LGPL-2.1",
    keywords=["drug design", "molecular dynamics", "free energy", "medicinal chemistry"],
    classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries'
    ]
)