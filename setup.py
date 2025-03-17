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
    scripts=['autopath/cli/run_AutoPath.py'
            ],
    zip_safe=False,
    install_requires=[''],
    python_requires='>=3.5',
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