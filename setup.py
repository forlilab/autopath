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
    packages=find_packages(),
    scripts=[
            ],
    zip_safe=False,
    install_requires=['openmm'],
    python_requires='>=3.5',
    license="LGPL-2.1",
    keywords=["drug design"],
    classifiers=[
        'Environment :: Console',
        'Environment :: Other Environment',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        #'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries'
    ]
)