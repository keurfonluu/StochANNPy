# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
import stochannpy

DISTNAME = "stochannpy"
DESCRIPTION = "StochANNPy"
LONG_DESCRIPTION = """StochANNPy (STOCHAstic Artificial Neural Network for PYthon) provides user-friendly routines compatible with Scikit-Learn for stochastic learning."""
AUTHOR = "Keurfon LUU"
AUTHOR_EMAIL = "keurfon.luu@mines-paristech.fr"
URL = "https://github.com/keurfonluu/stochannpy"
LICENSE = "MIT License"
REQUIREMENTS = [
    "numpy >= 1.9.0",
    "stochopy >= 1.5.2",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "joblib",
]
CLASSIFIERS = [
    "Programming Language :: Python",
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
]
 
if __name__ == "__main__":
    setup(
        name = DISTNAME,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        url = URL,
        license = LICENSE,
        install_requires = REQUIREMENTS,
        classifiers = CLASSIFIERS,
        version = stochannpy.__version__,
        packages = find_packages(),
        include_package_data = True,
    )