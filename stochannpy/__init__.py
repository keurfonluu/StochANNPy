# -*- coding: utf-8 -*-

"""
StochANNPy (STOCHAstic Artificial Neural Network for PYthon) provides
user-friendly routines compatible with Scikit-Learn for stochastic learning.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from .bayesian_neural_network import BNNClassifier
from .evolutionary_neural_network import ENNClassifier


__all__ = [ "BNNClassifier", "ENNClassifier" ]
__version__ = "0.0.1b2"