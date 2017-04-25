**********
StochANNPy
**********

StochANNPy (STOCHAstic Artificial Neural Network for PYthon) provides
user-friendly routines compatible with Scikit-Learn for stochastic learning.

:Version: 0.0.1b1
:Author: Keurfon Luu
:Web site: https://github.com/keurfonluu/stochannpy
:Copyright: This document has been placed in the public domain.
:License: StochOPy is released under the MIT License.

**NOTE**: StochANNPy has been implemented in the frame of my Ph. D. thesis. If
you find any error or bug, or if you have any suggestion, please don't hesitate
to contact me.


Features
========

StochANNPy provides routines for stochastic neural networks:

* Bayesian neural networks [1]
* Evolutionary neural networks

**NOTE**: for now, only classifier has been implemented. Regressor will come
later.


Installation
============

The recommended way to install StochANNPy is through pip:

.. code-block:: bash

    pip install stochannpy
    
Otherwise, download and extract the package, then run:

.. code-block:: bash

    python setup.py install
    

Usage
=====

First, import StochANNPy and initialize the classifier:

.. code-block:: python

    import numpy as np
    from stochannpy import BNNClassifier
    
    clf = BNNClassifier(hidden_layer_sizes = (5,))
    
Fit the training set:

.. code-block:: python

    clf.fit(X_train, y_train)
    
Predict the test set:

.. code-block:: python

    ypred = clf.predict(X_test)
    
Compute the accuracy:

.. code-block:: python

    print(np.mean(ypred == y_test))


References
==========
.. [1] N. Radford, *Bayesian Learning for Neural Networks*, Lecture Notes in
   Statistics, Springer, 1996
