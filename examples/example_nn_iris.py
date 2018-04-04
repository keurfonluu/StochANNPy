# -*- coding: utf-8 -*-

"""
This example shows how to use BNNClassifier and ENNClassifier on IRIS dataset.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
try:
    from stochannpy import BNNClassifier, ENNClassifier
except ImportError:
    import sys
    sys.path.append("../")
    from stochannpy import BNNClassifier, ENNClassifier

if __name__ == "__main__":
    # Load IRIS dataset
    iris = load_iris()
    
    # Get descriptors and target to predict
    X, y = iris.data, iris.target
    
    # Split data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    
    # Predict using ENNClassifier
    clf = ENNClassifier(hidden_layer_sizes = (5,),
                        activation = "relu",
                        max_iter = 500,
                        bounds = 1.,
                        alpha = 0.,
                        solver = "cpso",
                        popsize = 30)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print("Test Set Accuracy (ENNClassifier): %.2f" \
          % (np.mean(ypred == y_test) * 100))
    
    # Predict using BNNClassifier
    clf = BNNClassifier(hidden_layer_sizes = (5,),
                        activation = "relu",
                        max_iter = 5000,
                        alpha = 0.,
                        sampler = "hmc",
                        stepsize = 0.1,
                        n_leap = 10)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print("Test Set Accuracy (BNNClassifier): %.2f" \
          % (np.mean(ypred == y_test) * 100))
    
    # Plot sampled weights
    clf.plot_coefs(n_burnin = 100, ignore_bias = False)