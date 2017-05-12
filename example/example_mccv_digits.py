# -*- coding: utf-8 -*-

"""
This example shows how to boost a classifier performance using MCCVClassifier.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    # Import MCCVClassifier
    import sys
    sys.path.append("../")
    from stochannpy.mccv import MCCVClassifier
    
    # Load digits dataset
    digits = load_digits()
    
    # Get descriptors and target to predict
    X, y = digits.data, digits.target
    
    # Get all possible classes
    classes_list = np.unique(y).astype(int)
    
    # Display mean digits
    fig = plt.figure(figsize = (10, 5), facecolor = "white")
    fig.patch.set_alpha(0.)
    for idx in classes_list:
        ax = fig.add_subplot(2, 5, idx+1)
        img_mean = X[y == idx,:].mean(axis = 0)
        img = img_mean.reshape((8, 8))
        ax.imshow(img, cmap = "gray")
    fig.tight_layout()
        
    # Split data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    
    # Predict using MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes = (25,), alpha = 10, max_iter = 100,
                        activation = "relu", solver = "lbfgs")
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print("Test Set Accuracy (MLPClassifier): %.2f" \
          % (np.mean(ypred == y_test) * 100))
    
    # Boosting MLPClassifier using MCCVClassifier
    # Note that training is performed on the whole training set
    clf = MCCVClassifier(clf, n_split = 12, test_size = 0.5, scoring = "accuracy",
                         n_jobs = 4)
    clf.fit(X, y)
    ypred = clf.predict(X_test)
    print("Test Set Accuracy (MCCVlassifier): %.2f" \
          % (np.mean(ypred == y_test) * 100))