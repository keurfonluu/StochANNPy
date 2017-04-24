# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from ._base_neural_network import BaseNeuralNetwork
from stochopy import Evolutionary

__all__ = [ "ENNClassifier" ]


class ENNClassifier(BaseNeuralNetwork, ClassifierMixin):
    """
    Evolutionary neural network classifier.
    
    This model optimizes the log-loss function using Differential Evolution,
    Particle Swarm Optimization of Covariance Matrix Adaptation - Evolution
    Strategy.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple or list, length = n_layers-2, default (10,)
        The ith element represents the number of neurons in the ith hidden
        layer.
    activation : {'logistic', 'tanh'}, default 'tanh'
        Activation function the hidden layer.
        - 'logistic', the logistic sigmoid function.
        - 'tanh', the hyperbolic tan function.
    alpha : scalar, optional, default 0.
        L2 penalty (regularization term) parameter.
    max_iter : int, optional, default 100
        Maximum number of iterations.
    solver : {'de', 'pso', 'cmaes'}, default 'pso'
        Evolutionary Algorithm optimizer.
    popsize : int, optional, default 10
        Population size.
    w : scalar, optional, default 0.72
        Inertial weight. Only used when solver = 'pso'.
    c1 : scalar, optional, default 1.49
        Cognition parameter. Only used when solver = 'pso'.
    c2 : scalar, optional, default 1.49
        Sociability parameter. Only used when solver = 'pso'.
    l : scalar, optional, default 0.1
        Velocity clamping percentage. Only used when solver = 'pso'.
    F : scalar, optional, default 1.
        Differential weight. Only used when solver = 'de'.
    CR : scalar, optional, default 0.5
        Crossover probability. Only used when solver = 'de'.
    sigma : scalar, optional, default 1.
        Step size. Only used when solver = 'cmaes'.
    mu_perc : scalar, optional, default 0.5
        Number of parents as a percentage of population size. Only used
        when solver = 'cmaes'.
    eps1 : scalar, optional, default 1e-8
        Minimum change in best individual.
    eps2 : scalar, optional, default 1e-8
        Minimum objective function precision.
    random_state : int, optional, default None
        Seed for random number generator.
    
    Examples
    --------
    Import the module and initialize the classifier:
    
    >>> import numpy as np
    >>> from stochannpy import ENNClassifier
    >>> clf = ENNClassifier(hidden_layer_sizes = (5,))
    
    Fit the training set:
    
    >>> clf.fit(X_train, y_train)
    
    Predict the test set:
    
    >>> ypred = clf.predict(X_test)
    
    Compute the accuracy:
    
    >>> print(np.mean(ypred == y_test))
    """

    def __init__(self, hidden_layer_sizes = (10,), max_iter = 100, alpha = 0.,
                 activation = "logistic", solver = "pso", popsize = 10,
                 w = 0.72, c1 = 1.49, c2 = 1.49, l = 0.1, F = 1., CR = 0.5,
                 sigma = 1., mu_perc = 0.5, eps1 = 1e-8, eps2 = 1e-8,
                 random_state = None):
        super(ENNClassifier, self).__init__(
            hidden_layer_sizes = hidden_layer_sizes,
            activation = activation,
            alpha = alpha,
            max_iter = max_iter,
            random_state = random_state,
        )
        if not isinstance(solver, str) or solver not in [ "pso", "de", "cmaes" ]:
            raise ValueError("solver must either be 'pso', 'de' or 'cmaes', got %s" % solver)
        else:
            self.solver = solver
        if not isinstance(popsize, int) or popsize < 2:
            raise ValueError("popsize must be an integer > 1, got %s" % popsize)
        else:
            self._popsize = int(popsize)
        if not isinstance(eps1, float) and not isinstance(eps1, int) or eps1 < 0.:
            raise ValueError("eps1 must be positive, got %s" % eps1)
        else:
            self._eps1 = eps1
        if not isinstance(eps2, float) and not isinstance(eps2, int):
            raise ValueError("eps2 must be an integer or float, got %s" % eps2)
        else:
            self._eps2 = eps2
        self._hyperparams = {"w": w, "c1": c1, "c2": c2, "l": l,
                             "F": F, "CR": CR,
                             "sigma": sigma, "mu_perc": mu_perc}
        return
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : ndarray of length n_samples
            Target values.
        """
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [ hidden_layer_sizes ]
        hidden_layer_sizes = list(hidden_layer_sizes)
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = unique_labels(y)
        self.n_outputs_ = len(self.classes_)
        self.n_layers_ = len(hidden_layer_sizes) + 2
        self.layer_units_ = ( [ self.n_features_ ] + hidden_layer_sizes + [ self.n_outputs_ ] )
        
        # Convert y to a sparse matrix
        self.ymat_ = np.zeros((self.n_samples_, len(self.classes_)))
        for i in range(self.n_samples_):
            self.ymat_[i,y[i]] = 1.
        
        # Store meta information for the parameters
        self.coef_indptr_ = []
        start = 0
        for i in range(self.n_layers_-1):
            n_fan_in, n_fan_out = self.layer_units_[i]+1, self.layer_units_[i+1]
            end = start + (n_fan_in * n_fan_out)
            self.coef_indptr_.append((start, end, (n_fan_out, n_fan_in)))
            start = end
            
        # Initialize functions
        self._init_functions()
        
        # Initialize coefficients
        initial_coefs = self._init_coefs()
        packed_initial_coefs = self._pack(initial_coefs)
        
        # Optimize using L-BFGS
        self._optimizer = Evolutionary(self._loss,
                                       n_dim = len(packed_initial_coefs),
                                       max_iter = self.max_iter,
                                       popsize = self._popsize,
                                       eps1 = self._eps1,
                                       eps2 = self._eps2,
                                       args = (X,))
        if self.solver == "de":
            packed_coefs, self.loss_ = self._optimizer.optimize(solver = "de",
                                                                F = self._hyperparams["F"],
                                                                CR = self._hyperparams["CR"])
        elif self.solver == "pso":
            packed_coefs, self.loss_ = self._optimizer.optimize(solver = "pso",
                                                                w = self._hyperparams["w"],
                                                                c1 = self._hyperparams["c1"],
                                                                c2 = self._hyperparams["c2"],
                                                                l = self._hyperparams["l"])
        elif self.solver == "cmaes":
            packed_coefs, self.loss_ = self._optimizer.optimize(solver = "cmaes",
                                                                sigma = self._hyperparams["sigma"],
                                                                mu_perc = self._hyperparams["mu_perc"])
        self.coefs_ = self._unpack(packed_coefs)
        return
    
    def predict(self, X):
        """
        Predict using the trained model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        ypred : ndarray of length n_samples
            Predicted labels.
        """
        return self.classes_[np.argmax(self._predict(X), axis = 1)]
    
    def predict_log_proba(self, X):
        """
        Log of probability estimates.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        yprob : ndarray of shape (n_samples, n_outputs)
            The ith row and jth column holds the log-probability of the ith
            sample to the jth class
        """
        return np.log(self.predict_proba(X))
    
    def predict_proba(self, X):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        yprob : ndarray of shape (n_samples, n_outputs)
            The ith row and jth column holds the probability of the ith sample
            to the jth class
        """
        ypred = self._predict(X)
        if self.n_outputs_ == 1:
            ypred = ypred.ravel()
        if ypred.ndim == 1:
            return np.vstack([1. - ypred, ypred]).transpose()
        else:
            return ypred