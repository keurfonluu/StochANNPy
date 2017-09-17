# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from sklearn.base import ClassifierMixin
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
    activation : {'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function the hidden layer.
        - 'logistic', the logistic sigmoid function.
        - 'tanh', the hyperbolic tan function.
        - 'relu', the REctified Linear Unit function.
    alpha : scalar, optional, default 0.
        L2 penalty (regularization term) parameter.
    max_iter : int, optional, default 100
        Maximum number of iterations.
    solver : {'de', 'pso', 'cpso', 'cmaes'}, default 'cpso'
        Evolutionary Algorithm optimizer.
    popsize : int, optional, default 10
        Population size.
    w : scalar, optional, default 0.72
        Inertial weight. Only used when solver = {'pso', 'cpso'}.
    c1 : scalar, optional, default 1.49
        Cognition parameter. Only used when solver = {'pso', 'cpso'}.
    c2 : scalar, optional, default 1.49
        Sociability parameter. Only used when solver = {'pso', 'cpso'}.
    l : scalar, optional, default 0.1
        Velocity clamping percentage. Only used when solver = {'pso', 'cpso'}.
    gamma : scalar, optional, default 1.25
        Competitivity parameter. Only used when solver = 'cpso'.
    delta : None or scalar, optional, default None
        Swarm maximum radius. Only used when solver = 'cpso'.
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
    bounds : scalar, optional, default 1.
        Search space boundaries for initialization.
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
                 activation = "relu", solver = "cpso", popsize = 10,
                 w = 0.72, c1 = 1.49, c2 = 1.49, l = 0.1, gamma = 1.25,
                 delta = None, F = 1., CR = 0.5, sigma = 1., mu_perc = 0.5,
                 eps1 = 1e-8, eps2 = 1e-8, bounds = 1., random_state = None):
        super(ENNClassifier, self).__init__(
            hidden_layer_sizes = hidden_layer_sizes,
            activation = activation,
            alpha = alpha,
            max_iter = max_iter,
            bounds = bounds,
            random_state = random_state,
        )
        self.solver = solver
        self.popsize = int(popsize)
        self.eps1 = eps1
        self.eps2 = eps2
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.l = l
        self.gamma = gamma
        self.delta = delta
        self.F = F
        self.CR = CR
        self.sigma = sigma
        self.mu_perc = mu_perc
        
    def _validate_hyperparameters(self):
        self._validate_base_hyperparameters()
        if not isinstance(self.solver, str) or self.solver not in [ "cpso", "pso", "de", "cmaes" ]:
            raise ValueError("solver must either be 'cpso', 'pso', 'de' or 'cmaes', got %s" % self.solver)
        if not isinstance(self.popsize, int) or self.popsize < 2:
            raise ValueError("popsize must be an integer > 1, got %s" % self.popsize)
        if not isinstance(self.eps1, float) and not isinstance(self.eps1, int) or self.eps1 < 0.:
            raise ValueError("eps1 must be positive, got %s" % self.eps1)
        if not isinstance(self.eps2, float) and not isinstance(self.eps2, int):
            raise ValueError("eps2 must be an integer or float, got %s" % self.eps2)
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : ndarray of length n_samples
            Target values.
        
        Returns
        -------
        self : returns a trained ENNClassifier.
        """
        # Check inputs and initialize
        self._validate_hyperparameters()
        X, y = self._initialize(X, y)
        
        # Initialize boundaries
        n_dim = np.sum([ np.prod(i[-1]) for i in self.coef_indptr_ ])
        lower = np.full(n_dim, -self.bounds)
        upper = np.full(n_dim, self.bounds)
        
        # Optimize using DE, PSO, CPSO or CMA-ES
        ea = Evolutionary(self._loss,
                          lower = lower,
                          upper = upper,
                          max_iter = self.max_iter,
                          popsize = self.popsize,
                          eps1 = self.eps1,
                          eps2 = self.eps2,
                          constrain = False,
                          args = (X,))
        if self.solver == "de":
            packed_coefs, self.loss_ = ea.optimize(solver = "de",
                                                   F = self.F,
                                                   CR = self.CR)
        elif self.solver == "pso":
            packed_coefs, self.loss_ = ea.optimize(solver = "pso",
                                                   w = self.w,
                                                   c1 = self.c1,
                                                   c2 = self.c2,
                                                   l = self.l)
        elif self.solver == "cpso":
            packed_coefs, self.loss_ = ea.optimize(solver = "cpso",
                                                   w = self.w,
                                                   c1 = self.c1,
                                                   c2 = self.c2,
                                                   l = self.l,
                                                   gamma = self.gamma,
                                                   delta = self.delta)
        elif self.solver == "cmaes":
            packed_coefs, self.loss_ = ea.optimize(solver = "cmaes",
                                                   sigma = self.sigma,
                                                   mu_perc = self.mu_perc)
        self.coefs_ = self._unpack(packed_coefs)
        self.flag_ = ea.flag
        self.n_iter_ = ea.n_iter
        self.n_eval_ = ea.n_eval        
        return self
    
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
        y_pred = self._predict(X)
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        return self._label_binarizer.inverse_transform(y_pred)
    
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
        y_pred = self._predict(X)
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        if y_pred.ndim == 1:
            return np.vstack([1. - y_pred, y_pred]).transpose()
        else:
            return y_pred
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : ndarray of length n_samples
            Target values.
            
        Returns
        -------
        acc : scalar
            Accuracy of prediction.
        """
        return np.mean(self.predict(X) == y)
    
    @property
    def weights(self):
        """
        list of ndarray
        Neural network weights. The ith element holds the weighs for the
        layer i.
        """
        return [ np.array(coefs[:,1:].T) for coefs in self.coefs_ ]
    
    @property
    def biases(self):
        """
        list of ndarray
        Neural network biases. The ith element holds the biases for the
        layer i.
        """
        return [ np.array(coefs[:,0]) for coefs in self.coefs_ ]
    
    @property
    def flag(self):
        """
        int
        Stopping criterion:
            - -1, maximum number of iterations is reached.
            - 0, best individual position changes less than eps1.
            - 1, fitness is lower than threshold eps2.
            - 2, NoEffectAxis (only when solver = 'cmaes').
            - 3, NoEffectCoord (only when solver = 'cmaes').
            - 4, ConditionCov (only when solver = 'cmaes').
            - 5, EqualFunValues (only when solver = 'cmaes').
            - 6, TolXUp (only when solver = 'cmaes').
            - 7, TolFun (only when solver = 'cmaes').
            - 8, TolX (only when solver = 'cmaes').
        """
        return self.flag_
    
    @property
    def n_iter(self):
        """
        int
        Number of iterations required to reach stopping criterion.
        """
        return self.n_iter_
    
    @property
    def n_eval(self):
        """
        int
        Number of function evaluations performed.
        """
        return self.n_eval_