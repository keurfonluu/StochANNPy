# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.base import ClassifierMixin
from ._base_neural_network import BaseNeuralNetwork
from stochopy import MonteCarlo

__all__ = [ "BNNClassifier" ]


class BNNClassifier(BaseNeuralNetwork, ClassifierMixin):
    """
    Bayesian neural network classifier.
    
    This model samples the weights of the log-loss function using the
    Metropolis-Hastings algorithm or Hamiltonian (Hybrid) Monte-Carlo.
    
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
    max_iter : int, optional, default 1000
        Maximum number of iterations.
    sampler : {'mcmc', 'hmc'}, default 'mcmc'
        Sampling method.
    stepsize : scalar, optional, default 1e-1
        If sampler = 'mcmc', standard deviation of gaussian perturbation.
        If sampler = 'hmc', leap-frog step size.
    n_leap : int, optional, default 10
        Number of leap-frog steps. Only used when sampler = 'hmc'.
    bounds : scalar, optional, default 1.
        Search space boundaries for initialization.
    random_state : int, optional, default None
        Seed for random number generator.
    
    Examples
    --------
    Import the module and initialize the classifier:
    
    >>> import numpy as np
    >>> from stochannpy import BNNClassifier
    >>> clf = BNNClassifier(hidden_layer_sizes = (5,))
    
    Fit the training set:
    
    >>> clf.fit(X_train, y_train)
    
    Predict the test set:
    
    >>> ypred = clf.predict(X_test)
    
    Compute the accuracy:
    
    >>> print(np.mean(ypred == y_test))
    """

    def __init__(self, hidden_layer_sizes = (10,), activation = "tanh", alpha = 0.,
                 max_iter = 1000, sampler = "mcmc", stepsize = 1e-1, n_leap = 10,
                 bounds = 1., random_state = None):
        super(BNNClassifier, self).__init__(
            hidden_layer_sizes = hidden_layer_sizes,
            activation = activation,
            alpha = alpha,
            max_iter = max_iter,
            bounds = bounds,
            random_state = random_state,
        )
        if not isinstance(sampler, str) or sampler not in [ "mcmc", "hmc" ]:
            raise ValueError("sampler must either be 'mcmc' or 'hmc', got %s" % sampler)
        else:
            self.sampler = sampler
        if not isinstance(stepsize, float) and not isinstance(stepsize, int) or stepsize <= 0.:
            raise ValueError("stepsize must be positive, got %s" % stepsize)
        else:
            self.stepsize = stepsize
        if not isinstance(n_leap, int) or n_leap <= 0:
            raise ValueError("n_leap must be a positive integer, got %s" % n_leap)
        else:
            self.n_leap = n_leap
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
        # Initialize
        self._initialize(X, y)
        
        # Initialize boundaries
        n_dim = np.sum([ np.prod(i[-1]) for i in self.coef_indptr_ ])
        lower = np.full(n_dim, -self.bounds)
        upper = np.full(n_dim, self.bounds)
        
        # Optimize using Hamiltonian Monte-Carlo
        self._markov_chain = MonteCarlo(self._loss,
                                        lower = lower,
                                        upper = upper,
                                        max_iter = self.max_iter,
                                        args = (X,))
        if self.sampler == "hmc":
            self._markov_chain.sample(
                sampler = "hamiltonian",
                fprime = self._grad,
                stepsize = self.stepsize,
                n_leap = self.n_leap,
                args = (X,))
        elif self.sampler == "mcmc":
            self._markov_chain.sample(
                sampler = "hastings",
                stepsize = self.stepsize)
        return
    
    def predict(self, X, n_burnin = 0):
        """
        Predict using the trained model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        n_burnin : int, optional, default 0
            Number of samples corresponding to burn-in period (ignored in
            prediction).
        
        Returns
        -------
        ypred : ndarray of length n_samples
            Predicted labels.
        """
        ypred = self._predict(X)
        ypred = np.sum(self._integrate(ypred, n_burnin), 0)
        return self.classes_[np.argmax(ypred, axis = 1)]
    
    def predict_log_proba(self, X, n_burnin = 0):
        """
        Log of probability estimates.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        n_burnin : int, optional, default 0
            Number of samples corresponding to burn-in period (ignored in
            prediction).
        
        Returns
        -------
        yprob : ndarray of shape (n_samples, n_outputs)
            The ith row and jth column holds the log-probability of the ith
            sample to the jth class
        """
        return np.log(self.predict_proba(X, n_burnin))
    
    def predict_proba(self, X, n_burnin = 0):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        n_burnin : int, optional, default 0
            Number of samples corresponding to burn-in period (ignored in
            prediction).
        
        Returns
        -------
        yprob : ndarray of shape (n_samples, n_outputs)
            The ith row and jth column holds the probability of the ith sample
            to the jth class
        """
        ypred = self._predict(X)
        ypred = np.sum(self._integrate(ypred, n_burnin), 0)
        if self.n_outputs_ == 1:
            ypred = ypred.ravel()
        if ypred.ndim == 1:
            return np.vstack([1. - ypred, ypred]).transpose()
        else:
            return ypred
    
    def _predict(self, X):
        ypred = [ np.array([]) for i in range(self.max_iter) ]
        for i in range(self.max_iter):
            unpacked_coefs = self._unpack(self._markov_chain.models[:,i])
            Z, activations = self._forward_pass(unpacked_coefs, X)
            ypred[i] = activations[-1]
        return ypred
            
    def _integrate(self, ypred, n_burnin = 0):
        weights = np.exp(-self._markov_chain.energy[n_burnin:])
        weights /= np.sum(weights)
        ypred = [ ypred[n_burnin+i]*weights[i] for i in range(len(weights)) ]
        return ypred
    
    def plot_coefs(self, n_burnin = 0, ignore_bias = True):
        """
        Plot the posterior distribution of the sampled coefficients (biases
        and weights).
        
        Parameters
        ----------
        n_burnin : int, optional, default 0
            Number of samples corresponding to burn-in period to ignore.
        ignore_bias : bool, optional, default True
            If False, plot the posterior distribution of the biases.
        """
        weights = self.weights
        if not ignore_bias:
            biases = self.biases
            
        for i in range(self.n_layers_-1):
            fig = plt.figure(figsize = (8, 8), facecolor = "white")
            ax1 = fig.add_subplot(1, 1, 1)
            
            for j in range(weights[i].shape[0]):
                wmin = np.min(weights[i][j,:])
                wmax = np.max(weights[i][j,:])
                aw = np.linspace(wmin, wmax, 101)
                kde = gaussian_kde(weights[i][j,:])
                pdf = kde.pdf(aw)
                ax1.plot(aw, pdf, linestyle = "-")
            
            if not ignore_bias:
                for j in range(biases[i].shape[0]):
                    bmin = np.min(biases[i][j,:])
                    bmax = np.max(biases[i][j,:])
                    ab = np.linspace(bmin, bmax, 101)
                    kde = gaussian_kde(biases[i][j,:])
                    pdf = kde.pdf(ab)
                    ax1.plot(ab, pdf, linestyle = ":")
        return
    
    @property
    def weights(self):
        """
        list of ndarray
        Sampled neural network weights. The ith element holds all the weights
        sampled for the layer i.
        """
        weights = []
        for i in range(self.n_layers_-1):
            start, end, shape = self.coef_indptr_[i]
            nw = np.prod(shape)-shape[0]
            w_i = np.zeros((nw, self.max_iter))
            for j, k in enumerate(range(start+shape[0], end)):
                w_i[j,:] = self._markov_chain.models[k,:]
            weights.append(np.array(w_i))
        return weights
    
    @property
    def biases(self):
        """
        list of ndarray
        Sampled neural network biases. The ith element holds all the biases
        sampled for the layer i.
        """
        biases = []
        for i in range(self.n_layers_-1):
            start, end, shape = self.coef_indptr_[i]
            b_i = np.zeros((shape[0], self.max_iter))
            for j, k in enumerate(range(start, start+shape[0])):
                b_i[j,:] = self._markov_chain.models[k,:]
            biases.append(np.array(b_i))
        return biases
    
    @property
    def models(self):
        """
        ndarray of shape (n_dim, max_iter)
        Sampled models.
        """
        return self._markov_chain._models
    
    @property
    def energy(self):
        """
        ndarray of shape (max_iter)
        Energy of sampled models.
        """
        return self._markov_chain._energy
    
    @property
    def acceptance_ratio(self):
        """
        scalar between 0 and 1
        Acceptance ratio of sampler.
        """
        return self._markov_chain._acceptance_ratio