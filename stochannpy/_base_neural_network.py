# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from sklearn.base import BaseEstimator

__all__ = [ "BaseNeuralNetwork" ]


class BaseNeuralNetwork(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for artifical neural network.
    
    Do not use this class, please use derived classes.
    
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
    random_state : int, optional, default None
        Seed for random number generator.
    """
    
    @abstractmethod
    def __init__(self, hidden_layer_sizes = (10,), activation = "tanh", alpha = 0.,
                 max_iter = 100, random_state = None):
        # Check inputs
        if isinstance(hidden_layer_sizes, tuple) or isinstance(hidden_layer_sizes, list):
            if not np.all([ isinstance(l, int) and l > 0 for l in hidden_layer_sizes ]):
                raise ValueError("hidden_layer_sizes must contains positive integers only, got %s" % hidden_layer_sizes) 
            else:
                self.hidden_layer_sizes = hidden_layer_sizes
        else:
            raise ValueError("hidden_layer_sizes must be a tuple or a list")
        if not isinstance(activation, str) and activation not in [ "logistic", "tanh" ]:
            raise ValueError("activation must either be 'logistic' or 'tanh', got %s" % activation)
        else:
            self.activation = activation
        if not isinstance(alpha, float) and not isinstance(alpha, int) or alpha < 0.:
            raise ValueError("alpha must be positive, got %s" % alpha)
        else:
            self.alpha = alpha
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer, got %s" % max_iter)
        else:
            self.max_iter = max_iter
        if random_state is not None:
            np.random.seed(random_state)
        return
    
    def _predict(self, X):
        """
        Predict using the trained model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        h : ndarray
            Output layer activation values.
        """
        unpacked_coefs = self.coefs_
        Z, activations = self._forward_pass(unpacked_coefs, X)
        return activations[-1]
    
    def _forward_pass(self, unpacked_coefs, X):
        """
        Perform a feedforward pass on the network.
        
        Parameters
        ----------
        unpacked_coefs : list of ndarray
            The ith element of the list holds the biases and weights of the
            ith layer.
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        Z : list of ndarray
            Layers propagation matrices.
        activations : list of ndarray
            Layers activation values.
        """
        Z = [ np.array([]) ]
        activations = [ np.hstack((np.ones((X.shape[0],1)), X)) ]
        for i in range(1, self.n_layers_-1):
            Z.append(np.dot(activations[i-1], unpacked_coefs[i-1].transpose()))
            activations.append(np.hstack((np.ones((Z[i].shape[0],1)), self._func(Z[i]))))
        Z.append(np.dot(activations[-1], unpacked_coefs[-1].transpose()))
        activations.append(self._output_func(Z[-1]))
        activations[-1] = np.clip(activations[-1], 1e-10, 1 - 1e-10)
        return Z, activations
    
    def _backprop(self, Z, activations, unpacked_coefs):
        """
        Perform the backpropagation algorithm to compute the gradient.
        
        Parameters
        ----------
        Z : list of ndarray
            Layers propagation matrices.
        activations : list of ndarray
            Layers activation values.
        unpacked_coefs : list of ndarray
            The ith element of the list holds the biases and weights of the
            ith layer.
        
        Returns
        -------
        grad : list of ndarray
            The ith element of the list holds the gradient of the biases and
            weights of the ith layer.
        """
        sigma = [ np.array([]) for i in range(self.n_layers_) ]
        sigma[-1] = activations[-1] - self.ymat_
        for i in range(self.n_layers_-2, 0, -1):
            sigma[i] = np.dot(sigma[i+1], unpacked_coefs[i]) * self._fprime(np.hstack((np.ones((Z[i].shape[0],1)), Z[i])))
            sigma[i] = sigma[i][:,1:]
        
        # Accumulate gradient
        delta = []
        for i in range(self.n_layers_-1):
            delta.append(np.dot(sigma[i+1].T, activations[i]))
            
        # Unroll gradient
        grad = []
        for i in range(self.n_layers_-1):
            grad.append(delta[i] / self.n_samples_ + self.alpha / self.n_samples_ * np.hstack((np.zeros((unpacked_coefs[i].shape[0], 1)), unpacked_coefs[i][:,1:])))        
        return grad
    
    def _loss(self, packed_coefs, X):
        """
        Compute the log loss function.
        
        Parameters
        ----------
        packed_coefs : ndarray
            Neural network biases and weights.
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        loss : scalar
            Loss function value.
        """
        unpacked_coefs = self._unpack(packed_coefs)                 # Unpack
        Z, activations = self._forward_pass(unpacked_coefs, X)      # Forward pass
        values = np.sum(np.array([ np.dot(s[:,1:].ravel(), s[:,1:].ravel()) \
                                  for s in unpacked_coefs ]))       # L2 regularization penalty
        loss = np.sum(-self.ymat_*np.log(activations[-1]) \
                      - (1.-self.ymat_)*np.log(1.-activations[-1])) # Log loss function
        loss += 0.5 * self.alpha * values
        loss /= self.n_samples_
        return loss
    
    def _grad(self, packed_coefs, X):
        """
        Compute the gradient of the loss function.
        
        Parameters
        ----------
        packed_coefs : ndarray
            Neural network biases and weights.
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        grad : ndarray
            Gradient of neural network biases and weights.
        """
        unpacked_coefs = self._unpack(packed_coefs)                 # Unpack
        Z, activations = self._forward_pass(unpacked_coefs, X)      # Forward pass
        grad = self._backprop(Z, activations, unpacked_coefs)       # Backward propagation
        grad = self._pack(grad)                                     # Pack gradient coefficients
        return grad
    
    def _init_functions(self):
        if self.activation == "logistic":
            self._func = self._sigmoid
            self._fprime = self._sigmoid_grad
        elif self.activation == "tanh":
            self._func = self._tanh
            self._fprime = self._tanh_grad
        if self.n_outputs_ > 1:
            self._output_func = self._softmax
        else:
            self._output_func = self._sigmoid
        return
    
    def _init_coefs(self):
        coefs_init = []
        for i in range(self.n_layers_-1):
            if self.activation == "logistic":
                init_bound = np.sqrt(2. / (self.layer_units_[i] + self.layer_units_[i+1] + 1))
            else:
                init_bound = np.sqrt(6. / (self.layer_units_[i] + self.layer_units_[i+1] + 1))
            coefs_init.append(self._rand(self.layer_units_[i+1], self.layer_units_[i]+1, init_bound))
        return coefs_init
    
    def _rand(self, n1, n2, init_bound = 1.):
        """
        Generate uniform array between -init_bound and init_bound.
        
        Parameters
        ----------
        n1 : int
            First dimension of array.
        n2 : int
            Second dimension of array.
        init_bound : float, optional, default 1.
            Random number absolute maximum.
        
        Returns
        -------
        r : ndarray of shape (n1, n2)
            Uniform array between -init_bound and init_bound.
        """
        rnd = 2. * np.random.rand(n1, n2) - 1.
        return init_bound * rnd
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
        
    def _sigmoid_grad(self, x):
        sig = self._sigmoid(x)
        return sig * (1 - sig)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_grad(self, x):
        return 1. - self._tanh(x)**2
    
    def _softmax(self, x):
        tmp = x - x.max(axis = 1)[:,np.newaxis]
        x = np.exp(tmp)
        x /= x.sum(axis = 1)[:,np.newaxis]
        return x
    
    def _pack(self, unpacked_parameters):
        """
        Pack the coefficients into a 1-D array.
        
        Parameters
        ----------
        unpacked_parameters : list of ndarray
            The ith element of the list holds the biases and weights of the
            ith layer.
        
        Returns
        -------
        packed_parameters : ndarray
            Neural network biases and weights.
        """
        return np.hstack([ l.ravel(order = "F") for l in unpacked_parameters ])

    def _unpack(self, packed_parameters):
        """
        Unpack the coefficients into a list of NumPy arrays.
        
        Parameters
        ----------
        packed_parameters : ndarray
            Neural network biases and weights.
        
        Returns
        -------
        unpacked_parameters : list of ndarray
            The ith element of the list holds the biases and weights of the
            ith layer.
        """
        unpacked_parameters = []
        for i in range(self.n_layers_ - 1):
            start, end, shape = self.coef_indptr_[i]
            unpacked_parameters.append(np.reshape(packed_parameters[start:end], shape, order = "F"))
        return unpacked_parameters