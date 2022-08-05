""" Module implementing an Echo State Network. """

import numpy as np
from numpy.random import normal, uniform, rand
from scipy import linalg


class Esn:
    """ Class representing an Echo State Network. """

    def __init__(self, spectral_radius=1.0, W_in_scale=1.0, bias_scale=1.0, leaking_rate=1.0,
                 sparsity=0.0, reservoir_dim=20, beta=0.01, random_state=None):
        """ Initialize model parameters.

        :param spectral_radius: Spectral radius of ESN.
        :param W_in_scale: Scaling of input weights.
        :param bias_scale: Scaling of bias nodes.
        :param leaking_rate: Leaking rate of ESN.
        :param sparsity: Sparsity of the network.
        :param reservoir_dim: Reservoir dimensions.
        :param beta: Regularization coefficient.
        :param random_state: Random state.
        """

        self.spectral_radius = spectral_radius
        self.W_in_scale = W_in_scale
        self.bias_scale = bias_scale
        self.leaking_rate = leaking_rate
        self.sparsity = sparsity
        self.reservoir_dim = reservoir_dim
        self.beta = beta
        self.random_state = random_state
        self.in_dim = None

        self.W_in = None
        self.W_reservoir = None
        self.bias = None
        self.W_out = None

    def initialize_weights_gaussian(self):
        """
        Initialize network weights according to a gaussian distribution with a mean of 0 and
        standard deviation of 1. Afterwards, scale according to the parameters specified.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.W_in = normal(0, 1.0, (self.reservoir_dim, self.in_dim)) * self.W_in_scale
        self.W_reservoir = normal(0, 1.0, (self.reservoir_dim, self.reservoir_dim))
        # Make W_reservoir sparsely connected
        self.W_reservoir[rand(*self.W_reservoir.shape) < self.sparsity] = 0
        unit_radius = max(abs(linalg.eig(self.W_reservoir)[0]))
        self.W_reservoir *= self.spectral_radius / unit_radius
        self.bias = normal(0, 1.0, self.reservoir_dim) * self.bias_scale

    def initialize_weights_uniform(self):
        """
        Initialize network weights according to uniform distribution with a range of
        [-1, 1]. Afterwards, scale according to the parameters specified.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.W_in = uniform(-1.0, 1.0, (self.reservoir_dim, self.in_dim)) * self.W_in_scale
        self.W_reservoir = uniform(-0.5, 0.5, (self.reservoir_dim, self.reservoir_dim))
        # Make W_reservoir sparsely connected
        self.W_reservoir[rand(*self.W_reservoir.shape) < self.sparsity] = 0
        unit_radius = max(abs(linalg.eig(self.W_reservoir)[0]))
        self.W_reservoir *= self.spectral_radius / unit_radius
        self.bias = uniform(-1.0, 1.0, self.reservoir_dim) * self.bias_scale

    def run_reservoir(self, X, washout_period=0):
        """
        Run the ESN and obtain activation states (nr_timesteps x reservoir_dim).

        :param X: Input samples from which the activation states are obtained.
        :param washout_period: Amount of timesteps after which data is collected.
        :return: Activation states (nr_timesteps x reservoir_dim).
        """

        nr_timesteps = X.shape[0]

        states = np.zeros((nr_timesteps - washout_period, self.reservoir_dim))
        x = np.zeros(self.reservoir_dim)
        for n in range(nr_timesteps):
            pre_activation = np.dot(self.W_reservoir, x) + np.dot(self.W_in, X[n]) + self.bias

            x = (1 - self.leaking_rate) * x + self.leaking_rate * np.tanh(pre_activation)

            if n >= washout_period:
                states[:, n - washout_period] = x

        return states

    def get_W_out(self, X, y):
        """
        Compute output weights for each class via Ridge regression for computed reservoir
        activation states.

        :param X: Pre-computed reservoir activation states to learn with.
        :param y: Output to learn from.
        :return: Output weights.
        """

        return np.dot(np.dot(y.T, X), linalg.inv(np.dot(X.T, X) + self.beta * np.eye(self.reservoir_dim)))

    def get_params(self, deep=True):
        """ Get model parameter values. """

        return {'spectral_radius': self.spectral_radius,
                'W_in_scale': self.W_in_scale,
                'bias_scale': self.bias_scale,
                'leaking_rate': self.leaking_rate,
                'sparsity': self.sparsity,
                'reservoir_dim': self.reservoir_dim,
                'beta': self.beta,
                'random_state': self.random_state}

    def set_params(self, **parameters):
        """ Set model parameters. """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self
