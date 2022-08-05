"""
Helper class functioning as a base Echo State Network classifier. Stores necessary
data information.
"""

from abc import ABCMeta

import numpy as np
from sklearn.metrics import accuracy_score

from models.esn import Esn


class BaseEsnClassifier(Esn, metaclass=ABCMeta):
    """ Class representing base ESN classifier. """

    def __init__(self, **kwargs):
        """ Initialize model parameters.

        :param kwargs: Keyword arguments to be passed to the ESN.
        """

        super().__init__(**kwargs)

        self.nr_classes = None
        self.nr_timesteps = None
        self.nr_train_samples = None
        self.nr_test_samples = None

    def run_reservoir(self, X, washout_period=0):
        """
        Run the ESN and obtain activation states (nr_samples x nr_timesteps x reservoir_dim).

        :param X: Input samples from which the activation states are obtained.
        :param washout_period: Amount of timesteps after which data is collected.
        :return: Activation states (nr samples x nr_timesteps x reservoir_dim).
        """

        nr_samples = X.shape[0]
        nr_timesteps = X.shape[1]

        states = np.zeros((nr_samples, nr_timesteps - washout_period, self.reservoir_dim))
        x = np.zeros((nr_samples, self.reservoir_dim))
        for n in range(nr_timesteps):
            pre_activation = np.inner(x, self.W_reservoir) + np.inner(X[:, n], self.W_in) + self.bias

            x = (1 - self.leaking_rate) * x + self.leaking_rate * np.tanh(pre_activation)

            if n >= washout_period:
                states[:, n - washout_period] = x

        return states

    def fit(self, X_train, y_train):
        """ Store data information and initialize weights of ESN.

        :param X_train: training samples.
        :param y_train: class labels of training samples.
        """

        self.nr_classes = len(np.unique(y_train))
        self.nr_timesteps = X_train.shape[1]
        self.nr_train_samples = X_train.shape[0]
        self.in_dim = X_train.shape[2]

        super().initialize_weights_gaussian()

    def predict(self, X_test):
        """ Store size of test data.

        :param X_test: test samples.
        """

        self.nr_test_samples = X_test.shape[0]

    def score(self, X_test, y_test):
        """ Get accuracy score for the given test samples.

        :param X_test: test samples.
        :param y_test: test sample class labels.
        """

        pred = self.predict(X_test)
        return accuracy_score(y_test, np.argmax(pred, axis=1))

    def get_params(self, deep=True):
        """ Get model parameter values. """

        return super().get_params()
