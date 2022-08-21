"""
Module implementing an abstract conceptor classifier built upon an Echo State Network
classifier. It implements four purely conceptor and 2 ESN and conceptor combination
classification methods specified only by the "timestep" and "method" parameters.
"""

import numpy as np
from sklearn.preprocessing import normalize

from models.classifiers.base_esn_classifier import BaseEsnClassifier
from models.classifiers.esn_classifier import EsnClassifier
from models.conceptor import get_C, get_evidence, morph_C


class CClassifier(EsnClassifier):
    """ Class representing the conceptor classifier. """

    def __init__(self, method='combined', mu=0.5, aps_pos=None, aps_neg=None, **kwargs):
        """ Initialize model parameters.

        :param timestep: Method of creating conceptors.
        :param method: Method of classification.
        :param mu: evidence blending parameter.
        :param aps_pos: aperture values for the positive conceptors.
        :param aps_neg: aperture values for the negative conceptors.
        :param kwargs: arguments to be passed to the ESN classifier.
        """

        super().__init__(method=method, **kwargs)

        self.mu = mu
        self.aps_pos = aps_pos
        self.aps_neg = aps_neg

        self.C_poss = None
        self.C_negs = None

    def run_reservoir(self, X, Cs=None, washout_period=0):
        """
        Run the ESN, possibly with conceptor in the update loop, and obtain activation
        states (nr_samples x nr_timesteps x reservoir_dim).

        :param X: Input samples from which the activation states are obtained.
        :param Cs: Optional conceptors applied to the ESN update loop.
        :param washout_period: Amount of timesteps after which data is collected.
        :return: Activation states (nr samples x nr timesteps x reservoir_dim).
        """

        nr_samples = X.shape[0]
        nr_timesteps = X.shape[1]

        states = np.zeros((nr_samples, nr_timesteps - washout_period, self.reservoir_dim))
        x = np.zeros((nr_samples, self.reservoir_dim))
        for n in range(nr_timesteps):
            pre_activation = np.inner(x, self.W_reservoir) + np.inner(X[:, n], self.W_in) + self.bias

            x = (1 - self.leaking_rate) * x + self.leaking_rate * np.tanh(pre_activation)

            if Cs is not None and Cs.ndim == 3:
                x = np.inner(x, Cs[n])

            if Cs is not None and Cs.ndim == 2:
                x = np.inner(x, Cs)

            if n >= washout_period:
                states[:, n - washout_period] = x

        return states

    def fit(self, X_train, y_train):
        """ Fit model parameters according to chosen classification method.

        :param X_train: training samples.
        :param y_train: class labels of training samples.
        """

        BaseEsnClassifier.fit(self, X_train, y_train)
        train_states = self.run_reservoir(X_train)

        if self.method == 'standard':
            train_states = train_states.reshape(self.nr_train_samples, self.nr_timesteps * self.reservoir_dim)
            self.C_poss, self.C_negs = get_C(train_states, y_train, self.aps_pos, self.aps_neg)

        if self.method == 'combined':
            train_states = train_states.reshape(self.nr_train_samples * self.nr_timesteps, self.reservoir_dim)
            self.C_poss, self.C_negs = get_C(train_states, np.repeat(y_train, 36), self.aps_pos, self.aps_neg)

        if self.method == 'reduced':
            train_states = np.mean(train_states, axis=1)
            self.C_poss, self.C_negs = get_C(train_states, y_train, self.aps_pos, self.aps_neg)

        if (self.method == 'tubes') | (self.method == 'forecast') | (self.method == 'reservoir'):
            self.C_poss = np.zeros((self.nr_classes, self.nr_timesteps, self.reservoir_dim, self.reservoir_dim))
            self.C_negs = np.zeros((self.nr_classes, self.nr_timesteps, self.reservoir_dim, self.reservoir_dim))
            for n in range(self.nr_timesteps):
                self.C_poss[:, n], self.C_negs[:, n] = get_C(train_states[:, n, :], y_train, self.aps_pos, self.aps_neg)

        if self.method == 'forecast':
            super().fit_signal_prediction(train_states, X_train, y_train)

    def predict(self, X_test):
        """ Predict class labels according to classification method selected.

        :param X_test: test samples.
        :return: predictions.
        """

        BaseEsnClassifier.predict(self, X_test)
        test_states = self.run_reservoir(X_test)

        if self.method == 'standard':
            test_states = test_states.reshape(self.nr_test_samples, self.nr_timesteps * self.reservoir_dim)
            pred = get_evidence(test_states, morph_C(self.C_poss, self.C_negs, self.mu))

        if self.method == 'combined':
            pred = np.zeros((self.nr_test_samples, self.nr_classes))
            C_combined = morph_C(self.C_poss, self.C_negs, self.mu)
            for n in range(self.nr_timesteps):
                pred += get_evidence(test_states[:, n, :], C_combined)

        if self.method == 'reduced':
            test_states = np.mean(test_states, axis=1)
            pred = get_evidence(test_states, morph_C(self.C_poss, self.C_negs, self.mu))

        if self.method == 'tubes':
            pred = np.zeros((self.nr_test_samples, self.nr_classes))
            for n in range(self.nr_timesteps):
                pred += get_evidence(test_states[:, n, :], morph_C(self.C_poss[:, n], self.C_negs[:, n], self.mu))

        if self.method == 'reservoir':
            test_states_C = self.get_states_C(X_test)
            pred = -np.mean(np.sqrt(np.mean((test_states_C - test_states) ** 2, axis=2)), axis=2).T

        if self.method == 'forecast':
            pred = self.predict_signal(test_states, X_test)

        return pred

    def predict_signal(self, test_states, X_test):
        """
        Replicate signal with filtered and unfiltered states and compute differences.

        :param test_states: States obtained from running the ESN with X_test.
        :param X_test: test samples.
        :return predictions.
        """

        # Get filtered test states.
        test_states_C = self.get_states_C(X_test)

        pred_reservoir = -np.mean(np.sqrt(np.mean((test_states_C - test_states) ** 2, axis=2)), axis=2).T

        X_test_targets = np.delete(X_test, 0, axis=1)
        test_states = np.delete(test_states, -1, axis=1)
        test_states_C = np.delete(test_states_C, -1, axis=2)

        target_pred = np.einsum('hjk,ilk->hilj', self.W_out, test_states)
        target_pred_C = np.einsum('hjk,hilk->hilj', self.W_out, test_states_C)

        pred = -np.mean(np.sqrt(np.mean((target_pred - X_test_targets) ** 2, axis=2)), axis=2).T
        pred_C = -np.mean(np.sqrt(np.mean((target_pred_C - X_test_targets) ** 2, axis=2)), axis=2).T
        pred_C2 = -np.mean(np.sqrt(np.mean((target_pred_C - target_pred) ** 2, axis=2)), axis=2).T

        return normalize(pred) + normalize(pred_C) + normalize(pred_C2) + normalize(pred_reservoir)

    def get_states_C(self, X):
        """ Compute states filtered by conceptors for every class.

        :param X: Samples to get states for.
        :return States resulting from applied conceptors.
        """

        test_states_C = np.zeros((self.nr_classes, self.nr_test_samples, X.shape[1], self.reservoir_dim))
        for class_id in range(self.nr_classes):
            test_states_C[class_id] = self.run_reservoir(X, Cs=self.C_poss[class_id])

        return test_states_C

    def get_params(self, deep=False):
        """ Get model parameter values. """

        params = {'mu': self.mu,
                  'aps_pos': self.aps_pos,
                  'aps_neg': self.aps_neg}

        return params | super().get_params()
