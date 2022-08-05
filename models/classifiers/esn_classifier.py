"""
Module implementing an Echo State Network classifier with two methods for classification:
via signal prediction similarity and direct class output learning.
"""

import numpy as np

from models.classifiers.base_esn_classifier import BaseEsnClassifier


class EsnClassifier(BaseEsnClassifier):
    """ Class representing an Echo State Network classifier. """

    def __init__(self, method='class_label', **kwargs):
        """ Initialize model parameters.

        :param method: Esn classification method.
        :param kwargs: Keyword arguments to be passed to the ESN.
        """

        super().__init__(**kwargs)

        self.method = method

    def fit(self, X_train, y_train):
        """ Fit model parameters of chosen classification method.

        :param X_train: training samples.
        :param y_train: class labels of training samples.
        """

        super().fit(X_train, y_train)
        train_states = super().run_reservoir(X_train)

        if self.method == 'forecast':
            self.fit_signal_prediction(train_states, X_train, y_train)

        if self.method == 'class_label':
            train_states = np.mean(train_states, axis=1)
            y_train = np.eye(self.nr_classes)[y_train]
            self.W_out = super().get_W_out(train_states, y_train)

    def fit_signal_prediction(self, train_states, X_train, y_train):
        """
        Compute output weights of Echo State Network for the classification flavour of
        predicting signal prediction similarity.

        :param train_states: Reservoir states resulting from feeding ESN with X_train.
        :param X_train: train samples.
        :param y_train: Output training class labels.
        """

        train_states = np.delete(train_states, -1, axis=1)
        X_train_targets = np.delete(X_train, 0, axis=1)

        train_states = train_states.reshape(self.nr_train_samples * (self.nr_timesteps - 1), self.reservoir_dim)
        X_train_targets = X_train_targets.reshape(self.nr_train_samples * (self.nr_timesteps - 1), self.in_dim)

        # Get output weights per class for 1-step delay task
        y_train = np.repeat(y_train, (self.nr_timesteps - 1))
        self.W_out = np.zeros((self.nr_classes, self.in_dim, self.reservoir_dim))
        for class_id in range(self.nr_classes):
            class_ids = np.where(y_train == class_id)
            self.W_out[class_id] = super().get_W_out(train_states[class_ids], X_train_targets[class_ids])

    def predict(self, X_test):
        """ Predict class labels according to selected method.

        :param X_test: test samples.
        """

        super().predict(X_test)
        test_states = super().run_reservoir(X_test)

        if self.method == 'forecast':
            pred = self.predict_signal(test_states, X_test)

        if self.method == 'class_label':
            test_states = np.mean(test_states, axis=1)
            pred = np.inner(test_states, self.W_out)

        return pred

    def predict_signal(self, test_states, X_test):
        """ Predict class labels according to signal prediction method.

        :param test_states: States obtained from running the ESN with X_test.
        :param X_test: test samples.
        :return: predictions.
        """

        X_test_targets = np.delete(X_test, 0, axis=1)
        test_states = np.delete(test_states, -1, axis=1)

        # Predict signal from class output weights
        target_pred = np.einsum('hjk,ilk->hilj', self.W_out, test_states)

        # Compute RMSE of from all class predictions
        pred = -np.mean(np.sqrt(np.mean((target_pred - X_test_targets) ** 2, axis=2)), axis=2).T

        return pred

    def get_params(self, deep=True):
        """ Get model parameter values. """

        params = {'method': self.method}

        return params | super().get_params()
