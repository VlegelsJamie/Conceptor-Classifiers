"""
Module functioning as entry point to optimize classifier parameters via Bayesian
optimization.
"""

import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

from utils.IO import get_classifier, print_best_parameters
from utils.data import get_train_test_split


def get_args():
    """ Get command line arguments.

    :return: Command line arguments.
    """

    parser = argparse.ArgumentParser('Conceptor Classifiers')
    parser.add_argument('-cl', '--classifier', type=str, default='C_standard', help='Classifier type')
    parser.add_argument('-i', '--num_iters', type=int, default='50', help='Number of function evaluations')
    parser.add_argument('-f', '--num_folds', type=int, default='5', help='Number of folds to evaluate the dataset with')

    return parser.parse_args()


def optimize_params(model, param_space, X_train, y_train, num_iters, num_folds):
    """ Hyperparameter tuning with Bayesian optimization via stratified cross validation.

    :param model: Model to optimize hyperparameters of.
    :param param_space: Parameter space to sample values from for evaluation.
    :param X_train: Training samples.
    :param y_train: Training classes.
    :param num_iters: Number of function evaluations.
    :param num_folds: Number of cross-validation folds.
    """

    @use_named_args(param_space)
    def objective(**params):
        model.set_params(**params)
        return -np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=num_folds)))

    res_gp = gp_minimize(objective, param_space, n_calls=num_iters, random_state=42, noise=0.04, acq_func="LCB",
                         acq_optimizer="sampling", verbose=True)

    print_best_parameters(res_gp)

    return res_gp


if __name__ == '__main__':
    args = get_args()

    X_train, y_train, X_test, y_test = get_train_test_split()

    model, param_space = get_classifier(args.classifier)
    num_iters = args.num_iters
    num_folds = args.num_folds

    optimize_params(model, param_space, X_train, y_train, num_iters, num_folds)
