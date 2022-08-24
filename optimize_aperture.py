"""
Module functioning as entry point to optimize classifier aperture values via Bayesian
optimization with fixed classifier parameters. Optimized with fixed train states.
"""

import argparse

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args

from utils.IO import print_best_parameters, get_classifier
from utils.data import get_train_test_split


def get_args():
    """ Get command line arguments.

    :return: Command line arguments.
    """

    parser = argparse.ArgumentParser('Conceptor Classifiers')
    parser.add_argument('-cl', '--classifier', type=str, default='C_reduced', help='Classifier type')
    parser.add_argument('-i', '--num_iters', type=int, default='50', help='Number of function evaluations')
    parser.add_argument('-f', '--num_folds', type=int, default='5', help='Number of folds to evaluate the dataset with')

    return parser.parse_args()


def optimize_ap(model, param_space, X_train, y_train, num_iters, num_folds):
    """
    Adapted hyperparameter tuning function to optimize positive and negative aperture values
    with otherwise fixed classifier parameters.

    :param X_train: Training samples.
    :param y_train: Training classes.
    :param num_iters: Number of function evaluations.
    :param num_folds: Number of cross-validation folds.
    """

    @use_named_args(param_space)
    def objective(**params):
        if model.mu == 1.0:
            model.set_params(**{'aps_pos': list(params.values())})
        if model.mu == 0.0:
            model.set_params(**{'aps_neg': list(params.values())})
        return -np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=num_folds)))

    return gp_minimize(objective, param_space, n_calls=num_iters, noise=0.0000001, acq_func="LCB",
                             acq_optimizer="sampling", verbose=True)


if __name__ == '__main__':
    args = get_args()

    X_train, y_train, X_test, y_test = get_train_test_split()

    model, param_space = get_classifier(f"{args.classifier}_aperture")
    num_iters = args.num_iters
    num_folds = args.num_folds

    model.set_params(**{'mu': 1.0, 'aps_neg': list(np.ones(14))})
    res_gp_pos = optimize_ap(model, param_space[:14], X_train, y_train, num_iters, num_folds)

    model.set_params(**{'mu': 0.0, 'aps_pos': list(np.ones(14))})
    res_gp_neg = optimize_ap(model, param_space[14:], X_train, y_train, num_iters, num_folds)

    print_best_parameters(res_gp_pos)
    print_best_parameters(res_gp_neg)
