"""
Module functioning as entry point to optimize classifier aperture values via Bayesian
optimization with fixed classifier parameters. Optimized with fixed train states.
"""

import argparse
import warnings

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args

from params import classifier_params, param_spaces
from utils.figures import plot_partial_dependence
from utils.IO import print_best_parameters
from utils.data import get_train_test_split
from models.classifiers.C_classifier import CEsnClassifier

warnings.filterwarnings('ignore')


def get_args():
    """ Get command line arguments.

    :return: Command line arguments.
    """

    parser = argparse.ArgumentParser('Conceptor Classifiers')
    parser.add_argument('-i', '--num_iters', type=int, default='50', help='Number of function evaluations')
    parser.add_argument('-f', '--num_folds', type=int, default='5', help='Number of folds to evaluate the dataset with')

    return parser.parse_args()


def optimize_ap(X_train, y_train, num_iters, num_folds):
    """
    Adapted hyperparameter tuning function to optimize positive and negative aperture values
    with otherwise fixed classifier parameters.

    :param X_train: Training samples.
    :param y_train: Training classes.
    :param num_iters: Number of function evaluations.
    :param num_folds: Number of cross-validation folds.
    """

    param_space = param_spaces.aperture_pos_analysis_space
    model = CEsnClassifier(method='reduced', **classifier_params.C_reduced)

    @use_named_args(param_space)
    def objective(**params):
        if model.mu == 1.0:
            model.set_params(**{'aps_pos': list(params.values())})
        if model.mu == 0.0:
            model.set_params(**{'aps_neg': list(params.values())})
        return -np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=num_folds)))

    model.set_params(**{'mu': 1.0})
    res_gp_pos = gp_minimize(objective, param_space, n_calls=num_iters, random_state=42, noise=0.04, verbose=True)

    print_best_parameters(res_gp_pos)

    param_space = param_spaces.aperture_neg_analysis_space
    model.set_params(**{'mu': 0.0})
    res_gp_neg = gp_minimize(objective, param_space, n_calls=num_iters, random_state=42, noise=0.04, verbose=True)

    print_best_parameters(res_gp_neg)

    plot_partial_dependence(res_gp_pos, n_points=50, sample_source='expected_minimum', minimum='expected_minimum')
    plot_partial_dependence(res_gp_neg, n_points=50, sample_source='expected_minimum', minimum='expected_minimum')


if __name__ == '__main__':
    args = get_args()

    X_train, y_train, X_test, y_test = get_train_test_split()

    num_iters = args.num_iters
    num_folds = args.num_folds

    optimize_ap(X_train, y_train, num_iters, num_folds)
