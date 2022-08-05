"""
Module functioning as entry point to evaluate classifier performance over a number of trials.
"""

import argparse
import time

import numpy as np

from utils.IO import get_classifier
from utils.data import get_train_test_split


def get_args():
    parser = argparse.ArgumentParser('Conceptor Classifiers')
    parser.add_argument('-cl', '--classifier', type=str, default='C_standard', help='Classifier type')
    parser.add_argument('-t', '--num_trials', type=int, default='1', help='Number of trials to average results over')

    args = parser.parse_args()
    return args


def eval(model, params, X_train, y_train, X_test, y_test):
    """
    Function to approximate the accuracy of a classifier by averaging performance over
    multiple trials with randomly initialized ESN weights.

    :param: model: Classifier model.
    :param: params: List of hyperparameters.
    :param: X_train: train samples.
    :param: y_train: train class labels.
    :param: X_test: test samples.
    :param: y_test: test class labels.
    """

    start_time = time.process_time()

    # Define experiment parameters
    nr_trials = args.num_trials
    test_accs = np.zeros(nr_trials)

    model.set_params(**params)
    for trial in range(nr_trials):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        test_accs[trial] = score

        #print(f'Trial {trial} score: {score}')

    print(f'Mean test score: {np.mean(test_accs)}')
    print(f'Std test score: {np.std(test_accs)}')

    print('--- %s seconds ---' % (time.process_time() - start_time))


if __name__ == '__main__':
    args = get_args()

    X_train, y_train, X_test, y_test = get_train_test_split()

    model, params, _ = get_classifier(args.classifier)

    eval(model, params, X_train, y_train, X_test, y_test)
