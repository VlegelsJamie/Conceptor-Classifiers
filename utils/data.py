""" This module can load the LSST dataset data and preprocess it. """

import numpy as np
import arff
from scipy.stats import rankdata


def get_train_test_split():
    """ Get train and test data and preprocess them.

    :return preprocessed train and test samples and class labels.
    """

    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    X_train_processed, X_test_processed = preprocess(X_train.copy(), X_test.copy())

    return X_train_processed, y_train, X_test_processed, y_test


def get_train_data():
    """ Get input and output of LSST dataset train samples.

    :return: Input and output of LSST dataset train samples.
    """

    data = np.array([arff.load(open('../LSST dataset/LSSTDimension1_TRAIN.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension2_TRAIN.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension3_TRAIN.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension4_TRAIN.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension5_TRAIN.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension6_TRAIN.arff'))['data']], dtype=np.float32)

    # Get all output values from end of vector
    y = data[0, :, -1]
    # Replace class values with values from 0-13
    y = rankdata(list(y), method='dense') - 1

    # Delete label and Shape (nr samples x nr timesteps x nr passbands)
    X = np.delete(data, -1, 2).transpose((1, 2, 0))

    return X, y


def get_test_data():
    """ Get input and output of LSST dataset test samples.

    :return: Input and output of LSST dataset test samples.
    """

    data = np.array([arff.load(open('../LSST dataset/LSSTDimension1_TEST.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension2_TEST.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension3_TEST.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension4_TEST.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension5_TEST.arff'))['data'],
                     arff.load(open('../LSST dataset/LSSTDimension6_TEST.arff'))['data']], dtype=np.float32)

    # Get all output values from end of vector
    y = data[0, :, -1]
    # Replace class values with values from 0-13
    y = rankdata(list(y), method='dense') - 1

    # Delete label and Shape (nr samples x nr timesteps x nr passbands)
    X = np.delete(data, -1, 2).transpose((1, 2, 0))

    return X, y


def preprocess(X_train, X_test):
    """ Firstly robust shift and scale, then unpack tuple as arguments to normalize.

    :param X_train: Train input to preprocess.
    :param X_test: Test input to preprocess.
    :return: preprocessed train and test input.
    """

    return normalize(*robust_shift_scale(X_train, X_test))


def robust_shift_scale(X_train, X_test):
    """
    Robustly shift and scale input according to the median and interquartile range values
    per band.

    :param X_train: Train input to be robustly shifted and scaled.
    :param X_test: Test input to be robustly shifted and scaled.
    :return: Robustly shifted and scaled train and test input.
    """

    for i, band in enumerate(X_train.transpose(2, 0, 1)):
        q1_x = np.percentile(band, 25, interpolation='nearest')
        q3_x = np.percentile(band, 75, interpolation='nearest')
        median = np.median(band)

        X_train[:, :, i] = (X_train[:, :, i] - median) / (q3_x - q1_x)
        X_test[:, :, i] = (X_test[:, :, i] - median) / (q3_x - q1_x)

    return X_train, X_test


def normalize(X_train, X_test):
    """ Scale input to range of [-1, 1] without shifting.

    :param X_train: Train data to shift.
    :param X_test: Test data to shift.
    :return: scaled train and test data.
    """

    X_train = X_train / (np.max(X_train, axis=(1, 2), keepdims=True) - np.min(X_train, axis=(1, 2), keepdims=True))
    X_test = X_test / (np.max(X_test, axis=(1, 2), keepdims=True) - np.min(X_test, axis=(1, 2), keepdims=True))

    return X_train, X_test
