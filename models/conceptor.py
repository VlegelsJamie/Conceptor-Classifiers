""" Module for creating conceptors and computing evidence values. """

import numpy as np
from numpy.linalg import multi_dot, inv
from scipy.interpolate import CubicSpline


def get_C(X, y, aps_pos=None, aps_neg=None):
    """
    Compute conceptors for given samples. Creates a conceptor per class defined by the class
    labels. Either determines aperture values heuristically when aps_pos and aps_neg are None
    or directly computes final conceptors with specified aperture values.

    :param X: Samples to compute conceptors for.
    :param y: Class labels of samples.
    :param aps_pos: Optional positive conceptor aperture values.
    :param aps_neg: Optional negative conceptor aperture values.
    :return: Positive and negative conceptors.
    """

    # Define conceptor params
    nr_classes = len(np.unique(y))
    nr_features = X.shape[1]
    I = np.eye(nr_features)
    ap_exps = 15
    test_ap_exps = np.arange(0.0, ap_exps, 1.0)
    test_ap = 2 ** test_ap_exps

    # Compute correlation matrices
    R_norm = np.zeros((nr_classes, nr_features, nr_features))
    R_others_norm = np.zeros((nr_classes, nr_features, nr_features))
    R_all = np.dot(X.T, X)
    for class_id in range(nr_classes):
        class_X = X[np.where(y == class_id)]
        R = np.dot(class_X.T, class_X)
        R_norm[class_id] = R / len(class_X)
        R_other = R_all - R
        R_others_norm[class_id] = R_other / (len(X) - len(class_X))

    # Check if aperture needs to be determined heuristically
    if aps_pos is None:
        # Compute preliminary conceptors
        C_pre_poss = np.zeros((len(test_ap), nr_classes, nr_features, nr_features))
        for class_id in range(nr_classes):
            for ap_id in range(len(test_ap)):
                C_pre_poss[ap_id, class_id] = np.dot(R_norm[class_id],
                                                     inv(R_norm[class_id] + (test_ap[ap_id] ** -2) * I))

        # Heuristically determine aperture
        aps_pos = np.zeros(nr_classes)
        for class_id in range(nr_classes):
            norms_pos = np.zeros(len(test_ap))
            for ap_exp in range(len(test_ap)):
                norms_pos[ap_exp] = np.linalg.norm(C_pre_poss[ap_exp, class_id], 'fro') ** 2
            int_points = np.arange(0.0, ap_exps - 0.9, 0.1)
            norms_pos_int = CubicSpline(test_ap_exps, norms_pos)
            norms_pos_int_grad = norms_pos_int.derivative()
            norms_pos_int_grad = np.abs(norms_pos_int_grad(int_points))
            aps_pos[class_id] = 2 ** int_points[np.argmax(norms_pos_int_grad)]

    if aps_neg is None:
        C_pre_negs = np.zeros((len(test_ap), nr_classes, nr_features, nr_features))
        for class_id in range(nr_classes):
            for ap_id in range(len(test_ap)):
                C_pre_negs[ap_id, class_id] = I - np.dot(R_others_norm[class_id],
                                                         inv(R_others_norm[class_id] + (test_ap[ap_id] ** -2) * I))

        aps_neg = np.zeros(nr_classes)
        for class_id in range(nr_classes):
            norms_neg = np.zeros(len(test_ap))
            for ap_exp in range(len(test_ap)):
                norms_neg[ap_exp] = np.linalg.norm(I - C_pre_negs[ap_exp, class_id], 'fro') ** 2
            int_points = np.arange(0.0, ap_exps - 0.9, 0.1)
            norms_neg_int = CubicSpline(test_ap_exps, norms_neg)
            norms_neg_int_grad = norms_neg_int.derivative()
            norms_neg_int_grad = np.abs(norms_neg_int_grad(int_points))
            aps_neg[class_id] = 2 ** int_points[np.argmax(norms_neg_int_grad)]

    # Compute final conceptors.
    C_poss = np.zeros((nr_classes, nr_features, nr_features))
    C_negs = np.zeros((nr_classes, nr_features, nr_features))
    for class_id in range(nr_classes):
        C_poss[class_id] = np.dot(R_norm[class_id], inv(R_norm[class_id] +
                                                        (aps_pos[class_id] ** -2) * I))
        C_negs[class_id] = I - np.dot(R_others_norm[class_id],
                                      inv(R_others_norm[class_id] + (aps_neg[class_id] ** -2) * I))

    return C_poss, C_negs


def get_evidence(X, C):
    """ Compute evidence with conceptors from testing samples.

    :param X: States to compute evidences from.
    :param C: Conceptors.
    :return: positive and negative evidence.
    """

    ev = np.einsum('ij, hji->ih', X, np.inner(C, X))

    return ev


def morph_C(C_poss, C_negs, mu):
    """ Linearly blend positive and negative conceptors based on parameter mu.

    :param C_poss: Positive conceptors.
    :param C_negs: Negative conceptors.
    :param mu: Blending factor.
    :return: Morphed conceptors.
    """

    return (1 - mu) * C_negs + mu * C_poss
