""" Module to handle input from the user. """

from models.classifiers.C_classifier import CClassifier
from models.classifiers.esn_classifier import EsnClassifier
from params import classifier_params
from params import param_spaces


def get_classifier(classifier: str):
    """ Get classifier with appropriate hyperparameters and evaluation space.

    :param: classifier: String representation of the classifier name.
    :return: Classifier object, hyperparameters, and evaluation space.
    """

    match classifier:
        case 'C_standard':
            return CClassifier(method='standard', **classifier_params.C_standard), param_spaces.C_standard_space

        case 'C_reduced':
            return CClassifier(method='reduced', **classifier_params.C_reduced), param_spaces.C_reduced_space

        case 'C_combined':
            return CClassifier(method='combined', **classifier_params.C_combined), param_spaces.C_combined_space

        case 'C_tubes':
            return CClassifier(method='tubes', **classifier_params.C_tubes), param_spaces.C_tubes_space

        case 'C_reservoir':
            return CClassifier(method='reservoir', **classifier_params.C_reservoir), param_spaces.C_reservoir_space

        case 'C_forecast':
            return CClassifier(method='forecast', **classifier_params.C_forecast), param_spaces.C_forecast_space

        case 'ESN_one_hot':
            return (EsnClassifier(method='one_hot', **classifier_params.ESN_one_hot),
                    param_spaces.ESN_one_hot_space)

        case 'ESN_forecast':
            return EsnClassifier(method='forecast', **classifier_params.ESN_forecast), param_spaces.ESN_forecast_space

        case 'C_standard_aperture':
            return (CClassifier(method='standard', **classifier_params.C_standard_aperture),
                    param_spaces.C_standard_aperture_space)

        case 'C_reduced_aperture':
            return (CClassifier(method='reduced', **classifier_params.C_reduced_aperture),
                    param_spaces.C_reduced_aperture_space)

        case 'C_combined_aperture':
            return (CClassifier(method='combined', **classifier_params.C_combined_aperture),
                    param_spaces.C_combined_aperture_space)

        case _:
            print('Please select valid classifier.')


def print_best_parameters(res_gp):
    """ Print optimal parameters according to Bayesian Optimization search.

    :param res_gp: Optimization result.
    """

    print('Best parameters:')
    for param, value in zip(res_gp.space.dimension_names, res_gp.x):
        print(param, ': ', value)
