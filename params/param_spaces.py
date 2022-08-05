""" Parameter spaces used for Bayesian optimization of the different classifier parameters. """

from skopt.space import Real, Integer

C_standard_space = [Real(0.0, 1.0, prior='uniform', name='mu'),
                    Real(0.4, 1.4, prior='uniform', name='spectral_radius'),
                    Real(0.4, 1.4, prior='uniform', name='W_in_scale'),
                    Real(0.4, 1.4, prior='uniform', name='bias_scale'),
                    Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                    Integer(5, 25, 'uniform', name='reservoir_dim')]

C_combined_space = [Real(0.0, 1.0, prior='uniform', name='mu'),
                    Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                    Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                    Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                    Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                    Integer(5, 500, 'uniform', name='reservoir_dim')]

C_reduced_space = [Real(0.0, 1.0, prior='uniform', name='mu'),
                   Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                   Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                   Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                   Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                   Integer(5, 500, 'uniform', name='reservoir_dim')]

C_tubes_space = [Real(0.0, 1.0, prior='uniform', name='mu'),
                 Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                 Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                 Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                 Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                 Integer(5, 150, 'uniform', name='reservoir_dim')]

C_reservoir_space = [Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                     Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                     Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                     Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                     Integer(5, 150, 'uniform', name='reservoir_dim')]

C_forecast_space = [Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                    Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                    Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                    Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                    Integer(5, 150, 'uniform', name='reservoir_dim'),
                    Real(0.0001, 10.0, prior='log-uniform', name='beta')]

ESN_forecast_space = [Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                      Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                      Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                      Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                      Integer(5, 500, 'uniform', name='reservoir_dim'),
                      Real(0.0001, 10.0, prior='log-uniform', name='beta')]

ESN_class_label_space = [Real(0.01, 2.0, prior='uniform', name='spectral_radius'),
                         Real(0.01, 2.0, prior='uniform', name='W_in_scale'),
                         Real(0.01, 2.0, prior='uniform', name='bias_scale'),
                         Real(0.01, 1.0, prior='uniform', name='leaking_rate'),
                         Integer(5, 500, 'uniform', name='reservoir_dim'),
                         Real(0.0001, 10.0, prior='log-uniform', name='beta')]

aperture_pos_analysis_space = [Real(30.0, 100.0, prior='uniform', name='ap1'),
                               Real(110.0, 330.0, prior='uniform', name='ap2'),
                               Real(90.0, 270.0, prior='uniform', name='ap3'),
                               Real(80.0, 240.0, prior='uniform', name='ap4'),
                               Real(90.0, 270.0, prior='uniform', name='ap5'),
                               Real(2.5, 7.5, prior='uniform', name='ap6'),
                               Real(100.0, 300.0, prior='uniform', name='ap7'),
                               Real(35.0, 105.0, prior='uniform', name='ap8'),
                               Real(110.0, 330.0, prior='uniform', name='ap9'),
                               Real(90.0, 270.0, prior='uniform', name='ap10'),
                               Real(75.0, 225.0, prior='uniform', name='ap11'),
                               Real(90.0, 270.0, prior='uniform', name='ap12'),
                               Real(90.0, 270.0, prior='uniform', name='ap13'),
                               Real(40.0, 120.0, prior='uniform', name='ap14')]

aperture_neg_analysis_space = [Real(60.0, 180.0, prior='uniform', name='ap1'),
                               Real(60.0, 180.0, prior='uniform', name='ap2'),
                               Real(60.0, 180.0, prior='uniform', name='ap3'),
                               Real(60.0, 180.0, prior='uniform', name='ap4'),
                               Real(60.0, 180.0, prior='uniform', name='ap5'),
                               Real(60.0, 180.0, prior='uniform', name='ap6'),
                               Real(60.0, 180.0, prior='uniform', name='ap7'),
                               Real(60.0, 180.0, prior='uniform', name='ap8'),
                               Real(60.0, 180.0, prior='uniform', name='ap9'),
                               Real(60.0, 180.0, prior='uniform', name='ap10'),
                               Real(60.0, 180.0, prior='uniform', name='ap11'),
                               Real(60.0, 180.0, prior='uniform', name='ap12'),
                               Real(60.0, 180.0, prior='uniform', name='ap13'),
                               Real(60.0, 180.0, prior='uniform', name='ap14')]
