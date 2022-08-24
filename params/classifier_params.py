""" Parameters used for the different classifiers. """

C_standard = {'mu': 0.0,
              'spectral_radius': 1.06,
              'W_in_scale': 0.4,
              'bias_scale': 0.4,
              'leaking_rate': 0.28,
              'reservoir_dim': 25}

C_reduced = {'mu': 0.0,
             'spectral_radius': 1.07,
             'W_in_scale': 1.97,
             'bias_scale': 0.91,
             'leaking_rate': 0.38,
             'reservoir_dim': 500}

C_combined = {'mu': 0.28,
              'spectral_radius': 1.23,
              'W_in_scale': 1.81,
              'bias_scale': 1.9,
              'leaking_rate': 0.39,
              'reservoir_dim': 500}

C_tubes = {'mu': 0.80,
           'spectral_radius': 1.3,
           'W_in_scale': 1.82,
           'bias_scale': 1.40,
           'leaking_rate': 0.1,
           'reservoir_dim': 80}

C_reservoir = {'spectral_radius': 0.67,
               'W_in_scale': 0.62,
               'bias_scale': 0.71,
               'leaking_rate': 0.17,
               'reservoir_dim': 85}

C_forecast = {'spectral_radius': 0.67,
              'W_in_scale': 1.64,
              'bias_scale': 0.91,
              'leaking_rate': 0.17,
              'reservoir_dim': 100,
              'beta': 0.0001}

ESN_one_hot = {'spectral_radius': 1.33,
               'W_in_scale': 1.93,
               'bias_scale': 1.77,
               'leaking_rate': 0.73,
               'reservoir_dim': 500,
               'beta': 0.18}

ESN_forecast = {'spectral_radius': 1.57,
                'W_in_scale': 1.63,
                'bias_scale': 1.04,
                'leaking_rate': 0.15,
                'reservoir_dim': 420,
                'beta': 0.076}

C_standard_aperture = C_standard | {'aps_pos': [4.59, 77.93, 200.13, 143.29, 8.16, 1.23, 51.17,
                                                3.24, 117.22, 14.50, 39.87, 771.60, 18.44, 8.03],
                                    'aps_neg': [1950.21, 1739.29, 1934.67, 1787.22, 1983.06, 1748.58, 1723.30,
                                                2041.73, 1795.24, 1899.81, 1991.44, 1746.46, 1737.00, 1924.83],
                                    'mu': 0.0,
                                    'random_state': 42}

C_reduced_aperture = C_reduced | {'aps_pos': [19.88, 68.73, 164.88, 205.19, 48.43, 2.08, 65.50,
                                              14.76, 353.90, 32.12, 46.63, 170.82, 25.09, 16.59],
                                  'aps_neg': [177.27, 180.29, 200.33, 179.03, 190.54, 190.29, 176.76,
                                              167.49, 172.61, 176.81, 206.76, 166.58, 177.59, 170.30],
                                  'mu': 0.0,
                                  'random_state': 42}

C_combined_aperture = C_combined | {'aps_pos': [191.86, 105.98, 96.61, 96.51, 127.70, 85.87, 85.27,
                                                150.99, 154.18, 146.63, 100.94, 113.96, 89.31, 111.11],
                                    'aps_neg': [71.23, 59.01, 58.52, 56.85, 65.20, 64.27, 60.80,
                                                53.87, 52.27, 66.42, 60.53, 50.49, 56.69, 58.74],
                                    'mu': 0.9,
                                    'random_state': 42}
