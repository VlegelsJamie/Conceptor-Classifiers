""" Parameters used for the different classifiers. """

C_standard = {'mu': 0.0,
              'spectral_radius': 1.06,
              'W_in_scale': 0.4,
              'bias_scale': 0.4,
              'leaking_rate': 0.28,
              'reservoir_dim': 25}

C_combined = {'mu': 0.28,
              'spectral_radius': 1.23,
              'W_in_scale': 1.81,
              'bias_scale': 2.1,
              'leaking_rate': 0.39,
              'reservoir_dim': 500}

C_reduced = {'mu': 0.5,
             'spectral_radius': 0.54,
             'W_in_scale': 2.0,
             'bias_scale': 1.48,
             'leaking_rate': 1.0,
             'reservoir_dim': 94}

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

ESN_forecast = {'spectral_radius': 1.57,
                'W_in_scale': 1.63,
                'bias_scale': 1.04,
                'leaking_rate': 0.15,
                'reservoir_dim': 418,
                'beta': 0.076}

ESN_class_label = {'spectral_radius': 0.93,
                   'W_in_scale': 1.70,
                   'bias_scale': 0.88,
                   'leaking_rate': 0.72,
                   'reservoir_dim': 500,
                   'beta': 0.055}

C_reduced_aperture_adjusted = C_reduced | {'aps_pos': [30.0,
                                                       330.0,
                                                       191.30576370446795,
                                                       191.09108779659283,
                                                       240.4259712905323,
                                                       6.602270322234099,
                                                       208.03043037537574,
                                                       35.0,
                                                       330.0,
                                                       270.0,
                                                       225.0,
                                                       252.2348164228707,
                                                       270.0,
                                                       105.19588779062894],
                                           'aps_neg': [104.84917003078749,
                                                       104.31312254272845,
                                                       82.91676539402451,
                                                       151.79058833513687,
                                                       155.88962373107188,
                                                       139.10833744066042,
                                                       94.18686369301234,
                                                       176.70585538249728,
                                                       68.71082104955141,
                                                       131.64040819811248,
                                                       100.99685747152961,
                                                       66.68723260215484,
                                                       161.91417055421206,
                                                       168.54069730636905]}
