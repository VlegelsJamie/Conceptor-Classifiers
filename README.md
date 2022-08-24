# Conceptor Classifiers


This repository is the official implementation of the Bachelor's thesis [Multivariate Time Series Classification Using Conceptors: Exploring Methods Using Astronomical Object Data](). It implements the following lightweight time series conceptor classifiers built upon Echo State Networks (ESNs), as well as two standard ESN classifiers:
```
- C_standard
- C_reduced
- C_combined
- C_timestep
- C_reservoir
- C_forecast
- ESN_forecast
- ESN_one_hot
```

These classifiers were optimized and evaluated on the LSST dataset, but can also be run by defining your own parameters within the [classifier_params.py](params/classifier_params.py) module.
You can also optimize the model hyperparameters using a specified number of Bayesian optimization function evaluations and folds over the training set.

These models were created with extensibility in mind and can therefore also easily and independently be applied to other datasets.

## Project Structure

```
Conceptor Classifiers                
│   eval.py                         # Evaluation script
│   optimize_aperture.py            # Aperture optimization script
│   optimize_params.py              # Bayesian Optimization script   
│
├───figures                         # Generated figures
│
├───LSST dataset                    # LSST dataset
│                    
├───models                      
│   │   conceptor.py                # Conceptors
│   │   esn.py                      # Echo State Network
│   │
│   └───classifiers
│       │   base_esn_classifier.py  # Base Echo State Network classifier
│       │   C_classifier.py         # Conceptor Echo State Network Classifier
│       │   esn_classifier.py       # Echo State Network Classifier
│
├───params                      
│   │   classifier_params.py        # Stored parameters
│   │   param_spaces.py             # Parameter spaces for Bayesian optimization
│        
└───utils
    │   data.py                     # Dataloader script
    │   figures.py                  # Figure generator
    │   IO.py                       # Handle classifier selection   
```

## Requirements

To install, please run the following commands ([Python >= 3.10](https://www.python.org/downloads/)):

```setup
git clone
cd Conceptor Classifiers
pip install -r requirements.txt
```

While the LSST dataset is made available within this repository, it is also downloadable from the [Time Series Classification](http://www.timeseriesclassification.com/description.php?Dataset=LSST) website.

## Evaluation

To evaluate any classifier on the LSST dataset, run:

```eval
python eval.py --classifier C_standard  # Classifier to evaluate (see above for options)
               --num_trials 20          # Number of trials to average results over
```

This will allow you to evaluate the performance of the selected classifier using a specified 
number of random Echo State Network initializations.

The models are run using hyperparameters defined in the 
[classifier_params.py](params/classifier_params.py) module.

## Model Optimization

To optimize hyperparameters of a classifier using Bayesian optimization, run:

```cross-validate
python optimize_params.py --classifier C_standard  # Classifier to optimize (see above for options)
                          --num_iters 100          # Number of function evaluations of the Bayesian optimizer
                          --num_folds 5            # Number of cross-validation folds
```

This will allow you to perform Bayesian optimization evaluated with k-fold cross-validation 
on the LSST dataset with a specified number of iterations and folds over the dataset. 

The models are evaluated over hyperparameter spaces defined in the 
[param_spaces.py](params/param_spaces.py) module.

## Aperture Optimization

To optimize individual aperture values of a conceptor classifier, run:

```cross-validate
python optimize_aperture.py --classifier C_standard  # Classifier to optimize aperture values for (see above for options)
                            --num_iters 200          # Number of function evaluations of the Bayesian optimizer  
                            --num_folds 5            # Number of cross-validation folds
```

This will allow you to perform Bayesian optimization as described previously, but over 
individual aperture values with fixed classifier parameters. This is limited to the ```C_standard```, ```C_reduced```, and ```C_combined``` classifiers.

Both the optimal hyperparameters for evaluation as well as the hyperparameter evaluation spaces 
can be found in the [classifier_params.py](params/classifier_params.py) and [param_spaces.py](params/param_spaces.py) modules.