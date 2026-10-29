# ML Project 1

## Description

This is the repository for the first project of EPFL CS-433. The goal of the project is to predict heart disease using machine learning techniques. To do so, we train on the BFRSS databse.

## Setup and run

You will need a python environment with Numpy installed. You will also have to download the dataset following hte instructions on the project handout, then put the csv files inside a folder named `dataset` located at the root.

You can then open a terminal at the root of the repository and execute `python3 run.py`. A new `predictions.csv` file should appear after a few minutes.

## Files

On top of `implementations.py` and `run.py`, we added `helpers.py`, and the testing files `conftest.py` and `test_project1_public.py`. Those files were given. The dataset should be in a folder named `dataset`. You should never modify those files.

## Explanation of the code

`implementations.py` contains functions to compute linear regression (stochastic) gradient, ridge regression, logistic regression and regularized logistic regression.

Running `python run.py` performs seeded 5-fold cross-validation for regularized
logistic regression before fitting the model on the full training set. The
script reports validation accuracy for each fold and their mean.

The cross-validation settings are:

- Number of folds: 5
- Random seed: 12
- Training iterations: 300
- Learning rate: `0.02`
- Regularization parameter: `0.001`

The features are standardized. Then missing feature values are imputed using medians calculated from each fold's
training split. For the final fit, medians are calculated from the full
training set.

More details are available in the report.


### Authors:
Ferdinand Chiu, Simon Streuli