# Cross-validation

Running `python run.py` performs seeded 5-fold cross-validation for regularized
logistic regression before fitting the model on the full training set. The
script reports validation accuracy for each fold and their mean.

The cross-validation settings are:

- Number of folds: 5
- Random seed: 12
- Training iterations: 300
- Learning rate: `1e-10`
- Regularization parameter: `0.5`

Missing feature values are imputed using medians calculated from each fold's
training split. For the final fit, medians are calculated from the full
training set.
