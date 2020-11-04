# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectPercentile, SelectFpr, VarianceThreshold
import ds_functions as ds

# get data
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
print('X shape:', X.shape)
labels = pd.unique(y)

##### Filters - Classification #####

# chi2
chi, pval = chi2(X, y)
print('\nChi2 test scores:', chi)
print('\nChi2 p-values:', pval)

# f_classif
f, pval = f_classif(X, y)
print('\nChi2 test scores:', chi)
print('\nChi2 p-values:', pval)

# chi2
chi, pval = chi2(X, y)
print('\nChi2 test scores:', chi)
print('\nChi2 p-values:', pval)

# SelectKBest (highest scoring number)
selector = SelectKBest(mutual_info_classif, k=2)
X_new = selector.fit_transform(X, y)
print('Scores:', selector.scores_)
print('\nSelectKBest - Original data space:\n', X[0:3])
print('\nSelectKBest - New data space:\n', X_new[0:3])

# SelectPercentile (highest scoring percentage)
selector = SelectPercentile(f_classif, percentile=50)
X_new = selector.fit_transform(X, y)
print("P-values:",selector.pvalues_)
print('\nSelectKBest - Original data space:\n', X[0:3])
print('\nSelectKBest - New data space:\n', X_new[0:3])

# SelectFpr (false positive rate)
selector = SelectFpr(chi2, alpha=0.01)
X_new = selector.fit_transform(X, y)
print('\nSelectFpr - Original data space:\n', X[0:3])
print('\nSelectFpr - New data space:\n', X_new[0:3])

# SelectFdr (false discovery rate)
# SelectFwe (family wise error) 


##### Filters - Unsupervised #####

# variance
selector = VarianceThreshold()
selector.fit(X)
print('\nFeature variance =', selector.variances_)

selector = VarianceThreshold(threshold=0.2)
X_new = selector.fit_transform(X)
print('\nNumber of features =', len(X_new[0]))


##### Wrappers #####

