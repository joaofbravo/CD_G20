# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectPercentile, SelectFpr, VarianceThreshold
import ds_functions as ds

np.set_printoptions(suppress=True)

# get data
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
print('X shape:', X.shape)
labels = pd.unique(y)

##### Filters - Classification #####

# chi2
chi, pval = chi2(X, y)
print('\nChi2 test scores:\n', chi)
print('Chi2 p-values:\n', pval)

# f_classif (ANOVA)
fvalue, pval = f_classif(X, y)
print('\nANOVA test scores:\n', fvalue)
print('ANOVA p-values:\n', pval)

# mutual_info_classif
mutual_info = mutual_info_classif(X, y, n_neighbors=3)
print('\nMI scores:\n', mutual_info)

# SelectKBest (highest scoring number)
selector = SelectKBest(mutual_info_classif, k=2)
X_new = selector.fit_transform(X, y)
print('\nScores:', selector.scores_)
print('\nSelectKBest - Original data space:\n', X[0])
print('\nSelectKBest - New data space:\n', X_new[0])

# SelectPercentile (highest scoring percentage)
selector = SelectPercentile(f_classif, percentile=50)
X_new = selector.fit_transform(X, y)
print("P-values:",selector.pvalues_)
print('\nSelectKBest - Original data space:\n', X[0])
print('\nSelectKBest - New data space:\n', X_new[0])

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
print('\nNumber of features (threshold=0.2) =', len(X_new[0]))

selector = VarianceThreshold(threshold=0.25)
X_new = selector.fit_transform(X)
print('Number of features (threshold=0.25) =', len(X_new[0]), ', indices:', selector.get_support(indices=True))


##### Wrappers #####

