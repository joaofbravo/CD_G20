# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold

# from sklearn.model_selection import train_test_split, StratifiedKFold
# import sklearn.metrics as metrics
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from Heart_NaiveBayes import NaiveBayesModel, NaiveBayesEstimation
# import ds_functions as ds

np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)

# get data
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

print('X shape:', X.shape)
print('\nOriginal data space:\n', X[0])


##### Filters - Classification #####

# chi2
chi, pval = chi2(X, y)
print('\nChi2 test scores:\n', chi)
print('Chi2 p-values:\n', pval)

alpha = 0.01
print('\nalpha =', alpha)

# - SelectFpr (false positive rate)
selector = SelectFpr(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFpr - Number of features:', len(X_new[0]))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

# - SelectFdr (false discovery rate)
selector = SelectFdr(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFdr - Number of features:', len(X_new[0]))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

# - SelectFwe (family wise error) 
selector = SelectFwe(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFwe - Number of features:', len(X_new[0]))
print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])


# mutual_info_classif - SelectKBest (highest scoring number)
k = 2
selector = SelectKBest(mutual_info_classif, k)
X_new = selector.fit_transform(X, y)
print('\nSelectKBest scores:\n', selector.scores_)
print('SelectKBest (k = {}) - Selected indices: {}'.format(k, selector.get_support(indices=True)))
# print('New data space:', X_new[0])


# ANOVA (f_classif) - SelectPercentile (highest scoring percentage)
percentile = 50
selector = SelectPercentile(f_classif, percentile=percentile)
X_new = selector.fit_transform(X, y)
print('\nSelectPercentile p-values:\n', selector.pvalues_)
print('SelectPercentile ({}%) - Number of features: {}'.format(percentile, len(X_new[0])))
print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])


##### Filters - Unsupervised #####

# variance
selector = VarianceThreshold()
selector.fit(X)
print('\nFeature variance:\n', selector.variances_)

threshold = 0.2
selector = VarianceThreshold(threshold)
X_new = selector.fit_transform(X)
print('Variance (threshold={}) - Number of features: {}'.format(threshold, len(X_new[0])))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

threshold = 0.25
selector = VarianceThreshold(threshold)
X_new = selector.fit_transform(X)
print('Variance (threshold={}) - Number of features: {}'.format(threshold, len(X_new[0])))
print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])


##### Wrappers #####

