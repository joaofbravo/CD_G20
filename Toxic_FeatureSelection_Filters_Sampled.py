# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold

from sklearn.model_selection import train_test_split
from Toxic_NaiveBayes import NaiveBayesModel, NaiveBayesEstimation


def NaiveBayesTest(X, y, nsamples=10):
    score = np.empty((nsamples, 3))

    for i in range(nsamples):
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
        NaiveBayesModel(trnX, tstX, trnY, tstY)
        _, score[i] = NaiveBayesEstimation(trnX, tstX, trnY, tstY)
    
    print('Mean:', np.mean(score, axis=0))
    # print('Best:', np.max(score, axis=0))
    print('Std:', np.std(score, axis=0))
    return score


##### get data
data: pd.DataFrame = pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep =';')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
# labels = pd.unique(y)

# np.set_printoptions(precision=4, suppress=True)
nsamples = 100

##### original results
print('X shape:', X.shape)
# print('\nOriginal data space:\n', X[0])
NaiveBayesTest(X, y, nsamples)


# ##### Filters - Classification #####

##### chi2
# BETTER PERFORMANCE FOR alpha < 0.0001, DECREASES FOR BIGGER alpha

chi, pval = chi2(X, y)
# print('\nChi2 test scores:\n', chi)
# print('Chi2 p-values:\n', pval)

for alpha in (0.00001, 0.00005, 0.0001, 0.0005):
    print('\nalpha =', alpha)
    
    ### - SelectFpr (false positive rate)
    selector = SelectFpr(chi2, alpha=alpha)
    X_new = selector.fit_transform(X, y)
    print('SelectFpr - Number of features:', len(X_new[0]))
    # print('Selected indices:', selector.get_support(indices=True))
    # print('New data space:', X_new[0])
    
    NaiveBayesTest(X_new, y, nsamples)
    
    ### - SelectFdr (false discovery rate)
    selector = SelectFdr(chi2, alpha=alpha)
    X_new = selector.fit_transform(X, y)
    print('SelectFdr - Number of features:', len(X_new[0]))
    # print('Selected indices:', selector.get_support(indices=True))
    # print('New data space:', X_new[0])
    
    NaiveBayesTest(X_new, y, nsamples)
    
    ### - SelectFwe (family wise error) 
    selector = SelectFwe(chi2, alpha=alpha)
    X_new = selector.fit_transform(X, y)
    print('SelectFwe - Number of features:', len(X_new[0]))
    # print('Selected indices:', selector.get_support(indices=True))
    # print('New data space:', X_new[0])
    
    NaiveBayesTest(X_new, y, nsamples)


##### mutual_info_classif - SelectKBest (highest scoring number)
# BETTER PERFORMANCE FOR k<100, DECREASES FOR HIGHER k

for k in (10, 20, 50, 100, 200, 300):
    selector = SelectKBest(mutual_info_classif, k)
    X_new = selector.fit_transform(X, y)
    # print('\nSelectKBest scores:\n', selector.scores_)
    # print('\nSelectKBest (k = {}) - Selected indices: {}'.format(k, selector.get_support(indices=True)))
    # print('New data space:', X_new[0])
    
    print('\nSelectKBest (k = {})'.format(k))
    NaiveBayesTest(X_new, y, nsamples)


##### ANOVA (f_classif) - SelectPercentile (highest scoring percentage)
# BETTER PERFORMANCE FOR percentile < 5, DECREASES AS percentile INCREASES

for percentile in (3, 5, 7, 10, 15, 20):
    selector = SelectPercentile(f_classif, percentile=percentile)
    X_new = selector.fit_transform(X, y)
    # print('\nSelectPercentile p-values:\n', selector.pvalues_)
    print('\nSelectPercentile ({}%) - Number of features: {}'.format(percentile, len(X_new[0])))
    # print('Selected indices:', selector.get_support(indices=True))
    # print('New data space:', X_new[0])
    
    NaiveBayesTest(X_new, y, nsamples)


##### Filters - Unsupervised #####

##### variance
# NO FILTER FOR threshold < 0.01
# SAME PERFORMANCE AS ORIGINAL FOR threshold < 0.01, DECREASES FOR BIGGER threshold
# NO FEATURES IF threshold > 0.25

selector = VarianceThreshold()
selector.fit(X)
# print('\nFeature variance:\n', selector.variances_)

for threshold in (0.1, 0.15, 0.2):
    selector = VarianceThreshold(threshold)
    X_new = selector.fit_transform(X)
    print('\nVariance (threshold={}) - Number of features: {}'.format(threshold, len(X_new[0])))
    # print('Selected indices:', selector.get_support(indices=True))
    # print('New data space:', X_new[0])
    
    NaiveBayesTest(X_new, y, nsamples)
