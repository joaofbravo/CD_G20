# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold

from sklearn.model_selection import train_test_split
from Toxic_NaiveBayes import NaiveBayesModel, NaiveBayesEstimation


def NaiveBayesTest(X, y, savename=None):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=69)

    NaiveBayesModel(trnX, tstX, trnY, tstY)
    if savename is not None:
        plt.savefig('plots/Toxic_FeatureSelection_Filter_'+savename+'_NB.png')
    
    _, score = NaiveBayesEstimation(trnX, tstX, trnY, tstY)
    if savename is not None:
        plt.savefig('plots/Toxic_FetureSelection_Filter_'+savename+'_NBestimators.png')
    
    print('NB holdout score:', score)
    return score


##### get data
data: pd.DataFrame = pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep =';')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
# labels = pd.unique(y)

np.set_printoptions(precision=4, suppress=True)    


##### original results
print('X shape:', X.shape)
# print('\nOriginal data space:\n', X[0])
NaiveBayesTest(X, y)


##### Filters - Classification #####

##### chi2
chi, pval = chi2(X, y)
# print('\nChi2 test scores:\n', chi)
# print('Chi2 p-values:\n', pval)

alpha = 1e-10
print('\nalpha =', alpha)

### - SelectFpr (false positive rate)
selector = SelectFpr(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFpr - Number of features:', len(X_new[0]))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

NaiveBayesTest(X_new, y, 'chi2_FPR_alpha='+str(alpha))

### - SelectFdr (false discovery rate)
selector = SelectFdr(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFdr - Number of features:', len(X_new[0]))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

NaiveBayesTest(X_new, y, 'chi2_FDR_alpha='+str(alpha))

### - SelectFwe (family wise error) 
selector = SelectFwe(chi2, alpha=alpha)
X_new = selector.fit_transform(X, y)
print('SelectFwe - Number of features:', len(X_new[0]))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

NaiveBayesTest(X_new, y, 'chi2_FWE_alpha='+str(alpha))


##### mutual_info_classif - SelectKBest (highest scoring number)
k = 100
selector = SelectKBest(mutual_info_classif, k)
X_new = selector.fit_transform(X, y)
# print('\nSelectKBest scores:\n', selector.scores_)
# print('SelectKBest (k = {}) - Selected indices: {}'.format(k, selector.get_support(indices=True)))
# print('New data space:', X_new[0])

print('\nSelectKBest (k = {})'.format(k))
NaiveBayesTest(X_new, y, 'MI_kBest='+str(k))


##### ANOVA (f_classif) - SelectPercentile (highest scoring percentage)
percentile = 5
selector = SelectPercentile(f_classif, percentile=percentile)
X_new = selector.fit_transform(X, y)
# print('\nSelectPercentile p-values:\n', selector.pvalues_)
print('\nSelectPercentile ({}%) - Number of features: {}'.format(percentile, len(X_new[0])))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

NaiveBayesTest(X_new, y, 'ANOVA_percent='+str(percentile)+'%')


##### Filters - Unsupervised #####

##### variance
selector = VarianceThreshold()
selector.fit(X)
# print('\nFeature variance:\n', selector.variances_)

threshold = 0.2
selector = VarianceThreshold(threshold)
X_new = selector.fit_transform(X)
print('Variance (threshold={}) - Number of features: {}'.format(threshold, len(X_new[0])))
# print('Selected indices:', selector.get_support(indices=True))
# print('New data space:', X_new[0])

NaiveBayesTest(X_new, y, 'variance_trsh='+str(threshold))
