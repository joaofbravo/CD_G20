# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ds_functions as ds
from ScalingHeart import scale_heart


# Naive Bayes
def NaiveBayesModel(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)
    ds.plot_evaluation_results(pd.unique(y), y_train, prd_trn, y_test, prd_tst)
    return clf


# Estimation
def NaiveBayesEstimation(X_train, X_test, y_train, y_test):
    estimators = {'GaussianNB': GaussianNB(),
                  'MultinomialNB': MultinomialNB(),
                  'BernoulyNB': BernoulliNB()}
    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(X_train, y_train)
        prdY = estimators[clf].predict(X_test)
        yvalues.append(metrics.accuracy_score(y_test, prdY))
    
    plt.figure()
    ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    # plt.show()
    return xvalues, np.asarray(yvalues)


# get data (minmax)
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
norm_data_zscore, norm_data_minmax = scale_heart(data)

norm_data_minmax.pop('DEATH_EVENT')
y: np.ndarray = data['DEATH_EVENT']
X: np.ndarray = norm_data_minmax.values
labels = pd.unique(y)

# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf_holdout = NaiveBayesModel(trnX, tstX, trnY, tstY)
plt.savefig('plots/Heart_NaiveBayes_minmax_holdout.png')
_, score_holdout = NaiveBayesEstimation(trnX, tstX, trnY, tstY)
plt.savefig('plots/Heart_NaiveBayes_minmax_holdout_estimators.png')

print('Holdout score:', score_holdout)


# k-fold cross validation (StratifiedKFold)
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
skf.get_n_splits(X, y)

clf_crossval = np.empty(n_splits, dtype=GaussianNB)
score_crossval = np.empty(n_splits, dtype=np.ndarray)
i = 0
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf_crossval[i] = NaiveBayesModel(X_train, X_test, y_train, y_test)
    plt.savefig('plots/Heart_NaiveBayes_minmax_CrossVal5_#'+str(i)+'.png')
    xvalues, score_crossval[i] = NaiveBayesEstimation(X_train, X_test, y_train, y_test)
    plt.savefig('plots/Heart_NaiveBayes_minmax_CrossVal5_#'+str(i)+'_estimators.png')
    i+=1


# CrossVal: find max, mean, std
score_best = np.max([score_crossval[i] for i in range(n_splits)])
score_bestarg = np.argmax([score_crossval[i] for i in range(n_splits)])

print('CrossVal best estimator: %s with score %.2f' % (xvalues[score_bestarg % len(xvalues)], score_best))

score_mean = np.mean(score_crossval)
score_std = np.std([score_crossval[i][0] for i in range(n_splits)])
print('CrossVal mean score:', score_mean)
print('CrossVal std: %.4f' % score_std)

plt.figure()
ds.bar_chart(xvalues, score_mean, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
plt.savefig('plots/Heart_NaiveBayes_minmax_CrossVal5_mean_estimators.png')
plt.show()
