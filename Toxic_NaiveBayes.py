# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ds_functions as ds


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


# get data
data: pd.DataFrame = pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep =';')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)


# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf_holdout = NaiveBayesModel(trnX, tstX, trnY, tstY)
plt.savefig('plots/Toxic_NaiveBayes_holdout.png')
_, score_holdout = NaiveBayesEstimation(trnX, tstX, trnY, tstY)
plt.savefig('plots/Toxic_NaiveBayes_holdout_estimators.png')

print('Holdout score:', score_holdout)
