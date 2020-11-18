# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import ds_functions as ds


# KNN
def KNNModel(X_train, X_test, y_train, y_test, nvalues, dist):
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)
            prdY = knn.predict(X_test)
            yvalues.append(metrics.accuracy_score(y_test, prdY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
        values[d] = yvalues

    plt.figure()
    ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    # plt.show()

    print('Best results: %d neighbors, %s metric, %.2f accuracy' % (best[0], best[1], last_best))
    return best, last_best, values


# Performance
def KNNPerformance(X_train, X_test, y_train, y_test, best):
    clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(X_train, y_train)
    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)
    ds.plot_evaluation_results(pd.unique(y), y_train, prd_trn, y_test, prd_tst)
    # plt.show()    


# get data
data: pd.DataFrame = pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep =';')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)


# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']

print('-> Holdout:')
best, _, _ = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)
plt.savefig('plots/Toxic_KNN_holdout.png')
KNNPerformance(trnX, tstX, trnY, tstY, best)
plt.savefig('plots/Toxic_KNN_holdout_performance.png')
