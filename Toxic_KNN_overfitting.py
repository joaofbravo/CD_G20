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
        train_values = []
        test_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)
            prd_test = knn.predict(X_test)
            test_values.append(metrics.accuracy_score(y_test, prd_test))
            prd_train = knn.predict(X_train)
            train_values.append(metrics.accuracy_score(y_train, prd_train))
        values["test "+d] = test_values
        values["train "+d] = train_values
    return values

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
values = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)
plt.figure()
ds.multiple_line_chart(nvalues, values, title='Toxic_KNN_overfitting_holdout', xlabel='n', ylabel='accuracy', percentage=True)
plt.savefig('plots/Toxic_KNN_overfitting_holdout.png')

