# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import ds_functions as ds
from Heart_Scaling import scale_heart


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


# get data (z-score)
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
norm_data_zscore, norm_data_minmax = scale_heart(data)

norm_data_zscore.pop('DEATH_EVENT')
y: np.ndarray = data['DEATH_EVENT']
X: np.ndarray = norm_data_zscore.values
labels = pd.unique(y)


# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']

print('-> Holdout:')
values = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)
plt.figure()
ds.multiple_line_chart(nvalues, values, title='Heart KNN zscore overfitting holdout', xlabel='n', ylabel='accuracy', percentage=True)
plt.savefig('plots/Heart_KNN_zscore_overfitting_holdout.png')



# k-fold cross validation (StratifiedKFold)
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
# skf.get_n_splits(X, y)

print('\n-> 5-fold CrossVal:')
acc_crossval = np.empty(n_splits, dtype=dict)
i = 0
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    acc_crossval[i] = KNNModel(X_train, X_test, y_train, y_test, nvalues, dist)
    plt.figure()
    ds.multiple_line_chart(nvalues, values, title='Heart KNN zscore overfitting CrossVal5_#'+str(i), xlabel='n', ylabel='accuracy', percentage=True)
    plt.savefig('plots/Heart_KNN_zscore_overfitting_CrossVal5_#'+str(i)+'.png')
    i+=1


# CrossVal: find mean accuracies
dists = []
for d in dist:
    dists.append("test "+d)
    dists.append("train "+d)
acc_mean = {d: [] for d in dists}
for i, d in enumerate(dists):
    acc_mean[d] = np.mean([acc_crossval[j][d] for j in range(n_splits)], axis=0)
    
# score_std = np.std([score_crossval[i][0] for i in range(n_splits)])
# print('CrossVal mean score:', score_mean)
# print('CrossVal std: %.4f' % score_std)

plt.figure()
ds.multiple_line_chart(nvalues, acc_mean, title='Heart KNN zscore overfitting CrossVal5 mean', xlabel='n', ylabel='accuracy', percentage=True)
plt.savefig('plots/Heart_KNN_zscore_overfitting_CrossVal5_mean.png')

