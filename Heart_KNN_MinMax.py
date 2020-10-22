# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import ds_functions as ds
from ScalingHeart import scale_heart


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


# get data (minmax)
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
norm_data_zscore, norm_data_minmax = scale_heart(data)

norm_data_minmax.pop('DEATH_EVENT')
y: np.ndarray = data['DEATH_EVENT']
X: np.ndarray = norm_data_minmax.values
labels = pd.unique(y)


# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']

print('-> Holdout:')
best, _, _ = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)
plt.savefig('plots/Heart_KNN_minmax_holdout.png')
KNNPerformance(trnX, tstX, trnY, tstY, best)
plt.savefig('plots/Heart_KNN_minmax_holdout_performance.png')


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
    
    best, _, acc_crossval[i] = KNNModel(X_train, X_test, y_train, y_test, nvalues, dist)
    plt.savefig('plots/Heart_KNN_minmax_CrossVal5_#'+str(i)+'.png')
    KNNPerformance(X_train, X_test, y_train, y_test, best)
    plt.savefig('plots/Heart_KNN_minmax_CrossVal5_#'+str(i)+'_performance.png')
    i+=1


# CrossVal: find mean accuracies
acc_mean = {d: [] for d in dist}
for i, d in enumerate(dist):
    acc_mean[d] = np.mean([acc_crossval[j][d] for j in range(n_splits)], axis=0)
    
# score_std = np.std([score_crossval[i][0] for i in range(n_splits)])
# print('CrossVal mean score:', score_mean)
# print('CrossVal std: %.4f' % score_std)

plt.figure()
ds.multiple_line_chart(nvalues, acc_mean, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
plt.savefig('plots/Heart_KNN_minmax_CrossVal5_mean.png')

# CrossVal: find max accuracy
acc_best = np.max([acc_mean[i] for i in dist])
acc_bestarg = np.argmax([acc_mean[i] for i in dist])
n_best = nvalues[acc_bestarg % len(nvalues)]
dist_best = dist[acc_bestarg // len(nvalues)]
print('Best mean results: %d neighbors, %s metric, %.2f accuracy' % (n_best, dist_best, acc_best))

i = 0
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    KNNPerformance(X_train, X_test, y_train, y_test, (n_best, dist_best))
    plt.savefig('plots/Heart_KNN_minmax_CrossVal5_best'+str(i)+'_performance.png')
    i+=1
