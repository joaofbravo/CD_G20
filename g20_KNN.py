# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import ds_functions as ds
import g20_functions as g20
import seaborn as sns


def KNNModel(X_train, X_test, y_train, y_test, nvalues, dist):
    test_values = {}
    train_values = {}
    best = (0, '')
    last_best = 0

    for d in dist:
        y_train_values = []
        y_test_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)

            prd_test = knn.predict(X_test)
            y_test_values.append(metrics.accuracy_score(y_test, prd_test))
            prd_train = knn.predict(X_train)
            y_train_values.append(metrics.accuracy_score(y_train, prd_train))

            if y_test_values[-1] > last_best:
                best = (n, d)
                last_best = y_test_values[-1]
        test_values[d] = y_test_values
        train_values[d] = y_train_values

    print('Best results: %d neighbors, %s metric, %.2f accuracy' % (best[0], best[1], last_best))
    return best, last_best, test_values, train_values


def KNNPerformance(X_train, X_test, y_train, y_test, best, labels):
    clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(X_train, y_train)
    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)
    return prd_trn, prd_tst


# hold-out (train_test_split)
def holdoutKNN(X, y, labels, context, save_pics=False, train_size=0.7,
               nvalues=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
               dist=['manhattan', 'euclidean', 'chebyshev']):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
    print('-> Holdout for '+context+':')
    best, _, test_values, _ = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)

    title = 'KNN variants with holdout for '+context
    plt.figure()
    ds.multiple_line_chart(nvalues, test_values, title=title, xlabel='n', ylabel='accuracy', percentage=True)

    if save_pics:
        plt.savefig('plots/'+context+'_KNN_holdout.png')
    plt.show()
    prd_trn, prd_tst = KNNPerformance(trnX, tstX, trnY, trnY, best,labels)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    if save_pics:
        plt.savefig('plots/'+context+'_KNN_holdout_performance.png')
    plt.show()


# k-fold cross validation (StratifiedKFold)
def crossValKNN(X, y, labels, context, save_pics=False, n_splits=5,
                nvalues=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                dist=['manhattan', 'euclidean', 'chebyshev']):

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_crossval = np.empty(n_splits, dtype=dict)
    i = 0
    y_train_list = []
    prd_trn_list = []
    y_test_list  = []
    prd_tst_list = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best, _, acc_crossval[i], _ = KNNModel(X_train, X_test, y_train, y_test, nvalues, dist)

        title = 'KNN variants for fold '+str(i)+', '+context
        # plt.figure()
        # ds.multiple_line_chart(nvalues, acc_crossval[i], title=title, xlabel='n', ylabel='accuracy', percentage=True)
        # if save_pics:
        #     plt.savefig('plots/'+context+'_KNN_CrossVal'+str(n_splits)+'_#'+str(i)+'.png')
        # plt.show()
        prd_trn, prd_tst = KNNPerformance(X_train, X_test, y_train, y_test, best, labels)
        y_train_list.append(y_train)
        prd_trn_list.append(prd_trn)
        y_test_list.append(y_test)
        prd_tst_list.append(prd_tst)
        
        # ds.plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
        # if save_pics:
        #     plt.savefig('plots/'+context+'_KNN_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        # plt.show()
        i += 1
   
    g20.plot_avg_evaluation_results(labels, y_train_list, prd_trn_list, y_test_list, prd_tst_list)
    if save_pics:
        plt.savefig('plots/'+context+'_KNN_CrossVal'+str(n_splits)+'_average_performance.png')
    plt.show()
    # CrossVal: find mean accuracies
    acc_mean = {d: [] for d in dist}
    acc_std = {d: [] for d in dist}
    acc_95 = {d: [] for d in dist}
    for i, d in enumerate(dist):
        acc_mean[d] = np.mean([acc_crossval[j][d] for j in range(n_splits)], axis=0)
        acc_std[d] = np.std([acc_crossval[j][d] for j in range(n_splits)], axis=0)
        acc_95[d] = acc_std[d]*0.95/n_splits
    print(acc_95)
    plt.figure()
    title = 'KNN variants with '+str(n_splits)+'fold CrossVal for '+context+' (mean)'
    ds.multiple_line_chart(nvalues, acc_mean, title=title, xlabel='n', ylabel='accuracy', percentage=True)
    if save_pics:
        plt.savefig('plots/'+context+'_KNN_CrossVal'+str(n_splits)+'_mean.png')
    plt.show()
    
   
    # CrossVal: find max accuracy
    acc_best = np.max([acc_mean[i] for i in dist])
    acc_bestarg = np.argmax([acc_mean[i] for i in dist])
    n_best = nvalues[acc_bestarg % len(nvalues)]
    dist_best = dist[acc_bestarg // len(nvalues)]
    print('Best mean results: %d neighbors, %s metric, %.2f accuracy' % (n_best, dist_best, acc_best))

def overfitting_hoKNN(X, y, labels, train_size=0.7,
               nvalues=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
               dist=['manhattan', 'euclidean', 'chebyshev']):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
    best, _, test_values, train_values = KNNModel(trnX, tstX, trnY, tstY, nvalues, dist)
    
    title = 'KNN variants with holdout for Toxic'
    plt.figure()
    ds.multiple_line_chart(nvalues, test_values, title=title, xlabel='n', ylabel='accuracy', percentage=True)
    plt.show()
    prd_trn, prd_tst = KNNPerformance(trnX, tstX, trnY, trnY, best,labels)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    plt.show()

    for i, d in enumerate(dist):
        acc_mean={}
        acc_mean["test"] = test_values[d]
        acc_mean["train"] = train_values[d]
        plt.figure()
        title = 'KNN overfitting with '+d+' distance (mean)'
        ds.multiple_line_chart(nvalues, acc_mean, title=title, xlabel='n', ylabel='accuracy', percentage=True)
        plt.show()


def overfitting_cvKNN(X, y, labels,n_splits = 5,
                nvalues=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                dist=['manhattan', 'euclidean', 'chebyshev']):
    
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    # skf.get_n_splits(X, y)
    
    print('\n-> 5-fold CrossVal:')
    acc_test = np.empty(n_splits, dtype=dict)
    acc_train = np.empty(n_splits, dtype=dict)
    i = 0
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        _, _, acc_test[i], acc_train[i] = KNNModel(X_train, X_test, y_train, y_test, nvalues, dist)
        i+=1
    
    acc_mean = {}
    acc_std = {}
    acc_95 = {}
    for i, d in enumerate(dist):
        acc_mean[d]={}
        acc_mean[d]["test"] = np.mean([acc_test[j][d] for j in range(n_splits)], axis=0)
        acc_mean[d]["train"] = np.mean([acc_train[j][d] for j in range(n_splits)], axis=0)
        plt.figure()
        title = 'KNN overfitting with '+d+' distance (mean)'
        ds.multiple_line_chart(nvalues, acc_mean[d], title=title, xlabel='n', ylabel='accuracy', percentage=True)
        plt.show()
        acc_std[d+" test"] = np.std([acc_test[j][d] for j in range(n_splits)], axis=0)
        acc_std[d+" train"] = np.std([acc_train[j][d] for j in range(n_splits)], axis=0)
        acc_95[d+" test"] = acc_std[d+" test"]*0.95/n_splits
        acc_95[d+" train"] = acc_std[d+" train"]*0.95/n_splits
    print(acc_95)
    