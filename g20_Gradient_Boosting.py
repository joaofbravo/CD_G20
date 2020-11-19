# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
import ds_functions as ds


def GB(trnX, tstX, trnY, tstY, losses, criterion, learn_rates, n_estimators, max_depths, max_features, overfit=False):
    # yvalues = {}
    ytest_values = {}
    ytrain_values = {}
    best_par = ()
    best_tree = None
    last_best = 0

    for l in losses:
        for d in max_depths:
            for lr in learn_rates:
                for n in n_estimators:
                    for f in max_features:
                        clf = GradientBoostingClassifier(loss=l, criterion=criterion, learning_rate=lr, n_estimators=n, max_depth=d, max_features=f)
                        clf.fit(trnX, trnY)
                        # prdY = clf.predict(tstX)
                        prdY_tst = clf.predict(tstX)
                        prdY_trn = clf.predict(trnX)

                        key = (l, d, lr, n, f)
                        # yvalues[key] = metrics.accuracy_score(tstY, prdY)
                        ytest_values[key] = metrics.accuracy_score(tstY, prdY_tst)
                        ytrain_values[key] = metrics.accuracy_score(trnY, prdY_trn)
                        if ytest_values[key] > last_best:
                            best_par = key
                            best_tree = clf
                            last_best = ytest_values[key]
            # Progress flag
            print('--- done: {}, {}'.format(l, d))

    if overfit:
        return best_par, ytest_values, ytrain_values

    cols = len(max_depths)
    rows = len(losses)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)

    # Best parameters' selection & plot
    f_best = best_par[-1]
    for k1 in range(len(losses)):
        l = losses[k1]
        for k2 in range(len(max_depths)):
            d = max_depths[k2]
            values = {}
            for lr in learn_rates:
                ytest_filter = []
                for n in n_estimators:
                    ytest_filter.append(ytest_values[(l, d, lr, n, f_best)])
                values[lr] = ytest_filter

            ds.multiple_line_chart(n_estimators, values, ax=axs[k1, k2], title='Gradient Boosting (loss={}, max_depth={})'.format(l, d), xlabel='nr estimators', ylabel='accuracy', percentage=True)

    plt.show()
    print('\nBest results with loss={}, max_depth={}, learning_rate={}, {} estimators,  and {} features, with accuracy={:.3f}'.format(*best_par, last_best))
    return best_par, best_tree, last_best


def GBPerformance(tree, trnX, tstX, trnY, tstY, labels):
    prd_trn = tree.predict(trnX)
    prd_tst = tree.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)


def holdoutGB(X, y, labels, context, save_pics=False, train_size=0.7,
              losses=['deviance', 'exponential'],  # exponential == AdaBoost
              criterion='friedman_mse',  # friedman_mse, mae
              learn_rates=[0.01, 0.1, 0.5],
              n_estimators=[10, 100, 200, 300],
              max_depths=[5, 10, 25],
              max_features=[.25, 1]):

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    print('-> Holdout for '+context+':')
    best_par, best_tree, acc = GB(trnX, tstX, trnY, tstY, losses, criterion, learn_rates, n_estimators, max_depths, max_features)
    GBPerformance(best_tree, trnX, tstX, trnY, tstY,labels)
    if save_pics:
        plt.savefig('plots/'+context+'_GB_Holdout_performance.png')
    plt.show()


def crossValGB(X, y, labels, context, save_pics=False, n_splits=5,
               losses=['deviance', 'exponential'],  # exponential == AdaBoost
               criterion='friedman_mse',  # friedman_mse, mae
               learn_rates=[0.01, 0.1, 0.5],
               n_estimators=[10, 100, 200, 300],
               max_depths=[5, 10, 25],
               max_features=[.25, 1]):

    skf = StratifiedKFold(n_splits, shuffle=True)
    acc_crossval = np.empty(n_splits, dtype=dict)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]

        print('-> Fold '+str(i)+' for '+context+':')
        best_par, best_tree, acc_crossval[i] = GB(trnX, tstX, trnY, tstY, losses, criterion, learn_rates, n_estimators, max_depths, max_features)
        GBPerformance(best_tree, trnX, tstX, trnY, tstY, labels)
        if save_pics:
            plt.savefig('plots/'+context+'_GB_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        plt.show()
        i += 1

    print('\n-> Average for '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_mean = np.mean(acc_crossval)
    print('CrossVal mean score:', acc_mean)
    acc_std = np.std(acc_crossval)
    print('CrossVal std: %.4f' % acc_std)


def overfit_plot(ytest_values, ytrain_values, f_best, row_var, row_str,
                 col_var, col_str, x_var, x_str, legend_var,
                 key_order=(1,2,3,4,5)):
    cols = len(col_var)
    rows = len(row_var)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)

    for k1 in range(len(row_var)):
        rv = row_var[k1]
        for k2 in range(len(col_var)):
            cv = col_var[k2]
            values = {}
            for lv in legend_var:
                ytest_filter = []
                ytrain_filter = []
                for xv in x_var:
                    key = (rv, cv, lv, xv, f_best)  # (1, 2, 3, 4, 5)
                    key = tuple([var for _, var in sorted(zip(key_order, key))])
                    ytest_filter.append(ytest_values[key])
                    ytrain_filter.append(ytrain_values[key])

                values['test'+str(lv)] = ytest_filter
                values['train'+str(lv)] = ytrain_filter

            ds.multiple_line_chart(x_var, values, ax=axs[k1, k2], title='Gradient Boosting ({}={}, {}={})'.format(row_str, rv, col_str, cv), xlabel=x_str, ylabel='accuracy', percentage=True)
    plt.show()


def overfitting_hoGB(X, y, labels, context, save_pics=False, train_size=0.7,
                     losses=['deviance'],  # exponential == AdaBoost
                     criterion='friedman_mse',  # friedman_mse, mae
                     learn_rates=[0.01, 0.1, 0.5],
                     n_estimators=[10, 150, 300],
                     max_depths=[5, 10, 25],
                     max_features=[.25]):

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    print('-> Holdout for '+context+':')
    best_par, ytest_values, ytrain_values = GB(trnX, tstX, trnY, tstY, losses, criterion, learn_rates, n_estimators, max_depths, max_features, overfit=True)

    # Overfitting plots with best parameters
    f_best = best_par[-1]

    # testing n_estimators
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='loss',
                 col_var=max_depths, col_str='max_depth',
                 x_var=n_estimators, x_str='n_estimators',
                 legend_var=learn_rates, key_order=(1,2,3,4,5))

    # testing learn_rate
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='loss',
                 col_var=max_depths, col_str='max_depth',
                 x_var=learn_rates, x_str='learn_rates',
                 legend_var=n_estimators, key_order=(1,2,4,3,5))

    # testing max_depth
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='losses',
                 col_var=learn_rates, col_str='learn_rates',
                 x_var=max_depths, x_str='max_depths',
                 legend_var=n_estimators, key_order=(1,3,4,2,5))


def overfitting_cvGB(X, y, labels, context, save_pics=False, n_splits=5,
                     losses=['deviance'],  # exponential == AdaBoost
                     criterion='friedman_mse',  # friedman_mse, mae
                     learn_rates=[0.01, 0.1, 0.3, 0.5],
                     n_estimators=[10, 100, 200, 300],
                     max_depths=[5, 10, 25],
                     max_features=[.25, 0.5, 1]):

    skf = StratifiedKFold(n_splits, shuffle=True)
    ytest_values = np.empty(n_splits, dtype=dict)
    ytrain_values = np.empty(n_splits, dtype=dict)

    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]

        print('-> Fold '+str(i)+' for '+context+':')
        _, ytest_values[i], ytrain_values[i] = GB(trnX, tstX, trnY, tstY, losses, criterion, learn_rates, n_estimators, max_depths, max_features, overfit=True)
        i += 1

    # Averaging k-fold results
    keys = ytest_values[0].keys()
    ytest_values = {key: sum([ytest_values[i].get(key,0) for i in range(5)])
                    / float(sum([key in ytest_values[i] for i in range(5)]))
                    for key in keys}
    ytrain_values = {key: sum([ytrain_values[i].get(key,0) for i in range(5)])
                     / float(sum([key in ytrain_values[i] for i in range(5)]))
                     for key in keys}

    # Getting best result's parameters
    best_par = max([(value, key) for key, value in ytest_values.items()])[1]
    print('\nBest parameters:', best_par)
    f_best = best_par[-1]

    # Overfitting plots with best parameters
    # testing n_estimators
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='loss',
                 col_var=max_depths, col_str='max_depth',
                 x_var=n_estimators, x_str='n_estimators',
                 legend_var=learn_rates, key_order=(1,2,3,4,5))

    # testing learn_rate
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='loss',
                 col_var=max_depths, col_str='max_depth',
                 x_var=learn_rates, x_str='learn_rates',
                 legend_var=n_estimators, key_order=(1,2,4,3,5))

    # testing max_depth
    overfit_plot(ytest_values, ytrain_values, f_best,
                 row_var=losses, row_str='losses',
                 col_var=learn_rates, col_str='learn_rates',
                 x_var=max_depths, x_str='max_depths',
                 legend_var=n_estimators, key_order=(1,3,4,2,5))
