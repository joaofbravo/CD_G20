# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
import ds_functions as ds


def GB(trnX, tstX, trnY, tstY, losses, criterions, learn_rates, n_estimators, max_depths, max_features):
    yvalues = {}
    best_par = ()
    best_tree = None
    last_best = 0
    
    cols = len(losses)
    rows = len(criterions)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)
    
    for l in losses:            #---------------- 2
        for crit in criterions:    #------------- 2
            for lr in learn_rates:     #--------- 5
                for n in n_estimators:    #------ 5
                    for d in max_depths:     #--- 3
                        for f in max_features:  # 4
                            clf = GradientBoostingClassifier(loss=l, criterion=crit, learning_rate=lr, n_estimators=n, max_depth=d, max_features=f)
                            clf.fit(trnX, trnY)
                            prdY = clf.predict(tstX)
                            
                            key = (l, crit, lr, n, d, f)
                            yvalues[key] = metrics.accuracy_score(tstY, prdY)
                            if yvalues[key] > last_best:
                                best_par = key
                                best_tree = clf
                                last_best = yvalues[key]
            # Progress flag
            print('--- done: {}, {}'.format(l, crit))
    
    # Best parameters' selection & plot
    d_best, f_best = best_par[4:]
    for k1 in range(len(losses)):
        l = losses[k1]
        for k2 in range(len(criterions)):
            crit = criterions[k2]
            values = {}
            for lr in learn_rates:
                yfiltered = []
                for n in n_estimators:
                    yfiltered.append(yvalues[(l, crit, lr, n, d_best, f_best)])
                values[lr] = yfiltered
            ds.multiple_line_chart(n_estimators, values, ax=axs[k1, k2], title='Gradient Boosting (loss={}, crit={})'.format(l, crit), xlabel='nr estimators', ylabel='accuracy', percentage=True)
    
    plt.show()
    print('\nBest results with loss={}, criterion={}, learning_rate={}, {} estimators, depth={} and {} features, with accuracy={:.3f}'.format(*best_par, last_best))
    return best_par, best_tree, last_best


def GBPerformance(tree, trnX, tstX, trnY, tstY, labels):
    prd_trn = tree.predict(trnX)
    prd_tst = tree.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)


def holdoutGB(X, y, labels, context, save_pics=False, train_size=0.7,
               losses=['deviance', 'exponential'], # exponential == AdaBoost
               criterions=['friedman_mse', 'mae'],
               learn_rates=[0.01, 0.1, 0.5],
               n_estimators = [10, 100, 200, 300],
               max_depths=[5, 10, 25],
               max_features=[.25, .5, 1]):

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    print('-> Holdout for '+context+':')
    best_par, best_tree, acc = GB(trnX, tstX, trnY, tstY, losses, criterions, learn_rates, n_estimators, max_depths, max_features)
    GBPerformance(best_tree, trnX, tstX, trnY, tstY,labels)
    if save_pics:
        plt.savefig('plots/'+context+'_GB_Holdout_performance.png')
    plt.show()


def crossValGB(X, y, labels, context, save_pics=False, n_splits = 5,
               losses=['deviance', 'exponential'], # exponential == AdaBoost
               criterions=['friedman_mse', 'mae'],
               learn_rates=[0.01, 0.1, 0.5],
               n_estimators = [10, 100, 200, 300],
               max_depths=[5, 10, 25],
               max_features=[.25, .5, 1]):
    
    skf = StratifiedKFold(n_splits, shuffle=True)
    acc_crossval = np.empty(n_splits, dtype=dict)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        
        print('-> Fold '+str(i)+' for '+context+':')
        best_par, best_tree, acc_crossval[i] = GB(trnX, tstX, trnY, tstY, losses, criterions, learn_rates, n_estimators, max_depths, max_features)
        GBPerformance(best_tree, trnX, tstX, trnY, tstY, labels)
        if save_pics:
            plt.savefig('plots/'+context+'_GB_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        plt.show()
        i+=1
    
    print('\n-> Average for '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_mean = np.mean(acc_crossval)
    print('CrossVal mean score:', acc_mean)
    acc_std = np.std(acc_crossval)
    print('CrossVal std: %.4f' % acc_std)
