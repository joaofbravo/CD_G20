# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import ds_functions as ds

def RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context):
    best = ('', 0, 0)
    last_best = 0
    best_tree = None
    
    cols = len(max_depths)
    for o in range(len(criteria)):
        plt.figure()
        fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
        c = criteria[o]
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            for f in max_features:
                yvalues = []
                for n in n_estimators:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f,criterion=c)
                    rf.fit(trnX, trnY)
                    prdY = rf.predict(tstX)
                    yvalues.append(metrics.accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, f, n, c)
                        last_best = yvalues[-1]
                        best_tree = rf
        
                values[f] = yvalues
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d and criterion %s'%(d,c), xlabel='nr estimators', ylabel='accuracy', percentage=True)
        plt.show()
    
    print('Best results with depth=%d, %1.2f features, %s criterion  and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[3], best[2], last_best))
    return best, best_tree,last_best

def RFPerformance(tree,trnX, tstX, trnY, tstY,labels):
    prd_trn = tree.predict(trnX)
    prd_tst = tree.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    

def holdoutRF(X,y,labels,context,save_pics=False, train_size=0.7,
              n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300], max_depths = [5, 10, 25],
              max_features = [.1, .3, .5, .7, .9, 1],criteria = ['entropy', 'gini']):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    print('-> Holdout for '+context+':')
    best, best_tree, acc = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context)
    RFPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
    if save_pics:
        plt.savefig('plots/'+context+'_RF_Holdout_performance.png')
    plt.show()

def crossValRF(X,y,labels,context,save_pics=False, n_splits = 5,
              n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300], max_depths = [5, 10, 25],
              max_features = [.1, .3, .5, .7, .9, 1],criteria = ['entropy', 'gini']):
    skf = StratifiedKFold(n_splits, shuffle=True)
    acc_crossval = np.empty(n_splits, dtype=dict)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        
        print('-> Fold '+str(i)+' for '+context+':')
        best, best_tree, acc_crossval[i] = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context)
        RFPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
        if save_pics:
            plt.savefig('plots/'+context+'_RF_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        plt.show()
        i+=1
    
    print('\n-> Average for '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_mean = np.mean(acc_crossval)
    print('CrossVal mean score:', acc_mean)
    acc_std = np.std(acc_crossval)
    print('CrossVal std: %.4f' % acc_std)


# part 1


# part 2
