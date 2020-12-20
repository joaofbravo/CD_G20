# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import ds_functions as ds
import g20_functions as g20

def RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context,output=False):
    best = ('', 0, 0)
    last_best = 0
    best_tree = None
    
    cols = len(max_depths)
    output_data = {}
    for o in range(len(criteria)):
        plt.figure()
        fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
        c = criteria[o]
        output_data[c] = {}
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            output_data[c][d] = {}
            for f in max_features:
                yvalues = []
                output_data[c][d][f] = {}
                for n in n_estimators:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f,criterion=c, random_state=42)
                    rf.fit(trnX, trnY)
                    prd_trnY = rf.predict(trnX)
                    prdY = rf.predict(tstX)
                    output_data[c][d][f][n] = {"train":metrics.accuracy_score(trnY, prd_trnY),"test":metrics.accuracy_score(tstY, prdY)}
                    yvalues.append(metrics.accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, f, n, c)
                        last_best = yvalues[-1]
                        best_tree = rf
        
                values[f] = yvalues
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d and criterion %s'%(d,c), xlabel='nr estimators', ylabel='accuracy', percentage=True)
        plt.show()
    
    print('Best results with depth=%d, %1.2f features, %s criterion  and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[3], best[2], last_best))
    if output:
        return best, best_tree,last_best, output_data
    else:
        return best, best_tree,last_best

def RFPerformance(tree,trnX, tstX, trnY, tstY,labels):
    prd_trn = tree.predict(trnX)
    prd_tst = tree.predict(tstX)
    return prd_trn, prd_tst
    
    

def holdoutRF(X,y,labels,context,save_pics=False, train_size=0.7,output=False,
              n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300], max_depths = [5, 10, 25],
              max_features = [.1, .3, .5, .7, .9, 1],criteria = ['entropy', 'gini']):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    print('-> Holdout for '+context+':')
    if output:
        best, best_tree, acc, output_values = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context,output=True)
    else:
        best, best_tree, acc = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context)
    prd_trn, prd_tst = RFPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    if save_pics:
        plt.savefig('plots/'+context+'_RF_Holdout_performance.png')
    plt.show()
    if output:
        return output_values

def crossValRF(X,y,labels,context,save_pics=False, n_splits = 5,output=False,
              n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300], max_depths = [5, 10, 25],
              max_features = [.1, .3, .5, .7, .9, 1],criteria = ['entropy', 'gini']):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    acc_crossval = np.empty(n_splits, dtype=dict)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    y_train_list = []
    prd_trn_list = []
    y_test_list  = []
    prd_tst_list = []
    output_values = []
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        
        print('-> Fold '+str(i)+' for '+context+':')
        if output:
            best, best_tree, acc_crossval[i],output_value = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context,output=True)
            output_values.append(output_value)
        else:
            best, best_tree, acc_crossval[i] = RF(trnX, tstX, trnY, tstY,criteria,max_depths,n_estimators,max_features,context)
        prd_trn, prd_tst = RFPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
        # ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        # if save_pics:
        #     plt.savefig('plots/'+context+'_RF_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        # plt.show()
        y_train_list.append(trnY)
        prd_trn_list.append(prd_trn)
        y_test_list.append(tstY)
        prd_tst_list.append(prd_tst)
        i+=1
        
    g20.plot_avg_evaluation_results(labels, y_train_list, prd_trn_list, y_test_list, prd_tst_list)
    if save_pics:
        plt.savefig('plots/'+context+'_RF_CrossVal'+str(n_splits)+'_average_performance.png')
    plt.show()
    
    print('\n-> Average for '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_mean = np.mean(acc_crossval)
    print('CrossVal mean score:', acc_mean)
    acc_std = np.std(acc_crossval)
    print('CrossVal std: %.4f' % acc_std)
    if output:
        return output_values