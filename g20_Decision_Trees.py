# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
import ds_functions as ds
from sklearn.tree import export_graphviz
import pydot
import g20_functions as g20

def DT(trnX, tstX, trnY, tstY,criteria,max_depths,min_impurity_decrease,context):
    best = ('',  0, 0.0)
    last_best = 0
    best_tree = None
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp, random_state=42)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_tree = tree
    
            values[d] = yvalues
        ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f, xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    
    plt.show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
    return best, best_tree,last_best

def drawDT(tree,name,save_pics):
    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
   
    (graph,) = pydot.graph_from_dot_file('dtree.dot')
    graph.write_png('dtree.png')
    
    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')
    if save_pics:
        plt.savefig('plots/'+name+'.png')
    plt.show()
    
def DTPerformance(tree,trnX, tstX, trnY, tstY,labels):
    prd_trn = tree.predict(trnX)
    prd_tst = tree.predict(tstX)
    return prd_trn, prd_tst

def holdoutDT(X,y,labels,context,save_pics=False, train_size=0.7,
              min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001],
              max_depths = [2, 5, 10, 15, 20, 25],criteria = ['entropy', 'gini']):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    print('-> Holdout for '+context+':')
    best, best_tree, acc = DT(trnX, tstX, trnY, tstY,criteria,max_depths,min_impurity_decrease,context)
    drawDT(best_tree,'Best tree for '+context,save_pics)
    prd_trn, prd_tst = DTPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    if save_pics:
        plt.savefig('plots/'+context+'_DT_Holdout_performance.png')
    plt.show()

def crossValDT(X,y,labels,context,save_pics=False, n_splits = 5,
              min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001],
              max_depths = [2, 5, 10, 15, 20, 25],criteria = ['entropy', 'gini']):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    acc_crossval = np.empty(n_splits, dtype=dict)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    i = 0
    y_train_list = []
    prd_trn_list = []
    y_test_list  = []
    prd_tst_list = []
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        
        print('-> Fold '+str(i)+' for '+context+':')
        best, best_tree, acc_crossval[i] = DT(trnX, tstX, trnY, tstY,criteria,max_depths,min_impurity_decrease,context)
        drawDT(best_tree,'Best tree for '+context,save_pics)
        prd_trn, prd_tst = DTPerformance(best_tree,trnX, tstX, trnY, tstY,labels)
        y_train_list.append(trnY)
        prd_trn_list.append(prd_trn)
        y_test_list.append(tstY)
        prd_tst_list.append(prd_tst)
        # ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        # if save_pics:
        #     plt.savefig('plots/'+context+'_DT_CrossVal'+str(n_splits)+'_#'+str(i)+'_performance.png')
        # plt.show()
        i+=1
    
    g20.plot_avg_evaluation_results(labels, y_train_list, prd_trn_list, y_test_list, prd_tst_list)
    if save_pics:
        plt.savefig('plots/'+context+'_DT_CrossVal'+str(n_splits)+'_average_performance.png')
    plt.show()
    
    print('\n-> Average for '+str(n_splits)+'-fold CrossVal for '+context+':')
    acc_mean = np.mean(acc_crossval)
    print('CrossVal mean score:', acc_mean)
    acc_std = np.std(acc_crossval)
    print('CrossVal std: %.4f' % acc_std)