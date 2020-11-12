# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ds_functions as ds


# Naive Bayes
def NaiveBayesModel(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)
    ds.plot_evaluation_results(pd.unique(np.concatenate((y_train, y_test))), y_train, prd_trn, y_test, prd_tst)
    return clf


# Estimation
def NaiveBayesEstimation(X_train, X_test, y_train, y_test, context):
    if min(np.amin(X_train),np.amin(X_train))>=0:
        estimators = {'GaussianNB': GaussianNB(),
                      'MultinomialNB': MultinomialNB(),
                      'BernoulyNB': BernoulliNB()}
    else:
        estimators = {'GaussianNB': GaussianNB(),
                      'BernoulyNB': BernoulliNB()}
    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(X_train, y_train)
        prdY = estimators[clf].predict(X_test)
        yvalues.append(metrics.accuracy_score(y_test, prdY))
    
    plt.figure()
    ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models for '+context, ylabel='accuracy', percentage=True)
    return xvalues, np.asarray(yvalues)


# hold-out (train_test_split)
def holdoutNaiveBayes(X,y,labels,context,save_pics=False, train_size=0.7):
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    print('-> Holdout for '+context+':')
    NaiveBayesModel(trnX, tstX, trnY, tstY)
    if save_pics:
        plt.savefig('plots/'+context+'_NaiveBayes_holdout.png')
    plt.show()
    _, score_holdout = NaiveBayesEstimation(trnX, tstX, trnY, tstY, context)
    if save_pics:
        plt.savefig('plots/'+context+'_NaiveBayes_holdout_estimators.png')
    plt.show()
    
    print('Holdout score for '+context+':', score_holdout)
    
        
# k-fold cross validation (StratifiedKFold)
def crossValNaiveBayes(X,y,labels,context,save_pics=False, n_splits = 5):
    skf = StratifiedKFold(n_splits, shuffle=True)
    skf.get_n_splits(X, y)
    print('\n-> '+str(n_splits)+'-fold CrossVal for '+context+':')
    clf_crossval = np.empty(n_splits, dtype=GaussianNB)
    score_crossval = np.empty(n_splits, dtype=np.ndarray)
    i = 0
    for train_index, test_index in skf.split(X, y):
        print('\n-> fold '+str(i)+' for '+context+':')
        # print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf_crossval[i] = NaiveBayesModel(X_train, X_test, y_train, y_test)
        if save_pics:
            plt.savefig('plots/'+context+'_NaiveBayes_CrossVal'+str(n_splits)+'_#'+str(i)+'.png')
        plt.show()
        xvalues, score_crossval[i] = NaiveBayesEstimation(X_train, X_test, y_train, y_test,context)
        if save_pics:
            plt.savefig('plots/'+context+'_NaiveBayes_CrossVal'+str(n_splits)+'_#'+str(i)+'_estimators.png')
        plt.show()
        i+=1
    
    
    # CrossVal: find max, mean, std
    score_best = np.max([score_crossval[i] for i in range(n_splits)])
    score_bestarg = np.argmax([score_crossval[i] for i in range(n_splits)])
    print(context)
    print('CrossVal best estimator: %s with score %.2f' % (xvalues[score_bestarg % len(xvalues)], score_best))
    
    score_mean = np.mean(score_crossval)
    score_std = np.std([score_crossval[i][0] for i in range(n_splits)])
    print('CrossVal mean score:', score_mean)
    print('CrossVal std: %.4f' % score_std)
    
    plt.figure()
    ds.bar_chart(xvalues, score_mean, title='Comparison of Naive Bayes Models for '+context, ylabel='accuracy', percentage=True)
    if save_pics:
        plt.savefig('plots/'+context+'_NaiveBayes_CrossVal'+str(n_splits)+'_mean_estimators.png')
    plt.show()
