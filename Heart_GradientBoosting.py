# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
import ds_functions as ds

# get data
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

##### FOR TESTING
X = X[20:50]
y = y[20:50]

##### MUDAR PARA CROSSVAL #####
# hold-out (train_test_split) 
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

# Gradient Boosting parameters
losses = ['deviance', 'exponential'] # exponential == AdaBoost
criterions = ['friedman_mse', 'mae']
learn_rates = [0.01, 0.1, 0.5]
n_estimators = [10, 100, 200, 300]
max_depths = [5, 10, 25]
max_features = [.25, .5, 1]

yvalues = {}
best_par = ()
best_tree = None
last_best = 0

cols = len(losses)
rows = len(criterions)
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)

# Gradient Boosting fit
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
        # progress flag
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

# Best result's performance
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
