# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
import sklearn.metrics as metrics
import ds_functions as ds

# get data
data: pd.DataFrame = pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep =';')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)

# hold-out (train_test_split)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

# Gradient Boosting
losses = ['deviance', 'exponential'] # exponential == AdaBoost
criterions = ['friedman_mse', 'mae']
learn_rates = [0.01, 0.1, 0.3, 0.5, 1]
n_estimators = [10, 50, 100, 200, 300]
max_depths = [5, 10, 25]
max_features = [.25, .5, .75, 1]
best = ('', 0, 0)
last_best = 0
best_tree = None

cols = len(losses)
rows = len(criterions)
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)

for k1 in range(len(losses)): ############### 2
    l = losses[k1]
    values = {}
    for k2 in range(len(criterions)): ####### 2
        crit = criterions[k2]
        for lr in learn_rates: ############## 5
            yvalues = []
            for n in n_estimators: ########## 5
                for d in max_depths: ######## 3
                    for f in max_features: ## 4
                        clf = GradientBoostingClassifier(loss=l, criterion=crit, learning_rate=lr, n_estimators=n, max_depth=d, max_features=f)
                        clf.fit(trnX, trnY)
                        prdY = clf.predict(tstX)
                        
                        yvalues.append(metrics.accuracy_score(tstY, prdY))
                        if yvalues[-1] > last_best:
                            best = (l, crit, lr, n, d, f)
                            last_best = yvalues[-1]
                            best_tree = clf

            values[lr] = yvalues
        # progress flag
        print('--- done: {}, {}'.format(l, crit))
    ##############################################
    ########### TIRAR ISTO DAQUI #################
    ds.multiple_line_chart(n_estimators, values, ax=axs[k1, k2], title='Gradient Boosting with loss={}, crit={}'.format(l, crit), xlabel='nr estimators', ylabel='accuracy', percentage=True)

plt.show()
print('\nBest results with loss={}, criterion={}, learning_rate={}, {} estimators, depth={} and {} features, with accuracy={:.3f}'.format(*best, last_best))

# Best results
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)