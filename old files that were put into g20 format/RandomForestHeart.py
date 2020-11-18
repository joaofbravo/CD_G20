# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import ds_functions as ds

# part 1
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

criteria = ['entropy', 'gini']
n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
max_depths = [5, 10, 25]
max_features = [.1, .3, .5, .7, .9, 1]
best = ('', 0, 0, 0)
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

# part 2
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)