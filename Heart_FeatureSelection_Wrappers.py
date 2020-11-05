# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV, SelectFromModel

# get data
data: pd.DataFrame = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

##### RFECV (recursive feature elemination with crossVal)
classifier = DecisionTreeClassifier(min_samples_leaf=1, max_depth=None, criterion='entropy', min_impurity_decrease=0.02)
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)

# plot results
print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
plt.savefig('plots/Heart_FeatureSelection_Wrapper_DT.png')

##### SelectFromModel
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print("\nFeature ranking =", clf.feature_importances_)
print("Original data shape:", X.shape, "\nNew data shape:", X_new.shape)
