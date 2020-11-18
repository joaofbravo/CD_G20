# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import g20_functions as g20
from g20_functions import OUTLIER_METHODS

data = g20.loadHeart()
data.pop('time')
columns = data.columns.copy()

g20.correlationHeart(data, title = "Original Data")
y: np.ndarray = data.pop('DEATH_EVENT').values

X: np.ndarray = data.values
labels = pd.unique(y)
print(X.shape)
print(y.shape)

for outlier_method in OUTLIER_METHODS:
    for cont in [x / 100.0 for x in range(1,11)]:
        X_if, y_if = g20.outlierRemoval(X,y,outlier_method,cont)
        y_if = np.expand_dims(np.array(y_if),axis=1)
        print(X_if.shape)
        print(y_if.shape)
        data_if = pd.DataFrame(data = np.concatenate((X_if,y_if), axis=1), columns = columns)
        g20.correlationHeart(data_if, title = "{} with cont: {}".format(outlier_method,cont))