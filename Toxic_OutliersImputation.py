# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import g20_functions as g20
from g20_functions import OUTLIER_METHODS

data = g20.loadToxic()
print(data.shape)

corr = g20.correlationToxic(data)
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)
print(X.shape)
print(y.shape)

def thresholdsPrint(txt, c, ts = [0.99,0.95,0.90,0.85,0.80]):
    for t in ts:
        print("{}, {} found for {}".format(txt,sum(1 for i in c if i >= t and i < 1),t))


thresholdsPrint("Original Data",corr)
for outlier_method in OUTLIER_METHODS[1:]:
    for cont in [x / 100.0 for x in range(1,11)]:
        X_if, y_if = g20.outlierRemoval(X,y,outlier_method,cont)
        y_if = np.expand_dims(np.array(y_if),axis=1)
        print(X_if.shape)
        print(y_if.shape)
        data_if = pd.DataFrame(data = np.concatenate((X_if,y_if), axis=1))
        corr = g20.correlationToxic(data)
        thresholdsPrint("{} with cont: {}".format(outlier_method,cont),corr)