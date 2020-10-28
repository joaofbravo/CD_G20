# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import ds_functions as ds


#
def loadHeart():
    return pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

def loadToxic():
    pd.read_csv('data/qsar_oral_toxicity.csv',header= None, sep =';')


def dataShapeAndTypes(data):
    #Data records, variables and type
    
    print(data.shape); print(data.dtypes)

def correlationHeart(data, title = 'Correlation analysis'):
    register_matplotlib_converters()

    #Data correlation
    
    plt.figure(figsize=[12, 12])
    corr_mtx = data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()

OUTLIER_METHODS = ["Isolation Forest","Elliptic Envelope","Local Outlier Factor"]
def outlierRemoval(X,Y, method, cont):
    if method == "Isolation Forest":
        iso = IsolationForest(contamination=cont)
        yhat = iso.fit_predict(X)
    elif method == "Elliptic Envelope":
        ee = EllipticEnvelope(contamination=cont)
        yhat = ee.fit_predict(X)
    elif method == "Local Outlier Factor":
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X)
    mask = yhat != -1
    
    return X[mask, :],Y[mask]