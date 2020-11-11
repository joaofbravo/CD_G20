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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
register_matplotlib_converters()
from imblearn.over_sampling import SMOTE

# loads up the heart dataset
def loadHeart():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    
    # the time column has to do with how the study kept up with the patients, does not make sense to include in the analysis
    data.pop('time')
    return data

# loads up the toxic dataset
def loadToxic():
    return pd.read_csv('data/qsar_oral_toxicity.csv',header= None, sep =';')

# split the data into x & y
def xySplit(data,target='DEATH_EVENT'):
    y: np.ndarray = data.pop(target).values
    x: np.ndarray = data.values
    labels = pd.unique(y)
    return x,y,labels


# returns scaled data with two different methods
def scaleHeart(data):
    # exclude death_event from scaling, or you'll get problems later
    death_event = data.pop('DEATH_EVENT').values
    # print(data.describe(include='all'))
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
    norm_data_zscore = pd.DataFrame(transf.transform(data), columns= data.columns)
    norm_data_zscore['DEATH_EVENT'] = death_event
    # print(norm_data_zscore.describe(include='all'))
    
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
    norm_data_minmax = pd.DataFrame(transf.transform(data), columns= data.columns)
    norm_data_minmax['DEATH_EVENT'] = death_event
    # print(norm_data_minmax.describe(include='all'))
    return norm_data_zscore, norm_data_minmax

# returns base data, and data for the 3 balancing methods
def balanceData(data,dataset = "Heart",save_pics=False):
    if dataset == "Heart":
        target='DEATH_EVENT'
    elif dataset == "Toxic":
        target=1024
    
    target_count = data[target].value_counts()
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)
    if save_pics:
        plt.savefig('plots/Class balance for '+dataset+' dataset.png')
    plt.show()
    
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    
    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')
    
    RANDOM_STATE = 42 #The answer to the Ultimate Question of Life, the Universe, and Everything 
    
    
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}
    
    df_class_min = data[data[target] == min_class]
    df_class_max = data[data[target] != min_class]
    output = {'Original': pd.DataFrame().append(df_class_min).append(df_class_max)}
    
    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]
    output['UnderSample'] = pd.DataFrame().append(df_class_min).append(df_under)
    
    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]
    output['OverSample'] = pd.DataFrame().append(df_class_max).append(df_over)
    
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = data.pop(target).values
    X = data.values
    smote_X, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]
    output['SMOTE'] = [X,y]
    
    fig = plt.figure()
    ds.multiple_bar_chart([target_count.index[ind_min_class], target_count.index[1-ind_min_class]], values,
                          title='Target', xlabel='frequency', ylabel='Class balance')
    if save_pics:
        plt.savefig('plots/Target for '+dataset+' dataset.png')
    plt.show()
    return output

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

def correlationToxic(data):
    c = data.corr().abs()
    s = c.unstack()
    return s.sort_values(kind="quicksort")

def topCorr(so, thresholds = [0.99]):
    return 1

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