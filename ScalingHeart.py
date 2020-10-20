# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler
register_matplotlib_converters()

def scale_heart(data):
    
    # print(data.describe(include='all'))
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
    norm_data_zscore = pd.DataFrame(transf.transform(data), columns= data.columns)
    # norm_data_zscore.describe(include='all')
    
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
    norm_data_minmax = pd.DataFrame(transf.transform(data), columns= data.columns)
    # norm_data_minmax.describe(include='all')
    return norm_data_zscore, norm_data_minmax


data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
norm_data_zscore, norm_data_minmax = scale_heart(data)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(axis='x', labelrotation=90)
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 1].tick_params(axis='x', labelrotation=90)
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
axs[0, 2].tick_params(axis='x', labelrotation=90)
# plt.xticks(rotation= 90)
plt.show()