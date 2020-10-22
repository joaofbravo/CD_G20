# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds

data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
print(data.describe(), '\n')
# print(data)
print(data.columns)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

numeric_vars = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(numeric_vars))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
bins = 2
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[numeric_vars[n]].values, bins=bins)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.show()

variables = data.select_dtypes(include='number').columns
bins = (2, 10, 25, 50, 100, 299)
rows = len(variables)
cols = len(bins)
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
plt.show()
