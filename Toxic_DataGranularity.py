# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds

data = pd.read_csv('data/qsar_oral_toxicity.csv',header= None, sep =';')

print(data)
print(data.columns)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

variables = data.select_dtypes(include='number').columns
rows = 32
cols = 32
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=2)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.show()
