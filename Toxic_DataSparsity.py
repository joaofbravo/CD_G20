# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

data = pd.read_csv('data/qsar_oral_toxicity.csv',header= None, sep =';')

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

columns = data.select_dtypes(include='number').columns

size = len(columns) #full size
panel_size = int(size/(2**5)) #size within each panel

rows = cols = panel_size

for k in range(0,size,panel_size):
    for m in range(k,size,panel_size):
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
        for i in range(panel_size):
            var1 = columns[i+k]
            if k == m:
                edge = i+1
            else:
                edge = 0
            for j in range(edge, panel_size):
                var2 = columns[j+m]
                axs[i, j].set_title("%s x %s"%(var1,var2))
                axs[i, j].set_xlabel(var1)
                axs[i, j].set_ylabel(var2)
                axs[i, j].scatter(data[var1], data[var2])
        plt.show()