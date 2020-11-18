# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import scipy.stats as _stats
import numpy as np

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = _stats.lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    ds.multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

def data_dimensionality(data, dataset):
    print(data.shape)
    print()
    
    plt.figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    ds.bar_chart(values.keys(), values.values(), title='Nr of records vs nr variables')
    
    print(data.dtypes)
    print()
    
    if dataset == "Toxic":
        cat_vars = data.select_dtypes(include='object')
        data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        print(data.dtypes)
        print()
    
    plt.figure()
    mv = {}
    for var in data:
        mv[var] = data[var].isna().sum()
    ds.bar_chart(mv.keys(), mv.values(), title='Nr of missing values per variable',
                   xlabel='variables',
                   ylabel='nr missing values')
    plt.xticks(rotation=90)
    
def data_granularity(data, dataset):
    print(data.describe(), '\n')
    print(data.columns)
    print()
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    numeric_vars = data.select_dtypes(include='number').columns
    variables = data.select_dtypes(include='number').columns
    i, j = 0, 0
    if dataset == "Heart":
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
    elif dataset == "Toxic":
        rows = 32
        cols = 32
        fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
        
        for n in range(len(variables)):
            axs[i, j].set_title('Histogram for %s'%variables[n])
            axs[i, j].set_xlabel(variables[n])
            axs[i, j].set_ylabel('nr records')
            axs[i, j].hist(data[variables[n]].values, bins=2)
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()
    
def data_distribution(data, dataset):
    register_matplotlib_converters()
    
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

    print(data.describe())
    
    numeric_vars = data.select_dtypes(include='number').columns
    if dataset == "Heart":
        rows, cols = ds.choose_grid(len(numeric_vars))
    elif dataset == "Toxic":
        rows, cols = 32,32
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        sns.distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()

def data_sparsity(data, dataset):
    register_matplotlib_converters()
    
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    
    columns = data.select_dtypes(include='number').columns
    
    if dataset == "Heart":
        rows, cols = len(columns)-1, len(columns)-1
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
        for i in range(len(columns)):
            var1 = columns[i]
            for j in range(i+1, len(columns)):
                var2 = columns[j]
                axs[i, j-1].set_title("%s x %s"%(var1,var2))
                axs[i, j-1].set_xlabel(var1)
                axs[i, j-1].set_ylabel(var2)
                axs[i, j-1].scatter(data[var1], data[var2])
        plt.show()
    elif dataset == "Toxic":
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

def data_correlation(data, dataset):
    register_matplotlib_converters()
   
    if dataset == "Heart":
        fig = plt.figure(figsize=[12, 12])
        corr_mtx = data.corr()
        sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        plt.title('Correlation analysis')
        plt.show()
    
    elif dataset == "Toxic":
        #Change last variable from non-numeric to symbolic
        cat_vars = data.select_dtypes(include='object')
        data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    
        c = data.corr().abs()
        
        s = c.unstack()
        so = s.sort_values(kind="quicksort")
        
        #pd.set_option('display.max_rows', None)
        print(so[-1400:-1025])




def topCorr(so, thresholds = [0.99]):
    return 1
