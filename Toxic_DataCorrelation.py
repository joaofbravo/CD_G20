# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

data = pd.read_csv('data/qsar_oral_toxicity.csv',header= None, sep =';')


#Data records, variables and type
print(data.shape); print(data.dtypes); print()

#Change last variable from non-numeric to symbolic
cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes); print()

#data
print(data)


#Data correlation
c = data.corr().abs()

s = c.unstack()
so = s.sort_values(kind="quicksort")


#pd.set_option('display.max_rows', None)
print(so[-1400:-1025])



