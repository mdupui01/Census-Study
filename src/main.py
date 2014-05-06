#!/bin/python

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = ['age', 'workclass', 'fnlwgt', 'education', 'unknown', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capita_loss', 'hours-per-week', 'native_country', 'income']
data = pd.read_csv('../data/adult_data.csv', index_col = False, header = None, names = header)
data = data.ix[0:32560,:]

print data

print np.mean(data.age)
plt.hist(data.age, bins = 37)
plt.show()
