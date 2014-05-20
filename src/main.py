#!/bin/python

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = ['age', 'workclass', 'fnlwgt', 'education', 'unknown', 'marital_status', 'occupation', 'relationship',
          'race', 'sex', 'capital_gain', 'capita_loss', 'hours-per-week', 'native_country', 'income']
data = pd.read_csv('../data/adult_data.csv', index_col=False, header=None, names=header)
data = data.ix[0:32560, :]

# Getting a feel for the data and looking at the age histogram.

print np.mean(data.age)
plt.hist(data.age, bins=37)
plt.show()


# Replace the income columns with 1 or 0 instead of <=50K and >50K

def income_replacement(value):

    if value == ' <=50K':
        return 0
    else:
        return 1

data['income'] = data['income'].apply(income_replacement)


# Prior to applying the random forest, I vectorize the string variables
# into binary values to process them with the random forest.

features = data.ix[:, 0:-1]
labels = data['income']

features = [dict(r.iteritems()) for _, r in features.iterrows()]
vec = DictVectorizer()
features = vec.fit_transform(features).toarray()

labels = [x for x in labels]

print features
print labels

# Apply the sklearn random forest classifier to the data

clf = RandomForestClassifier(n_estimators=100, criterion='gini')
clf.fit(features, labels)
print clf.score(features, labels)