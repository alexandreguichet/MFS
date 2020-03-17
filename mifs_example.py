#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:32:06 2020

@author: alexandreguichet
"""

from mifs import MIFS
from mutual_information import mutual_information

import numpy as np 
import pandas as pd
import time

mifs = MIFS()
mifs.load_pickle("titanic_train.pkl")

features = mifs.df.drop(columns = ["Survived"])

#Convert categorical
cat_columns = features.select_dtypes(['category', 'object']).columns
features[cat_columns] = features[cat_columns].astype('category').apply(lambda x: x.cat.codes)
    
labels = mifs.df["Survived"].to_frame()

start = time.time()
mi = mutual_information(features, labels, downsample = True)
print(time.time() - start)

start = time.time()
from sklearn.feature_selection import mutual_info_regression
aa = dict()
for i in features.columns:
    x = features[i].values.reshape(1, -1)
    y = labels.values.reshape(1, -1)
    
    missing_array = np.any(np.isnan(np.transpose(np.vstack([x, y]))), axis = 1) #TODO: Check if it returns 1D-array of booleans                
    
    x = x[np.invert(missing_array).reshape(1, -1)]
    y = y[np.invert(missing_array).reshape(1, -1)]

    aa[i] = mutual_info_regression(x.reshape(-1, 1), y.reshape(-1, 1))
print(time.time() - start)

#Check what is the value of 1-to-1 ratio    
bb = mutual_information(features["Fare"].values, features["Fare"].values)
cc = mutual_info_regression(features["Fare"].values.reshape(-1, 1), features["Fare"].values.reshape(-1, 1))
