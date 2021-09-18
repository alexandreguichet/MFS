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

# mifs = MIFS()
# mifs.load_file("datasets\\IPODataFull.csv")

# #extract features/labels
# features = mifs.df.drop(columns = ["Survived"])
# labels = mifs.df["Survived"].to_frame()

# #Convert categorical
# cat_columns = features.select_dtypes(['category', 'object']).columns
# features[cat_columns] = features[cat_columns].astype('category').apply(lambda x: x.cat.codes)

# #Calculate mutual information
# start = time.time()
# mi = mutual_information(features, labels, downsample = True)
# print(time.time() - start)

# #Scikit-learn implementation
# start = time.time()
# from sklearn.feature_selection import mutual_info_regression
# aa = dict()
# for i in features.columns:
#     x = features[i].values.reshape(1, -1)
#     y = labels.values.reshape(1, -1)
    
#     missing_array = np.any(np.isnan(np.transpose(np.vstack([x, y]))), axis = 1) #TODO: Check if it returns 1D-array of booleans                
    
#     x = x[np.invert(missing_array).reshape(1, -1)]
#     y = y[np.invert(missing_array).reshape(1, -1)]

#     aa[i] = mutual_info_regression(x.reshape(-1, 1), y.reshape(-1, 1))
# print(time.time() - start)

# #Check what is the value of 1-to-1 ratio - Comparing Scikit-learn and MIFS absolute value for a 1-1 ratio.    
# bb = mutual_information(features["Fare"].values, features["Fare"].values)
# cc = mutual_info_regression(features["Fare"].values.reshape(-1, 1), features["Fare"].values.reshape(-1, 1))

# ##Feature Selection
mifs = MIFS()
mifs.load_file(r"datasets/IPODataFull.pkl")

from tools.convert_categorical import convert_cat
#deal categorical data
mifs.df = convert_cat(mifs.df)

#Select the labels we want to optimize for, arrays of labels can be passed
mifs.separate_labels(["Rate of return over 261 days"])

#results is a dictionary!
# check results['selected'] for the final answer: unique features with most redundancies removed (final value is normalized)
# check results['threshold'] for all features above a threshold (n = 50 here), redundancies are still present
# check results['all'] for the mutual information matrix of the 50 selected features (that are above a threshold)
# check results['labels'] for the mutual information value of all features
results = mifs.select_n_features(n = 50, downsample = True) 
