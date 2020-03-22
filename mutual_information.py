#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alexandre Guichet
"""
import time
import os
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

from tools.convert_categorical import convert_cat

def mutual_information(feature, target, n_neighbors = 3, ordered = True, downsample = False):
    """
    Compute Mutual Information using the k-neighbor approach for continuous-discrete mixtures
    _________
    Examples:
        mi = mutual_information(feature, target) 
            - feature is a N-by-MX np.array/pd.DataFrame/pd.Series
            - target is a N-by-MY np.array/pd.DataFrame/pd.Series
        
        returns a Mx-by-My pd.DataFrame with column and label names
    
    ___________
    Parameters:
        'n_neighbors' - default: 5
            <integer>, positive. Number of nighbors used in the knnsearch. Recommended values is 3 [1]
            
        'ordered' - default: True
            <bool>. If the return dataframe should be ordered in descending order for the first label
        
        'downsample' - default: False
            <bool>, Allow resampling of parameters when sample size is huge. 
            Also remove nan's and make features/target of the same length in case nx != ny
            Downsample up to 500 samples/parameters if True
            
        References: 
            	- B.C. Ross, "Mutual Information between Discrete and Continuous Data Sets". PloS ONE 9(2), 2014.
                https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357
            	- A. Kraskov, H. Stogbauer, and P. Grassberger, "Estimating Mutual Information". Phys. Rev. E 69, 2004. 
                https://arxiv.org/abs/cond-mat/0305641
            	- W. Gao, S. Kannan, S. Oh, P. Viswanath, "Estimating Mutual Information for discrete-continuous mixtures". arXiv preprint, arXiv:1709.06212, 2017.
                https://arxiv.org/pdf/1709.06212.pdf
    """       
    if isinstance(feature, pd.Series):
        mx = 1
        mi_features = [feature.name]
        feature = feature.values
    elif isinstance(feature, (np.ndarray, np.generic)):
        try: 
            _, mx = feature.shape
        except ValueError:
            mx = 1
        mi_features = ["Feature_" + str(i) for i in range(mx)]
    elif isinstance(feature, pd.DataFrame):
        _, mx = feature.shape
        mi_features = feature.columns
        feature = feature.values
        
    if isinstance(target, pd.Series):
        my = 1
        mi_labels = [target.name]
        target = target.values
    elif isinstance(target, (np.ndarray, np.generic)):
        try: 
            _, my = target.shape
        except ValueError:
            my = 1
        mi_labels = ["Label_" + str(i) for i in range(my)]
    elif isinstance(target, pd.DataFrame):
        _, my = target.shape
        mi_labels = target.columns
        target = target.values
            
    mi = np.zeros([mx, my])
             
    with tqdm(total = my*mx, desc="computing regression", bar_format = "{l_bar}{bar}]| {n_fmt}/{total_fmt} [time left: {remaining}]", ascii = True) as pbar:
        for i in range(my):
            mi[:,i] = _estimate_mi(feature, target[:,i].reshape(-1, 1) if my > 1 else target.reshape(-1 ,1), n_neighbors = n_neighbors, downsample = downsample, pbar = pbar)                
        pbar.close()
    
    mi = pd.DataFrame(mi, columns = mi_labels, index = mi_features)
        
    if ordered:
        mi.sort_values(by = mi_labels[0])
    
    return mi

def _estimate_mi(X, y, n_neighbors = 3, downsample = False, pbar = None):
    
    try:
        n_samples, n_features = X.shape
    except ValueError:
        n_samples = X.shape[0]
        n_features = 1
        
    ny = y.shape[0]
    
    if n_samples != ny and downsample == False:
        raise ValueError("The number of samples in X does not match y, use 'downsampling' to remove 'nan'.")
    
    #Scale and add noise to X
    X = scale(X)
    if n_features > 1:
        X = X + 1e-10 * np.maximum(1, np.nanmean(np.abs(X), axis = 0)) * np.random.normal(size = [n_samples, 1])
    else:
        X = X + 1e-10 * np.maximum(1, np.nanmean(np.abs(X))) * np.random.normal(size = n_samples)
    #Scale and add noise to y
    y = scale(y)
    y = y + 1e-10 * np.maximum(1, np.nanmean(np.abs(y))) * np.random.normal(size = [n_samples, 1])

    mi = np.zeros(n_features) 
    
    for i in range(n_features):
        x = X[:, i].reshape(-1, 1) if n_features > 1 else X
        if downsample:
            missing_array = np.any(np.isnan(np.hstack([x, y])), axis = 1) #TODO: Check if it returns 1D-array of booleans                
            [x, yi] = resample(x, y, missing_array)
        else:
            yi = y
            
        mi[i] = _compute_mi(x, yi, n_neighbors)
        
        if pbar is not None:
            pbar.update(1)
    return mi

def _compute_mi(x, y, n_neighbors):
    n_samples = x.shape[0]
    x = x.reshape((-1, 1))    
    y = y.reshape((-1, 1))
    xy = np.hstack([x,y])
    
    nn = NearestNeighbors(metric = 'chebyshev', n_neighbors = n_neighbors)    
    if len(x) == 0 or len(y) == 0:
        return 0
    nn.fit(xy)
    
    try: #Added 21.04.2020, to discuss with Adrien   
        radius = nn.kneighbors()[0]
    except ValueError:
        return 0

    radius = np.nextafter(radius[:, -1], 0)
    
    nx = np.sum((np.repeat(x, n_samples, axis = 1) > (np.transpose(x) - radius)) & (np.repeat(x, 
                n_samples, axis = 1) < (np.transpose(x) + radius)), axis = 0)
    ny = np.sum((np.repeat(y, n_samples, axis = 1) > (np.transpose(y) - radius)) & (np.repeat(y, 
                n_samples, axis = 1) < (np.transpose(y) + radius)), axis = 0)
    
    #B.C.Ross - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357
    mi = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx)) - np.mean(digamma(ny))
    
    return max(0, mi)

def resample(X, y, missing_array):
    
    X = X[np.invert(missing_array)]
    y = y[np.invert(missing_array)]

    nnx = X.shape[0]
    nny = y.shape[0]
    
    resample_size = 891
    if (nnx < resample_size) & (nny < resample_size) & (nnx == nny):
        return X, y
    else:
        if nnx < nny & nnx< resample_size: 
            steps = int(np.floor(nny/nnx)) #TODO: Check if it works
            y = y[0::steps]
        elif nny < nnx & nny < resample_size:
            steps = int(np.floor(nnx/nny)) #TODO: Check if it works
            X = X[0::steps]
        elif nny <= nnx:
            steps = int(np.floor(nny/resample_size)) #TODO: Check if it works
            X = X[0::steps]
            y = y[0::steps]
        elif nnx < nny:
            steps = int(np.floor(nnx/resample_size)) #TODO: Check if it works
            X = X[0::steps]
            y = y[0::steps]         
    return X, y
            
    
