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

def mutual_information(feature, target, n_neighbors = 3, ordered = True, downsampling = False):
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
        
        'downsampling' - default: False
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


    try: 
        _, mx = feature.shape
        _, my = target.shape
    except:
        mx = feature.shape #Cases: pd.Series (1D) and 1D nump.arrays
        my = target.shape #Cases: pd.Series (1D) and 1D nump.arrays
    
    mi = np.zeros([mx, my])
    
    try: 
        mi_labels = target.columns
    except:
        mi_labels = ["Labels_" + str(i) for i in range(my)]
    
    if isinstance(feature, pd.DataFrame):
        mi_features = feature.columns
    else: 
        mi_features = ["Feature_" + str(i) for i in range(mx)]
        
    with tqdm(total = my*mx, desc=_desc, bar_format = "{1_bar}{bar}]| {n_fmt}/{total_fmt} [time left: {remianing}]", ascii = True) as pbar:
        for i in range(my):
            mi[:,i] = _estimate_mi(X, y, n_neighbors = n_neighbors, downsampling = downsampling, pbar = pbar)
        pbar.close()
    
    mi = pd.DataFrame(mi, columns = mi_labels, index = mi_features)
    
    if ordered:
        mi.sort_values(by = mi_babels[0])
    
    return mi

def _estimate_mi(X, y, n_neighbors = 3, downsampling = False, pbar = None):
    
    n_samples, n_features = X.shape
    ny = y.shape[0]
    
    if n_samples != ny and downsampling == False:
        raise ValueError("The number of samples in X does not match y, use 'downsampling' to remove 'nan'.")
    
    #Scale and add noise to X
    X = scale(X)
    X = X + 1e-10 * np.maximum(1, np.mean(np.abs(X), axis = 0)) * np.random.normal(size = [n_samples, 1])
    
    #Scale and add noise to y
    y = scale(y)
    y = y + 1e-10 * np.maximum(1, np.mean(np.abs(y))) * np.random.normal(size = n_samples)

    mi = np.zeros(n_features) 
    
    for i in range(n_features):
        x = X[:, i]
        if downsampling:
            missing_aray = np.isnan(np.hstack([x, y])) #TODO: Check if it returns 1D-array of booleans

            [x, yi] = resample(x, y)
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
    
    nn = NearestNeighbors(metric = 'chebyshev', n_neighbors = n_neighbors)
    
    nn.fit(np.hstack([x,y]))
    radius = nn.kneighbors()[0]
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
    
    resample_size = 500
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