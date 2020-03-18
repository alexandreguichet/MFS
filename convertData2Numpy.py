# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:10:33 2020

@author: a.guichet
"""
import numpy as np
import pandas as pd

x = np.array([2,3,1,0])

def _is_pd_series(data):
    mx = 1
    my = data.size
    column_names = [data.name]
    np_data = data.values
    return mx, my, column_names, np_data

def _is_pd_dataframe(data):
    my, mx = data.shape
    column_names = data.columns
    np_data = data.values
    return mx, my, column_names, np_data

def _is_numpy_array(data):
    data = np.array(data, ndmin=2)
    my, mx = data.shape
    column_names = ["Feature_" + str(i) for i in range(mx)]
    np_data = data
    return mx, my, column_names, np_data
    
def convert_data_2_numpy(data):
    if isinstance(data, pd.Series):
        return _is_pd_series(data)
    elif isinstance(data, pd.DataFrame):
        return _is_pd_dataframe(data)
    elif isinstance(data, np.ndarray):
        return _is_numpy_array(data)
    else:
        raise ValueError("The input data type is neither: pd.Series or pd.DataFrame or np.ndarray")
        
convert_data_2_numpy("0")