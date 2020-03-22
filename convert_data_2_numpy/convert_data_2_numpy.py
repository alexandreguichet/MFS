# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:10:33 2020

@author: a.guichet
"""
import numpy as np
import pandas as pd

def __is_pd_series(data):
    mx = 1
    my = data.size
    column_names = [data.name]
    np_data = np.array(data.values, ndmin=2).T
    return mx, my, column_names, np_data

def __is_pd_dataframe(data):
    my, mx = data.shape
    column_names = data.columns
    np_data = np.array(data.values, ndmin=2)
    return mx, my, column_names, np_data

def __is_numpy_array(data, column_name):
    data = np.array(data, ndmin=2)
    my, mx = data.shape
    column_names = [column_name + "_" + str(i) for i in range(mx)]
    np_data = data
    return mx, my, column_names, np_data
    
def convert_data_2_numpy(data, column_name="column"):
    if isinstance(data, pd.Series):
        return __is_pd_series(data)
    elif isinstance(data, pd.DataFrame):
        return __is_pd_dataframe(data)
    elif isinstance(data, np.ndarray):
        return __is_numpy_array(data, column_name)
    else:
        raise ValueError("The input data type is neither: pd.Series or pd.DataFrame or np.ndarray")