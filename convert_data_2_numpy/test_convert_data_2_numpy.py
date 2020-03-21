# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:06:46 2020

@author: a.guichet
"""

# Std libraries
import numpy as np
import pandas as pd

# Library to test
import convert_data_2_numpy as cvd2np
#----------------------------------------------------------------------------#

ref_data = [2,3.5,3,1,0,2] # WARNING numpy casts to str if any item is str while panda.values return the right type
ref_column_name = "test_column_name"
ref_pd_series_ = pd.Series(data=ref_data, name=ref_column_name)
ref_dataframe = pd.DataFrame(data=ref_data, columns=[ref_column_name])

def test__is_pd_series():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.__is_pd_series(ref_pd_series_)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
def test__is_pd_dataframe():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.__is_pd_dataframe(ref_dataframe)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
def test__is_numpy_array():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name + "_0"
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.__is_numpy_array(ref_np_data, ref_column_name)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
def test_convert_data_2_numpy__series():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.convert_data_2_numpy(ref_pd_series_)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
def test_convert_data_2_numpy__dataframe():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.convert_data_2_numpy(ref_dataframe)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
def test_convert_data_2_numpy__numpy():
    # setup reference variables
    ref_mx = 1
    ref_my = len(ref_data)
    ref_column_names = ref_column_name + "_0"
    ref_np_data = np.array(ref_data, ndmin=2).T
    
    # get test variables
    t_mx, t_my, t_column_names, t_np_data = cvd2np.convert_data_2_numpy(ref_np_data, ref_column_name)
    
    # Test
    assert (ref_mx == t_mx)
    assert (ref_my == t_my)
    assert (ref_column_names == t_column_names[0])
    assert (ref_np_data == t_np_data).all()
    
test_convert_data_2_numpy__dataframe()