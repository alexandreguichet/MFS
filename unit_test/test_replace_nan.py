# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:35:18 2020

@author: a.guichet
"""

import copy
import numpy as np
import pandas as pd
import replace_nan as rn

#----------------------------------------------------------------------------#
numpy_array_with_nan = np.array([2,np.nan,3,1,0,np.nan], ndmin=2).T

def test__remove_all_nan__int():
    ##Base test
    test_nan_indexes = np.argwhere(np.isnan(numpy_array_with_nan))[:,0]
    test_trimmed_array = np.delete(numpy_array_with_nan, test_nan_indexes)
    
    ##Function to test
    new_values, indexes_removed = rn.__remove_all_nan(numpy_array_with_nan)
    assert (new_values == test_trimmed_array).all()
    assert (indexes_removed == test_nan_indexes).all()

test__remove_all_nan__int()
    
categorical_str_array = np.array(["male","female",np.nan, 
                                  'male',"robot",np.nan], ndmin=2).T
    
def test__remove_all_nan__str():
    ##Base test
    test_nan_indexes = np.argwhere(categorical_str_array == 'nan')
    test_trimmed_array = np.delete(categorical_str_array, test_nan_indexes)
    
    #Function to test
    new_values, indexes_removed = rn.__remove_all_nan(categorical_str_array)
    assert (new_values == test_trimmed_array).all()
    assert (indexes_removed == test_nan_indexes).all()

#----------------------------------------------------------------------------#
categorical_str_array = np.array(["male","female",np.nan, 
                                  'male',"robot",np.nan], ndmin=2).T
categorical_int_array = np.array([0,1,np.nan, 0,2, np.nan], ndmin=2).T

def test_make_into_new_class__str(): 
    
    #Convert to int categorical
    length_array = np.size(categorical_str_array)
    codes, uniques = pd.factorize(categorical_str_array.reshape(length_array)) # nan are interpreted as a string

    #Function to test
    new_values, indexes_removed = rn.__make_into_new_class(categorical_str_array)
    assert (new_values == codes).all()
    assert indexes_removed == [None]
        
def test_make_into_new_class__int(): 
    
    #Convert to int categorical
    length_array = np.size(categorical_int_array)
    codes, uniques = pd.factorize(categorical_int_array.reshape(length_array)) # nan are interpreted as a string

    #Function to test
    new_values, indexes_removed = rn.__make_into_new_class(categorical_int_array)
    assert (new_values == codes).all()
    assert indexes_removed == [None]
    
#----------------------------------------------------------------------------#
    
def test_replace_nan_in_vector__remove_all_nan__str():
    #Base result
    test_new_values, test_indexes_removed = rn.__remove_all_nan()
    
    #Function to test
    new_values, indexes_removed = replace_nan_in_vector(array, mode="auto")
    