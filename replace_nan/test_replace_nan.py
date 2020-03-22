# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:35:18 2020

@author: a.guichet
"""

import numpy as np
import pandas as pd
import replace_nan as rn



def test__remove_all_nan__int():
    ref_array_with_nan = np.array([2,np.nan,3,1,0,np.nan], ndmin=2).T
    
    ##Reference test
    ref_nan_indexes = np.argwhere(np.isnan(ref_array_with_nan))[:,0]
    ref_trimmed_array = np.delete(ref_array_with_nan, ref_nan_indexes)
    
    ##Function to test
    t_new_values, t_indexes_2_remove = rn.__remove_all_nan(ref_array_with_nan)
    
    # (=).all() returns false when arrays have nan; therefore we have to do 
    # a comparison of only the numeral values"
    t_trimmed_array = np.delete(t_new_values, t_indexes_2_remove)
    assert (t_trimmed_array == ref_trimmed_array).all()
    assert (t_indexes_2_remove == ref_nan_indexes).all()
  
def test__remove_all_nan__str():
    ref_array_categorical_str = np.array(["male","female",np.nan, 
                                  'male',"robot",np.nan], ndmin=2).T
    
    ##Reference test
    test_nan_indexes = np.argwhere(ref_array_categorical_str == 'nan')
    #test_trimmed_array = np.delete(ref_array_categorical_str, test_nan_indexes)
    
    #Function to test
    new_values, indexes_removed = rn.__remove_all_nan(ref_array_categorical_str)
    assert (new_values == ref_array_categorical_str).all()
    assert (indexes_removed == test_nan_indexes).all()

def test__replace_by_mean():
    #Reference test
    ref_numpy_array = np.array([2,np.nan,3,0,0,np.nan], ndmin=2).T
    ref_numpy_array__mean = np.array([2,1.25,3,0,0,1.25], ndmin=2).T
    
    #Function to test
    t_new_values, t_indexes_removed = rn.__replace_by_mean(ref_numpy_array)
    assert (ref_numpy_array__mean == t_new_values).all()
    assert [None] == t_indexes_removed

def test__replace_by_median():
    #Reference test
    ref_numpy_array = np.array([2,np.nan,3,7,0,np.nan], ndmin=2).T
    ref_numpy_array__median = np.array([2,2.5,3,7,0,2.5], ndmin=2).T
    
    #Function to test
    t_new_values, t_indexes_removed = rn.__replace_by_median(ref_numpy_array)
    assert (ref_numpy_array__median == t_new_values).all()
    assert [None] == t_indexes_removed
    
def test_make_into_new_class__str(): 
    ref_array_categorical_str = np.array(["male","female",np.nan, 
                                  'male',"robot",np.nan], ndmin=2).T
    #Convert to int categorical
    length_array = np.size(ref_array_categorical_str)
    codes, uniques = pd.factorize(ref_array_categorical_str.reshape(length_array)) # nan are interpreted as a string

    #Function to test
    new_values, indexes_removed = rn.__make_into_new_class(ref_array_categorical_str)
    assert (new_values == codes).all()
    assert indexes_removed == [None]
        
def test_make_into_new_class__int(): 
    ref_array_categorical_int = np.array([0,1,np.nan, 0,2, np.nan], ndmin=2).T
    #Convert to int categorical
    length_array = np.size(ref_array_categorical_int)
    codes, uniques = pd.factorize(ref_array_categorical_int.reshape(length_array)) # nan are interpreted as a string

    #Function to test
    new_values, indexes_removed = rn.__make_into_new_class(ref_array_categorical_int)
    assert (new_values == codes).all()
    assert indexes_removed == [None]
    
ref_str_array = np.array(["male","female",np.nan, 
                                  'male',"robot",np.nan], ndmin=2).T

def test_replace_nan_in_column__remove_all_nan__str():
    #Reference result
    ref_new_values, ref_indexes_removed = rn.__remove_all_nan(ref_str_array)
    
    #Function to test
    t_new_values, t_indexes_removed = rn.replace_nan_in_column(ref_str_array, mode="remove")
    assert (ref_new_values == t_new_values).all()
    assert (ref_indexes_removed == t_indexes_removed).all()
    
ref_numpy_array = np.array([2,np.nan,3,1,0,np.nan], ndmin=2).T

def test_replace_nan_in_column__remove_all_nan__int():
    
    ref_array_with_nan = np.array([2,np.nan,3,1,0,np.nan], ndmin=2).T
    
    ##Reference test
    ref_new_values, ref_indexes_2_remove = rn.__remove_all_nan(ref_array_with_nan)   
    ##Function to test
    t_new_values, t_indexes_2_remove = rn.replace_nan_in_column(ref_numpy_array, mode="remove")
    
    # (=).all() returns false when arrays have nan; therefore we have to do 
    # a comparison of only the numeral values"
    ref_trimmed_array = np.delete(ref_new_values, ref_indexes_2_remove)
    t_trimmed_array = np.delete(t_new_values, t_indexes_2_remove)
    assert (t_trimmed_array == ref_trimmed_array).all()
    assert (t_indexes_2_remove == ref_indexes_2_remove).all()
    
def test_replace_nan_in_column__replace_by_mean():
    #Reference test
    ref_numpy_array = np.array([2,np.nan,3,0,0,np.nan], ndmin=2).T
    ref_numpy_array__mean = np.array([2,1.25,3,0,0,1.25], ndmin=2).T
    
    #Function to test
    t_new_values, t_indexes_removed = rn.replace_nan_in_column(ref_numpy_array, mode="mean")
    assert (ref_numpy_array__mean == t_new_values).all()
    assert [None] == t_indexes_removed


def test_replace_nan_in_column__replace_by_median():
     #Reference test
    ref_numpy_array = np.array([2,np.nan,3,7,0,np.nan], ndmin=2).T
    ref_numpy_array__median = np.array([2,2.5,3,7,0,2.5], ndmin=2).T
    
    #Function to test
    t_new_values, t_indexes_removed = rn.replace_nan_in_column(ref_numpy_array, mode="median")
    assert (ref_numpy_array__median == t_new_values).all()
    assert [None] == t_indexes_removed