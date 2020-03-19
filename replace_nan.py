# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:37:05 2020

@author: a.guichet

(Priority: Easy - Time: Low - Difficulty: Easy) Enlever nan’s automatically
default: let the algorithm pick for you

Utiliser plusieur méthodes pour deal with nans 
Replace median,
Replace by mean,
Replace by esperance, (if we have time as it is an embeded technique)
Remove completely

"""
import numpy as np
import pandas as pd

def __remove_all_nan(array):
    nan_indexes = np.where(array == np.nan)
    trimmed_array = np.delete(array, nan_indexes)
    return trimmed_array, nan_indexes

def __replace_by_mean(array):
    nan_indexes = np.where(array == np.nan)
    array_mean = np.mean(array)
    np.put(array, nan_indexes, array_mean)
    return array, [None]
    
def __replace_by_median(array):
    nan_indexes = np.where(array == np.nan)
    array_median = np.median(array)
    np.put(array, nan_indexes, array_median)
    return array, [None]

def __make_into_new_class(array):
    nan_indexes = np.where(array == np.nan)
    
    # Convert data to int categorical (in case its not already done)
    # and find next integer
    codes, _ = pd.factorize(array)
    sorted_codes = np.sort(codes.unique)
    new_class_id = sorted_codes[-1] + 1
    
    # Replace all Nan by new class ID
    np.put(array, nan_indexes, new_class_id)
    return array, [None]

def __auto_handle_nan(array, ratio_unique_length_max, ratio_nan_length_max, 
                      percent_deviation_std_max):
            
    #Get ratio of unique values vs length
    nb_unique_values = array.unique.size
    length_array = array.size
    ratio_unique_length = nb_unique_values / length_array
    #Get ratio of nan / length of array 
    ratio_nan_length = np.count_nonzero(np.isnan(array))
    # Get median and mean
    array_mean = np.mean(array)
    array_median = np.median(array)
    array_std = np.std(array)
    
    cond_is_int = (array.dtype == 'int64' )
    cond_is_float = (array.dtype == 'float64' )
    cond_is_object = ( array.dtype == 'object' )
    
    # If data is categorical
    if cond_is_int or cond_is_object:
        if ratio_nan_length > ratio_nan_length_max:
            return __make_into_new_class(array)
        # not sure if categorical
        elif ratio_unique_length > ratio_unique_length_max:
            return __remove_all_nan(array)
        #is likely categorical
        else:
            return __make_into_new_class(array) 
    
    # If data is continuous
    elif cond_is_float:
        if ratio_unique_length > ratio_unique_length_max:
            return __remove_all_nan(array)
        elif np.abs(array_mean - array_median) < (array_std * percent_deviation_std_max):
            return __replace_by_mean(array)
        else:
            return __replace_by_median(array)
    else:
        exception_text = ('The datatype of the array given as argument is invalid.'
                         'Accepted datatypes: int64, float64, object.'
                        'datatype given: {}')
        raise Exception(exception_text.format(array.dtype))

def replace_nan_in_vector(array, mode="auto", ratio_unique_length_max = 0.2, 
                          ratio_nan_length_max=0.7, 
                          percent_deviation_std_max = 0.1):
    """
    

    Parameters
    ----------
    array : np.ndarray (n,1)
        DESCRIPTION. Input data to process

    Returns
    -------
    new values; index of values removed

    """
    if(array.shape[1] != 1):
         raise Exception('The array should only have one column. The number of columns was: {}'.format(array.shape[1])) 
    
    if mode == "mean":
        new_values, indexes_removed = __replace_by_mean(array)
    elif mode == "median":
        new_values, indexes_removed = __replace_by_median(array)
    elif mode == "remove":
        new_values, indexes_removed = __remove_all_nan(array)
    elif mode == "newclass":
        new_values, indexes_removed = __make_into_new_class(array)
    elif mode == "auto":
        new_values, indexes_removed = __auto_handle_nan(array, 
                                                        ratio_unique_length_max, 
                                                        ratio_nan_length_max, 
                                                        percent_deviation_std_max)
    else:
        raise Exception('The mode given as argument is invalid. Only remove, mean, median, newclass & auto are accepted. Mode given: {}'.format(mode))
    
    return new_values, indexes_removed