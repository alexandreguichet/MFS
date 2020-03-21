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
    # Check if data is in string format
    if pd.api.types.is_string_dtype (array):
        nan_indexes = np.argwhere(array == 'nan')
    else:
        nan_indexes = np.argwhere(np.isnan(array))[:,0]
    trimmed_array = np.delete(array, nan_indexes)
    return trimmed_array, nan_indexes

def __replace_by_mean(array):
    array_mean = np.nanmean(array)
    array = np.nan_to_num(array, nan=array_mean)
    return array, [None]
    
def __replace_by_median(array):
    array_median = np.nanmedian(array)
    array = np.nan_to_num(array, nan=array_median)
    return array, [None]

def __make_into_new_class(array):
    
    # Check if data is in string format
    if isinstance(array, object):
        #Convert to int categorical
        length_array = np.size(array)
        codes, uniques = pd.factorize(array.reshape(length_array)) # nan are interpreted as a string
    
    # Data is in numeral format
    else:
        length_array = np.size(array)
        codes, uniques = pd.factorize(array.reshape(length_array)) # all nan / none will be set to -1
        sorted_codes = np.sort(np.unique(codes))
        new_class_id = sorted_codes[-1] + 1
        codes[codes == -1] = new_class_id
        
    array = codes
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

def replace_nan_in_column(array, mode="auto", ratio_unique_length_max = 0.2, 
                          ratio_nan_length_max=0.7, 
                          percent_deviation_std_max = 0.1):
    """
    

    Parameters
    ----------
    array : np.ndarray (n,1)
        DESCRIPTION. Input data to process
        
    mode : str
        DESCRIPTION. Which operation to perform on the NaN. 
        Values accepted:
        remove, mean, median, newclass & auto

    Returns
    -------
    new_values : np.ndarray (m,1)
        DESCRIPTION. Processed Data

    indexes_removed : np.ndarray (k,1)
        DESCRIPTION. Indexes which were removed in the input array
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

