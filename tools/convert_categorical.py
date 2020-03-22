#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:53:20 2020

@author: alexandreguichet
"""
import pandas as pd

def convert_cat(df):
    #Convert categorical
    if isinstance(df, pd.DataFrame):
        cat_columns = df.select_dtypes(['category', 'object']).columns
        df[cat_columns] = df[cat_columns].astype('category').apply(lambda x: x.cat.codes)
    return df
