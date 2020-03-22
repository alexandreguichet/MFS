#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:44:22 2020

@author: Alexandre Guichet
"""
import re 
import pandas as pd

def load_file(path):
    try:
        ftype = re.search("\.\w+", path).group()
    except:
        print("Could not understand the path clearly. Returning without loading dataset")
    
    if ftype == ".pkl":
        return pd.read_pickle(path)
    elif ftype == ".csv":
        return pd.read_csv(path)
    elif ftype == ".xlsx" or ftype == ".xls":
        return pd.read_excel(path)
