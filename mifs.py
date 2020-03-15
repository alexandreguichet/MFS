#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:10:08 2020

@author: Alexandre Guichet
"""

import pandas as pd
from mutual_information import mutual_information

class MIFS():
    
    def __init__(self):
        self.df = None
        self.mi = None
        self.mifs = dict()
    
        
    def load_pickle(self, path):
        self.df= pd.read_pickle(path)
        
    def select_k_features(self, features, labels, k = 5):
       """
       Select the k-best features for each labels. 
       """ 
      
        
       return self.mifs
   
   
    def _select_features(self, df, mi):
        pass
        