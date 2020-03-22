#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:10:08 2020

@author: Alexandre Guichet
"""

import pandas as pd
import numpy as np

import time
import warnings
from tools.load import load_file

from mutual_information import mutual_information as mi

class MIFS():
    
    def __init__(self):
        self.df = None
        self.mi = None
        self.mifs = dict()
        self.mi_time = 0
        self.features = None
        self.labels = None
        
        self._cost_function = self._min_redundancy_cost_function
        self._feature_selector = self._max_mi_min_redundancy
        self._lambda = 1
        
        
        
    def load_file(self, path):
        self.df= load_file(path)
               
    def separate_labels(self, labels = list()):
       self.labels = self.df[labels].copy()
       self.features = self.df.drop(columns = labels)
    
    def select_n_features(self, n = 5, **kwargs):
       """
       Select the k-best features for each labels. 
       """ 
       if isinstance(self.labels, pd.DataFrame):
           joint_data = self.labels.join(self.features)     
       else:
           joint_data = np.hstack([self.labels, self.features])
          
       mi_time = time.time()
       self.mifs["labels"] = mi(joint_data, self.labels, **kwargs)
       self.mi_time = time.time() - mi_time
       
       df_temps = self.mifs["labels"].drop(index = self.mifs["labels"].columns)
       
       temp = pd.DataFrame()
       
       for c in df_temps:
           temp = temp.append(df_temps.nlargest(n = n, columns = c)).drop_duplicates()
       
       self.mifs["threshold"] = temp
            
       return self._select_features(**kwargs)

    
    def select_features_threshold(self, n = 5):
       """
       Select the k-best features for each labels. 
       """ 
       
       return self._select_features()
    
    def _select_features(self, **kwargs):
        _, mx = self.features.shape
        _, my = self.labels.shape
        
        candidate_names = self.mifs["threshold"].index.values
        
        time_appx = (len(candidate_names) + my)**2 * self.mi_time/my/mx
        
        if time_appx >= 3600:
            warnings.warn("The feature selection for {:d} feature will take approximately {:.2g}hours to compute. \n".format(len(candidate_names), time_appx/3600), Warning)
        elif time_appx >= 60:
            warnings.warn("The feature selection for {:d} feature will take approximately {:.2g}mins to compute. \n".format(len(candidate_names), time_appx/60), Warning)
        
        joint_data = self.labels.join(self.features[candidate_names])
        
        self.mifs["all"] = mi(joint_data, joint_data, **kwargs)
        self.mifs["all"] = self.mifs["all"]/self.mifs["all"].max(axis = 0)
        
        return self.feature_selector()
    
    def _max_mi_min_redundancy(self):
        _, mx = self.features.shape
        _, my = self.labels.shape

        redundancies = self.mifs["all"][self.mifs["all"][self.mifs["all"].columns[my:]].iloc[my:] >= 0.2]
        
        self.mifs["selected"] = self.mifs["all"][self.labels.columns].drop(index = self.labels.columns)
        
        trim = redundancies.where(np.triu(np.ones(redundancies.shape)).astype(np.bool)).stack().reset_index()
        trim.columns = ["Rows", "Columns", "Values"]
        trim = trim.drop(index = trim[trim["Values"] == 1].index)
        
        uniques = {tuple(item) for item in map(sorted, trim[["Rows", "Columns"]].values)}
        
        for i in uniques:
            try: 
                Cz, Cx = self.cost_function(i)
                if Cz > Cx:
                    self.mifs["selected"] = self.mifs["selected"].drop(i[1])
                elif Cx > Cz: 
                    self.mifs["selected"] = self.mifs["selected"].drop(i[0])
                else: 
                    print("that should never happen as mutual information is not symetric, you are lucky! Well done!")
            except KeyError:
                pass #The Value was already removed
        
        self.mifs["selected"] = self.mifs["selected"].sort_values(self.mifs["selected"].columns[0], ascending = False)
        return self.mifs  
    
    def cost_function(self, *args):
        print("here with")
        return self._cost_function(*args)
        
    @property
    def Lambda(self):
        return self._lambda
    
    @Lambda.setter
    def Lambda(self, value):
        self._lambda = value     
  
    @property
    def feature_selector(self):
        return self._feature_selector
    
    @feature_selector.setter
    def feature_selector(self, value):
        self._feature_selector = value
    
    def _min_redundancy_cost_function(self, i):   
        _, my = self.labels.shape
        
        Cz = self.mifs["selected"].loc[i[0]].max()/((self.mifs["all"][self.mifs["all"].columns[my:]].loc[i[0]].drop(i[0])).mean()**self.Lambda)
        Cx = self.mifs["selected"].loc[i[1]].max()/((self.mifs["all"][self.mifs["all"].columns[my:]].loc[i[1]].drop(i[1])).mean()**self.Lambda)        
        return Cz, Cx