#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:32:06 2020

@author: alexandreguichet
"""

from mifs import MIFS

mifs = MIFS()
mifs.load_pickle("titanic_train.pkl")