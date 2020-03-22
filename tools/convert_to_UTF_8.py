#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:59:21 2020

@author: alexandreguichet
"""

import csv

path = r"./datasets/IPODataFull.csv"
with open(path, 'r', encoding='utf-8', errors='ignore') as infile, open(path + 'final.csv', 'w') as outfile:
     inputs = csv.reader(infile)
     output = csv.writer(outfile)

     for index, row in enumerate(inputs):
         # Create file with no header
         if index == 0:
             continue
         output.writerow(row)    