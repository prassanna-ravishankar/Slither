# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:06:50 2016

@author: prassanna
"""
import numpy as np

test_file = "sample_predict.txt"
bla = open(test_file,"r").readlines()
pred_results = np.array([[float(b) for b in c.split("\t")[:-1]] for c in bla])