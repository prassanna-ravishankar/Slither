# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:52:56 2016

@author: prassanna
"""

import numpy as np
import subprocess
from sklearn import datasets, preprocessing
#import rfsvm
import random

#rf_release = "/home/prassanna/Development/workspace/Sherwood2/bin/rf"

datasets.load_digits(n_class=2)
bla = datasets.load_digits(n_class=2)

#Sample DAta
a = bla['data']
b = bla['target']

newbla = zip(a,b)
random.shuffle(newbla)
a,b = zip(*newbla)

#Require Array to be specific way
X_train = np.array(a[:len(a)/2], dtype=np.float64);
Y_train = np.array(b[:len(b)/2], dtype=np.float64);

bla = np.column_stack((Y_train, X_train))
np.savetxt('sample_train.txt',bla,delimiter = '\t')

