# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:52:56 2016

@author: prassanna
"""

import numpy as np
import subprocess
from sklearn import datasets, preprocessing
import random
#import rfsvm

rf_release = "/home/prassanna/Development/workspace/Sherwood2/bin/rf"
folder_dump = "/home/prassanna/Development/workspace/Sherwood_bak/bin/"

datasets.load_digits(n_class=2)
bla = datasets.load_digits(n_class=2)

#Sample DAta
a = bla['data']
b = bla['target']
bla = zip(a,b)
random.shuffle(bla)
a[:],b[:] = zip(*bla)

#Require Array to be specific way
X_train = np.array(a[:len(a)/2], dtype=np.float64);
Y_train = np.array(b[:len(b)/2], dtype=np.float64);
Y_train.shape  = (len(Y_train),)
cooltrain = np.column_stack((Y_train, X_train))
np.savetxt(folder_dump+"sample_train.txt", cooltrain, delimiter ="\t")

X_test = np.array(a[len(a)/2:], dtype=np.float64);
Y_test = np.array(b[len(b)/2:], dtype=np.float64);
cooltrain = np.column_stack((Y_test, X_test))
np.savetxt(folder_dump+"sample_test.txt", cooltrain, delimiter ="\t")
#rfsvm.setDefaultParams()
#bla = rfsvm.loadData(X_train,Y_train);
#print bla
#rfsvm.onlyTrain();
#rfsvm.saveModel('model.xyz')