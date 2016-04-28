# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:52:56 2016

@author: prassanna
"""

import numpy as np
import subprocess
from sklearn import datasets, preprocessing
import rfsvm
import random

rf_release = "/home/prassanna/Development/workspace/Sherwood2/bin/rf"

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

#cooltrain = np.column_stack((Y_train, X_train))
#np.savetxt("sample_train.txt", cooltrain, delimiter ="\t")

traindata = np.loadtxt('_400traindata.csv')
X_train = traindata[:,1:]
Y_train = traindata[:,0]

Y_train.shape  = (len(Y_train),)
rfsvm.setDefaultParams()
bla = rfsvm.loadData(X_train,Y_train);
#print bla
rfsvm.onlyTrain();
rfsvm.saveModel('model.xyz')