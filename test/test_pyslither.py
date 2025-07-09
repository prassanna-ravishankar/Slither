# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:52:56 2016
Modified heavily on Oct 19

This is a test end-to-end operation of slither_py
1. Loading of Numpy Data
2. Training a model
3. Serialization (i.e Saving the model)
4. Using the trained model to test against another half of data

@author: Prass, The Nomadic Chef
"""

# For the data
import numpy as np
import random
import os

# for the dataset
from sklearn import datasets, preprocessing

# for the random forest library
from slither_py import SlitherWrapper

LOAD_IF_EXISTS = True

folder_dump = os.getcwd()+"/test_output/"
folder_exists = False
data_exists = False
model_exists = False
if not os.path.exists(folder_dump):
    print("Creating folder : " + folder_dump)
    os.mkdir(folder_dump)
else:
    print("Using existing dump folder " + folder_dump)
    folder_exists = True
    if "sample_train.txt" and "sample_test.txt" in os.listdir(folder_dump):
        data_exists = True
    if "model.xyz" and os.listdir(folder_dump):
        model_exists = True

# Prepare data if not present
if not data_exists:
    print("Loading and shuffling fresh digits data")
    digits_data = datasets.load_digits(n_class=2)
    digits_data = list(zip(digits_data['data'], digits_data['target']))
    random.shuffle(digits_data)
    X, Y = zip(*digits_data)

    # Require Array to be specific way
    X_train = np.array(X[:int(len(X) / 2)], dtype=np.float64)
    Y_train = np.array(Y[:int(len(Y) / 2)], dtype=np.float64)
    Y_train.shape  = (len(Y_train),)
    cooltrain = np.column_stack((Y_train, X_train))
    np.savetxt(folder_dump+"sample_train.txt", cooltrain, delimiter ="\t")

    X_test = np.array(X[int(len(X) / 2):], dtype=np.float64)
    Y_test = np.array(Y[int(len(Y) / 2):], dtype=np.float64)
    cooltrain = np.column_stack((Y_test, X_test))
    np.savetxt(folder_dump+"sample_test.txt", cooltrain, delimiter ="\t")
else:
    print ("Loading existing data")
    cooltrain = np.loadtxt(folder_dump+"sample_train.txt")
    cooltest = np.loadtxt(folder_dump+"sample_test.txt")
    Y_train, X_train = cooltrain[:, 0], cooltrain[:, 1:]
    Y_test, X_test = cooltest[:, 0], cooltest[:, 1:]

# Initialize the model
my_slither = SlitherWrapper()
my_slither.setDefaultParams()

if model_exists and data_exists:
    # Trust the model only if we use the same training/testing data
    print("Loading existing model")
    my_slither.loadModel(folder_dump+"model.xyz")
else:
    print("Training a fresh model")
    # Training a model
    my_slither.loadData(X_train, Y_train)
    my_slither.onlyTrain()
    my_slither.saveModel(folder_dump + 'model.xyz')

# Testing a model - Always happens
my_slither.loadData(X_test, Y_test)
res_prob = my_slither.onlyTest()res_clf = np.argmax(res_prob, axis=1)
print("I got : " + str(np.sum(res_clf == Y_test)) + " correct out of : " + str(len(Y_test)))

