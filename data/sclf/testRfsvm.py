
#all_train = np.column_stack((Y_train,X_train))
#np.savetxt("trainfile.txt",all_train,delimiter="\t")  
#subprocess.call([rf_release,"--train", "trainfile.txt","--model", "forestloc.out","--op_mode","train"," . "])

import numpy as np
import subprocess
from sklearn import datasets
import rfsvm
datasets.load_digits(n_class=2)
bla = datasets.load_digits(n_class=2)

#Sample DAta
a = bla['data']
b = bla['target']
X_test = np.array(a[len(a)/2:], dtype=np.float64);
Y_test = np.array(b[len(b)/2:], dtype=np.float64);
#all_test = np.column_stack((Y_test,X_test))
#cooltrain = np.column_stack((Y_test, X_test))
#np.savetxt("sample_test.txt", cooltrain, delimiter ="\t")

testdata = np.loadtxt('_400traindata.csv')
X_test = testdata[:,1:]
Y_test = testdata[:,0]


rfsvm.setDefaultParams()
bla = rfsvm.loadData(X_test,Y_test);
print bla
bla = rfsvm.loadModel('model.xyz')
print bla
results = rfsvm.onlyTest();
print "Woah", results
#np.savetxt("testfile.txt",all_test,delimiter="\t")  
#subprocess.call([rf_release,"--test", "testfile.txt","--model", "forestloc.out","--op_mode","test"," . "])