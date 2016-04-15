import numpy as np
import rfsvm as Rfsvm

#Sample DAta
a = np.random.rand(20,20)
b = np.array(range(0,20), dtype=np.float64) % 2

#Require Array to be specific way
X = np.array(a, dtype=np.float64);
Y = np.array(b, dtype=np.float64);
Y.shape  = (len(Y),)

#Actual Operations
Rfsvm.setDefaultParams()
Rfsvm.loadData(X,Y);
Rfsvm.onlyTrain();
Rfsvm.loadData(X,Y);
results = Rfsvm.onlyTest()
print results
