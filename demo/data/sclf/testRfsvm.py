import numpy as np
import Rfsvm

a = np.random.rand(20,20)
b = np.array(range(0,20), dtype=np.float64) % 2

Rfsvm.setDefaultParams()
Rfsvm.loadData(a,b);
Rfsvm.onlyTrain();
Rfsvm.loadData(a,b);
results = Rfsvm.onlyTest()
print results
