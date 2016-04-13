# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:47:34 2016

@author: prassanna
"""
import numpy as np
import random

def create_random_3d(above,negate=False):
    x,y,z = random.randint(0,10), random.randint(0,10),random.randint(0,10)
    if(above):
        x,y,z = -x,-y,-z;        
    if(negate):
        return str(int(above)+2), str(-x),str(-y),str(-z)
    return str(int(above)), str(x),str(y),str(z)
    
    
def other_random_3d(above):
    x,y,z = random.randint(0,10), random.randint(0,10), random.randint(0,10)
    x,y,z = x+20, y+20,z+20;
    if(above):
        x,y,z = x+10,y+10,z+10;        
    return str(int(above)+1), str(x),str(y),str(z)    
    
def tuple_str(tup):
    tup_str = str(tup[0])
    for t in tup[1:]:
        tup_str = tup_str + "\t"+str(t)
    return tup_str
    
nums = [a%2 == 0 for a in range(0,200)]


#random.shuffle(nums)

#sample_data = ['\t'.join(create_random_3d(n))+'\n' for n in nums]
#sample_data_str = ['\t'.join(s)+'\n' for s in sample_data]
data = ['\t'.join(create_random_3d(n))+'\n' for n in random.sample(nums,len(nums))]
#data.extend(['\t'.join(create_random_3d(n, True))+'\n' for n in random.sample(nums,len(nums))])

#data = ['\t'.join(create_random_3d(n))+'\n' for n in random.sample(nums,len(nums))]
#data.extend(['\t'.join(create_random_3d(n, True))+'\n' for n in random.sample(nums,len(nums))])
open('sample_train.txt','w').writelines(random.sample(data,len(data)/2))
open('sample_test.txt','w').writelines(random.sample(data,len(data)/2))
