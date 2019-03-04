import io
import pickle as pk
import numpy as np
import time
from simulationManager import simulationManager as simMan
#from dataHandler2 import dataHandler2 as dataHandler

class test:
    def __init__(self):
        pass

    def threshold(self,input_array):
        x = np.less(input_array, 1e-12)
        
             
        return x
    
    def listSpeed(self,list1):
        x = 1
        t1 = time.time()
        if len(list1) > 0:  
            for entry in list1:
                x += x**2
        t2 = time.time()
        print(t2-t1)
        
        return x





mylist = []

tst = test()
out = tst.listSpeed(mylist)

vec = np.array([[1,1,1],[1,1,1],[1,1,1]])
print(vec.sum())