import io
import pickle as pk
import numpy as np
from simulationManager import simulationManager as simMan
#from dataHandler2 import dataHandler2 as dataHandler

def hook1(x):
    x+=5
    
    return x



class test:
    def __init__(self,hooklist):
        self.methods = []
        self.hooklist = hooklist

    def run(self):
        for hook in self.hooklist:
            try:
                self.methods.append(getattr(self,hook))
            except TypeError:
                self.methods.append(hook)
            
          
        x=0
        for method in self.methods:
            x = method(x)
        print(x)
             
    def hook2(self,x):
        x+=5
        
        return x





hooklist = [hook1, 'hook2']

tst = test(hooklist)

tst.run()