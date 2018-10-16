#!/usr/bin/env python3

## Dependencies
import numpy as np

class mesh:
    def __init__(self,**kwargs):
        ## Default values
        self.domain_type = 'box'
        self.cells = 1
        
        self.xlimits = np.array([0,1],dtype=np.float)
        self.ylimits = np.array([0,1],dtype=np.float)
        self.zlimits = np.array([0,1],dtype=np.float)
        
        self.dh = 1
        self.dx = 1
        self.dy = 1
        self.dz = 1
        
        self.res = 1
        self.xres = 1
        self.yres = 1
        self.zres = 1
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
    
        self.nn = (self.xres+1)*(self.yres+1)*(self.zres+1) #calculate no. of cell nodes
        self.pos = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.q = np.zeros((self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.E = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.B = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
