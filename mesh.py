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
        self.dv = 1
        
        self.res = 1
        self.xres = 1
        self.yres = 1
        self.zres = 1
        
        self.gather_count = 0
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
        self.setup()
        
        
    def setup(self):
        self.nn = (self.xres+1)*(self.yres+1)*(self.zres+1) #calculate no. of cell nodes
        self.pos = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.q = np.zeros((self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.rho = np.zeros((self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.phi = np.zeros((self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.E = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.CE = 0
        self.PE = np.zeros((self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.PE_sum = 0
        self.B = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        
        self.q_bk = np.zeros((self.q.shape),dtype=np.float)
        self.rho_bk = np.zeros((self.rho.shape),dtype=np.float)
        self.E_bk = np.zeros((self.E.shape),dtype=np.float)
        self.B_bk = np.zeros((self.B.shape),dtype=np.float)
