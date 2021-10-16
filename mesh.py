#!/usr/bin/env python3

## Dependencies
import numpy as np

class mesh:
    def __init__(self,**kwargs):
        ## Default values
        self.domain_type = 'box'
        self.cells = 8
        
        self.xlimits = np.array([-1,1],dtype=np.float)
        self.ylimits = np.array([-1,1],dtype=np.float)
        self.zlimits = np.array([-1,1],dtype=np.float)
        
        self.res = np.array([2,2,2])
        self.xres = 2
        self.yres = 2
        self.zres = 2
        
        self.dh = 1
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.dv = 1
        
        self.x, self.y, self.z = np.mgrid[-1:1:3*1j, 
                                          -1:1:3*1j, 
                                          -1:1:3*1j]
        
        self.gather_count = 0
        self.gmres_iters = 0
        
        self.Rx = np.zeros((3,3))
        self.Rv = np.zeros((3,3))
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
        self.setup()
        
        
    def setup(self):
        self.nn = (self.xres+1)*(self.yres+1)*(self.zres+1) #calculate no. of cell nodes
        self.pos = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.q = np.zeros((self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.rho = np.zeros((self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.phi = np.zeros((self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.E = np.zeros((3,self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.CE = 0
        self.PE = np.zeros((self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.PE_sum = 0
        self.B = np.zeros((3,self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        
        BC_vector = np.zeros((self.xres+2,self.yres+2,self.zres+2),dtype=np.float)
        self.BC_vector = np.ravel(BC_vector[1:-2,1:-2,1:-2])

        self.q_bk = np.zeros((self.q.shape),dtype=np.float)
        self.rho_bk = np.zeros((self.rho.shape),dtype=np.float)
        self.E_bk = np.zeros((self.E.shape),dtype=np.float)
        self.B_bk = np.zeros((self.B.shape),dtype=np.float)

