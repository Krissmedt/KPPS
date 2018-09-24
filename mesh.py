#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import pi

## Class
class mesh:
    ## Main Methods
    def __init__(self,**kwargs):
        if 'box' in kwargs:
            self.params = kwargs['box']
            self.xlim = self.params['xlim']
            self.ylim = self.params['ylim']
            self.zlim = self.params['zlim']
            self.pos_init = self.box_pos
        else:
            self.xlim = [0,1]
            self.ylim = [0,1]
            self.zlim = [0,1]
            self.pos_init = self.nope
            
        if 'resolution' in kwargs:
            if len(kwargs['resolution']) == 1:
                self.xres = kwargs['resolution'][0]
                self.yres = kwargs['resolution'][0]
                self.zres = kwargs['resolution'][0]
                
            elif len(kwargs['resolution']) == 3:
                self.xres = kwargs['resolution'][0]
                self.yres = kwargs['resolution'][1]
                self.zres = kwargs['resolution'][2]

            self.dx = abs(self.xlim[1] - self.xlim[0])/self.xres
            self.dy = abs(self.ylim[1] - self.ylim[0])/self.yres
            self.dz = abs(self.zlim[1] - self.zlim[0])/self.zres
            
        elif 'dh' in kwargs:
            if len(kwargs['resolution']) == 1:
                self.dx = kwargs['dh'][0]
                self.dy = kwargs['dh'][0]
                self.dz = kwargs['dh'][0]
                
            elif len(kwargs['resolution']) == 3:
                self.dx = kwargs['dh'][0]
                self.dy = kwargs['dh'][1]
                self.dz = kwargs['dh'][2]

            self.xres = abs(self.xlim[1] - self.xlim[0])/self.dx
            self.yres = abs(self.ylim[1] - self.ylim[0])/self.dy
            self.zres = abs(self.zlim[1] - self.zlim[0])/self.dz
            
        else:
            self.xres = 1
            self.yres = 1
            self.zres = 1
            self.pos_init = self.nope
    
        self.nn = (self.xres+1)*(self.yres+1)*(self.zres+1) #calculate no. of cell nodes
        self.pos = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.q = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.E = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)
        self.B = np.zeros((3,self.xres+1,self.yres+1,self.zres+1),dtype=np.float)

        self.pos_init()
    
    def box_pos(self):
        for xi in range(0,self.xres+1):
            self.pos[0,xi,:,:] = self.xlim[0] + self.dx * xi
        for yi in range(0,self.yres+1):
            self.pos[1,:,yi,:] = self.ylim[0] + self.dy * yi
        for zi in range(0,self.yres+1):
            self.pos[2,:,:,zi] = self.zlim[0] + self.dz * zi
            
    def nope(self):
        return
            
    ## Additional Methods
