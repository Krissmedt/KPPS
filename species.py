#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import pi

## Class
class species:
    ## Main Methods
    def __init__(self,**kwargs):
        ## Default values
        self.name = 'none'
        self.nq = 1
        self.q = 1
        self.mq = 1
        self.a = 1    #charge-to-mass ratio alpha
        self.energy = 0.
        self.cm = np.array([0,0,0])
        
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
            
        if 'qtype' in self.params:
            if self.params['qtype'] == "proton":
                self.qtype = 'proton'
                self.q = 1
                self.mq = 1836.152672197
            elif self.params['qtype'] == "electron":
                self.qtype = 'electron'
                self.q = -1
                self.mq = 1
            else:
                self.qtype = 'custom'

        try:
            self.mq = self.params['q'] / self.params['a']
        except KeyError:
            pass
        
        try:
            self.q = self.params['a'] * self.params['mq']
        except KeyError:
            pass
        
        try:
            self.a = self.params['q'] / self.params['mq']
        except KeyError:
            pass
        
        ## Initialise number-dependent arrays
        self.E = np.zeros((self.nq,3),dtype=np.float) 
        self.B = np.zeros((self.nq,3),dtype=np.float) 
        self.F = np.zeros((self.nq,3),dtype=np.float) 
        self.vel = np.zeros((self.nq,3),dtype=np.float)
        self.pos = np.zeros((self.nq,3),dtype=np.float)
        
    ## Additional Methods
    def toVector(self,storageMatrix):
        self.vector = []
        for pii in range(0,self.nq):
            self.vector.append(storageMatrix[pii,0])
            self.vector.append(storageMatrix[pii,1])
            self.vector.append(storageMatrix[pii,2])
            
        self.vector = np.array(self.vector)
        return self.vector
    
    def toMatrix(self,vector):
        self.matrix = np.zeros((self.nq,3),dtype=np.float)
        for pii in range(0,self.nq):
            self.matrix[pii,0] = vector[3*pii]
            self.matrix[pii,1] = vector[3*pii+1]
            self.matrix[pii,2] = vector[3*pii+2]

        return self.matrix
    
