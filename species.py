#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import pi

## Class
class species:
    ## Main Methods
    def __init__(self,**kwargs):
        if 'nq' in kwargs:
            self.nq = kwargs['nq']
        else:
            self.nq = 1
            
        if 'q' in kwargs:
            self.q = kwargs['q']
        else:
            self.q = 1
            
        if 'mq' in kwargs:
            self.mq = kwargs['mq']
        else:
            self.mq = 1
            
        if 'qtype' in kwargs:
            if kwargs['qtype'] == "proton":
                self.qtype = 'proton'
                self.q = 1
                self.mq = 1836.152672197
            elif kwargs['qtype'] == "electron":
                self.qtype = 'electron'
                self.q = -1
                self.mq = 1
            else:
                self.qtype = 'custom'
    

        self.a = self.q/self.mq    #define mass-charge ratio alpha
        self.E = np.zeros((self.nq,3),dtype=np.float) 
        self.B = np.zeros((self.nq,3),dtype=np.float) 
        self.F = np.zeros((self.nq,3),dtype=np.float) 
        self.vel = np.zeros((self.nq,3),dtype=np.float)
        self.pos = np.zeros((self.nq,3),dtype=np.float)
        self.energy = 0.
        self.cm = np.array([0,0,0])
        
        ## Physical constants
        self.mu0 = 1
        self.ep0 = 1
        self.q0 = 1
        
        
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
    
    def makeSI(self):
        mElectron = 1/1822.8884845
        
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
        self.mq = self.mq * mElectron
        