#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import pi

## Class
class species:
    ## Attributes
    nq = 1
    q = 1
    mq = 1
    
    qtype = 'custom'
    
    ## Physical constants
    mu0 = 1
    ep0 = 1
    q0 = 1

    ## Main Methods
    def __init__(self,**kwargs):
        if 'nq' in kwargs:
            self.nq = kwargs['nq']
        if 'q' in kwargs:
            self.q = kwargs['q']
        if 'mq' in kwargs:
            self.mq = kwargs['mq']
            
        if 'qtype' in kwargs:
            if kwargs['qtype'] == "proton":
                self.qtype = 'proton'
                self.q = 1
                self.mq = 1836.152672197
            elif kwargs['qtype'] == "electron":
                self.qtype = 'electron'
                self.q = -1
                self.mq = 1
        
        
        self.E = np.zeros((self.nq,3),dtype=np.float) 
        self.B = np.zeros((self.nq,3),dtype=np.float) 
        self.F = np.zeros((self.nq,3),dtype=np.float) 
        self.vel = np.zeros((self.nq,3),dtype=np.float)
        self.pos = np.zeros((self.nq,3),dtype=np.float)
        
        
    ## Additional Methods
    def makeSI(self):
        mElectron = 1/1822.8884845
        
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
        self.mq = self.mq * mElectron
        