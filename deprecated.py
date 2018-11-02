#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:30:21 2018

@author: mn12kms
"""

import numpy as np
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
import time

def poisson_cube2nd_setup_old(self,species,fields,simulationManager,**kwargs):
    ## WON'T WORK FOR dx =/= dy =/= dz
    n = np.int(fields.res[0 ]+1)
    Dk = np.zeros((n,n),dtype=np.float)
    Ek = np.zeros((n**2,n**2),dtype=np.float)
    Fk = np.zeros((n**3,n**3),dtype=np.float)
    
    k = 2/(fields.dx + fields.dy + fields.dz)
    
    # Setup 1D FD matrix
    Dk[0,0] = -k
    Dk[0,1] = 1
    Dk[-1,-1] = -k
    Dk[-1,-2] = 1/1
    
    for i in range(1,n-1):
        Dk[i,i] = -k
        Dk[i,i-1] = 1/fields.dy
        Dk[i,i+1] = 1/fields.dy
    
    # Setup 2D FD matrix
    I = np.identity(n)/1
    
    Ek[0:n,0:n] = Dk
    Ek[0:n,n:(2*n)] = I
    Ek[(n-1)*n:,(n-1)*n:] = Dk
    Ek[(n-1)*n:,(n-2)*n:(n-1)*n] = I
    
    for i in range(n,((n-1)*n-1),n):
        Ek[i:(i+n),i:(i+n)] = Dk
        Ek[i:(i+n),(i-n):i] = I
        Ek[i:(i+n),(i+n):(i+2*n)] = I
        
    # Setup 3D FD matrix
    J = np.identity(n**2)/fields.dx
    
    Fk[0:n**2,0:n**2] = Ek
    Fk[0:n**2,n**2:(2*n**2)] = J
    Fk[(n-1)*n**2:,(n-1)*n**2:] = Ek
    Fk[(n-1)*n**2:,(n-2)*n**2:(n-1)*n**2] = J
    
    for i in range(n**2,((n-1)*n**2-1),n**2):
        Fk[i:(i+n**2),i:(i+n**2)] = Ek
        Fk[i:(i+n**2),(i-n**2):i] = J
        Fk[i:(i+n**2),(i+n**2):(i+2*n**2)] = J
    
    self.Fk = Fk
    return self.Fk