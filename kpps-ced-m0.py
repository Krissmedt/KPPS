################### Kris' Plasma Particle Simulator (KPPS) ####################
"""
Coulomb electrodynamic, non-magnetic 'ced-m0': LINUX
"""

## Dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import pyTools as pt
import random as rand
from copy import copy, deepcopy
from math import exp, sqrt, floor, fabs, fsum, pi, e

## Global constants
global mu0
global ep0
global q0

#mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
#ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
#q0 = 1.602176620898*10**(-19) #Elementary charge (C)

mu0 = 1
ep0 = 1
q0 = 1

## Methods
def coulomb(q1,q2,pos1,pos2):
    rpos = pos1-pos2
    r = sqrt(fsum(rpos**2))
    rUnit = rpos/r
    
    Fc = 1/(4*pi*ep0) * q1*q2/r**2 * rUnit
    
    return Fc

def randPos(pos,ndim):
    for nd in range(0,ndim):
        for xi in range(0,len(pos)):
            pos[xi,nd] = rand.random()
    
    return pos

def mkDataDir(*args):
    foldername = ""
    for a in range(0,len(args)):
        try:
            foldername = foldername + args[a]
        except TypeError:
            foldername = foldername + str(args[a])
        except:
            print(
                  "Unexptected error, did you enter a string or otherwise" +
                  " parsable value? Code:", sys_exc_info()[0])
            
    os.mkdir("./" + foldername)
    return foldername

def writePData(foldername,tstep,tsteps,positionArray):
    filename = "./" + foldername + "/" + str(ts) + ".vtu"
    vtk_writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
    
    if tstep == (tsteps-1):
        vtk_writer.writePVD("./" + foldername + "/" + "run" + ".pvd")
 


## Problem variables
#mp = 1.67262189821*10**(-27)
mp = 2000
nq = 20
ndim = 2

te = 10
tsteps = 1000
samples = floor(tsteps/5)
tsample = floor(tsteps/samples)
dt = te/tsteps
tArray = np.zeros(tsteps,dtype=np.float)

pos = np.zeros((nq,3),dtype=np.float)
vel = np.zeros((nq,3),dtype=np.float)
Fe = np.zeros((nq,3),dtype=np.float)

vtk_writer = pt.VTK_XML_Serial_Unstructured()
results = []

foldername = mkDataDir(ndim,"D_",nq,"p_",te,"s_",tsteps,"k")

pos = randPos(pos,ndim)
#pos = np.array([[1.,0.],[0.,0.]])

for ts in range(0,tsteps):
    tArray[ts] = dt*ts
    for pii in range(0,nq):
        for pjj in range(0,nq):
            if pii==pjj:
                continue
            Fe[pii,:] = Fe[pii,:] + coulomb(1,1,pos[pii,:],pos[pjj,:])
        
        Fe = Fe*q0**2
        a = Fe[pii,:]/mp

        vel[pii,:] = vel[pii,:] + dt*a
        pos[pii,:] = pos[pii,:] + dt*vel[pii,:]
        
    if ts % tsample == 0 or ts == (tsteps-1):
        writePData(foldername,ts,tsteps,pos)

