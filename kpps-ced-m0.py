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
def coulomb(q2,pos1,pos2):
    rpos = pos1-pos2
    r = sqrt(fsum(rpos**2))
    rUnit = rpos/r
    
    Ec = 1/(4*pi*ep0) * q2/r**2 * rUnit
    return Ec

def eFieldCont(pos,**kwargs):
    Ef = np.zeros(3, dtype=np.float)
    boost = 1
    eftype = "sPenning"
    
    if "ftype" in kwargs:
        eftype = kwargs["ftype"]
    
    if "boost" in kwargs:
        boost = kwargs["boost"]
    
    if eftype == "sPenning":
        Ef[0] = -pos[0] * boost
    elif eftype == "custom":
        if "F" in kwargs:
            customF = kwargs["F"]
            Ef = customF(pos,boost)
            
    return Ef
    
    
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
    
    k = 1
    error = True
    append = ""
    while error == True:
        try:
            os.mkdir("./" + foldername + append)
            error = False
        except FileExistsError:
            append = "(" + str(k) + ")"
            error = True
            k = k+1
            
    foldername = foldername + append
    
    return foldername

def writePData(foldername,tstep,tsteps,positionArray):
    filename = "./" + foldername + "/" + str(ts) + ".vtu"
    vtk_writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
    
    if tstep == (tsteps-1):
        vtk_writer.writePVD("./" + foldername + "/" + "run" + ".pvd")
 


## Problem variables
#mp = 1.67262189821*10**(-27)
mp = 2000
nq = 10
ndim = 2

te = 20
tsteps = 2000
samples = floor(tsteps/5)
tsample = floor(tsteps/samples)
dt = te/tsteps
tArray = np.zeros(tsteps+1,dtype=np.float)
xArray = np.zeros((tsteps+1,nq),dtype=np.float)

pos = np.zeros((nq,3),dtype=np.float)
vel = np.zeros((nq,3),dtype=np.float)
Ee = np.zeros((nq,3),dtype=np.float)

vtk_writer = pt.VTK_XML_Serial_Unstructured()
results = []

foldername = mkDataDir(ndim,"D_",nq,"p_",te,"s_",tsteps,"k")

pos = randPos(pos,ndim)
print(pos)
#pos = np.array([[1.,0.],[0.,0.]])


## Acceleration and velocity initialisation
#Update acceleration
for pii in range(0,nq):
    #Initialise acceleration (at n=0)
    for pjj in range(0,nq):
        if pii==pjj:
            continue
        Ee[pii,:] = Ee[pii,:] + coulomb(1,pos[pii,:],pos[pjj,:])
        
    Ee[pii,:] = Ee[pii,:] + eFieldCont(pos[pii,:],
                                       ftype="sPenning",
                                       boost=5000)
    Fe = Ee*1
    
    Fe = Fe*q0**2
    a = Fe[pii,:]/mp
    
    #Initialise velocity (at n=1/2)
    vel[pii,:] = vel[pii,:] + dt/2*a


## Main time loop
for ts in range(1,tsteps+1):
    tArray[ts] = dt*ts
    Ee = np.zeros((nq,3),dtype=np.float)
    for pii in range(0,nq):
        xArray[ts,pii] = pos[pii,0]
        
        #Update position
        pos[pii,:] = pos[pii,:] + dt*vel[pii,:]
        
        #Update acceleration
        for pjj in range(0,nq):
            if pii==pjj:
                continue
            Ee[pii,:] = Ee[pii,:] + coulomb(1,pos[pii,:],pos[pjj,:])
            
        Ee[pii,:] = Ee[pii,:] + eFieldCont(pos[pii,:],
                                           ftype="sPenning",
                                           boost=5000)
        Fe = Ee*1
        
        Fe = Fe*q0**2
        a = Fe[pii,:]/mp

        
        vel[pii,:] = vel[pii,:] + dt*a    #Update velocity

  
    if ts % tsample == 0 or ts == (tsteps-1):
        writePData(foldername,ts,tsteps,pos)


plt.plot(tArray,xArray)
plt.axis([0,20,-2,2])
