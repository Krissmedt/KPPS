################### Kris' Plasma Particle Simulator (KPPS) ####################
"""
Coulomb electrodynamic, magnetostatic 'ced-ms': LINUX
"""

## Dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import pyTools as pt
import random as rand
from kpps_methods import *
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

## Problem variables
#mp = 1.67262189821*10**(-27)
q = 1
mp = 2000
nq = 10
ndim = 3

te = 10
tsteps = 2000
samples = floor(tsteps/5)
tsample = floor(tsteps/samples)
dt = te/tsteps

writeData = True
if writeData == True:
    vtk_writer = pt.VTK_XML_Serial_Unstructured()
    foldername = pt.mkDataDir("ms_",ndim,"D_",nq,"p_",te,"s_",tsteps,"k")

tArray = np.zeros(tsteps+1,dtype=np.float)
xArray = np.zeros((tsteps+1,nq),dtype=np.float)
yArray = np.zeros((tsteps+1,nq),dtype=np.float)
zArray = np.zeros((tsteps+1,nq),dtype=np.float)

pos = np.zeros((nq,3),dtype=np.float)
vel = np.zeros((nq,3),dtype=np.float)
Ee = np.zeros((nq,3),dtype=np.float)



pos = randPos(pos,ndim)
xArray[0,:] = pos[:,0]
yArray[0,:] = pos[:,1]
zArray[0,:] = pos[:,2]


## Acceleration and velocity initialisation
k = dt*nq/(2*mp)
vel = vel


## Main time loop
for ts in range(1,tsteps+1):
    tArray[ts] = dt*ts
    
    E = eField(pos,ftype="sPenning",boost=5000)
    B = bField(pos,magnitude=500)
    

    update = boris(pos,vel,E,B,dt,mp)
    vel = update['vel']
    pos = update['pos']
    
    xArray[ts,:] = pos[:,0]
    yArray[ts,:] = pos[:,1]
    zArray[ts,:] = pos[:,2]
  
    if (ts % tsample == 0 or ts == (tsteps-1)) and writeData == True:
        pt.writePData(pos,ts,tsteps,
                      vtk_writer=vtk_writer,foldername=foldername)
        

plt.figure(1)
plt.plot(tArray,xArray)

plt.figure(2)
plt.plot(tArray,yArray)

plt.figure(3)
plt.plot(tArray,zArray)
