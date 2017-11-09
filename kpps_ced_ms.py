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
from species import species
from simulationManager import simulationManager
from dataHandler import dataHandler
from kpps_methods import *
from copy import copy, deepcopy
from math import exp, sqrt, floor, fabs, fsum, pi, e

## Problem variables
ndim = 3
nq = 10

te = 10
tsteps = 100
sampleEvery = 5


## Initialise required modules
particles = species(nq=nq,qtype="proton")
sim = simulationManager(tEnd=te,tSteps=tsteps,dimensions=ndim)
dHandler = dataHandler(particles,sim, 
                       sampleRate=sampleEvery,foldername='simple')

tArray = np.zeros(tsteps+1,dtype=np.float)
xArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)
yArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)
zArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)

randPos(particles,sim.ndim)

xArray[0,:] = particles.pos[:,0]
yArray[0,:] = particles.pos[:,1]
zArray[0,:] = particles.pos[:,2]


## Main time loop
for ts in range(1,tsteps+1):

    eField(particles,ftype="sPenning",boost=5000)
    bField(particles,magnitude=500)
    boris(particles,sim)
    sim.updateTime()

    xArray[ts,:] = particles.pos[:,0]
    yArray[ts,:] = particles.pos[:,1]
    zArray[ts,:] = particles.pos[:,2]
  
    dHandler.writeData(particles,sim)
        

plt.figure(1)
plt.plot(sim.tArray,xArray)

plt.figure(2)
plt.plot(sim.tArray,yArray)

plt.figure(3)
plt.plot(sim.tArray,zArray)