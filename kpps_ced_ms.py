################### Kris' Plasma Particle Simulator (KPPS) ####################
"""
Coulomb electrodynamic, magnetostatic 'ced-ms': LINUX
"""

## Dependencies
import numpy as np
import matplotlib.pyplot as plt
from species import species
from simulationManager import simulationManager
from dataHandler import dataHandler
from kpps_methods import *
from kpps_analysis import kpps_analysis

## Problem variables
simSettings = {'ndim':3,'tEnd':10,'tSteps':100}
particleSettings = {'nq':10,'qtype':'proton'}

eFieldSettings = {'ftype':'sPenning', 'magnitude':1000}
bFieldSettings = {'uniform':[0,0,1], 'magnitude':100}
analysisSettings = {'electricField':eFieldSettings,
                    'interactionModelling':'intra',
                    'magneticField':bFieldSettings,
                    'timeIntegration':'boris'}

dataSettings = {'sampleRate':5,'foldername':'simple'}


## Initialise required modules
particles = species(**particleSettings)
sim = simulationManager(**simSettings)
analyser = kpps_analysis(**analysisSettings)
dHandler = dataHandler(particles,sim,**dataSettings)


## Initialise results arrays and particle positions
xArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)
yArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)
zArray = np.zeros((tsteps+1,particles.nq),dtype=np.float)

randPos(particles,sim.ndim)
#particles.vel[:,0] = np.ones(particles.nq,dtype=np.float)

xArray[0,:] = particles.pos[:,0]
yArray[0,:] = particles.pos[:,1]
zArray[0,:] = particles.pos[:,2]


## Main time loop
for ts in range(1,tsteps+1):
    analyser.electric(particles)
    analyser.magnetic(particles)
    analyser.timeIntegrator(particles,sim)
    sim.updateTime()

    xArray[ts,:] = particles.pos[:,0]
    yArray[ts,:] = particles.pos[:,1]
    zArray[ts,:] = particles.pos[:,2]
  
    dHandler.writeData(particles,sim)
        

## Plot position results singularly
plt.figure(1)
plt.plot(sim.tArray,xArray)

plt.figure(2)
plt.plot(sim.tArray,yArray)

plt.figure(3)
plt.plot(sim.tArray,zArray)

plt.figure(4)
plt.plot(xArray,yArray)
