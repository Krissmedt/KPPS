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
from caseHandler import caseHandler
from kpps_analysis import kpps_analysis

## Simulation settings
simSettings = {'ndim':3,'tEnd':10,'tSteps':100}
speciesSettings = {'nq':1,'qtype':'proton'}
caseSettings = {'distribution':{'random':''},
                'explicitSetup':{'velocities':np.array([0.,1.,0.])}}

eFieldSettings = {'ftype':'sPenning', 'magnitude':1000}
bFieldSettings = {'uniform':[0,0,1], 'magnitude':1000}
analysisSettings = {'electricField':eFieldSettings,
                    'interactionModelling':'intra',
                    'magneticField':bFieldSettings,
                    'timeIntegration':'boris'}

dataSettings = {'write':{'sampleRate':1,'foldername':'simple'},
                'record':{'sampleRate':1},
                'plot':{'tPlot':'xyz','sPlot':''}}


## Load required modules
particles = species(**speciesSettings)
sim = simulationManager(**simSettings)
case = caseHandler(particles,**caseSettings)
analyser = kpps_analysis(**analysisSettings)
dHandler = dataHandler(particles,sim,case,**dataSettings)


## Main time loop
dHandler.run(particles,sim)
for ts in range(1,sim.tSteps+1):
    sim.updateTime()
    analyser.electric(particles)
    analyser.magnetic(particles)
    analyser.timeIntegrator(particles,sim)
    dHandler.run(particles,sim)

## Plot position results singularly
dHandler.plot()
print(dHandler.tArray)