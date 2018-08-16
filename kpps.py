################### Kris' Plasma Particle Simulator (KPPS) ####################

## Dependencies
import numpy as np
import matplotlib.pyplot as plt
from species import species
from simulationManager import simulationManager
from dataHandler import dataHandler
from caseHandler import caseHandler
from kpps_analysis import kpps_analysis
import time

class kpps:
    simSettings = {}
    speciesSettings = {}
    caseSettings = {}
    
    eFieldSettings = {}
    bFieldSettings = {}
    analysisSettings = {}
    
    dataSettings = {}
    
    def __init__(self,**kwargs):
        if 'simSettings' in kwargs:
            self.simSettings = kwargs['simSettings']
            
        if 'speciesSettings' in kwargs:
            self.speciesSettings = kwargs['speciesSettings']
            
        if 'caseSettings' in kwargs:
            self.caseSettings = kwargs['caseSettings']
            
        if 'analysisSettings' in kwargs:
            self.analysisSettings = kwargs['analysisSettings']
            
        if 'dataSettings' in kwargs:
            self.dataSettings = kwargs['dataSettings']


    def run(self):
        ## Load required modules
        particles = species(**self.speciesSettings)
        sim = simulationManager(**self.simSettings)
        case = caseHandler(particles,**self.caseSettings)
        analyser = kpps_analysis(**self.analysisSettings)
        dHandler = dataHandler(particles,sim,case,**self.dataSettings)
        
        
        ## Main time loop
        analyser.preAnalyser(particles,sim)
        dHandler.run(particles,sim)
        

        for ts in range(1,sim.tSteps+1):
            sim.updateTime()
            analyser.fieldIntegrator(particles)
            analyser.particleIntegrator(particles,sim) 
            analyser.runHooks(particles,sim)
            dHandler.run(particles,sim)

        
        ## Post-analysis and data plotting
        analyser.postAnalyser(particles,sim)
        dHandler.post()
        dHandler.plot()

        return dHandler
