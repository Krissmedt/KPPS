################### Kris' Plasma Particle Simulator (KPPS) ####################

## Dependencies
import numpy as np
import matplotlib.pyplot as plt
from species import species
from mesh import mesh
from simulationManager import simulationManager
from dataHandler import dataHandler
from caseHandler import caseHandler
from kpps_analysis import kpps_analysis

class kpps:
    simSettings = {}
    speciesSettings = {}
    fieldSettings = {}
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
            
        if 'fieldSettings' in kwargs:
            self.fieldSettings = kwargs['fieldSettings']
            
        if 'caseSettings' in kwargs:
            self.caseSettings = kwargs['caseSettings']
            
        if 'analysisSettings' in kwargs:
            self.analysisSettings = kwargs['analysisSettings']

        if 'dataSettings' in kwargs:
            self.dataSettings = kwargs['dataSettings']
            

    def run(self):
        ## Load required modules
        particles = species(**self.speciesSettings)
        fields = mesh(**self.fieldSettings)
        sim = simulationManager(**self.simSettings)
        
        case = caseHandler(species=particles,
                           mesh=fields,
                           **self.caseSettings)
        
        analyser = kpps_analysis(simulationManager=sim,
                                 **self.analysisSettings)
        
        dHandler = dataHandler(species=particles,
                               mesh=fields,
                               caseHandler=case,
                               simulationManager=sim,
                               **self.dataSettings)
        
        
        ## Main time loop
        analyser.preAnalyser(particles,fields,sim)
        dHandler.run(particles,fields,sim)

        for ts in range(1,sim.tSteps+1):
            sim.updateTime()
            analyser.fieldIntegrator(particles,fields,sim)
            analyser.particleIntegrator(particles,fields,sim) 
            analyser.runHooks(particles,fields,sim)
            dHandler.run(particles,fields,sim)

        
        ## Post-analysis and data plotting
        analyser.postAnalyser(particles,fields,sim)
        dHandler.post(particles,fields,sim)
        dHandler.plot()

        return dHandler
