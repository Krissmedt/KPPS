################### Kris' Plasma Particle Simulator (KPPS) ####################

## Dependencies
import copy as cp
from species import species as species_class
from mesh import mesh
from simulationManager import simulationManager
from dataHandler2 import dataHandler2 as dataHandler
from caseHandler import caseHandler
from kpps_analysis import kpps_analysis

class kpps:
    def __init__(self,**kwargs):          
        self.simSettings = {}
        self.speciesSettings = {}
        self.meshSettings = {}
        self.caseSettings = {}
        self.analysisSettings = {}
        self.dataSettings = {}
        
        if 'simSettings' in kwargs:
            self.simSettings = kwargs['simSettings']
        self.simSettings['simSettings'] = cp.copy(self.simSettings)
            
        if 'speciesSettings' in kwargs:
            self.speciesSettings = kwargs['speciesSettings']
        self.simSettings['speciesSettings'] = self.speciesSettings
            
        if 'meshSettings' in kwargs:
            self.meshSettings = kwargs['meshSettings']
        self.simSettings['meshSettings'] = self.meshSettings
            
        if 'caseSettings' in kwargs:
            self.caseSettings = kwargs['caseSettings']
        self.simSettings['caseSettings'] = self.caseSettings
            
        if 'analysisSettings' in kwargs:
            self.analysisSettings = kwargs['analysisSettings']
        self.simSettings['analysisSettings'] = self.analysisSettings
            
        if 'dataSettings' in kwargs:
            self.dataSettings = kwargs['dataSettings']
        self.simSettings['dataSettings'] = self.dataSettings
            
            
    def run(self):
        ## Load required modules
        particles = species_class(**self.speciesSettings)   
        fields = mesh(**self.meshSettings)
        sim = simulationManager(**self.simSettings)
        
        case = caseHandler(species=particles,
                           mesh=fields,
                           sim=sim,
                           **self.caseSettings)

        analyser = kpps_analysis(simulationManager=sim,
                                 **self.analysisSettings)
        
        dHandler = dataHandler(controller_obj=sim,
                               **self.dataSettings)
        

        ## Main time loop
        analyser.run_preAnalyser(particles,fields,sim)
        dHandler.run_setup()
        dHandler.run(particles,fields,sim)
        sim.inputPrint()

        for ts in range(1,sim.tSteps+1):
            sim.updateTime()
            analyser.run_fieldIntegrator(particles,fields,sim)
            analyser.run_particleIntegrator(particles,fields,sim) 
            analyser.runHooks(particles,fields,sim)
            dHandler.run(particles,fields,sim)

        
        ## Post-analysis and data plotting
        analyser.run_postAnalyser(particles,fields,sim)
        dHandler.post(particles,fields,sim)
        dHandler.plot()

        return dHandler
