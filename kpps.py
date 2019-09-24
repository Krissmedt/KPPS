################### Kris' Plasma Particle Simulator (KPPS) ####################

## Dependencies
import matplotlib.pyplot as plt
import copy as cp
import time
from species import species as species_class
from mesh import mesh
from particleLoader import particleLoader as pLoader_class
from meshLoader import meshLoader
from controller import controller
from dataHandler2 import dataHandler2 as dataHandler
from kpps_analysis import kpps_analysis

class kpps:
    def __init__(self,**kwargs):          
        self.simSettings = {}
        self.speciesSettings = []
        self.pLoaderSettings = []
        self.meshSettings = {}
        self.mLoaderSettings = {}
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
            
        if 'pLoaderSettings' in kwargs:
            self.pLoaderSettings = kwargs['pLoaderSettings']
        self.simSettings['pLoaderSettings'] = self.pLoaderSettings
        
        if 'mLoaderSettings' in kwargs:
            self.mLoaderSettings = kwargs['mLoaderSettings']
        self.simSettings['mLoaderSettings'] = self.mLoaderSettings
            
        if 'analysisSettings' in kwargs:
            self.analysisSettings = kwargs['analysisSettings']
        self.simSettings['analysisSettings'] = self.analysisSettings
            
        if 'dataSettings' in kwargs:
            self.dataSettings = kwargs['dataSettings']
        self.simSettings['dataSettings'] = self.dataSettings
            
            
    def run(self):
        tStart = time.time()
        
        ## Load required modules
        sim = controller(**self.simSettings)
        sim.inputPrint()
        
        print("Setting up...")
        
        species_list = []
        for setting in self.speciesSettings:
            species = species_class(**setting)
            species_list.append(species)
            
        pLoader_list = []
        for setting in self.pLoaderSettings:
            pLoader = pLoader_class(**setting)
            pLoader_list.append(pLoader)
        
        fields = mesh(**self.meshSettings)
        mLoader = meshLoader(**self.mLoaderSettings)
        

        analyser = kpps_analysis(simulationManager=sim,
                                 **self.analysisSettings)
        
        dHandler = dataHandler(controller_obj=sim,
                               **self.dataSettings)

        ## Main time loop
        for loader in pLoader_list:
            loader.run(species_list,sim)
        
        mLoader.run(fields,sim)

        analyser.run_preAnalyser(species_list,fields,controller=sim)

        dHandler.run_setup()
        dHandler.run(species_list,fields,sim)
        
        tRun = time.time()
        for ts in range(1,sim.tSteps+1):
            sim.updateTime()
            analyser.run_Integrator(species_list,fields,sim)
#            analyser.run_fieldIntegrator(species_list,fields,sim)
#            analyser.run_particleIntegrator(species_list,fields,sim) 
            analyser.runHooks(species_list,fields,controller=sim)
            dHandler.run(species_list,fields,sim)


        ## Post-analysis and data plotting
        analyser.run_postAnalyser(species_list,fields,sim)
        
        tStop = time.time()   
        sim.setupTime = tRun-tStart 
        sim.runTime = tStop-tRun
        
        dHandler.post(species_list,fields,sim)
        dHandler.plot()

        return dHandler
