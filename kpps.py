################### Kris' Plasma Particle Simulator (KPPS) ####################

## Dependencies
import matplotlib.pyplot as plt
import copy as cp
import time
import io
import pickle as pk
from species import species as species_class
from mesh import mesh
from particleLoader import particleLoader as pLoader_class
from meshLoader import meshLoader
from controller import controller
from dataHandler2 import dataHandler2 as dataHandler
from kpps_analysis import kpps_analysis

class kpps:
    def __init__(self,**kwargs):
        print("Welcome to KPPS")
    
    def restart(self,folder,sim_name,tstep,**kwargs):
        self.tStart = time.time()
        
        ## Load required modules
        sim_file = io.open(folder + sim_name + "/sim",mode='rb')
        sim = pk.load(sim_file)
        sim_file.close()
        
        print("Restarting ' "+sim_name+" ' at time "+str(tstep*sim.dt)+"s...")
        
        sim.ts = tstep
        sim.t = tstep*sim.dt
        sim.restarted = True
        
        p = species_class()
        fields = mesh(**sim.meshSettings)
        mLoader = meshLoader(**sim.mLoaderSettings)
        mLoader.run(fields,sim)
        
        species_list = [p]

        analyser = kpps_analysis(simulationManager=sim,
                                 **sim.analysisSettings)
        analyser.run_preAnalyser(species_list,fields,controller=sim)
        
        species_list = []
        speciesSettings = sim.speciesSettings
        for setting in speciesSettings:
            spec_file = io.open(folder + sim_name + "/p_" + setting['name'] + "_t" + str(tstep),mode='rb')
            species = pk.load(spec_file)
            species_list.append(species)
            spec_file.close()
        
        mesh_file = io.open(folder + sim_name + "/m" + "_t" + str(tstep),mode='rb')
        fields = pk.load(mesh_file)
        mesh_file.close()
        
        
        dHandler = dataHandler(controller_obj=sim,
                               **sim.dataSettings)
        dHandler.run_setup(sim)
        ########################## RUN TIME! ##################################
        dHandler = self.run(species_list,fields,sim,analyser,dHandler)
        
        return dHandler
    
            
    def start(self,**kwargs):
        self.tStart = time.time()
        
        ## Read input settings
        self.simSettings = {}
        self.speciesSettings = []
        self.pLoaderSettings = []
        self.meshSettings = {}
        self.mLoaderSettings = {}
        self.analysisSettings = {}
        self.dataSettings = {}
        
        if 'simSettings' in kwargs:
            simSettings = kwargs['simSettings']
        simSettings['simSettings'] = cp.copy(simSettings)
            
        if 'speciesSettings' in kwargs:
            speciesSettings = kwargs['speciesSettings']
        simSettings['speciesSettings'] = speciesSettings
            
        if 'meshSettings' in kwargs:
            meshSettings = kwargs['meshSettings']
        simSettings['meshSettings'] = meshSettings
            
        if 'pLoaderSettings' in kwargs:
            pLoaderSettings = kwargs['pLoaderSettings']
        simSettings['pLoaderSettings'] = pLoaderSettings
        
        if 'mLoaderSettings' in kwargs:
            mLoaderSettings = kwargs['mLoaderSettings']
        simSettings['mLoaderSettings'] = mLoaderSettings
            
        if 'analysisSettings' in kwargs:
            analysisSettings = kwargs['analysisSettings']
        simSettings['analysisSettings'] = analysisSettings
            
        if 'dataSettings' in kwargs:
            dataSettings = kwargs['dataSettings']
        simSettings['dataSettings'] = dataSettings
        
        ## Load required modules
        sim = controller(**simSettings)
        sim.inputPrint()
        
        print("Setting up...")
        
        species_list = []
        for setting in speciesSettings:
            species = species_class(**setting)
            species_list.append(species)
            
        pLoader_list = []
        for setting in pLoaderSettings:
            pLoader = pLoader_class(**setting)
            pLoader_list.append(pLoader)
        
        fields = mesh(**meshSettings)
        mLoader = meshLoader(**mLoaderSettings)
        

        analyser = kpps_analysis(simulationManager=sim,
                                 **analysisSettings)
        
        dHandler = dataHandler(controller_obj=sim,
                               **dataSettings)

        for loader in pLoader_list:
            loader.run(species_list,sim)
        
        mLoader.run(fields,sim)

        analyser.run_preAnalyser(species_list,fields,controller=sim)

        dHandler.run_setup(sim)
        dHandler.run(species_list,fields,sim)
        
        ########################## RUN TIME! ##################################
        dHandler = self.run(species_list,fields,sim,analyser,dHandler)
        
        return dHandler
    
        
    def run(self,species_list,fields,sim,analyser,dHandler):
        ## Main time loop
        tRun = time.time()
        for ts in range(sim.ts+1,sim.tSteps+1):
            sim.updateTime()
            analyser.run_particleIntegrator(species_list,fields,sim) 
            analyser.runHooks(species_list,fields,controller=sim)
            dHandler.run(species_list,fields,sim)


        ## Post-analysis and data plotting
        analyser.run_postAnalyser(species_list,fields,sim)
        
        tStop = time.time()   
        sim.setupTime = tRun-self.tStart 
        sim.runTime = tStop-tRun
        
        dHandler.post(species_list,fields,sim)
        dHandler.plot()

        return dHandler
