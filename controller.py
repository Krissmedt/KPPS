#!/usr/bin/env python3
from math import floor
import numpy as np
import time

## Class
class controller:
    ## Main Methods
    def __init__(self,**kwargs):
        ## Default values
        self.simID = 'no_name'
        self.ndim = 3
        self.t0 = 0
        self.tEnd = 1
        self.dt = 1
        self.tSteps = 1
        
        self.setupTime = 0
        self.runTime = 0

        self.percentBar = False
        self.restarted = False
        
        self.xlimits = np.array([-1,1],dtype=np.float)
        self.ylimits = np.array([-1,1],dtype=np.float)
        self.zlimits = np.array([-1,1],dtype=np.float)
        
        self.speciesSettings = {}
        self.meshSettings = {}
        self.caseSettings = {}
        self.analysisSettings = {}
        self.dataSettings = {}

        
        ## Dummy values, must be set in parameters or elsewhere!
        self.rhs_dt = None
        self.rhs_eval = None
        
        self.simType = None
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
        # check for other intuitive parameter names
        name_dict = {}
        name_dict['ndim'] = ['dimensions','dimension']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    setattr(self,key,getattr(self,name))
                except AttributeError:
                    pass

        ## Try to determine correct end-time, time-step -size and -number combo 
        try:
            self.dt = (self.params['tEnd']-self.t0)/self.params['tSteps']
        except KeyError:
            pass
        
        try:
            self.tSteps = floor((self.params['tEnd']-self.t0)/self.params['dt'])
        except KeyError:
            pass
        
        try:
            self.tEnd = self.t0 + self.params['dt'] * self.params['tSteps']
        except KeyError:
            pass
        
        try:
            self.simType = self.analysisSettings['fieldAnalysis']
        except KeyError:
            pass

        
        self.hookFunctions = []
        try: 
            if self.percentBar == True:
                self.hookFunctions.append(self.displayProgress)
        except AttributeError:
            pass
        
        self.ts = 0
        self.t = self.t0
        
        self.tArray = []
        self.tArray.append(self.t)
        
        self.percentTime = self.tEnd/100
        self.percentCounter = self.percentTime
        
    
        
    def updateTime(self):
        self.ts += 1
        self.t += self.dt
        self.tArray.append(self.t)
        
        for method in self.hookFunctions:
            method()
        

    def inputPrint(self):
        print("Simulation '" + self.simID
                + "' will now run from t = " + str(self.t0)
                + " to t = " + str(self.tEnd) + " in " 
                + str(self.tSteps) + " time-steps. Time-step size is " 
                + str(self.dt) + ".")    
            
    def displayProgress(self):
        if self.t >= self.percentCounter:
            print("Simulation progress: " 
                  + str(int(self.t/self.percentTime)) + "%" 
                  + " - " + str(self.ts) + "/" + str(self.tSteps)
                  + " - at " + time.strftime("%d/%m/%y  %H:%M:%S",time.localtime()))
            self.percentCounter += self.percentTime