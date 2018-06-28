#!/usr/bin/env python3
from math import floor

## Class
class simulationManager:
    ## Main Methods
    def __init__(self,**kwargs):
        if 't0' in kwargs:
            self.t0 = kwargs['t0']
        else:
            self.t0 = 0
        
        if 'tEnd' in kwargs and 'tSteps' in kwargs:
            self.tEnd = kwargs['tEnd']
            self.tSteps = kwargs['tSteps']
            self.dt = (self.tEnd-self.t0)/self.tSteps
            
        elif 'tEnd' in kwargs and 'dt' in kwargs:
            self.tEnd = kwargs['tEnd']
            self.dt = kwargs['dt']
            self.tSteps = floor((self.tEnd-self.t0)/self.dt)
            
        elif 'dt' in kwargs and 'tSteps' in kwargs:
            self.dt = kwargs['dt']
            self.tSteps = kwargs['tSteps']
            self.tEnd = self.t0 + self.dt * self.tSteps
            
        else:
            print("No valid combination of inputs end-time tEnd, time-steps "+
                  "tSteps and time-step-length dt specified, resorting to "+
                  "default simulation parameters: tEnd=1, tSteps=100, dt=0.01")
            
        self.inputPrint()
        
        self.ts = 0
        self.t = self.t0
        
        self.tArray = []
        self.tArray.append(self.t)
        
        self.percentTime = self.tEnd/100
        self.percentCounter = self.percentTime
        
    def inputPrint(self):
        print("Simulation will now run from t = " + str(self.t0)
                + " to t = " + str(self.tEnd) + " in " 
                + str(self.tSteps) + " time-steps. Time-step size is " 
                + str(self.dt) + ".")
        
    def updateTime(self):
        self.ts = self.ts + 1
        self.t = self.t + self.dt
        self.tArray.append(self.t)
        
        if self.t >= self.percentCounter:
            print("Simulation progress: " 
                  + str(int(self.t/self.percentTime)) + "%" 
                  + " - " + str(self.ts) + "/" + str(self.tSteps))
            self.percentCounter += self.percentTime
        