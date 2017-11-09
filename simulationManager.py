#!/usr/bin/env python3
## Class
class simulationManager:
    tStart = 0.
    
    ts = 0
    t = tStart
    
    tEnd = 1.
    tSteps = 100
    dt = tEnd/tSteps
    
    ndim = 3
    
    
    ## Main Methods
    def __init__(self,**kwargs):
        if 'tEnd' in kwargs and 'tSteps' in kwargs:
            self.tEnd = kwargs['tEnd']
            self.tSteps = kwargs['tSteps']
            self.dt = self.tEnd/self.tSteps
            
        elif 'tEnd' in kwargs and 'dt' in kwargs:
            self.tEnd = kwargs['tEnd']
            self.dt = kwargs['dt']
            self.tSteps = self.tEnd/self.dt
            
        elif 'dt' in kwargs and 'tSteps' in kwargs:
            self.dt = kwargs['dt']
            self.tSteps = kwargs['tSteps']
            self.tEnd = self.dt * self.tSteps
            
        else:
            print("No valid combination of inputs end-time tEnd, time-steps "+
                  "tSteps and time-step-length dt specified, resorting to "+
                  "default simulation parameters: tEnd=1, tSteps=100, dt=0.01")
        
        self.tArray = []
        self.tArray.append(self.t)
        
        if 'dimensions' in kwargs:
            self.ndim = kwargs['dimensions']
            
        
    def updateTime(self):
        if self.t < self.tEnd:
            self.ts = self.ts + 1
            self.t = self.t + self.dt
            self.tArray.append(self.t)
        else:
            print("End time reached!")
        