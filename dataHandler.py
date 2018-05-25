## Dependencies ##
import vtk_writer as vtk_writer
import os
import numpy as np
import matplotlib.pyplot as plt

## Class ##
class dataHandler:
    writeEvery = 1
    recordEvery = 1
    recordIndex = 0
    dataFoldername = "./"
    runOps = []
    plotOps = []
    writer = vtk_writer.VTK_XML_Serial_Unstructured()
    
    def __init__(self, species, simulationManager, caseHandler, **kwargs):
        if 'write' in kwargs:
            self.writeSetup(species,
                            simulationManager,
                            caseHandler,
                            **kwargs['write'])
            
            self.runOps.append(self.writeData)
                    
        if 'record' in kwargs:
            self.recordSetup(species,simulationManager, **kwargs['record'])
            self.runOps.append(self.recordData)
                    
        if 'plot' in kwargs:
            self.plotSettings = kwargs['plot']
            self.plotOps.append(self.xyzPlot)
                    

    def run(self,species,simulationManager):
        for method in self.runOps:
            method(species,simulationManager)

    def plot(self):
        for method in self.plotOps:
            method()
    
    def mkDataDir(self):
        k = 1
        error = True
        append = ""
        while error == True:
            try:
                os.mkdir(self.dataFoldername + append)
                error = False
            except FileExistsError:
                append = "(" + str(k) + ")"
                error = True
                k = k+1
                
        self.dataFoldername = self.dataFoldername + append
        
        
    def recordSetup(self,species,simulationManager,**kwargs):
        if 'sampleRate' in kwargs:
            self.recordEvery = kwargs['sampleRate']
          
        self.tArray = []
        
        self.xArray = []
        self.yArray = []
        self.zArray = []
        
    def recordData(self,species,simulationManager):
        if simulationManager.ts % self.recordEvery == 0:
            self.tArray.append(simulationManager.t)
            self.xArray.append(species.pos[:,0])
            self.yArray.append(species.pos[:,1])
            self.zArray.append(species.pos[:,2])
    
    def xyzPlot(self,**plotSettings):
        figureNo = 0
        if 'tPlot' in self.plotSettings:
            for char in self.plotSettings['tPlot']:
                figureNo += 1
                if char == 'x':
                    fig = plt.figure(figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.xArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$x$')
                    
                elif char == 'y':
                    fig = plt.figure(figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.yArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$y$')
                    
                elif char == 'z':
                    fig = plt.figure(figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.zArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$z$')
                    
        if 'sPlot' in self.plotSettings:
            figureNo += 1
            plt.figure(figureNo)
            plt.plot(self.xArray,self.yArray)
        
    
    def writeSetup(self,species,simulationManager,caseHandler,**kwargs):
        if 'foldername' in kwargs:
            if kwargs['foldername'] == "simple":
                delimiter = "_"
                
                entries = ['kpps',
                           str(caseHandler.ndim) + 'D',
                           str(species.nq) + 'p',
                           str(simulationManager.tSteps) + 'k']
                
                self.dataFoldername += delimiter.join(entries)
            
            self.mkDataDir()
            
        if 'sampleRate' in kwargs:
            self.writeEvery = kwargs['sampleRate']
            

        
        
    def writeData(self,species,simulationManager):
        ts = simulationManager.ts
        pos = species.pos
        
        if ts % self.writeEvery == 0:
            filename = self.dataFoldername + "/" + str(ts) + ".vtu"
            self.writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
            
            filename = self.dataFoldername + "/" + self.dataFoldername[2:]
            self.writer.writePVD(filename + ".pvd")
            
            
    def convertToNumpy(self):
        self.tArray = np.array(self.tArray)
        self.xArray = np.array(self.xArray)
        self.yArray = np.array(self.yArray)
        self.zArray = np.array(self.zArray)