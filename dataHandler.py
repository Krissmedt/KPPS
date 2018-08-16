## Dependencies ##
import vtk_writer as vtk_writer
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Class ##
class dataHandler:
    def __init__(self, species, simulationManager, caseHandler, **kwargs):
        self.writeEvery = 1
        self.recordEvery = 1
        self.recordIndex = 0
        self.dataFoldername = "./"
        self.runOps = []
        self.postOps = []
        self.plotOps = []
        self.writer = vtk_writer.VTK_XML_Serial_Unstructured()
        self.figureNo = 0
        self.label = simulationManager.simID
        
        if 'write' in kwargs:
            self.writeSetup(species,
                            simulationManager,
                            caseHandler,
                            **kwargs['write'])
            
            self.runOps.append(self.writeData)
                    
        if 'record' in kwargs:
            self.recordSetup(species,simulationManager, **kwargs['record'])
            self.runOps.append(self.recordData)
            self.postOps.append(self.convertToNumpy)
                    
        if 'plot' in kwargs:
            self.plotSettings = kwargs['plot']
            self.plotOps.append(self.xyzPlot)
            
        if 'trajectory_plot' in kwargs:
            self.trajectorySettings = kwargs['trajectory_plot']
            self.plotOps.append(self.trajectoryPlot)
            
                    

    def run(self,species,simulationManager):
        for method in self.runOps:
            method(species,simulationManager)
            
    def post(self):
        for method in self.postOps:
            method()    

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
        ## NOTE: For small sampleRate compared to large number of time-steps,
        ## data held in memory can quickly exceed system capabilities!
        if 'sampleRate' in kwargs:
            self.recordEvery = kwargs['sampleRate']
          
        self.tArray = []
        
        self.xArray = []
        self.yArray = []
        self.zArray = []
        self.vxArray = []
        self.vyArray = []
        self.vzArray = []
        
        self.hArray = []
        self.cmArray = []
        
        
    def recordData(self,species,simulationManager):
        ## NOTE: For small sampleRate compared to large number of time-steps,
        ## data held in memory can quickly exceed system capabilities!
        if simulationManager.ts % self.recordEvery == 0:
            self.tArray.append(simulationManager.t)
            
            self.xArray.append(species.pos[:,0])
            self.yArray.append(species.pos[:,1])
            self.zArray.append(species.pos[:,2])
            
            self.vxArray.append(species.vel[:,0])
            self.vyArray.append(species.vel[:,1])
            self.vzArray.append(species.vel[:,2])
            
            self.hArray.append(species.energy)
            self.cmArray.append(np.copy(species.cm))


    def xyzPlot(self):
        if 'tPlot' in self.plotSettings:
            for char in self.plotSettings['tPlot']:
                self.figureNo += 1
                if char == 'x':
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.xArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$x$')
                    
                elif char == 'y':
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.yArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$y$')
                    
                elif char == 'z':
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.zArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$z$')
                    
                elif char == 'v':
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.vxArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$vx$')
                    
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.vyArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$vy$')
                    
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.vzArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$vz$')
                    
                elif char == 'E':
                    fig = plt.figure(self.figureNo)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(self.tArray,self.hArray)
                    ax.set_xscale('linear')
                    ax.set_xlabel('$t$')
                    ax.set_yscale('linear')
                    ax.set_ylabel('$h$')
                    
        if 'sPlot' in self.plotSettings:
            self.figureNo += 1
            plt.figure(self.figureNo)
            plt.plot(self.xArray,self.yArray)
            
            
    def trajectoryPlot(self):
        self.figureNo += 1
        if 'particles' in self.trajectorySettings:
            particles = np.array(self.trajectorySettings['particles']) - 1
        else:
            particles = [0]
            
        if 'limits' in self.trajectorySettings:
            limits = np.array(self.trajectorySettings['limits'],dtype=np.float)
        else:
            limits = np.array([20,20,15],dtype=np.float)
        
        fig = plt.figure(self.figureNo)
        ax = fig.gca(projection='3d')
        for pii in range(0,len(particles)):
            traj = "p" + str(particles[pii])
            ax.plot3D(self.xArray[:,particles[pii]],
                      self.yArray[:,particles[pii]],
                      zs=self.zArray[:,particles[pii]])
            
        ax.set_xlim([-limits[0], limits[0]])
        ax.set_ylim([-limits[1], limits[1]])
        ax.set_zlim([-limits[2], limits[2]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
    
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
        self.vxArray = np.array(self.vxArray)
        self.vyArray = np.array(self.vyArray)
        self.vzArray = np.array(self.vzArray)
        self.hArray = np.array(self.hArray)
        self.cmArray =  np.array(self.cmArray)