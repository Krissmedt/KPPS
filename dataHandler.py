## Dependencies ##
import vtk_writer as vtk_writer
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Class ##
class dataHandler:
    def __init__(self, **kwargs):
        self.writeEvery = 1
        self.recordEvery = 1
        self.recordIndex = 0
        self.dataFoldername = "./"
        self.runOps = []
        self.postOps = []
        self.plotOps = []
        self.writer = vtk_writer.VTK_XML_Serial_Unstructured()
        self.figureNo = 0
        
        if 'species_obj' in kwargs:
            species = kwargs['species_obj']
            
        if 'mesh_obj' in kwargs:
            fields = kwargs['mesh_obj']
            
        if 'caseHandler_obj' in kwargs:
            caseHandler = kwargs['caseHandler_obj']
            
        if 'simManager_obj' in kwargs:
            simulationManager = kwargs['simManager_obj']
            self.label = simulationManager.simID

        
        if 'write' in kwargs:
            self.writeSetup(species,
                            simulationManager,
                            caseHandler,
                            **kwargs['write'])
            
            self.runOps.append(self.writeData)
                    
        if 'record' in kwargs:
            self.recordSetup(species,fields,simulationManager, **kwargs['record'])
            self.runOps.append(self.recordData)
            self.postOps.append(self.convertToNumpy)
            self.postOps.append(self.rhs_tally)
            
        if 'plot' in kwargs:
            self.plotSettings = kwargs['plot']
            self.plotOps.append(self.xyzPlot)
            
        if 'trajectory_plot' in kwargs:
            self.trajectorySettings = kwargs['trajectory_plot']
            self.plotOps.append(self.trajectoryPlot)
            

    def run(self,species,fields,simulationManager):
        for method in self.runOps:
            method(species,fields,simulationManager)
            
    def post(self,species,fields,simulationManager):
        for method in self.postOps:
            method(species,fields,simulationManager)    

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
        
        
    def recordSetup(self,species,fields,simulationManager,**kwargs):
        ## NOTE: For small sampleRate compared to large number of time-steps,
        ## data held in memory can quickly exceed system capabilities!
        if 'sampleInterval' in kwargs:
            self.recordEvery = kwargs['sampleInterval']
        elif 'sampleNo' in kwargs:
            self.recordEvery = math.floor(simulationManager.tSteps/kwargs['sampleNo'])
            
        self.tArray = []
        
        self.xArray = []
        self.yArray = []
        self.zArray = []
        self.vxArray = []
        self.vyArray = []
        self.vzArray = []
        
        self.hArray = []
        self.cmArray = []
        
        
    def recordData(self,species,fields,simulationManager):
        ## NOTE: For small sampleInterval compared to large number of time-steps,
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

    
    def writeSetup(self,species,fields,simulationManager,caseHandler,**kwargs):
        if 'foldername' in kwargs:
            if kwargs['foldername'] == "simple":
                delimiter = "_"
                
                entries = ['kpps',
                           str(caseHandler.ndim) + 'D',
                           str(species.nq) + 'p',
                           str(simulationManager.tSteps) + 'k']
                
                self.dataFoldername += delimiter.join(entries)
            
            self.mkDataDir()
            
        if 'sampleInterval' in kwargs:
            self.writeEvery = kwargs['sampleInterval']
        elif 'sampleNo' in kwargs:
            self.writeEvery = math.floor(simulationManager.tSteps/kwargs['sampleNo'])
            

        
        
    def writeData(self,species,fields,simulationManager):
        ts = simulationManager.ts
        pos = species.pos
        
        if ts % self.writeEvery == 0:
            filename = self.dataFoldername + "/" + str(ts) + ".vtu"
            self.writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
            
            filename = self.dataFoldername + "/" + self.dataFoldername[2:]
            self.writer.writePVD(filename + ".pvd")
    
    def rhs_tally(self,species,fields,simulationManager):
        self.rhs_eval = simulationManager.rhs_dt * simulationManager.tSteps
            
    def convertToNumpy(self,*args):
        self.tArray = np.array(self.tArray)
        self.xArray = np.array(self.xArray)
        self.yArray = np.array(self.yArray)
        self.zArray = np.array(self.zArray)
        self.vxArray = np.array(self.vxArray)
        self.vyArray = np.array(self.vyArray)
        self.vzArray = np.array(self.vzArray)
        self.hArray = np.array(self.hArray)
        self.cmArray =  np.array(self.cmArray)
        
    def loadData(self,filename,variable):
        """ Read data file and set the desired array attribute accordingly.
            For preset data arrays, use variable = 'x','y','z','vx','vy',
            'vz','cm' or 'h'. Input own variable name for custom variable."""
            
        array_name = variable + 'Array'
        setattr(self,array_name,np.loadtxt(filename))