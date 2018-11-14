## Dependencies ##
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Class ##
class dataHandler:
    def __init__(self,**kwargs):
        ## Default values and initialisation
        self.writeEvery = 1
        self.recordEvery = 1
        self.record = False
        self.write = False
        self.component_plots = False
        self.trajectory_plots = False
        self.domain_limits = [[0,1],[0,1],[0,1]]
        
        self.recordIndex = 0
        self.vtkFoldername = "./"
        self.runOps = []
        self.postOps = []
        self.plotOps = []
        self.figureNo = 0
        self.plot_params = {}
        self.components = ' '
        
        ## Dummy values - Need to be set in params!
        self.pos = np.zeros((1,3),dtype=np.float)
        self.vel = np.zeros((1,3),dtype=np.float)
        self.species = None
        self.mesh = None
        self.simulationManager = None
        self.caseHandler = None
        self.samplePeriod_w = None
        self.samples_w = None
        self.samplePeriod_rec = None
        self.samples_rec = None
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
         # check for other intuitive parameter names
        name_dict = {}
        name_dict['samplePeriod_wrt'] = ['samplePeriod']
        name_dict['samplePeriod_rec'] = ['samplePeriod']
        name_dict['samples_wrt'] = ['samples']
        name_dict['samples_rec'] = ['samples']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    getattr(self,name)
                    setattr(self,key,self.params[name])
                except AttributeError:
                    pass
                
        # Transform potential lists into numpy arrays
        self.domain_limits = np.array(self.domain_limits,dtype=np.float)            
        
        
        try:
            self.label = self.simulationManager.simID
            self.simType = self.simulationManager.simType
        except AttributeError:
            self.label = 'none'
            self.simType = 'none'
        
        if self.domain_limits.shape != (3,2):
            try:
                lim = self.domain_limits
                self.domain_limits = np.zeros((3,2),dtype=np.float)
                self.domain_limits[:,0] = -lim
                self.domain_limits[:,1] = lim
            except IndexError:
                print("Domain limit input not recognised")
        else:
            try: 
                self.domain_limits[0,:] = self.caseHandler.limits
                self.domain_limits[1,:] = self.caseHandler.limits
                self.domain_limits[2,:] = self.caseHandler.limits
            except AttributeError:
                try:
                    self.domain_limits[0,:] = self.caseHandler.xlimits
                    self.domain_limits[1,:] = self.caseHandler.ylimits
                    self.domain_limits[2,:] = self.caseHandler.zlimits
                except AttributeError:
                    print("Domain limit input not recognised")

            
        if self.write == True:
            self.writeSetup(self.species,
                            self.mesh,
                            self.simulationManager,
                            self.caseHandler)
            
            self.runOps.append(self.writeData)
                    
        if self.record == True:
            self.recordSetup(self.species,self.mesh,self.simulationManager)
            self.runOps.append(self.record_particleData)
            
            if self.simType == 'pic':
                self.runOps.append(self.record_meshData)
                
            self.postOps.append(self.convertToNumpy)
            
            
        if self.component_plots == True:
            self.plotOps.append(self.xyzPlot)
            
        if self.trajectory_plots == True:
            self.plotOps.append(self.trajectoryPlot)
            
        self.postOps.append(self.rhs_tally)   
        plt.rcParams.update(self.plot_params)
        


            

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
        try:
            self.recordEvery = 0 + self.samplePeriod_rec
        except TypeError:
            try:
                self.recordEvery = math.floor(simulationManager.tSteps
                                             /self.samples_rec)
            except TypeError:
                pass
            
        self.tArray = []
        
        self.xArray = []
        self.yArray = []
        self.zArray = []
        self.vxArray = []
        self.vyArray = []
        self.vzArray = []
        
        self.mesh_q = []
        self.mesh_E = []
        self.mesh_CE = []
        self.mesh_B = []
        self.mesh_phi = []
        self.mesh_pos = fields.pos
        self.hArray = []
        self.cmArray = []
        
        
    def record_particleData(self,species,fields,simulationManager):
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
            
    def record_meshData(self,species,fields,simulationManager):
        ## NOTE: For small sampleInterval compared to large number of time-steps,
        ## data held in memory can quickly exceed system capabilities!
        if simulationManager.ts % self.recordEvery == 0:
            self.mesh_q.append(fields.q)
            self.mesh_phi.append(fields.phi)
            self.mesh_E.append(fields.E)
            self.mesh_B.append(fields.B)
            self.mesh_CE.append(fields.CE)

    def xyzPlot(self):
        limits = self.domain_limits
        for char in self.components:
            self.figureNo += 1
            if char == 'x':
                fig = plt.figure(self.figureNo)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(self.tArray,self.xArray)
                ax.set_xscale('linear')
                ax.set_xlabel('$t$')
                ax.set_yscale('linear')
                ax.set_ylabel('$x$')
                ax.set_ylim([limits[0,0], limits[0,1]])
                
            elif char == 'y':
                fig = plt.figure(self.figureNo)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(self.tArray,self.yArray)
                ax.set_xscale('linear')
                ax.set_xlabel('$t$')
                ax.set_yscale('linear')
                ax.set_ylabel('$y$')
                ax.set_ylim([limits[1,0], limits[1,1]])
                
            elif char == 'z':
                fig = plt.figure(self.figureNo)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(self.tArray,self.zArray)
                ax.set_xscale('linear')
                ax.set_xlabel('$t$')
                ax.set_yscale('linear')
                ax.set_ylabel('$z$')
                ax.set_ylim([limits[2,0], limits[2,1]])
                
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
            
            
    def trajectoryPlot(self):
        self.figureNo += 1
        
        try:
            particles = np.array(self.trajectories,dtype=np.int) - 1
        except (TypeError,ValueError):
            if self.trajectories == 'all':
                particles = np.linspace(0,len(self.xArray[0])-1,
                                        len(self.xArray[0])-1,
                                        dtype=np.int)

        limits = self.domain_limits
        
        fig = plt.figure(self.figureNo)
        ax = fig.gca(projection='3d')
        for pii in range(0,len(particles)):
            ax.plot3D(self.xArray[:,particles[pii]],
                      self.yArray[:,particles[pii]],
                      zs=self.zArray[:,particles[pii]])
            
        ax.set_xlim([limits[0,0], limits[0,1]])
        ax.set_ylim([limits[1,0], limits[1,1]])
        ax.set_zlim([limits[2,0], limits[2,1]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    
    def writeSetup(self,species,fields,simulationManager,caseHandler,**kwargs):
        import vtk_writer as vtk_writer
        self.writer = vtk_writer.VTK_XML_Serial_Unstructured()
        if self.foldernaming == 'simple':
            delimiter = "_"
            
            entries = ['kpps',
                       str(caseHandler.ndim) + 'D',
                       str(species.nq) + 'p',
                       str(simulationManager.tSteps) + 'k']
            
            self.vtkFoldername += delimiter.join(entries)
            
        try:
            self.writeEvery = 0 + self.samplePeriod_wrt
        except TypeError:
            try:
                self.writeEvery = math.floor(simulationManager.tSteps
                                             /self.samples_wrt)
            except TypeError:
                pass
        
        self.mkDataDir()
            
        
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
        
        
    def orderLines(self,order,xRange,yRange):
        if order < 0:
            a = yRange[1]/xRange[0]**order
        else:
            a = yRange[0]/xRange[0]**order    
        
        oLine = [a*xRange[0]**order,a*xRange[1]**order]
            
        return oLine
        
    def loadData(self,filename,variables,**kwargs):
        """ Read data file and set the desired array attribute accordingly.
            For preset data arrays, use variable = 'x','y','z','vx','vy',
            'vz','cm' or 'h'. Input own variable name for custom variable."""
        
        data = np.loadtxt(filename)
        
        i = 0
        for var in variables:
            array_name = var + 'Array'
            
            if 'columns' in kwargs:
                j = kwargs['columns'][i]
            else:
                j = i
                
            setattr(self,array_name,data[:,j])
            i += 1