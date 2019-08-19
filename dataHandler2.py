## Dependencies ##
import os
import io
import numpy as np
import math as math
import matplotlib.pyplot as plt
import copy as cp
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D

## Class ##
class dataHandler2:
    def __init__(self,**kwargs):
        ## Default values and initialisation
        self.params = {}
        
        self.samplePeriod = 1
        self.dataPath = 'relative'
        self.dataFoldername = "./"
        self.vtkFoldername = "./vtk"
        
        self.write = True
        self.write_p = True
        self.write_m = True
        self.write_vtk = False
        self.trajectory_plotting = False
        self.time_plotting = False
        self.time_plot_vars = [] 
        
        self.runOps = []
        self.postOps = []
        self.plotOps = []
        
        self.figureNo = 0
        
        self.tagged_particles = []

        self.vecPlot_list = []
        self.pTimePlot_list = []
        
        self.rhs_eval = 0
        
        self.plot_params = {}
        self.plot_params['legend.fontsize'] = 12
        self.plot_params['figure.figsize'] = (12,8)
        self.plot_params['axes.labelsize'] = 20
        self.plot_params['axes.titlesize'] = 20
        self.plot_params['xtick.labelsize'] = 16
        self.plot_params['ytick.labelsize'] = 16
        self.plot_params['lines.linewidth'] = 3
        self.plot_params['axes.titlepad'] = 10
     
        ## Dummy values - Need to be set in params or during run!
        self.samples = None
        self.controller_obj = None
        self.plot_limits = None
        
        
        ## Iterate through keyword arguments and store all in object (self)
        self.set_params(kwargs)
            
         # check for other intuitive parameter names
        self.input_translator(self.params)
        
        ## Determine dependent inputs
        self.set_plot_lims(self.plot_limits)


        if self.time_plotting == True:
            self.plotOps.append(self.particle_time_plot)
        
        if self.trajectory_plotting == True:
            self.plotOps.append(self.trajectory_plot)
            
        plt.rcParams.update(self.plot_params)
                    
###################### Simulation Run Functionality ###########################
                    
    def run(self,species_list,fields,simulationManager):
        if simulationManager.ts % self.samplePeriod == 0:
            for method in self.runOps:
                method(species_list,fields,simulationManager)
            

    def run_setup(self):

        self.controller_check()
        self.sampling_setup()
            
        self.dataFoldername = self.controller_obj.simID
        
        if self.write == True:
            if self.write_p == True:
                self.runOps.append(self.p_dumper)
            if self.write_m == True:
                self.runOps.append(self.m_dumper)
            self.dataFoldername = self.mkDataDir(self.dataFoldername)
        
        if self.write_vtk == True:
            import vtk_writer as vtk_writer
            self.vtk_writer_class = vtk_writer.VTK_XML_Serial_Unstructured()
            self.runOps.append(self.vtk_dumper)
            try:
                self.vtkFoldername = self.dataFoldername + "/vtk"
                self.mkDataDir(self.vtkFoldername)
            except FileNotFoundError:
                self.dataFoldername = self.mkDataDir(self.dataFoldername)
                self.mkDataDir(self.vtkFoldername)
        
        self.controller_obj.simID = self.dataFoldername  
            
        
    def p_dumper(self,species_list,fields,simulationManager):
        for species in species_list:
            p_filename = self.dataFoldername + "/p_" + species.name + "_t" + str(simulationManager.ts)
            p_file = io.open(p_filename,mode='wb')
            pk.dump(species,p_file)
            p_file.close()
            
            
    def m_dumper(self,species_list,fields,simulationManager):
        m_filename = self.dataFoldername + "/m_t" + str(simulationManager.ts)
        m_file = io.open(m_filename,mode='wb')
        pk.dump(fields,m_file)
        m_file.close()
        
    
    def vtk_dumper(self,species_list,fields,simulationManager):
        ts = simulationManager.ts
        
        for species in species_list:
            pos = species.pos
            
            filename = self.vtkFoldername + "/" + str(ts) + ".vtu"
            self.vtk_writer_class.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
            
            filename = self.vtkFoldername + "/PVD"
            self.vtk_writer_class.writePVD(filename + ".pvd")
        
        
######################### Post-Run Functionality ##############################
        
    def post(self,species_list,fields,controller):
        for method in self.postOps:
            method(species_list,fields,controller)
            
        sim_filename = self.dataFoldername + "/sim"
        sim_file = io.open(sim_filename,mode='wb')
        self.controller_obj = controller
        pk.dump(controller,sim_file)
            
    def load_sim(self,sim_name=None,overwrite=False):
        try:
            sim_file = io.open("./" + sim_name + "/sim",mode='rb')
        except TypeError:
            try:
                sim_file = io.open("./" + str(self.controller_obj.simID) + "/sim",mode='rb')
                sim_name = str(self.controller_obj.simID)
            except AttributeError:
                print('DataHandler: No valid simulation ID given and no controller object' +
                      ' with associated ID found, load failed.')

        sim = pk.load(sim_file)

        if overwrite == True:
            self.controller_obj = sim
            self.set_params(sim.dataSettings)
            self.set_plot_lims(self.plot_limits)
            self.input_translator(sim.dataSettings)
            self.sampling_setup()

                
        return sim, sim_name

    def load(self,dataType,variables,sim_name=None,overwrite=False,load_limit=None):
        # Will load specified variables for all dumps for data objects 'species'
        # or 'mesh' identified by dataType input, strings 'p' and 'm' for 
        # species and mesh objects respectively.
        # 'load_limit' can be used to load fewer than the max time-steps
        
        sim, sim_name = self.load_sim(sim_name,overwrite)
        try:
            skip = np.int(sim.tSteps/load_limit)
        except TypeError:
            skip = 1
        
        
        return_dict = {}
        for var in variables:
            return_dict[var] = []
        
        return_dict['t'] = []
        for ti in range(0,self.samples+1,skip):
            data_file = io.open("./" + str(sim_name) + "/" + dataType + "_t" 
                                + str(self.samplePeriod*ti),mode='rb')
            dataObject = pk.load(data_file)
            data_file.close()
            
            t = sim.t0 + self.samplePeriod*ti*sim.dt
            return_dict['t'].append(t)
            for var in variables:
                return_dict[var].append(getattr(dataObject,var))
        
        for key, value in return_dict.items():
            return_dict[key] = np.array(value)
            
        return return_dict, sim
        
    def load_p(self,variables,species=['none'],sim_name=None,overwrite=False,load_limit=None):
        species_list = []
        for spec in species:
            dtype = 'p_' + spec
            species_dict, sim = self.load(dtype,variables,sim_name=sim_name)
            species_list.append(species_dict)
            
        return species_list
    
    def load_m(self,variables,sim_name=None,overwrite=False,load_limit=None):
        return_dict, sim = self.load('m',variables,sim_name=sim_name)
        return return_dict
        
    def load_parameters(self,modules=None,sim_name=None):
        sim, sim_name = self.load_sim(sim_name)
        
        try:
            check = modules[0]
        except TypeError:
            modules = ['simSettings','speciesSettings','meshSettings',
                       'caseSettings','analysisSettings','dataSettings']

        for mod in modules:
            print()
            mod_dict = getattr(sim,mod)
            for key,value in mod_dict.items():
                print(key + " = " + str(value))
                
    

                
######################### Plotting Functionality ##############################
    def plot(self):
        for method in self.plotOps:
            method()
            
    def particle_time_plot(self,species=['none'],variables=None,particles=None,sim_name=None):
        try:
            check = variables[0]
        except TypeError:
            variables = self.time_plot_vars
            
        for spec in species:
            dtype = "p_" + spec
            if 'pos' not in variables:
                variables.append('pos')
                data,sim = self.load(dtype,variables,sim_name=sim_name)
                variables.remove('pos')
            else:
                data,sim = self.load(dtype,variables,sim_name=sim_name)

            particles = self.set_taggedList(data,particles)
            
            for var in variables:
                vlabel = var
                try:
                    assert data[var].shape[1] >= particles.shape[0]
                except (AssertionError,IndexError):
                    continue
                
                self.figureNo += 1
                fig = plt.figure(self.figureNo)
                ax = fig.add_subplot(1, 1, 1)
                for pii in range(0,particles.shape[0]):
                    try:
                        ax.plot(data['t'],data[var][:,particles[pii],0],label=vlabel+"_x") 
                        ax.plot(data['t'],data[var][:,particles[pii],1],label=vlabel+"_y") 
                        ax.plot(data['t'],data[var][:,particles[pii],2],label=vlabel+"_z") 
                    except IndexError:
                        ax.plot(data['t'],data[var][:,particles[pii]],label=vlabel) 
                    
                ax.set_xscale('linear')
                ax.set_xlabel('$t$')
                ax.set_yscale('linear')
                ax.set_ylabel(var)
                ax.legend()

                
    
    def trajectory_plot(self,species=['none'],particles=None,sim_name=None):
        for spec in species: 
            dtype = "p_" + spec
            posData,sim = self.load(dtype,['pos'],sim_name=0)
            xArray = posData['pos'][:,:,0]
            yArray = posData['pos'][:,:,1]
            zArray = posData['pos'][:,:,2]
            
            particles = self.set_taggedList(posData,particles)
                    
            limits = np.array(self.plot_limits)
            
            self.figureNo += 1
            fig = plt.figure(self.figureNo)
            ax = fig.gca(projection='3d')
            for pii in range(0,len(particles)):
                plabel = "p = " + str(particles[pii]+1)
                ax.plot3D(xArray[:,particles[pii]],
                          yArray[:,particles[pii]],
                          zs=zArray[:,particles[pii]],label=plabel)
                
            ax.set_xlim([limits[0,0], limits[0,1]])
            ax.set_ylim([limits[1,0], limits[1,1]])
            ax.set_zlim([limits[2,0], limits[2,1]])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            
            if particles.shape[0] <= 10:
                ax.legend()
        
        
    def vector_plot(self,variables=0):
        pass
    
    
######################### Setup Functionality #################################
    def set_params(self,input_dict):
        self.params = input_dict
        for key, value in self.params.items():
            setattr(self,key,value)
            
        
        
    def input_translator(self, param_dict):
        name_dict = {}
        name_dict['samplePeriod'] = ['sampleInterval']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    getattr(self,name)
                    setattr(self,key,param_dict[name])
                except AttributeError:
                    pass
    
    def set_plot_lims(self, limit_input):
        limit_input = np.array(limit_input)
        try:
            if limit_input.shape == (3,2):
                pass
            else:
                lims = limit_input
                self.plot_limits = np.zeros((3,2),dtype=np.float)
                self.plot_limits[:,0] = -lims
                self.plot_limits[:,1] = lims
    
        except (AttributeError, TypeError, IndexError):
                try: 
                    self.plot_limits[0,:] = self.controller_obj.caseSettings['xlimits']
                    self.plot_limits[1,:] = self.controller_obj.caseSettings['ylimits']
                    self.plot_limits[2,:] = self.controller_obj.caseSettings['zlimits']
                except (AttributeError, KeyError):
                    print("DataHandler: Plot limit input not recognised, defaulting to 1x1x1")
                    self.plot_limits = np.array([[0,1],[0,1],[0,1]])


    def mkDataDir(self,foldername):
        k = 1
        error = True
        append = ""
        while error == True:
            try:
                os.mkdir(foldername + append)
                error = False
            except FileExistsError:
                append = "(" + str(k) + ")"
                error = True
                k = k+1
                
        foldername = foldername + append
        
        return foldername
        
    
    def sampling_setup(self):
        try:
            self.samplePeriod = math.floor(self.controller_obj.tSteps
                                           /self.samples)
        except TypeError:
            self.samples = math.floor(self.controller_obj.tSteps
                                           /self.samplePeriod)
            
        self.samplePeriod = int(self.samplePeriod)
        self.int = int(self.samples)
            
    
######################### Other Functionality #################################    
    def orderLines(self,order,xRange,yRange):
        if order < 0:
            a = yRange[1]/xRange[0]**order
        else:
            a = yRange[0]/xRange[0]**order    
        
        oLine = [a*xRange[0]**order,a*xRange[1]**order]
        return oLine
    
                
    def controller_check(self):
        try:
            self.dt = self.controller_obj.dt
        
        except AttributeError:
            print("DataHandler:  No valid controller object detected in dataHandler module" 
                  + ", cancelling run.")
            raise SystemExit(0)
            
            
    def set_taggedList(self,p_data,tagged_list):
        nq = p_data['pos'].shape[1]
        try:
            tagged_list = np.array(tagged_list,dtype=np.int) - 1
        except (TypeError,ValueError):
            if self.tagged_particles == 'all':
                tagged_list = np.linspace(0,nq-1,nq,dtype=np.int)
            else:
                tagged_list = np.array(self.tagged_particles,dtype=np.int) - 1
                
        return tagged_list