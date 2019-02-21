#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np
import math as math

## Class
class caseHandler:
    def __init__(self,**kwargs):
        
        ## Default values
        self.ndim = 3
        self.custom_case = None  #Assign custom case class to this
        
        ## Default species values
        self.particle_init = 'none'
        self.dx = 1
        self.dv = 10

        
        ## Default mesh values
        self.mesh_init = 'none'
        
        self.xlimits = np.array([0,1],dtype=np.float)
        self.ylimits = np.array([0,1],dtype=np.float)
        self.zlimits = np.array([0,1],dtype=np.float)
        
        self.mesh_res = np.array([1,1,1],dtype=np.int)
        self.store_node_pos = False
        
        self.BC_scaling = 1
        self.BC_box = self.node_set
        self.BC_function = self.BC_node_template
        
        
        ## Dummy values - Need to be set in params for class to work!
        self.pos = np.zeros((1,3),dtype=np.float)
        self.vel = np.zeros((1,3),dtype=np.float)
        self.sim = None
        self.species = None
        self.mesh = None
        
        self.mesh_dh = None
        

        ## Iterate through keyword arguments and store all in object
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
         # check for other intuitive parameter names
        name_dict = {}
        name_dict['pos'] = ['positions']
        name_dict['vel'] = ['velocities']
        name_dict['mesh_dh'] = ['spacing']
        name_dict['mesh_res'] = ['resolution','res']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    getattr(self,name)
                    setattr(self,key,self.params[name])
                except AttributeError:
                    pass

        # Transform potential list inputs into numpy
        self.pos = np.array(self.pos)
        self.vel = np.array(self.vel)
        
        ## Case dimensionality setup
        try:
            self.ndim = self.sim.ndim
        except AttributeError:
            print('No controller referenced, defaulting to 3D')
            self.ndim = 3
        
        ## Main functionality - setup mesh and species for specific case
        ## Mesh setup
        # Take single digit inputs and assign to each axis
        try:
            self.xlimits = self.limits
            self.ylimits = self.limits
            self.zlimits = self.limits
        except AttributeError:
            pass
        
        try: 
            dh = np.zeros(3,dtype=np.float)
            dh[:] = self.mesh_dh
            self.mesh_dh = dh
        except ValueError:
            res = np.zeros(3,dtype=np.int)
            res[:] = self.mesh_res
            self.mesh_res = res

        self.set_mesh_inputs()
        
        if self.mesh_init == 'none':
            pass
        elif self.mesh_init == 'box':
            self.mesh.domain_type = self.mesh_init
            self.box(self.mesh)     
        elif self.mesh_init == 'custom' and self.particle_init != 'custom':
            self.custom_case(self.mesh)
        elif self.mesh_init == 'custom' and self.particle_init == 'custom':
            self.custom_case(self.species,self.mesh)

        ## Species setup              
        if self.particle_init == 'none':
            pass
        elif self.particle_init == 'direct':
            self.direct(self.species)
        elif self.particle_init == 'clouds':
            self.clouds(self.species)
        elif self.particle_init == 'random':
            self.randDis(self.species)
        elif self.particle_init == 'even':
            self.evenPos(self.species)
        elif self.particle_init == 'custom' and self.mesh_init != 'custom':
            self.custom_case(self.species)
            
        self.set_particle_inputs()

        
            
            
    ## Species methods
    def direct(self,species,**kwargs):
        nPos = self.pos.shape[0]
        if nPos <= species.nq:
            species.pos[:nPos,:] = self.pos
        elif nPos > species.nq:
            print("More positions than particles specified, ignoring excess entries.")
            species.pos = self.pos[:species.nq,:]
            
        nVel = self.vel.shape[0]
        if nVel <= species.nq:
            species.vel[:nVel,:] = self.vel
        elif nVel > species.nq:
            print("More velocities than particles specified, ignoring excess entries.")
            species.vel = self.vel[:species.nq,:]
                
    def clouds(self,species,**kwargs):
        ppc = math.floor(species.nq/self.pos.shape[0])
        for xi in range(0,len(self.pos)):
            species.pos[xi*ppc:(xi+1)*ppc,:] = self.pos[xi] + self.random(ppc,self.dx)
            species.vel[xi*ppc:(xi+1)*ppc,:] = self.vel[xi] + self.random(ppc,self.dv)
     
    def evenPos(self,species):
        return species
    
    def randDis(self,species):
        species.pos = self.pos[0] + self.random(species.nq,self.dx)
        species.vel = self.vel[0] + self.random(species.nq,self.dv)
        
    def set_particle_inputs(self):
        if self.ndim == 2:
            try:
                self.species.pos[0] = self.xlimits[0] + (self.xlimits[1]-self.xlimits[0])/2.
                self.species.vel[0] = 0
   
            except AttributeError:
                pass
 
        elif self.ndim == 1:
            try:
                self.species.pos[:,0] = self.xlimits[0] + (self.xlimits[1]-self.xlimits[0])/2.
                self.species.pos[:,1] = self.ylimits[0] + (self.ylimits[1]-self.ylimits[0])/2.
                self.species.vel[:,0] = 0
                self.species.vel[:,1] = 0
   
            except AttributeError:
                pass    
           
       
    ## Mesh methods
    def box(self,mesh):
        mesh.xlimits = self.xlimits 
        mesh.ylimits = self.ylimits
        mesh.zlimits = self.zlimits
        
        mesh.dh = self.mesh_dh
        mesh.dx = self.mesh_dh[0]
        mesh.dy = self.mesh_dh[1]
        mesh.dz = self.mesh_dh[2]
        mesh.dv = self.cell_volume

        mesh.res = np.array(self.mesh_res)
        mesh.xres = self.mesh_res[0]
        mesh.yres = self.mesh_res[1]
        mesh.zres = self.mesh_res[2]
        
        mesh.cells = np.prod(mesh.res)
        mesh.nn = np.prod(mesh.res+1)
        
        mesh.q = np.zeros((mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        mesh.rho = np.zeros((mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        mesh.E = np.zeros((3,mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        mesh.B = np.zeros((3,mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        
        mesh.phi = np.zeros((mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        mesh.BC_vector = np.zeros((mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
        mesh = self.BC_box(mesh)
        
        if self.store_node_pos == True:
            mesh.pos = np.zeros((3,mesh.xres+1,mesh.yres+1,mesh.zres+1),dtype=np.float)
            for xi in range(0,mesh.xres+1):
                mesh.pos[0,xi,:,:] = mesh.xlimits[0] + mesh.dx * xi
            for yi in range(0,mesh.yres+1):
                mesh.pos[1,:,yi,:] = mesh.ylimits[0] + mesh.dy * yi
            for zi in range(0,mesh.zres+1):
                mesh.pos[2,:,:,zi] = mesh.zlimits[0] + mesh.dz * zi
        return mesh


    def node_set(self,mesh):
        for yi in range(0,mesh.yres+1):
            for xi in range(0,mesh.xres+1):
                y = mesh.ylimits[0] + mesh.dy * yi
                x = mesh.xlimits[0] + mesh.dx * xi
                
                z0 = mesh.zlimits[0]
                zn = mesh.zlimits[1]
                mesh.phi[xi,yi,0] = self.BC_function(np.array([x,y,z0]))
                mesh.phi[xi,yi,-1] = self.BC_function(np.array([x,y,zn]))
                
        mesh.BC_vector[:,:,1] += mesh.phi[:,:,0]/mesh.dz**2
        mesh.BC_vector[:,:,-2] += mesh.phi[:,:,-1]/mesh.dz**2    
        
        if self.ndim >= 2:
            for zi in range(0,mesh.zres+1):
                for xi in range(0,mesh.xres+1):
                    x = mesh.ylimits[0] + mesh.dx * xi
                    z = mesh.zlimits[0] + mesh.dz * zi
                    
                    y0 = mesh.ylimits[0]
                    yn = mesh.ylimits[1]
                    mesh.phi[xi,0,zi] = self.BC_function(np.array([x,y0,z]))
                    mesh.phi[xi,-1,zi] = self.BC_function(np.array([x,yn,z]))
                    
            mesh.BC_vector[:,1,:] += mesh.phi[:,0,:]/mesh.dy**2
            mesh.BC_vector[:,-2,:] += mesh.phi[:,-1,:]/mesh.dy**2
        
        if self.ndim == 3:
            for zi in range(0,mesh.zres+1):
                for yi in range(0,mesh.yres+1):
                    y = mesh.ylimits[0] + mesh.dy * yi
                    z = mesh.zlimits[0] + mesh.dz * zi
                    
                    x0 = mesh.xlimits[0]
                    xn = mesh.xlimits[1]
                    mesh.phi[0,yi,zi] = self.BC_function(np.array([x0,y,z]))
                    mesh.phi[-1,yi,zi] = self.BC_function(np.array([xn,y,z]))
                    
            mesh.BC_vector[1,:,:] += mesh.phi[0,:,:]/mesh.dx**2
            mesh.BC_vector[-2,:,:] += mesh.phi[-1,:,:]/mesh.dx**2
                
        mesh.BC_vector = mesh.BC_vector * self.BC_scaling                
        mesh.BC_vector = self.meshtoVector(mesh.BC_vector[1:-1,1:-1,1:-1])
        
        return mesh

    
    def set_mesh_inputs(self):
    # Set dependent mesh parameters and enforce input dimensionality.
        try:
            assert self.xlimits[1] - self.xlimits[0] > 0
            assert self.ylimits[1] - self.ylimits[0] > 0
            assert self.zlimits[1] - self.zlimits[0] > 0
        except AssertionError:
            print("One of the input box-edge limits is not positive " +
                  "in length. Reverting to default 1x1x1 cube.")
            self.xlimits = np.array([0,1])
            self.ylimits = np.array([0,1])
            self.zlimits = np.array([0,1])
    
        try:
            xres = (self.xlimits[1] - self.xlimits[0])/self.mesh_dh[0]
            yres = (self.ylimits[1] - self.ylimits[0])/self.mesh_dh[1]
            zres = (self.zlimits[1] - self.zlimits[0])/self.mesh_dh[2]
            self.mesh_res = np.array([xres,yres,zres],dtype=np.int)
        except ValueError:
            dx = (self.xlimits[1] - self.xlimits[0])/self.mesh_res[0]
            dy = (self.ylimits[1] - self.ylimits[0])/self.mesh_res[1]
            dz = (self.zlimits[1] - self.zlimits[0])/self.mesh_res[2]
            self.mesh_dh = np.array([dx,dy,dz])   
            
        self.cell_volume = np.prod(self.mesh_dh[0:])
        
        if self.ndim == 2:
            try:
                assert self.mesh_res[0] == 2

            except AssertionError:
                self.mesh_res[0] = 2
                self.mesh_dh[0] = (self.xlimits[1] - self.xlimits[0])/self.mesh_res[0]
            
            self.cell_volume = np.prod(self.mesh_dh[1:])
            
        elif self.ndim == 1:
            try:
                assert self.mesh_res[0] == 2
                assert self.mesh_res[1] == 2
                
            except AssertionError:
                self.mesh_res[0] = 2
                self.mesh_res[1] = 2
                self.mesh_dh[0] = (self.xlimits[1] - self.xlimits[0])/self.mesh_res[0]
                self.mesh_dh[1] = (self.ylimits[1] - self.ylimits[0])/self.mesh_res[1]
                
            self.cell_volume = np.prod(self.mesh_dh[2:])
    
    def BC_node_template(self,pos):
        node_value = 0
        
        return node_value
        
    ## Additional methods
    def meshtoVector(self,mesh):
        shape = np.shape(mesh)
        x = np.zeros(shape[0]*shape[1]*shape[2],dtype=np.float)
        xi = 0
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    x[xi] = mesh[i,j,k]
                    xi += 1
        return x
    
    def random(self,rows,deviance):
        output = np.zeros((rows,self.ndim),dtype=np.float)
        for nd in range(0,self.ndim):
            for i in range(0,rows):
                output[i,nd] = np.random.uniform(-deviance,deviance)
        
        return output