#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np
import math as math

## Class
class meshLoader:
    def __init__(self,**kwargs):
        
        ## Default mesh values
        self.load_type = 'custom'
        self.custom = self.custom_ph
        
        self.grid_input = None
        self.mesh_res = np.array([2,2,2],dtype=np.int)
        self.mesh_dh = None
        self.store_node_pos = False
        
        self.BC_scaling = 1
        self.BC_box = self.node_set
        self.BC_function = self.BC_node_template
    
        

        ## Iterate through keyword arguments and store all in object
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
         # check for other intuitive parameter names
        name_dict = {}
        name_dict['mesh_dh'] = ['spacing']
        name_dict['mesh_res'] = ['resolution','res']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    setattr(self,key,getattr(self,name))
                except AttributeError:
                    pass

        
        ## Translate input to load method
        self.load_type = self.stringtoMethod(self.load_type)
        
    def run(self,mesh,controller):
        print("Loading mesh...")
        self.set_mesh_inputs(mesh,controller)
        self.load_type(mesh,controller)
            
       
    ## Mesh methods
    def box(self,mesh,controller):
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
        
        mesh.q = np.zeros((mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        mesh.rho = np.zeros((mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        mesh.E = np.zeros((3,mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        mesh.B = np.zeros((3,mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        
        mesh.phi = np.zeros((mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        mesh.BC_vector = np.zeros((mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
        
        mesh.q_bk = np.zeros((mesh.q.shape),dtype=np.float)
        mesh.rho_bk = np.zeros((mesh.rho.shape),dtype=np.float)
        mesh.E_bk = np.zeros((mesh.E.shape),dtype=np.float)
        mesh.B_bk = np.zeros((mesh.B.shape),dtype=np.float)
        
        mesh = self.BC_box(mesh,controller)
        
        if self.store_node_pos == True:
            mesh.pos = np.zeros((3,mesh.xres+2,mesh.yres+2,mesh.zres+2),dtype=np.float)
            for xi in range(0,mesh.xres+2):
                mesh.pos[0,xi,:,:] = mesh.xlimits[0] + mesh.dx * xi
            for yi in range(0,mesh.yres+2):
                mesh.pos[1,:,yi,:] = mesh.ylimits[0] + mesh.dy * yi
            for zi in range(0,mesh.zres+2):
                mesh.pos[2,:,:,zi] = mesh.zlimits[0] + mesh.dz * zi

        return mesh
        

    def node_set(self,mesh,controller):
        for yi in range(0,mesh.yres+1):
            for xi in range(0,mesh.xres+1):
                y = mesh.ylimits[0] + mesh.dy * yi
                x = mesh.xlimits[0] + mesh.dx * xi
                
                z0 = mesh.zlimits[0]
                zn = mesh.zlimits[1]
                mesh.phi[xi,yi,0] = self.BC_function(np.array([x,y,z0]))
                mesh.phi[xi,yi,-2] = self.BC_function(np.array([x,y,zn]))

        mesh.BC_vector[:,:,1] += mesh.phi[:,:,0]/mesh.dz**2
        mesh.BC_vector[:,:,-3] += mesh.phi[:,:,-2]/mesh.dz**2    
        
        if controller.ndim >= 2:
            for zi in range(0,mesh.zres+1):
                for xi in range(0,mesh.xres+1):
                    x = mesh.ylimits[0] + mesh.dx * xi
                    z = mesh.zlimits[0] + mesh.dz * zi
                    
                    y0 = mesh.ylimits[0]
                    yn = mesh.ylimits[1]
                    mesh.phi[xi,0,zi] = self.BC_function(np.array([x,y0,z]))
                    mesh.phi[xi,-2,zi] = self.BC_function(np.array([x,yn,z]))
                    
            mesh.BC_vector[:,1,:] += mesh.phi[:,0,:]/mesh.dy**2
            mesh.BC_vector[:,-3,:] += mesh.phi[:,-2,:]/mesh.dy**2
        
        if controller.ndim == 3:
            for zi in range(0,mesh.zres+1):
                for yi in range(0,mesh.yres+1):
                    y = mesh.ylimits[0] + mesh.dy * yi
                    z = mesh.zlimits[0] + mesh.dz * zi
                    
                    x0 = mesh.xlimits[0]
                    xn = mesh.xlimits[1]
                    mesh.phi[0,yi,zi] = self.BC_function(np.array([x0,y,z]))
                    mesh.phi[-2,yi,zi] = self.BC_function(np.array([xn,y,z]))
                    
            mesh.BC_vector[1,:,:] += mesh.phi[0,:,:]/mesh.dx**2
            mesh.BC_vector[-3,:,:] += mesh.phi[-2,:,:]/mesh.dx**2
        
        mesh.BC_vector = mesh.BC_vector * self.BC_scaling                
        mesh.BC_vector = self.meshtoVector(mesh.BC_vector[1:-2,1:-2,1:-2])

        return mesh

    
    def set_mesh_inputs(self,mesh,controller):
    # Set dependent mesh parameters and enforce input dimensionality.
        try:
            assert controller.xlimits[1] - controller.xlimits[0] > 0
            assert controller.ylimits[1] - controller.ylimits[0] > 0
            assert controller.zlimits[1] - controller.zlimits[0] > 0
            
            mesh.xlimits = controller.xlimits
            mesh.ylimits = controller.ylimits
            mesh.zlimits = controller.zlimits

        except AssertionError:
            print("Mesh Loader: One of the input box-edge limits is not positive " +
                  "in length. Reverting to default 2x2x2 cube centered on origin.")
            mesh.xlimits = np.array([-1,1])
            mesh.ylimits = np.array([-1,1])
            mesh.zlimits = np.array([-1,1])
            
        if self.grid_input == 'cell_count' or self.mesh_dh == None:
            try:
                res = np.zeros(3,dtype=np.int)
                res[:] = self.mesh_res
                self.mesh_res = res
                
                dx = (mesh.xlimits[1] - mesh.xlimits[0])/self.mesh_res[0]
                dy = (mesh.ylimits[1] - mesh.ylimits[0])/self.mesh_res[1]
                dz = (mesh.zlimits[1] - mesh.zlimits[0])/self.mesh_res[2]
                self.mesh_dh = np.array([dx,dy,dz])
                self.grid_input = 'cell_count'
                
            except:
                print("Exception in handling mesh resolution input, trying by cell length.")
                self.grid_iput = 'cell_length'
            
                
        if self.grid_input == 'cell_length':
            try:
                type_test = 1/self.mesh_dh
            except:
                print("Exception in handling mesh resolution input, defaulting to 2x2x2 resolution.")
                self.mesh_dh = (mesh.xlimits[1] - mesh.xlimits[0])/2
                self.mesh_dh = (mesh.ylimits[1] - mesh.ylimits[0])/2
                self.mesh_dh = (mesh.ylimits[1] - mesh.ylimits[0])/2

            xres = (mesh.xlimits[1] - mesh.xlimits[0])/self.mesh_dh[0]
            yres = (mesh.ylimits[1] - mesh.ylimits[0])/self.mesh_dh[1]
            zres = (mesh.zlimits[1] - mesh.zlimits[0])/self.mesh_dh[2]
            self.mesh_res = np.array([xres,yres,zres],dtype=np.int)
            self.grid_iput = 'cell_length'
                
        self.cell_volume = np.prod(self.mesh_dh[0:])
        
        if controller.ndim == 2:
            try:
                assert self.mesh_res[0] == 2

            except AssertionError:
                self.mesh_res[0] = 2
                self.mesh_dh[0] = (self.xlimits[1] - self.xlimits[0])/self.mesh_res[0]
            
            self.cell_volume = np.prod(self.mesh_dh[1:])
            
        elif controller.ndim == 1:
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
    
    
    def custom_ph(self,mesh,controller):
        print('No custom case method specified, mesh loader will do nothing.')        
    
    def stringtoMethod(self,front):
        try:
            function = getattr(self,front)
            front = function
        except TypeError:
            pass
        
        return front
            