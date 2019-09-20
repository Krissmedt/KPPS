#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np
import math as math

## Class
class particleLoader:
    def __init__(self,**kwargs):
        
        
        self.speciestoLoad = [0]
        
        ## Default values
        self.custom_case = self.custom_case_ph  #Assign custom case method to this
        
        ## Default species values
        self.load_type = 'custom_case'
        self.clean_load = True
        self.dx = 1
        self.dv = 10

        
        ## Dummy values - Need to be set in params for class to work!
        self.pos = np.zeros((1,3),dtype=np.float)
        self.vel = np.zeros((1,3),dtype=np.float)
        self.sim = None
        self.species_list = [None]
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
                    setattr(self,key,getattr(self,name))
                except AttributeError:
                    pass

        # Transform potential list inputs into numpy
        self.pos = np.array(self.pos)
        self.vel = np.array(self.vel)
        
        ## Translate input to load method
        self.load_type = self.stringtoMethod(self.load_type)

            
    ## Loader run loop
    def run(self,species_list,controller,**kwargs):
        for index in self.speciestoLoad:
            print("Loading species {}...".format(index+1))
            if self.clean_load == True:
                self.reset_species(species_list[index])
            self.load_type(species_list[index],controller,**kwargs)
            self.enforce_dimensionality(species_list[index],controller)
            
    ## Species methods
    def direct(self,species,controller,**kwargs):
        nPos = self.pos.shape[0]
        if nPos <= species.nq:
            species.pos[:nPos,:] = self.pos
        elif nPos > species.nq:
            print("Particle Loader: More positions than particles specified, ignoring excess entries.")
            species.pos = self.pos[:species.nq,:]
        
        nVel = self.vel.shape[0]
        if nVel <= species.nq:
            species.vel[:nVel,:] = self.vel
        elif nVel > species.nq:
            print("Particle Loader: More velocities than particles specified, ignoring excess entries.")
            species.vel = self.vel[:species.nq,:]
        
        #print(species.pos)
                
    def clouds(self,species,controller,**kwargs):
        ppc = math.floor(species.nq/self.pos.shape[0])
        for xi in range(0,len(self.pos)):
            species.pos[xi*ppc:(xi+1)*ppc,:] = self.pos[xi] + self.random(ppc,self.dx)
            species.vel[xi*ppc:(xi+1)*ppc,:] = self.vel[xi] + self.random(ppc,self.dv)
     
    def evenPos(self,species,controller):
        return species
    
    def randDis(self,species,controller):
        species.pos = self.pos[0] + self.random(species.nq,self.dx)
        species.vel = self.vel[0] + self.random(species.nq,self.dv)
        
    def random(self,rows,deviance):
        output = np.zeros((rows,3),dtype=np.float)
        for j in range(0,3):
            for i in range(0,rows):
                output[i,j] = np.random.uniform(-deviance,deviance)
        
        return output
        
    def enforce_dimensionality(self,species,controller):
        if controller.ndim == 2:
            try:
                species.pos[:,0] = controller.xlimits[0] + (controller.xlimits[1]-controller.xlimits[0])/2.
                species.vel[:,0] = 0
   
            except AttributeError:
                pass
 
        elif controller.ndim == 1:
            try:
                species.pos[:,0] = controller.xlimits[0] + (controller.xlimits[1]-controller.xlimits[0])/2.
                species.pos[:,1] = controller.ylimits[0] + (controller.ylimits[1]-controller.ylimits[0])/2.
                species.vel[:,0] = 0
                species.vel[:,1] = 0
   
            except AttributeError:
                pass    
            
    def reset_species(self,species):
        species.E = np.zeros((species.nq,3),dtype=np.float) 
        species.B = np.zeros((species.nq,3),dtype=np.float) 
        species.F = np.zeros((species.nq,3),dtype=np.float) 
        species.vel = np.zeros((species.nq,3),dtype=np.float)
        species.pos = np.zeros((species.nq,3),dtype=np.float)
            
    def custom_case_ph(self,species_list):
        print('No custom case method specified, particle loader will do nothing.')        
    
    def stringtoMethod(self,front):
        try:
            function = getattr(self,front)
            front = function
        except TypeError:
            pass
        
        return front

            