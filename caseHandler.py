#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np
import math as math

## Class
class caseHandler:
    def __init__(self,**kwargs):
        
        ## Default values
        self.particle_init = 'direct'
        self.ndim = 3
        self.dx = 1
        self.dv = 10
        
        ## Dummy values - Need to be set in params for class to work!
        self.pos = np.zeros((1,3),dtype=np.float)
        self.vel = np.zeros((1,3),dtype=np.float)
        self.species = None
        self.mesh = None
        
        ## Iterate through keyword arguments and store all in object
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
         # check for other intuitive parameter names
        try:
            self.ndim = self.dimensions
            self.pos = self.positions
            self.vel = self.velocities
        except AttributeError:
            pass
        
        ## Main functionality - setup mesh and species for specific case
        if 'distribution' in self.params:
            self.setupDistribute(self.species)
            
        if 'explicit' in self.params:
            self.setupExplicit(self.species,**self.params['explicit'])
            
    
        if self.particle_init == 'direct':
            self.direct(self.species)
        elif self.particle_init == 'clouds':
            self.clouds(self.species)
        elif self.particle_init == 'random':
            self.randDis(self.species)
        elif self.particle_init == 'even':
            self.evenPos(self.species)
                

    ## Setup Methods
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
        species.pos = self.random(species.nq,self.dx)
        species.vel = self.random(species.nq,self.dv)
    
    def random(self,rows,deviance):
        output = np.zeros((rows,self.ndim),dtype=np.float)
        for nd in range(0,self.ndim):
            for i in range(0,rows):
                output[i,nd] = rand.uniform(-deviance,deviance)
        
        return output