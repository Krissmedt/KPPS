#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np

## Class
class caseHandler:
    ndim = 3
    def __init__(self,species,**kwargs):
        if 'dimensions' in kwargs:
            self.ndim = kwargs['dimensions']
        
        if 'explicitSetup' in kwargs:
            self.setupExplicit(species,**kwargs['explicitSetup'])
            
        if 'distribution' in kwargs:
            self.setupDistribute(species,**kwargs['distribution'])
    
    
    def setupExplicit(self,species,**kwargs):
        if 'positions' in kwargs:
            posInput = np.array(kwargs['positions'])
            nPos = posInput.size / 3
            nPos = int(nPos)
            if nPos <= species.nq:
                species.pos[:nPos,:] = posInput
                
        if 'velocities' in kwargs:
            velInput = np.array(kwargs['velocities'])
            nVel = velInput.size / 3
            nVel = int(nVel)
            if nVel <= species.nq:
                species.vel[:nVel,:] = velInput
                
        return species
                
                
    def setupDistribute(self,species,**kwargs):
        if 'even' in kwargs:
            self.evenPos(species)
        elif 'random' in kwargs:
            self.randPos(species)
    
    
    def evenPos(self,species):
        return species
    
    
    def randPos(self,species):
        for nd in range(0,self.ndim):
            for i in range(0,len(species.pos)):
                species.pos[i,nd] = rand.random()
        
        return species