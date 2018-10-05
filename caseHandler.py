#!/usr/bin/env python3

## Dependencies
import random as rand
import numpy as np
import math as math

## Class
class caseHandler:
    def __init__(self,species,fields,dx=1,dv=10,**kwargs):
        self.ndim = 3
        self.dx = dx
        self.dv = dv
        self.pos = np.zeros((species.nq,3),dtype=np.float)
        self.vel = np.zeros((species.nq,3),dtype=np.float)
        
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
        
        try:
            self.ndim = self.dimensions
            self.pos = self.positions
            self.vel = self.velocities
        except AttributeError:
            pass
            
        if 'distribution' in self.params:
            self.setupDistribute(species)
            
        if 'explicit' in self.params:
            self.setupExplicit(species,**self.params['explicit'])
            
    
    
    def setupExplicit(self,species,expType='direct',**kwargs):
        if 'positions' in kwargs:
            self.pos = np.array(kwargs['positions'])
                
        if 'velocities' in kwargs:
            self.vel = np.array(kwargs['velocities'])
    
        if expType == 'direct':
            self.direct(species)
        elif expType == 'clouds':
            self.clouds(species)
            
        return species
                
                
    def setupDistribute(self,species,disType='random',**kwargs):
        if disType == 'even':
            self.evenPos(species)
        elif disType == 'random':
            self.randDis(species)
        elif disType == 'point_clouds':
            self.pointClouds
    

## Explicit setup methods:
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

## Distributed setup methods:     
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