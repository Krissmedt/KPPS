#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import sqrt, fsum, pi

## Class
class kpps_analysis:
    ## Physical constants
    mu0 = 1
    ep0 = 1
    q0 = 1
    
    def __init__(self,**kwargs):
        # Load required electric methods
        self.electricAnalysis = []
        if 'electricField' in kwargs:
            self.electricAnalysis.append(self.eField)
            self.electricFieldInput = kwargs['electricField']
            
        if 'interactionModelling' in kwargs:
            if kwargs['interactionModelling'] == 'full':
                self.electricAnalysis.append(self.nope)
            elif kwargs['interactionModelling'] == 'intra':
                self.electricAnalysis.append(self.coulombIntra)   
            else:
                self.electricAnalysis.append(self.nope) 
                
        # Load required magnetic methods
        self.magneticAnalysis = []
        if 'magneticField' in kwargs:
            self.magneticAnalysis.append(self.bField)
            self.magneticFieldInput = kwargs['magneticField']
        
        # Load required time-integration methods
        self.timeIntegration = []
        if 'timeIntegration' in kwargs:
            if kwargs['timeIntegration'] == 'boris':
                self.timeIntegration.append(self.boris)
        
        
    ## Analysis modules
    def electric(self,species,**kwargs):
        species.E = np.zeros((len(species.E),3),dtype=np.float)
        for method in self.electricAnalysis:
            method(species)
        
        return species


    def magnetic(self,species,**kwargs):
        species.B = np.zeros((len(species.E),3),dtype=np.float)
        for method in self.magneticAnalysis:
            method(species)
        
        return species


    def timeIntegrator(self,species,simulationManager, **kwargs):
        for method in self.timeIntegration:
            method(species, simulationManager)
        
        return species


    ## Electric field methods
    def eField(self,species,**kwargs):
        k = 1
        if "magnitude" in self.electricFieldInput:
            k = self.electricFieldInput["magnitude"]
            
        if "sPenning" in self.electricFieldInput:
            direction = np.array(self.electricFieldInput['sPenning'])
            species.E += - species.pos * direction * k
        elif "general" in self.electricFieldInput:
            inputMatrix = np.array(self.electricFieldInput['general'])
            for pii in range(0,species.nq):
                direction = np.dot(inputMatrix,species.pos[pii,:])
                species.E[pii,:] += direction * k

        return species


    def coulombIntra(self, species,**kwargs):
        try:
            pos = species.pos
            E = species.E
        except AttributeError:
            print("Input species object either has no position array named"
                  + " 'pos' or electric field array named 'E'.")
        
        nq = len(pos)
        
        for pii in range(0,nq):
            for pjj in range(0,nq):
                if pii==pjj:
                    continue
                E[pii,:] = E[pii,:] + self.coulombForce(species.q,
                                                   pos[pii,:],
                                                   pos[pjj,:])
        
        species.E += E
        return species
    
    
    def coulombForce(self,q2,pos1,pos2):
        """
        Returns the electric field contribution on particle 1 w.r.t. 
        particle 2, where the charge of particle 2 'q2' is given in units 
        of the elementary charge q0 (i.e. actual charge = q2*q0).
        """
        
        rpos = pos1-pos2
        r = sqrt(fsum(rpos**2))
        rUnit = rpos/r
        
        Ec = 1/(4*pi*self.ep0) * q2/r**2 * rUnit
        return Ec
    
    
    
    ## Magnetic field methods
    def bField(self,species,**kwargs):
        species.B = np.array(species.B)
        settings = self.magneticFieldInput
        
        if "magnitude" in settings:
            bMag = settings["magnitude"]
        else:
            bMag = 1
        
        if "uniform" in settings:
            direction = np.array(settings["uniform"])
            try:
                species.B[:,0:] = np.multiply(bMag,direction)
            except TypeError:
                print("TypeError raised, did you input a length 3 vector "
                      + "to define the uniform magnetic field?")

        return species
        
    
    
    ## Time-integration methods
    def boris(self, species, simulationParameters):
        nq = len(species.pos)
        k = simulationParameters.dt * species.nq /(2*species.mq)
        vPlus = np.zeros((nq,3),dtype=np.float)
        
        t = k*species.B
        vMinus = species.vel + k*species.E
        for pii in range(0,nq):
            tMag = np.linalg.norm(t[pii,:])
            vDash = vMinus[pii,:] + np.cross(vMinus[pii,:],t[pii,:])
            vPlus[pii,:] = vMinus[pii,:] + np.cross( 2/(1+tMag**2)*vDash,
                                                     t[pii,:] )
        
        species.vel = vPlus + k*species.E
        species.pos = species.pos + simulationParameters.dt * species.vel
        
        return species
    
    
    
    ## Additional methods
    def nope(self,species):
        return species
    
    def makeSI(self):
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
    