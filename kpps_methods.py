## Dependencies
import numpy as np
import random as rand
from math import exp, sqrt, floor, fabs, fsum, pi, e


## Universal constants
#mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
#ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
#q0 = 1.602176620898*10**(-19) #Elementary charge (C)

mu0 = 1
ep0 = 1
q0 = 1


## Electric Field Methods
def eField(species,**kwargs):
    try:
        pos = species.pos
    except AttributeError:
        print("Input species object has no position array named 'pos'.")
    
    nq = len(pos)
    E = np.zeros((nq,3),dtype=np.float)
    
    for pii in range(0,nq):
        for pjj in range(0,nq):
            if pii==pjj:
                continue
            E[pii,:] = E[pii,:] + coulomb(1,pos[pii,:],pos[pjj,:])
            
        E[pii,:] = E[pii,:] + eFieldSet(pos[pii,:],**kwargs)
   
    species.E = E
    return E



def eFieldSet(pos,**kwargs):
    Ef = np.zeros(3, dtype=np.float)
    boost = 1
    eftype = "sPenning"
    
    if "ftype" in kwargs:
        eftype = kwargs["ftype"]
    
    if "boost" in kwargs:
        boost = kwargs["boost"]
    
    if eftype == "sPenning":
        Ef[0] = -pos[0] * boost
    elif eftype == "custom":
        if "F" in kwargs:
            customF = kwargs["F"]
            Ef = customF(pos,boost)
            
    return Ef


def coulomb(q2,pos1,pos2):
    """
    Returns the electric field contribution on particle 1 w.r.t. particle 2,
    where the charge of particle 2 'q2' is given in units of the elementary
    charge q0 (i.e. actual charge = q2*q0).
    """
    
    rpos = pos1-pos2
    r = sqrt(fsum(rpos**2))
    rUnit = rpos/r
    
    Ec = 1/(4*pi*ep0) * q2/r**2 * rUnit
    return Ec



## Magnetic field methods
def bField(species,**kwargs):
    try:
        pos = species.pos
    except AttributeError:
        print("Input object has no position array named 'pos'.")
        
    nq = len(pos)
    B = np.zeros((nq,3),dtype=np.float)
    
    if "magnitude" in kwargs:
        bMag = kwargs["magnitude"]
    else:
        bMag = 5000
        
    for pii in range(0,nq):
        B[pii,0] = bMag 
        
    species.B = B
    
    return species


## Time integration methods
def boris(species, simulationParameters):
    nq = len(species.pos)
    k = simulationParameters.dt * species.nq /(2*species.mq)
    vPlus = np.zeros((nq,3),dtype=np.float)
    
    t = k*species.B
    vMinus = species.vel + k*species.E
    for pii in range(0,nq):
        tMag = np.linalg.norm(t[pii,:])
        vDash = vMinus[pii,:] + np.cross(vMinus[pii,:],t[pii,:])
        vPlus[pii,:] = vMinus[pii,:] + np.cross(2/(1+tMag**2)*vDash,t[pii,:])
    
    species.vel = vPlus + k*species.E
    species.pos = species.pos + simulationParameters.dt * species.vel
    
    return species

    
def qAccel(species,**kwargs):
    """
    Calculates acceleration for a charged particle as a function of position,
    for a given electric field at the particle position returned by function
    'E'.
    """
    
    species.F = species.E * species.q
    species.F = species.F * (species.q0)**2
    a = species.F/species.mq
    
    return [a, species]
    
    
## Other methods
def randPos(species,ndim):
    for nd in range(0,ndim):
        for xi in range(0,len(species.pos)):
            species.pos[xi,nd] = rand.random()
    
    return species