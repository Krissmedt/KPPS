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


## Methods
def coulomb(q2,pos1,pos2):
    # Returns the electric field contribution on particle 1 w.r.t. particle 2,
    # where the charge of particle 2 'q2' is given in units of the elementary
    # charge q0 (i.e. actual charge = q2*q0).
    
    rpos = pos1-pos2
    r = sqrt(fsum(rpos**2))
    rUnit = rpos/r
    
    Ec = 1/(4*pi*ep0) * q2/r**2 * rUnit
    return Ec

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


def eField(pos,**kwargs):
    nq = len(pos)
    Ee = np.zeros((nq,3),dtype=np.float)
    for pii in range(0,nq):
        for pjj in range(0,nq):
            if pii==pjj:
                continue
            Ee[pii,:] = Ee[pii,:] + coulomb(1,pos[pii,:],pos[pjj,:])
            
        Ee[pii,:] = Ee[pii,:] + eFieldSet(pos[pii,:],**kwargs)
   
    return Ee


def bField(pos,**kwargs):
    nq = len(pos)
    B = np.zeros((nq,3),dtype=np.float)
    if "magnitude" in kwargs:
        bMag = kwargs["magnitude"]
    else:
        bMag = 5000
        
    for pii in range(0,nq):
        B[pii,0] = bMag 
        
    return B

def boris(pos,vel,E,B,dt,mp):
    nq = len(pos)
    k = dt*nq/(2*mp)
    t = k*B
    
    for pii in range(0,nq):
        vMinus = vel[pii,:] + k*E[pii,:]
        tMag = np.linalg.norm(t[pii,:])
        vDash = vMinus + np.cross(vMinus,t[pii,:])
        vPlus = vMinus + np.cross(2/(1+tMag**2)*vDash,t[pii,:])
    
        vel[pii,:] = vPlus + k*E[pii,:]
    
    pos = pos + dt*vel
    
    update = dict([('pos',pos),('vel',vel)])
    return update

    
def qAccel(E,pos,**kwargs):
    # Calculates acceleration for a charged particle as a function of position,
    # for a given electric field at the particle position returned by function
    # 'E'.
    
    q1 = 1 #particle charge as multiple of elementary charge q0.
    
    Ee = E(pos,**kwargs)
           
    Fe = Ee*q1
    
    Fe = Fe*q0**2
    a = Fe/mp
    
    return a
    
    
def randPos(pos,ndim):
    for nd in range(0,ndim):
        for xi in range(0,len(pos)):
            pos[xi,nd] = rand.random()
    
    return pos