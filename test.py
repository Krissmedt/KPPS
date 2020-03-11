import io
import pickle as pk
import numpy as np
import time
import copy
import cmath as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh
from species import species
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor

def boris(vel, E, B, dt, alpha, ck=0):
    """
    Applies Boris' trick for given velocity, electric and magnetic 
    field for vector data in the shape (N x 3), i.e. particles as rows 
    and x,y,z components for the vector as the columns.
    k = delta_t * alpha / 2
    """ 

    k = dt*alpha/2
    
    tau = k*B

    vMinus = vel + dt/2 * (alpha*E + ck)
    
    tauMag = np.linalg.norm(tau,axis=1)
    vDash = vMinus + np.cross(vMinus,tau)
    
    tm = 2/(1+tauMag**2)

    for col in range(0,3):
        vDash[:,col] = tm[:] * vDash[:,col]

    vPlus = vMinus + np.cross(vDash,tau)
    
    vel_new = vPlus + dt/2 * (alpha*E + ck)

    return vel_new


def vel_2015():
    half_E = (species.En_m+species.E)/2
    species.En_m = species.E              #Save m+1 value as next node's m value
    
    sumS = 0
    for l in range(1,M+1):
        sumS += Smat[m+1,l] * species.F[:,l]
    
    species.vQuad = species.vn[:,m] + sumS
    
    species.ck_dm = -1/2 * (species.F[:,m+1]+species.F[:,m]) + 1/dm[m] * sumS

    v_oldNode = self.toMatrix(species.vn[:,m])
    species.ck_dm = self.toMatrix(species.ck_dm)
    
    ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
    v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
    
    
    
def vel_2018():
    half_E = (species.En_m+species.E)/2
    species.En_m = species.E              #Save m+1 value as next node's m value
    species.Bn_m = species.B
    
    
    # Calculate collocation terms required for pos update
    IF = 0
    for j in range(1,M+1):
        IF += (q[m+1,j]-q[m,j])*species.F[:,j]
        
    c = -dm[m]/2 * np.cross(species.vn[:,m].reshape((species.nq,3)),
                                                    species.Bn_m)

    c += -dm[m]/2 * np.reshape(species.F[:,m]+species.F[:,m+1],
                              (species.nq,3)) + IF.reshape((species.nq,3))
            
    c += -np.cross(species.vn[:,m].reshape((species.nq,3)),
                                           species.B)
    
    #Resort all other 3d vectors to shape Nx3 for use in Boris function
    v_oldNode = self.toMatrix(species.vn[:,m])
    species.ck_dm = c
    
    ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
    v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)