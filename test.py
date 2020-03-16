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
from gauss_lobatto import CollGaussLobatto

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

def vel_sync(species,dt):

    alpha = species.a
    E_old = species.E
    E_new = species.E

    species.E_half = (E_old+E_new)/2
    
    species.vel = boris(species.vel,species.E_half,species.B,dt,alpha)
    print('output velocity Boris: {0}'.format(species.vel))


def vel_2015(species,dm,Smat):
    species.En_m = species.E 
    half_E = (species.En_m+species.E)/2
    species.En_m = species.E              #Save m+1 value as next node's m value
    
    for m in range(1,2):
        sumS = 0
        for l in range(1,M+1):
            sumS += Smat[m+1,l] * species.F[:,l]
        
        species.vQuad = species.vn[:,m] + sumS
        
        species.ck_dm = -1/2 * (species.F[:,m+1]+species.F[:,m]) + 1/dm[m] * sumS
        
    
        v_oldNode = np.reshape(species.vn[:,m],(species.nq,3))
        species.ck_dm = np.reshape(species.ck_dm,(species.nq,3))
#        print(species.ck_dm)
        ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
        v_new = boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
        print('output velocity Boris-SDC 2015: {0}'.format(v_new))
    
    
    
def vel_2018(species,dm,q):
    species.En_m = species.E
    half_E = (species.En_m+species.E)/2
    species.En_m = species.E              #Save m+1 value as next node's m value
    species.Bn_m = species.B/2
    
    for m in range(1,2):
        # Calculate collocation terms required for pos update
        IF = 0
        for j in range(1,M+1):
            IF += (q[m+1,j]-q[m,j])*species.F[:,j]
            
        c = -dm[m]/2 * np.cross(species.vn[:,m].reshape((species.nq,3)),
                                                        species.Bn_m)
    
        c += -dm[m]/2 * np.reshape(species.F[:,m]+species.F[:,m+1],
                                  (species.nq,3)) + IF.reshape((species.nq,3))
#        print(c)
                
        c += dm[m]/2 * np.cross(species.vn[:,m].reshape((species.nq,3)),
                                               species.B)
#        print(c)
        #Resort all other 3d vectors to shape Nx3 for use in Boris function
        v_oldNode = np.reshape(species.vn[:,m],(species.nq,3))
        species.ck_dm = c
        
        ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
        v_new = boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
        print('output velocity Boris-SDC 2018: {0}'.format(v_new))
        
    
    

M = 3
K = 1
    
coll = CollGaussLobatto(M,0,1) #Initialise collocation/quadrature analysis object (class is Daniels old code)
nodes = coll._getNodes
weights = coll._getWeights(coll.tleft,coll.tright) #Get M  nodes and weights 

Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix
delta_m = coll._gen_deltas         #Generate vector of node spacings

settings = {}
settings['q'] = 1
settings['a'] = 1
settings['nq'] = 5
prtls = species(**settings)

prtls.pos = np.array([[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]])
prtls.vel = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]])
prtls.E = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
prtls.B = np.array([[0,0,2.5],[0,0,2.5],[0,0,2.5],[0,0,2.5],[0,0,2.5]])

prtls.lntz = prtls.a*(prtls.E + np.cross(prtls.vel,prtls.B))

d = 3*prtls.nq
prtls.x0 = np.zeros((d,M+1),dtype=np.float)
prtls.v0 = np.zeros((d,M+1),dtype=np.float)

prtls.xn = np.zeros((d,M+1),dtype=np.float)
prtls.vn = np.zeros((d,M+1),dtype=np.float)

prtls.F = np.zeros((d,M+1),dtype=np.float)
prtls.Fn = np.zeros((d,M+1),dtype=np.float)

prtls.x0[:,0] = np.ravel(prtls.pos)
prtls.v0[:,0] = np.ravel(prtls.vel)
prtls.F[:,0] = np.ravel(prtls.lntz)
prtls.En_m0 = prtls.E

for m in range(1,M+1):
    prtls.x0[:,m] = prtls.x0[:,0]
    prtls.v0[:,m] = prtls.v0[:,0]
    prtls.F[:,m] = prtls.F[:,0]

prtls.x = np.copy(prtls.x0)
prtls.v = np.copy(prtls.v0)

prtls.xn[:,:] = prtls.x[:,:]
prtls.vn[:,:] = prtls.v[:,:]
prtls.Fn[:,:] = prtls.F[:,:]


print(prtls.vel)
vel_2015(prtls,delta_m,Smat)
vel_2018(prtls,delta_m,Qmat)
vel_sync(prtls,0.5)