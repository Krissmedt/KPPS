import io
import pickle as pk
import numpy as np
import time
import cmath as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh
from caseFile_landau1D import *
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor

def lower_index(pos,O,dh):
    li = np.floor((pos-O)/dh)
    li = np.array(li,dtype=np.int)
    
    return li

def trilinear_weights(rpos,dh):
    h = rpos/dh
    
    w = np.zeros((rpos.shape[0],8),dtype=np.float)
    w[:,0] = (1-h[:,0])*(1-h[:,1])*(1-h[:,2])
    w[:,1] = (1-h[:,0])*(1-h[:,1])*(h[:,2])
    w[:,2] = (1-h[:,0])*(h[:,1])*(1-h[:,2])
    w[:,3] = (1-h[:,0])*(h[:,1])*(h[:,2])
    w[:,4] = (h[:,0])*(1-h[:,1])*(1-h[:,2])
    w[:,5] = (h[:,0])*(1-h[:,1])*(h[:,2])
    w[:,6] = (h[:,0])*(h[:,1])*(1-h[:,2])
    w[:,7] = (h[:,0])*(h[:,1])*(h[:,2])
    
    return w


pos = np.random.rand(20,3)
field = np.zeros((11,11,11),dtype=np.float)
field2 = np.zeros((11,11,11),dtype=np.float)
O = np.array([0,0,0])
dh = 0.1
li = lower_index(pos,O,dh)
rpos = pos - O - li*dh

rpos = pos - O - li*dh
w = trilinear_weights(rpos,dh)
nq = pos.shape[0]

i = li[:,0]
j = li[:,1]
k = li[:,2]

mE = np.random.rand(3,11,11,11)
E = np.zeros((nq,3),dtype=np.float)

for pii in range(0,nq):
    i,j,k = li[pii,:]
    E[pii] = (w[pii,0]*mE[:,i,j,k] +
              w[pii,1]*mE[:,i,j,k+1] +
              w[pii,2]*mE[:,i,j+1,k] + 
              w[pii,3]*mE[:,i,j+1,k+1] +
              w[pii,4]*mE[:,i+1,j,k] +
              w[pii,5]*mE[:,i+1,j,k+1] +
              w[pii,6]*mE[:,i+1,j+1,k] + 
              w[pii,7]*mE[:,i+1,j+1,k+1])
    
E2 = np.zeros((nq,3),dtype=np.float)
i = li[:,0]
j = li[:,1]
k = li[:,2]

for comp in range(0,3):
    E2[:,comp] += w[:,0]*mE[comp,i,j,k]
    E2[:,comp] += w[:,1]*mE[comp,i,j,k+1]
    E2[:,comp] += w[:,2]*mE[comp,i,j+1,k]
    E2[:,comp] += w[:,3]*mE[comp,i,j+1,k+1]
    E2[:,comp] += w[:,4]*mE[comp,i+1,j,k]
    E2[:,comp] += w[:,5]*mE[comp,i+1,j,k+1]
    E2[:,comp] += w[:,6]*mE[comp,i+1,j+1,k]
    E2[:,comp] += w[:,7]*mE[comp,i+1,j+1,k+1]