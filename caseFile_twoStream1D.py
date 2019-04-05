from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def particle_pos_init(ppc,res,L,dx_mag,dx_mode):
    nq = np.int(ppc*res)
    spacing = L/nq
    
    
    x0 = [(i+0.5)*spacing for i in range(0,nq)]
    xi = [dx_mag*math.cos(2*math.pi*dx_mode*x0i/L) for x0i in x0]
    x = [x0[i]+xi[i] for i in range(0,nq)]
    
    pos_list = np.zeros((nq,3),dtype=np.float)
    pos_list[:,2] = np.array(x)

    return pos_list

def particle_pos_init_2sp(ppc,res,L,dist_type='linear'):
    nq = np.int(ppc*res)
    dz = L/res
    spacing = dz/ppc
    offset = 0.0001
    pos_list = np.zeros((nq,3),dtype=np.float)

    pos_list[0:np.int(nq/2),2] = np.linspace(0,L-spacing,np.int(nq/2))
    pos_list[np.int(nq/2):nq,2] = np.linspace(0,L-spacing+offset,np.int(nq/2))
 
    return pos_list


def particle_vel_init(pos_list,v,dv_mag,dv_mode):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    for pii in range(0,pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  v + dv_mag*sin(dv_mode*z)
        
    return vel_list

def particle_vel_init_2sp(pos_list,v,k,a):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    for pii in range(0,np.int(pos_list.shape[0]/2)):
        z = pos_list[pii,2]
        vel_list[pii,2] =  v + k*sin(a*z)
        
    for pii in range(np.int(pos_list.shape[0]/2),pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  -v + k*sin(a*z)

    return vel_list


def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species_list,mesh,controller):
    threshold = 1e-10
    mesh.rho[1,1,:-1] += mesh.node_charge/mesh.dz
    mesh.rho[np.abs(mesh.rho) < threshold] = 0

