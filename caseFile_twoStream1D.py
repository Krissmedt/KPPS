from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def particle_pos_init(ppc,res,L,dist_type='linear'):
    nq = ppc*res
    method = None
    pos_list = np.zeros((nq,3),dtype=np.float)
    dz = L/res
    
    if dist_type == 'linear':
        method = linear_dist
    
    if dist_type == 'linear_2sp':
        method = linear_dist_2sp
    
    #try:
    for i in range(0,res):
        pos_list[ppc*i:ppc*(i+1),2] = method(ppc,i,dz)
    #except:
    #    print('No valid cell particle distribution specified.')
    
    return pos_list


def particle_vel_init(pos_list,v,k,a):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    for pii in range(0,pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  v + k*sin(a*z)
        
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

def linear_dist(ppc,cell_no,dz):
    cell_O = cell_no*dz
    pos_subList = np.linspace(cell_O,cell_O+dz,ppc+1)
    pos_subList = pos_subList[0:-1]
        
    return pos_subList

def linear_dist_2sp(ppc,cell_no,dz):
    cell_O = cell_no*dz
    pos_subList = np.zeros(ppc,dtype=np.float)
    pos_subList[0:np.int(ppc/2)] = np.linspace(cell_O,cell_O+dz,(ppc/2)+1)[0:-1]
    pos_subList[np.int(ppc/2):ppc] = np.linspace(cell_O,cell_O+dz,(ppc/2)+1)[0:-1]
    return pos_subList

def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species,mesh,controller):
    threshold = 1e-12
    mesh.rho[1,1,:-1] -= mesh.node_charge
    mesh.rho[np.abs(mesh.rho) < threshold] = 0