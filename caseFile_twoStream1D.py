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
    
    #try:
    for i in range(0,res):
        print(ppc*i)
        pos_list[ppc*i:ppc*(i+1),2] = method(ppc,i,dz)
    #except:
    #    print('No valid cell particle distribution specified.')
    
    return pos_list


def particle_vel_init(pos_list,k,a):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    
    for pii in range(0,pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  k*sin(a*z)
        
    return vel_list

def linear_dist(ppc,cell_no,dz):
    cell_O = cell_no*dz
    pos_subList = np.linspace(cell_O,cell_O+dz,ppc+2)
    pos_subList = pos_subList[1:-2]
        
    return pos_subList