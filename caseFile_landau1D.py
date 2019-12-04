from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D

def particle_vel_maxwellian(pos_list,v,v_th):
    # initialise an evenly populated velocity phase-spaced corresponding to:
    # pos_list: NQx3 array of particle positions
    # v: Bulk particle velocity (mean vel of all particles together)
    # v_th: Thermal speed defined as v_th = sqrt(2*k_B*T/mq)
    
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    vel_list[:] = v
    for pii in range(0,pos_list.shape[0]-1,2):
        U1 = random.random()
        U2 = random.random()
        Z0 = np.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)
        Z1 = np.sqrt(-2*math.log(U1))*math.sin(2*math.pi*U2)
        
        V0 = v_th/np.sqrt(2) * Z0
        V1 = v_th/np.sqrt(2) * Z1
        vel_list[pii,2] += V0
        vel_list[pii+1,2] += V1

    return vel_list

def particle_pos_init(nq,L,dx_mag,dx_mode):
    spacing = L/nq
    
    
    x0 = [(i+0.5)*spacing for i in range(0,nq)]
    xi = [dx_mag*math.sin(2*math.pi*dx_mode*x0i/L) for x0i in x0]
    x = [x0[i]+xi[i] for i in range(0,nq)]
    
    pos_list = np.zeros((nq,3),dtype=np.float)
    pos_list[:,2] = np.array(x)

    return pos_list

def ppos_init_sin(nq,L,dx_mag,dx_mode,ftype='sin'):
    spacing = L/nq
    n0 = 1
    
    x0 = [(i+0.5)*spacing for i in range(0,nq)]
    
    if ftype == 'sin':
        xi = [-dx_mag/(n0*dx_mode)*math.cos(dx_mode*x0i) for x0i in x0]
    elif ftype == 'cos':
        xi = [dx_mag/(n0*dx_mode)*math.sin(dx_mode*x0i) for x0i in x0]
        
    x = [x0[i]+xi[i] for i in range(0,nq)]
    
    pos_list = np.zeros((nq,3),dtype=np.float)
    pos_list[:,2] = np.array(x)

    return pos_list


def particle_vel_init(pos_list,v,dv_mag,dv_mode):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    for pii in range(0,pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  v + dv_mag*sin(dv_mode*z)
        
    return vel_list


def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species_list,mesh,controller=None,q_bk=None):
    threshold = 1e-10

    q_bk[1,1,:-2] += mesh.node_charge
    q_bk[np.abs(q_bk) < threshold] = 0
    
    mesh.q_bk = q_bk
    return q_bk


def vel_dist(vel_data_list,res,v_min,v_max):
    # Produce the probability distribution function f, from particle samples
    # of velocity. 
    # vel_data: List of 1D arrays of velocity values (each array a species)
    # res: Resolution desired in plotting (balance vs. total particle count)
    # v_min: Minimum velocity (smaller than smallest sample)
    # v_max: Maximum velocity (larger than largest sample)
    
    conc_data = np.array([])
    for i in range(0,len(vel_data_list)):
        conc_data = np.concatenate((conc_data,vel_data_list[i]))

    sorted_data = np.sort(conc_data)
    mapped_data = sorted_data - v_min

    dv = (v_max-v_min)/res
    v_array = np.linspace(v_min,v_max-dv,res) + dv/2
    cells = np.floor(mapped_data/dv)
    unique, counts = np.unique(cells,return_counts=True)
    
    vel_dist = np.zeros(res,dtype=np.float)
    
    j = 0
    for i in unique:
        i = int(i)
        vel_dist[i] = counts[j]
        j += 1
        
    vel_dist = vel_dist/conc_data.shape[0]/dv # probability per unit dv
    
    return v_array, vel_dist


