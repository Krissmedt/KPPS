from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def particle_pos_init(ppc,res,L,dx_mag,dx_mode):
    nq = int(ppc*res)
    spacing = L/nq
    
    
    x0 = [(i+0.5)*spacing for i in range(0,nq)]
    xi = [dx_mag*math.cos(2*math.pi*dx_mode*x0i/L) for x0i in x0]
    x = [x0[i]+xi[i] for i in range(0,nq)]
    
    pos_list = np.zeros((nq,3),dtype=np.float)
    pos_list[:,2] = np.array(x)

    return pos_list

def ppos_init_sin(nq,L,dx_mag,dx_mode,ftype='sin'):
    spacing = L/nq
    n0 = 1
    
    x0 = [(i+0.5)*spacing for i in range(0,nq)]
    
    try:
        if ftype == 'sin':
            xi = [dx_mag/(n0*dx_mode)*math.cos(dx_mode*x0i) for x0i in x0]
        elif ftype == 'cos':
            xi = [-dx_mag/(n0*dx_mode)*math.sin(dx_mode*x0i) for x0i in x0]
    except:
        xi = [0*x0i for x0i in x0]
        
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

def particle_vel_init_2sp(pos_list,v,k,a):
    vel_list = np.zeros(pos_list.shape,dtype=np.float)
    for pii in range(0,int(pos_list.shape[0]/2)):
        z = pos_list[pii,2]
        vel_list[pii,2] =  v + k*sin(a*z)
        
    for pii in range(int(pos_list.shape[0]/2),pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] =  -v + k*sin(a*z)

    return vel_list


def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species_list,mesh,controller=None,q_bk=None):
    threshold = 1e-10

    q_bk[1,1,:-1] += mesh.node_charge
    q_bk[np.abs(q_bk) < threshold] = 0
    
    mesh.q_bk = q_bk
    return q_bk

    
def calc_density_mesh(pos_data_list,vel_data_list,xres,vres,v_off,L):
    # Use linear interpolation to establish particle density in 1D phase-space
    # as mesh data for use in contour plotting (numpy).
    # pos_data: list of 1D arrays of particle position (each array a species)
    # vel_data: list of 1D arrays of particle velocity (each array a species)
    # xres: desired density data resolution in position
    # xres: desired density data resolution in velocity
    # v_off: cutoff velocity for domain v = [-v_off, v_off]
    # L: domain length in x for domain x = [0,L]
    
    dx = L/xres
    dv = 2*v_off/vres
    
    xi = np.linspace(0,L+dx,xres+2)
    vi = np.linspace(-v_off,v_off+dv,vres+2)
    
    grid_x, grid_v = np.meshgrid(xi,vi)
    f = np.zeros(grid_x.shape,dtype=np.float)
    n = np.zeros((xres+2),dtype=np.float)
    n1 = np.zeros((xres+2),dtype=np.float)
    
    pos_data = np.array([])
    vel_data = np.array([])
    for i in range(0,len(vel_data_list)):
        pos_data = np.concatenate((pos_data,pos_data_list[i]))
        vel_data = np.concatenate((vel_data,vel_data_list[i]))
        
    nq = pos_data.shape[0]
        
    over_max = vel_data > v_off
    under_min = vel_data < -v_off

    vel_data[over_max] = v_off
    vel_data[under_min] = -v_off
    
    for pii in range(0,pos_data.shape[0]):
        lix = int(pos_data[pii]/dx)
        liv = int((vel_data[pii]+v_off)/dv) 
        hx = (pos_data[pii] - lix*dx)/dx
        hv = (vel_data[pii] + v_off - liv*dv)/dv
        
        f[liv,lix] += (1-hx)*(1-hv)
        f[liv+1,lix] += (1-hx)*(hv)
        f[liv,lix+1] += (hx)*(1-hv)
        f[liv+1,lix] += (hx)*(hv)
        
        n1[lix] += (1-hx)
        n1[lix+1] += hx

    f[:,0] += f[:,-2]
    f[:,-2] = f[:,0]
    
    n1[0] += n1[-2]
    n1[-2] = n1[0]
    
    f = f/(dx*dv)/nq * L 
    n1 = n1[0:-1]/dx * L/nq

    n = np.sum(f[0:-1,0:-1],axis=0) * dv
    fv = np.sum(f[0:-1,0:-1],axis=1) * dx
    f_int = np.sum(fv) * dv

    
    return grid_x[0:-1,0:-1],grid_v[0:-1,0:-1],f[0:-1,0:-1],n,fv