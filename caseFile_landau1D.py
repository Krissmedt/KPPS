from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D

def particle_vel_maxwellian(pos_list,v,v_th,v_off=4):
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
            
#        V0 = -v_off if V0 < -v_off else V0
#        V0 = v_off if V0 > v_off else V0
#        V1 = -v_off if V1 < -v_off else V1
#        V1 = v_off if V1 > v_off else V1

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

def perturb_vel(pos_list,vel_list,dv_mag,dv_mode):
    for pii in range(0,pos_list.shape[0]):
        z = pos_list[pii,2]
        vel_list[pii,2] += dv_mag*sin(dv_mode*z)
        
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
        
    over_max = conc_data > v_max
    under_min = conc_data < v_min

    conc_data[over_max] = v_max
    conc_data[under_min] = v_min

    sorted_data = np.sort(conc_data)
    mapped_data = sorted_data - v_min

    dv = (v_max-v_min)/res
    v_array = np.linspace(v_min,v_max-dv,res+1) + dv/2
    cells = np.floor(mapped_data/dv)
    unique, counts = np.unique(cells,return_counts=True)

    vel_dist = np.zeros(res+1,dtype=np.float)
    
    j = 0
    for i in unique:
        i = int(i)
        vel_dist[i] = counts[j]
        j += 1
    
    # probability distribution from particle count in each cell given by
    # probability per cell: distribution / particle count
    # probability per unit velocity volume (f): prob. per cell/cell size:
    vel_dist = vel_dist/conc_data.shape[0]/dv
    
    return v_array, vel_dist


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

    pos_data = np.array([])
    vel_data = np.array([])
    for i in range(0,len(vel_data_list)):
        pos_data = np.concatenate((pos_data,pos_data_list[i]))
        vel_data = np.concatenate((vel_data,vel_data_list[i]))
        
    over_max = vel_data > v_off
    under_min = vel_data < -v_off

    vel_data[over_max] = v_off
    vel_data[under_min] = -v_off

    for pii in range(0,pos_data.shape[0]):
        lix = np.int(pos_data[pii]/dx)
        liv = np.int((vel_data[pii]+v_off)/dv) 
        hx = (pos_data[pii] - lix*dx)/dx
        hv = (vel_data[pii] + v_off - liv*dv)/dv
        
        f[liv,lix] += (1-hx)*(1-hv)
        f[liv+1,lix] += (1-hx)*(hv)
        f[liv,lix+1] += (hx)*(1-hv)
        f[liv+1,lix] += (hx)*(hv)
        
    f[:,0] += f[:,-1]
    
    f = f/pos_data.shape[0]/(dx*dv)
    
    pn = np.sum(f[0:-1,0:-1],axis=0) * dv
    pdist = np.sum(f[0:-1,0:-1],axis=1) * dx
    
    return grid_x[0:-1,0:-1],grid_v[0:-1,0:-1],f[0:-1,0:-1],pn,pdist


def lit_iv(res,mag,mode,L,v_off):
    xi = np.linspace(0,L,res+1)
    vi = np.linspace(-v_off,v_off,res+1)
    
    grid_x, grid_v = np.meshgrid(xi,vi)
    dx = L/res
    dv = 2*v_off/res

    f = np.zeros(grid_x.shape,dtype=np.float)
    for xi in range(0,res+1):
        for vi in range(0,res+1):
            x = xi*dx
            v = -v_off + vi*dv
            f[vi,xi] = 1/np.sqrt(2*np.pi) * (1+mag*np.cos(mode*x)) * np.exp(-v**2/2)
            
    n = np.sum(f,axis=0) * dv
    dist = np.sum(f,axis=1) * dx / L
    
    return grid_x, grid_v, f, n, dist


res = 100
mag = 0.0
mode = 0.5
L = 4*np.pi
v_off = 4
nq = 2**14
v_th = 2

ppos = ppos_init_sin(nq,L,mag,mode,ftype='cos')
pvel = particle_vel_maxwellian(ppos,0,v_th)

v_array, pdist = vel_dist([pvel[:,2]],res,-v_off,v_off)
grid_x,grid_v, fp, pn, pdist2 = calc_density_mesh([ppos[:,2]],[pvel[:,2]],res,res,v_off,L)


grid_x, grid_v, f, n, dist = lit_iv(res,mag,mode,L,v_off)

fig = plt.figure(1)
ax_f = fig.add_subplot(111)
cont = ax_f.contourf(grid_x,grid_v,f,cmap='inferno')
#cont.set_clim(0,np.max(f))
cbar = plt.colorbar(cont,ax=ax_f)


fig = plt.figure(2)
ax_f = fig.add_subplot(111)
cont = ax_f.contourf(grid_x,grid_v,fp,cmap='inferno')
#cont.set_clim(0,np.max(f))
cbar = plt.colorbar(cont,ax=ax_f)

fvel = 1/np.sqrt(2*np.pi)*np.exp(-np.power(grid_v[:,0],2)/2)
fig = plt.figure(3)
ax_vel = fig.add_subplot(111)
ax_vel.plot(grid_v[:,0],pdist)
ax_vel.plot(grid_v[:,0],pdist2)
ax_vel.plot(grid_v[:,0],fvel)
ax_vel.legend()

fig = plt.figure(4)
ax_pos = fig.add_subplot(111)
#ax_pos.plot(grid_x[0,:],n)
ax_pos.plot(grid_x[0,:],pn)
#ax_pos.plot(grid_x[0,:],(1+mag*np.cos(mode*grid_x[0,:])))
ax_pos.legend()

dist_int = np.sum(dist[:17]) * 2*v_off/res * nq
print(dist_int)