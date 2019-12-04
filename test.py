import io
import pickle as pk
import numpy as np
import time
import cmath as math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh
from caseFile_landau1D import *
import scipy.interpolate as scint

#def f(x,v,A,k):
#    f = 1/np.sqrt(2*np.pi) * (1+A*np.cos(k*x))*np.exp(-v**2/2)
#    return f
#
#def n(x,A,k):
#    n = (1+A*np.cos(k*x))
#    return n
#
#N = 1000
#L = 4*math.pi
#x = np.linspace(0,L,N)
#v = np.linspace(-4,4,N)
#
#A = 0.01
#k = 0.5
#
#fmat = np.zeros((len(x),len(v)),dtype=np.float)
#for i in range(0,len(x)):
#    for j in range(0,len(v)):
#        fmat[j,i] = f(x[i],v[j],A,k)
#
#
#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#ax.plot(x,n(x,A,k),label='n')
#ax.legend()
#
#fig = plt.figure(2)
#ax = fig.add_subplot(111)
#ax.contourf(x,v,fmat)


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
    
    # probability distribution from particle count in each cell given by
    # probability per cell: distribution / particle count
    # probability per unit velocity volume (f): prob. per cell/cell size:
    vel_dist = vel_dist/conc_data.shape[0]/dv
    
    return v_array, vel_dist


def calc_density_mesh(pos_data_list,vel_data_list,xres,vres,v_off,L):
    # Use griddata library to establish particle density in 1D phase-space
    # as mesh data for use in contour plotting (numpy).
    # pos_data: list of 1D arrays of particle position (each array a species)
    # vel_data: list of 1D arrays of particle velocity (each array a species)
    # xres: desired density data resolution in position
    # xres: desired density data resolution in velocity
    
    xi = np.linspace(-1,1,xres)
    vi = np.linspace(-1,1,vres)
    
    grid_x, grid_v = np.meshgrid(xi,vi)
    
    pos_data = np.array([])
    vel_data = np.array([])
    for i in range(0,len(vel_data_list)):
        pos_data = np.concatenate((pos_data,pos_data_list[i]))
        vel_data = np.concatenate((vel_data,vel_data_list[i]))
        
    xv = np.array(pos_data,vel_data)
        
    dx = L/xres
    dv = 2*v_off/vres

    for species in species_list:
        for pii in range(0,species.nq):
            li = 
            rpos = species.pos[pii] - O - li*mesh.dh
            w = self.trilinear_weights(rpos,mesh.dh)

            mesh.q[li[0],li[1],li[2]] += species.q * w[0]
            mesh.q[li[0],li[1],li[2]+1] += species.q * w[1]
            mesh.q[li[0],li[1]+1,li[2]] += species.q * w[2]
            mesh.q[li[0],li[1]+1,li[2]+1] += species.q * w[3]
            mesh.q[li[0]+1,li[1],li[2]] += species.q * w[4]
            mesh.q[li[0]+1,li[1],li[2]+1] += species.q * w[5]
            mesh.q[li[0]+1,li[1]+1,li[2]] += species.q * w[6]
            mesh.q[li[0]+1,li[1]+1,li[2]+1] += species.q * w[7]
        
    return grid_x,grid_v
    
        
nq = 100000
res = 100
v_th = 1
v_off = 4
dv = 2*v_off/res
spw = 2*v_off/nq

pos1_seed = np.random.random(int(nq/2))
pos1 = np.zeros((int(nq/2),3),dtype=np.float)
pos1[:,2] = pos1_seed

pos2_seed = np.random.random(int(nq/2))
pos2 = np.zeros((int(nq/2),3),dtype=np.float)
pos2[:,2] = pos1_seed
pos_list = [pos1_seed,pos2_seed]

vel1 = particle_vel_maxwellian(pos1,0,1)[:,2]
vel2 = particle_vel_maxwellian(pos2,0,1)[:,2]
vel_list = [vel1,vel2]

gridx, gridv = calc_density_mesh(pos_list,vel_list,res,res)


#vs, dist = vel_dist(dist,res,-v_off,v_off)
#f = np.sqrt(1/np.pi) * 1/v_th * np.exp(-np.power(vs,2)/(v_th**2))
#
#dist = dist/nq/dv
#dist_int = np.sum(dist) * dv
#
#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#ax.plot(vs,dist)
#ax.plot(vs,f)
#ax.legend()


