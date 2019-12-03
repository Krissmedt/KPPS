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
    # takes list of 1D velocity data numpy arrays
    
    conc_data = np.array([])
    for i in range(0,len(vel_data_list)):
        conc_data = np.concatenate((conc_data,vel_data_list[i]))

    sorted_data = np.sort(conc_data)
    
    dv = (v_max-v_min)/res
    bins = np.floor(sorted_data/dv)
    
    for i in range(0,res):
        # assign number in each bin to dv bins

        
    
    return bins
        
        
        
a = np.array([1,3,8,1,9,2])
b = np.array([5,5,5,6,3,7,10])
c = np.array([9,7,8,6,3,5,4,1,2])
d = np.array([6,3,4,5,8,9,7,8,2,1,3])
dist = [a,b,c,d]

data = vel_dist(dist,10,0,10)
