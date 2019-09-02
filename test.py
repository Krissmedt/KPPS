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

def trilinear_weights(rpos,dh):
    h = rpos/dh
    
    w = np.zeros(8,dtype=np.float)
    w[0] = (1-h[0])*(1-h[1])*(1-h[2])
    w[1] = (1-h[0])*(1-h[1])*(h[2])
    w[2] = (1-h[0])*(h[1])*(1-h[2])
    w[3] = (1-h[0])*(h[1])*(h[2])
    w[4] = (h[0])*(1-h[1])*(1-h[2])
    w[5] = (h[0])*(1-h[1])*(h[2])
    w[6] = (h[0])*(h[1])*(1-h[2])
    w[7] = (h[0])*(h[1])*(h[2])
    
    return w

def cell_index(pos,O,dh):
    li = np.floor((pos-O)/dh)
    li = np.array(li,dtype=np.int)
    
    return li

limits = [-1,1]
O = limits[0]
dx_mag = 1
qSum = -2

res = 20
ppc = 1
q = qSum/(res*ppc)
nq = np.int(res*ppc)
meshq = np.zeros((res+1))
meshpos = np.linspace(limits[0],limits[1],res+1)

L = limits[1] - limits[0]
spacing = L/nq

x0 = [(i+0.5)*spacing for i in range(np.int(-nq/2),np.int(nq/2))]
xi = [-dx_mag*np.power(x0i/L,3) for x0i in x0]
x2 = [-3*dx_mag*np.power(x0i,2) for x0i in x0]

x = np.zeros(len(x0))
for i in range(0,nq):
    x[i] = x0[i]+xi[i]
    print(x0[i])
    if x[i] >= limits[1]:
        x[i] = limits[0] + (x[i]-limits[1])
    elif x[i] < limits[0]:
        x[i] = limits[1] - (limits[0]-x[i])
        
pos_list = np.zeros((nq,3),dtype=np.float)
y = np.zeros((nq,3),dtype=np.float)
pos_list[:,2] = np.array(x)
print(pos_list[:,2])


dh = L/res
for pii in range(0,nq):
    li = cell_index(pos_list[pii,2],O,dh)
    rpos = pos_list[pii,2] - O - li*dh
    h = rpos/dh
    
    meshq[li] += q*(1-h)
    meshq[li+1] += q*(h)

#meshq[0] += meshq[-1]
#meshq[-1] = meshq[0]

spacing = []
for pii in range(1,nq):
    spacing.append(abs(pos_list[pii,2]-pos_list[pii-1,2]))

print(np.sum(meshq))
meshrho = meshq/dh

fig1 = plt.figure(1)
plt.plot(x0,x2,'bo')

fig2 = plt.figure(2)
plt.plot(meshpos[1:-1],meshq[1:-1])

fig3 = plt.figure(3)
plt.plot(meshpos[1:-1],meshrho[1:-1])

fig4 = plt.figure(4)
plt.plot(meshpos[1:-1],-3*np.power(meshpos[1:-1],2))