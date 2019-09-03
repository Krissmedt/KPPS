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

def scatter_1d(pos_list,q):
    for pii in range(0,nq):
        li = cell_index(pos_list[pii,2],O,dh)

        rpos = pos_list[pii,2] - O - li*dh
        h = rpos/dh
        
        meshq[li] += q*(1-h)
        meshq[li+1] += q*(h)
    return meshq

outerLimits = [-3,3]
limits = [-1,1]
plot_ilims = [0,None]
O = limits[0]
dx_mag = 10
qSum = -(limits[1]**3 - limits[0]**3)

res = 200
ppc = 1000

nq = np.int(res*ppc)
meshq = np.zeros((res+1))
meshpos = np.linspace(limits[0],limits[1],res+1)

L = limits[1] - limits[0]
outerL = outerLimits[1] - outerLimits[0]
spacing = outerL/nq

x0 = [(i+0.5)*spacing for i in range(np.int(-nq/2),np.int(nq/2))]
xi = [-dx_mag*np.power(x0i/outerL,3) for x0i in x0]
x2 = [-3*dx_mag*np.power(x0i,2) for x0i in x0]

x = []
for i in range(0,nq):
    x.append(x0[i]+xi[i])
    
    if x[i] >= limits[1] or x[i] < limits[0]:
        x[i] = 'out'
        xi[i] = 'out'
        x0[i] = 'out'
        
        
x = [y for y in x if y != 'out']
x0 = [y for y in x0 if y != 'out']
xi = [y for y in xi if y != 'out']

nq = len(x)

pos_spec1 = np.zeros((nq,3),dtype=np.float)
pos_spec2 = np.zeros((nq,3),dtype=np.float)

pos_spec1[:,2] = np.array(x)
pos_spec2[:,2] = np.array(x0)

q = qSum/nq
q2 = -q

dh = L/res

meshq = scatter_1d(pos_spec1,q)
#meshq = scatter_1d(pos_spec2,q2)

meshq[0] = meshq[0]*2
meshq[-1] = meshq[-1]*2
#meshq = meshq + nq*q2/(res+1)
meshrho = meshq/dh


spacing = []
p_midpoints = []
for pii in range(1,nq):
    space = abs(pos_spec1[pii,2]-pos_spec1[pii-1,2])
    spacing.append(space)
    p_midpoints.append(x[pii]-space)


si = plot_ilims[0]
ei = plot_ilims[1]

fig1 = plt.figure(1)
plt.plot(p_midpoints,spacing,'bo')

fig2 = plt.figure(2)
plt.plot(x0,xi,'b')


fig3, ax3 = plt.subplots()
ax3.plot(meshpos[si:ei],meshq[si:ei],label=r'Mesh $q$')
ax3.set_xlim([-1,1])
#ax4.set_ylim([-0.21,0.21])

ax3b = ax3.twinx()
ax3.plot(meshpos[si:ei],-3*np.power(meshpos[si:ei],2)*dh,'r',label=r'Real $q$')
#ax4b.set_ylim([-1,1])
fig3.legend(loc='upper right',bbox_to_anchor=(0.85,0.83))


fig4, ax4 = plt.subplots()
ax4.plot(meshpos[si:ei],meshrho[si:ei],label=r'Mesh $\rho$')
ax4.set_xlim([-1,1])
#ax4.set_ylim([-0.21,0.21])

ax4b = ax4.twinx()
ax4.plot(meshpos[si:ei],-3*np.power(meshpos[si:ei],2),'r',label=r'Real $\rho$')
#ax4b.set_ylim([-1,1])
fig4.legend(loc='upper right',bbox_to_anchor=(0.85,0.83))

print("")

print("Total charge:")
print("Particles: " + str(nq*q))
print("Mesh: " + str(np.sum(meshq)))

print("")

print("Average density:")
print("Particles: " + str(nq*q/L))
print("Mesh: " + str(np.sum(meshrho)/(len(meshrho))))
