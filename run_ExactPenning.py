from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from dataHandler2 import dataHandler2 as DH
from mpl_toolkits.mplot3d import Axes3D
import time


sim2match = 'simple_penning(1)'
DHO = DH()
sim, sim_name = DHO.load_sim(sim_name=sim2match) 
tsteps = sim.tSteps+1

nq = 1
mq = 1
alpha = 1.
q = alpha*mq

omegaB = 25.0
omegaE = 4.9
epsilon = -1

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])

bMag = omegaB/alpha
eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])



## Analytical solution ##
x0 = np.array([[10,0,0]])
v0 = np.array([[100,0,100]])
omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
Rplus = x0[0,0] - Rminus
Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
Iplus = x0[0,1] - Iminus

tArray = np.zeros(tsteps,dtype=np.float)
xAnalyt = np.zeros(tsteps,dtype=np.float)
yAnalyt = np.zeros(tsteps,dtype=np.float)
zAnalyt = np.zeros(tsteps,dtype=np.float)

vxAnalyt = np.zeros(tsteps,dtype=np.float)
vyAnalyt = np.zeros(tsteps,dtype=np.float)
vzAnalyt = np.zeros(tsteps,dtype=np.float)
exactEnergy = np.zeros(tsteps,dtype=np.float)

for ts in range(0,tsteps):
    t = sim.t0 + sim.dt * ts
    
    tArray[ts] = t
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
    exactEnergy[ts] = u.transpose() @ H @ u

    
dataArray = np.zeros((tsteps,8),dtype=np.float)

dataArray[:,0] = tArray
dataArray[:,1] = xAnalyt
dataArray[:,2] = yAnalyt
dataArray[:,3] = zAnalyt
dataArray[:,4] = vxAnalyt
dataArray[:,5] = vyAnalyt
dataArray[:,6] = vzAnalyt
dataArray[:,7] = exactEnergy

filename = './' + sim2match +"/exactPenning"
np.savetxt(filename,dataArray)


##Energy Plot
fig2 = plt.figure(51)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(tArray[1:],exactEnergy[1:])

## Energy plot finish
ax2.set_xlim(0,sim.tEnd)
ax2.set_xlabel('$t$')
ax2.set_ylim(0,10**4)
ax2.set_ylabel('$\Delta E$')
