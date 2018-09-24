from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time


#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'lobatto':'boris_SDC'}

M = 3
iterations = [3]

tEnd = 8
#tEnd = 16.0
#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4])
#dt = dt/omegaB 
                        
dt = np.array([0.01])

sampleRate = 1

log = False

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

tsteps = floor(tEnd/dt[-1]) +1
xAnalyt = np.zeros(tsteps,dtype=np.float)
yAnalyt = np.zeros(tsteps,dtype=np.float)
zAnalyt = np.zeros(tsteps,dtype=np.float)

vxAnalyt = np.zeros(tsteps,dtype=np.float)
vyAnalyt = np.zeros(tsteps,dtype=np.float)
vzAnalyt = np.zeros(tsteps,dtype=np.float)
exactEnergy = []


xRel = np.zeros(len(dt),dtype=np.float)
yRel = np.zeros(len(dt),dtype=np.float)
zRel = np.zeros(len(dt),dtype=np.float)



t = 0
for ts in range(0,tsteps):
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    if ts%sampleRate == 0:
        u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
        exactEnergy.append(u.transpose() @ H @ u)

    
    t += dt[-1]


## Numerical solution ##
for key, value in schemes.items():
    for K in iterations:
        tNum = []
        xNum = []
        yNum = []
        zNum = []
        dNum = []
        for i in range(0,len(dt)):
            xMod = Rplus*cos(omegaPlus*dt[i]) + Rminus*cos(omegaMinus*dt[i]) + Iplus*sin(omegaPlus*dt[i]) + Iminus*sin(omegaMinus*dt[i])
            yMod = Iplus*cos(omegaPlus*dt[i]) + Iminus*cos(omegaMinus*dt[i]) - Rplus*sin(omegaPlus*dt[i]) - Rminus*sin(omegaMinus*dt[i])
            zMod = x0[0,2] * cos(omegaTilde * dt[i]) + v0[0,2]/omegaTilde * sin(omegaTilde*dt[i])
            
            
            v_half_dt = [(xMod-x0[0,0])/(dt[i]),(yMod-x0[0,1])/(dt[i]),(zMod-x0[0,2])/(dt[i])]
        
            xOne = [xMod,yMod,zMod]
            vHalf = v_half_dt
            
            
            finalTs = floor(tEnd/dt[i])
            model = dict(
                    simSettings = {'t0':0,'tEnd':tEnd,'dt':dt[i],'percentBar':False},
                
                    speciesSettings = {'nq':nq,'mq':mq,'q':q},
                    
                    caseSettings = {'dimensions':3,
                                    'explicit':{'expType':'direct','positions':x0,'velocities':v0}},
                    
                    analysisSettings = {'imposedElectricField':{'general':eTransform, 'magnitude':eMag},
                                        'imposedMagneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                                        'particleIntegration':value,
                                        'M':M,
                                        'K':K,
                                        'nodeType':key,
                                        'penningEnergy':H,
                                        'centreMass':True},
                    
                    dataSettings = {#'write':{'sampleRate':1,'foldername':'simple'},
                                    'record':{'sampleInterval':sampleRate}
                                    ,'plot':{'tPlot':'xyz'}
                                    ,'trajectory_plot':{'particle':1,'limits':[20,20,15]}
                                    })
            
            kppsObject = kpps(**model)
            data = kppsObject.run()

            if log == True:
                filename = key + "_" + value + "_"  + str(M) + "_" + str(K) + "_" + str(dt[i]) + "dt.txt"
                np.savetxt(filename,data.cmArray)
            
        
            xRel[i] = abs(data.xArray[-1] - xAnalyt[-1])/abs(xAnalyt[-1])
            yRel[i] = abs(data.yArray[-1] - yAnalyt[-1])/abs(yAnalyt[-1])
            zRel[i] = abs(data.zArray[-1] - zAnalyt[-1])/abs(zAnalyt[-1])

        exactEnergy = np.array(exactEnergy)
        energyError = abs(data.hArray[1:]-exactEnergy[1:])
        energyConvergence = energyError - energyError[0]
        
        for i in range(0,len(energyConvergence)-1):
            energyConvergence[i] = energyConvergence[i+1]-energyConvergence[i]
            
        
        #Second-order line
        a = xRel[0]
        orderOne = np.zeros(len(dt),dtype=np.float)
        orderTwo = np.zeros(len(dt),dtype=np.float)
        orderM = np.zeros(len(dt),dtype=np.float)
        order2M = np.zeros(len(dt),dtype=np.float)
        
        for i in range(0,len(dt)):
            orderOne[i] = a*(dt[i]/dt[-1])
            orderTwo[i] = a*(dt[i]/dt[-1])**2
            orderM[i] = a*(dt[i]/dt[-1])**(2*M-2)
            order2M[i] = a*(dt[i]/dt[-1])**(2*M)
        
        label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
        label_traj = label_order + ", dt=" + str(dt[-1])
        
        
        ##Order Plot
        fig = plt.figure(50)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(dt,xRel,label=label_order)
        
        ##Energy Plot
        fig2 = plt.figure(51)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(data.tArray[1:],data.hArray[1:],label=label_order)
        
        if key == 'boris':
            break

## Order plot finish
#ax.plot(dt,orderOne,label='1st Order')
ax.plot(dt,orderTwo,label='2nd Order')
ax.plot(dt,orderM,label='Order 2M-2')
ax.plot(dt,order2M,label='Order 2M')
ax.set_xscale('log')
#ax.set_xlim(10**-1,11)
ax.set_xlabel('$\omega_B \Delta t$')

ax.set_yscale('log')
#ax.set_ylim(10**(-10),10)
ax.set_ylabel('$\Delta x$')
ax.legend()

## Energy plot finish
ax2.set_xlim(0,tEnd)
ax2.set_xlabel('$t$')
ax2.set_ylim(0,10**4)
ax2.set_ylabel('$\Delta E$')
ax2.legend()

#runtime = t1-t0
#print(runtime)