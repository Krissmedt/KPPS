from kpps_ced_ms import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

nq = 1
mq = 1
alpha = 1.
q = alpha*mq

omegaB = 25.0
omegaE = 4.9
epsilon = -1

bMag = omegaB/alpha
eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])

tEnd = 16
#tEnd = 16.0
#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.01,0.005,0.0025,0.00125])
#dt = dt/omegaB 
                        
dt = np.array([0.01])

tNum = []
xNum = []
yNum = []
zNum = []
dNum = []

## Analytical solution ##
x0 = np.array([10,0,0])
v0 = np.array([100,0,100])
omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0] + v0[1])/(omegaPlus - omegaMinus)
Rplus = x0[0] - Rminus
Iminus = (omegaPlus*x0[1] - v0[0])/(omegaPlus - omegaMinus)
Iplus = x0[1] - Iminus

tsteps = floor(tEnd/dt[-1]) +1
xAnalyt = np.zeros(tsteps,dtype=np.float)
yAnalyt = np.zeros(tsteps,dtype=np.float)
zAnalyt = np.zeros(tsteps,dtype=np.float)

vxAnalyt = np.zeros(tsteps,dtype=np.float)
vyAnalyt = np.zeros(tsteps,dtype=np.float)
vzAnalyt = np.zeros(tsteps,dtype=np.float)


t = 0
for ts in range(0,tsteps):
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[2] * cos(omegaTilde * t) + v0[2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[2] * -omegaTilde * sin(omegaTilde * t) + v0[2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    pos = np.array([xAnalyt[ts],yAnalyt[ts],zAnalyt[ts]])
    vel = np.array([vxAnalyt[ts],vyAnalyt[ts],vzAnalyt[ts]])
    k = dt[-1]*q/(2*mq)
    E = eMag * eTransform @ pos
    B = bMag * np.array([0,0,1])
    #print(k*B)
    t += dt[-1]

## Numerical solution ##    
for i in range(0,len(dt)):
    xMod = Rplus*cos(omegaPlus*dt[i]) + Rminus*cos(omegaMinus*dt[i]) + Iplus*sin(omegaPlus*dt[i]) + Iminus*sin(omegaMinus*dt[i])
    yMod = Iplus*cos(omegaPlus*dt[i]) + Iminus*cos(omegaMinus*dt[i]) - Rplus*sin(omegaPlus*dt[i]) - Rminus*sin(omegaMinus*dt[i])
    zMod = x0[2] * cos(omegaTilde * dt[i]) + v0[2]/omegaTilde * sin(omegaTilde*dt[i])
    
    
    v_half_dt = [(xMod-x0[0])/(dt[i]),(yMod-x0[1])/(dt[i]),(zMod-x0[2])/(dt[i])]

    xOne = [xMod,yMod,zMod]
    vHalf = v_half_dt
    
    
    finalTs = floor(tEnd/dt[i])
    model = dict(
            simSettings = {'t0':0,'tEnd':tEnd,'dt':dt[i]},
        
            speciesSettings = {'nq':nq,'mq':mq,'q':q},
            
            caseSettings = {'dimensions':3,
                            'explicitSetup':{'positions':x0,'velocities':v0}},
            
            analysisSettings = {'imposedElectricField':{'general':eTransform, 'magnitude':eMag},
                                'imposedMagneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                                'particleIntegration':'boris_synced',
                                'M':3,
                                'K':3},
            
            dataSettings = {#'write':{'sampleRate':1,'foldername':'simple'},
                            'record':{'sampleRate':1}
                            #,'plot':{'tPlot':'xyz'}
                            ,'single_trajectory_plot':{'particle':1,'limits':[20,20,15]}
                            })
    
    kppsObject = kpps(**model)
    data = kppsObject.run()
    
    # Distance calculation
    xDis = xAnalyt[-1] - data.xArray[-1]
    yDis = yAnalyt[-1] - data.yArray[-1]
    zDis = zAnalyt[-1] - data.zArray[-1]
    dis = sqrt(xDis**2 + yDis**2 + zDis**2)

    #print(data.tArray)
    #print(data.xArray)
    #print(data.yArray)
    #print(data.zArray)

    tNum.append(data.tArray[-1])
    xNum.append(data.xArray[-1])
    yNum.append(data.yArray[-1])
    zNum.append(data.zArray[-1])
    dNum.append(dt[i])
    dNum.append(dis)
    
    dataArray = []
    dataArray.append(data.xArray[-1])
    dataArray.append(dis)
    np.array(dataArray)
    filename = str(dt[i]) + "dt.txt"
    np.savetxt(filename,dataArray)


xRel = abs(np.array(xNum) - xAnalyt[-1])/abs(xAnalyt[-1]) 
yRel = abs(np.array(yNum) - yAnalyt[-1])/abs(yAnalyt[-1])
zRel = abs(np.array(zNum - zAnalyt[-1]))/abs(zAnalyt[-1])


#Second-order line
a = 0.0005
orderOne = np.zeros(len(dt),dtype=np.float)
orderTwo = np.zeros(len(dt),dtype=np.float)
for i in range(0,len(dt)):
    orderOne[i] = a*(dt[i]/dt[-1])
    orderTwo[i] = a*(dt[i]/dt[-1])**2


fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot3D(xAnalyt, yAnalyt, zAnalyt,label='Exact solution')
ax.plot3D(data.xParticle,data.yParticle,data.zParticle,label='KPPS Boris')
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-15,15])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

fig = plt.figure(50)
ax = fig.add_subplot(1, 1, 1)
ax.plot(dt,xRel,label='Boris')
ax.plot(dt,orderOne,label='1st Order')
ax.plot(dt,orderTwo,label='2nd Order')
ax.set_xscale('log')
#ax.set_xlim(10**-1,11)
ax.set_xlabel('$\omega_B \Delta t$')

ax.set_yscale('log')
ax.set_ylim(10**(-5),10)
ax.set_ylabel('$\Delta x$')
ax.legend()

