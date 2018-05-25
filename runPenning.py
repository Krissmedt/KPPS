from kpps_ced_ms import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np


nq = 1
mq = 1
alpha = 1.
q = alpha*mq

x0 = [10,0,0]
v0 = [100,0,100]

omegaB = 25.0
omegaE = 4.9
epsilon = -1

bMag = omegaB/alpha
eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])

tEnd = 16.0
#normalisedTs = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125,
                         #0.00625,0.003125,0.0015625,0.00078125])
normalisedTs = np.array([0.05,0.025,0.0125,0.00625,0.00625/2,0.00625/4,0.00625/8,0.00625/16])
dt = normalisedTs/omegaB 

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

xAnalyt = Rplus*cos(omegaPlus*tEnd) + Rminus*cos(omegaMinus*tEnd) + Iplus*sin(omegaPlus*tEnd) + Iminus*sin(omegaMinus*tEnd)
yAnalyt = Iplus*cos(omegaPlus*tEnd) + Iminus*cos(omegaMinus*tEnd) - Rplus*sin(omegaPlus*tEnd) - Rminus*sin(omegaMinus*tEnd)
zAnalyt = x0[2] * cos(omegaTilde * tEnd) + v0[2]/omegaTilde * sin(omegaTilde*tEnd)

## Numerical solution ##
for i in range(0,len(dt)):
    print(i)
    finalTs = floor(tEnd/dt[i])
    model = dict(
            simSettings = {'tEnd':tEnd,'dt':dt[i]},
        
            speciesSettings = {'nq':1,'mq':mq,'q':q},
            
            caseSettings = {'dimensions':3,
                            'explicitSetup':{'positions':x0,'velocities':v0}},
            
            analysisSettings = {'electricField':{'general':eTransform, 'magnitude':eMag},
                                'magneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                                'timeIntegration':'boris'},
            
            dataSettings = {#'write':{'sampleRate':1,'foldername':'simple'},
                            'record':{'sampleRate':finalTs},
                            #'plot':{'tPlot':'xyz'}
                            })
    
    kppsObject = kpps(**model)
    data = kppsObject.run()
    
    # Distance calculation
    xDis = xAnalyt - data.xArray[-1]
    yDis = yAnalyt - data.yArray[-1]
    zDis = zAnalyt - data.zArray[-1]
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


xRel = abs(np.array(xNum) - xAnalyt)/abs(xAnalyt) 
yRel = abs(np.array(yNum) - yAnalyt)/abs(yAnalyt)
zRel = abs(np.array(zNum - zAnalyt))/abs(zAnalyt)


#Second-order line
a = 0.0005
orderTwo = np.zeros(len(normalisedTs),dtype=np.float)
for i in range(0,len(normalisedTs)):
    orderTwo[i] = a*(normalisedTs[i]/normalisedTs[-1])**2

fig = plt.figure(4)
ax = fig.add_subplot(1, 1, 1)
ax.plot(normalisedTs,xRel)
ax.plot(normalisedTs,orderTwo)
ax.set_xscale('log')
#ax.set_xlim(10**-1,11)
ax.set_xlabel('$\omega_B \Delta t$')

ax.set_yscale('log')
ax.set_ylim(10**(-5),10)
ax.set_ylabel('$\Delta x$')

