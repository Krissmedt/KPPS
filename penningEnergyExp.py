from kpps import kpps
from dataHandler import dataHandler
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time


#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'lobatto':'boris_SDC'}

M = 3
iterations = [3]


#dt = np.array([1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8,0.0125/16,0.0125/32,0.0125/64])
#dt = dt/omegaB 
#dt = 0.01
tEnd = 8                    
tsteps = 800
samples = 800
sampleInterval = floor(tsteps/samples)
dt = tEnd/tsteps

log = True

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


xAnalyt = np.zeros(tsteps+1,dtype=np.float)
yAnalyt = np.zeros(tsteps+1,dtype=np.float)
zAnalyt = np.zeros(tsteps+1,dtype=np.float)

vxAnalyt = np.zeros(tsteps+1,dtype=np.float)
vyAnalyt = np.zeros(tsteps+1,dtype=np.float)
vzAnalyt = np.zeros(tsteps+1,dtype=np.float)
exactEnergy = []


t = 0
for ts in range(0,tsteps+1):
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    if ts%sampleInterval == 0:
        u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
        exactEnergy.append(u.transpose() @ H @ u)

    t += dt


## Numerical solution ##
t1 = time.time()
for key, value in schemes.items():
    for K in iterations:
        tNum = []
        xNum = []
        yNum = []
        zNum = []
        dNum = []
        model = dict(
                simSettings = {'t0':0,'dt':dt,'tSteps':tsteps,'percentBar':True,'id':value},
            
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
                                    'centreMass':False},
                
                dataSettings = {#'write':{'sampleRate':1,'foldername':'simple'},
                                'record':{'sampleNo':samples}
                                ,'plot':{'tPlot':'xyz'}
                                ,'trajectory_plot':{'particle':1,'limits':[20,20,15]}
                                })
        

        kppsObject = kpps(**model)
        data = kppsObject.run()


        if log == True:
            filename = 't_' + key + "_" + value + "_"  + str(M) + "_" + str(K)
            filename = 'h_' + key + "_" + value + "_"  + str(M) + "_" + str(K)
            filename2 = 'h_exact_' + key + "_" + value + "_"  + str(M) + "_" + str(K)
            np.savetxt(filename,data.tArray) 
            np.savetxt(filename,data.hArray)   
            np.savetxt(filename2,exactEnergy)
        

        exactEnergy = np.array(exactEnergy)
        energyError = abs(data.hArray-exactEnergy)
        energyConvergence = energyError - energyError
        
        for i in range(0,len(energyConvergence)-1):
            energyConvergence = energyConvergence-energyConvergence
            
        
        label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)

        
        ##Energy Plot
        h_fig = plt.figure(52)
        h_ax = h_fig.add_subplot(1, 1, 1)
        h_ax.scatter(data.tArray[1:],energyError[1:],label=label_order)
        
        if key == 'boris':
            break
        
t2 = time.time()
print("t=" + str(t2-t1))

## energy plot finish
h_ax.set_xscale('log')
h_ax.set_xlim(10**-1,10**5)
h_ax.set_xlabel('$t$')

h_ax.set_yscale('log')
h_ax.set_ylim(10**-12,10**6)
h_ax.set_ylabel('$\Delta E$')
h_ax.legend()

#runtime = t1-t0
#print(runtime)