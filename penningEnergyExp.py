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
samples = 800

log = True

omegaB = 25.0
omegaE = 4.9
epsilon = -1

sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = 8
sim_params['tSteps'] = 800
sim_params['percentBar'] = False
dt = sim_params['tEnd']/sim_params['tSteps']


species_params['mq'] = 1
species_params['q'] = 1
alpha = species_params['q']/species_params['mq']

case_params['dimensions'] = 3
case_params['explicit'] = {}
case_params['explicit']['expType'] = 'direct'
case_params['dx'] = 0.01
case_params['dv'] = 5

#case_params['positions'] = np.array([[10,0,0]])
#case_params['velocities'] = np.array([[100,0,100]])

case_params['pos'] = np.array([[10,0,0]])
case_params['vel'] = np.array([[100,0,100]])

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = species_params['mq'] /2 * np.diag(H[0])

analysis_params['M'] = 3
analysis_params['centreMass_check'] = False
analysis_params['residual_check'] = False
analysis_params['fieldAnalysis'] = 'coulomb'
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/alpha
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/alpha

data_params['record'] = {}
data_params['record']['sampleInterval'] = floor(sim_params['tSteps']/samples)

data_params['plotSettings'] = {}
data_params['plotSettings']['legend.fontsize'] = 12
data_params['plotSettings']['figure.figsize'] = (12,8)
data_params['plotSettings']['axes.labelsize'] = 20
data_params['plotSettings']['axes.titlesize'] = 20
data_params['plotSettings']['xtick.labelsize'] = 16
data_params['plotSettings']['ytick.labelsize'] = 16
data_params['plotSettings']['lines.linewidth'] = 3

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


xAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)
yAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)
zAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)

vxAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)
vyAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)
vzAnalyt = np.zeros(sim_params['tSteps']+1,dtype=np.float)
exactEnergy = []


t = 0
for ts in range(0,sim_params['tSteps']+1):
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    if ts%data_params['record']['sampleInterval'] == 0:
        u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
        exactEnergy.append(u.transpose() @ H @ u)

    t += dt


## Numerical solution ##
t1 = time.time()
for key, value in schemes.items():
    analysis_params['particleIntegration'] = value
    analysis_params['nodeType'] = key
    for K in iterations:
        analysis_params['K'] = K
        
        tNum = []
        xNum = []
        yNum = []
        zNum = []
        dNum = []
        

        model = dict(simSettings=sim_params,
                     speciesSettings=species_params,
                     analysisSettings=analysis_params,
                     caseSettings=case_params,
                     dataSettings=data_params)


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