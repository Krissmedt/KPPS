from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'lobatto':'boris_SDC'}

M = 3
iterations = [1,2,4,8,16]
          
omegaB = 25.0
omegaE = 4.9
epsilon = -1

dt = np.linspace(10**-1,10,10)
dt = dt/omegaB
dt = np.flip(dt,axis=0)

log = False

omegaB = 25.0
omegaE = 4.9
epsilon = -1

sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = 1
sim_params['percentBar'] = False

species_params['mq'] = 1
species_params['q'] = 1
alpha = species_params['q']/species_params['mq']

case_params['dimensions'] = 3
case_params['explicit'] = {}
case_params['explicit']['expType'] = 'clouds'
case_params['dx'] = 0.01
case_params['dv'] = 0

#case_params['positions'] = np.array([[10,0,0]])
#case_params['velocities'] = np.array([[100,0,100]])

case_params['pos'] = np.array([[10,0,0]])
case_params['vel'] = np.array([[100,0,100]])

analysis_params['M'] = 3
analysis_params['centreMass_check'] = True
analysis_params['residual_check'] = False
analysis_params['fieldAnalysis'] = 'coulomb'
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/alpha
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/alpha

data_params['record'] = {}
data_params['record']['sampleInterval'] = 1

data_params['plot'] = {}
data_params['plot']['tPlot'] = 'xyz'
data_params['trajectory_plot'] = {}
data_params['trajectory_plot']['particles'] = [1]
data_params['trajectory_plot']['limits'] = [20,20,15]

data_params['plotSettings'] = {}
data_params['plotSettings']['legend.fontsize'] = 12
data_params['plotSettings']['figure.figsize'] = (12,8)
data_params['plotSettings']['axes.labelsize'] = 20
data_params['plotSettings']['axes.titlesize'] = 20
data_params['plotSettings']['xtick.labelsize'] = 16
data_params['plotSettings']['ytick.labelsize'] = 16
data_params['plotSettings']['lines.linewidth'] = 3


## Analytical solution ##
x0 = case_params['pos']
v0 = case_params['vel']
omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
Rplus = x0[0,0] - Rminus
Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
Iplus = x0[0,1] - Iminus

tsteps = floor(sim_params['tEnd']/dt[-1]) +1
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
dataArray = np.zeros((len(dt),3),dtype=np.float) 
rhs_evals = np.zeros(len(dt),dtype=np.float)

t = 0
for ts in range(0,tsteps):
    xAnalyt[ts] = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    yAnalyt[ts] = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    zAnalyt[ts] = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vxAnalyt[ts] = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vyAnalyt[ts] = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vzAnalyt[ts] = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)
    
    if ts%data_params['record']['sampleInterval'] == 0:
        u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
        exactEnergy.append(u.transpose() @ H @ u)

    
    t += dt[-1]


## Numerical solution ##
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
        for i in range(0,len(dt)):
            sim_params['dt'] = dt[i]
            
            xMod = Rplus*cos(omegaPlus*dt[i]) + Rminus*cos(omegaMinus*dt[i]) + Iplus*sin(omegaPlus*dt[i]) + Iminus*sin(omegaMinus*dt[i])
            yMod = Iplus*cos(omegaPlus*dt[i]) + Iminus*cos(omegaMinus*dt[i]) - Rplus*sin(omegaPlus*dt[i]) - Rminus*sin(omegaMinus*dt[i])
            zMod = x0[0,2] * cos(omegaTilde * dt[i]) + v0[0,2]/omegaTilde * sin(omegaTilde*dt[i])
            
            
            v_half_dt = [(xMod-x0[0,0])/(dt[i]),(yMod-x0[0,1])/(dt[i]),(zMod-x0[0,2])/(dt[i])]
        
            xOne = [xMod,yMod,zMod]
            vHalf = v_half_dt
            
            
            finalTs = floor(sim_params['tEnd']/dt[i])
            model = dict(simSettings=sim_params,
                         speciesSettings=species_params,
                         analysisSettings=analysis_params,
                         caseSettings=case_params,
                         dataSettings=data_params)
            

            kppsObject = kpps(**model)
            data = kppsObject.run()
            
            
            rhs_evals[i] = data.rhs_eval
            
            xRel[i] = abs(data.xArray[-1] - xAnalyt[-1])/abs(xAnalyt[-1])
            yRel[i] = abs(data.yArray[-1] - yAnalyt[-1])/abs(yAnalyt[-1])
            zRel[i] = abs(data.zArray[-1] - zAnalyt[-1])/abs(zAnalyt[-1])
            
            
        if log == True:
            dataArray[:,0] = dt
            dataArray[:,1] = rhs_evals
            dataArray[:,2] = xRel
            
            filename = key + "_" + value + "_"  + str(M) + "_" + str(K) + "_" + "winkel"
            np.savetxt(filename,dataArray)
            
            
        exactEnergy = np.array(exactEnergy)
        energyError = abs(data.hArray[1:]-exactEnergy[1:])
        energyConvergence = energyError - energyError[0]
        
        for i in range(0,len(energyConvergence)-1):
            energyConvergence[i] = energyConvergence[i+1]-energyConvergence[i]
            
        
        #Second-order line
        a = xRel[0]
        orderFour = np.zeros(len(dt),dtype=np.float)
        orderTwo = np.zeros(len(dt),dtype=np.float)
        orderM = np.zeros(len(dt),dtype=np.float)
        order2M = np.zeros(len(dt),dtype=np.float)
        
        for i in range(0,len(dt)):
            orderFour[i] = a*(dt[i]/dt[-1])**4
            orderTwo[i] = a*(dt[i]/dt[-1])**2
            orderM[i] = a*(dt[i]/dt[-1])**(2*M-2)
            order2M[i] = a*(dt[i]/dt[-1])**(2*M)
        
        label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
        label_traj = label_order + ", dt=" + str(dt[-1])
        
        
        ##Order Plot w/ rhs
        fig_rhs = plt.figure(50)
        ax_rhs = fig_rhs.add_subplot(1, 1, 1)
        ax_rhs.plot(rhs_evals,xRel,label=label_order)

        
        ##Order Plot w/ dt
        fig_dt = plt.figure(51)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dt*omegaB,xRel,label=label_order)
        
        
        ##Energy Plot
        fig2 = plt.figure(52)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(data.tArray[1:],data.hArray[1:],label=label_order)
        
        if key == 'boris':
            break
        

## Order plot finish
ax_rhs.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_rhs.set_xlabel('Number of RHS evaluations')
ax_rhs.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_rhs.set_ylabel('$\Delta x^{(rel)}$')

xRange = ax_rhs.get_xlim()
yRange = ax_rhs.get_ylim()

ax_rhs.plot(xRange,data.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_rhs.plot(xRange,data.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_rhs.plot(xRange,data.orderLines(-8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_rhs.legend()


## Order plot finish
ax_dt.set_xscale('log')
#ax_dt.set_xlim(10**-3,10**-1)
ax_dt.set_xlabel('$\Omega_B \Delta t$')
ax_dt.set_yscale('log')
#ax_dt.set_ylim(10**(-7),10**1)
ax_dt.set_ylabel('$\Delta x^{(rel)}$')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,data.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,data.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.plot(xRange,data.orderLines(8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_dt.legend()



## Energy plot finish
ax2.set_xlim(0,sim_params['tEnd'])
ax2.set_xlabel('$t$')
ax2.set_ylim(0,10**4)
ax2.set_ylabel('$\Delta E$')
ax2.legend()

