from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2 as DH


simulate = True
sim_no = 0

#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'lobatto':'boris_SDC'}

M = 5
iterations = [1,5]

omegaB = 25.0
omegaE = 4.9
epsilon = -1

#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8,0.0125/16])              
dt = np.array([0.02,0.01,0.005])
#dt = np.array([0.01])

sim_params = {}
species_params = {}
ploader_params = {}
analysis_params = {}
data_params = {}


sim_params['t0'] = 0
sim_params['tEnd'] = 1
sim_params['percentBar'] = True
sim_params['dimensions'] = 3
sim_params['xlimits'] = [0,20]
sim_params['ylimits'] = [0,20]
sim_params['zlimits'] = [0,15]
sim_params['simID'] = 'simple_penning'

species_params['q'] = 1
species_params['a'] = 1
mq = species_params['q']/species_params['a']
ploader_params['load_type'] = 'direct'
ploader_params['speciestoLoad'] = [0]
ploader_params['pos'] = np.array([[10,0,0]])
ploader_params['vel'] = np.array([[100,0,100]])

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])


analysis_params['particleIntegration'] = True
analysis_params['M'] = M
analysis_params['K'] = 3
analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'coulomb'
analysis_params['external_fields'] = True
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']
analysis_params['hooks'] = ['energy_calc_penning']
analysis_params['H'] = H

analysis_params['centreMass_check'] = True
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = True


data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['write_vtk'] = False
data_params['time_plotting'] = False
data_params['tagged_particles'] = 'all'
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = True
data_params['trajectories'] = [1]
data_params['plot_limits'] = [20,20,15]


plot_params = {}
plot_params['legend.fontsize'] = 12
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 10
data_params['plot_params'] = plot_params


species_params = [species_params]
loader_params = [ploader_params]

run_times_inner = np.zeros((dt.shape[0],len(iterations)),dtype=np.float)
run_times = []

## Analytical solution ##
x0 = ploader_params['pos']
v0 = ploader_params['vel']
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
    
    if ts%data_params['samplePeriod'] == 0:
        u = np.array([xAnalyt[ts],vxAnalyt[ts],yAnalyt[ts],vyAnalyt[ts],zAnalyt[ts],vzAnalyt[ts]])
        exactEnergy.append(u.transpose() @ H @ u)

    
    t += dt[-1]


## Numerical solution ##
for key, value in schemes.items():
    analysis_params['particleIntegrator'] = value
    analysis_params['nodeType'] = key
    
    j = 0
    for K in iterations:
        analysis_params['K'] = K
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
                         pLoaderSettings=loader_params,
                         analysisSettings=analysis_params,
                         dataSettings=data_params)
            
            kppsObject = kpps(**model)
            
            
            if simulate == True:
                dHandler = kppsObject.run()
                s_name = dHandler.controller_obj.simID
            elif simulate == False:
                dHandler = DH()
                s_name = sim_params['simID'] + '(' + str(sim_no) + ')'
                if sim_no == 0:
                    s_name = sim_params['simID']

            sim, garbage = dHandler.load_sim(sim_name=s_name,overwrite=True)
            rhs_evals[i] = sim.rhs_eval
            
            var_list = ['pos','energy']
            data_list = dHandler.load_p(var_list,sim_name=s_name)
            data_dict = data_list[0]
            
            tArray = data_dict['t']
            xArray = data_dict['pos'][:,0,0]
            yArray = data_dict['pos'][:,0,1]
            zArray = data_dict['pos'][:,0,2]
            
            hArray = data_dict['energy']
            
            xRel[i] = abs(xArray[-1] - xAnalyt[-1])/abs(xAnalyt[-1])
            yRel[i] = abs(yArray[-1] - yAnalyt[-1])/abs(yAnalyt[-1])
            zRel[i] = abs(zArray[-1] - zAnalyt[-1])/abs(zAnalyt[-1])
            
            run_times_inner[i,j] = sim.runTime
            
            sim_no += 1
            
        j += 1
        
        exactEnergy = np.array(exactEnergy)
        energyError = abs(hArray[1:]-exactEnergy[1:])
        energyConvergence = energyError - energyError[0]
        
        for i in range(0,len(energyConvergence)-1):
            energyConvergence[i] = energyConvergence[i+1]-energyConvergence[i]
            

        label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
        label_traj = label_order + ", dt=" + str(dt[-1])
        
        
        ##Order Plot w/ rhs
        fig_rhs = plt.figure(dHandler.figureNo+1)
        ax_rhs = fig_rhs.add_subplot(1, 1, 1)
        ax_rhs.plot(rhs_evals,xRel,label=label_order)

        
        ##Order Plot w/ dt
        fig_dt = plt.figure(dHandler.figureNo+2)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dt,xRel,label=label_order)
        
        
        ##Energy Plot
        fig2 = plt.figure(dHandler.figureNo+3)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(tArray[1:],hArray[1:],label=label_order)
        
        if key == 'boris':
            break
    
    run_times.append(run_times_inner)
        

## Order plot finish
ax_rhs.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_rhs.set_xlabel('Number of RHS evaluations')
ax_rhs.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_rhs.set_ylabel('$\Delta x^{(rel)}$')

xRange = ax_rhs.get_xlim()
yRange = ax_rhs.get_ylim()

ax_rhs.plot(xRange,dHandler.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_rhs.plot(xRange,dHandler.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_rhs.plot(xRange,dHandler.orderLines(-8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_rhs.legend()


## Order plot finish
ax_dt.set_xscale('log')
#ax_dt.set_xlim(10**-3,10**-1)
ax_dt.set_xlabel('\Delta t$')
ax_dt.set_yscale('log')
#ax_dt.set_ylim(10**(-7),10**1)
ax_dt.set_ylabel('$\Delta x^{(rel)}$')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,dHandler.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,dHandler.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.plot(xRange,dHandler.orderLines(8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_dt.legend()



## Energy plot finish
ax2.set_xlim(0,sim_params['tEnd'])
ax2.set_xlabel('$t$')
ax2.set_ylim(0,10**4)
ax2.set_ylabel('$\Delta E$')
ax2.legend()

print(run_times[0])

