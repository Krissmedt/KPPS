from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2 as DH


simulate =  True
plot = True
prefix = ""

schemes = {'boris_SDC'}
#schemes = {'boris':'boris_synced'}

M = 5
iterations = [4]

omegaB = 25.0
omegaE = 4.9
epsilon = -1
tend = 16

x_plot_range = [-1,1]
runs = 10

omega_dt = np.logspace(x_plot_range[0],x_plot_range[1],runs)
omega_dt = np.flip(omega_dt)
dt = omega_dt/omegaB
tsteps = tend/dt
tsteps = np.floor(tend/dt)
dt = tend/tsteps
omega_dt = omegaB*dt 

#dt = np.array([0.02,0.01,0.005])
#dt = np.array([0.25,0.125,0.125/2,0.125/4,0.125/8,0.125/16])
#omega_dt= dt*omegaB

sim_params = {}
species_params = {}
ploader_params = {}
analysis_params = {}
data_params = {}


sim_params['t0'] = 0
sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 3
sim_params['xlimits'] = [0,20]
sim_params['ylimits'] = [0,20]
sim_params['zlimits'] = [0,15]
sim_params['simID'] = 'penning'

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
analysis_params['nodeType'] = 'lobatto'
analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'coulomb'
analysis_params['external_fields'] = True
analysis_params['E_type'] = 'transform'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']
analysis_params['hooks'] = ['energy_calc_penning']
analysis_params['H'] = H

analysis_params['centreMass_check'] = False
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = True


data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['write_vtk'] = False
data_params['time_plotting'] = False
data_params['tagged_particles'] = 'all'
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = False
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

exactEnergy = []

xRel = np.zeros(len(dt),dtype=np.float)
yRel = np.zeros(len(dt),dtype=np.float)
zRel = np.zeros(len(dt),dtype=np.float)
dataArray = np.zeros((len(dt),3),dtype=np.float) 
rhs_evals = np.zeros(len(dt),dtype=np.float)

t = tend
xAnalyt = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
yAnalyt = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
zAnalyt = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)

vxAnalyt = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
vyAnalyt = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
vzAnalyt = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)

u = np.array([xAnalyt,vxAnalyt,yAnalyt,vyAnalyt,zAnalyt,vzAnalyt])
exactEnergy.append(u.transpose() @ H @ u)



## Numerical solution ##
for scheme in schemes:
    analysis_params['particleIntegrator'] = scheme
    
    j = 0
    for K in iterations:
        analysis_params['K'] = K
        for i in range(0,len(dt)):
            sim_params['dt'] = dt[i]
            Nt = floor(sim_params['tEnd']/dt[i]) +1
            
            xMod = Rplus*cos(omegaPlus*dt[i]) + Rminus*cos(omegaMinus*dt[i]) + Iplus*sin(omegaPlus*dt[i]) + Iminus*sin(omegaMinus*dt[i])
            yMod = Iplus*cos(omegaPlus*dt[i]) + Iminus*cos(omegaMinus*dt[i]) - Rplus*sin(omegaPlus*dt[i]) - Rminus*sin(omegaMinus*dt[i])
            zMod = x0[0,2] * cos(omegaTilde * dt[i]) + v0[0,2]/omegaTilde * sin(omegaTilde*dt[i])
            
            
            v_half_dt = [(xMod-x0[0,0])/(dt[i]),(yMod-x0[0,1])/(dt[i]),(zMod-x0[0,2])/(dt[i])]
        
            xOne = [xMod,yMod,zMod]
            vHalf = v_half_dt
            
            if scheme == 'boris_staggered':
                analysis_params['pre_hook_list'] = ['ES_vel_rewind']
            elif scheme == 'boris_SDC':
                analysis_params['pre_hook_list'] = []
                scheme += '_M' + str(M) + 'K' + str(K)
            else:
                analysis_params['pre_hook_list'] = []
            
            sim_name = 'penning_' + prefix + '_' + scheme + '_TE' + str(tend) + '_NT' + str(Nt) 
            sim_params['simID'] = sim_name
            
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
            
            xRel[i] = abs(xArray[-1] - xAnalyt)/abs(xAnalyt)
            yRel[i] = abs(yArray[-1] - yAnalyt)/abs(yAnalyt)
            zRel[i] = abs(zArray[-1] - zAnalyt)/abs(zAnalyt)
            
            run_times_inner[i,j] = sim.runTime
            
            
        j += 1
        
        if scheme != 'boris_SDC':
            break
        
    
    run_times.append(run_times_inner)
        

if plot == True:
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_workprec" + ".h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        zrels = file["fields/errors"][:]
        label = file.attrs['label_res']
        nlo_type = file.attrs['type']


        ##Order Plot w/ rhs
        fig_rhs = plt.figure(dHandler.figureNo+1)
        ax_rhs = fig_rhs.add_subplot(1, 1, 1)
        ax_rhs.plot(rhs_evals,xRel,label=label_order)

        
        ##Order Plot w/ dt
        fig_dt = plt.figure(dHandler.figureNo+2)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(omega_dt,xRel,label=label_order)
        



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
ax_dt.set_xlabel('$\omega_B \Delta t$')
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



print(run_times[0])

