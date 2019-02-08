from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


simulate = True
sim_no = 0


schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC'}

M = 3
iterations = [1,2,4,8,16]
          
omegaB = 25.0
omegaE = 4.9
epsilon = -1

omegaB = 25.0
omegaE = 4.9
epsilon = -1

sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = 26000
sim_params['tSteps'] = 160000
sim_params['percentBar'] = True
sim_name = 'energy_exp_'

species_params['mq'] = 1
species_params['q'] = 1
species_params['a'] = 1

case_params['dimensions'] = 3
case_params['particle_init'] = 'direct'
case_params['pos'] = np.array([[10,0,0]])
case_params['vel'] = np.array([[100,0,100]])

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = species_params['mq']/2 * np.diag(H[0])

analysis_params['M'] = M
analysis_params['centreMass_check'] = False
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = False
analysis_params['fieldAnalysis'] = 'coulomb'
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']

data_params['samplePeriod'] = 100
data_params['write'] = True
data_params['write_vtk'] = False
data_params['time_plotting'] = True
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = False
data_params['trajectories'] = [1]
data_params['domain_limits'] = [20,20,15]

plot_params = {}
plot_params['legend.fontsize'] = 12
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 3

data_params['plot_params'] = plot_params


## Analytical solution ##
exactEnergy = 8799.5

## Numerical solution ##
for key, value in schemes.items():
    analysis_params['particleIntegration'] = value
    analysis_params['nodeType'] = key
    
    for K in iterations:
        analysis_params['K'] = K
        sim_params['simID'] = sim_name + str(key) + "_k=" + str(K)
        
        model = dict(simSettings=sim_params,
                     speciesSettings=species_params,
                     analysisSettings=analysis_params,
                     caseSettings=case_params,
                     dataSettings=data_params)
        

        kppsObject = kpps(**model)
        
        
        if simulate == True:
            dHandler = kppsObject.run()
            s_name = dHandler.controller_obj.simID
        elif simulate == False:
            s_name = sim_params['simID']
        
        sim, garbage = dHandler.load_sim(sim_name=s_name)
        
        var_list = ['energy']
        data_dict = dHandler.load_p(var_list,sim_name=s_name)
        
        tArray = data_dict['t']
        hArray = data_dict['energy']
  
        energyError = abs(hArray[1:]-exactEnergy)
        energyConvergence = energyError - energyError[0]
        
        for i in range(0,len(energyConvergence)-1):
            energyConvergence[i] = energyConvergence[i+1]-energyConvergence[i]
        
        ##Energy Plot
        fig2 = plt.figure(52)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(tArray[1:],hArray[1:],label=s_name)
        
        if key == 'boris':
            break 


## Energy plot finish
ax2.set_xlim(0,sim_params['tEnd'])
ax2.set_xlabel('$t$')
ax2.set_ylim(0,10**4)
ax2.set_ylabel('$\Delta E$')
ax2.legend()

