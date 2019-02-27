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

M = 3
iterations = [3]

omegaB = 25.0
omegaE = 4.9
epsilon = -1

#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8,0.0125/16])              
dt = np.array([0.01])


sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = 1
sim_params['percentBar'] = True
sim_params['simID'] = 'multi_penning'

species_params['mq'] = 1
species_params['q'] = 1
species_params['nq'] = 5
species_params['a'] = species_params['q']/species_params['mq']

case_params['dimensions'] = 3
case_params['particle_init'] = 'clouds'
case_params['dx'] = 0.01
case_params['dv'] = 0
case_params['pos'] = np.array([[0,0,0]])
case_params['vel'] = np.array([[1,0,0]])

case_params['mesh_init'] = 'box'
case_params['resolution'] = [1,2,3]
case_params['store_node_pos'] = True

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = species_params['mq']/2 * np.diag(H[0])

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_SDC'
analysis_params['M'] = M

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'coulomb'
analysis_params['external_fields'] = False
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']

analysis_params['centreMass_check'] = True
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = True

data_params['samplePeriod'] = 2
data_params['write'] = True
data_params['time_plotting'] = True
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


tsteps = floor(sim_params['tEnd']/dt[-1]) +1


## Numerical solution ##
for key, value in schemes.items():
    analysis_params['particleIntegrator'] = value
    analysis_params['nodeType'] = key
    
    for K in iterations:
        analysis_params['K'] = K
        for i in range(0,len(dt)):
            sim_params['dt'] = dt[i]
            
            finalTs = floor(sim_params['tEnd']/dt[i])
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
                dHandler = DH()
                s_name = sim_params['simID'] + '(' + str(sim_no) + ')'
                if sim_no == 0:
                    s_name = sim_params['simID']

            sim, garbage = dHandler.load_sim(sim_name=s_name,overwrite=True)
            rhs_evals[i] = sim.rhs_eval
            
            var_list = ['pos','energy']
            data_dict = dHandler.load_p(var_list,sim_name=s_name)
            
            tArray = data_dict['t']
            xArray = data_dict['pos'][:,0,0]
            yArray = data_dict['pos'][:,0,1]
            zArray = data_dict['pos'][:,0,2]
            
            sim_no += 1

        #label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
        #label_traj = label_order + ", dt=" + str(dt[-1])
        
        
        if key == 'boris':
            break
