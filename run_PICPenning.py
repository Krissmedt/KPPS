from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'boris':'boris_synced'}

M = 3
iterations = [3]

omegaB = 25.0
omegaE = 4.9
epsilon = -1
        
dt = np.array([0.01])

log = False


sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = 0.01
sim_params['percentBar'] = False

species_params['nq'] = 100
species_params['q'] = 1
species_params['a'] = 1

case_params['dimensions'] = 3
case_params['particle_init'] = 'random'
case_params['dx'] = 14
case_params['dv'] = 0
case_params['pos'] = np.array([[0,0,0]])
case_params['vel'] = np.array([[0,0,0]])

case_params['mesh_init'] = 'box'
case_params['xlimits'] = [-15,15]
case_params['ylimits'] = [-15,15]
case_params['zlimits'] = [-15,15]
case_params['resolution'] = [2]
case_params['store_node_pos'] = True

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = species_params['q']/species_params['a']/2 * np.diag(H[0])

analysis_params['M'] = M
analysis_params['centreMass_check'] = False
analysis_params['residual_check'] = False
analysis_params['fieldAnalysis'] = 'pic'
analysis_params['E_type'] = 'none'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'none'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']

data_params['sampleInterval'] = 1
data_params['record'] = True
data_params['write'] = False
data_params['component_plots'] = False
data_params['components'] = 'xyz'
data_params['trajectory_plots'] = True
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


tsteps = floor(sim_params['tEnd']/dt[-1]) +1
dataArray = np.zeros((len(dt),3),dtype=np.float) 
rhs_evals = np.zeros(len(dt),dtype=np.float)


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

            finalTs = floor(sim_params['tEnd']/dt[i])
            model = dict(simSettings=sim_params,
                         speciesSettings=species_params,
                         analysisSettings=analysis_params,
                         caseSettings=case_params,
                         dataSettings=data_params)
            

            kppsObject = kpps(**model)
            data = kppsObject.run()
            print(data.mesh_q)
            rhs_evals[i] = data.rhs_eval
            
            
        if log == True:
            dataArray[:,0] = dt
            dataArray[:,1] = rhs_evals
            
            filename = key + "_" + value + "_"  + str(M) + "_" + str(K) + "_order"
            np.savetxt(filename,dataArray)
            
        label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
        label_traj = label_order + ", dt=" + str(dt[-1])
        
        

