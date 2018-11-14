from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'lobatto':'boris_SDC'}

M = 3
iterations = [3]

#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8,0.0125/16])
#dt = dt/omegaB                     
dt = np.array([0.01])

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
sim_params['tEnd'] = 1
sim_params['percentBar'] = False


species_params['nq'] = 10
species_params['mq'] = 100
species_params['q'] = 1
alpha = species_params['q']/species_params['mq']

case_params['dimensions'] = 3
case_params['particle_init'] = 'clouds'
case_params['dx'] = 0.01
case_params['dv'] = 5
case_params['pos'] = np.array([[10,0,0]])
case_params['vel'] = np.array([[100,0,100]])

analysis_params['M'] = M
analysis_params['centreMass_check'] = True
analysis_params['residual_check'] = False
analysis_params['fieldAnalysis'] = 'coulomb'
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/alpha
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/alpha

data_params['sampleInterval'] = 1
data_params['record'] = True
data_params['write'] = False
data_params['component_plots'] = True
data_params['components'] = 'xyz'
data_params['trajectory_plots'] = True
data_params['trajectories'] = 'all'
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



## Numerical solution ##
figNo = 50
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
            
            label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
            label = label_order + ", dt=" + str(dt[-1])
            
            kppsObject = kpps(**model)
            data = kppsObject.run()
            data.convertToNumpy()
    
            if log == True:
                filename = key + "_" + value + "_"  + str(M) + "_" + str(K) + "_" + str(dt[i]) + "dt.txt"              
                np.savetxt('x_' + filename,data.xArray)
                np.savetxt('y_' + filename,data.yArray)
                np.savetxt('z_' + filename,data.zArray)
                np.savetxt('h_' + filename,data.hArray)
                np.savetxt('cm_' + filename,data.cmArray)
            
            label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
            label_traj = label_order + ", dt=" + str(dt[-1])
            

            ## CoM Trajectory Plot
            fig = plt.figure(figNo+1)
            ax = fig.gca(projection='3d')
            ax.plot3D(data.cmArray[:,0],data.cmArray[:,1],data.cmArray[:,2],label=label_traj)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([-15,15])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            
            figNo += 2
        
