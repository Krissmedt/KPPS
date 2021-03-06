from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sim_name = 'playground'

omegaB = 25.0
omegaE = 4.9
epsilon = -1


sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['tSteps'] = 100
sim_params['simID'] = sim_name
sim_params['t0'] = 0
sim_params['tEnd'] = 1
sim_params['percentBar'] = True

species_params['nq'] = 5
species_params['mq'] = 1
species_params['q'] = 10
species_params['a'] = species_params['q']/species_params['mq']

case_params['dimensions'] = 3
case_params['particle_init'] = 'clouds'
case_params['dx'] = 10
case_params['dv'] = 0
case_params['pos'] = np.array([[0,0,0]])
case_params['vel'] = np.array([[0,0,0]])
case_params['xlimits'] = [-20,20]
case_params['ylimits'] = [-20,20]
case_params['zlimits'] = [-15,15]

case_params['mesh_init'] = 'box'
case_params['resolution'] = [1,1,1]
case_params['store_node_pos'] = False

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = species_params['mq']/2 * np.diag(H[0])

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_SDC'
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['periodic_axes'] = ['x','y','z']
analysis_params['external_fields'] = False
analysis_params['centreMass_check'] = False
analysis_params['hook_list'] = ['display_residuals']
analysis_params['fieldAnalysis'] = 'coulomb'
analysis_params['E_type'] = 'custom'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']

data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['write_vtk'] = False
data_params['time_plotting'] = False
data_params['tagged_particles'] = 'all'
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = True
data_params['trajectories'] = [1]
data_params['domain_limits'] = [[-20,20],[-20,20],[-15,15]]

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


## Numerical solution ##
model = dict(simSettings=sim_params,
             speciesSettings=species_params,
             analysisSettings=analysis_params,
             caseSettings=case_params,
             dataSettings=data_params)


kppsObject = kpps(**model)
data = kppsObject.run()


## Analysis and Visualisation
data.trajectory_plot()
data.particle_time_plot(variables=['pos'])

