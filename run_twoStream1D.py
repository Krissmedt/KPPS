from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *

ppc = 20
L = 2*pi
res = 10
T = 0.1
Nt = 10

k = 10
a = 1

sim_name = 'two_stream_1d'


############################ Setup and Run ####################################
sim_params = {}
species_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['tSteps'] = Nt
sim_params['simID'] = sim_name
sim_params['t0'] = 0
sim_params['tEnd'] = T
sim_params['percentBar'] = True

species_params['nq'] = ppc*res
species_params['mq'] = 1
species_params['q'] = 1

case_params['dimensions'] = 1
case_params['particle_init'] = 'direct'
case_params['pos'] = particle_pos_init(ppc,res,L)
case_params['vel'] = particle_vel_init(case_params['pos'],k,a)
case_params['zlimits'] = [0,L]

case_params['mesh_init'] = 'box'
case_params['resolution'] = [2,2,res]
case_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_SDC'
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['periodic_axes'] = ['z']
analysis_params['centreMass_check'] = False
analysis_params['hook_list'] = ['display_residuals']

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
#analysis_params['background'] = 
analysis_params['units'] = 'custom'
analysis_params['periodic_mesh'] = True

data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['time_plotting'] = True
data_params['time_plotting_vars'] = ['pos'] 
data_params['tagged_particles'] = 'all'

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


####################### Analysis and Visualisation ############################


