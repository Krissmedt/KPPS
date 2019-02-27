from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *

ppc = 20
L = 4*pi
res = 10
T = 0.1
Nt = 1

v = 1
vmod = 0.01
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
sim_params['dimensions'] = 1

species_params['nq'] = ppc*res
species_params['mq'] = 1
species_params['q'] = 1

case_params['particle_init'] = 'direct'
case_params['pos'] = particle_pos_init(ppc,res,L)
#print(particle_pos_init(ppc,res,L))
case_params['vel'] = particle_vel_init(case_params['pos'],v,vmod,a)
case_params['zlimits'] = [-L/2,L/2]

case_params['mesh_init'] = 'box'
case_params['resolution'] = [2,2,res]
case_params['BC_function'] = bc_pot
case_params['store_node_pos'] = False

analysis_params['particleIntegration'] = False
analysis_params['particleIntegrator'] = 'boris_synced'
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['periodic_axes'] = ['z']
analysis_params['centreMass_check'] = False
analysis_params['hook_list'] = ['display_residuals']

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
analysis_params['fieldIntegrator_methods'] = ['scatter'] 
analysis_params['units'] = 'custom'
analysis_params['periodic_mesh'] = True

data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['time_plotting'] = False
data_params['time_plot_vars'] = ['pos'] 
data_params['tagged_particles'] = [1]
data_params['plot_limits'] = [1,1,L]

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
DH = kppsObject.run()


####################### Analysis and Visualisation ############################
pData_dict = DH.load_p(['pos','vel','E'])
mData_dict = DH.load_m(['phi','E','rho'])

tsPlot = 1
Z = np.linspace(-L/2,L/2,res+1)

print(mData_dict['phi'][tsPlot,1,1,:])

fig = plt.figure(DH.figureNo+1)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(pData_dict['pos'][tsPlot,:,2],pData_dict['vel'][tsPlot,:,2])
ax.set_xscale('linear')
ax.set_xlabel('$z$')
ax.set_yscale('linear')
ax.set_ylabel('vz')

fig = plt.figure(DH.figureNo+2)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(Z,mData_dict['rho'][tsPlot,1,1,:])
ax.set_xscale('linear')
ax.set_xlabel('$z$')
ax.set_yscale('linear')
ax.set_ylabel(r'$\rho $')

fig = plt.figure(DH.figureNo+3)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(Z,mData_dict['phi'][tsPlot,1,1,:])
ax.set_xscale('linear')
ax.set_xlabel('$z$')
ax.set_yscale('linear')
ax.set_ylabel(r'$\phi $')

fig = plt.figure(DH.figureNo+4)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(Z,mData_dict['E'][tsPlot,2,1,1,:])
ax.set_xscale('linear')
ax.set_xlabel('$z$')
ax.set_yscale('linear')
ax.set_ylabel(r'$E_z $')