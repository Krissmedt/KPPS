from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *
from dataHandler2 import dataHandler2
import matplotlib.animation as animation

def update_line(num, xdata,ydata, line):
        line.set_data(xdata[num,:],ydata[num,:])
        return line


ppc = 20
L = 2*pi
res = 63
dt = 0.01
Nt = 100

v = 1
vmod = 0.01
a = 1

simulate = False
sim_name = 'two_stream_1d(1)'


############################ Setup and Run ####################################
sim_params = {}
species_params = {}
mesh_params = {}
case_params = {}
analysis_params = {}
data_params = {}

sim_params['tSteps'] = Nt
sim_params['simID'] = sim_name
sim_params['t0'] = 0
sim_params['dt'] = dt
sim_params['percentBar'] = True
sim_params['dimensions'] = 1

species_params['nq'] = ppc*res
species_params['mq'] = 1
species_params['q'] = 1

mesh_params['node_charge'] = ppc*species_params['q']

case_params['particle_init'] = 'direct'
case_params['pos'] = particle_pos_init(ppc,res,L)
case_params['vel'] = particle_vel_init(case_params['pos'],v,vmod,a)
case_params['zlimits'] = [0,L]

case_params['mesh_init'] = 'box'
case_params['resolution'] = [2,2,res]
#case_params['BC_function'] = bc_pot
case_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_synced'
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['looped_axes'] = ['z']
analysis_params['centreMass_check'] = False

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
analysis_params['background'] = ion_bck
analysis_params['units'] = 'custom'
analysis_params['mesh_boundary_z'] = 'open'
analysis_params['poisson_M_adjust_1d'] = 'constant_phi_1d'

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
             meshSettings=mesh_params,
             analysisSettings=analysis_params,
             caseSettings=case_params,
             dataSettings=data_params)

if simulate == True:
    kppsObject = kpps(**model)
    DH = kppsObject.run()
else:
    DH = dataHandler2()
    DH.load_sim(sim_name=sim_name,overwrite=True)
    ow = False

####################### Analysis and Visualisation ############################
pData_dict = DH.load_p(['pos','vel','E'],sim_name=sim_name,overwrite=ow)
mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name,overwrite=ow)

#tsPlots = [ts for ts in range(Nt)]
tsPlots = [0,floor(Nt/4),floor(2*Nt/4),floor(3*Nt/4),-1]
Z = np.linspace(0,L,res+1)

#print(mData_dict['phi'][tsPlot,1,1,:])
fps = 10

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)


# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
line = ax.plot(pData_dict['pos'][0,:,2],pData_dict['vel'][0,:,2],'bo')[0]

# Setting the axes properties
ax.set_xlim([0.0, L])
ax.set_xlabel('z')

ax.set_ylabel('vz')
ax.set_ylim([-15, 15])
ax.set_title('Two-stream instability')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_line, DH.samples, 
                                   fargs=(pData_dict['pos'][:,:,2],pData_dict['vel'][:,:,2],line),
                                   interval=1000/fps)

plt.show()

"""
fig = plt.figure(DH.figureNo+1)
for ts in range(0,DH.samples):
    print(ts)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pData_dict['pos'][ts,:,2],pData_dict['vel'][ts,:,2],label=str(ts))
    ax.set_xscale('linear')
    ax.set_xlabel('$z$')
    ax.set_yscale('linear')
    ax.set_ylabel('vz')
    ax.legend()
    plt.show()
    plt.pause(1/(DH.samples/fps))


    fig = plt.figure(DH.figureNo+2)
    ax = fig.add_subplot(1, 1, 1)
    for ts in tsPlots:
        ax.plot(Z,mData_dict['rho'][ts,1,1,:-1],label=str(ts))
    ax.set_xscale('linear')
    ax.set_xlabel('$z$')
    ax.set_yscale('linear')
    ax.set_ylabel(r'$\rho $')
    ax.legend()
    
    fig = plt.figure(DH.figureNo+3)
    ax = fig.add_subplot(1, 1, 1)
    for ts in tsPlots:
        ax.plot(Z,mData_dict['phi'][ts,1,1,:-1],label=str(ts))
    ax.set_xscale('linear')
    ax.set_xlabel('$z$')
    ax.set_yscale('linear')
    ax.set_ylabel(r'$\phi $')
    ax.legend()
    
    fig = plt.figure(DH.figureNo+4)
    ax = fig.add_subplot(1, 1, 1)
    for ts in tsPlots:
        ax.plot(Z,mData_dict['E'][ts,2,1,1,:-1],label=str(ts))
    ax.set_xscale('linear')
    ax.set_xlabel('$z$')
    ax.set_yscale('linear')
    ax.set_ylabel(r'$E_z $')
    ax.legend()
    """