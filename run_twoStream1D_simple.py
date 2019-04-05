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

def update_lines(num, xdata,ydata, lines):
    for xdat,ydat,line in zip(xdata,ydata,lines):
        line.set_data(xdat[num,:],ydat[num,:])
        
    return lines

ppc = 20
L = 2*pi
res = 63
dt = 0.01
Nt = 100

dx_mag = 0.0001
dx_mode = 1

v = 1
dv_mag = 0
dv_mode = 1

simulate = True
sim_name = 'two_stream_1d_1beam'



############################ Setup and Run ####################################
sim_params = {}
beam1_params = {}
loader1_params = {}
species_params = []
loader_params = []
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}

sim_params['tSteps'] = Nt
sim_params['simID'] = sim_name
sim_params['t0'] = 0
sim_params['dt'] = dt
sim_params['percentBar'] = True
sim_params['dimensions'] = 1

sim_params['zlimits'] = [0,L]

beam1_params['nq'] = ppc*res
beam1_params['mq'] = 1
beam1_params['q'] = 1

loader1_params['load_type'] = 'direct'
loader1_params['speciestoLoad'] = [0]
loader1_params['pos'] = particle_pos_init(ppc,res,L,dx_mag,dx_mode)
loader1_params['vel'] = particle_vel_init(loader1_params['pos'],v,dv_mag,dv_mode)

species_params = [beam1_params]
loader_params = [loader1_params]

mesh_params['node_charge'] = 2*ppc*q
mLoader_params['zlimits'] = [0,L]
mLoader_params['load_type'] = 'box'
mLoader_params['resolution'] = [2,2,res]
#mLoader_params['BC_function'] = bc_pot
mLoader_params['store_node_pos'] = False

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
analysis_params['poisson_M_adjust_1d'] = 'integral_phi_1d'

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
             pLoaderSettings=loader_params,
             meshSettings=mesh_params,
             analysisSettings=analysis_params,
             mLoaderSettings=mLoader_params,
             dataSettings=data_params)

if simulate == True:
    kppsObject = kpps(**model)
    DH = kppsObject.run()
    sim_name = DH.controller_obj.simID
else:
    DH = dataHandler2()
    DH.load_sim(sim_name=sim_name,overwrite=True)

####################### Analysis and Visualisation ############################
pData_list = DH.load_p(['pos','vel','E'],sim_name=sim_name)
pData_dict = pData_list[0]
mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name)

#tsPlots = [ts for ts in range(Nt)]
tsPlots = [0,floor(Nt/4),floor(2*Nt/4),floor(3*Nt/4),-1]

Z = np.zeros((DH.samples,res+1),dtype=np.float)
Z[:] = np.linspace(0,L,res+1)

rho_data = mData_dict['rho'][:,1,1,:-1]
rho_min = np.abs(np.min(rho_data))
rho_max = np.abs(np.max(rho_data))
rho_h = rho_min+rho_max
rho_data = (rho_data+rho_min)/rho_h

phi_data = mData_dict['phi'][:,1,1,:-1]
phi_min = np.abs(np.min(phi_data))
phi_max = np.abs(np.max(phi_data))
phi_h = phi_min+phi_max
phi_data = (phi_data+phi_min)/phi_h

fps = 10

fig = plt.figure(DH.figureNo+1)
p_ax = fig.add_subplot(1,1,1)
p_line = p_ax.plot(pData_dict['pos'][0,:,2],pData_dict['vel'][0,:,2],'bo')[0]
p_ax.set_xlim([0.0, L])
p_ax.set_xlabel('$z$')
p_ax.set_ylabel('$v_z$')
p_ax.set_ylim([0, 2])
p_ax.set_title('Two stream instability phase space, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))
p_ax.legend()

fig2 = plt.figure(DH.figureNo+2)
dist_ax = fig2.add_subplot(1,1,1)
rho_line = dist_ax.plot(Z[0,:],rho_data[0,:],label=r'charge dens. $\rho_z$')[0]
phi_line = dist_ax.plot(Z[0,:],phi_data[0,:],label=r'potential $\phi_z$')[0]
dist_ax.set_xlim([0.0, L])
dist_ax.set_xlabel('$z$')
dist_ax.set_ylabel(r'$\rho_z$/$\phi_z$')
dist_ax.set_ylim([-1, 1])
dist_ax.set_title('Two stream instability potential, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))
dist_ax.legend()

# Setting data/line lists:
xdata = [Z,Z]
ydata = [rho_data,phi_data]
lines = [rho_line,phi_line]

# Creating the Animation object
phase_ani = animation.FuncAnimation(fig, update_line, DH.samples, 
                                   fargs=(pData_dict['pos'][:,:,2],pData_dict['vel'][:,:,2],p_line),
                                   interval=1000/fps)


dist_ani = animation.FuncAnimation(fig2, update_lines, DH.samples, 
                                   fargs=(xdata,ydata,lines),
                                   interval=1000/fps)


phase_ani.save(sim_name+'_phase.mp4')
dist_ani.save(sim_name+'_dist.mp4')

plt.show()
