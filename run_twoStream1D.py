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


ppc = 40
L = 2*pi
res = 63
dt = 0.01
Nt = 300

v = 1
vmod = 0.
a = 1

omega = 1

nq = ppc*res
q = omega**2 *(1/1) * 1 * L/(nq/2)

simulate = True
sim_name = 'two_stream_1d_fixed_phi'


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

species_params['nq'] = nq
species_params['q'] = 1
species_params['mq'] = species_params['q']

mesh_params['node_charge'] = ppc*species_params['q']

case_params['particle_init'] = 'direct'
case_params['pos'] = particle_pos_init_2sp(ppc,res,L,dist_type='linear')
case_params['vel'] = particle_vel_init_2sp(case_params['pos'],v,vmod,a)
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
analysis_params['poisson_M_adjust_1d'] = 'periodic_fixed_1d'

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
    sim_name = DH.controller_obj.simID
else:
    DH = dataHandler2()
    DH.load_sim(sim_name=sim_name,overwrite=True)

####################### Analysis and Visualisation ############################
pData_dict = DH.load_p(['pos','vel','E'],sim_name=sim_name)
mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name)

Z = np.linspace(0,L,res+1)

pps = np.int(ppc*res/2)
p1_data = pData_dict['pos'][:,0:pps,2]
p2_data = pData_dict['pos'][:,pps:pps*2,2]

v_data = pData_dict['vel'][:,:,2] 
v_max = np.abs(np.max(v_data))
v_min = np.abs(np.min(v_data))
v_h = v_max+v_min
v_data = v_data/v_h

v1_data = v_data[:,0:pps]
v2_data = v_data[:,pps:pps*2]

fps = 10

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)


# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
line_p1 = ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo')[0]
line_p2 = ax.plot(p2_data[0,0:1],v2_data[0,0:1],'ro')[0]

# Setting the axes properties
ax.set_xlim([0.0, L])
ax.set_xlabel('$z$')

ax.set_ylabel('$v_z$')
ax.set_ylim([-0.75, 0.75])
ax.set_title('Phase space')

# Setup data/line lists
pps = np.int(ppc*res/2)
xdata = [p1_data,p2_data]
ydata = [v1_data,v2_data]
lines = [line_p1,line_p2]

# Creating the Animation object
phase_ani = animation.FuncAnimation(fig, update_lines, DH.samples, 
                                   fargs=(xdata,ydata,lines),
                                   interval=1000/fps)


####################### Analysis and Visualisation ############################
pData_dict = DH.load_p(['pos','vel','E'],sim_name=sim_name)
mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name)

#tsPlots = [ts for ts in range(Nt)]
tsPlots = [0,floor(Nt/4),floor(2*Nt/4),floor(3*Nt/4),-1]

Z = np.zeros((DH.samples,res+1),dtype=np.float)
Z[:] = np.linspace(0,L,res+1)

pps = np.int(ppc*res/2)
p1_data = pData_dict['pos'][:,0:pps,2]
p2_data = pData_dict['pos'][:,pps:pps*2,2]

v_data = pData_dict['vel'][:,:,2] 
v_max = np.abs(np.max(v_data))
v_min = np.abs(np.min(v_data))
v_h = v_max+v_min
v_data = v_data/v_h

v1_data = v_data[:,0:pps]
v2_data = v_data[:,pps:pps*2]


rho_data = mData_dict['rho'][:,1,1,:-1]
rho_max = np.max(rho_data)
rho_data = rho_data/rho_max

phi_data = mData_dict['phi'][:,1,1,:-1]
phi_min = np.abs(np.min(phi_data))
phi_max = np.abs(np.max(phi_data))
phi_h = phi_min+phi_max
phi_data = (phi_data+phi_min)/phi_h

fps = 10
# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()




fig = plt.figure(DH.figureNo+1)
p_ax = fig.add_subplot(1,1,1)
line_p1 = p_ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo',label='Beam 1, v=1')[0]
line_p2 = p_ax.plot(p2_data[0,0:1],v2_data[0,0:1],'ro',label='Beam 2, v=-1')[0]
p_ax.set_xlim([0.0, L])
p_ax.set_xlabel('$z$')
p_ax.set_ylabel('$v_z$')
p_ax.set_ylim([-0.5, 0.5])
p_ax.legend()

fig2 = plt.figure(DH.figureNo+2)
dist_ax = fig2.add_subplot(1,1,1)
rho_line = dist_ax.plot(Z[0,:],rho_data[0,:],label=r'charge dens. $\rho_z$')[0]
phi_line = dist_ax.plot(Z[0,:],phi_data[0,:],label=r'potential $\phi_z$')[0]
dist_ax.set_xlim([0.0, L])
dist_ax.set_xlabel('$z$')
dist_ax.set_ylabel('$rho_z$/$\phi_z$')
dist_ax.set_ylim([0, 1])
dist_ax.legend()


# Setting data/line lists:
pdata = [p1_data,p2_data]
vdata = [v1_data,v2_data]
phase_lines = [line_p1,line_p2]


xdata = [Z,Z]
ydata = [rho_data,phi_data]
lines = [rho_line,phi_line]

# Creating the Animation object
phase_ani = animation.FuncAnimation(fig, update_lines, DH.samples, 
                                   fargs=(pdata,vdata,phase_lines),
                                   interval=1000/fps)


dist_ani = animation.FuncAnimation(fig2, update_lines, DH.samples, 
                                   fargs=(xdata,ydata,lines),
                                   interval=1000/fps)


phase_ani.save(sim_name+'_phase.mp4')
dist_ani.save(sim_name+'_dist.mp4')

plt.show()
