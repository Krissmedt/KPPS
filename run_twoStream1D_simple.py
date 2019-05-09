from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
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

def update_phase(num,xdata,ydata,lines,KE):
    t = '%.2E' % Decimal(str(num*dt))
    text = r't = '+ t + '; KE = ' + str(KE[num])
    p_text.set_text(text)
    
    lines = update_lines(num,xdata,ydata,lines)
    
    return lines

def update_dist(num,xdata,ydata,lines,PE):
    for ydat in ydata:
        ymin = np.min(ydat[num,:])
        ymax = np.max(ydat[num,:])
        ydat[num,:] = ydat[num,:] - ymin
        yh = ymax-ymin + 0.0001
        ydat[num,:] = ydat[num,:]/yh
    
    mintext = '%.2E' % Decimal(str(ymin))
    maxtext = '%.2E' % Decimal(str(ymax))
    t = '%.2E' % Decimal(str(num*dt))
    text = (r't = '+ t +'; $\phi$ = [' + mintext +' : '+ maxtext + ']' 
                 + '; PE = ' + str(PE[num]))
    dist_text.set_text(text)
        
    lines = update_lines(num,xdata,ydata,lines)
    
    return lines

def update_hist(num, data, histogram_axis,bins,xmin,xmax,ymax):
    histogram_axis.cla()
    histogram_axis.hist(data[num,:],bins)
    histogram_axis.set_xlim([xmin,xmax])
    histogram_axis.set_xlabel('No.')
    histogram_axis.set_ylabel(r'f')
    histogram_axis.set_ylim([0, ymax])
    t = '%.2E' % Decimal(str(num*dt))
    time = r't = '+ t
    hist_text = histogram_axis.text(.05,.95,time,
                                    transform=dist_ax.transAxes,
                                    verticalalignment='top',fontsize=14)

    return histogram_axis

ppc = 20
L = 2*pi
res = 64
dt = 0.01
Nt = 10000
tend = 30

dx_mag = 0.0001
dx_mode = 1

v = 1
dv_mag = 0
dv_mode = 1

omega = 1

nq = ppc*res

a1 = -1
q1 = omega*omega *(1/a1) * 1 * L/(nq/2)

a2 = 0.1
q2 = -q1

max_gRate = ( (q1/a1) / (q2/a2) )**(1/3)

simulate = True


############################ Setup and Run ####################################
sim_params = {}
beam1_params = {}
loader1_params = {}
beam2_params = {}
loader2_params = {}
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}

sim_params['tSteps'] = Nt
sim_params['t0'] = 0
sim_params['dt'] = dt
#sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 1
sim_params['zlimits'] = [0,L]

beam1_params['name'] = 'electrons'
beam1_params['nq'] = np.int(ppc*res)
beam1_params['a'] = a1
beam1_params['q'] = q1
loader1_params['load_type'] = 'direct'
loader1_params['speciestoLoad'] = [0]
loader1_params['pos'] = particle_pos_init(ppc,res,L,dx_mag,dx_mode)
loader1_params['vel'] = particle_vel_init(loader1_params['pos'],v,dv_mag,dv_mode)

beam2_params['name'] = 'ions'
beam2_params['nq'] = np.int(ppc*res)
beam2_params['a'] = a2
beam2_params['q'] = q2

loader2_params['load_type'] = 'direct'
loader2_params['speciestoLoad'] = [1]
loader2_params['pos'] = particle_pos_init(ppc,res,L,0,1)
loader2_params['vel'] = particle_vel_init(loader2_params['pos'],0,0,0)

species_params = [beam1_params,beam2_params]
loader_params = [loader1_params,loader2_params]

mLoader_params['load_type'] = 'box'
mLoader_params['resolution'] = [2,2,res]
#mLoader_params['BC_function'] = bc_pot
mLoader_params['store_node_pos'] = True

analysis_params['particleIntegration'] = True
#analysis_params['particleIntegrator'] = 'boris_synced'
analysis_params['particleIntegrator'] = 'boris_staggered'
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['looped_axes'] = ['z']
analysis_params['centreMass_check'] = False

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
analysis_params['units'] = 'custom'
analysis_params['mesh_boundary_z'] = 'open'
analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
analysis_params['pre_hook_list'] = ['ES_vel_rewind']
analysis_params['hooks'] = ['kinetic_energy','field_energy']

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

sim_name = 'tsi_simple_1d_' + analysis_params['particleIntegrator'] + '_' + str(Nt) + '_' + str(Nt*dt) 
sim_params['simID'] = sim_name

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
    
    sim = DH.controller_obj
    sim_name = sim.simID
else:
    DH = dataHandler2()
    sim, name = DH.load_sim(sim_name=sim_name,overwrite=True)


####################### Analysis and Visualisation ############################
pData_list = DH.load_p(['pos','vel','KE_sum'],species=['electrons','ions'],sim_name=sim_name)

p1Data_dict = pData_list[0]
p2Data_dict = pData_list[1]

mData_dict = DH.load_m(['phi','E','rho','PE_sum'],sim_name=sim_name)

tArray = mData_dict['t']
Z = np.zeros((DH.samples,res+1),dtype=np.float)
Z[:] = np.linspace(0,L,res+1)

p1_data = p1Data_dict['pos'][:,:,2]
p2_data = p2Data_dict['pos'][:,:,2]

v1_data = p1Data_dict['vel'][:,:,2] 
v2_data = p2Data_dict['vel'][:,:,2] 

vdiff = np.zeros((v1_data.shape[0]-1,v1_data.shape[1]),dtype=np.float)
for ti in range(0,v1_data.shape[0]-1):
    vdiff[ti,:] = v1_data[ti,:]-v1_data[ti+1,:]
    
max_vdiff = np.max(np.abs(vdiff))
avg_vdiff = np.mean(np.abs(vdiff),axis=1)
num_gRate = max_vdiff/dt
num_gRate_avg = np.max(avg_vdiff)/dt

v1_max = np.max(v1_data)
v2_min = np.min(v2_data)
KE_data = p1Data_dict['KE_sum'] + p2Data_dict['KE_sum']

rho_data = mData_dict['rho'][:,1,1,:-1]

phi_data = mData_dict['phi'][:,1,1,:-1]
PE_data = mData_dict['PE_sum']

fps = 1/sim.dt


## Phase animation setup
fig = plt.figure(DH.figureNo+4,dpi=150)
p_ax = fig.add_subplot(1,1,1)
line_p1 = p_ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo',label='Electrons, v=1')[0]
line_p2 = p_ax.plot(p2_data[0,0:1],v2_data[0,0:1],'ro',label='Ions, v=0')[0]
p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
p_ax.set_xlim([0.0, L])
p_ax.set_xlabel('$z$')
p_ax.set_ylabel('$v_z$')
p_ax.set_ylim([-4,4])
p_ax.set_title('Two stream instability phase space, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))
p_ax.legend()

# Setting data/line lists:
pdata = [p1_data,p2_data]
vdata = [v1_data,v2_data]
phase_lines = [line_p1,line_p2]

## Field value animation setup
fig2 = plt.figure(DH.figureNo+5,dpi=150)
dist_ax = fig2.add_subplot(1,1,1)
rho_line = dist_ax.plot(Z[0,:],rho_data[0,:],label=r'charge dens. $\rho_z$')[0]
phi_line = dist_ax.plot(Z[0,:],phi_data[0,:],label=r'potential $\phi_z$')[0]
dist_text = dist_ax.text(.05,.05,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
dist_ax.set_xlim([0.0, L])
dist_ax.set_xlabel('$z$')
dist_ax.set_ylabel(r'$\rho_z$/$\phi_z$')
dist_ax.set_ylim([-0.2, 1.2])
dist_ax.set_title('Two stream instability potential, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))
dist_ax.legend()

## Velocity histogram animation setup
hist_data = np.concatenate((v1_data,v2_data),axis=1)
fig3 = plt.figure(DH.figureNo+6,dpi=150)
hist_ax = fig3.add_subplot(1,1,1)
hist_text = hist_ax.text(.95,.95,'',transform=dist_ax.transAxes,verticalalignment='top',fontsize=14)
hist_ymax = ppc*res
hist_xmax = np.max(hist_data)
hist_xmin = np.min(hist_data)
n_bins = 40
vmag0 = 0.5*(abs(2*v))
min_vel = -3*vmag0
hist_dv = (-2*min_vel)/n_bins
hist_bins = []
for b in range(0,n_bins):
    hist_bins.append(min_vel+b*hist_dv)


# Setting data/line lists:
xdata = [Z,Z]
ydata = [rho_data,phi_data]
lines = [rho_line,phi_line]

# Creating the Animation object
phase_ani = animation.FuncAnimation(fig, update_phase, DH.samples, 
                                   fargs=(pdata,vdata,phase_lines,KE_data),
                                   interval=1000/fps)


dist_ani = animation.FuncAnimation(fig2, update_dist, DH.samples, 
                                   fargs=(xdata,ydata,lines,PE_data),
                                   interval=1000/fps)

hist_ani = animation.FuncAnimation(fig3, update_hist, Nt, 
                                   fargs=(hist_data,hist_ax,hist_bins,
                                          hist_xmin,hist_xmax,hist_ymax),
                                   interval=1000/fps)

phase_ani.save(sim_name+'_phase.mp4')
dist_ani.save(sim_name+'_dist.mp4')
hist_ani.save(sim_name+'_hist.mp4')
plt.show()

print("Setup time = " + str(sim.setupTime))
print("Run time = " + str(sim.runTime))
#print("Maximum particle growth rate in beam 1 velocity: "+str(num_gRate))
#print("Maximum average growth rate in beam 1 velocity: "+str(num_gRate_avg))