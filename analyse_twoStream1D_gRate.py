from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *
from dataHandler2 import dataHandler2
import matplotlib.animation as animation

#sim_name = 'hail_mary3_double'
sim_name = 'tsi_1d_boris_staggered_1000_100.0'
omega_p = 1
res = 64

k = 1
v = 1

k2 = k**2
v2 = v**2

roots = [None,None,None,None]
roots[0] = cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[1] = cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[2] = -cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[3] = -cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))

real_slope = roots[1].imag

DH = dataHandler2()
sim, name = DH.load_sim(sim_name=sim_name,overwrite=True)


plot_params = {}
plot_params['legend.fontsize'] = 8
plot_params['figure.figsize'] = (6,4)
plot_params['axes.labelsize'] = 12
plot_params['axes.titlesize'] = 12
plot_params['xtick.labelsize'] = 8
plot_params['ytick.labelsize'] = 8
plot_params['lines.linewidth'] = 2
plot_params['axes.titlepad'] = 5
plt.rcParams.update(plot_params)
####################### Analysis and Visualisation ############################
dt = sim.dt
Nt = sim.tSteps

pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)

p1Data_dict = pData_list[0]
p2Data_dict = pData_list[1]

mData_dict = DH.load_m(['phi','E','rho','PE_sum'],sim_name=sim_name)

tArray = mData_dict['t']

phi_data = mData_dict['phi'][:,1,1,:-1]
PE_data = mData_dict['PE_sum']


## Growth rate phi plot setup
start_dt = 0
max_t = 12
max_steps = np.int(max_t/sim.dt)
g_tArray = tArray[5:max_steps]

## Growth rate phi plot setup
max_phi_data = np.amax(np.abs(phi_data[5:max_steps]),axis=1)
max_phi_data_log = np.log(max_phi_data)

g_slope = (max_phi_data_log[(start_dt+1):-1] - max_phi_data_log[start_dt:-2])/dt
avg_slope = round(np.average(g_slope),2)

errors = g_slope - real_slope
avg_error = np.average(errors)
avg_error = avg_error/real_slope * 100
slope_text = 'Avg. slope from 6s to ' + str(max_t) + 's: ' + str(avg_slope)
fig = plt.figure(DH.figureNo+3,dpi=150)
gphi_ax = fig.add_subplot(1,1,1)
line_gphi = gphi_ax.plot(g_tArray,max_phi_data_log)
text_gphi = gphi_ax.text(.25,.05,slope_text,transform=gphi_ax.transAxes,
                         verticalalignment='bottom',fontsize=8)
#g_ax.set_xlim([0.0, sim.dt*sim.tSteps])
gphi_ax.set_xlabel('$t$')
gphi_ax.set_ylabel(r'$\log(|\phi|_{max}$)')
#g_ax.set_ylim([0,2])
gphi_ax.set_title('Two stream instability phi growth, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))



## Growth rate rho plot setup
rho_data = mData_dict['rho'][:,1,1,:-1]
max_rho_data = np.amax(np.abs(rho_data[5:max_steps]),axis=1)
max_rho_data_log = np.log(max_rho_data)

g_rho_slope = (max_rho_data_log[(start_dt+1):-1] - max_rho_data_log[start_dt:-2])/dt
avg_rho_slope = round(np.average(g_rho_slope),2)

errors_rho = g_rho_slope - real_slope
avg_error_rho = np.average(errors_rho)
avg_error_rho = avg_error_rho/real_slope * 100
slope_text = 'Avg. slope from 6s to ' + str(max_t) + 's: ' + str(avg_rho_slope)
fig = plt.figure(DH.figureNo+4,dpi=150)
grho_ax = fig.add_subplot(1,1,1)
line_grho = grho_ax.plot(g_tArray,max_rho_data_log)
text_grho = grho_ax.text(.25,.05,slope_text,transform=gphi_ax.transAxes,
                         verticalalignment='bottom',fontsize=8)
#g_ax.set_xlim([0.0, sim.dt*sim.tSteps])
grho_ax.set_xlabel('$t$')
grho_ax.set_ylabel(r'$\log(|\rho|_{max}$)')
#g_ax.set_ylim([0,2])
grho_ax.set_title('Two stream instability rho growth, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))



## Growth rate vel plot setup
vel_data = p1Data_dict['vel'][:,:,2]
max_vel_data = np.amax(np.abs(vel_data[5:max_steps]),axis=1)
max_vel_data_log = np.log(max_vel_data)

g_vel_slope = (max_vel_data_log[(start_dt+1):-1] - max_vel_data_log[start_dt:-2])/dt
avg_vel_slope = round(np.average(g_vel_slope),2)

errors_vel = g_vel_slope - real_slope
avg_error_vel = np.average(errors_vel)
avg_error_vel = avg_error/real_slope * 100
slope_text = 'Avg. slope from 6s to ' + str(max_t) + 's: ' + str(avg_vel_slope)

fig = plt.figure(DH.figureNo+5,dpi=150)
gvel_ax = fig.add_subplot(1,1,1)
line_gvel = gvel_ax.plot(g_tArray,max_vel_data_log)
text_gvel = gvel_ax.text(.05,.95,slope_text,transform=gvel_ax.transAxes,
                         verticalalignment='bottom',fontsize=8)
#g_ax.set_xlim([0.0, sim.dt*sim.tSteps])
gvel_ax.set_xlabel('$t$')
gvel_ax.set_ylabel(r'$\log(|v_1|_{max}$)')
#g_ax.set_ylim([0,2])
gvel_ax.set_title('Two stream instability vel growth, dt=' + str(dt) + ', Nt=' + str(Nt) +', Nz=' + str(res+1))
