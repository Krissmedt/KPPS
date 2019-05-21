from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2
import matplotlib.animation as animation

plot = False

start_time = 15
max_time = 45

sims = {}


sims['tsi_long_boris_staggered_NZ64_NQ1280_NT'] = [50,100,200]

comp_run = 'tsi_long_boris_staggered_NZ64_NQ1280_NT400'

omega_p = 1

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

DH_comp = dataHandler2()
comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
mData_comp = DH_comp.load_m(['phi','zres'],sim_name=comp_sim_name)
start_dt = np.int(start_time/(comp_sim.dt*DH_comp.samplePeriod))
max_steps = np.int(max_time/(comp_sim.dt*DH_comp.samplePeriod))+1

print(start_dt)
print(max_steps)
tArray_comp = mData_comp['t'][0:max_steps]
tArray_comp_spec = tArray_comp[start_dt:]

phi_comp_data = mData_comp['phi'][:,1,1,:-1]
phi_comp = phi_comp_data[start_dt:max_steps,:]

for key, value in sims.items():
    dts = []
    Nts = []
    rhs_evals = []
    avg_errors = []
    convergence = []
    
    phi_spec_old = 0
    
    for tsteps in value:
        DH = dataHandler2()
        sim_name = key + str(tsteps)
        sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)
        
        ####################### Analysis and Visualisation ############################
        dt = sim.dt
        Nt = sim.tSteps
        
        start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
        max_steps = np.int(max_time/(sim.dt*DH.samplePeriod))+1

        pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
        
        p1Data_dict = pData_list[0]
        p2Data_dict = pData_list[1]

        mData_dict = DH.load_m(['zres','phi','E','rho','PE_sum'],sim_name=sim_name)
        
        tArray = mData_dict['t'][0:max_steps]
        tArray_spec = tArray[start_dt:]
        
        phi_data = mData_dict['phi'][:,1,1,:-1]
        PE_data = mData_dict['PE_sum']
        
        start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
        max_steps = np.int(max_time/(sim.dt*DH.samplePeriod))+1
        
        skip_time = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
        skip_time_int = np.int(skip_time)
        
        skip_space = mData_dict['zres'][0]/(mData_comp['zres'][0])
        skip_space_int = np.int(skip_space)
        
        tArray_comp_slice = tArray_comp_spec[0::skip_time_int]
        
        ## Phi line matching
        phi_spec = phi_data[start_dt:max_steps,:]
        phi_comp_spec = phi_comp[0::skip_time_int,0::skip_space_int]
        
        diff_array = np.abs(phi_comp_spec-phi_spec)
        avg_diffs_space = np.average(diff_array,axis=1)
        avg_diffs_time = np.average(avg_diffs_space)
        error = avg_diffs_time
        """
        diff_array_conv = np.abs(phi_spec-phi_spec_old)
        avg_diffs_space_conv = np.average(diff_array,axis=1)
        avg_diffs_time_conv = np.average(avg_diffs_space)
        error_conv = avg_diffs_time
        phi_spec_old = phi_spec
        """
        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)
        avg_errors.append(error)
        #convergence.append(error_conv)
        
        
    label_order = sim_name[:-6]
    """
    ##Convergence Plot w/ rhs
    fig_con = plt.figure(DH.figureNo+4)
    ax_con = fig_con.add_subplot(1, 1, 1)
    ax_con.plot(rhs_evals[1:],convergence[1:],label=label_order)
    """
    ##Order Plot w/ rhs
    fig_rhs = plt.figure(DH.figureNo+1)
    ax_rhs = fig_rhs.add_subplot(1, 1, 1)
    ax_rhs.plot(rhs_evals,avg_errors,label=label_order)
    
    ##Order Plot w/ dt
    fig_dt = plt.figure(DH.figureNo+2)
    ax_dt = fig_dt.add_subplot(1, 1, 1)
    ax_dt.plot(dts,avg_errors,label=label_order)

## Convergence plot finish
ax_con.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_con.set_xlabel('Number of RHS evaluations')
ax_con.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_con.set_ylabel('Avg. slope difference')

xRange = ax_con.get_xlim()
yRange = ax_con.get_ylim()

ax_con.plot(xRange,DH.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_con.plot(xRange,DH.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_con.legend()


## Order plot finish
ax_rhs.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_rhs.set_xlabel('Number of RHS evaluations')
ax_rhs.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_rhs.set_ylabel('Avg. relative slope error')

xRange = ax_rhs.get_xlim()
yRange = ax_rhs.get_ylim()

ax_rhs.plot(xRange,DH.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_rhs.plot(xRange,DH.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_rhs.legend()


## Order plot finish
ax_dt.set_xscale('log')
#ax_dt.set_xlim(10**-3,10**-1)
ax_dt.set_xlabel(r'$\Delta t$')
ax_dt.set_yscale('log')
#ax_dt.set_ylim(10**(-7),10**1)
ax_dt.set_ylabel('Avg. relative slope error')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.legend()