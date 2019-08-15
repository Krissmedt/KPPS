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

def phase_snap(tstep,beamData1,beamData2,figNo=3):
    fig = plt.figure(figNo,dpi=150)
    p_ax = fig.add_subplot(1,1,1)
    line_p1 = p_ax.plot(beamData1['pos'][tstep,:,2],beamData1['vel'][tstep,:,2],'bo',label='Beam 1, v=1')
    line_p2 = p_ax.plot(beamData2['pos'][tstep,:,2],beamData2['vel'][tstep,:,2],'ro',label='Beam 2, v=-1')
    p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
    p_ax.set_xlim([0.0, 2*pi])
    p_ax.set_xlabel('$z$')
    p_ax.set_ylabel('$v_z$')
    p_ax.set_ylim([-4,4])
    p_ax.set_title('Two stream instability phase space')
    p_ax.legend()
    plt.show()
    print("hey")

plot = True

start_time = 0
max_time = 20

sims = {}

sims['tsi__boris_synced_NZ128_NQ2560_NT'] = [400]
#sims['tsi__boris_staggered_NZ128_NQ2560_NT'] = [50,100,200,400,800,1600]
#sims['tsi__boris_SDC_M5K5_NZ128_NQ2560_NT'] = [50,100,200]


comp_run = 'tsi__boris_SDC_M5K5_NZ128_NQ2560_NT400'

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
mData_comp = DH_comp.load_m(['phi'],sim_name=comp_sim_name)
pDataList_comp = DH_comp.load_p(['pos'],species=['beam1','beam2'],sim_name=comp_sim_name)
p1Data_comp = pDataList_comp[0] 


for key, value in sims.items():
    dts = []
    Nts = []
    rhs_evals = []
    avg_errors = []
    avg_slopes = []
    avg_errors_nonlinear = []
    
    for tsteps in value:
        DH = dataHandler2()
        sim_name = key + str(tsteps)
        sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

        ####################### Analysis and Visualisation ############################
        dt = sim.dt
        Nt = sim.tSteps
        
        start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
        max_dt = np.int(max_time/(sim.dt*DH.samplePeriod))+1
        
        pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
        
        p1Data_dict = pData_list[0]
        p2Data_dict = pData_list[1]

        mData_dict = DH.load_m(['phi','E','rho','PE_sum'],sim_name=sim_name)
        
        
        ## particle position comparison
        skip = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
        skip_int = np.int(skip)
        
        tArray = p1Data_dict['t'][:]
        tArray_comp = p1Data_comp['t'][:]
        tArray_comp = tArray_comp[0::skip_int]
        
        beam1_pos = p1Data_dict['pos'][:,:,2]
        beam2_pos = p2Data_dict['pos'][:,:,2]
        comp_beam1_pos = p1Data_comp['pos'][:,:,2]
        comp_beam1_pos = comp_beam1_pos[0::skip_int,:]
        
        
        tArray_slice = tArray[start_dt:max_dt]
        tArray_comp_slice = tArray_comp[start_dt:max_dt]
        beam1_pos_slice = beam1_pos[start_dt:max_dt]
        comp_beam1_pos_slice = comp_beam1_pos[start_dt:max_dt]
        
        pos_diff = np.abs(comp_beam1_pos_slice-beam1_pos_slice)
        rel_pos_diff = pos_diff/np.abs(comp_beam1_pos_slice)
        final_errors = rel_pos_diff[-1,:]
        
        avg_error = np.average(final_errors)
        
        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)
        avg_errors.append(avg_error)
        
        if plot == True:
            phase_snap(0,p1Data_dict,p2Data_dict,figNo=3)
            phase_snap(-1,p1Data_dict,p2Data_dict,figNo=4)

    label_order = sim_name[:-6]

    ##Convergence Plot w/ rhs
    #fig_con = plt.figure(DH.figureNo+4)
    #ax_con = fig_con.add_subplot(1, 1, 1)
    #ax_con.plot(rhs_evals[1:],avg_slope_diff,label=label_order)
    
    ##Order Plot w/ rhs
    fig_rhs = plt.figure(DH.figureNo+1)
    ax_rhs = fig_rhs.add_subplot(1, 1, 1)
    ax_rhs.plot(rhs_evals,avg_errors,label=label_order)
    
    ##Order Plot w/ dt
    fig_dt = plt.figure(DH.figureNo+2)
    ax_dt = fig_dt.add_subplot(1, 1, 1)
    ax_dt.plot(dts,avg_errors,label=label_order)

"""
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
"""

## Order plot finish
ax_rhs.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_rhs.set_xlabel('Number of RHS evaluations')
ax_rhs.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_rhs.set_ylabel('Avg. relative particle $\Delta z$')

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
ax_dt.set_ylabel('Avg. relative particle $\Delta z$')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.legend()
