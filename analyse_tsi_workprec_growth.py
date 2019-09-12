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

start_time = 10
max_time = 17.5


sims = {}

sims['tsi_TE20_boris_SDC_M3K3_NZ10_PPC20_NT'] = [50,100,200,400]
sims['tsi_TE20_boris_SDC_M3K3_NZ100_PPC20_NT'] = [50,100,200,400]

sims['tsi_TE20_boris_synced_NZ10_ppc20_NT'] = [50,100,200,400]
sims['tsi_TE20_boris_synced_NZ100_ppc20_NT'] = [50,100,200]


comp_run = 'tsi_TE20_boris_synced_NZ100_ppc20_NT400'


################################ Linear analysis ##############################
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

############################### Setup #########################################

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
start_dt = np.int(start_time/(comp_sim.dt*DH_comp.samplePeriod))
max_steps = np.int(max_time/(comp_sim.dt*DH_comp.samplePeriod))+1

tArray_comp = mData_comp['t']

phi_comp_data = mData_comp['phi'][:,1,1,:-1]
max_phi_comp = np.amax(np.abs(phi_comp_data),axis=1)
max_phi_comp_log = np.log(max_phi_comp)
comp_growth_fit = np.polyfit(tArray_comp[start_dt:max_steps],
                             max_phi_comp_log[start_dt:max_steps],1)


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
        print(sim.runTime)

        ####################### Analysis and Visualisation ############################
        dt = sim.dt
        Nt = sim.tSteps
        
        start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
        max_steps = np.int(max_time/(sim.dt*DH.samplePeriod))+1
        NA = start_dt
        NB = max_steps
        
        pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
        
        p1Data_dict = pData_list[0]
        p2Data_dict = pData_list[1]

        mData_dict = DH.load_m(['phi','E','rho','PE_sum'],sim_name=sim_name)
        
        tArray = mData_dict['t']
        phi_data = mData_dict['phi'][:,1,1,:-1]
        PE_data = mData_dict['PE_sum']
        
        ## Growth rate phi plot setup
        max_phi_data = np.amax(np.abs(phi_data),axis=1)
        max_phi_data_log = np.log(max_phi_data)

        growth_fit = np.polyfit(tArray[NA:NB],max_phi_data_log[NA:NB],1)
        growth_line = growth_fit[0]*tArray[NA:NB] + growth_fit[1]
        
        error_linear = abs(real_slope - growth_fit[0])/real_slope
        error_nl = abs(comp_growth_fit[0] - growth_fit[0])/comp_growth_fit[0]
        
        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)
        avg_slopes.append(growth_fit[0])
        avg_errors.append(error_linear)
        avg_errors_nonlinear.append(error_nl)
        
        if plot == True:
            fig = plt.figure(DH.figureNo+3)
            gphi_ax = fig.add_subplot(1,1,1)
            line_gphi = gphi_ax.plot(tArray,max_phi_data_log,label=sim_name)
            text_gphi = gphi_ax.text(.25,.05,transform=gphi_ax.transAxes,
                                     verticalalignment='bottom',fontsize=8)
            #g_ax.set_xlim([0.0, sim.dt*sim.tSteps])
            gphi_ax.set_xlabel('$t$')
            gphi_ax.set_ylabel(r'$\log(|\phi|_{max}$)')
            #g_ax.set_ylim([0,2])
            gphi_ax.set_title('Linear Instability Growth')
            #gphi_ax.legend()
        
        
    label_order = sim_name[:-6]
    
    avg_slopes = np.array(avg_slopes)
    avg_slope_diff = np.abs(avg_slopes[0:-1]-avg_slopes[1:])/avg_slopes[0]
    ##Convergence Plot w/ rhs
    fig_con = plt.figure(DH.figureNo+4)
    ax_con = fig_con.add_subplot(1, 1, 1)
    ax_con.plot(rhs_evals[1:],avg_slope_diff,label=label_order)
    
    ##Order Plot w/ rhs
    fig_rhs = plt.figure(DH.figureNo+1)
    ax_rhs = fig_rhs.add_subplot(1, 1, 1)
    #ax_rhs.plot(rhs_evals,avg_errors,label=label_order+'vsLinear')
    ax_rhs.plot(rhs_evals,avg_errors_nonlinear,label=label_order)
    
    ##Order Plot w/ dt
    fig_dt = plt.figure(DH.figureNo+2)
    ax_dt = fig_dt.add_subplot(1, 1, 1)
    #ax_dt.plot(dts,avg_errors,label=label_order+'vsLinear')
    ax_dt.plot(dts,avg_errors_nonlinear,label=label_order)

## Convergence plot finish
ax_con.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_con.set_xlabel('Number of RHS evaluations')
ax_con.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_con.set_ylabel('Slope difference')

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
ax_rhs.set_ylabel('Relative slope error')

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
ax_dt.set_ylabel('Relative slope error')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.legend()