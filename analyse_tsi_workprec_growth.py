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
import h5py as h5

analyse = True
plot = True
snapPlot = True
compare_linear = True

start_time = 7.5
max_time = 15

data_root = "../data/"
sims = {}

sims['tsi_TE20_boris_SDC_M3K3_NZ100_NQ2000_NT'] = [20,40,80,160]
sims['tsi_TE20_boris_synced_NZ100_NQ2000_NT'] = [20,40,80,160]

#sims['tsi_TE1_boris_staggered_NZ10_NQ2000_NT'] = [1,2,4,8,16,32,64]
#sims['tsi_TE1_boris_staggered_NZ100_NQ2000_NT'] = [1,2,4,8,16,32,64]
#sims['tsi_TE1_boris_synced_NZ1000_NQ20000_NT'] = [1,2,4,8,16,32,64,128]
#sims['tsi_TE1_boris_synced_NZ10000_NQ20000_NT'] = [1,2,4,8,16,32,64,128]


comp_run = 'tsi_TE20_boris_SDC_M3K3_NZ100_NQ2000_NT160'


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
data_params = {}
data_params['dataRootFolder'] = data_root
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

DH_comp = dataHandler2(**data_params)
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

filenames = []
if analyse == True:
    for key, value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        avg_slopes = []
        avg_errors = []
        avg_errors_nonlinear = []
        
        filename = key + "_workprec_growth" + ".h5"
        filenames.append(filename)
        file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
        
        for tsteps in value:
            DH = dataHandler2(**data_params)
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
    
            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres'],sim_name=sim_name)
            
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
            
            if snapPlot == True:
                fig = plt.figure(DH.figureNo+3)
                gphi_ax = fig.add_subplot(1,1,1)
                line_gphi = gphi_ax.plot(tArray,max_phi_data_log,label=sim_name)
                #text_gphi = gphi_ax.text(.25,.05,transform=gphi_ax.transAxes,
                 #                        verticalalignment='bottom',fontsize=8)
                #g_ax.set_xlim([0.0, sim.dt*sim.tSteps])
                gphi_ax.set_xlabel('$t$')
                gphi_ax.set_ylabel(r'$\log(|\phi|_{max}$)')
                #g_ax.set_ylim([0,2])
                gphi_ax.set_title('Linear Instability Growth')
                #gphi_ax.legend()
            
            
        file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
        file.attrs["res"] = str(mData_dict['zres'][0])
        try:
            file.attrs["M"] = str(3)
            file.attrs["K"] = str(3)
        except KeyError:
            pass
        
        grp.create_dataset('dts',data=dts)
        grp.create_dataset('Nts',data=Nts)
        grp.create_dataset('rhs_evals',data=rhs_evals)
        grp.create_dataset('errors',data=avg_errors)
        grp.create_dataset('errors_nonlinear',data=avg_errors_nonlinear)
        file.close()
    
    
    
if plot == True:
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key + "_workprec_pos" + ".h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        avg_errors = file["fields/errors"][:]
        avg_errors_nonlinear = file["fields/errors_nonlinear"][:]
        
        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris Synced" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC" + ", Nz=" + file.attrs["res"]
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]

        ##Convergence Plot w/ rhs
        #fig_con = plt.figure(DH.figureNo+4)
        #ax_con = fig_con.add_subplot(1, 1, 1)
        #ax_con.plot(rhs_evals[1:],avg_slope_diff,label=label_order)
        
        ##Order Plot w/ rhs
        fig_rhs = plt.figure(10)
        ax_rhs = fig_rhs.add_subplot(1, 1, 1)
        #ax_rhs.plot(rhs_evals,avg_errors_nonlinear,label=label)
        if compare_linear == True:
            ax_rhs.plot(rhs_evals,avg_errors,label=label+" vs. Linear")
            
        ##Order Plot w/ dt
        fig_dt = plt.figure(11)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        #ax_dt.plot(dts,avg_errors_nonlinear,label=label)
        if compare_linear == True:
            ax_dt.plot(dts,avg_errors,label=label+" vs. Linear")
            
        file.close()
        
        
    ax_rhs.set_xscale('log')
    #ax_rhs.set_xlim(10**3,10**5)
    ax_rhs.set_xlabel('Number of RHS evaluations')
    ax_rhs.set_yscale('log')
    ax_rhs.set_ylim(10**(-6),10)
    ax_rhs.set_ylabel('Growth rate error')
    
    xRange = ax_rhs.get_xlim()
    yRange = ax_rhs.get_ylim()
    
    ax_rhs.plot(xRange,DH.orderLines(-2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_rhs.plot(xRange,DH.orderLines(-4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    ax_rhs.legend()
    
    ax_dt.set_xscale('log')
    #ax_dt.set_xlim(10**-3,10**-1)
    ax_dt.set_xlabel(r'$\Delta t$')
    ax_dt.set_yscale('log')
    ax_dt.set_ylim(10**(-4),10)
    ax_dt.set_ylabel('Growth rate error')
    
    xRange = ax_dt.get_xlim()
    yRange = ax_dt.get_ylim()
    
    ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    ax_dt.legend()