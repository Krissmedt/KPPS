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
from collections import OrderedDict


analyse = False
plot = True
snapPlot = False
compare_reference = True

start_time = 0
max_time = 4

fig_type = 'versus'
data_root = "../data_tsi_old/"
sims = {}


sims['tsi_TE10_boris_staggered_NZ10_NQ20000_NT'] = []
sims['tsi_TE10_boris_staggered_NZ100_NQ20000_NT'] = []
sims['tsi_TE10_boris_staggered_NZ1000_NQ20000_NT'] = []


sims['tsi_TE10_boris_SDC_M3K3_NZ10_NQ20000_NT'] = []
sims['tsi_TE10_boris_SDC_M3K3_NZ100_NQ20000_NT'] = []
sims['tsi_TE10_boris_SDC_M3K3_NZ1000_NQ20000_NT'] = []

comp_run = 'tsi_TE10_boris_SDC_M5K5_NZ10000_NQ200000_NT1000'


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
plot_params['legend.fontsize'] = 14
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['legend.loc'] = 'lower left'
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    if compare_reference == True:
        DH_comp = dataHandler2(**data_params)
        comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
        mData_comp = DH_comp.load_m(['phi','E','dz'],sim_name=comp_sim_name)
        start_dt = np.int(start_time/(comp_sim.dt*DH_comp.samplePeriod))
        max_steps = np.int(max_time/(comp_sim.dt*DH_comp.samplePeriod))
        
        tArray_comp = mData_comp['t']
        
        phi_comp_data = mData_comp['phi'][:,1,1,:-1]
        max_phi_comp = np.amax(np.abs(phi_comp_data),axis=1)
        max_phi_comp_log = np.log(max_phi_comp)
        comp_growth_fit = np.polyfit(tArray_comp[start_dt:max_steps],
                                     max_phi_comp_log[start_dt:max_steps],1)

        dz_comp = mData_comp['dz'][0] 
        E_comp = mData_comp['E'][max_steps,2,1,1,:-1]
        UE_comp = np.sum(E_comp*E_comp/2)*dz_comp
    
    for key, value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        avg_slopes = []
        avg_errors = []
        avg_errors_nonlinear = []
        energy_errors = []
        
        filename = key[:-3] + "_workprec_growth_"  + str(max_time)+ "s"  ".h5"
        filenames.append(filename)
        file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
        
        for tsteps in value:
            DH = dataHandler2(**data_params)
            sim_name = key + str(tsteps)
            sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)
    
            ####################### Analysis and Visualisation ############################
            dt = sim.dt
            Nt = sim.tSteps
            
            start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
            max_steps = np.int(max_time/(sim.dt*DH.samplePeriod))
            NA = start_dt
            NB = max_steps

            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz'],sim_name=sim_name)
            tArray = mData_dict['t']

            E = mData_dict['E'][NB,2,1,1,:-1]            
            UE = np.sum(E*E/2)*mData_dict['dz'][0]
            print(UE)
            print(UE_comp)
            tArray = mData_dict['t']
            phi_data = mData_dict['phi'][:,1,1,:-1]
            PE_data = mData_dict['PE_sum']
            
            ## Growth rate phi plot setup
            max_phi_data = np.amax(np.abs(phi_data),axis=1)
            max_phi_data_log = np.log(max_phi_data)
    
            growth_fit = np.polyfit(tArray[NA:NB],max_phi_data_log[NA:NB],1)
            growth_line = growth_fit[0]*tArray[NA:NB] + growth_fit[1]
            
            error_linear = abs(real_slope - growth_fit[0])/real_slope
            
            
            dts.append(sim.dt)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            avg_slopes.append(growth_fit[0])
            avg_errors.append(error_linear)
            
            if compare_reference == True:
                energy_error = abs(UE_comp-UE)/UE_comp
                energy_errors.append(energy_error)
            
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
                gphi_ax.set_title('Electric Potential Growth')
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
        grp.create_dataset('energy',data=UE)
        grp.create_dataset('energy_reference',data=UE_comp)
        grp.create_dataset('energy_errors',data=energy_errors)
        file.close()
    
    
    
if plot == True:
    DH = dataHandler2(**data_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_workprec_growth_" + str(max_time)+ "s" + ".h5"
            filenames.append(filename)
            

    plt.rcParams.update(plot_params)
    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        avg_errors = file["fields/errors"][:]
        energy_errors = file["fields/energy_errors"][:]

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
            label = "Boris"
            c = '#0080FF'
        elif file.attrs["integrator"] == "boris_synced":
            c = '#0080FF'
            label = "Boris"
        elif file.attrs["integrator"] == "boris_SDC":
            c = '#F9004B'
            label = "Boris-SDC"
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]
        
        if compare_reference == True:
            ##Order Plot w/ rhs
            fig_nl_rhs = plt.figure(10)
            ax_nl_rhs = fig_nl_rhs.add_subplot(1, 1, 1)
            ax_nl_rhs.plot(rhs_evals,energy_errors,color=c,label=label)
                
            ##Order Plot w/ dt
            fig_nl_dt = plt.figure(11)
            ax_nl_dt = fig_nl_dt.add_subplot(1, 1, 1)
            ax_nl_dt.plot(dts,energy_errors,color=c,label=label)
            
    handles, labels = fig_nl_rhs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax_nl_rhs.legend(by_label.values(), by_label.keys(),loc='best')
    
    handles, labels = fig_nl_dt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax_nl_dt.legend(by_label.values(), by_label.keys(),loc='best')
    
    if compare_reference == True:
        axnl_list = []
        axnl_list.append(ax_nl_rhs)
        axnl_list.append(ax_nl_dt)
        
        i = 0
        for ax in axnl_list:
            i +=1
            if i == 1:
                orderSlope = -1
                ax.set_xlabel('Number of RHS evaluations')
            else:
                ax.set_xlabel(r'$\Delta t$')
                orderSlope = 1
            
            ax.set_xscale('log')
            #ax_rhs.set_xlim(10**3,10**5)
            ax.set_yscale('log')
            ax.set_ylim(10**(-5),10)
            ax.set_ylabel(r'Energy Error $\Delta (\sum \frac{E_i^2}{2} \Delta x )$')
            
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75')
            
            ax.set_title('Two-Stream Instability Work Precision at {0}s'.format(str(max_time)))
            
            
            
#            ax.legend(loc = 'best')
            fig_nl_rhs.savefig(data_root + 'tsi_growth_'+ fig_type +"_"+ str(max_time) + 's_rhs.svg', dpi=300, facecolor='w', edgecolor='w',orientation='portrait')
            fig_nl_dt.savefig(data_root + 'tsi_growth_' + fig_type +"_"+ str(max_time) + 's_dt.svg', dpi=300, facecolor='w', edgecolor='w',orientation='portrait')