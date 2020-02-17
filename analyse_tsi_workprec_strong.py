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

analyse = False
plot = True
snapPlot = False
compare_reference = True

analysis_times = [0,1,2,3,4,5,6,7,8,9,10]
compare_times = [10]

fit_start = analysis_times[0]
fit_stop = analysis_times[-1]

fig_type = 'versus'
data_root = "../data_tsi_strong/"
sims = {}

sims['tsi_TE10_a0.1_boris_SDC_M3K3_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200]
sims['tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200]
sims['tsi_TE10_a0.1_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200]

sims['tsi_TE10_a0.1_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]
sims['tsi_TE10_a0.1_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]
sims['tsi_TE10_a0.1_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]

comp_run = 'tsi_TE10_a0.1_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


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
plot_params['legend.fontsize'] = 12
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 14
plot_params['axes.titlesize'] = 14
plot_params['xtick.labelsize'] = 10
plot_params['ytick.labelsize'] = 10
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 5
plot_params['legend.loc'] = 'lower left'
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    if compare_reference == True:
        DH_comp = dataHandler2(**data_params)
        comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
        mData_comp = DH_comp.load_m(['phi','rho','E','dz'],sim_name=comp_sim_name,max_t=analysis_times[-1])

        analysis_ts = []
        for time in analysis_times:
            analysis_ts.append(np.int(time/(comp_sim.dt*DH_comp.samplePeriod)))
        analysis_ts = np.array(analysis_ts)

        tArray_comp = mData_comp['t']
        dz_comp = mData_comp['dz'][0] 
        
        rho_comp = mData_comp['rho'][analysis_ts,1,1,:-2]
        E = mData_comp['E'][analysis_ts,2,1,1,:-2]
        E2 = E*E
        UE =  np.sum(E2/2,axis=1)*dz_comp
        UE_log = np.log(UE)
        UE_comp = UE/UE[0]
        
        EL2 = np.sum(E*E,axis=1)
        EL2_comp = np.sqrt(EL2*dz_comp)
        
        
    for key, value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        linear_errors = []
        energy_errors = []
        E_errors = []
        
        
        filename = key[:-3] + "_wp_strong.h5"
        filenames.append(filename)
        try:
            file = h5.File(data_root+filename,'w')
        except OSError:
            file.close()
            file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
        
        for tsteps in value:
            DH = dataHandler2(**data_params)
            sim_name = key + str(tsteps)
            sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)
    
            ####################### Analysis and Visualisation ############################
            dt = sim.dt
            Nt = sim.tSteps
            
            NA = np.int(fit_start/(sim.dt*DH.samplePeriod))
            NB = np.int(fit_stop/(sim.dt*DH.samplePeriod))
            
            analysis_ts = []
            for time in analysis_times:
                analysis_ts.append(np.int(time/(sim.dt*DH.samplePeriod)))
            analysis_ts = np.array(analysis_ts)
                

            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz'],sim_name=sim_name,max_t=analysis_times[-1])
            tArray = mData_dict['t']

            phi_data = mData_dict['phi'][analysis_ts,1,1,:-1]
            E = mData_dict['E'][analysis_ts,2,1,1,:-1]
            E2 = E*E
            UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0] 
            UE_log = np.log(UE)
            UE_comp = UE/UE[0]
            
            EL2 = np.sum(E*E,axis=1) * mData_dict['dz'][0]
            EL2 = np.sqrt(EL2)  
            
            ## Growth rate phi plot setup
            max_phi_data = np.amax(np.abs(phi_data),axis=1)
            max_phi_data_log = np.log(max_phi_data)
    
            try:
                growth_fit = np.polyfit(tArray[NA:NB],max_phi_data_log[NA:NB],1)
                growth_line = growth_fit[0]*tArray[NA:NB] + growth_fit[1]
                
                error_linear = abs(real_slope - growth_fit[0])/real_slope
                linear_errors.append(error_linear)
            except:
                pass
            
            
            dts.append(sim.dt)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            
            if compare_reference == True:
                E_error = np.abs(EL2_comp-EL2)/np.abs(EL2_comp)
                E_errors.append(E_error)
            
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
        grp.create_dataset('E_errors',data=E_errors)
        grp.create_dataset('energy',data=UE)
        grp.create_dataset('energy_reference',data=UE_comp)
        grp.create_dataset('energy_errors',data=energy_errors)
        file.close()
    
    
    
if plot == True:
    DH = dataHandler2(**data_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_wp_strong.h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        energy_errors = file["fields/energy_errors"][:]
        E_errors = file["fields/E_errors"][:]
        E_errors = np.array(E_errors)

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
            label = "Boris" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris Synced" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC" + ", Nz=" + file.attrs["res"]
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]
        
        if compare_reference == True:
            ##Order Plot w/ rhs
            fig_nl_rhs = plt.figure(10)
            ax_nl_rhs = fig_nl_rhs.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_rhs.plot(rhs_evals,E_errors[:,time],marker="o",label=label_line)
                
            ##Order Plot w/ dt
            fig_nl_dt = plt.figure(11)
            ax_nl_dt = fig_nl_dt.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_dt.plot(dts,E_errors[:,time],marker="o",label=label_line)
            
        
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
            ax.set_ylabel(r'E L2 Error $\Delta \sqrt{\sum \frac{E_i^2}{2} \Delta x}$')
            
            ax.set_title('Convergence vs. Ref Solution')
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25',label='2nd Order')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75',label='4th Order')
            
            ax.legend(loc = 'best')
            compare_times = np.array(compare_times,dtype=np.int)
            fig_nl_rhs.savefig(data_root + 'tsi_strong_'+ fig_type +"_"+ str(compare_times) + 's_rhs.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
            fig_nl_dt.savefig(data_root + 'tsi_strong_' + fig_type +"_"+ str(compare_times) + 's_dt.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')