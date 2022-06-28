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
from caseFile_landau1D import *

analyse = True
fieldPlot = False
snapPlot = False
resPlot = True
compare_reference = True
plot = True


analysis_times = [0,1,2,3,4,5,6,7,8,9,10]
compare_times = [1]

fit_start = 10
fit_stop = 16

snaps = [0,100]

fig_type = 'full'
data_root = "../data/"
sims = {}

#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
##
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]

#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#
#sims['tsi_TE10_a0.0001_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]

#sims['tsi_TE50_a0.0001_boris_SDC_M3K2_NZ100_NQ20000_NT'] = [500]

comp_run = 'tsi_TE10_a0.0001_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


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
plot_params['legend.fontsize'] = 16
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['axes.linewidth'] = 1.5
plot_params['ytick.major.width'] = 2
plot_params['ytick.minor.width'] = 2
plot_params['xtick.major.width'] = 2
plot_params['xtick.minor.width'] = 2
plot_params['legend.loc'] = 'best'
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    if compare_reference == True:
        DH_comp = dataHandler2(**data_params)
        comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
        mData_comp = DH_comp.load_m(['phi','rho','E','dz'],sim_name=comp_sim_name,max_t=analysis_times[-1])

        analysis_ts = []
        for time in analysis_times:
            analysis_ts.append(int(time/(comp_sim.dt*DH_comp.samplePeriod)))
        analysis_ts = np.array(analysis_ts)

        tArray_comp = mData_comp['t']
        dz_comp = mData_comp['dz'][0] 
        
        rho_comp = mData_comp['rho'][analysis_ts,1,1,:-2]
        E = mData_comp['E'][:,2,1,1,:-2]
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
        
        
        filename = key[:-3] + "_wp_weak.h5"
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
            
            NA = int(fit_start/(sim.dt*DH.samplePeriod))
            NB = int(fit_stop/(sim.dt*DH.samplePeriod))
            
            analysis_ts = []
            for time in analysis_times:
                analysis_ts.append(int(time/(sim.dt*DH.samplePeriod)))
            analysis_ts = np.array(analysis_ts)
                

            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz','Rx','Rv'],sim_name=sim_name,max_t=analysis_times[-1])
            tArray = mData_dict['t']

            phi_data = mData_dict['phi'][analysis_ts,1,1,:-1]
            E = mData_dict['E'][:,2,1,1,:-1]
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
                c1 = EL2[NA]*0.008
                E_fit = np.around(np.polyfit(tArray[NA:NB],np.log(EL2[NA:NB]),1),decimals=4)
                E_line = c1*np.exp(tArray[NA:NB]*E_fit[0])
                
                real_slope = np.around(real_slope,decimals=4)
                lit_line = c1*np.exp(tArray[NA:NB]*real_slope)
                
                error_linear = abs(real_slope - E_fit[0])/real_slope
                linear_errors.append(error_linear)
            except:
                pass
            
            
            dts.append(sim.dt)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            
            if compare_reference == True:
                E_error = np.abs(EL2_comp-EL2[analysis_ts])/np.abs(EL2_comp)
                E_errors.append(E_error)
            
            if fieldPlot == True:
                print("Drawing field plot...")
                fig_el2 = plt.figure(DH.figureNo+5,dpi=150)
                el2_ax = fig_el2.add_subplot(1,1,1)
                el2_ax.plot(tArray[1:],EL2[1:],'blue',label="$E$")
                el2_ax.plot(tArray[NA:NB],E_line,'red',label="Fitted $\gamma$")
                el2_ax.plot(tArray[NA:NB],lit_line,'orange',label="Literature $\gamma$")
                E_text1 = el2_ax.text(.1,0.02,'',transform=el2_ax.transAxes,verticalalignment='bottom',fontsize=16)
                text1 = (r'$\gamma_{fit}$ = ' + str(E_fit[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(real_slope))
                E_text1.set_text(text1)
                el2_ax.set_xlabel('$t$')
                el2_ax.set_ylabel(r'$||E||_{L2}$')
                el2_ax.set_yscale('log')
                el2_ax.set_xlim(0,50)
#                el2_ax.set_ylim(10**(-4),3)
                el2_ax.legend()
                fig_el2.savefig(data_root + 'tsi_weak_growth.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
            
                
            if snapPlot == True:
                print("Loading particle data...")
                pData_list = DH.load_p(['pos','vel'],species=['beam1','beam2'],sim_name=sim_name)
                
                p1Data_dict = pData_list[0]
                p2Data_dict = pData_list[1]
    
                
                p1_data = p1Data_dict['pos'][:,:,2]
                p2_data = p2Data_dict['pos'][:,:,2]
                
                v1_data = p1Data_dict['vel'][:,:,2] 
                v2_data = p2Data_dict['vel'][:,:,2] 
                
                no = 0
                for snap in snaps:
                    no +=1
                    print("Drawing snap no. {0}...".format(no))
                    fig_snap = plt.figure(DH.figureNo+10+no,dpi=150)
                    p_ax = fig_snap.add_subplot(1,1,1)
                    line_p1 = p_ax.plot(p1_data[snap,:],v1_data[snap,:],'bo',ms=2,c=(0.2,0.2,0.75,1),label='Beam 1, v=1')[0]
                    line_p2 = p_ax.plot(p2_data[snap,:],v2_data[snap,:],'ro',ms=2,c=(0.75,0.2,0.2,1),label='Beam 2, v=-1')[0]
                    p_ax.set_xlim([0.0, sim.zlimits[1]])
                    p_ax.set_xlabel('$z$')
                    p_ax.set_ylabel('$v_z$')
                    p_ax.set_ylim([-3,3])
                    fig_snap.savefig(data_root + 'tsi_weak_snap_ts{0}.svg'.format(snap), dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
                    
                    
            if resPlot == True:
                fig_R = plt.figure(DH.figureNo+6,dpi=150)
                R_ax = fig_R.add_subplot(1,1,1)
                R_ax.plot(tArray[1:],mData_dict['Rx'][1:,-1,-1],label='Beam 1, Rx')
                R_ax.plot(tArray[1:],mData_dict['Rv'][1:,-1,-1],label='Beam 1, Rv')
                R_ax.set_xlabel('$t$')
                R_ax.set_ylabel(r'$||R||_{L2}$')
                R_ax.set_yscale('log')
                R_ax.legend()
                fig_R.savefig(data_root + 'tsi_weak_residual.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
                  
            
            
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
            filename = key[:-3] + "_wp_weak.h5"
            filenames.append(filename)
            
    plt.rcParams.update(plot_params)
    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        energy_errors = file["fields/energy_errors"][:]
        E_errors = file["fields/E_errors"][:]
        E_errors = np.array(E_errors)
        
        K = filename[filename.find('K')+1]
        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
            label = "Boris"
            c = '#0080FF'
        elif file.attrs["integrator"] == "boris_synced":
            c = '#0080FF'
            label = "Boris"
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC"
            if K == '1':
                c = '#00d65d'
                label += ", M=" + file.attrs["M"] + ", K=" + K
            elif K == '2':
                c = '#F9004B'
                label += ", M=" + file.attrs["M"] + ", K=" + K
            elif K == '3':
                c = '#FFD738'
                label += ", M=" + file.attrs["M"] + ", K=" + K
                
        if compare_reference == True:
            ##Order Plot w/ rhs
            fig_nl_rhs = plt.figure(10)
            ax_nl_rhs = fig_nl_rhs.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_rhs.plot(rhs_evals,E_errors[:,time],marker="o",color=c,label=label_line)
                
            ##Order Plot w/ dt
            fig_nl_dt = plt.figure(11)
            ax_nl_dt = fig_nl_dt.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_dt.plot(dts,E_errors[:,time],marker="o",color=c,label=label_line)
                
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
            ax.set_ylim(10**(-6),1)
#            ax.set_ylabel(r'E L2 Error $\Delta \sqrt{\sum \frac{E_i^2}{2} \Delta z}$')
            ax.set_ylabel(r'Rel. $||E||_{L2}$ Error')
            
#            ax.set_title('Weak two-stream instability, convergence vs. ref solution')
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75')
            
        compare_times = np.array(compare_times,dtype=int)
        fig_nl_rhs.savefig(data_root + 'tsi_weak_'+ fig_type +"_"+ str(compare_times) + 's_rhs.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.0,bbox_inches = 'tight')
        fig_nl_dt.savefig(data_root + 'tsi_weak_' + fig_type +"_"+ str(compare_times) + 's_dt.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.0,bbox_inches = 'tight')
