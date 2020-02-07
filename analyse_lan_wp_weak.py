from caseFile_landau1D import *
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
import sys
import traceback

def plot_density_1d(species_list,fields,controller='',**kwargs):
    plot_res = controller.plot_res
    v_off = controller.v_off
    
    pos_data_list = [species_list[0].pos[:,2]]
    vel_data_list = [species_list[0].vel[:,2]]
    fields.grid_x,fields.grid_v,fields.f,fields.pn,fields.vel_dist = calc_density_mesh(pos_data_list,vel_data_list,plot_res,plot_res,v_off,L)
    
    return species_list, fields

def find_peaks(peak_intervals,EL2,dt,samplePeriod):
    peaks = []
    for interval in peak_intervals:
        NA = np.int(interval[0]/(dt*samplePeriod))
        NB = np.int(interval[1]/(dt*samplePeriod))
        mi = NA + np.argmax(EL2[NA:NB])
        peaks.append(mi)
        
    return peaks

analyse = False
plot = False
snapPlot = False
compare_reference = False
compare_linear = False

peak_intervals = [[2,4],[4,5]]
#peak_intervals = [[2,4],[4,6],[6,8]]
#peak_intervals = [[0,2],[2,4],[4,6]]

fit1_start = peak_intervals[0][0]
fit1_stop = peak_intervals[-1][-1]

analysis_times = [0,2,4,6,8,10]
compare_times = [5]

fig_type = 'versus3'
data_root = "../data_landau_weak/"
sims = {}


sims['lan_TE10_a0.05_boris_SDC_M3K1_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]

#sims['lan_TE10_a0.05_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]
#sims['lan_TE10_a0.05_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]
sims['lan_TE10_a0.05_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,500,1000]

#sims['lan_TE10_a0.05_boris_SDC_M3K3_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,500]
#sims['lan_TE10_a0.05_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,500]
sims['lan_TE10_a0.05_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,500]


comp_run = 'lan_TE20_a0.05_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


################################ Linear analysis ##############################
nq = 20000
k = 0.5
v_th = 1
L = 4*np.pi
a = 1
q = L/nq

omega_p = np.sqrt(q*nq*a*1/L)
omega = np.sqrt(omega_p**2  +3*k**2*v_th**2)
#omega = 1.4436
vp = omega/k
vp = 2.84
#omega2 = 2.8312 * k
#omegap2 = np.sqrt(omega2**2 - 3*k**2*v_th**2)
#
#df_vp = (2*np.pi)**(-1/2)*(1/v_th) * np.exp(-vp**2/(2*v_th**2)) * -vp/v_th**2
df_vp = -vp/np.sqrt(2*np.pi) * np.exp(-vp**2/2)

gamma = (np.pi*omega_p**3)/(2*k**2) * df_vp

gamma_lit1 = -0.1533


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
plot_params['legend.loc'] = 'upper right'
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
        #EL2_comp = EL2_comp/EL2_comp[0]
        
        try:
            comp_peaks = find_peaks(peak_intervals,EL2_comp,comp_sim.dt,DH_comp.samplePeriod)
            comp_fit = np.polyfit(tArray_comp[comp_peaks],
                                 EL2_comp[comp_peaks],1)
        except:
            pass
        

    sim_no = 0
    for key, value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        avg_slopes = []
        avg_errors = []
        avg_errors_nonlinear = []
        E_errors = []
        
        filename = key[:-3] + "_wp_weak.h5"
        filenames.append(filename)
        try:
            file = h5.File(data_root+filename,'w')
        except OSError:
            file.close()
            file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
        try:
            for tsteps in value:
                sim_no += 1
                DH = dataHandler2(**data_params)
                sim_name = key + str(tsteps)
                sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)
        
                ####################### Analysis and Visualisation ############################
                dt = sim.dt
                Nt = sim.tSteps
                
                NA = np.int(fit1_start/(sim.dt*DH.samplePeriod))
                NB = np.int(fit1_stop/(sim.dt*DH.samplePeriod))
    
                analysis_ts = []
                for time in analysis_times:
                    analysis_ts.append(np.int(time/(sim.dt*DH.samplePeriod)))
                analysis_ts = np.array(analysis_ts)

                mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz'],sim_name=sim_name,max_t=analysis_times[-1])
                tArray = mData_dict['t']
                
                NQ = key[key.find('NQ')+2:key.find('_NT')] 
                rho = mData_dict['rho'][analysis_ts,1,1,:-2]
                E = mData_dict['E'][analysis_ts,2,1,1,:-2]
                E2 = E*E
                UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0] 
                UE_log = np.log(UE)
                UE_comp = UE/UE[0]
                
                EL2 = np.sum(E*E,axis=1)
                EL2 = np.sqrt(EL2*mData_dict['dz'][0])  


                try:
                    peaks = find_peaks(peak_intervals,EL2,sim.dt,DH.samplePeriod)
                    c1 = EL2[peaks[0]]*1.05
                    
                    E_fit = np.polyfit(tArray[peaks],np.log(EL2[peaks]),1)
                    E_fit = np.around(E_fit,decimals=6)
                    E_fit_line = c1*np.exp(E_fit[0]*tArray[NA:NB])
                    
                    gamma_line = c1*np.exp(gamma*tArray[NA:NB])
                    lit_line = c1*np.exp(gamma_lit1*tArray[NA:NB])
                    
                    error_linear = abs(gamma_lit1 - E_fit[0])/abs(gamma_lit1)
                    avg_slopes.append(E_fit[0])
                    avg_errors.append(error_linear)
                except:
                    pass
        
                dts.append(sim.dt)
                Nts.append(sim.tSteps)
                rhs_evals.append(sim.rhs_eval)

                
                if compare_reference == True:
#                    skip = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
#                    skip_int = np.int(skip)
                    E_error = np.abs(EL2_comp-EL2)/np.abs(EL2_comp)
                    E_errors.append(E_error)

                
                if snapPlot == True:
                    fig = plt.figure(DH.figureNo+sim_no)
                    E_ax = fig.add_subplot(1,1,1)
                    E_ax.plot(tArray,EL2,'blue',label="$E$-field")
                    
                    E_ax.scatter(tArray[peaks],EL2[peaks])
                    E_ax.plot(tArray[NA:NB],E_fit_line,'orange',label="Fitted $\gamma$")
                    #E_ax.plot(tArray[NA:NB],gamma_line,'red',label="$Analyt$")
                    E_ax.plot(tArray[NA:NB],lit_line,'green',label="Literature $\gamma$")
                    
                    E_text1 = E_ax.text(.05,0,'',transform=E_ax.transAxes,verticalalignment='bottom',fontsize=14)
                    text1 = (r'$\gamma_{fit}$ = ' + str(E_fit[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(gamma_lit1))
                    E_text1.set_text(text1)
                    
                    
                    E_ax.set_xlabel('$t$')
                    E_ax.set_ylabel(r'$||E||_{L2}$')
                    E_ax.set_yscale('log')
                    #E_ax.set_ylim([10**-7,10**-1])
                    E_ax.set_title('Linear Landau damping, a=0.05, NQ=' + NQ + ', NZ=' + str(mData_dict['zres'][0]) 
                                    + ', NT=' + str(sim.tSteps)) 
                    E_ax.legend()
                    fig.savefig(data_root + sim_name + '_EL2.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                    
            file.attrs["reference"] = comp_run
            file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
            file.attrs["res"] = str(mData_dict['zres'][0])
            try:
                file.attrs["M"] = str(sim.analysisSettings['M'])
                file.attrs["K"] = str(sim.analysisSettings['K'])
            except KeyError:
                pass
            
            grp.create_dataset('dts',data=dts)
            grp.create_dataset('Nts',data=Nts)
            grp.create_dataset('rhs_evals',data=rhs_evals)
            grp.create_dataset('errors',data=avg_errors)
            grp.create_dataset('energy',data=UE)
            grp.create_dataset('energy_reference',data=UE_comp)
            grp.create_dataset('E_errors',data=np.array(E_errors))
            file.close()
        except Exception as err:
            file.close()
            traceback.print_tb(err.__traceback__)
            print(err)
    
    
if plot == True:
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_wp"  + "_weak.h5"
            filenames.append(filename)
            


    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        avg_errors = file["fields/errors"][:]
        E_errors = file["fields/E_errors"][:]
        E_errors = np.array(E_errors)

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
            label = "Boris" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris Synced" + ", Nz=" + file.attrs["res"]
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC" + ", Nz=" + file.attrs["res"]
            try:
                label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]
            except:
                label += ", M=3, K=1"
                
        
        if compare_reference == True:
            ##Order Plot w/ rhs
            fig_nl_rhs = plt.figure(10)
            ax_nl_rhs = fig_nl_rhs.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_rhs.plot(rhs_evals,E_errors[:,time],label=label_line)
                
            ##Order Plot w/ dt
            fig_nl_dt = plt.figure(11)
            ax_nl_dt = fig_nl_dt.add_subplot(1, 1, 1)
            for time in compare_times:
                label_line = label + ', ' + str(analysis_times[time]) + 's'
                ax_nl_dt.plot(dts,E_errors[:,time],label=label_line)
            
        if compare_linear == True:
            ##Order Plot w/ rhs
            fig_rhs = plt.figure(12)
            ax_rhs = fig_rhs.add_subplot(1, 1, 1)
            ax_rhs.plot(rhs_evals,avg_errors,label=label)
                
            ##Order Plot w/ dt
            fig_dt = plt.figure(13)
            ax_dt = fig_dt.add_subplot(1, 1, 1)
            ax_dt.plot(dts,avg_errors,label=label)
            
        file.close()
        

    
    if compare_linear == True:
        ax_list = []  
        ax_list.append(ax_rhs)
        ax_list.append(ax_dt)
        i = 0
        for ax in ax_list:
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
            ax.set_ylim(10**(-4),10)
            ax.set_ylabel('damping rate error')
            
            ax.set_title('Convergence vs. Linear Theory')
            
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(1*orderSlope,xRange,yRange),
                        ls='dashdot',c='0.5',label='1st Order')
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25',label='2nd Order')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75',label='4th Order')
            
        ax.legend()
        
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
            ax.set_ylabel(r'E L2 Error $\Delta (\sum \frac{E_i^2}{2} \Delta x)$')
            
            ax.set_title('Convergence vs. Ref Solution')
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(1*orderSlope,xRange,yRange),
                        ls='dashdot',c='0.5',label='1st Order')
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25',label='2nd Order')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75',label='4th Order')
            
            ax.legend(loc = 'best')
            
            compare_times = np.array(compare_times,dtype=np.int)
            fig_nl_rhs.savefig(data_root + 'landau_weak_'+ fig_type +"_"+ str(compare_times) + 's_rhs.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
            fig_nl_dt.savefig(data_root + 'landau_weak_' + fig_type +"_"+ str(compare_times) + 's_dt.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')