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
from collections import OrderedDict

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

analyse = True
fieldPlot = False
snapPlot = False
compare_reference = False
plot = True


peak_intervals = [[0,2],[2,4],[4,6]]
peak_intervals2 = [[22.5,25],[25,27.5],[27.5,30]]
#peak_intervals = [[2,4],[4,6],[6,8]]
#peak_intervals = [[0,2],[2,4],[4,6]]

fit1_start = peak_intervals[0][0]
fit1_stop = peak_intervals[-1][-1]
fit2_start = peak_intervals2[0][0]
fit2_stop = peak_intervals2[-1][-1]

analysis_times = [0,1,2,3,4,5,6,7,8,9,10]
compare_times = [1]

snaps = [0,60,120,180,240,300]

fig_type = 'versus'
data_root = "../data_landau_strong/"
sims = {}

#sims['lan_TE10_a0.5_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#
#sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#
#sims['lan_TE10_a0.5_boris_SDC_M3K3_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]


#sims['lan_TE10_a0.5_boris_synced_NZ1000_NQ2000000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
#sims['lan_TE10_a0.5_boris_SDC_M3K3_NZ1000_NQ2000000_NT'] = [10,20,40,50,80,100,200,400,500]

#sims['lan_TE30_a0.5_boris_SDC_M3K3_NZ100_NQ200000_NT300] = [300]

comp_run = 'lan_TE10_a0.5_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


################################ Linear analysis ##############################
nq = 163000
k = 0.5
v_th = 1
L = 4*np.pi
a = 1
q = L/nq

omega_p = np.sqrt(q*nq*a*1/L)
omega = np.sqrt(omega_p**2  +3*k**2*v_th**2)
#omega = 1.4436
vp = omega/k
#omega2 = 2.8312 * k
#omegap2 = np.sqrt(omega2**2 - 3*k**2*v_th**2)

df_vp = (2*np.pi)**(-1/2)*(1/v_th) * np.exp(-vp**2/(2*v_th**2)) * -vp/v_th**2

gamma = (np.pi*omega_p**3)/(2*k**2) * df_vp

gamma_lit1 = -0.2922
gamma_lit2 = 0.08612


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
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    if compare_reference == True:
        DH_comp = dataHandler2(**data_params)
        comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
        mData_comp = DH_comp.load_m(['phi','rho','E','dz','q','vel_dist','grid_x','grid_v','f'],sim_name=comp_sim_name,max_t=analysis_times[-1])

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
        gamma1_errors = []
        gamma2_errors = []
        E_errors = []
        
        filename = key[:-3] + "_wp_strong.h5"
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
                NC = np.int(fit2_start/(sim.dt*DH.samplePeriod))
                ND = np.int(fit2_stop/(sim.dt*DH.samplePeriod))
                
                analysis_ts = []
                for time in analysis_times:
                    analysis_ts.append(np.int(time/(sim.dt*DH.samplePeriod)))
                analysis_ts = np.array(analysis_ts)

                mData_dict = DH.load_m(['phi','E','rho','zres','dz','q','grid_x','grid_v','f'],sim_name=sim_name,max_t=analysis_times[-1])
                tArray = mData_dict['t']
    
                E = mData_dict['E'][:,2,1,1,:-1]
                E2 = E*E
                UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0] 
                UE_log = np.log(UE)
                UE_comp = UE/UE[0]
                
                EL2 = np.sum(E*E,axis=1) * mData_dict['dz'][0]
                EL2 = np.sqrt(EL2)  

                try:
                    peaks = find_peaks(peak_intervals,EL2,sim.dt,DH.samplePeriod)
                    c1 = EL2[peaks[0]]*1.05
                    E_fit = np.polyfit(tArray[peaks],np.log(EL2[peaks]),1)
                    E_fit = np.around(E_fit,decimals=5)
                    E_fit_line = c1*np.exp(E_fit[0]*tArray[NA:NB])
                    gamma_line = c1*np.exp(gamma*tArray[NA:NB])
                    lit_line = c1*np.exp(gamma_lit1*tArray[NA:NB])
                    error_gamma1 = abs(gamma_lit1 - E_fit[0])/abs(gamma_lit1)
                    gamma1_errors.append(error_gamma1)
                except Exception:
                    pass
                
                try:
                    peaks2 = find_peaks(peak_intervals2,EL2,sim.dt,DH.samplePeriod)
                    c2 = EL2[peaks2[0]]*0.17
                    E_fit2 = np.polyfit(tArray[peaks2],np.log(EL2[peaks2]),1)
                    E_fit2 = np.around(E_fit2,decimals=5)
                    E_fit_line2 = c2*np.exp(E_fit2[0]*tArray[NC:ND])
                    lit_line2 = c2*0.7*np.exp(gamma_lit2*tArray[NC:ND])
                    error_gamma2 = abs(gamma_lit2 - E_fit2[0])/abs(gamma_lit2)
                    gamma2_errors.append(error_gamma2)
                except Exception:
                    pass
                
                
                dts.append(sim.dt)
                Nts.append(sim.tSteps)
                rhs_evals.append(sim.rhs_eval)

                if compare_reference == True:
#                    skip = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
#                    skip_int = np.int(skip)
                    E_error = np.abs(EL2_comp-EL2[analysis_ts])/np.abs(EL2_comp)
                    E_errors.append(E_error)

                    
                if fieldPlot == True:
                    plt.rcParams.update(plot_params)
                    print("Drawing field plot...")
                    fig = plt.figure(DH.figureNo+sim_no)
                    E_ax = fig.add_subplot(1,1,1)
                    E_ax.plot(tArray,EL2,'blue',label="$E$-field")
                    
                    E_ax.scatter(tArray[peaks],EL2[peaks])
                    E_ax.plot(tArray[NA:NB],E_fit_line,'orange',label="Fitted $\gamma$")
                    E_ax.plot(tArray[NA:NB],lit_line,'green',label="Literature $\gamma$")
                    
                    E_text1 = E_ax.text(.02,0.02,'',transform=E_ax.transAxes,verticalalignment='bottom',fontsize=16)
                    text1 = (r'$\gamma_{fit}$ = ' + str(E_fit[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(gamma_lit1))
                    E_text1.set_text(text1)
                    
                    E_ax.set_yscale('log')
                    E_ax.set_xlabel(r'$\Delta t$')
                    E_ax.set_ylabel(r'$||E||_{L2}$')
                    E_ax.legend()
                    
                    try:
                        E_ax.scatter(tArray[peaks2],EL2[peaks2])
                        E_ax.plot(tArray[NC:ND],E_fit_line2,'orange')
                        E_ax.plot(tArray[NC:ND],lit_line2,'green')
                        
                        E_text2 = E_ax.text(.6,0.02,'',transform=E_ax.transAxes,verticalalignment='bottom',fontsize=16)
                        text2 = (r'$\gamma_{fit}$ = ' + str(E_fit2[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(gamma_lit2))
                        E_text2.set_text(text2)
                    except Exception:
                        pass
                    fig.savefig(data_root + 'landau_strong_field.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                
                    
                if snapPlot == True:
                    print("Loading density data...")
                    gridx = mData_dict['grid_x'][0,:,:]
                    gridv = mData_dict['grid_v'][0,:,:]
                    f = mData_dict['f']
                    
                    no = 0
                    for snap in snaps:
                        no +=1
                        print("Drawing snap no. {0}...".format(no))
                        fig_f = plt.figure(DH.figureNo+5+no,dpi=150)
                        f_ax = fig_f.add_subplot(1,1,1)
                        cont = f_ax.contourf(gridx,gridv,f[snap,:,:],cmap='inferno')
                        cont.set_clim(0,np.max(f))
                        cbar = plt.colorbar(cont,ax=f_ax)
                        f_ax.set_xlim([0.0, L])
                        f_ax.set_xlabel('$z$')
                        f_ax.set_ylabel('$v_z$')
                        f_ax.set_ylim([-4,4])
    #                    f_ax.set_title('Landau density distribution, Nt=' + str(Nt) +', Nz=' + str(res+1))
                        fig_f.savefig(data_root + 'landau_strong_snap_ts{}.pdf'.format(snap), dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                        
                    
                    
            
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
            grp.create_dataset('gamma1_errors',data=gamma1_errors)
            grp.create_dataset('gamma2_errors',data=gamma2_errors)
            grp.create_dataset('energy',data=UE)
            grp.create_dataset('energy_reference',data=UE_comp)
            grp.create_dataset('E_errors',data=E_errors)
            file.close()
        except Exception as err:
            file.close()
            traceback.print_tb(err.__traceback__)
            print(err)
    
    
if plot == True:
    DH = dataHandler2(**data_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_wp"  + "_strong.h5"
            filenames.append(filename)
            

    plt.rcParams.update(plot_params)
    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        gamma1_errors = file["fields/gamma1_errors"][:]
        gamma2_errors = file["fields/gamma2_errors"][:]
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
                c = '#FFD738'
                label += ", M=" + file.attrs["M"] + ", K=" + K
            elif K == '3':
                c = '#F9004B'
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
            ax.set_ylabel(r'Rel. $||E||_{L2}$ Error')
            
#            ax.set_title('Non-linear Landau damping, convergence vs. ref solution')
            xRange = ax.get_xlim()
            yRange = ax.get_ylim()
            
            ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                        ls='dotted',c='0.25')
            ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                        ls='dashed',c='0.75')
            
            compare_times = np.array(compare_times,dtype=np.int)
            fig_nl_rhs.savefig(data_root + 'landau_strong_'+ fig_type +"_"+ str(compare_times) + 's_rhs.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.0,bbox_inches = 'tight')
            fig_nl_dt.savefig(data_root + 'landau_strong_' + fig_type +"_"+ str(compare_times) + 's_dt.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.0,bbox_inches = 'tight')
