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

analyse = True
plot = True
snapPlot = False

y_max = 10**(-4)
fig_type = 'versus'
h5_suffix = ''
data_root = "../data_tsi_particles/"
start_time = 0
max_time = 1

sims = {}


#sims['tsi_TE1_boris_staggered_NZ10_NQ200000_NT'] = [1,2,4,8,16,32,64,128,256,512,1024]
sims['tsi_TE1_boris_staggered_NZ100_NQ200000_NT'] = [1,2,4,8,16,32,64,128,256,512,1024]
sims['tsi_TE1_boris_staggered_NZ1000_NQ200000_NT'] = [1,2,4,8,16,32,64,128,256,512,1024]
##
sims['tsi_TE1_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [1,2,4,8,16,32,64,128,256,512]
sims['tsi_TE1_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [1,2,4,8,16,32,64,128,256,512]


comp_run = 'tsi_TE1_boris_SDC_M5K5_NZ10000_NQ200000_NT1024'

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
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    DH_comp = dataHandler2(**data_params)
    comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
    mData_comp = DH_comp.load_m(['phi','E','dz'],sim_name=comp_sim_name)
    pDataList_comp = DH_comp.load_p(['pos'],species=['beam1','beam2'],sim_name=comp_sim_name)
    p1Data_comp = pDataList_comp[0] 
    tArray_comp = mData_comp['t']
    dz_comp = mData_comp['dz'][0] 
    E_comp = mData_comp['E'][-1,2,1,1,:-1]
    UE_comp = np.sum(E_comp*E_comp/2)*dz_comp

    for key, value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        avg_errors = []
        avg_slopes = []
        energy_errors = []
        
        filename = key + "_workprec_pos" + h5_suffix + ".h5"
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
            max_dt = np.int(max_time/(sim.dt*DH.samplePeriod))+1
            
            pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
            
            p1Data_dict = pData_list[0]
            p2Data_dict = pData_list[1]
    
            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz'],sim_name=sim_name)
            
            E = mData_dict['E'][-1,2,1,1,:-1]            
            UE = np.sum(E*E/2)*mData_dict['dz'][0]

            ## particle position comparison
            skip = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
            skip_int = np.int(skip)
            skip_p = np.int(p1Data_comp['pos'].shape[1]/p1Data_dict['pos'].shape[1])
            
            tArray = mData_dict['t']
            
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
            final_errors = rel_pos_diff[max_dt-1,:]
            
            avg_error = np.average(final_errors)
            dts.append(sim.dt)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            energy_errors.append(UE)
            avg_errors.append(avg_error)
            
            if snapPlot == True:
                phase_snap(0,p1Data_dict,p2Data_dict,figNo=3)
                phase_snap(-1,p1Data_dict,p2Data_dict,figNo=4)
        
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
        grp.create_dataset('energy_errors',data=energy_errors)
        file.close()

if plot == True:
    DH = dataHandler2(**data_params)
    plt.rcParams.update(plot_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key + "_workprec_pos" + h5_suffix + ".h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        avg_errors = file["fields/errors"][:]
        energy_errors = file["fields/energy_errors"][:]
        
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
        ax_rhs.plot(rhs_evals,avg_errors,label=label)
        
        ##Order Plot w/ dt
        fig_dt = plt.figure(11)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dts,avg_errors,label=label)
        
        ##Order Plot w/ rhs
        fig_E_rhs = plt.figure(12)
        ax_E_rhs = fig_E_rhs.add_subplot(1, 1, 1)
        ax_E_rhs.plot(rhs_evals,energy_errors,label=label)
        
        ##Order Plot w/ dt
        fig_E_dt = plt.figure(13)
        ax_E_dt = fig_E_dt.add_subplot(1, 1, 1)
        ax_E_dt.plot(dts,energy_errors,label=label)
        
    file.close()
        
        
    ax_rhs.set_xscale('log')
    #ax_rhs.set_xlim(10**3,10**5)
    ax_rhs.set_xlabel('Number of RHS evaluations')
    ax_rhs.set_yscale('log')
    ax_rhs.set_ylim(10**(-10),y_max)
    ax_rhs.set_ylabel('Avg. relative particle $\Delta z$')
    
    xRange = ax_rhs.get_xlim()
    yRange = ax_rhs.get_ylim()
    
    ax_rhs.plot(xRange,DH.orderLines(-2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_rhs.plot(xRange,DH.orderLines(-4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    
    plot_params['legend.loc'] = 'upper right'
    plt.rcParams.update(plot_params)
    ax_rhs.legend()
    
    ax_dt.set_xscale('log')
    #ax_dt.set_xlim(10**-3,10**-1)
    ax_dt.set_xlabel(r'$\Delta t$')
    ax_dt.set_yscale('log')
    ax_dt.set_ylim(10**(-10),y_max)
    ax_dt.set_ylabel('Avg. relative particle $\Delta z$')
    
    xRange = ax_dt.get_xlim()
    yRange = ax_dt.get_ylim()
    
    ax_dt.plot(xRange,DH.orderLines(1,xRange,yRange),
                ls='-.',c='0.1',label='1st Order')
    ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    
    plot_params['legend.loc'] = 'upper left'
    plt.rcParams.update(plot_params)
    ax_dt.legend()
    
    ax_E_rhs.set_xscale('log')
    #ax_rhs.set_xlim(10**3,10**5)
    ax_E_rhs.set_xlabel('Number of RHS evaluations')
    ax_E_rhs.set_yscale('log')
    ax_E_rhs.set_ylim(10**(-10),y_max)
    ax_E_rhs.set_ylabel(r'Energy Error $\Delta (\sum \frac{E_i^2}{2} \Delta x)$')
    
    xRange = ax_rhs.get_xlim()
    yRange = ax_rhs.get_ylim()
    
    ax_E_rhs.plot(xRange,DH.orderLines(-2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_E_rhs.plot(xRange,DH.orderLines(-4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    
    plot_params['legend.loc'] = 'upper right'
    plt.rcParams.update(plot_params)
    ax_E_rhs.legend()
    
    ax_E_dt.set_xscale('log')
    #ax_dt.set_xlim(10**-3,10**-1)
    ax_E_dt.set_xlabel(r'$\Delta t$')
    ax_E_dt.set_yscale('log')
    ax_E_dt.set_ylim(10**(-10),y_max)
    ax_E_dt.set_ylabel(r'Energy Error $\Delta (\sum \frac{E_i^2}{2} \Delta x)$')
    
    xRange = ax_dt.get_xlim()
    yRange = ax_dt.get_ylim()
    
    ax_E_dt.plot(xRange,DH.orderLines(1,xRange,yRange),
                ls='-.',c='0.1',label='1st Order')
    ax_E_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_E_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    
    plot_params['legend.loc'] = 'upper left'
    plt.rcParams.update(plot_params)
    ax_E_dt.legend()
    
    fig_rhs.savefig(data_root + 'tsi_particles_'+ fig_type + "_pos_" + str(max_time) + 's_rhs.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
    fig_dt.savefig(data_root + 'tsi_particles_' + fig_type + "_pos_" + str(max_time) + 's_dt.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
    fig_E_rhs.savefig(data_root + 'tsi_particles_'+ fig_type + "_energy_" + str(max_time) + 's_rhs.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
    fig_E_dt.savefig(data_root + 'tsi_particles_' + fig_type + "_energy_" + str(max_time) + 's_dt.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')