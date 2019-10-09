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

h5_suffix = ''
data_root = "../data/"
start_time = 0
max_time = 1

sims = {}


sims['tsi_1_TE1_boris_staggered_NZ_NQ2000_NT32'] = [5,10,20,40,80,160]
sims['tsi_1_TE1_boris_staggered_NZ_NQ2000_NT64'] = [5,10,20,40,80,160]
sims['tsi_2_TE1_boris_staggered_NZ_NQ2000_NT64'] = [5,10,20,40,80]

comp_run = 'tsi_2_TE1_boris_staggered_NZ160_NQ2000_NT64'

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
plot_params['legend.fontsize'] = 10
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 12
plot_params['axes.titlesize'] = 12
plot_params['xtick.labelsize'] = 8
plot_params['ytick.labelsize'] = 8
plot_params['lines.linewidth'] = 2
plot_params['axes.titlepad'] = 5
plt.rcParams.update(plot_params)

filenames = []
if analyse == True:
    DH_comp = dataHandler2(**data_params)
    comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
    mData_comp = DH_comp.load_m(['phi'],sim_name=comp_sim_name)
    pDataList_comp = DH_comp.load_p(['pos'],species=['beam1','beam2'],sim_name=comp_sim_name)
    p1Data_comp = pDataList_comp[0] 

    for key, value in sims.items():
        dzs = []
        Nts = []
        rhs_evals = []
        avg_errors = []
        avg_slopes = []
        avg_errors_nonlinear = []
        
        filename = key + "_workprec_dz" + h5_suffix + ".h5"
        filenames.append(filename)
        file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
    
        for ncells in value:
            DH = dataHandler2(**data_params)
            insert = key.find('NZ')
            sim_name = key[:insert+2] + str(ncells) + key[insert+2:]
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
            dz = mData_dict['dz'][0]

            ## particle position comparison
            skip = (sim.dt*DH.samplePeriod)/(comp_sim.dt*DH_comp.samplePeriod)
            skip_int = np.int(skip)
            skip_p = np.int(p1Data_comp['pos'].shape[1]/p1Data_dict['pos'].shape[1])
            
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
            final_errors = rel_pos_diff[max_dt-1,:]
            
            avg_error = np.average(final_errors)
            dzs.append(dz)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            avg_errors.append(avg_error)
            print(avg_errors)
            
            if snapPlot == True:
                phase_snap(0,p1Data_dict,p2Data_dict,figNo=3)
                phase_snap(-1,p1Data_dict,p2Data_dict,figNo=4)
        
        file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
        file.attrs["dt"] = str(dt)
        try:
            file.attrs["M"] = str(sim.analysisSettings['M'])
            file.attrs["K"] = str(sim.analysisSettings['K'])
        except KeyError:
            pass
        
        grp.create_dataset('dzs',data=dzs)
        grp.create_dataset('Nts',data=Nts)
        grp.create_dataset('rhs_evals',data=rhs_evals)
        grp.create_dataset('errors',data=avg_errors)
        file.close()

if plot == True:
    plt.rcParams.update(plot_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key + "_workprec_dz" + h5_suffix + ".h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dzs = file["fields/dzs"][:]
        avg_errors = file["fields/errors"][:]

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", dt=" + file.attrs["dt"]
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris Synced" + ", dt=" + file.attrs["dt"]
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC" + ", dt=" + file.attrs["dt"]
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]
        print(dzs)
        ##Order Plot w/ dz
        fig_dz = plt.figure(11)
        ax_dz = fig_dz.add_subplot(1, 1, 1)
        ax_dz.plot(dzs,avg_errors,label=label)
        
    file.close()
    
    ax_dz.set_xscale('log')
    #ax_dz.set_xlim(10**-3,10**-1)
    ax_dz.set_xlabel(r'$\Delta z$')
    ax_dz.set_yscale('log')
    ax_dz.set_ylim(10**(-12),10)
    ax_dz.set_ylabel('Avg. relative particle $\Delta z$')
    
    xRange = ax_dz.get_xlim()
    yRange = ax_dz.get_ylim()
    
    ax_dz.plot(xRange,DH.orderLines(1,xRange,yRange),
                ls='-.',c='0.1',label='1st Order')
    ax_dz.plot(xRange,DH.orderLines(2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_dz.plot(xRange,DH.orderLines(4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    ax_dz.legend()
