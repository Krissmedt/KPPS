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

analyse = False
plot = True
snapPlot = False

h5_suffix = ''
data_root = "../data_tsi_spatial/"
fig_type = 'NQ'
start_time = 0
max_time = 1

sims = {}

sims['tsi_TE1_boris_SDC_NZ1000_NQ_NT100'] = [20,200,2000,20000]
sims['tsi_TE1_boris_SDC_NZ1000_NQ_NT100'] = [20,200,2000,20000]

comp_run = 'tsi_TE1_boris_SDC_NZ5000_NQ200000_NT1000'

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
    mData_comp = DH_comp.load_m(['E','phi','dz'],sim_name=comp_sim_name)
    dz_comp = mData_comp['dz'][0] 
    E_comp = mData_comp['E'][-1,2,1,1,:-2]
    UE_comp = np.sum(E_comp*E_comp/2)*dz_comp

    for key, value in sims.items():
        dzs = []
        Nts = []
        Nqs = []
        rhs_evals = []
        errors = []

        filename = key + "_workprec_nq" + h5_suffix + ".h5"
        filenames.append(filename)
        file = h5.File(data_root+filename,'w')
        grp = file.create_group('fields')
    
        for nq in value:
            DH = dataHandler2(**data_params)
            insert = key.find('NQ')
            sim_name = key[:insert+2] + str(nq) + key[insert+2:]
            sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)
    
            ####################### Analysis and Visualisation ############################
            dt = sim.dt
            Nt = sim.tSteps
            
            start_dt = np.int(start_time/(sim.dt*DH.samplePeriod))
            max_dt = np.int(max_time/(sim.dt*DH.samplePeriod))+1
    
            mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz'],sim_name=sim_name)
            dz = mData_dict['dz'][0]
            
            
            E = mData_dict['E'][-1,2,1,1,:-2]            
            UE = np.sum(E*E/2)*dz
            error = abs(UE_comp-UE)/UE_comp

            dzs.append(dz)
            Nqs.append(nq)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            errors.append(error)
        
        
        file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
        file.attrs["dt"] = str(dt)
        file.attrs["Nt"] = str(Nt)

        try:
            file.attrs["M"] = str(sim.analysisSettings['M'])
            file.attrs["K"] = str(sim.analysisSettings['K'])
        except KeyError:
            pass
        
        grp.create_dataset('dzs',data=dzs)
        grp.create_dataset('Nts',data=Nts)
        grp.create_dataset('Nqs',data=Nqs)
        grp.create_dataset('rhs_evals',data=rhs_evals)
        grp.create_dataset('errors',data=errors)
        grp.create_dataset('energy',data=UE)
        grp.create_dataset('energy_reference',data=UE_comp)
        file.close()

if plot == True:
    plt.rcParams.update(plot_params)
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key + "_workprec_nq" + h5_suffix + ".h5"
            filenames.append(filename)
            

    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        nqs = file["fields/Nqs"][:]
        errors = file["fields/errors"][:]

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nt=" + file.attrs["Nt"]
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris Synced" + ", Nt=" + file.attrs["Nt"]
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC" + ", Nt=" + file.attrs["Nt"]
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]

        ##Order Plot w/ dz
        fig_dz = plt.figure(11)
        ax_dz = fig_dz.add_subplot(1, 1, 1)
        ax_dz.plot(nqs,errors,label=label)
    file.close()
    
    ax_dz.set_xscale('log')
    ax_dz.set_xlim(10**1,10**5)
    ax_dz.set_xlabel(r'Particle Count $N_q$')
    ax_dz.set_yscale('log')
    ax_dz.set_ylim(10**(-5),10**7)
    ax_dz.set_ylabel(r'Energy Error $\Delta (\sum \frac{E_i^2}{2} \Delta x)$')
    
    xRange = ax_dz.get_xlim()
    yRange = ax_dz.get_ylim()
    
    ax_dz.plot(xRange,DH.orderLines(-1,xRange,yRange),
                ls='-.',c='0.1',label='1st Order')
    ax_dz.plot(xRange,DH.orderLines(-2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_dz.plot(xRange,DH.orderLines(-4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    ax_dz.legend()
    fig_dz.savefig(data_root + 'tsi_spatial_' + fig_type + '.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
