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

def nonLinear_ext_E(species,mesh,controller=None):
    nq = species.pos.shape[0]
    for pii in range(0,nq):
        species.E[pii,2] += -np.power(species.pos[pii,2],3)
    
    return species

def nonLinear_mesh_E(species_list,mesh,controller=None):
    for zi in range(0,mesh.E[2,1,1,:].shape[0]-1):
        z = mesh.zlimits[0] + zi * mesh.dz
        mesh.E[2,1,1,zi] += -np.power(z,3)

    static_E = np.zeros(mesh.E.shape)
    static_E[:] = mesh.E[:]

    return mesh, static_E

def nonLinear_ion_bck(species_list,mesh,controller):
    pass

sims = {}

particle = 0


#sims['NLO__type2_boris_SDC_M3K3_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]


sims['NLO__type2_boris_SDC_M3K3_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
sims['NLO__type2_boris_synced_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254]
sims['NLO__type2_boris_synced_NZ100_TE1_NT1'] = [1,2,4,8,16,32,64,128,254,512]
sims['NLO__type1_boris_SDC_M5K5_NZ1_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
comp_run = 'NLO__type1_boris_SDC_M5K5_NZ1_TE1_NT1024'

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
pDataList_comp = DH_comp.load_p(['pos'],species=['spec1'],sim_name=comp_sim_name)
p1Data_comp = pDataList_comp[0] 


for key, value in sims.items():
    dts = []
    Nts = []
    rhs_evals = []
    zrels = []
    
    for tsteps in value:
        DH = dataHandler2()
        sim_name = key + str(tsteps)
        sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

        ####################### Analysis and Visualisation ############################
        dt = sim.dt
        Nt = sim.tSteps
        
        #DH.particle_time_plot(species=['spec1'],variables=['pos'],particles=[particle+1],sim_name=sim_name)
        
        pData_list = DH.load_p(['pos','vel'],species=['spec1'],sim_name=sim_name)
        
        p1Data_dict = pData_list[0]

        #mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name)
        
        ## particle position comparison 
        spec1_pos = p1Data_dict['pos'][:,:,2]
        comp_pos = p1Data_comp['pos'][:,:,2]

        zrel = np.abs(comp_pos[-1,particle] - spec1_pos[-1,particle])/np.abs(comp_pos[-1,particle])
        
        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)
        zrels.append(zrel)
        
    if sim.analysisSettings['particleIntegrator'] == 'boris_SDC':
        label_order = 'Boris-SDC,' + ' M=' + str(sim.analysisSettings['M']) + ', K=' + str(sim.analysisSettings['K'])
    elif sim.analysisSettings['particleIntegrator'] == 'boris_staggered':
        label_order = 'Staggered Boris'
    elif sim.analysisSettings['particleIntegrator'] == 'boris_synced':
        label_order = 'Synchronised Boris'
        

    ##Convergence Plot w/ rhs
    #fig_con = plt.figure(DH.figureNo+4)
    #ax_con = fig_con.add_subplot(1, 1, 1)
    #ax_con.plot(rhs_evals[1:],avg_slope_diff,label=label_order)
    
    ##Order Plot w/ rhs
    fig_rhs = plt.figure(DH.figureNo+1)
    ax_rhs = fig_rhs.add_subplot(1, 1, 1)
    ax_rhs.plot(rhs_evals,zrels,label=label_order)
    
    ##Order Plot w/ dt
    fig_dt = plt.figure(DH.figureNo+2)
    ax_dt = fig_dt.add_subplot(1, 1, 1)
    ax_dt.plot(dts,zrels,label=label_order)

"""
## Convergence plot finish
ax_con.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_con.set_xlabel('Number of RHS evaluations')
ax_con.set_yscale('log')
#ax_rhs.set_ylim(10**(-5),10**1)
ax_con.set_ylabel('Avg. slope difference')

xRange = ax_con.get_xlim()
yRange = ax_con.get_ylim()

ax_con.plot(xRange,DH.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_con.plot(xRange,DH.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_con.legend()
"""

## Order plot finish
ax_rhs.set_xscale('log')
#ax_rhs.set_xlim(10**3,10**5)
ax_rhs.set_xlabel('Number of RHS evaluations')
ax_rhs.set_yscale('log')
ax_rhs.set_ylim(10**(-16),1)
ax_rhs.set_ylabel('Avg. relative particle $\Delta z$')

xRange = ax_rhs.get_xlim()
yRange = ax_rhs.get_ylim()

ax_rhs.plot(xRange,DH.orderLines(-2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_rhs.plot(xRange,DH.orderLines(-4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_rhs.plot(xRange,DH.orderLines(-8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_rhs.legend()


## Order plot finish
ax_dt.set_xscale('log')
#ax_dt.set_xlim(10**-3,10**-1)
ax_dt.set_xlabel(r'$\Delta t$')
ax_dt.set_yscale('log')
ax_dt.set_ylim(10**(-16),1)
ax_dt.set_ylabel('Avg. relative particle $\Delta z$')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.plot(xRange,DH.orderLines(8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_dt.legend()
