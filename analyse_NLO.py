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
from caseFile_nonLinearOsc import *


sims = {}

particle = 0

#sims['NLO__type2_boris_staggered_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]


#sims['NLO__type2_boris_staggered_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]

#sims['NLO__type2_boris_SDC_M3K3_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254]

#sims['NLO__type2_boris_SDC_M3K3_NZ10_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ1000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ10000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M3K3_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]

#sims['NLO__type2_boris_SDC_M5K5_NZ10_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ1000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ10000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_SDC_M5K5_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]

#sims['NLO__type2_boris_staggered_NZ10_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_staggered_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_staggered_NZ1000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_staggered_NZ10000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_staggered_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_staggered_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]

#sims['NLO__type2_boris_synced_NZ10_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ100_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ1000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ10000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ100000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]
#sims['NLO__type2_boris_synced_NZ1000000_TE1_NT'] = [1,2,4,8,16,32,64,128,254,512]

sims['NLO__type3_boris_synced_NZ100_TE1_NT'] = [512]
#sims['NLO__type2_boris_synced_NZ100_TE1_NT'] = [512]

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
plot_params['legend.loc'] = 'upper right'
plot_params['legend.loc'] = 'lower left'
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

        mData_dict = DH.load_m(['phi','E','rho'],sim_name=sim_name)
        
        ## particle position comparison 
        spec1_pos = p1Data_dict['pos'][:,:,2]
        comp_pos = p1Data_comp['pos'][:,:,2]

        zrel = np.abs(comp_pos[-1,particle] - spec1_pos[-1,particle])/np.abs(comp_pos[-1,particle])
        
        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)
        zrels.append(zrel)
        
    if sim.analysisSettings['particleIntegrator'] == 'boris_SDC':
        label_order = 'Boris-SDC,' + ' M=' + str(sim.analysisSettings['M']) + ', K=' + str(sim.analysisSettings['K']) + ', Nz=' + str(sim.mLoaderSettings['resolution'][2])
    elif sim.analysisSettings['particleIntegrator'] == 'boris_staggered':
        label_order = 'Staggered Boris' + ', Nz=' + str(sim.mLoaderSettings['resolution'][2])
    elif sim.analysisSettings['particleIntegrator'] == 'boris_synced':
        label_order = 'Synchronised Boris' + ', Nz=' + str(sim.mLoaderSettings['resolution'][2])
        

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
    
    
res = 101
z = np.linspace(-1,1,res)
rho = 3*np.power(z,2)
E = -np.power(z,3)
phi = -0.25 * np.power(z,4)

fig = plt.figure(10)
ax1 = fig.add_subplot(111)
#ax1.plot(z,rho)
ax1.plot(z,phi)
ax1.plot(z,mData_dict['phi'][0,1,1,:-1])
ax1.plot(z,mData_dict['phi'][256,1,1,:-1])
ax1.plot(z,mData_dict['phi'][-1,1,1,:-1])

fig = plt.figure(11)
ax2 = fig.add_subplot(111)
#ax1.plot(z,rho)
#ax1.plot(z,E)
#ax1.plot(z,phi)
ax2.plot(z,mData_dict['E'][0,2,1,1,:-1])
ax2.plot(z,mData_dict['E'][256,2,1,1,:-1])
ax2.plot(z,mData_dict['E'][-1,2,1,1,:-1])

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
