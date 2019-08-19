from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *
from dataHandler2 import dataHandler2
import matplotlib.animation as animation
from caseFile_nonLinearOsc import *

"""
Experiment type:
    1 Direct E
    2 Gather E from mesh
    3 Solve for E -> Gather E from mesh
    4 Scatter q -> Solve for E -> Gather E from mesh
"""

exptype = 3
prefix = ''

schemes = ['boris_SDC']
steps = [1]
resolutions = [10]

M = 5
K = 5

tend = 1

ppc = 20




############################ Setup and Run ####################################
sim_params = {}
spec1_params = {}
loader1_params = {}
beam2_params = {}
loader2_params = {}
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 1
sim_params['zlimits'] = [-1,1]

spec1_params['name'] = 'spec1'
spec1_params['nq'] = 2
spec1_params['q'] = 1
spec1_params['mq'] = 1


loader1_params['load_type'] = 'direct'
loader1_params['speciestoLoad'] = [0]
loader1_params['pos'] = np.array([[0,0,0.5],[0,0,0.75]])
loader1_params['vel'] = np.array([[0,0,0],[0,0,0]])

#mesh_params['node_charge'] = -2*ppc*q
mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['nodeType'] = 'lobatto'
analysis_params['M'] = M
analysis_params['K'] = K

analysis_params['E_type'] = 'custom'
analysis_params['custom_external_E'] = nonLinear_ext_E
analysis_params['custom_static_E'] = nonLinear_mesh_E

analysis_params['units'] = 'custom'
analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
analysis_params['rhs_check'] = True

data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['write_m'] = False


plot_params = {}
plot_params['legend.fontsize'] = 8
plot_params['figure.figsize'] = (6,4)
plot_params['axes.labelsize'] = 12
plot_params['axes.titlesize'] = 12
plot_params['xtick.labelsize'] = 8
plot_params['ytick.labelsize'] = 8
plot_params['lines.linewidth'] = 2
plot_params['axes.titlepad'] = 5
data_params['plot_params'] = plot_params


analysis_params = type_setup(exptype,analysis_params)

for scheme in schemes:
    analysis_params['particleIntegrator'] = scheme
    
    if scheme == 'boris_staggered':
        analysis_params['pre_hook_list'] = ['ES_vel_rewind']
    elif scheme == 'boris_SDC':
        analysis_params['pre_hook_list'] = []
        scheme += '_M' + str(M) + 'K' + str(K)
    else:
        analysis_params['pre_hook_list'] = []
        

    for res in resolutions:
        mLoader_params['resolution'] = [2,2,res]
        dts = []
        
        for Nt in steps:
            sim_params['tSteps'] = Nt
            dt = tend/Nt
            dts.append(dt)

            species_params = [spec1_params]
            loader_params = [loader1_params]
    
            sim_name = 'NLO_' + prefix + '_' + 'type' + str(exptype) + '_' + scheme + '_NZ' + str(res) + '_TE' + str(tend) + '_NT' + str(Nt) 
            sim_params['simID'] = sim_name
            
            ## Numerical solution ##
            model = dict(simSettings=sim_params,
                         speciesSettings=species_params,
                         pLoaderSettings=loader_params,
                         meshSettings=mesh_params,
                         analysisSettings=analysis_params,
                         mLoaderSettings=mLoader_params,
                         dataSettings=data_params)
            
            kppsObject = kpps(**model)
            DH = kppsObject.run()

            
"""          
            ####################### Analysis and Visualisation ############################
            
            
            pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
            
            p1Data_dict = pData_list[0]
            p2Data_dict = pData_list[1]
            
            mData_dict = DH.load_m(['phi','E','rho','PE_sum'],sim_name=sim_name)
            
            tArray = mData_dict['t']
            
            zRel_sync_pic = np.abs(pos_sync_pic - spec_comp.pos[:,2])
            zRel_stag_pic = np.abs(pos_stag_pic- spec_comp.pos[:,2])
            zRel_sdc_pic = np.abs(pos_sdc_pic - spec_comp.pos[:,2])
            
            
        ##Order Plot w/ dt
        fig_dt = plt.figure(2)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dts,zRel[:,0],label=sim_name[0:-6])

            
            
## Order plot finish
ax_dt.set_xscale('log')
#ax_dt.set_xlim(10**-3,10**-1)
ax_dt.set_xlabel('$\Delta t$')
ax_dt.set_yscale('log')
#ax_dt.set_ylim(10**(-7),10**1)
ax_dt.set_ylabel('$\Delta x^{(rel)}$')

xRange = ax_dt.get_xlim()
yRange = ax_dt.get_ylim()

ax_dt.plot(xRange,DH.orderLines(1,xRange,yRange),
            ls='-',c='0.25',label='1st Order')
ax_dt.plot(xRange,DH.orderLines(2,xRange,yRange),
            ls='dotted',c='0.25',label='2nd Order')
ax_dt.plot(xRange,DH.orderLines(4,xRange,yRange),
            ls='dashed',c='0.75',label='4th Order')
ax_dt.plot(xRange,DH.orderLines(8,xRange,yRange),
            ls='dashdot',c='0.1',label='8th Order')
ax_dt.legend()



"""