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

def type_setup(exptype,analysis_params):
    if exptype == 1:
        analysis_params['fieldIntegration'] = False
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'none'
        analysis_params['external_fields'] = True
        analysis_params['external_fields_mesh'] = False
        
    elif exptype == 2:
        analysis_params['fieldIntegration'] = True
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'trilinear_gather'
        analysis_params['external_fields'] = False
        analysis_params['external_fields_mesh'] = True
        
    elif exptype == 3:
        analysis_params['background'] = nonLinear_ion_bck
        
    return analysis_params
        
        
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


"""
Experiment type:
    1 Direct E
    2 Gather E from mesh
    3 Solve for E -> Gather E from mesh
    4 Scatter q -> Solve for E -> Gather E from mesh
"""

exptype = 2
prefix = ''

#schemes = ['boris_SDC']
schemes = ['boris_synced','boris_staggered']
steps = [1,2,4,8,16,32]
resolutions = [10000]

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
analysis_params['M'] = 5
analysis_params['K'] = 5

analysis_params['E_type'] = 'custom'
analysis_params['custom_external_E'] = nonLinear_ext_E
analysis_params['custom_static_E'] = nonLinear_mesh_E

analysis_params['units'] = 'custom'
analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
analysis_params['rhs_check'] = True

data_params['samplePeriod'] = 1
data_params['write'] = True

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
    if scheme == 'boris_staggered':
        analysis_params['pre_hook_list'] = ['ES_vel_rewind']
        
    analysis_params['particleIntegrator'] = scheme
    
    for res in resolutions:
        mLoader_params['resolution'] = [2,2,res]
        dts = []
        
        for Nt in steps:
            sim_params['tSteps'] = Nt
            dt = tend/Nt
            dts.append(dt)

            species_params = [spec1_params]
            loader_params = [loader1_params]
    
            sim_name = 'NLO_' + prefix + '_' + 'type' + str(exptype) + '_' + analysis_params['particleIntegrator'] + '_NZ' + str(res) + '_TE' + str(tend) + '_NT' + str(Nt) 
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