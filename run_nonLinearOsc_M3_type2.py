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

exptype = 2
prefix = ''

schemes = ['boris_staggered','boris_synced','boris_SDC']
steps = [1,5,10,50,100,200,250,500,1000]
resolutions = [10,100,1000,10000,100000,1000000]

M = 3
K = 3

tend = 1

ppc = 20




############################ Setup and Run ####################################
sim_params = {}
spec1_params = {}
loader1_params = {}
spec2_params = {}
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


loader1_params['load_type'] = 'direct'
loader1_params['speciestoLoad'] = [0]
loader1_params['pos'] = np.array([[0,0,0.5],[0,0,0.75]])
loader1_params['vel'] = np.array([[0,0,0],[0,0,0]])

#mesh_params['node_charge'] = -2*ppc*q
mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False
mLoader_params['BC_function'] = quartic_potential

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

data_params['write'] = True
data_params['write_m'] = False
data_params['dataRootFolder'] = "../data_nlo/" 

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
sim_params['nlo_type'] = exptype
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
            data_params['samples'] = 1

            species_params, loader_params = type_setup_spec(exptype,res,ppc,spec1_params,loader1_params,spec2_params,loader2_params)
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
            
            kppsObject = kpps()
            DH = kppsObject.start(**model)
