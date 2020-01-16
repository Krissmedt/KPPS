from kpps import kpps as kpps_class
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2


schemes = ['boris_SDC']
node_type = 'lobatto'
M = 3
iterations = [3]
dtwb = [10,1,0.1]


tend = 16

omegaB = 25.0
omegaE = 4.9
epsilon = -1

prefix = 'TE'+str(tend)



sim_params = {}
species_params = {}
ploader_params = {}
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}


sim_params['t0'] = 0
sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 3
sim_params['xlimits'] = [0,20]
sim_params['ylimits'] = [0,20]
sim_params['zlimits'] = [0,15]

species_params['q'] = 1
species_params['a'] = 1
mq = species_params['q']/species_params['a']
ploader_params['load_type'] = 'direct'
ploader_params['speciestoLoad'] = [0]
ploader_params['pos'] = np.array([[10,0,0]])
ploader_params['vel'] = np.array([[100,0,100]])

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])


analysis_params['particleIntegration'] = True
analysis_params['M'] = M
analysis_params['K'] = 3
analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'coulomb'
analysis_params['external_fields'] = True
analysis_params['E_type'] = 'transform'
analysis_params['E_transform'] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
analysis_params['E_magnitude'] = -epsilon*omegaE**2/species_params['a']
analysis_params['B_type'] = 'uniform'
analysis_params['B_transform'] = [0,0,1]
analysis_params['B_magnitude'] = omegaB/species_params['a']
analysis_params['hooks'] = ['energy_calc_penning']
analysis_params['H'] = H

analysis_params['centreMass_check'] = True
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = True


data_params['dataRootFolder'] = '../data_penning/'
data_params['write'] = True
data_params['write_m'] = False
data_params['write_vtk'] = False
data_params['time_plotting'] = False
data_params['tagged_particles'] = 'all'
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = True
data_params['trajectories'] = [1]
data_params['plot_limits'] = [20,20,15]


plot_params = {}
plot_params['legend.fontsize'] = 12
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 10
data_params['plot_params'] = plot_params


species_params = [species_params]
loader_params = [ploader_params]

run_times_inner = np.zeros((len(dtwb),len(iterations)),dtype=np.float)
run_times = []

## Calculated Params ##
x0 = ploader_params['pos']
v0 = ploader_params['vel']
nq = x0.shape[0]

omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
Rplus = x0[0,0] - Rminus
Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
Iplus = x0[0,1] - Iminus

kppsObject = kpps_class()

## Numerical solution ##
for scheme in schemes:
    analysis_params['particleIntegrator'] = scheme
    analysis_params['nodeType'] = node_type
    
    j = 0
    for K in iterations:
        analysis_params['K'] = K
        for dt in dtwb:
            sim_params['dt'] = dt/omegaB
            Nt = floor(sim_params['tEnd']/sim_params['dt'])
            data_params['samplePeriod'] = 1
            
            xMod = Rplus*cos(omegaPlus*dt) + Rminus*cos(omegaMinus*dt) + Iplus*sin(omegaPlus*dt) + Iminus*sin(omegaMinus*dt)
            yMod = Iplus*cos(omegaPlus*dt) + Iminus*cos(omegaMinus*dt) - Rplus*sin(omegaPlus*dt) - Rminus*sin(omegaMinus*dt)
            zMod = x0[0,2] * cos(omegaTilde * dt) + v0[0,2]/omegaTilde * sin(omegaTilde*dt)
            
            
            v_half_dt = [(xMod-x0[0,0])/(dt),(yMod-x0[0,1])/(dt),(zMod-x0[0,2])/(dt)]
        
            xOne = [xMod,yMod,zMod]
            vHalf = v_half_dt
            
            sim_name = 'pen_' + prefix + '_' + analysis_params['particleIntegrator'] + '_NQ' + str(int(nq)) + '_NT' + str(Nt) 
            sim_params['simID'] = sim_name
            
            
            model = dict(simSettings=sim_params,
                         speciesSettings=species_params,
                         pLoaderSettings=loader_params,
                         meshSettings=mesh_params,
                         analysisSettings=analysis_params,
                         mLoaderSettings=mLoader_params,
                         dataSettings=data_params)
                
            
            DH = kppsObject.start(**model)
            sim = DH.controller_obj
            sim_name = sim.simID
        
        if scheme == 'boris_staggered':
            break
        if scheme == 'boris_synced':
            break

