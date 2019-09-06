from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2 as DH
import h5py as h5

analyse = True
simulate =  True
plot = True
prefix = "energy"

points_to_plot = 100

schemes = {'boris_SDC','boris_synced'}
#schemes = {'boris':'boris_synced'}

M = 3
iterations = [3]

omegaB = 25.0
omegaE = 4.9
epsilon = -1

tend = 10
Nt = 1000
#omega_dt = 0.25
#omega_dt = np.flip(omega_dt,0)
#dt = omega_dt/omegaB

sim_params = {}
species_params = {}
ploader_params = {}
analysis_params = {}
data_params = {}


sim_params['t0'] = 0
sim_params['tEnd'] = tend
#sim_params['dt'] = dt
sim_params['tSteps'] = Nt
sim_params['percentBar'] = True
sim_params['dimensions'] = 3
sim_params['xlimits'] = [0,20]
sim_params['ylimits'] = [0,20]
sim_params['zlimits'] = [0,15]
sim_params['simID'] = 'penning'

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
analysis_params['nodeType'] = 'lobatto'
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

analysis_params['centreMass_check'] = False
analysis_params['residual_check'] = False
analysis_params['rhs_check'] = True


data_params['samplePeriod'] = 1
data_params['write'] = True
data_params['write_vtk'] = False
data_params['time_plotting'] = False
data_params['tagged_particles'] = 'all'
data_params['time_plot_vars'] = ['pos']
data_params['trajectory_plotting'] = False
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


## Analytical solution ##
x0 = ploader_params['pos']
v0 = ploader_params['vel']
omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
Rplus = x0[0,0] - Rminus
Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
Iplus = x0[0,1] - Iminus

t = tend
xAnalyt = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
yAnalyt = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
zAnalyt = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)

vxAnalyt = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
vyAnalyt = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
vzAnalyt = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)

u = np.array([xAnalyt,vxAnalyt,yAnalyt,vyAnalyt,zAnalyt,vzAnalyt])
exactEnergy = u.transpose() @ H @ u



## Numerical solution ##
filenames = []
for scheme in schemes:    
    for K in iterations:
        analysis_params['particleIntegrator'] = scheme
        analysis_params['K'] = K
        
        if scheme == 'boris_staggered':
            integrator = 'boris'
        elif scheme == 'boris_SDC':
            integrator = 'boris_' + 'M' + str(M) + 'K' + str(K)
        elif scheme == 'boris_synced':
            integrator = 'boris'
            
        sim_name = 'penning_' + prefix + '_' + integrator
        filename = sim_name + ".h5"
        filenames.append(filename)

        
        if analyse == True:
            file = h5.File(filename,'w')
            grp = file.create_group('fields')
        
            sim_params['simID'] = sim_name
            
            model = dict(simSettings=sim_params,
                         speciesSettings=species_params,
                         pLoaderSettings=loader_params,
                         analysisSettings=analysis_params,
                         dataSettings=data_params)
            
            kppsObject = kpps(**model)
            
            
            if simulate == True:
                dHandler = kppsObject.run()
                s_name = dHandler.controller_obj.simID
            elif simulate == False:
                dHandler = DH()
                s_name = sim_params['simID']

            sim, garbage = dHandler.load_sim(sim_name=s_name,overwrite=True)


            var_list = ['pos','energy']
            data_list = dHandler.load_p(var_list,sim_name=s_name)
            data_dict = data_list[0]
            
            tArray = data_dict['t']
            hArray = data_dict['energy']
            
            skip = np.int((sim_params['tSteps']/data_params['samplePeriod'])/points_to_plot)
            
            tRed = tArray[0::skip]
            hRed = hArray[0::skip]

            
            if sim.analysisSettings['particleIntegrator'] == 'boris_SDC':
                label_res = 'Boris-SDC,' + ' M=' + str(sim.analysisSettings['M']) + ', K=' + str(sim.analysisSettings['K'])
                
            elif sim.analysisSettings['particleIntegrator'] == 'boris_staggered':
                label_res = 'Boris'
            elif sim.analysisSettings['particleIntegrator'] == 'boris_synced':
                label_res = 'Boris'
            
            file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
            file.attrs["label_res"] = label_res
            file.attrs["omegaB"] = omegaB
            
            try:
                file.attrs["M"] = str(sim.analysisSettings['M'])
                file.attrs["K"] = str(sim.analysisSettings['K'])
            except KeyError:
                pass
                
            grp.create_dataset('t',data=tRed)
            grp.create_dataset('energy',data=hRed)
            
            file.close()            
                
        if scheme  == 'boris_staggered':
            break
        elif scheme == 'boris_synced':
            break
        

if plot == True:
    dHandler = DH()
    
    for filename in filenames:
        file = h5.File(filename,'r')
        t = file["fields/t"][:]
        H = file["fields/energy"][:]
        label = file.attrs['label_res']


        ## Energy plot
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.scatter(t,H,label=label)
        ax1.set_xlim(0,sim_params['tEnd'])
        ax1.set_xlabel('$t$')
        ax1.set_ylim(0,10**4)
        ax1.set_ylabel('$\Delta E$')
        ax1.legend()
        
    plt.show()

