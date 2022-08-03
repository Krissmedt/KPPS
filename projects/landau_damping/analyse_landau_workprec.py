from projects.landau_damping.caseFile_landau1D import *
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict
from scipy.interpolate import interp1d

analysis_times = [0,1,2,3,4,5,6,7,8,9,10]

data_root = "/home/krissmedt/data/landau/strong/"

sims = {}


sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
sims['lan_TE10_a0.5_boris_SDC_M3K2_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]

sims['lan_TE10_a0.5_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
sims['lan_TE10_a0.5_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]
sims['lan_TE10_a0.5_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,400,500,1000]

#sims['tsi_TE50_a0.0001_boris_SDC_M3K2_NZ100_NQ20000_NT'] = [500]

comp_run = 'lan_TE10_a0.5_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


data_params = {}
data_params['dataRootFolder'] = data_root


############################### Data Analysis #########################################
filenames = []
analysis_ts = []

DH_comp = DataHandler(**data_params)
comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
mData_comp = DH_comp.load_m(['phi','rho','E','dz', 'zres', 'zlimits'],sim_name=comp_sim_name,max_t=analysis_times[-1])

for time in analysis_times:
    analysis_ts.append(int(time/(comp_sim.dt*DH_comp.samplePeriod)))
analysis_ts = np.array(analysis_ts)

tArray_comp = mData_comp['t']
dz_comp = mData_comp['dz'][0]
zlimits_comp = mData_comp['zlimits'][0]
zres_comp = round((zlimits_comp[1] - zlimits_comp[0])/dz_comp)
z_comp = np.linspace(zlimits_comp[0], zlimits_comp[1], zres_comp+1)
rho_comp = mData_comp['rho'][analysis_ts,1,1,:-1]
E_comp = mData_comp['E'][:,2,1,1,:-1]
comp_field_interpol = interp1d(z_comp, E_comp, kind="cubic")
E2 = E_comp*E_comp
UE =  np.sum(E2/2,axis=1)*dz_comp
UE_log = np.log(UE)
UE_comp = UE/UE[0]

EL2 = np.sum(E_comp*E_comp,axis=1)
EL2_comp = np.sqrt(EL2*dz_comp)


for key, value in sims.items():
    dts = []
    Nts = []
    rhs_evals = []
    linear_errors = []
    energy_errors = []
    E_errors_avg = []
    E_errors_max = []
    EL2_errors = []
    times = []

    filename = key[:-3] + "_wp_strong.h5"
    filenames.append(filename)

    try:
        file = h5.File(data_root+filename,'w')
    except OSError:
        file.close()
        file = h5.File(data_root+filename,'w')
    grp = file.create_group('fields')

    for tsteps in value:
        DH = DataHandler(**data_params)
        sim_name = key + str(tsteps)
        sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

        ####################### Analysis and Visualisation ############################
        dt = sim.dt
        Nt = sim.tSteps

        analysis_ts = []
        for time in analysis_times:
            analysis_ts.append(int(time/(sim.dt*DH.samplePeriod)))
        analysis_ts = np.array(analysis_ts)


        mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz','Rx','Rv','dz', 'zres', 'zlimits'],sim_name=sim_name,max_t=analysis_times[-1])
        tArray = mData_dict['t']
        dz = mData_dict['dz'][0]
        zlimits= mData_dict['zlimits'][0]
        zres = round((zlimits[1] - zlimits[0]) / dz)
        zArray = np.linspace(zlimits[0], zlimits[1], zres+1)

        phi_data = mData_dict['phi'][analysis_ts,1,1,:-1]
        E = mData_dict['E'][:,2,1,1,:-1]
        E2 = E*E
        UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0]
        UE_log = np.log(UE)
        UE_comp = UE/UE[0]

        EL2 = np.sum(E*E,axis=1) * mData_dict['dz'][0]
        EL2 = np.sqrt(EL2)

        dts.append(sim.dt)
        Nts.append(sim.tSteps)
        rhs_evals.append(sim.rhs_eval)

        E_comp_ip = comp_field_interpol(zArray)
        E_error_array = np.abs(E_comp_ip - E)/np.abs(E_comp_ip)
        E_error_avg = np.average(E_error_array, axis=1)
        E_error_max = np.max(E_error_array, axis=1)
        E_errors_avg.append(E_error_avg)
        E_errors_max.append(E_error_max)

        EL2_error = np.abs(EL2_comp - EL2[analysis_ts]) / np.abs(EL2_comp)
        EL2_errors.append(EL2_error)


    file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
    file.attrs["res"] = str(mData_dict['zres'][0])
    try:
        file.attrs["M"] = str(3)
        file.attrs["K"] = filename[filename.find('K')+1]
    except KeyError:
        pass

    grp.create_dataset('times',data=analysis_times)
    grp.create_dataset('dts',data=dts)
    grp.create_dataset('Nts',data=Nts)
    grp.create_dataset('rhs_evals',data=rhs_evals)
    grp.create_dataset('EL2_errors', data=EL2_errors)
    grp.create_dataset('Emax_errors', data=E_errors_max)
    grp.create_dataset('Eavg_errors', data=E_errors_avg)
    grp.create_dataset('energy',data=UE)
    grp.create_dataset('energy_reference',data=UE_comp)
    grp.create_dataset('energy_errors',data=energy_errors)
    file.close()
