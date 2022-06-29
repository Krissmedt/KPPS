import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict

analyse = True
fieldPlot = False
snapPlot = False
resPlot = False
compare_reference = True
plot = False


analysis_times = [0,1,2,3,4,5,6,7,8,9,10]

data_root = "/home/krissmedt/data/tsi/weak/"

sims = {}

#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K1_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
##
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]
sims['tsi_TE10_a0.0001_boris_SDC_M3K2_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]

#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#sims['tsi_TE10_a0.0001_boris_SDC_M3K3_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500]
#
sims['tsi_TE10_a0.0001_boris_synced_NZ10_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]
sims['tsi_TE10_a0.0001_boris_synced_NZ100_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]
sims['tsi_TE10_a0.0001_boris_synced_NZ1000_NQ200000_NT'] = [10,20,40,50,80,100,200,300,400,500,1000]

#sims['tsi_TE50_a0.0001_boris_SDC_M3K2_NZ100_NQ20000_NT'] = [500]

comp_run = 'tsi_TE10_a0.0001_boris_SDC_M3K3_NZ5000_NQ200000_NT5000'


data_params = {}
data_params['dataRootFolder'] = data_root


############################### Data Analysis #########################################
filenames = []
analysis_ts = []

DH_comp = DataHandler(**data_params)
comp_sim, comp_sim_name = DH_comp.load_sim(sim_name=comp_run,overwrite=True)
mData_comp = DH_comp.load_m(['phi','rho','E','dz'],sim_name=comp_sim_name,max_t=analysis_times[-1])

for time in analysis_times:
    analysis_ts.append(int(time/(comp_sim.dt*DH_comp.samplePeriod)))
analysis_ts = np.array(analysis_ts)

tArray_comp = mData_comp['t']
dz_comp = mData_comp['dz'][0]

rho_comp = mData_comp['rho'][analysis_ts,1,1,:-2]
E = mData_comp['E'][:,2,1,1,:-2]
E2 = E*E
UE =  np.sum(E2/2,axis=1)*dz_comp
UE_log = np.log(UE)
UE_comp = UE/UE[0]

EL2 = np.sum(E*E,axis=1)
EL2_comp = np.sqrt(EL2*dz_comp)


for key, value in sims.items():
    dts = []
    Nts = []
    rhs_evals = []
    linear_errors = []
    energy_errors = []
    E_errors = []
    times = []

    filename = key[:-3] + "_wp_weak.h5"
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


        mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz','Rx','Rv'],sim_name=sim_name,max_t=analysis_times[-1])
        tArray = mData_dict['t']

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

        E_error = np.abs(EL2_comp-EL2[analysis_ts])/np.abs(EL2_comp)
        E_errors.append(E_error)


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
    grp.create_dataset('E_errors',data=E_errors)
    grp.create_dataset('energy',data=UE)
    grp.create_dataset('energy_reference',data=UE_comp)
    grp.create_dataset('energy_errors',data=energy_errors)
    file.close()
