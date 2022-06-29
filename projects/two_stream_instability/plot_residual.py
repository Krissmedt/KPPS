import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict

data_root = "/home/krissmedt/data/tsi/weak/"

sims = [
    "tsi_TE50_a0.0001_boris_SDC_M3K4_NZ100_NQ20000_NT500"
]

############################### Setup #########################################
data_params = {}
data_params['dataRootFolder'] = data_root
plot_params = {}
plot_params['legend.fontsize'] = 16
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['axes.linewidth'] = 1.5
plot_params['ytick.major.width'] = 2
plot_params['ytick.minor.width'] = 2
plot_params['xtick.major.width'] = 2
plot_params['xtick.minor.width'] = 2
plot_params['legend.loc'] = 'best'
plt.rcParams.update(plot_params)

filenames = []

for sim_name in sims:
    DH = DataHandler(**data_params)
    sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

    ####################### Analysis and Visualisation ############################
    dt = sim.dt
    Nt = sim.tSteps

    pData_list = DH.load_p(['x_res', 'v_res'], species=['beam1', 'beam2'], sim_name=sim_name)
    p1Data_dict = pData_list[0]
    p2Data_dict = pData_list[1]
    tArray = p1Data_dict['t']
    xRes = np.array(p1Data_dict['x_res'][1:,:,-1], dtype=float)
    vRes = np.array(p1Data_dict['v_res'][1:,:,-1], dtype=float)

    print("Drawing x-residual plot for Beam 1 of " + sim_name + "...")
    fig_p1 = plt.figure(1,dpi=150)
    R_ax = fig_p1.add_subplot(1,1,1)
    for i in range(0, xRes.shape[1]):
        R_ax.plot(tArray[1:],xRes[:, i],label='k = ' + str(i+1))
    R_ax.set_xlabel('$t$')
    R_ax.set_ylabel(r'$||R||_{L2}$')
    R_ax.set_yscale('log')
    R_ax.legend()
    fig_p1.savefig(data_root + sim_name + '_xResidual.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')

    print("Drawing v-residual plot for Beam 1 of " + sim_name + "...")
    fig_p2 = plt.figure(1,dpi=150)
    R_ax = fig_p2.add_subplot(1,1,1)
    for i in range(0, vRes.shape[1]):
        R_ax.plot(tArray[1:],vRes[:, i],label='k = ' + str(i+1))
    R_ax.set_xlabel('$t$')
    R_ax.set_ylabel(r'$||R||_{L2}$')
    R_ax.set_yscale('log')
    R_ax.legend()
    fig_p2.savefig(data_root + sim_name + '_vResidual.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')

