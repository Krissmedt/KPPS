import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict

snapshot_timesteps = [0, 100, 200, 300, 400, 500]

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

    print("Loading particle data...")
    pData_list = DH.load_p(['pos', 'vel'], species=['beam1', 'beam2'], sim_name=sim_name)

    p1Data_dict = pData_list[0]
    p2Data_dict = pData_list[1]

    p1_data = p1Data_dict['pos'][:, :, 2]
    p2_data = p2Data_dict['pos'][:, :, 2]

    v1_data = p1Data_dict['vel'][:, :, 2]
    v2_data = p2Data_dict['vel'][:, :, 2]

    no = 0
    for snap in snapshot_timesteps:
        no += 1
        print("Drawing snap no. {0}...".format(no))
        fig_snap = plt.figure(DH.figureNo + 10 + no, dpi=150)
        p_ax = fig_snap.add_subplot(1, 1, 1)
        line_p1 = p_ax.plot(p1_data[snap, :], v1_data[snap, :], 'bo', ms=2, c=(0.2, 0.2, 0.75, 1), label='Beam 1, v=1')[
            0]
        line_p2 = \
        p_ax.plot(p2_data[snap, :], v2_data[snap, :], 'ro', ms=2, c=(0.75, 0.2, 0.2, 1), label='Beam 2, v=-1')[0]
        p_ax.set_xlim([0.0, sim.zlimits[1]])
        p_ax.set_xlabel('$z$')
        p_ax.set_ylabel('$v_z$')
        p_ax.set_ylim([-3, 3])
        fig_snap.savefig(data_root + sim_name + '_snap_ts{0}.svg'.format(snap), dpi=150, facecolor='w', edgecolor='w',
                         orientation='portrait', pad_inches=0.05, bbox_inches='tight')