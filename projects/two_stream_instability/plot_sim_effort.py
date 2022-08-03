import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict


output_label = "tsi"
rhs_eval = 1

data_root = "/home/krissmedt/data/tsi/timing/"

setups = {
    "Boris" : [
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT10",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT20",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT40",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT50",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT80",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT100",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT200",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT400",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT500",
        "tsi_TE10_a0.1_boris_synced_NZ100_NQ20000_NT1000"
    ],
    "Boris-SDC M3K1": [
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT10",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT20",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT40",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT50",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT80",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT100",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT200",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT400",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT500",
        "tsi_TE10_a0.1_boris_SDC_M3K1_NZ100_NQ20000_NT1000"
    ],
    "Boris-SDC M3K2": [
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT10",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT20",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT40",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT50",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT80",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT100",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT200",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT400",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT500",
        "tsi_TE10_a0.1_boris_SDC_M3K2_NZ100_NQ20000_NT1000"
    ],
    "Boris-SDC M3K3": [
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT10",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT20",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT40",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT50",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT80",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT100",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT200",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT400",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT500",
        "tsi_TE10_a0.1_boris_SDC_M3K3_NZ100_NQ20000_NT1000"
    ]
}

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
plot_params['lines.linewidth'] = 6
plot_params['axes.titlepad'] = 5
plot_params['axes.linewidth'] = 1.5
plot_params['ytick.major.width'] = 2
plot_params['ytick.minor.width'] = 2
plot_params['xtick.major.width'] = 2
plot_params['xtick.minor.width'] = 2
plot_params['legend.loc'] = 'best'
plt.rcParams.update(plot_params)

results = {}
for key, value in setups.items():
    sim_times = []
    rhs_evals = []
    for sim_name in value:
        DH = DataHandler(**data_params)
        sim, sim_name = DH.load_sim(sim_name=sim_name, overwrite=True)

        dt = sim.dt
        Nt = sim.tSteps
        runTimes = sim.runTimeDict

        sim_times.append(runTimes["sim_time"])
        rhs_evals.append(sim.rhs_eval)

    result = np.array([np.array(rhs_evals, dtype=int), np.array(sim_times, dtype=float)])
    results[key] = result


print("Drawing timing vs. NL evaluations plot...")
fig = plt.figure(1, dpi=150)
ax = fig.add_subplot(1, 1, 1)
for key, value in results.items():
    ax.plot(value[0,:], value[1,:], label=key, marker="o")

ax.set_xlabel(r'RHS Evaluations')
ax.set_ylabel(r'Simulation time $(s)$')
ax.legend()
fig.savefig(data_root + output_label + '_timings.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
