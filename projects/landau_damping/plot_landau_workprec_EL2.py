import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict

compare_samples = [1,2,3,4,5,6,7,8,9,10]

data_root = "/home/krissmedt/data/landau/strong/"

datafiles = [
    'lan_TE10_a0.5_boris_synced_NZ10_NQ200000_wp_strong',
    'lan_TE10_a0.5_boris_synced_NZ100_NQ200000_wp_strong',
    'lan_TE10_a0.5_boris_synced_NZ1000_NQ200000_wp_strong',
    'lan_TE10_a0.5_boris_SDC_M3K2_NZ10_NQ200000_wp_strong',
    'lan_TE10_a0.5_boris_SDC_M3K2_NZ100_NQ200000_wp_strong',
    'lan_TE10_a0.5_boris_SDC_M3K2_NZ1000_NQ200000_wp_strong'
]

output_file_label = "strong_EL2_"

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

DH = DataHandler(**data_params)

plt.rcParams.update(plot_params)

for sample in compare_samples:
    fig_nl_rhs = plt.figure(1)
    ax_nl_rhs = fig_nl_rhs.add_subplot(1, 1, 1)
    fig_nl_dt = plt.figure(2)
    ax_nl_dt = fig_nl_dt.add_subplot(1, 1, 1)
    for filename in datafiles:
        file = h5.File(data_root+filename + ".h5",'r')
        dts = file["fields/dts"][:]
        times = file["fields/times"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        energy_errors = file["fields/energy_errors"][:]
        EL2_errors = file["fields/EL2_errors"][:]
        EL2_errors = np.array(EL2_errors)
        Eavg_errors = file["fields/Eavg_errors"][:]
        Eavg_errors = np.array(Eavg_errors)
        Emax_errors = file["fields/Emax_errors"][:]
        Emax_errors = np.array(Emax_errors)

        K = filename[filename.find('K')+1]
        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered" + ", Nz=" + file.attrs["res"]
            label = "Boris"
            c = '#0080FF'
        elif file.attrs["integrator"] == "boris_synced":
            c = '#0080FF'
            label = "Boris"
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC"
            if K == '1':
                c = '#00d65d'
                label += ", M=" + file.attrs["M"] + ", K=" + K
            elif K == '2':
                c = '#F9004B'
                label += ", M=" + file.attrs["M"] + ", K=" + K
            elif K == '3':
                c = '#FFD738'
                label += ", M=" + file.attrs["M"] + ", K=" + K

        file.close()

        ##Order Plot w/ rhs
        label_line = label + ', ' + str(times[sample]) + 's'
        ax_nl_rhs.plot(rhs_evals, EL2_errors[:, sample], marker="o", color=c, label=label_line)

        ##Order Plot w/ dt
        label_line = label + ', ' + str(times[sample]) + 's'
        ax_nl_dt.plot(dts, EL2_errors[:, sample], marker="o", color=c, label=label_line)

    handles, labels = fig_nl_rhs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax_nl_rhs.legend(by_label.values(), by_label.keys(),loc='best')

    handles, labels = fig_nl_dt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax_nl_dt.legend(by_label.values(), by_label.keys(),loc='best')

    axnl_list = []
    axnl_list.append(ax_nl_rhs)
    axnl_list.append(ax_nl_dt)

    i = 0
    for ax in axnl_list:
        i +=1
        if i == 1:
            orderSlope = -1
            ax.set_xlabel('Number of RHS evaluations')
        else:
            ax.set_xlabel(r'$\Delta t$')
            orderSlope = 1

        ax.set_xscale('log')
        #ax_rhs.set_xlim(10**3,10**5)
        ax.set_yscale('log')
        ax.set_ylim(10**(-6),1)
        #x.set_ylabel(r'E L2 Error $\Delta \sqrt{\sum \frac{E_i^2}{2} \Delta z}$')
        ax.set_ylabel(r'Rel. $||E||_{L2}$ Error')
        xRange = ax.get_xlim()
        yRange = ax.get_ylim()

        ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                    ls='dotted',c='0.25')
        ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                    ls='dashed',c='0.75')

    fig_nl_rhs.savefig(data_root + "landau_wp_" + output_file_label + "_" + str(times[sample]) + 's_rhs.pdf',
                       dpi=150, facecolor='w',
                       edgecolor='w',
                       orientation='portrait',
                       pad_inches=0.0,
                       bbox_inches ='tight'
                       )
    fig_nl_dt.savefig(data_root + "landau_wp_" + output_file_label + "_" + str(times[sample]) + 's_dt.pdf',
                      dpi=150, facecolor='w',
                      edgecolor='w',
                      orientation='portrait',
                      pad_inches=0.0,
                      bbox_inches ='tight'
                      )
