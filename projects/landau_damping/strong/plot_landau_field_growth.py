import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler

def find_peaks(peak_intervals,EL2,dt,samplePeriod):
    peaks = []
    for interval in peak_intervals:
        NA = int(interval[0]/(dt*samplePeriod))
        NB = int(interval[1]/(dt*samplePeriod))
        mi = NA + np.argmax(EL2[NA:NB])
        peaks.append(mi)

    return peaks

peak_intervals = [[0,2],[2,4]]
peak_intervals2 = [[22.5,25],[25,27.5]]

fit1_start = peak_intervals[0][0]
fit1_stop = 5
fit2_start = 22
fit2_stop = 30


data_root = "/home/krissmedt/data/landau/strong/"

sims = [
    "lan_TE30_a0.5_boris_SDC_M3K2_NZ100_NQ200000_NT300"
]

################################ Linear analysis ##############################

gamma_lit1 = -0.2922
gamma_lit2 = 0.08612


############################### Setup #########################################
data_params = {}
data_params['dataRootFolder'] = data_root
plot_params = {}
plot_params['legend.fontsize'] = 22
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 24
plot_params['axes.titlesize'] = 24
plot_params['xtick.labelsize'] = 24
plot_params['ytick.labelsize'] = 24
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['axes.linewidth'] = 1.5
plot_params['ytick.major.width'] = 2
plot_params['ytick.minor.width'] = 2
plot_params['xtick.major.width'] = 2
plot_params['xtick.minor.width'] = 2
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)

filenames = []

for sim_name in sims:
    DH = DataHandler(**data_params)
    sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

    ####################### Analysis and Visualisation ############################
    dt = sim.dt
    Nt = sim.tSteps

    NA = int(fit1_start / (sim.dt * DH.samplePeriod))
    NB = int(fit1_stop / (sim.dt * DH.samplePeriod))
    NC = int(fit2_start / (sim.dt * DH.samplePeriod))
    ND = int(fit2_stop / (sim.dt * DH.samplePeriod))

    mData_dict = DH.load_m(['phi', 'E', 'rho', 'zres', 'dz', 'q', 'grid_x', 'grid_v', 'f', 'Rx', 'Rv'],
                           sim_name=sim_name, max_t='all')
    tArray = mData_dict['t']

    E = mData_dict['E'][:, 2, 1, 1, :-1]
    E2 = E * E
    UE = np.sum(E2 / 2, axis=1) * mData_dict['dz'][0]
    UE_log = np.log(UE)
    UE_comp = UE / UE[0]

    EL2 = np.sum(E * E, axis=1) * mData_dict['dz'][0]
    EL2 = np.sqrt(EL2)

    try:
        peaks = find_peaks(peak_intervals, EL2, sim.dt, DH.samplePeriod)
        c1 = EL2[peaks[0]] * 1.05
        E_fit = np.polyfit(tArray[peaks], np.log(EL2[peaks]), 1)
        E_fit = np.around(E_fit, decimals=5)
        E_fit_line = c1 * np.exp(E_fit[0] * tArray[NA:NB])
        lit_line = 1.4*c1 * np.exp(gamma_lit1 * tArray[NA:NB])
        error_gamma1 = abs(gamma_lit1 - E_fit[0]) / abs(gamma_lit1)
    except Exception:
        pass

    # try:
    #     peaks2 = find_peaks(peak_intervals2, EL2, sim.dt, DH.samplePeriod)
    #     c2 = EL2[peaks2[0]] * 0.18
    #     E_fit2 = np.polyfit(tArray[peaks2], np.log(EL2[peaks2]), 1)
    #     E_fit2 = np.around(E_fit2, decimals=5)
    #     E_fit_line2 = c2 * 0.75 * np.exp(E_fit2[0] * tArray[NC:ND])
    #     lit_line2 = c2 * 0.7 * np.exp(gamma_lit2 * tArray[NC:ND])
    #     error_gamma2 = abs(gamma_lit2 - E_fit2[0]) / abs(gamma_lit2)
    # except Exception:
    #     pass

    plt.rcParams.update(plot_params)
    print("Drawing field plot...")
    fig = plt.figure(1, dpi=150)
    E_ax = fig.add_subplot(1, 1, 1)
    E_ax.plot(tArray, EL2, 'blue', label="$E$-field")

    E_ax.scatter(tArray[peaks], EL2[peaks])
    E_ax.plot(tArray[NA:NB], E_fit_line,'red', marker="s", markevery=10, label="Fitted $\gamma$")
    E_ax.plot(tArray[NA:NB], lit_line,'orange', marker="x", markevery=10, label="Literature $\gamma$")

    E_text1 = E_ax.text(.1, 0.92, '', transform=E_ax.transAxes, verticalalignment='bottom', fontsize=22)
    text1 = (r'$\gamma_{fit}$ = ' + str(E_fit[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(gamma_lit1))
    E_text1.set_text(text1)

    E_ax.set_yscale('log')
    E_ax.set_xlabel(r'$t$')
    E_ax.set_ylabel(r'$||E||_{L2}$')
    E_ax.legend()

    try:
        E_ax.scatter(tArray[peaks2], EL2[peaks2])
        E_ax.plot(tArray[NC:ND], E_fit_line2,'red', marker="s", markevery=10)
        E_ax.plot(tArray[NC:ND],'orange', marker="x", markevery=10)

        E_text2 = E_ax.text(.6, 0.02, '', transform=E_ax.transAxes, verticalalignment='bottom', fontsize=24)
        text2 = (r'$\gamma_{fit}$ = ' + str(E_fit2[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(gamma_lit2))
        E_text2.set_text(text2)
    except Exception:
        pass
    fig.savefig(data_root + 'landau_strong_field.pdf', dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                pad_inches=0.05, bbox_inches='tight')
