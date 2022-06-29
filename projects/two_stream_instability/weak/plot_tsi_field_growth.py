import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from kpps.output.data_handler import DataHandler
import h5py as h5
from collections import OrderedDict

fit_start = 10
fit_stop = 16

data_root = "/home/krissmedt/data/tsi/weak/"

sims = [
    "tsi_TE50_a0.0001_boris_SDC_M3K4_NZ100_NQ20000_NT500"
]



################################ Linear analysis ##############################
omega_p = 1

k = 1
v = 1

k2 = k**2
v2 = v**2

roots = [None,None,None,None]
roots[0] = cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[1] = cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[2] = -cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[3] = -cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))

real_slope = roots[1].imag

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

    NA = int(fit_start/(sim.dt*DH.samplePeriod))
    NB = int(fit_stop/(sim.dt*DH.samplePeriod))

    mData_dict = DH.load_m(['phi','E','rho','PE_sum','zres','dz','Rx','Rv'],sim_name=sim_name,max_t='all')
    tArray = mData_dict['t']

    phi_data = mData_dict['phi'][:,1,1,:-1]
    E = mData_dict['E'][:,2,1,1,:-1]
    E2 = E*E
    UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0]
    UE_log = np.log(UE)
    UE_comp = UE/UE[0]

    EL2 = np.sum(E*E,axis=1) * mData_dict['dz'][0]
    EL2 = np.sqrt(EL2)

    ## Growth rate phi plot setup
    max_phi_data = np.amax(np.abs(phi_data),axis=1)
    max_phi_data_log = np.log(max_phi_data)
    try:
        c1 = EL2[NA]*0.008
        E_fit = np.around(np.polyfit(tArray[NA:NB],np.log(EL2[NA:NB]),1),decimals=4)
        E_line = c1*np.exp(tArray[NA:NB]*E_fit[0])

        real_slope = np.around(real_slope,decimals=4)
        lit_line = c1*np.exp(tArray[NA:NB]*real_slope)

        error_linear = abs(real_slope - E_fit[0])/real_slope
    except:
        pass

    print("Drawing field plot...")
    fig_el2 = plt.figure(1,dpi=150)
    el2_ax = fig_el2.add_subplot(1,1,1)
    el2_ax.plot(tArray[1:],EL2[1:],'blue',label="$E$")
    el2_ax.plot(tArray[NA:NB],E_line,'red',label="Fitted $\gamma$")
    el2_ax.plot(tArray[NA:NB],lit_line,'orange',label="Literature $\gamma$")
    E_text1 = el2_ax.text(.1,0.02,'',transform=el2_ax.transAxes,verticalalignment='bottom',fontsize=16)
    text1 = (r'$\gamma_{fit}$ = ' + str(E_fit[0]) + ' vs. ' + r'$\gamma_{lit}$ = ' + str(real_slope))
    E_text1.set_text(text1)
    el2_ax.set_xlabel('$t$')
    el2_ax.set_ylabel(r'$||E||_{L2}$')
    el2_ax.set_yscale('log')
    el2_ax.set_xlim(0,50)
#   el2_ax.set_ylim(10**(-4),3)
    el2_ax.legend()
    fig_el2.savefig(data_root + sim_name + '_eField.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
