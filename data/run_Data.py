from kpps import kpps
from dataHandler import dataHandler
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

schemes = {'lobatto':'boris_SDC'}
M = 3
K = [1,2,4,8]
dt = []
energy_data = True
omegaB = 25.0


params = {'legend.fontsize': 12,
  'figure.figsize': (12, 8),
  'axes.labelsize': 20,
  'axes.titlesize': 20,
  'xtick.labelsize': 16,
  'ytick.labelsize': 16,
  'lines.linewidth': 3
  }
plt.rcParams.update(params)

if energy_data == True:
    data = dataHandler()
    data.loadData("exactPenning_262136.txt",['exact_h_'],columns=[7])
    
    for key, value in schemes.items():
        for k in K:
            filename = key + "_" + value + "_"  + str(M) + "_" + str(k)
            data.loadData(filename,['t','h'])
            data.tArray = data.tArray * omegaB

            energyError = abs(data.hArray-data.exact_h_Array)/data.exact_h_Array
            label = key + "-" + value + ", M=" + str(M) + ", K=" + str(k)

            ##Energy Plot
            h_fig = plt.figure(1)
            h_ax = h_fig.add_subplot(1, 1, 1)
            h_ax.scatter(data.tArray,energyError,label=label)
            


## energy plot finish
h_ax.set_xscale('log')
h_ax.set_xlim(10**2,data.tArray[-1])
#h_ax.set_xlabel('$\omega_b t$')
h_ax.set_xlabel('$t$')

h_ax.set_yscale('log')
h_ax.set_ylim(10**-11,10**0)
h_ax.set_ylabel('$\Delta E^{rel}$')
h_ax.legend()