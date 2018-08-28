from kpps import kpps
from dataHandler import dataHandler
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

schemes = {'lobatto':'boris_SDC'}
M = 3
K = [1,2,4,8]
dt = []
energy_data = True
omegaB = 25.0

tstart = 0
tEnd = 26000
samples = 1001
tArray = omegaB * np.linspace(tstart,tEnd,samples)


if energy_data == True:
    data = dataHandler()
    data.tArray = tArray
    data.loadData('h_exact_lobatto_boris_SDC_3_1','exact_h_')
    
    for key, value in schemes.items():
        for k in K:
            filename = key + "_" + value + "_"  + str(M) + "_" + str(k)
            data.loadData(filename,'h')
            
            data.exact_h_Array = np.array(data.exact_h_Array)
            energyError = abs(data.hArray-data.exact_h_Array)/data.exact_h_Array
            label = key + "-" + value + ", M=" + str(M) + ", K=" + str(k)
            
            ##Energy Plot
            h_fig = plt.figure(1)
            h_ax = h_fig.add_subplot(1, 1, 1)
            h_ax.scatter(tArray[1:],energyError[1:],label=label)
            
            
## energy plot finish
h_ax.set_xscale('log')
h_ax.set_xlim(10**3,10**6)
h_ax.set_xlabel('$\omega_b t$')

h_ax.set_yscale('log')
h_ax.set_ylim(10**-12,10**6)
h_ax.set_ylabel('$\Delta E^{rel}$')
h_ax.legend()