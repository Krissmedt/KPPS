from kpps import kpps
from dataHandler import dataHandler
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def orderLines(order,xRange,yRange):
    if order < 0:
        a = yRange[1]/xRange[0]**order
    else:
        a = yRange[0]/xRange[0]**order    
    
    oLine = [a*xRange[0]**order,a*xRange[1]**order]
        
    return oLine


schemes = {'lobatto':'boris_SDC'}
M = 3
K = [1,2,4,8]
dt = []
order_data = True
energy_data = False
omegaB = 25.0
orderEntry = [10**5,10**-5]

params = {'legend.fontsize': 12,
  'figure.figsize': (12, 8),
  'axes.labelsize': 20,
  'axes.titlesize': 20,
  'xtick.labelsize': 16,
  'ytick.labelsize': 16,
  'lines.linewidth': 3
  }
plt.rcParams.update(params)

if order_data == True:
    data = dataHandler()
    for key, value in schemes.items():
        for k in K:
            filename = key + "_" + value + "_"  + str(M) + "_" + str(k) + "_" + "winkel"
            data.loadData(filename,['dt','rhs','xRel'])

            
            label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(k)
            
            
            ##Order Plot w/ rhs
            fig_rhs = plt.figure(50)
            ax_rhs = fig_rhs.add_subplot(1, 1, 1)
            ax_rhs.plot(data.rhsArray,data.xRelArray,label=label_order)
            
            ##Order Plot w/ dt
            fig_dt = plt.figure(51)
            ax_dt = fig_dt.add_subplot(1, 1, 1)
            ax_dt.plot(data.dtArray*omegaB,data.xRelArray,label=label_order)
       

    ## Order plot finish
    ax_rhs.set_xscale('log')
    ax_rhs.set_xlim(10**3,5*10**5)
    ax_rhs.set_xlabel('Number of RHS evaluations')
    ax_rhs.set_yscale('log')
    ax_rhs.set_ylim(10**(-5),10**1)
    ax_rhs.set_ylabel('$\Delta x^{(rel)}$')
    
    xRange = ax_rhs.get_xlim()
    yRange = ax_rhs.get_ylim()

    ax_rhs.plot(xRange,orderLines(-2,xRange,yRange),ls='dotted',c='0.25',label='2nd Order')
    ax_rhs.plot(xRange,orderLines(-4,xRange,yRange),ls='dashed',c='0.75',label='4th Order')
    ax_rhs.plot(xRange,orderLines(-8,xRange,yRange),ls='dashdot',c='0.1',label='8th Order')
    ax_rhs.legend()
    
    
    ## Order plot finish
    ax_dt.set_xscale('log')
    ax_dt.set_xlim(10**-1,10**1)
    ax_dt.set_xlabel('$\omega_B \Delta t$')
    ax_dt.set_yscale('log')
    ax_dt.set_ylim(10**(-5),10**1)
    ax_dt.set_ylabel('$\Delta x^{(rel)}$')
    
    xRange = ax_dt.get_xlim()
    yRange = ax_dt.get_ylim()

    ax_dt.plot(xRange,orderLines(2,xRange,yRange),ls='dotted',c='0.25',label='2nd Order')
    ax_dt.plot(xRange,orderLines(4,xRange,yRange),ls='dashed',c='0.75',label='4th Order')
    ax_dt.plot(xRange,orderLines(8,xRange,yRange),ls='dashdot',c='0.1',label='8th Order')
    ax_dt.legend()


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