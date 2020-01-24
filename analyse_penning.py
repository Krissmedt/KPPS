# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:19:24 2020

@author: Kristoffer Smedt
"""
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2
import h5py as h5


def analytical_sol(t,omegaE,omegaB,H,x0,v0,epsilon=-1):
    omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
    omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
    Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
    Rplus = x0[0,0] - Rminus
    Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
    Iplus = x0[0,1] - Iminus
    omegaTilde = sqrt(-2 * epsilon) * omegaE
    
    x = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    y = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    z = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vx = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vy = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vz = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)

    u = np.array([x,vx,y,vy,z,vz])
    energy = u.transpose() @ H @ u
    
    return u, energy 
    
    
    

analyse = True
plot = False
snapPlot = True
data_root = "../data_penning/"
plot_suffix = 'M5'


sims = {}
#sims['pen_TE16_boris_SDC_M5K1_NQ1_NT'] = [8,40,80,400,800,4000,8000]
#sims['pen_TE16_boris_SDC_M5K2_NQ1_NT'] = [8,40,80,400,800,4000,8000]
#sims['pen_TE16_boris_SDC_M5K4_NQ1_NT'] = [8,40,80,400,800,4000,8000]
#sims['pen_TE16_boris_SDC_M5K8_NQ1_NT'] = [8,40,80,400,800,4000,8000]
#sims['pen_TE16_boris_SDC_M5K16_NQ1_NT'] = [8,40,80,400,800,4000,8000]
#sims['pen_TE16_boris_synced_NQ1_NT'] = [8,40,80,400,800,4000,8000]
sims['pen_TE16_boris_SDC_M5K5_NQ1_NT'] = [4000]
nsims = len(sims)

omegaB = 25.0
omegaE = 4.9
epsilon = -1
mq = 1

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])


data_params = {}
plot_params = {}
plot_params['legend.fontsize'] = 16
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 16
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 10
data_params['plot_params'] = plot_params
data_params['dataRootFolder'] = data_root
map_end = 10

filenames = []
if analyse == True:
    sim_no = 0
    for key,value in sims.items():
        dts = []
        Nts = []
        rhs_evals = []
        xErrors = []
        hErrors = []

        filename = key[:-3] + "_wp.h5"
        filenames.append(filename)
        try:
            file = h5.File(data_root+filename,'w')
        except OSError:
            file.close()
            file = h5.File(data_root+filename,'w')
            
        grp = file.create_group('fields')
        
        for tsteps in value:
            DH = dataHandler2(**data_params)
            
            sim_no += 1
            sim_name = key + str(tsteps)
            sim, sim_name = DH.load_sim(sim_name=sim_name,overwrite=True)

            data_list = DH.load_p(['pos','vel','energy'],sim_name=sim_name)
            data_dict = data_list[0]
            
            tArray = data_dict['t']
            xArray = data_dict['pos'][:,0,0]
            yArray = data_dict['pos'][:,0,1]
            zArray = data_dict['pos'][:,0,2]
            hArray = data_dict['energy']

            xAnalyt = np.zeros(tArray.shape,dtype=np.float)
            yAnalyt = np.zeros(tArray.shape,dtype=np.float)
            zAnalyt = np.zeros(tArray.shape,dtype=np.float)
            hAnalyt = np.zeros(tArray.shape,dtype=np.float)
            for ti in range(0,tArray.shape[0]):
                u_lit, UE_lit = analytical_sol(tArray[ti],omegaE,omegaB,H,data_dict['pos'][0,:],data_dict['vel'][0,:])
                xAnalyt[ti] = u_lit[0]
                yAnalyt[ti] = u_lit[2]
                zAnalyt[ti] = u_lit[4]
                hAnalyt[ti] = UE_lit
                
            hArray[0] = hAnalyt[0]
            xRel = abs(xArray - xAnalyt)/abs(xAnalyt)
            hRel = abs(hArray - hAnalyt)/abs(hAnalyt)
            
            xErrors.append(xRel[-1])
            hErrors.append(hRel[-1])
            dts.append(sim.dt)
            Nts.append(sim.tSteps)
            rhs_evals.append(sim.rhs_eval)
            
            if snapPlot == True:
                plot_params['legend.loc'] = 'upper right'
                plt.rcParams.update(plot_params)
                limits = np.array(DH.plot_limits)
                
                DH.figureNo += 1
                fig = plt.figure(DH.figureNo)
                ax = fig.gca(projection='3d')
                plabel = "Boris-SDC, M5K5"
                ax.plot3D(data_dict['pos'][:,0,0],
                          data_dict['pos'][:,0,1],
                          zs=data_dict['pos'][:,0,2],label=plabel)
                
                ax.plot3D(xAnalyt,
                          yAnalyt,
                          zs=zAnalyt,ls='dashed',label='Exact Trajectory')
                    
                ax.set_xlim([limits[0,0], limits[0,1]])
                ax.set_ylim([limits[1,0], limits[1,1]])
                ax.set_zlim([limits[2,0], limits[2,1]])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend()
                
                DH.figureNo += 1
                fig_H = plt.figure(DH.figureNo)
                ax_H = fig_H.add_subplot(1, 1, 1)
                ax_H.plot(tArray,hRel,label=sim_name)
                #ax_H.set_yscale('log')
                ax_H.set_xlabel('$t$')
                ax_H.set_ylabel('$\Delta E_{rel}$')
                
                fig.savefig(data_root + key + str(tsteps) + '_trajectory.svg', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                fig_H.savefig(data_root + key + str(tsteps) + '_energy.svg', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
    
        file.attrs["integrator"] = sim.analysisSettings['particleIntegrator']
        try:
            file.attrs["M"] = key[key.find('M')+1]
            file.attrs["K"] = key[key.find('K')+1]
        except KeyError:
            pass
        
        grp.create_dataset('dts',data=dts)
        grp.create_dataset('Nts',data=Nts)
        grp.create_dataset('rhs_evals',data=rhs_evals)
        grp.create_dataset('xErrors',data=np.array(xErrors))
        grp.create_dataset('hErrors',data=np.array(hErrors))
        file.close()
                
            
if plot == True:
    if len(filenames) == 0:
        for key, value in sims.items():
            filename = key[:-3] + "_wp.h5"
            filenames.append(filename)
    
    
    cmap = DH.get_cmap(map_end)
    i = 0
    for filename in filenames:
        file = h5.File(data_root+filename,'r')
        dts = file["fields/dts"][:]
        rhs_evals = file["fields/rhs_evals"][:]
        xErrors = file["fields/xErrors"][:]
        hErrors = file["fields/hErrors"][:]

        if file.attrs["integrator"] == "boris_staggered":
            label = "Boris Staggered"
            label = "Boris"
            color = 'black'
        elif file.attrs["integrator"] == "boris_synced":
            label = "Boris"
            color = 'black'
        elif file.attrs["integrator"] == "boris_SDC":
            label = "Boris-SDC"
            label += ", M=" + file.attrs["M"] + ", K=" + file.attrs["K"]
            color = cmap(i)
        
        dts = np.array(dts)

        ##Order Plot w/ rhs
        fig_rhs = plt.figure(DH.figureNo+2)
        ax_rhs = fig_rhs.add_subplot(1, 1, 1)
        ax_rhs.plot(rhs_evals,xErrors,c=color,label=label)
        ax_rhs.set_xlim(10**(3),10**6)
        ax_rhs.set_ylim(10**(-5),10**1)
        plot_params['legend.loc'] = 'upper right'
        plt.rcParams.update(plot_params)
        ax_rhs.legend()
        
        ##Order Plot w/ dt
        fig_dt = plt.figure(DH.figureNo+3)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dts*omegaB,xErrors,c=color,label=label)
        ax_dt.set_xlim(10**(-1),10**1)
        ax_dt.set_ylim(10**(-5),10**1)
        plot_params['legend.loc'] = 'lower right'
        plt.rcParams.update(plot_params)
        ax_dt.legend()
        i += 1
        
        file.close()
        
    ax_list = []  
    ax_list.append(ax_rhs)
    ax_list.append(ax_dt)
    
    i = 0
    for ax in ax_list:
        i +=1
        if i == 1:
            orderSlope = -1
            ax.set_xlabel('Number of RHS evaluations')
        else:
            ax.set_xlabel(r'$\omega_B \cdot \Delta t $')
            orderSlope = 1
        
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylabel('$\Delta x^{(rel)}$')
        
        ax.set_title(plot_suffix)
        
        xRange = ax.get_xlim()
        yRange = ax.get_ylim()
        
        ax.plot(xRange,DH.orderLines(2*orderSlope,xRange,yRange),
                    ls='dotted',c='0.25',label='2nd Order')
        ax.plot(xRange,DH.orderLines(4*orderSlope,xRange,yRange),
                    ls='dashed',c='0.75',label='4th Order')
        ax.plot(xRange,DH.orderLines(8*orderSlope,xRange,yRange),
                    ls='dashdot',c='0.5',label='8th Order')
        
        
        
        fig_rhs.savefig(data_root + 'penning_M' + plot_suffix + '_wp_rhs.svg', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
        fig_dt.savefig(data_root + 'penning_M' + plot_suffix + '_wp_dt.svg', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
