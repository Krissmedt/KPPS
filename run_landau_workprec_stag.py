from kpps import kpps as kpps_class
from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_landau1D import *
from dataHandler2 import dataHandler2
import matplotlib.animation as animation
import cmath as cm

def update_line(num, xdata,ydata, line):
    line.set_data(xdata[num,:],ydata[num,:])
        
    return line

def update_lines(num, xdata,ydata, lines):
    for xdat,ydat,line in zip(xdata,ydata,lines):
        line.set_data(xdat[num,:],ydat[num,:])
        
    return lines

def update_phase(num,xdata,ydata,lines,KE,dt):
    t = '%.2E' % Decimal(str(num*dt))
    text = r't = '+ t + '; KE = ' + str(KE[num])
    p_text.set_text(text)
    
    lines = update_lines(num,xdata,ydata,lines)
    
    return lines


def update_dist(num,xdata,ydata,lines,PE):
    for ydat in ydata:
        ymin = np.min(ydat[num,:])
        ymax = np.max(ydat[num,:])
        ydat[num,:] = ydat[num,:] - ymin
        yh = ymax-ymin + 0.0001
        ydat[num,:] = ydat[num,:]/yh
    
    mintext = '%.2E' % Decimal(str(ymin))
    maxtext = '%.2E' % Decimal(str(ymax))
    t = '%.2E' % Decimal(str(num*dt))
    text = (r't = '+ t +'; $\phi$ = [' + mintext +' : '+ maxtext + ']' 
                 + '; PE = ' + str(PE[num]))
    dist_text.set_text(text)
        
    lines = update_lines(num,xdata,ydata,lines)
    
    return lines

def update_hist(num, data, histogram_axis,bins,xmin,xmax,ymax):
    histogram_axis.cla()
    histogram_axis.hist(data[num,:],bins)
    histogram_axis.set_xlim([xmin,xmax])
    histogram_axis.set_xlabel('No.')
    histogram_axis.set_ylabel(r'f')
    histogram_axis.set_ylim([0, ymax])
    t = '%.2E' % Decimal(str(num*dt))
    time = r't = '+ t
    hist_text = histogram_axis.text(.05,.95,time,
                                    transform=dist_ax.transAxes,
                                    verticalalignment='top',fontsize=14)

    return histogram_axis

steps = [200]
resolutions = [100]

dataRoot = "../data_tsi_growth/"

L = 2*pi
tend = 20

dx_mag = 0.02
dx_mode = 1

v = 0
v_th = 0.5

dv_mag = 0
dv_mode = 1

a = -1
omega_p = 1

#Nq is particles per species, total nq = 2*nq
#ppc = 20
nq = 2000

prefix = 'TE'+str(tend)
simulate = True
plot = True

restart = False
restart_ts = 14


slow_factor = 1

############################# Linear Analysis ##################################
#k2 = dx_mode**2
#v2 = v**2
#
#roots = [None,None,None,None]
#roots[0] = cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
#roots[1] = cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
#roots[2] = -cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
#roots[3] = -cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
#
#real_slope = roots[1].imag

############################ Setup and Run ####################################
sim_params = {}
beam1_params = {}
loader1_params = {}
beam2_params = {}
loader2_params = {}
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 1
sim_params['zlimits'] = [0,L]

hot_params['name'] = 'hot'
hotLoader_params['load_type'] = 'direct'
hotLoader_params['speciestoLoad'] = [0]

mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_staggered'
analysis_params['looped_axes'] = ['z']
analysis_params['centreMass_check'] = False

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
analysis_params['custom_rho_background'] = ion_bck
analysis_params['units'] = 'custom'
analysis_params['mesh_boundary_z'] = 'open'
analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
analysis_params['hooks'] = ['kinetic_energy','field_energy']
analysis_params['rhs_check'] = True
analysis_params['pre_hook_list'] = ['ES_vel_rewind']

data_params['dataRootFolder'] = dataRoot
data_params['write'] = True
data_params['plot_limits'] = [1,1,L]

plot_params = {}
plot_params['legend.fontsize'] = 10
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 12
plot_params['axes.titlesize'] = 16
plot_params['xtick.labelsize'] = 12
plot_params['ytick.labelsize'] = 12
plot_params['lines.linewidth'] = 2
plot_params['axes.titlepad'] = 5
data_params['plot_params'] = plot_params

kppsObject = kpps_class()

for Nt in steps:
    sim_params['tSteps'] = Nt
    data_params['samples'] = 10
    for res in resolutions:
        ppc = nq/res
        #nq = ppc*res
        
        q = omega_p**2 * L / (nq*a*1)
        
        hot_params['nq'] = np.int(nq)
        hot_params['mq'] = -q
        hot_params['q'] = q
        hotLoader_params['pos'] = particle_pos_init(nq,L,dx_mag,dx_mode)
        hotLoader_params['vel'] = particle_vel_maxwellian(hotLoader_params['pos'],v,v_th)
        
        mLoader_params['resolution'] = [2,2,res]
        mesh_params['node_charge'] = -2*ppc*q
        
        species_params = [beam1_params,beam2_params]
        loader_params = [loader1_params,loader2_params]

        sim_name = 'lan_' + prefix + '_' + analysis_params['particleIntegrator'] + '_NZ' + str(res) + '_NQ' + str(int(nq)) + '_NT' + str(Nt) 
        sim_params['simID'] = sim_name
        
        ## Numerical solution ##
        model = dict(simSettings=sim_params,
                     speciesSettings=species_params,
                     pLoaderSettings=loader_params,
                     meshSettings=mesh_params,
                     analysisSettings=analysis_params,
                     mLoaderSettings=mLoader_params,
                     dataSettings=data_params)
        
        if simulate == True:
            if restart == True:
                DH = kppsObject.restart(dataRoot,sim_name,restart_ts)
                sim = DH.controller_obj
                sim_name = sim.simID
            else:
                DH = kppsObject.start(**model)
                sim = DH.controller_obj
                sim_name = sim.simID
        else:
            DH = dataHandler2(**data_params)
            sim, name = DH.load_sim(sim_name=sim_name,overwrite=True)
        
        
        ####################### Analysis and Visualisation ###########################
            if plot == True:       
                pData_list = DH.load_p(['pos','vel','KE_sum'],species=['hot',sim_name=sim_name)
                
                p1Data_dict = pData_list[0]
                
                mData_dict = DH.load_m(['phi','E','rho','dz'],sim_name=sim_name)
                
                tArray = mData_dict['t']
                Z = np.zeros((DH.samples,res+1),dtype=np.float)
                Z[:] = np.linspace(0,L,res+1)
                
                p1_data = p1Data_dict['pos'][:,:,2]      
                v1_data = p1Data_dict['vel'][:,:,2] 
                v1_max = np.max(v1_data)
                KE_data = p1Data_dict['KE_sum'] + p2Data_dict['KE_sum']
                
                rho_data = mData_dict['rho'][:,1,1,:-1]
                
                phi_data = mData_dict['phi'][:,1,1,:-1]
                
                ## Growth rate phi plot setup
                tA = 12.5
                tB = 17.5
                
                NA = int(np.floor(tA/(sim.dt*DH.samplePeriod)))
                NB = int(np.floor(tB/(sim.dt*DH.samplePeriod)))
                
                E = mData_dict['E'][:,2,1,1,:-1]
                E2 = E*E
                UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0]
                UE_log = np.log(UE)
                energy_fit = np.polyfit(tArray[NA:NB],UE_log[NA:NB],1)
                energy_line = energy_fit[0]*tArray[NA:NB] + energy_fit[1]
                
                max_phi_data = np.amax(np.abs(phi_data),axis=1)
                max_phi_data_log = np.log(max_phi_data)
                g_slope = (max_phi_data_log[2:] - max_phi_data_log[1:-1])/dt
                growth_fit = np.polyfit(tArray[NA:NB],max_phi_data_log[NA:NB],1)
                growth_line = growth_fit[0]*tArray[NA:NB] + growth_fit[1]
                
                linear_g_error = abs(real_slope - growth_fit[0])/real_slope
                
                uniform_dist = particle_pos_init(ppc,res,L,0,dx_mode)
                uni_time_evol = np.zeros((DH.samples+1,floor(nq)),dtype=np.float)
                uni_time_evol[0,:] = uniform_dist[:,2]
                for ti in range(1,DH.samples+1):
                    uni_time_evol[ti,:] = uni_time_evol[ti-1,:] + sim.dt* v1_data[0,0] 
                    
                    for pii in range(0,floor(nq)):
                        if uni_time_evol[ti,pii] < 0:
                            overshoot = 0 - uni_time_evol[ti,pii]
                            uni_time_evol[ti,pii] = L - overshoot % L
                
                        elif uni_time_evol[ti,pii] >= L:
                            overshoot = uni_time_evol[ti,pii] - L
                            uni_time_evol[ti,pii] = 0 + overshoot % L
                            
                dx_evol = p1_data - uni_time_evol

                
                
                ## Phase animation setup
                fig = plt.figure(DH.figureNo+4,dpi=150)
                p_ax = fig.add_subplot(1,1,1)
                line_p1 = p_ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo',ms=2,c=(0.2,0.2,0.75,1),label='Plasma, v=0')[0]
                p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
                p_ax.set_xlim([0.0, L])
                p_ax.set_xlabel('$z$')
                p_ax.set_ylabel('$v_z$')
                p_ax.set_ylim([-4,4])
                p_ax.set_title('Landau phase space, Nt=' + str(Nt) +', Nz=' + str(res+1))
                p_ax.legend()
                
                # Setting data/line lists:
                pdata = [p1_data]
                vdata = [v1_data]
                phase_lines = [line_p1]
                
                ## Field value animation setup
                fig2 = plt.figure(DH.figureNo+5,dpi=150)
                dist_ax = fig2.add_subplot(1,1,1)
                rho_line = dist_ax.plot(Z[0,:],rho_data[0,:],label=r'charge dens. $\rho_z$')[0]
                phi_line = dist_ax.plot(Z[0,:],phi_data[0,:],label=r'potential $\phi_z$')[0]
                dist_text = dist_ax.text(.05,.05,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
                dist_ax.set_xlim([0.0, L])
                dist_ax.set_xlabel('$z$')
                dist_ax.set_ylabel(r'$\rho_z$/$\phi_z$')
                dist_ax.set_ylim([-0.2, 1.2])
                dist_ax.set_title('Landau potential, Nt=' + str(Nt) +', Nz=' + str(res+1))
                dist_ax.legend()
                
                ## Velocity histogram animation setup
                hist_data = np.concatenate((v1_data,v2_data),axis=1)
                fig3 = plt.figure(DH.figureNo+6,dpi=150)
                hist_ax = fig3.add_subplot(1,1,1)
                hist_text = hist_ax.text(.95,.95,'',transform=dist_ax.transAxes,verticalalignment='top',fontsize=14)
                hist_ymax = ppc*res
                hist_xmax = np.max(hist_data)
                hist_xmin = np.min(hist_data)
                n_bins = 40
                vmag0 = 0.5*(abs(2*v+v_th))
                min_vel = -3*vmag0
                hist_dv = (-2*min_vel)/n_bins
                hist_bins = []
                for b in range(0,n_bins):
                    hist_bins.append(min_vel+b*hist_dv)
                
                
                ## Growth rate plot
                fig5 = plt.figure(DH.figureNo+8,dpi=150)
                growth_ax = fig5.add_subplot(1,1,1)
                growth_ax.plot(tArray,max_phi_data_log,'blue',label="$\phi$ growth")
                growth_ax.plot(tArray[NA:NB],growth_line,'orange',label="slope")
                growth_text = growth_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
                text = (r'$\gamma$ = ' + str(growth_fit[0]))
                growth_text.set_text(text)
                growth_ax.set_xlabel('$t$')
                growth_ax.set_ylabel('log $\phi_{max}$')
                #growth_ax.set_ylim([-0.001,0.001])
                growth_ax.set_title('Landau growth rate, Nt=' + str(Nt) +', Nz=' + str(res+1))
                growth_ax.legend()
                
                ## Energy plot
                fig6 = plt.figure(DH.figureNo+9,dpi=150)
                energy_ax = fig6.add_subplot(1,1,1)
                energy_ax.plot(tArray,UE_log,'blue',label="Energy growth")
                energy_ax.plot(tArray[NA:NB],energy_line,'orange',label="slope")
                energy_text = energy_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
                text = (r'$\gamma_E$ = ' + str(energy_fit[0]))
                energy_ax.set_xlabel('$t$')
                energy_ax.set_ylabel('$\sum E^2/2 \Delta x$')
                energy_ax.set_ylim([-20,1])
                energy_ax.set_title('Landau energy, Nt=' + str(Nt) +', Nz=' + str(res+1))
                energy_ax.legend()
                
                
                # Setting data/line lists:
                xdata = [Z,Z]
                ydata = [rho_data,phi_data]
                lines = [rho_line,phi_line]
    
                fps = 1/(sim.dt*DH.samplePeriod)
                # Creating the Animation object
                perturb_ani = animation.FuncAnimation(fig4, update_line, DH.samples, 
                                                   fargs=(uni_time_evol,dx_evol,line_perturb),
                                                   interval=1000/fps)
                
                phase_ani = animation.FuncAnimation(fig, update_phase, DH.samples, 
                                                   fargs=(pdata,vdata,phase_lines,KE_data,sim.dt),
                                                   interval=1000/fps)
                
                
                dist_ani = animation.FuncAnimation(fig2, update_dist, DH.samples, 
                                                   fargs=(xdata,ydata,lines),
                                                   interval=1000/fps)
                
                hist_ani = animation.FuncAnimation(fig3, update_hist, DH.samples, 
                                                   fargs=(hist_data,hist_ax,hist_bins,
                                                          hist_xmin,hist_xmax,hist_ymax),
                                                   interval=1000/fps)
                
                perturb_ani.save(sim_name+'_perturb.mp4')
                phase_ani.save(sim_name+'_phase.mp4')
                dist_ani.save(sim_name+'_dist.mp4')
                hist_ani.save(sim_name+'_hist.mp4')
                fig5.savefig(dataRoot + sim_name + '_growth.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                fig6.savefig(dataRoot + sim_name + '_energy.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                plt.show()
        
        print("Setup time = " + str(sim.setupTime))
        print("Run time = " + str(sim.runTime))