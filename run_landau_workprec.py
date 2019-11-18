from kpps import kpps
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

steps = [1]
resolutions = [26]
iterations = [3]

dataRoot = "../data_landau/"

L = 2*pi
tend = 100

dx_mag = 0.02
dx_mode = 1

v0 = 0.9
vt = 0.9

dv_mag = 2.5*10**(-4)
dv_mode = 1

qm_h = 1
qm_c = 0.01
qm_m = -1
 
omega_ph = 0.383
omega_pc = 0.924
omega_pm = 10**(-10)

nq_h = 1600 #2**14
nq_c = 26
nq_m = 30

nq = nq_h+nq_c+nq_m

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
hot_params = {}
hLoader_params = {}
cold_params = {}
cLoader_params = {}
marker_params = {}
markLoader_params = {}
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
hLoader_params['load_type'] = 'direct'
hLoader_params['speciestoLoad'] = [2]

cold_params['name'] = 'cold'
cLoader_params['load_type'] = 'direct'
cLoader_params['speciestoLoad'] = [1]

marker_params['name'] = 'marker'
markLoader_params['load_type'] = 'direct'
markLoader_params['speciestoLoad'] = [0]

mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_SDC'
analysis_params['M'] = 3
analysis_params['looped_axes'] = ['z']
analysis_params['centreMass_check'] = False

analysis_params['fieldIntegration'] = True
analysis_params['field_type'] = 'pic'
analysis_params['custom_q_background'] = ion_bck
analysis_params['units'] = 'custom'
analysis_params['mesh_boundary_z'] = 'open'
analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
analysis_params['hooks'] = ['kinetic_energy','field_energy']
analysis_params['rhs_check'] = True
analysis_params['pre_hook_list'] = []   

data_params['write'] = True
data_params['plot_limits'] = [1,1,L]
data_params['dataRootFolder'] = dataRoot

plot_params = {}
plot_params['legend.fontsize'] = 8
plot_params['figure.figsize'] = (6,4)
plot_params['axes.labelsize'] = 12
plot_params['axes.titlesize'] = 12
plot_params['xtick.labelsize'] = 8
plot_params['ytick.labelsize'] = 8
plot_params['lines.linewidth'] = 2
plot_params['axes.titlepad'] = 5
data_params['plot_params'] = plot_params

kppsObject = kpps()
for Nt in steps:
    sim_params['tSteps'] = Nt
    data_params['samples'] = Nt
    dt = tend/Nt
    for res in resolutions:
        mLoader_params['resolution'] = [2,2,res]
        for K in iterations:
            analysis_params['K'] = K
            
            h_ppc = nq_h/res
            c_ppc = nq_c/res
            m_ppc = nq_m/res
            
            q_h = -omega_ph**2 * L / (nq_h*qm_h*1)
            q_c = -omega_pc**2 * L / (nq_c*qm_c*1)
            q_m = -omega_pm**2 * L / (nq_m*qm_m*1)
            
            hot_params['nq'] = np.int(nq_h)
            hot_params['q'] = q_h
            hLoader_params['pos'] = particle_pos_init(nq_h,L,0,0)
            hLoader_params['vel'] = particle_vel_maxwellian(hLoader_params['pos'],0,vt)
            
            cold_params['nq'] = np.int(nq_c)
            cold_params['q'] = q_c
            cLoader_params['pos'] = particle_pos_init(nq_c,L,0,0)
            cLoader_params['vel'] = particle_vel_init(cLoader_params['pos'],v0,dv_mag,dv_mode)
            
            marker_params['nq'] = np.int(nq_h)
            marker_params['q'] = q_h
            markLoader_params['pos'] = particle_pos_init(nq_m,L,0,0)
            markLoader_params['vel'] = particle_vel_init(markLoader_params['pos'],v0,0,0)
            
            mesh_params['node_charge'] = -(nq_h*q_h + nq_c*q_c + nq_m*q_m)/(res+1)
            mesh_params['node_charge'] = 0
            print((nq_h*q_h + nq_c*q_c + nq_m*q_m)/(L/res))
            print(mesh_params['node_charge']*(res+1)/(L/res))
            
            species_params = [hot_params,cold_params,marker_params]
            loader_params = [hLoader_params,cLoader_params,markLoader_params]
    
            sim_name = 'landau_' + prefix + '_' + analysis_params['particleIntegrator'] + '_M3K' + str(K) + '_NZ' + str(res) + '_NQ' + str(nq) + '_NT' + str(Nt) 
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
                pData_list = DH.load_p(['pos','vel','KE_sum'],species=['hot','cold','marker'],sim_name=sim_name)
                
                mark_data_dict = pData_list[0]
                cold_data_dict = pData_list[1]
                hot_data_dict = pData_list[2]
                
                mData_dict = DH.load_m(['phi','E','rho','PE_sum','dz'],sim_name=sim_name)
                
                tArray = mData_dict['t']
                Z = np.zeros((DH.samples,res+1),dtype=np.float)
                Z[:] = np.linspace(0,L,res+1)
                
                mark_p_data = mark_data_dict['pos'][:,:,2]             
                mark_v_data = mark_data_dict['vel'][:,:,2] 
                
                cold_p_data = cold_data_dict['pos'][:,:,2]             
                cold_v_data = cold_data_dict['vel'][:,:,2] 
                
                hot_p_data = hot_data_dict['pos'][:,:,2]             
                hot_v_data = hot_data_dict['vel'][:,:,2] 
                
                KE_data = mark_data_dict['KE_sum'] + cold_data_dict['KE_sum'] + hot_data_dict['KE_sum']
                
                rho_data = mData_dict['rho'][:,1,1,:-1]
                phi_data = mData_dict['phi'][:,1,1,:-1]
                E_data = mData_dict['E'][:,2,1,1,:-1]
                
                UE = np.zeros(E_data.shape[0],dtype=np.float)

                for ti in range(0,E_data.shape[0]):
                    UE[ti] = np.sum(E_data[ti,:-1]*E_data[ti,:-1]/2) * mData_dict['dz'][0]
                
                ## Phase animation setup
                fig = plt.figure(DH.figureNo+4,dpi=150)
                p_ax = fig.add_subplot(1,1,1)
                line_mark = p_ax.plot(mark_p_data[0,0:1],mark_v_data[0,0:1],'go',label='Markers')[0]
                line_cold = p_ax.plot(cold_p_data[0,0:1],cold_v_data[0,0:1],'bo',label='Cold')[0]
                line_hot = p_ax.plot(hot_p_data[0,0:1],hot_v_data[0,0:1],'ro',label='Hot')[0]
                p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
                p_ax.set_xlim([0.0, L])
                p_ax.set_xlabel('$z$')
                p_ax.set_ylabel('$v_z$')
                p_ax.set_ylim([-2,2])
                p_ax.set_title('Landau damping phase-space, Nt=' + str(Nt) +', Nz=' + str(res+1))
                p_ax.legend()
                
                # Setting data/line lists:
                pdata = [mark_p_data,cold_p_data,hot_p_data]
                vdata = [mark_v_data,cold_v_data,hot_v_data]
                phase_lines = [line_mark,line_cold,line_hot]
                
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
                dist_ax.set_title('Landau damping potential, Nt=' + str(Nt) +', Nz=' + str(res+1))
                dist_ax.legend()
                
                ## Velocity histogram animation setup
                hist_data = np.concatenate((mark_v_data,cold_v_data),axis=1)
                fig3 = plt.figure(DH.figureNo+6,dpi=150)
                hist_ax = fig3.add_subplot(1,1,1)
                hist_text = hist_ax.text(.95,.95,'',transform=dist_ax.transAxes,verticalalignment='top',fontsize=14)
                hist_ymax = m_ppc*res + c_ppc*res
                hist_xmax = np.max(hist_data)
                hist_xmin = np.min(hist_data)
                n_bins = 40
                vmag0 = 0.5*(abs(2*v0))
                min_vel = -3*vmag0
                hist_dv = (-2*min_vel)/n_bins
                hist_bins = []
                for b in range(0,n_bins):
                    hist_bins.append(min_vel+b*hist_dv)
                
#                ## Perturbation animation setup
#                fig4 = plt.figure(DH.figureNo+7,dpi=150)
#                pert_ax = fig4.add_subplot(1,1,1)
#                line_perturb = pert_ax.plot(uni_time_evol[0,0:1],dx_evol[0,0:1],'bo')[0]
#                pert_ax.set_xlim([0.0, L])
#                pert_ax.set_xlabel('$z$')
#                pert_ax.set_ylabel('$dz$')
#                pert_ax.set_title('Two stream instability perturbation, Nt=' + str(Nt) +', Nz=' + str(res+1))
#                pert_ax.legend()
                
                
                ## Growth rate plot
                fig5 = plt.figure(DH.figureNo+8,dpi=150)
                growth_ax = fig5.add_subplot(1,1,1)
                growth_ax.plot(tArray,UE,'blue')
                growth_ax.set_xlabel('$t$')
                growth_ax.set_ylabel("$\sum E^2/2$")
                growth_ax.set_yscale('log')
                growth_ax.set_title('Landau damping field energy, Nt=' + str(Nt) +', Nz=' + str(res+1))
                growth_ax.legend()
                
                # Setting data/line lists:
                xdata = [Z,Z]
                ydata = [rho_data,phi_data]
                lines = [rho_line,phi_line]
    
                fps = 1/(sim.dt*DH.samplePeriod)
                # Creating the Animation object
#                perturb_ani = animation.FuncAnimation(fig4, update_line, DH.samples, 
#                                                   fargs=(uni_time_evol,dx_evol,line_perturb),
#                                                   interval=1000/fps)
#                
                phase_ani = animation.FuncAnimation(fig, update_phase, DH.samples, 
                                                   fargs=(pdata,vdata,phase_lines,KE_data,sim.dt),
                                                   interval=1000/fps)
                
                
                dist_ani = animation.FuncAnimation(fig2, update_dist, DH.samples, 
                                                   fargs=(xdata,ydata,lines,UE),
                                                   interval=1000/fps)
                
                hist_ani = animation.FuncAnimation(fig3, update_hist, DH.samples, 
                                                   fargs=(hist_data,hist_ax,hist_bins,
                                                          hist_xmin,hist_xmax,hist_ymax),
                                                   interval=1000/fps)
                
#                perturb_ani.save(sim_name+'_perturb.mp4')
                phase_ani.save(sim_name+'_phase.mp4')
                dist_ani.save(sim_name+'_dist.mp4')
                hist_ani.save(sim_name+'_hist.mp4')
                plt.show()
        
        print("Setup time = " + str(sim.setupTime))
        print("Run time = " + str(sim.runTime))
