from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from caseFile_twoStream1D import *
from dataHandler2 import dataHandler2
import matplotlib.animation as animation
import cmath as cm

def update_line(num, xdata,ydata, line):
    line.set_data(xdata,ydata[num,:])
        
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

def update_contour(num, xgrid,ygrid,fgrid, contour_obj,f_ax):
    for c in contour_obj.collections:
        contour_obj.collections.remove(c)
        
    contour_obj.collections = []
        
    contour_obj = f_ax.contourf(xgrid,ygrid,fgrid[num,:,:],cmap='inferno')
        
    return contour_obj


def update_field(num,xdata,ydata,lines,rho_mag,phi_mag):

    phitext = '%.2E' % Decimal(str(phi_mag[num]))
    rhotext = '%.2E' % Decimal(str(rho_mag[num]))
    t = '%.2E' % Decimal(str((num)*dt))
    text = (r't = '+ t +r'; $\phi$ = ' + phitext + r'; $\rho$ = ' + rhotext)
    field_text.set_text(text)
    print(xdata[1].shape)
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

def plot_density_1d(species_list,fields,controller='',**kwargs):
    plot_res = controller.plot_res
    v_off = controller.v_off
    
    pos_data_list = [species_list[0].pos[:,2]]
    vel_data_list = [species_list[0].vel[:,2]]
    pos_data_list.append(species_list[1].pos[:,2])
    vel_data_list.append(species_list[1].vel[:,2])
    fields.grid_x,fields.grid_v,fields.f,fields.pn,fields.vel_dist = calc_density_mesh(pos_data_list,vel_data_list,plot_res,plot_res,v_off,L)
    
    return species_list, fields


steps = [300]
resolutions = [100]
iterations = [1]

dataRoot = "../data_tsi_weak/"

L = 2*pi
tend = 30

dx_mag = 1e-4
dx_mode = 1

v = 1
dv_mag = 0
dv_mode = 1

a = -1
omega_p = 1

#Nq is particles per species, total nq = 2*nq
#ppc = 20
nq = 2000

prefix = 'TE'+str(tend) + '_a' + str(dx_mag)
simulate = False
plot = True

restart = False
restart_ts = 14

slow_factor = 1
############################ Linear Analysis ##################################
k2 = dx_mode**2
v2 = v**2

roots = [None,None,None,None]
roots[0] = cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[1] = cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[2] = -cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[3] = -cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))

real_slope = roots[1].imag

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
sim_params['plot_res'] = 100
sim_params['v_off'] = 4

beam1_params['name'] = 'beam1'
loader1_params['load_type'] = 'direct'
loader1_params['speciestoLoad'] = [0]

beam2_params['name'] = 'beam2'
loader2_params['load_type'] = 'direct'
loader2_params['speciestoLoad'] = [1]

mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False

mesh_params['pn'] = 0
mesh_params['vel_dist'] = 0
mesh_params['grid_x'] = 0
mesh_params['grid_v'] = 0
mesh_params['f'] = 0

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

if plot == True:
    analysis_params['hooks'].append(plot_density_1d)
    analysis_params['pre_hook_list'].append(plot_density_1d)

data_params['write'] = True
data_params['plot_limits'] = [1,1,L]
data_params['dataRootFolder'] = dataRoot

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


kppsObject = kpps()
for Nt in steps:
    sim_params['tSteps'] = Nt
    data_params['samples'] = Nt
    dt = tend/Nt
    for res in resolutions:
        mLoader_params['resolution'] = [2,2,res]
        for K in iterations:
            analysis_params['K'] = K
            ppc = nq/res
            #nq = ppc*res
            
            q = omega_p**2 * L / (nq*a*1)
            
            beam1_params['nq'] = np.int(nq)
            beam1_params['mq'] = -q
            beam1_params['q'] = q
            loader1_params['pos'] = ppos_init_sin(nq,L,dx_mag,dx_mode,ftype='cos')
            loader1_params['vel'] = particle_vel_init(loader1_params['pos'],v,dv_mag,dv_mode)
            
            beam2_params['nq'] = np.int(nq)
            beam2_params['mq'] = -q
            beam2_params['q'] = q
            loader2_params['pos'] = ppos_init_sin(nq,L,-dx_mag,dx_mode,ftype='cos')
            loader2_params['vel'] = particle_vel_init(loader2_params['pos'],-v,dv_mag,dv_mode)
            
            mesh_params['node_charge'] = -2*ppc*q
            
            species_params = [beam1_params,beam2_params]
            loader_params = [loader1_params,loader2_params]
    
            sim_name = 'tsi_' + prefix + '_' + analysis_params['particleIntegrator'] + '_M3K' + str(K) + '_NZ' + str(res) + '_NQ' + str(nq) + '_NT' + str(Nt) 
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
                print("Loading data...")
                pData_list = DH.load_p(['pos','vel','KE_sum'],species=['beam1','beam2'],sim_name=sim_name)
                
                p1Data_dict = pData_list[0]
                p2Data_dict = pData_list[1]
                
                mData_dict = DH.load_m(['phi','E','rho','q','dz','vel_dist','grid_x','grid_v','f'],sim_name=sim_name)
                
                print("Analysing data...")
                tArray = mData_dict['t']
                Z = np.zeros((DH.samples+1,res+1),dtype=np.float)
                Z[:] = np.linspace(0,L,res+1)
                
                p1_data = p1Data_dict['pos'][:,:,2]
                p2_data = p2Data_dict['pos'][:,:,2]
                
                v1_data = p1Data_dict['vel'][:,:,2] 
                v2_data = p2Data_dict['vel'][:,:,2] 
                
                v1_max = np.max(v1_data)
                v2_min = np.min(v2_data)
                KE_data = p1Data_dict['KE_sum'] + p2Data_dict['KE_sum']
                
                phi_data = mData_dict['phi'][:,1,1,:-1]
                rho_data = mData_dict['rho'][:,1,1,:-1]
                q_data = mData_dict['q'][:,1,1,:-1]
                
                rho_sum = np.sum(rho_data[1:-1],axis=1)
                q_sum = np.sum(q_data[1:-1],axis=1)
                
                phi_min = np.min(phi_data,axis=1)
                rho_min = np.min(rho_data,axis=1)
                
                phi_mag = np.max(phi_data,axis=1) - phi_min
                rho_mag = np.max(rho_data,axis=1) - rho_min
                
                phi1 = (phi_data-phi_min[:,np.newaxis])/phi_mag[:,np.newaxis]
                rho1 = (rho_data-rho_min[:,np.newaxis])/rho_mag[:,np.newaxis]

                ## Growth rate phi plot setup
                tA = 12.5
                tB = 17.5
                
                NA = int(np.floor(tA/(sim.dt*DH.samplePeriod)))
                NB = int(np.floor(tB/(sim.dt*DH.samplePeriod)))
                
                E = mData_dict['E'][:,2,1,1,:-1]
                E2 = E*E
                EL2 = np.sum(E*E,axis=1)
                EL2 = np.sqrt(EL2*mData_dict['dz'][0])

                UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0]
                UE_log = np.log(UE)
                
                c1 = 10e-4
                E_fit = np.polyfit(tArray[NA:NB],np.log(EL2[NA:NB]),1)
                E_line = c1*np.exp(E_fit[0]*tArray[NA:NB])
                
                max_phi_data = np.amax(np.abs(phi_data),axis=1)
                max_phi_data_log = np.log(max_phi_data)
                    
                growth_fit = np.polyfit(tArray[NA:NB],max_phi_data_log[NA:NB],1)
                growth_line = growth_fit[0]*tArray[NA:NB] + growth_fit[1]
                
                exact_line = real_slope*tArray[NA:NB] + growth_fit[1]
                
                linear_g_error = abs(real_slope - growth_fit[0])/real_slope
                
                vel_dist = mData_dict['vel_dist']
            
                v_off = sim_params['v_off']
                gridx = mData_dict['grid_x'][0,:,:]
                gridv = mData_dict['grid_v'][0,:,:]
                f = mData_dict['f']

                
                print("Setting up plots...")
                # Phase animation setup
                fig = plt.figure(DH.figureNo+4,dpi=150)
                p_ax = fig.add_subplot(1,1,1)
                line_p1 = p_ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo',ms=2,c=(0.2,0.2,0.75,1),label='Beam 1, v=1')[0]
                line_p2 = p_ax.plot(p2_data[0,0:1],v2_data[0,0:1],'ro',ms=2,c=(0.75,0.2,0.2,1),label='Beam 2, v=-1')[0]
                p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
                p_ax.set_xlim([0.0, L])
                p_ax.set_xlabel('$z$')
                p_ax.set_ylabel('$v_z$')
                p_ax.set_ylim([-v_off,v_off])
                p_ax.set_title('Two stream instability phase space, Nt=' + str(Nt) +', Nz=' + str(res+1))
                p_ax.legend()
                
                # Setting data/line lists:
                pdata = [p1_data,p2_data]
                vdata = [v1_data,v2_data]
                phase_lines = [line_p1,line_p2]


                ## Field value animation setup
                fig_field = plt.figure(DH.figureNo+5,dpi=150)
                field_ax = fig_field.add_subplot(1,1,1)
                rho_line = field_ax.plot(Z[0,:],rho1[0,:],label=r'charge dens. $\rho_z$')[0]
                phi_line = field_ax.plot(Z[0,:],phi1[0,:],label=r'potential $\phi_z$')[0]
                field_text = field_ax.text(.05,.05,'',transform=field_ax.transAxes,verticalalignment='bottom',fontsize=14)
                field_ax.set_xlim([0.0, L])
                field_ax.set_xlabel('$z$')
                field_ax.set_ylabel(r'$\rho_z$/$\phi_z$')
                field_ax.set_ylim([-0.2, 1.2])
                field_ax.set_title('Two-stream instability potential, Nt=' + str(Nt) +', Nz=' + str(res+1))
                field_ax.legend()
                fig_field.savefig(dataRoot + sim_name + '_rho.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                
                ## Velocity histogram animation setup
                hist_data = np.concatenate((v1_data,v2_data),axis=1)
                fig_hist = plt.figure(DH.figureNo+6,dpi=150)
                hist_ax = fig_hist.add_subplot(1,1,1)
                hist_text = hist_ax.text(.95,.95,'',transform=hist_ax.transAxes,verticalalignment='top',fontsize=14)
                hist_ymax = ppc*res
                hist_xmax = np.max(hist_data)
                hist_xmin = np.min(hist_data)
                n_bins = 40
                vmag0 = 0.5*(abs(2*v))
                min_vel = -3*vmag0
                hist_dv = (-2*min_vel)/n_bins
                hist_bins = []
                for b in range(0,n_bins):
                    hist_bins.append(min_vel+b*hist_dv)
                    
                ## Velocity distribution animation setup
                fig_dist = plt.figure(DH.figureNo+7,dpi=150)
                dist_ax = fig_dist.add_subplot(1,1,1)
                dist_line = dist_ax.plot(gridv[:,0],vel_dist[0,:])[0]
                dist_ax.set_xlim([-v_off, v_off])
                dist_ax.set_xlabel('$v_z$')
                dist_ax.set_ylabel(r'$f$')
                dist_ax.set_title('Two-stream velocity distribution, Nt=' + str(Nt) +', Nz=' + str(res+1))
                
                ## Phase density animation setup
                fig_f = plt.figure(DH.figureNo+8,dpi=150)
                f_ax = fig_f.add_subplot(1,1,1)
                cont = f_ax.contourf(gridx,gridv,f[0,:,:],cmap='inferno')
                cont.set_clim(0,np.max(f))
                cbar = plt.colorbar(cont,ax=f_ax)
                f_ax.set_xlim([0.0, L])
                f_ax.set_xlabel('$z$')
                f_ax.set_ylabel('$v_z$')
                f_ax.set_ylim([-v_off,v_off])
                f_ax.set_title('Two-stream density distribution, Nt=' + str(Nt) +', Nz=' + str(res+1))
                f_ax.legend()
                
                
                ## Growth rate plot
                fig_gr = plt.figure(DH.figureNo+9,dpi=150)
                growth_ax = fig_gr.add_subplot(1,1,1)
                growth_ax.plot(tArray,max_phi_data_log,'blue',label="$\phi$ growth")
                growth_ax.plot(tArray[NA:NB],growth_line,'orange',label="slope")
                growth_text = growth_ax.text(.5,0,'',transform=growth_ax.transAxes,verticalalignment='bottom',fontsize=14)
                text = (r'$\gamma$ = ' + str(growth_fit[0]))
                growth_text.set_text(text)
                growth_ax.set_xlabel('$t$')
                growth_ax.set_ylabel('log $\phi_{max}$')
                #growth_ax.set_ylim([-0.001,0.001])
                growth_ax.set_title('Two stream instability growth rate, Nt=' + str(Nt) +', Nz=' + str(res+1))
                growth_ax.legend()
                
                fig_el2 = plt.figure(DH.figureNo+10,dpi=150)
                el2_ax = fig_el2.add_subplot(1,1,1)
                el2_ax.plot(tArray,EL2,'blue',label="$E$")
                el2_ax.plot(tArray[NA:NB],E_line,'orange',label="slope")
                el2_text = el2_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
                el2_ax.set_xlabel('$t$')
                el2_ax.set_ylabel(r'log $||E||_{L2}$')
                el2_ax.set_yscale('log')
                #el2_ax.set_ylim([10**-7,10**-1])
                el2_ax.set_title('Two-stream E-field L2 norm, Nt=' + str(Nt) +', Nz=' + str(res+1))
                el2_ax.legend()
                
                ## Energy plot
                fig_UE = plt.figure(DH.figureNo+11,dpi=150)
                energy_ax = fig_UE.add_subplot(1,1,1)
                energy_ax.plot(tArray,UE_log,'blue')
                energy_text = energy_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
                energy_ax.set_xlabel('$t$')
                energy_ax.set_ylabel('$\sum E^2/2 \Delta x$')
                #energy_ax.set_ylim([-20,1])
                energy_ax.set_title('Two stream instability energy, Nt=' + str(Nt) +', Nz=' + str(res+1))
                energy_ax.legend()
                
                print("Drawing animations...")
                # Setting data/line lists:
                xdata = [Z,Z]
                ydata = [rho1,phi1]
                lines = [rho_line,phi_line]
    
                fps = 1/(sim.dt*DH.samplePeriod)
                
                # Creating the Animation object
                phase_ani = animation.FuncAnimation(fig, update_phase, DH.samples+1, 
                                                   fargs=(pdata,vdata,phase_lines,KE_data,sim.dt),
                                                   interval=1000/fps)
                
                
                hist_ani = animation.FuncAnimation(fig_hist, update_hist, DH.samples+1, 
                                                   fargs=(hist_data,hist_ax,hist_bins,
                                                          hist_xmin,hist_xmax,hist_ymax),
                                                   interval=1000/fps)
                                                   
                dens_dist_ani = animation.FuncAnimation(fig_f, update_contour, DH.samples+1, 
                                                   fargs=(gridx,gridv,f,cont,f_ax),
                                                   interval=1000/fps)
#                
                
                field_ani = animation.FuncAnimation(fig_field, update_field, DH.samples+1, 
                                                   fargs=(xdata,ydata,lines,rho_mag,phi_mag),
                                                   interval=1000/fps)
                
                vel_dist_ani = animation.FuncAnimation(fig_dist, update_line, DH.samples+1, 
                                                   fargs=(gridv[:,0],vel_dist,dist_line),
                                                   interval=1000/fps)
                
                dens_dist_ani.save(dataRoot + sim_name+'_dens.mp4')
                vel_dist_ani.save(dataRoot + sim_name+'_veldist.mp4')
                phase_ani.save(dataRoot + sim_name+'_phase.mp4')
                field_ani.save(dataRoot + sim_name+'_field.mp4')
                hist_ani.save(dataRoot + sim_name+'_hist.mp4')
                fig_gr.savefig(dataRoot + sim_name + '_growth.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                fig_el2.savefig(dataRoot + sim_name + '_el2.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                fig_UE.savefig(dataRoot + sim_name + '_energy.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
                plt.show()
#        
        print("Setup time = " + str(sim.setupTime))
        print("Run time = " + str(sim.runTime))
