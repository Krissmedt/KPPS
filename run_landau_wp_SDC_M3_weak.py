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
    line.set_data(xdata,ydata[num,:])
        
    return line

def update_lines(num, xdata,ydata, lines):
    for xdat,ydat,line in zip(xdata,ydata,lines):
        line.set_data(xdat[0,:],ydat[num,:])
        
    return lines

def update_phase(num,xdata,ydata,lines,KE,dt):
    t = '%.2E' % Decimal(str(num*dt))
    text = r't = '+ t + '; KE = ' + str(KE[num])
    p_text.set_text(text)
    
    for xdat,ydat,line in zip(xdata,ydata,lines):
        line.set_data(xdat[num,:],ydat[num,:])
    
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
    
    lines = update_lines(num,xdata,ydata,lines)
    
    return lines

def update_hist(num, data, histogram_axis,bins,xmin,xmax,ymax):
    histogram_axis.cla()
    histogram_axis.hist(data[num,:],bins)
    histogram_axis.set_xlim([xmin,xmax])
    histogram_axis.set_xlabel('v')
    histogram_axis.set_ylabel(r'f')
    histogram_axis.set_ylim([0, ymax])
    t = '%.2E' % Decimal(str(num*dt))
    time = r't = '+ t
    hist_text = histogram_axis.text(.05,.95,time,
                                    transform=dist_ax.transAxes,
                                    verticalalignment='top',fontsize=14)

    return histogram_axis


def vel_dist_1d(species_list,fields,controller='',**kwargs):
    plot_res = controller.plot_res
    v_off = controller.v_off
    
    vel_data_list = [species_list[0].vel[:,2]]
    fields.v_array, fields.dist = vel_dist(vel_data_list,plot_res,-v_off,v_off)

    return species_list,fields


def plot_density_1d(species_list,fields,controller='',**kwargs):
    plot_res = controller.plot_res
    v_off = controller.v_off
    
    pos_data_list = [species_list[0].pos[:,2]]
    vel_data_list = [species_list[0].vel[:,2]]
    fields.grid_x,fields.grid_v,fields.f,fields.pn,fields.vel_dist = calc_density_mesh(pos_data_list,vel_data_list,plot_res,plot_res,v_off,L)
    
    return species_list, fields
    
    

steps = [300]
resolutions = [100]

dataRoot = "../data_landau_weak/"

L = 4*pi
tend = 30

dx_mag = 0.05
dx_mode = 0.5

v = 0
v_th = 1

dv_mag = 0
dv_mode = 0

v_off = 4
plot_res = 100

#Nq is particles per species, total nq = 2*nq
#ppc = 20
nq = 200000

#q = omega_p**2 * L / (nq*a*1)
q = L/nq
#m = 1
a = 1

omega_p = np.sqrt(q*nq*a*1/L)

prefix = 'TE'+str(tend) + '_a' + str(dx_mag)
simulate = True
plot = True

restart = False
restart_ts = 14


slow_factor = 1

############################# Linear Analysis ##################################
k = dx_mode
omega = np.sqrt(omega_p**2  +3*k**2*v_th**2)
#omega = 1.4436
vp = omega/k
omega2 = 2.8312 * k
omegap2 = np.sqrt(omega2**2 - 3*k**2*v_th**2)

df_vp = (2*np.pi)**(-1/2)*(1/v_th) * np.exp(-vp**2/(2*v_th**2)) * -vp/v_th**2

damp_rate = - (np.pi*omega_p**3)/(2*k**2) * df_vp

c1 = 2
gamma_lit = 0.1533

############################ Setup and Run ####################################
sim_params = {}
hot_params = {}
hotLoader_params = {}
mesh_params = {}
mLoader_params = {}
analysis_params = {}
data_params = {}

sim_params['t0'] = 0
sim_params['tEnd'] = tend
sim_params['percentBar'] = True
sim_params['dimensions'] = 1
sim_params['zlimits'] = [0,L]
sim_params['plot_res'] = plot_res
sim_params['v_off'] = v_off

hot_params['name'] = 'hot'
hotLoader_params['load_type'] = 'direct'
hotLoader_params['speciestoLoad'] = [0]

mesh_params['pn'] = 0
mesh_params['vel_dist'] = 0
mesh_params['grid_x'] = 0
mesh_params['grid_v'] = 0
mesh_params['f'] = 0

mLoader_params['load_type'] = 'box'
mLoader_params['store_node_pos'] = False

analysis_params['particleIntegration'] = True
analysis_params['particleIntegrator'] = 'boris_SDC'
analysis_params['M'] = 3
analysis_params['K'] = 3
analysis_params['looped_axes'] = ['z']

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

data_params['dataRootFolder'] = dataRoot
data_params['write'] = True
data_params['write_p'] = True
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
    data_params['samples'] = Nt
    for res in resolutions:
        ppc = nq/res
        #nq = ppc*res

        hot_params['nq'] = np.int(nq)
        hot_params['a'] = a
        hot_params['q'] = q
        
        pos_list = ppos_init_sin(nq,L,dx_mag,dx_mode,ftype='cos')
        hotLoader_params['pos'] = pos_list
        vel_list = particle_vel_maxwellian(hotLoader_params['pos'],v,v_th,rand_seed=1)
        hotLoader_params['vel'] = perturb_vel(pos_list,vel_list,dv_mag,dv_mode)
        
        mLoader_params['resolution'] = [2,2,res]
        mesh_params['node_charge'] = -ppc*q
        
        species_params = [hot_params]
        loader_params = [hotLoader_params]

        sim_name = 'lan_' + prefix + '_' + analysis_params['particleIntegrator']+ '_M3K3' + '_NZ' + str(res) + '_NQ' + str(int(nq)) + '_NT' + str(Nt) 
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
            dt = sim.dt
            pData_list = DH.load_p(['pos','vel','KE_sum','q'],species=['hot'],sim_name=sim_name)
            
            p1Data_dict = pData_list[0]
            
            mData_dict = DH.load_m(['phi','E','rho','q','dz','vel_dist','grid_x','grid_v','f'],sim_name=sim_name)
            
            tArray = mData_dict['t']
            Z = np.zeros((DH.samples,res+1),dtype=np.float)
            Z[:] = np.linspace(0,L,res+1)
            
            p1_data = p1Data_dict['pos'][:,:,2]      
            v1_data = p1Data_dict['vel'][:,:,2] 
            v1_max = np.max(v1_data)
            KE_data = p1Data_dict['KE_sum']
            
            rho_data = mData_dict['rho'][:,1,1,:-1]
            q_data = mData_dict['q'][:,1,1,:-1]
            
            rho_sum = np.sum(rho_data[1:-1],axis=1)
            q_sum = np.sum(q_data[1:-1],axis=1)

            phi_data = mData_dict['phi'][:,1,1,:-1]
            
            ## Growth rate phi plot setup
            tA = 0
            tB = 5
            
            NA = int(np.floor(tA/(sim.dt*DH.samplePeriod)))
            NB = int(np.floor(tB/(sim.dt*DH.samplePeriod)))
            
            E = mData_dict['E'][:,2,1,1,:-1]
            E2 = E*E
            UE =  np.sum(E2/2,axis=1)*mData_dict['dz'][0]
            UE_log = np.log(UE)
            UE_norm = UE/UE[0]
            
            EL2 = np.sum(E*E,axis=1)
            EL2 = np.sqrt(EL2)
            
            phi_min = np.min(phi_data,axis=1)
            rho_min = np.min(rho_data,axis=1)
            
            phi_mag = np.max(phi_data,axis=1) - phi_min
            rho_mag = np.max(rho_data,axis=1) - rho_min
            
            phi1 = (phi_data-phi_min[:,np.newaxis])/phi_mag[:,np.newaxis]
            rho1 = (rho_data-rho_min[:,np.newaxis])/rho_mag[:,np.newaxis]
            
            max_phi_data = np.amax(np.abs(phi_data),axis=1)
            max_phi_data_log = np.log(max_phi_data)
            max_phi_data_norm = max_phi_data/max_phi_data[0]
            
            try:
                energy_fit = np.polyfit(tArray[NA:NB],UE_log[NA:NB],1)
                energy_line = energy_fit[0]*tArray[NA:NB] + energy_fit[1]
                enorm_fit = np.polyfit(tArray[NA:NB],np.log(EL2[NA:NB]),1)
                enorm_line = enorm_fit[1]*np.exp(enorm_fit[0]*tArray[NA:NB])
                lit_line = c1*np.exp(-gamma_lit*tArray[NA:NB])
            except:
                pass
            
            vel_dist = mData_dict['vel_dist']
            
            gridx = mData_dict['grid_x'][0,:,:]
            gridv = mData_dict['grid_v'][0,:,:]
            f = mData_dict['f']
                 
#            ## Phase animation setup
#            fig = plt.figure(DH.figureNo+4,dpi=150)
#            p_ax = fig.add_subplot(1,1,1)
#            line_p1 = p_ax.plot(p1_data[0,0:1],v1_data[0,0:1],'bo',ms=2,c=(0.2,0.2,0.75,1),label='Plasma, v=0')[0]
#            p_text = p_ax.text(.05,.05,'',transform=p_ax.transAxes,verticalalignment='bottom',fontsize=14)
#            p_ax.set_xlim([0.0, L])
#            p_ax.set_xlabel('$z$')
#            p_ax.set_ylabel('$v_z$')
#            p_ax.set_ylim([-4,4])
#            p_ax.set_title('Landau phase space, Nt=' + str(Nt) +', Nz=' + str(res+1))
#            p_ax.legend()
#            
#            # Setting data/line lists:
#            pdata = [p1_data]
#            vdata = [v1_data]
#            phase_lines = [line_p1]
#            
            ## Phase density animation setup
            fig_f = plt.figure(DH.figureNo+5,dpi=150)
            f_ax = fig_f.add_subplot(1,1,1)
            cont = f_ax.contourf(gridx,gridv,f[0,:,:],cmap='inferno')
            cont.set_clim(0,np.max(f))
            cbar = plt.colorbar(cont,ax=f_ax)
            f_ax.set_xlim([0.0, L])
            f_ax.set_xlabel('$z$')
            f_ax.set_ylabel('$v_z$')
            f_ax.set_ylim([-v_off,v_off])
            f_ax.set_title('Landau density distribution, Nt=' + str(Nt) +', Nz=' + str(res+1))
            f_ax.legend()
            
            ## Field value animation setup
            fig2 = plt.figure(DH.figureNo+6,dpi=150)
            field_ax = fig2.add_subplot(1,1,1)
            rho_line = field_ax.plot(Z[0,:],rho1[0,:],label=r'charge dens. $\rho_z$')[0]
            phi_line = field_ax.plot(Z[0,:],phi1[0,:],label=r'potential $\phi_z$')[0]
            field_text = field_ax.text(.05,.05,'',transform=field_ax.transAxes,verticalalignment='bottom',fontsize=14)
            field_ax.set_xlim([0.0, L])
            field_ax.set_xlabel('$z$')
            field_ax.set_ylabel(r'$\rho_z$/$\phi_z$')
            field_ax.set_ylim([-0.2, 1.2])
            field_ax.set_title('Landau potential, Nt=' + str(Nt) +', Nz=' + str(res+1))
            field_ax.legend()
            fig2.savefig(dataRoot + sim_name + '_rho.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
            
            # Setting data/line lists:
            xdata = [Z,Z]
            ydata = [rho1,phi1]
            lines = [rho_line,phi_line]
            
            ## Velocity distribution animation setup
            fig3 = plt.figure(DH.figureNo+7,dpi=150)
            dist_ax = fig3.add_subplot(1,1,1)
            dist_line = dist_ax.plot(gridv[:,0],vel_dist[0,:])[0]
            dist_ax.set_xlim([-v_off, v_off])
            dist_ax.set_xlabel('$v_z$')
            dist_ax.set_ylabel(r'$f$')
            dist_ax.set_title('Landau velocity distribution, Nt=' + str(Nt) +', Nz=' + str(res+1))
            
            ## Perturbation plot
            fig5 = plt.figure(DH.figureNo+8,dpi=75)
            perturb_ax = fig5.add_subplot(1,1,1)
            nq = p1_data.shape[1]
            spacing = L/nq
            x0 = [(i+0.5)*spacing for i in range(0,nq)]
            xi2 = ppos_init_sin(nq,L,dx_mag,dx_mode,ftype='cos')
            perturb_ax.plot(x0,p1_data[0,:]-x0,'blue')
            perturb_ax.plot(x0,xi2[:,2]-x0,'orange')
            perturb_ax.set_xlabel('$x_{uniform}$')
            perturb_ax.set_ylabel('$x_i$')
            perturb_ax.set_ylim([-dx_mag*1.2,dx_mag*1.2])
            
            ## Electric field norm plot
            fig6 = plt.figure(DH.figureNo+9,dpi=150)
            growth_ax = fig6.add_subplot(1,1,1)
            growth_ax.plot(tArray,EL2,'blue',label="$E$")
            growth_text = growth_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
            growth_ax.set_xlabel('$t$')
            growth_ax.set_ylabel(r'$||E||_{L2}$')
            growth_ax.set_yscale('log')
            #growth_ax.set_ylim([10**-7,10**-1])
            growth_ax.set_title('Landau electric potential, Nt=' + str(Nt) +', Nz=' + str(res+1))
            growth_ax.legend()
            
            
            ## Energy plot
            fig7 = plt.figure(DH.figureNo+10,dpi=150)
            energy_ax = fig7.add_subplot(1,1,1)
            energy_ax.plot(tArray,UE_log,'blue',label="Energy growth")
            energy_text = energy_ax.text(.5,0,'',transform=dist_ax.transAxes,verticalalignment='bottom',fontsize=14)
            energy_ax.set_xlabel('$t$')
            #energy_ax.set_yscale('log')
            energy_ax.set_ylabel('$\sum E^2/2 \Delta x$')
            energy_ax.set_title('Landau energy, Nt=' + str(Nt) +', Nz=' + str(res+1))
            energy_ax.legend()
            
            try:
                growth_ax.plot(tArray[NA:NB],enorm_line,'orange',label="actual damping")
                growth_ax.plot(tArray[NA:NB],lit_line,'red',label="analytical damping")
                text = (r'$\gamma$ = ' + str(enorm_fit[0]))
                growth_text.set_text(text)
                
                energy_ax.plot(tArray[NA:NB],energy_line,'orange',label="damping rate")
                text = (r'$\gamma_E$ = ' + str(energy_fit[0]))
                energy_text.set_text(text)
            except:
                pass
                
            fps = 1/(sim.dt*DH.samplePeriod)
            #fps = 1
            # Creating the Animation object
#            phase_ani = animation.FuncAnimation(fig, update_phase, DH.samples+1, 
#                                               fargs=(pdata,vdata,phase_lines,KE_data,sim.dt),
#                                               interval=1000/fps)

            dens_dist_ani = animation.FuncAnimation(fig_f, update_contour, DH.samples+1, 
                                               fargs=(gridx,gridv,f,cont,f_ax),
                                               interval=1000/fps)
            
            
            field_ani = animation.FuncAnimation(fig2, update_field, DH.samples+1, 
                                               fargs=(xdata,ydata,lines,rho_mag,phi_mag),
                                               interval=1000/fps)
            
            vel_dist_ani = animation.FuncAnimation(fig3, update_line, DH.samples+1, 
                                               fargs=(gridv[:,0],vel_dist,dist_line),
                                               interval=1000/fps)
            
#            phase_ani.save(dataRoot +sim_name+'_phase.mp4')
            dens_dist_ani.save(dataRoot +sim_name+'_density.mp4')
            field_ani.save(dataRoot +sim_name+'_field.mp4')
            vel_dist_ani.save(dataRoot +sim_name+'_dist.mp4')
            fig6.savefig(dataRoot + sim_name + '_growth.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
            fig7.savefig(dataRoot + sim_name + '_energy.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
            plt.show()
    
    print("Setup time = " + str(sim.setupTime))
    print("Run time = " + str(sim.runTime))
