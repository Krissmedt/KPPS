import numpy as np
import scipy as sp
import copy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from math import floor
from species import species
from mesh import mesh
from caseHandler import caseHandler
from simulationManager import simulationManager
from kpps_analysis import kpps_analysis
from kpps import kpps

class Test_fields:
    def setup(self):
        case_params = {}
        
        case_params['dimensions'] = 3
        case_params['particle_init'] = 'none'
        case_params['dx'] = 0.1
        case_params['dv'] = 0
        case_params['pos'] = np.array([[10,0,0],[-10,0,0]])
        case_params['vel'] = np.array([[0,0,0],[0,0,0]])
        
        case_params['mesh_init'] = 'box'
        case_params['xlimits'] = [0,1]
        case_params['ylimits'] = [0,1]
        case_params['zlimits'] = [0,1]
        case_params['resolution'] = [5,5,5]
        case_params['store_node_pos'] = True
        
        plot_params = {}
        plot_params['legend.fontsize'] = 12
        plot_params['figure.figsize'] = (12,8)
        plot_params['axes.labelsize'] = 20
        plot_params['axes.titlesize'] = 20
        plot_params['xtick.labelsize'] = 16
        plot_params['ytick.labelsize'] = 16
        plot_params['lines.linewidth'] = 3
        plot_params['axes.titlepad'] = 10
        
        data_params = {}
        data_params['plot_params'] = plot_params
        
        self.case_params = case_params
        self.data_params = data_params
        
    def test_laplacian_1d(self):
        sim = simulationManager()
        p = species()
        kpa = kpps_analysis()
        
        m = mesh()
        case = caseHandler(mesh=m,**self.case_params)
        sim.ndim = 1
        
        z = m.pos[2,0,0,:]
        Lz = m.zlimits[1]
        phi = z*(z-Lz)
        phi = phi[np.newaxis,np.newaxis,:] 
        phi_D_vector = kpa.meshtoVector(phi)
        Dk = kpa.poisson_cube2nd_setup(p,m,sim).toarray()
        b = np.dot(Dk,phi_D_vector)
        
        bSum = (m.res[2]-1)*2
        assert np.allclose(b[1:-1],np.ones(m.res[2]-1)*2)
        assert bSum-0.001 <= np.sum(b[1:-1]) <= bSum+0.001
        return b, phi
    
    def test_laplacian_2d(self):
        sim = simulationManager()
        p = species()
        kpa = kpps_analysis()
        
        m = mesh()
        case = caseHandler(mesh=m,**self.case_params)
        sim.ndim = 2
        
        z = m.pos[2,0,0,:]
        Lz = m.zlimits[1]
        y = m.pos[1,0,:,0]
        Ly = m.ylimits[1]
        
        
        phi = np.zeros((m.res[1:3]+1),dtype=np.float)
        sol = np.zeros((m.res[1:3]+1),dtype=np.float)
        for zi in range(0,m.res[2]+1):
            for yi in range(0,m.res[1]+1):
                phi[yi,zi] = z[zi]*(z[zi]-Lz)*y[yi]*(y[yi]-Ly)
                sol[yi,zi] = 2*z[zi]*(z[zi]-Lz)+2*y[yi]*(y[yi]-Ly)
                
        phi = phi[np.newaxis,:,:] 
        phi_E_vector = kpa.meshtoVector(phi)
        
        Ek = kpa.poisson_cube2nd_setup(p,m,sim).toarray()
        
        b = np.dot(Ek,phi_E_vector)
        b = kpa.vectortoMesh(b,phi.shape)
        
        assert np.allclose(b[0,1:-1,1:-1],sol[1:-1,1:-1])
        assert np.sum(b[0,1:-1,1:-1]) <= np.sum(sol[1:-1,1:-1])+0.01
        assert np.sum(b[0,1:-1,1:-1]) >= np.sum(sol[1:-1,1:-1])-0.01
        return b, phi, phi_E_vector, sol
        
    def test_laplacian_3d(self):
        sim = simulationManager()
        p = species()
        kpa = kpps_analysis()
        
        m = mesh()
        case = caseHandler(mesh=m,**self.case_params)
        sim.ndim = 3
        
        z = m.pos[2,0,0,:]
        Lz = m.zlimits[1]
        y = m.pos[1,0,:,0]
        Ly = m.ylimits[1]
        x = m.pos[0,:,0,0]
        Lx = m.xlimits[1]
        
        phi = np.zeros((m.res[0:3]+1),dtype=np.float)
        sol = np.zeros((m.res[0:3]+1),dtype=np.float)
        b = np.zeros((m.res[0:3]+1),dtype=np.float) 
        
        for zi in range(0,m.res[2]+1):
            for yi in range(0,m.res[1]+1):
                for xi in range(0,m.res[0]+1):
                    phi[xi,yi,zi] = (z[zi]*(z[zi]-Lz)
                                    *y[yi]*(y[yi]-Ly)
                                    *x[xi]*(x[xi]-Lx))
                    sol[xi,yi,zi] = (2*z[zi]*(z[zi]-Lz)*y[yi]*(y[yi]-Ly)
                                    +2*y[yi]*(y[yi]-Ly)*x[xi]*(x[xi]-Lx)
                                    +2*x[xi]*(x[xi]-Lx)*z[zi]*(z[zi]-Lz))
        
        phi_F_vector = kpa.meshtoVector(phi[1:-1,1:-1,1:-1])
        Fk = kpa.poisson_cube2nd_setup(p,m,sim).toarray()
        
        b_vector = np.dot(Fk,phi_F_vector)
        b[1:-1,1:-1,1:-1] = kpa.vectortoMesh(b_vector,phi[1:-1,1:-1,1:-1].shape)

        assert np.allclose(b[1:-1,1:-1,1:-1],sol[1:-1,1:-1,1:-1])
        assert np.sum(b[1:-1,1:-1,1:-1]) <= np.sum(sol[1:-1,1:-1,1:-1])+0.01
        assert np.sum(b[1:-1,1:-1,1:-1]) >= np.sum(sol[1:-1,1:-1,1:-1])-0.01
        return b, phi, phi_F_vector, sol
    
    
    def test_2particles(self):
        p_params = {}
        sim_params = {}
        case_params = cp.copy(self.case_params)
        analysis_params = {}
        data_params = {}
        
        initial_pos = [5.,5.,5.]
        offset = [1.,0.,0.]
        
        p_params['nq'] = 2
        p_params['q'] = 2
        p_params['mq'] = 1
        
        sim_params['dt'] = 0.01
        sim_params['tEnd'] = 0.5
        sim_params['percentBar'] = True
        sim_params['simID'] = 'test_2p'
        
        case_params['particle_init'] = 'direct'
        case_params['pos'] =  [np.array(initial_pos)-np.array(offset),
                               np.array(initial_pos)+np.array(offset)]
        self.part_pos = case_params['pos']

        case_params['xlimits'] = [0,10]
        case_params['ylimits'] = [0,10]
        case_params['zlimits'] = [0,10]
        case_params['BC_function'] = self.pot
        case_params['store_node_pos'] = True
        
        analysis_params['fieldAnalysis'] = 'coulomb'
        analysis_params['particleIntegration'] = 'boris_synced'
        
        data_params = self.data_params
        data_params['samplePeriod'] = sim_params['tEnd']/sim_params['dt']
        data_params['write'] = True
        data_params['domain_limits'] = [[0,10],[0,10],[0,10]]
        data_params['component_plots'] = True
        data_params['components'] = 'x'
        
        model_clmb = {'simSettings':sim_params,'speciesSettings':p_params,
         'caseSettings':case_params,'analysisSettings':analysis_params,
         'dataSettings':data_params}
            
        kpps_clmb = kpps(**model_clmb)
        clmb_data = kpps_clmb.run()
        clmb_pos = clmb_data.load_p(['pos'])
        ppos_exact = clmb_pos['pos'][-1,0,0]
        
        resolutions = np.array([[6,5,5],[21,20,20],[31,30,30],[41,40,40]])
        #resolutions = np.array([[21,20,20]])
        errors = np.zeros((resolutions.shape[0],1),dtype = np.float)
        node_count = np.zeros((resolutions.shape[0],1),dtype = np.float)
        
        for n in range(0,resolutions.shape[0]):
            case_params['resolution'] = resolutions[n,:]
            node_count[n] = np.prod(resolutions[n,:])
            mid = floor(case_params['resolution'][2]/2)
            xlimiter = floor(case_params['resolution'][0]/10*2)
            
            analysis_params['fieldAnalysis'] = 'pic'
            model_pic = {'simSettings':sim_params,'speciesSettings':p_params,
                     'caseSettings':case_params,'analysisSettings':analysis_params,
                     'dataSettings':data_params}
            
            kpps_pic = kpps(**model_pic)
            pic_data = kpps_pic.run()
            pic_pos = pic_data.load_p(['pos'])
            
            ppos = pic_pos['pos'][-1,0,0]
            
            
            errors[n] = abs(ppos_exact-ppos)
            
        fig = plt.figure(2)
        ax4 = fig.add_subplot(1, 1, 1)
        ax4.plot(node_count,errors,label='Errors')
        ax4.plot([1100,27900],self.orderLines(-1,[1100,27900],[10**-3,10**-1]),
                ls='dashed',c='0.5',label='1st Order')
        ax4.plot([1100,27900],self.orderLines(-2,[1100,27900],[10**-3,10**-1]),
                ls='dotted',c='0.25',label='2nd Order')
        ax4.set_xlabel('Mesh node count')
        ax4.set_ylabel('$\Delta x_{rel}$')
        ax4.set_title('Error in x-position vs. node-count for PIC',fontsize=16)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend()
        
        
        mesh_data = pic_data.load_m(['pos','E','phi'])
        tslice = 1
        zplane = mid
        

        phi = mesh_data['phi'][tslice,:,:,:]
        
        E = mesh_data['E'][tslice,:,:,:,:]
        E_ppos_x = pic_pos['pos'][tslice,:,0]
        E_ppos_y = pic_pos['pos'][tslice,:,1]
        
        
        X = mesh_data['pos'][tslice,0,:,:,zplane]
        Y = mesh_data['pos'][tslice,1,:,:,zplane]
        
        EE = np.hypot(E[0,:,:,zplane],E[1,:,:,zplane])
        Ex = E[0,:,:,zplane]
        Ey = E[1,:,:,zplane]
        
        fig = plt.figure(3)
        ax1 = fig.add_subplot(1, 1, 1)
        Eplot = ax1.quiver(X,Y,Ex/EE,Ey/EE,EE,pivot='mid',units='width',cmap='coolwarm')
                    #norm=colors.LogNorm(vmin=EE.min(),vmax=EE.max()))
        ax1.scatter(X,Y,s=2,c='black')
        ax1.scatter(E_ppos_x,E_ppos_y,c='grey')
        ax1.set_xlabel('x (cm)')
        ax1.set_ylabel('y (cm)')
        cbar = fig.colorbar(Eplot,extend='max')
        cbar.set_label('E (statvolt/cm)')
            
        return errors
    
    
    def test_poisson_poly(self):
        self.poly_order = 4
        
        p_params = {}
        sim_params = {}
        case_params = cp.copy(self.case_params)
        analysis_params = {}
        data_params = {}
        
        p_params['nq'] = 1
        p_params['q'] = 1
        p_params['mq'] = 1
        
        sim_params['simID'] = 'test_poisson'
        sim_params['dt'] = 0.01
        sim_params['tEnd'] = 0.01
        sim_params['ndim'] = 3
        
        case_params['particle_init'] = 'direct'
        case_params['pos'] =  [[0.,0.,0.]]
        self.part_pos = np.array(case_params['pos'])

        case_params['xlimits'] = [-5,5]
        case_params['ylimits'] = [-5,5]
        case_params['zlimits'] = [-5,5]
        case_params['BC_function'] = self.poly_phi
        case_params['store_node_pos'] = True
        
        analysis_params['fieldIntegration'] = True
        analysis_params['field_type'] = 'pic'
        analysis_params['background'] = self.poly_charge
        analysis_params['units'] = 'custom'
        analysis_params['periodic_mesh'] = True
        
        data_params['samplePeriod'] = 1
        data_params['write'] = True
        data_params['domain_limits'] = [[0,10],[0,10],[0,10]]
        
        resolutions = np.array([[5,5,5],[10,10,10],[20,20,20]])
        errors = np.zeros((resolutions.shape[0],1),dtype = np.float)
        node_count = np.zeros((resolutions.shape[0],1),dtype = np.float)
        
        for n in range(0,resolutions.shape[0]):
            case_params['resolution'] = resolutions[n,:]
            #node_count[n] = np.prod(resolutions[n,:])
            node_count[n] = resolutions[n,2]
            midx = floor(case_params['resolution'][0]/2)
            midy = floor(case_params['resolution'][1]/2)
            xlimiter = floor(case_params['resolution'][0]/10*2)
            model1 = {'simSettings':sim_params,'speciesSettings':p_params,
                     'caseSettings':case_params,'analysisSettings':analysis_params,
                     'dataSettings':data_params}
            
            kpps1 = kpps(**model1)
            dh = kpps1.run()
            pot_mesh_dict = dh.load_m(['pos','phi'])


            X = pot_mesh_dict['pos'][-1,:,:,:,:]
            
            phi_full = pot_mesh_dict['phi'][-1,:,:,:]
            phi = pot_mesh_dict['phi'][-1,midx,midy,:]
            
            p = self.poly_order
            phi_exact = X[0,:,:,:]**p+ X[1,:,:,:]**p + X[2,:,:,:]**p
        
            errors[n] = np.mean(abs(phi-phi_exact[midx,midy,:]))
            fig = plt.figure(n)
            ax4 = fig.add_subplot(1, 1, 1)
            ax4.plot(X[2,midx,midy,:],phi,label='potential numerical')
            ax4.plot(X[2,midx,midy,:],phi_exact[midx,midy,:],label='potential exact')
            ax4.set_xlabel('z (cm)')
            ax4.set_ylabel('E potential (statV/cm)')
            ax4.set_title('Potential in z slice in ' +str(sim_params['ndim']) +'D, for [nx,ny,nz] = ' 
                          + str(case_params['resolution']) 
                          + ' and poly order = ' + str(self.poly_order),fontsize=16)
            ax4.legend()
            
        
        fig = plt.figure(20)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(node_count,errors,label='Errors')
        ax.plot([node_count[0],node_count[-1]],self.orderLines(-1,[node_count[0],node_count[-1]],[errors[-1],errors[0]]),
                ls='dashed',c='0.5',label='1st Order')
        ax.plot([node_count[0],node_count[-1]],self.orderLines(-2,[node_count[0],node_count[-1]],[errors[-1],errors[0]]),
                ls='dotted',c='0.25',label='2nd Order')
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_yscale('log')
        ax.set_ylabel('$\Delta \phi$ for x in range [0,2]')
        ax.set_title('Error convergence for Poisson eq. in ' +str(sim_params['ndim']) +'D, for polynomial order = ' + str(self.poly_order),fontsize=16)
        ax.legend()
        
        return phi_full, phi_exact, errors
    
    
    def test_poisson_deposit(self):
        self.poly_order = 2
        
        p_params = {}
        sim_params = {}
        case_params = cp.copy(self.case_params)
        analysis_params = {}
        data_params = {}
        
        p_params['q'] = 1
        p_params['mq'] = 1
        
        sim_params['simID'] = 'test_poisson'
        sim_params['dt'] = 0.01
        sim_params['tEnd'] = 0.01
        sim_params['ndim'] = 1
        
        case_params['particle_init'] = 'direct'
        case_params['xlimits'] = [-5,5]
        case_params['ylimits'] = [-5,5]
        case_params['zlimits'] = [-5,5]
        case_params['BC_function'] = self.poly_phi
        case_params['store_node_pos'] = True
        
        analysis_params['fieldIntegration'] = True
        analysis_params['preAnalysis_methods'] = [self.set_dv]
        #analysis_params['background'] = self.poly_charge
        analysis_params['fieldIntegrator_methods'] = ['scatter']
        analysis_params['field_type'] = 'pic'
        analysis_params['units'] = 'custom'
        analysis_params['periodic_axes'] = ['x','y','z']
        
        data_params['samplePeriod'] = 1
        data_params['write'] = True
        data_params['domain_limits'] = [[0,10],[0,10],[0,10]]
        
        resolutions = np.array([[2,2,5],[2,2,10],[2,2,100],[2,2,100]])
        errors = np.zeros((resolutions.shape[0],1),dtype = np.float)
        node_count = np.zeros((resolutions.shape[0],1),dtype = np.float)
        
        for n in range(0,resolutions.shape[0]):
            nn = resolutions[n,2]+1
            #nq = (resolutions[n,2])*2
            nq=20
            L = case_params['zlimits'][1]-case_params['zlimits'][0]
            dz = L/resolutions[n,2]
            pos_list = np.zeros((nq,3),dtype=np.float)
            pos_list[:,2] = np.linspace(-5+dz/4,5-dz/4,nq)
            p_params['nq'] = nq
            case_params['resolution'] = resolutions[n,:]
            case_params['pos'] = pos_list
            
            node_count[n] = resolutions[n,2]
            midx = floor(case_params['resolution'][0]/2)
            midy = floor(case_params['resolution'][1]/2)
            xlimiter = floor(case_params['resolution'][0]/10*2)
            model1 = {'simSettings':sim_params,'speciesSettings':p_params,
                     'caseSettings':case_params,'analysisSettings':analysis_params,
                     'dataSettings':data_params}
            
            kpps1 = kpps(**model1)
            dh = kpps1.run()
            pot_mesh_dict = dh.load_m(['pos','phi'])


            X = pot_mesh_dict['pos'][-1,:,:,:,:]
            
            phi_full = pot_mesh_dict['phi'][-1,:,:,:]
            phi = pot_mesh_dict['phi'][-1,midx,midy,:]
            
            p = self.poly_order
            phi_exact = X[0,:,:,:]**p+ X[1,:,:,:]**p + X[2,:,:,:]**p
        
            errors[n] = np.mean(abs(phi-phi_exact[midx,midy,:]))
            fig = plt.figure(n)
            ax4 = fig.add_subplot(1, 1, 1)
            ax4.plot(X[2,midx,midy,:],phi,label='potential numerical')
            ax4.plot(X[2,midx,midy,:],phi_exact[midx,midy,:],label='potential exact')
            ax4.set_xlabel('z (cm)')
            ax4.set_ylabel('E potential (statV/cm)')
            ax4.set_title('Potential in z slice in ' +str(sim_params['ndim']) +'D, for [nx,ny,nz] = ' 
                          + str(case_params['resolution']) 
                          + ' and poly order = ' + str(self.poly_order),fontsize=16)
            ax4.legend()
            
        
        fig = plt.figure(20)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(node_count,errors,label='Errors')
        ax.plot([node_count[0],node_count[-1]],self.orderLines(-1,[node_count[0],node_count[-1]],[errors[-1],errors[0]]),
                ls='dashed',c='0.5',label='1st Order')
        ax.plot([node_count[0],node_count[-1]],self.orderLines(-2,[node_count[0],node_count[-1]],[errors[-1],errors[0]]),
                ls='dotted',c='0.25',label='2nd Order')
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_yscale('log')
        ax.set_ylabel('$\Delta \phi$ for x in range [0,2]')
        ax.set_title('Error convergence for Poisson eq. in ' +str(sim_params['ndim']) +'D, for polynomial order = ' + str(self.poly_order),fontsize=16)
        ax.legend()
        
        return phi_full, phi_exact, errors
    
    def set_dv(self,species,mesh,controller):
        mesh.dv = 1
        
    def constant_charge(self,species,mesh,simulationManager):
        mesh.rho[:,:,:] = 2*simulationManager.ndim
    
    def poly_charge(self,species,mesh,simulationManager):
        n = self.poly_order
        if simulationManager.ndim >= 3:
            mesh.rho[:,:,:] += n*(n-1)*mesh.pos[0,:,:,:]**(n-2)
            
        if simulationManager.ndim >= 2:
            mesh.rho[:,:,:] += n*(n-1)*mesh.pos[1,:,:,:]**(n-2) 
        
        mesh.rho[:,:,:] += n*(n-1)*mesh.pos[2,:,:,:]**(n-2)
        

    def stationary(self,species,mesh):
        species.E[:,:] = 0
        species.B[:,:] = 0
        
    def poly_phi(self,pos):
        n = self.poly_order
        phi = pos[0]**n+pos[1]**n+pos[2]**n
        return phi
    
    def pot(self,pos):
        phi = 0
        ppos = np.array(self.part_pos)
        for p in range(0,ppos.shape[0]):
            r = np.sqrt((pos[0]-ppos[p,0])**2+(pos[1]-ppos[p,1])**2+(pos[2]-ppos[p,2])**2)
            phi += -1/r
        return phi
    
    def orderLines(self,order,xRange,yRange):
        if order < 0:
            a = yRange[1]/xRange[0]**order
        else:
            a = yRange[0]/xRange[0]**order    
        
        oLine = [a*xRange[0]**order,a*xRange[1]**order]
            
        return oLine

tf = Test_fields()
tf.setup()
phi, phi_exact, errors = tf.test_poisson_deposit()
