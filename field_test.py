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
        
        self.case_params = case_params
        
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
        
        p_params['nq'] = 1
        p_params['q'] = 1
        p_params['mq'] = 1
        
        sim_params['dt'] = 0.01
        sim_params['tEnd'] = 0.01
        sim_params['percentBar'] = True
        
        case_params['particle_init'] = 'direct'
        self.potpos = [5.0,5.0,5.0]
        case_params['pos'] =  [self.potpos]
        #,[52.1,0.5,0.5]]
        case_params['resolution'] = [21,20,20]
        self.res = case_params['resolution']
        case_params['xlimits'] = [0,10]
        case_params['ylimits'] = [0,10]
        case_params['zlimits'] = [0,10]
        case_params['BC_function'] = self.pot
        self.middom = floor(case_params['resolution'][1]/2)
        
        analysis_params['fieldAnalysis'] = 'pic'
        analysis_params['particleIntegration'] = 'boris_synced'
        analysis_params['coulomb_field_check'] = True
        
        data_params['samplePeriod'] = 1
        data_params['record'] = True
        data_params['component_plots'] = True
        data_params['components'] = 'x'
        data_params['domain_limits'] = [[0,10],[0,10],[0,10]]
        
        model1 = {'simSettings':sim_params,'speciesSettings':p_params,
                 'caseSettings':case_params,'analysisSettings':analysis_params,
                 'dataSettings':data_params}
        
        kpps1 = kpps(**model1)
        pic_data = kpps1.run()
        
        analysis_params['fieldAnalysis'] = 'coulomb'
        
        model2 = {'simSettings':sim_params,'speciesSettings':p_params,
                 'caseSettings':case_params,'analysisSettings':analysis_params,
                 'dataSettings':data_params}
        
        kpps2 = kpps(**model2)
        colmb_data = kpps2.run()
        
        return pic_data, colmb_data
    
    
    def test_poisson(self):
        p_params = {}
        sim_params = {}
        case_params = cp.copy(self.case_params)
        analysis_params = {}
        data_params = {}
        
        p_params['nq'] = 1
        p_params['q'] = 1
        p_params['mq'] = 1
        
        sim_params['dt'] = 0.01
        sim_params['tEnd'] = 0.01
        
        case_params['particle_init'] = 'direct'
        case_params['pos'] =  [[5.0,5.0,5.0]]
        self.part_pos = np.array(case_params['pos'])

        case_params['xlimits'] = [0,10]
        case_params['ylimits'] = [0,10]
        case_params['zlimits'] = [0,10]
        case_params['BC_function'] = self.pot
        case_params['store_node_pos'] = True
        
        analysis_params['fieldAnalysis'] = 'pic'
        analysis_params['particleIntegration'] = 'boris_synced'
        
        data_params['samplePeriod'] = 1
        data_params['record'] = True
        data_params['domain_limits'] = [[0,10],[0,10],[0,10]]
        
        #resolutions = np.array([[11,10,10],[21,20,20],[31,30,30],[41,40,40]])
        resolutions = np.array([[21,20,20]])
        errors = np.zeros((resolutions.shape[0],1),dtype = np.float)
        node_count = np.zeros((resolutions.shape[0],1),dtype = np.float)
        
        for n in range(0,resolutions.shape[0]):
            case_params['resolution'] = resolutions[n,:]
            node_count[n] = np.prod(resolutions[n,:])
            mid = floor(case_params['resolution'][1]/2)
            xlimiter = floor(case_params['resolution'][0]/10*2)
            model1 = {'simSettings':sim_params,'speciesSettings':p_params,
                     'caseSettings':case_params,'analysisSettings':analysis_params,
                     'dataSettings':data_params}
            
            kpps1 = kpps(**model1)
            pot_data = kpps1.run()
            
            X = pot_data.mesh_pos[0,:,mid,mid]
            phi = pot_data.mesh_phi[1][:,mid,mid]
            phi_exact = -1/(abs(X - self.part_pos[0,0]))

            errors[n] = np.mean(abs(phi[0:xlimiter]-phi_exact[0:xlimiter]))
            
            fig = plt.figure(n)
            ax4 = fig.add_subplot(1, 1, 1)
            ax4.plot(X,phi,label='potential numerical')
            ax4.plot(X,phi_exact,label='potential exact')
            ax4.set_xlabel('x (cm)')
            ax4.set_ylabel('E potential (statV/cm)')
            ax4.set_title('Potential in X slice at y=z=5 for [nx,ny,nz] = ' 
                          + str(case_params['resolution']) 
                          + ' and unit charge particle at x=y=z=5',fontsize=16)
            ax4.legend()
            
            fig = plt.figure(n+10)
            ax4 = fig.add_subplot(1, 1, 1)
            ax4.plot(X,abs(phi-phi_exact))
            ax4.set_xlabel('x (cm)')
            ax4.set_ylabel('E diff (statV/cm)')
            ax4.legend()
        
        fig = plt.figure(20)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(node_count,errors,label='Errors')
        ax.plot([1100,127500],self.orderLines(-1,[1100,127500],[10**-5,10**-2]),
                ls='dashed',c='0.5',label='1st Order')
        ax.plot([1100,127500],self.orderLines(-2,[1100,127500],[10**-5,10**-2]),
                ls='dotted',c='0.25',label='2nd Order')
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_yscale('log')
        ax.set_ylabel('$\Delta \phi$ for x in range [0,2]')
        ax.set_title('Error convergence for Poisson eq. in 3D, unit charge particle at mid-domain',fontsize=16)
        ax.legend()
        
        return errors

    
    def pot(self,pos):
        ppos = self.part_pos
        r = np.sqrt((pos[0]-ppos[0,0])**2+(pos[1]-ppos[0,1])**2+(pos[2]-ppos[0,2])**2)
        phi = -1/r
        
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
b, phi, phi_F_vector, sol = tf.test_laplacian_3d()
error = tf.test_poisson()
"""
tf = Test_fields()
tf.setup()
#tf.test_laplacian_3d()
pic_data, colmb_data = tf.test_2particles()


pos = pic_data.mesh_pos
tslice = 1
zplane = tf.middom

Q = pic_data.mesh_q[tslice][:,:,zplane]
rho = pic_data.mesh_q[tslice][:,:,zplane]
phi = pic_data.mesh_phi[tslice][:,:,zplane]

E = pic_data.mesh_E[tslice]
E_ppos_x = pic_data.xArray[tslice]
E_ppos_y = pic_data.yArray[tslice]


X = pos[0,:,:,zplane]
Y = pos[1,:,:,zplane]

EE = np.hypot(E[0,:,:,zplane],E[1,:,:,zplane])
Ex = E[0,:,:,zplane]
Ey = E[1,:,:,zplane]

fig = plt.figure(4)
ax1 = fig.add_subplot(1, 1, 1)
Eplot = ax1.quiver(X,Y,Ex/EE,Ey/EE,EE,pivot='mid',units='width',cmap='coolwarm')
         #norm=colors.LogNorm(vmin=EE.min(),vmax=EE.max()))
ax1.scatter(X,Y,s=2,c='black')
ax1.scatter(E_ppos_x,E_ppos_y,c='grey')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
cbar = fig.colorbar(Eplot,extend='max')
cbar.set_label('E (statvolt/cm)')



CE = pic_data.mesh_CE[tslice]
CE_ppos_x = pic_data.xArray[tslice]
CE_ppos_y = pic_data.yArray[tslice]

CEE = np.hypot(CE[0,:,:,zplane],CE[1,:,:,zplane])
CEx = CE[0,:,:,zplane]
CEy = CE[1,:,:,zplane]

fig = plt.figure(5)
ax2 = fig.add_subplot(1, 1, 1)
CEplot = ax2.quiver(X,Y,CEx/CEE,CEy/CEE,CEE,pivot='mid',units='width',cmap='coolwarm')
          #norm=colors.LogNorm(vmin=CEE.min(),vmax=CEE.max()))
ax2.scatter(X,Y,s=2,c='black')
ax2.scatter(CE_ppos_x,CE_ppos_y,c='grey')
ax2.set_xlabel('x (cm)')
ax2.set_ylabel('y (cm)')
cbar = fig.colorbar(CEplot,extend='max')
cbar.set_label('E (statvolt/cm)')

fig = plt.figure(6)
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(X[:,0],E[0,:,tf.middom,zplane],label='pic')
ax3.plot(X[:,0],CE[0,:,tf.middom,zplane],label='coulomb')
ax3.set_xlabel('x (cm)')
ax3.set_ylabel('E (statV/cm)')
ax3.legend()

phi_exact = -1/(abs(X[:,0] - E_ppos_x[0])) 
fig = plt.figure(7)
ax4 = fig.add_subplot(1, 1, 1)
ax4.plot(X[:,0],phi[:,tf.middom],label='potential numerical')
ax4.plot(X[:,0],phi_exact,label='potential exact')
ax4.set_xlabel('x (cm)')
ax4.set_ylabel('E potential (statV/cm)')
ax4.legend()

fig = plt.figure(8)
ax5 = fig.add_subplot(111,projection='3d')
ax5.plot_surface(X,Y,phi,cmap='coolwarm',label='potential numerical')
ax5.set_xlabel('x (cm)',labelpad=15)
ax5.set_ylabel('y (cm)',labelpad=15)
ax5.set_zlabel('$\phi$ (statvolt/cm)',labelpad=15)
ax5.set_title('Potential in XY plane at z=5 for [nx,ny,nz]= '+str(tf.res),fontsize=16)

"""