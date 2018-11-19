import numpy as np
import scipy as sp
import copy as cp
import matplotlib.pyplot as plt
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
        for zi in range(0,m.res[2]+1):
            for yi in range(0,m.res[1]+1):
                for xi in range(0,m.res[0]+1):
                    phi[xi,yi,zi] = (z[zi]*(z[zi]-Lz)
                                    *y[yi]*(y[yi]-Ly)
                                    *x[xi]*(x[xi]-Lx))
                    sol[xi,yi,zi] = (2*z[zi]*(z[zi]-Lz)*y[yi]*(y[yi]-Ly)
                                    +2*y[yi]*(y[yi]-Ly)*x[xi]*(x[xi]-Lx)
                                    +2*x[xi]*(x[xi]-Lx)*z[zi]*(z[zi]-Lz))
        
        phi_F_vector = kpa.meshtoVector(phi)
                    
        Fk = kpa.poisson_cube2nd_setup(p,m,sim).toarray()
        
        b = np.dot(Fk,phi_F_vector)
        b = kpa.vectortoMesh(b,phi.shape)

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
        case_params['pos'] =  [[51,50,50]]
        #,[52.1,0.5,0.5]]
        case_params['resolution'] = [20,20,20]
        case_params['xlimits'] = [0,100]
        case_params['ylimits'] = [0,100]
        case_params['zlimits'] = [0,100]
        
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
    
tf = Test_fields()
tf.setup()
#tf.test_laplacian_3d()
pic_data, colmb_data = tf.test_2particles()


pos = pic_data.mesh_pos
tslice = 1
zplane = 1

Q = pic_data.mesh_q[tslice][:,:,zplane]
rho = pic_data.mesh_q[tslice][:,:,zplane]/(pos[0,1,0,0]-pos[0,0,0,0])
phi = pic_data.mesh_phi[tslice][:,:,zplane]

E = pic_data.mesh_E[tslice]
E_ppos_x = pic_data.xArray[tslice]
E_ppos_y = pic_data.yArray[tslice]


X = pos[0,:,:,zplane]
Y = pos[1,:,:,zplane]

EE = np.hypot(E[0,:,:,zplane],E[1,:,:,zplane])
Ex = E[0,:,:,zplane]/EE
Ey = E[1,:,:,zplane]/EE



fig = plt.figure(4)
ax = fig.add_subplot(1, 1, 1)
Eplot = ax.quiver(X,Y,Ex,Ey,EE,pivot='mid',units='width',cmap='coolwarm')
         #norm=colors.LogNorm(vmin=EE.min(),vmax=EE.max()))
ax.scatter(X,Y,s=2,c='black')
ax.scatter(E_ppos_x,E_ppos_y,c='grey')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
cbar = fig.colorbar(Eplot,extend='max')
cbar.set_label('E (statvolt/cm)')



CE = pic_data.mesh_CE[tslice]
CE_ppos_x = pic_data.xArray[tslice]
CE_ppos_y = pic_data.yArray[tslice]

CEE = np.hypot(CE[0,:,:,zplane],CE[1,:,:,zplane])
CEx = CE[0,:,:,zplane]/CEE
CEy = CE[1,:,:,zplane]/CEE

fig = plt.figure(5)
ax = fig.add_subplot(1, 1, 1)
CEplot = ax.quiver(X,Y,CEx,CEy,CEE,pivot='mid',units='width',cmap='coolwarm')
          #norm=colors.LogNorm(vmin=CEE.min(),vmax=CEE.max()))
ax.scatter(X,Y,s=2,c='black')
ax.scatter(CE_ppos_x,CE_ppos_y,c='grey')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
cbar = fig.colorbar(CEplot,extend='max')
cbar.set_label('E (statvolt/cm)')

fig = plt.figure(6)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X[:,0],E[0,:,0,zplane],label='pic')
ax.plot(X[:,0],CE[0,:,0,zplane],label='coulomb')
ax.set_xlabel('x (cm)')
ax.set_ylabel('E (statV/cm)')
ax.legend()

phi_exact = -1/(abs(X[:,0] - E_ppos_x[0])) 
#+ -1/(abs(X[:,0] - E_ppos_x[1]))
fig = plt.figure(7)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X[:,0],phi[:,0],label='potential numerical')
ax.plot(X[:,0],phi_exact,label='potential exact')
ax.set_xlabel('x (cm)')
ax.set_ylabel('E potential (statV/cm)')
ax.legend()
