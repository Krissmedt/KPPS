#!/usr/bin/env python3

## Dependencies
import numpy as np
import random as rand
import copy as cp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Required modules
from kpps_analysis import kpps_analysis
from species import species
from mesh import mesh
from particleLoader import particleLoader as pLoader
from meshLoader import meshLoader
from controller import controller
from dataHandler2 import dataHandler2 as DH

class Test_boris:
    def setup(self):
        self.dt = 0.1
        self.steps = 10
        self.print = False
        
        sim_params = {}
        spec1_params = {}
        loader1_params = {}
        mLoader_params = {}
        analysis_params = {}
        
        sim_params['t0'] = 0
        sim_params['tSteps'] = self.steps
        sim_params['dimensions'] = 1
        sim_params['zlimits'] = [-1,1]
        
        spec1_params['nq'] = 2
        spec1_params['mq'] = 1
        spec1_params['q'] = 1
        spec1_params['name'] = 'spec1'
        
        loader1_params['load_type'] = 'direct'
        loader1_params['speciestoLoad'] = [0]
        loader1_params['pos'] = np.array([[0,0,0.5],[0,0,0.75]])
        loader1_params['vel'] = np.array([[0,0,0],[0,0,0]])

        mLoader_params['load_type'] = 'box'
        mLoader_params['resolution'] = [2,2,10000]
        mLoader_params['store_node_pos'] = True
        
        analysis_params['particleIntegration'] = True
        analysis_params['nodeType'] = 'lobatto'
        analysis_params['M'] = 5
        analysis_params['K'] = 5
        analysis_params['fieldIntegration'] = False
        analysis_params['field_type'] = 'pic'
        analysis_params['looped_axes'] = ['z']
        analysis_params['residual_check'] = False
        analysis_params['units'] = 'custom'
        analysis_params['mesh_boundary_z'] = 'open'
        analysis_params['poisson_M_adjust_1d'] = 'simple_1d'
        analysis_params['hooks'] = ['kinetic_energy','field_energy']
        analysis_params['rhs_check'] = True
        
        analysis_params['external_fields'] = True
        analysis_params['E_type'] = 'transform'
        analysis_params['E_transform'] = np.array([[0,0,0],[0,0,0],[0,0,-1]])
        analysis_params['E_magnitude'] = 1

        self.sim_params = sim_params
        self.species_params = [spec1_params]
        self.pLoader_params = [loader1_params]
        self.mLoader_params = mLoader_params
        self.analysis_params = analysis_params
        
        
        ### Trick test parameters ###
        self.e_z = np.array([0, 0, 1])
        self.omega_b = 25.0
        self.alpha   = 1.0
        self.epsilon = -1.0 # set E-field to zero for testing
        self.omega_e = 4.9
        self.E_mat = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, -2] ])
        self.alpha = 1.0

    def getB(self,x):
        return (self.omega_b/self.alpha)*self.e_z

    def getE(self,x):
        return -self.epsilon*(self.omega_e**2/self.alpha)*self.E_mat.dot(x)

    def test_trick(self):
        kpa = kpps_analysis()
        dt = 0.01
        
        # Generate starting position and velocity randomly
        x_old = np.random.rand(3)
        v_old = np.random.rand(3)
        
        # Perform forward Euler step to obtain new position (would be Verlet in actual Boris method)
        x_new = x_old + dt*v_old
        
        # Generate a random c_i term
        c_i   = np.random.rand(3)
        
        '''
        Boris' trick should solve v_new = v_old + dt*E + 0.5*dt*(v_old + v_new) x B
        with B and E being taken at the same position.
        '''
        E      = 0.5*(self.getE(x_old) + self.getE(x_new))
        B      = self.getB(x_new)
        v_new  = self.boris_trick(v_old, E, B, c_i,dt)
        v_avg  = 0.5*(v_old + v_new)
        B_m    = np.cross( v_avg, self.getB(x_new) )
        defect = v_new - v_old - dt*(0.5*(self.getE(x_old) + self.getE(x_new))) - dt*B_m - dt*c_i
        print ("Defect with trapezoidal: %5.3e" % np.linalg.norm(defect, np.inf))
        
        '''
        Midpoint rule based
        '''
        
        x_avg_mp = 0.5*(x_old + x_new)
        E      = self.getE(x_avg_mp)
        B      = self.getB(x_avg_mp)
        v_new_mp = self.boris_trick(v_old, E, B, c_i,dt)
        
        v_avg_mp = 0.5*(v_old + v_new)
        B_m_mp = np.cross( v_avg, self.getB(x_avg_mp) )
        defect = v_new_mp - v_old - dt*self.getE(0.5*(x_old + x_new)) - dt*B_m_mp - dt*c_i
        print ("Defect with midpoint: %5.3e" % np.linalg.norm(defect, np.inf))
        
        
        '''
        Kris' Boris trick
        '''
        E      = 0.5*(self.getE(x_old) + self.getE(x_new))
        B      = self.getB(x_new)
        v_new  = kpa.boris(np.array([v_old]), np.array([E]), np.array([B]), dt, self.alpha, c_i)
        v_avg  = 0.5*(v_old + v_new)
        B_m    = np.cross( v_avg, self.getB(x_new) )
        defect = v_new - v_old - dt*(0.5*(self.getE(x_old) + self.getE(x_new))) - dt*B_m - dt*c_i
        print ("Defect with Kris Boris: %5.3e" % np.linalg.norm(defect, np.inf))
        
        
    def test_linear(self):
        real_pos = np.zeros((self.steps+1,2),dtype=np.float)
        real_pos[0,:] = [0.5,0.75]
        
        real_vel = np.zeros((self.steps+1,2),dtype=np.float)
        real_vel[0,:] = [0.,0.]
        
        for t in range(1,test_obj.steps+1):
            real_pos[t,0] = 0.5 * math.cos(t*self.dt)
            real_pos[t,1] = 0.75 * math.cos(t*self.dt)
            real_vel[t,0] = -0.5 * math.sin(t*self.dt)
            real_vel[t,1] = -0.75 * math.sin(t*self.dt)
        
        
        end = 1
        dts = [1,0.5,0.25,0.125]
        
        pos_sync = []
        pos_stag = []
        pos_sdc = []
        
        pos_sync_pic = []
        pos_stag_pic = []
        pos_sdc_pic = []
        
        #solutions for simple harmonic osc
        real_pos[-1,0] = 0.5 * math.cos(end)
        real_pos[-1,1] = 0.75 * math.cos(end)
        real_vel[-1,0] = -0.5 * math.sin(end)
        real_vel[-1,1] = -0.75 * math.sin(end)
        
        
        for dt in dts:
            rewind_vel = -0.5 * math.sin(-dt)
            self.dt = dt
            self.steps = np.int(end/dt)
            
            self.analysis_params['gather'] = 'none'
            self.analysis_params['external_fields'] = True
            self.analysis_params['external_fields_mesh'] = False
            spec_sync = self.test_synced()
            spec_stag = self.test_staggered()
            spec_sdc = self.test_SDC()
            
            pos_sync.append(spec_sync.pos[:,2])
            pos_stag.append(spec_stag.pos[:,2])
            pos_sdc.append(spec_sdc.pos[:,2])
            
            self.analysis_params['gather'] = 'trilinear_gather'
            self.analysis_params['external_fields'] = False
            self.analysis_params['external_fields_mesh'] = True
            spec_sync_pic = self.test_synced()
            spec_stag_pic = self.test_staggered()
            spec_sdc_pic = self.test_SDC()    
        
            pos_sync_pic.append(spec_sync_pic.pos[:,2])
            pos_stag_pic.append(spec_stag_pic.pos[:,2])
            pos_sdc_pic.append(spec_sdc_pic.pos[:,2])
            
            
        pos_sync = np.array(pos_sync)
        pos_stag = np.array(pos_stag)
        pos_sdc = np.array(pos_sdc)
        pos_sync_pic = np.array(pos_sync_pic)
        pos_stag_pic = np.array(pos_stag_pic)
        pos_sdc_pic = np.array(pos_sdc_pic)
        
        zRel_sync = np.abs(pos_sync - real_pos[-1,:])
        zRel_stag = np.abs(pos_stag- real_pos[-1,:])
        zRel_sdc = np.abs(pos_sdc - real_pos[-1,:])
            
        zRel_sync_pic = np.abs(pos_sync_pic - real_pos[-1,:])
        zRel_stag_pic = np.abs(pos_stag_pic- real_pos[-1,:])
        zRel_sdc_pic = np.abs(pos_sdc_pic - real_pos[-1,:])
        
        ##Order Plot w/ dt
        fig_dt = plt.figure(1)
        ax_dt = fig_dt.add_subplot(1, 1, 1)
        ax_dt.plot(dts,zRel_sync[:,0],label='Synced')
        ax_dt.plot(dts,zRel_stag[:,0],label='Staggered')
        ax_dt.plot(dts,zRel_sync_pic[:,0],label='Synced PIC')
        ax_dt.plot(dts,zRel_stag_pic[:,0],label='Staggered PIC')
        ax_dt.plot(dts,zRel_sdc[:,0],label='SDC')
        ax_dt.plot(dts,zRel_sdc_pic[:,0],label='SDC PIC')
        
        ## Order plot finish
        dHandler = DH()
        ax_dt.set_xscale('log')
        #ax_dt.set_xlim(10**-3,10**-1)
        ax_dt.set_xlabel('$\Delta t$')
        ax_dt.set_yscale('log')
        #ax_dt.set_ylim(10**(-7),10**1)
        ax_dt.set_ylabel('$\Delta x^{(rel)}$')
        
        xRange = ax_dt.get_xlim()
        yRange = ax_dt.get_ylim()
        
        ax_dt.plot(xRange,dHandler.orderLines(1,xRange,yRange),
                    ls='-',c='0.25',label='1st Order')
        ax_dt.plot(xRange,dHandler.orderLines(2,xRange,yRange),
                    ls='dotted',c='0.25',label='2nd Order')
        ax_dt.plot(xRange,dHandler.orderLines(4,xRange,yRange),
                    ls='dashed',c='0.75',label='4th Order')
        ax_dt.plot(xRange,dHandler.orderLines(8,xRange,yRange),
                    ls='dashdot',c='0.1',label='8th Order')
        ax_dt.legend()
        
    def test_synced(self):
        self.analysis_params['particleIntegrator'] = 'boris_synced'
        self.analysis_params['pre_hook_list'] = []
        self.sim_params['dt'] = self.dt
        
        species_list, mesh1 = self.test_kpps()
            
        return species_list[0]
            
    def test_SDC(self):
        self.analysis_params['particleIntegrator'] = 'boris_SDC'
        self.analysis_params['pre_hook_list'] = []
        self.sim_params['dt'] = self.dt
        
        species_list, mesh1 = self.test_kpps()
            
        return species_list[0]
    
    def test_staggered(self):
        self.analysis_params['particleIntegrator'] = 'boris_staggered'
        self.analysis_params['pre_hook_list'] = ['ES_vel_rewind']
        self.sim_params['dt'] = self.dt
        
        species_list, mesh1 = self.test_kpps()
            
        return species_list[0]
    
    
    def test_kpps(self):
        print("Running: " + self.analysis_params['particleIntegrator'] + " at dt = " + str(self.dt))
        print("************************ Setup *******************************")
        sim = controller(**self.sim_params)
        analyser = kpps_analysis(**self.analysis_params)
        spec1 = species(**self.species_params[0])
        species_list = [spec1]
        pLoader1 = pLoader(**self.pLoader_params[0])
        mLoader = meshLoader(**self.mLoader_params)
        mesh1 = mesh()
        
        pLoader1.run(species_list,sim)
        mLoader.run(mesh1,sim)
        
        print(analyser.fieldIntegrator_methods)

        analyser.run_preAnalyser(species_list,mesh1,controller=sim)
        print("*************************** Run ******************************")

        for t in range(1,self.steps+1):
            self.print_step(species_list,t)
            sim.updateTime()
            analyser.run_fieldIntegrator(species_list,mesh1,sim)
            analyser.run_particleIntegrator(species_list,mesh1,sim)
            analyser.runHooks(species_list,mesh1,controller=sim)

        print("*************************** Done *****************************")            
        return species_list, mesh1
    
    
    def print_step(self,species_list,t):
        if self.print == True:
            print("*************** t = " + str(t-1) + " ***********************")
            print(species_list[0].pos)
            print(species_list[0].E)
            print(species_list[0].vel)
            print("")

    def F(self,x,v):
        return self.getE(x) + np.cross(v, self.getB(x))
    
    def boris_trick(self,v_old, E_np12, B, c_i,dt):
        t      = 0.5*dt*B
        s      = 2.0*t/(1.0 + np.dot(t, t))
        v_min  = v_old + 0.5*dt*E_np12 + 0.5*dt*c_i
        v_star = v_min + np.cross(v_min, t)
        v_plu  = v_min + np.cross(v_star, s)
        return v_plu + 0.5*dt*E_np12 + 0.5*dt*c_i
    
    def exponential_pos(self,t,c1,c2):
        x = math.log(1/2 *c1*(math.tanh**2 * (1/2 * math.sqrt(c1*(c2 + t)**2)) - 1))
        return x
    
    def exponential_vel(self,t,c1,c2):
        v_num = c1*(c2+t)*math.tanh(1/2 * math.sqrt(c1*(c2+t)**2)) * math.sech(1/2 * math.sqrt(c1*(c2+t)**2))
        v_den = math.sqrt(c1*(c2+t)**2) * (math.tanh(1/2 * math.sqrt(c1*(c2+t)**2))-1)
        v1 = v_num/v_den
        v2 = -math.sqrt(c1) * math.tanh(1/2 * math.sqrt(c1)*(c2+t))
        
        return v1,v2
    
    def nonLinear_ext_E(self,species,mesh,controller=None):
        nq = species.pos.shape[0]
        for pii in range(0,nq):
            species.E[pii,2] += -np.power(species.pos[pii,2],3)
        
        return species


    def nonLinear_mesh_E(self,species_list,mesh,controller=None):
        for zi in range(0,mesh.E[2,1,1,:].shape[0]-1):
            z = mesh.zlimits[0] + zi * mesh.dz
            mesh.E[2,1,1,zi] += -np.power(z,3)
    
        static_E = np.zeros(mesh.E.shape)
        static_E[:] = mesh.E[:]
    
        return mesh, static_E
    
test_obj = Test_boris()
test_obj.setup()
test_obj.test_trick()

run_linear = False
run_nonLinear = True

######################## Linear oscillator test ###############################
if run_linear == True:
    test_obj.test_linear()


############################# Non-linear oscillator test ######################
if run_nonLinear == True:
    end = 1
    dts = [1,0.5,0.25,0.125]
    #dts = [1]
    
    pos_sync = []
    pos_stag = []
    pos_sdc = []
    
    pos_sync_pic = []
    pos_stag_pic = []
    pos_sdc_pic = []
    
    #solutions for non-linear osc
    test_obj.analysis_params['E_type'] = 'custom'
    test_obj.analysis_params['custom_external_E'] = test_obj.nonLinear_ext_E
    test_obj.analysis_params['custom_static_E'] = test_obj.nonLinear_mesh_E
    
    #test_obj.analysis_params['gather'] = 'trilinear_gather'
    #test_obj.analysis_params['external_fields'] = False
    #test_obj.analysis_params['external_fields_mesh'] = True
    test_obj.analysis_params['gather'] = 'none'
    test_obj.analysis_params['external_fields'] = True
    test_obj.analysis_params['external_fields_mesh'] = False

    test_obj.dt = dts[-1]/2
    test_obj.steps = np.int(end/test_obj.dt)
    spec_comp = test_obj.test_SDC()
    real_pos = spec_comp.pos[:,2]
    
    for dt in dts:
        rewind_vel = -0.5 * math.sin(-dt)
        test_obj.dt = dt
        test_obj.steps = np.int(end/dt)
        
        print('')
        print('********************** Direct Runs *************************')
        print('')
        
        test_obj.analysis_params['gather'] = 'none'
        test_obj.analysis_params['external_fields'] = True
        test_obj.analysis_params['external_fields_mesh'] = False
        spec_sync = test_obj.test_synced()
        spec_stag = test_obj.test_staggered()
        spec_sdc = test_obj.test_SDC()
        
        pos_sync.append(spec_sync.pos[:,2])
        pos_stag.append(spec_stag.pos[:,2])
        pos_sdc.append(spec_sdc.pos[:,2])
        
        print('')
        print('*********************** PIC Runs ***************************')
        print('')
        
        test_obj.analysis_params['gather'] = 'trilinear_gather'
        test_obj.analysis_params['external_fields'] = False
        test_obj.analysis_params['external_fields_mesh'] = True
        spec_sync_pic = test_obj.test_synced()
        spec_stag_pic = test_obj.test_staggered()
        spec_sdc_pic = test_obj.test_SDC()    
    
        pos_sync_pic.append(spec_sync_pic.pos[:,2])
        pos_stag_pic.append(spec_stag_pic.pos[:,2])
        pos_sdc_pic.append(spec_sdc_pic.pos[:,2])
        
        
    pos_sync = np.array(pos_sync)
    pos_stag = np.array(pos_stag)
    pos_sdc = np.array(pos_sdc)
    pos_sync_pic = np.array(pos_sync_pic)
    pos_stag_pic = np.array(pos_stag_pic)
    pos_sdc_pic = np.array(pos_sdc_pic)
    
    zRel_sync = np.abs(pos_sync - spec_comp.pos[:,2])
    zRel_stag = np.abs(pos_stag- spec_comp.pos[:,2])
    zRel_sdc = np.abs(pos_sdc - spec_comp.pos[:,2])
        
    zRel_sync_pic = np.abs(pos_sync_pic - spec_comp.pos[:,2])
    zRel_stag_pic = np.abs(pos_stag_pic- spec_comp.pos[:,2])
    zRel_sdc_pic = np.abs(pos_sdc_pic - spec_comp.pos[:,2])
    
    ##Order Plot w/ dt
    fig_dt = plt.figure(2)
    ax_dt = fig_dt.add_subplot(1, 1, 1)
    ax_dt.plot(dts,zRel_sync[:,0],label='Synced')
    ax_dt.plot(dts,zRel_stag[:,0],label='Staggered')
    ax_dt.plot(dts,zRel_sync_pic[:,0],label='Synced PIC')
    ax_dt.plot(dts,zRel_stag_pic[:,0],label='Staggered PIC')
    ax_dt.plot(dts,zRel_sdc[:,0],label='SDC')
    ax_dt.plot(dts,zRel_sdc_pic[:,0],label='SDC PIC')
    
    ## Order plot finish
    dHandler = DH()
    ax_dt.set_xscale('log')
    #ax_dt.set_xlim(10**-3,10**-1)
    ax_dt.set_xlabel('$\Delta t$')
    ax_dt.set_yscale('log')
    #ax_dt.set_ylim(10**(-7),10**1)
    ax_dt.set_ylabel('$\Delta x^{(rel)}$')
    
    xRange = ax_dt.get_xlim()
    yRange = ax_dt.get_ylim()
    
    ax_dt.plot(xRange,dHandler.orderLines(1,xRange,yRange),
                ls='-',c='0.25',label='1st Order')
    ax_dt.plot(xRange,dHandler.orderLines(2,xRange,yRange),
                ls='dotted',c='0.25',label='2nd Order')
    ax_dt.plot(xRange,dHandler.orderLines(4,xRange,yRange),
                ls='dashed',c='0.75',label='4th Order')
    ax_dt.plot(xRange,dHandler.orderLines(8,xRange,yRange),
                ls='dashdot',c='0.1',label='8th Order')
    ax_dt.legend()



    
    
    
    
    
    