#!/usr/bin/env python3

## Dependencies
import numpy as np
import random as rand
import copy as cp

## Required modules
from kpps_analysis import kpps_analysis
from species import species
from mesh import mesh
from controller import controller
from meshLoader import meshLoader
from particleLoader import particleLoader
import math

class Test_units_analysis:
    def setup(self):
        species_params = {}
        pLoader_params = {}
        mLoader_params = {}
        sim_params = {}
        
        sim_params['xlimits'] = [-2,2]
        sim_params['ylimits'] = [-2,2]
        sim_params['zlimits'] = [-2,2]
        self.sim = controller(**sim_params)
        
        species_params['nq'] = 8
        species_params['mq'] = 1
        species_params['q'] = 8
        self.p = species(**species_params)
        
        pLoader_params['load_type'] = 'direct'
        pLoader_params['vel'] = np.zeros((8,3),dtype=np.float)
        pLoader_params['pos'] = np.array([[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],
                                      [-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1]])
        self.pLoader_params = pLoader_params
        self.pLoader = particleLoader(**pLoader_params)
        
        mLoader_params['load_type'] = 'box'
        mLoader_params['resolution'] = [2]
        mLoader_params['store_node_pos'] = True
        self.mLoader_params = mLoader_params
        self.mLoader = meshLoader(**mLoader_params)
        self.m = mesh()

        self.kpa = kpps_analysis()
        
        
    def test_trilinearScatter(self):
        p = cp.copy(self.p)
        m = cp.copy(self.m)
        
        self.pLoader.run([p],self.sim)
        self.mLoader.run(m,self.sim)
        
        # Simple test - '8 particles, 8 cells, 1 cube'
        assert m.q[0,0,0] == 0
        self.kpa.trilinear_qScatter([p],m,self.sim)
        assert m.q[0,0,0] == 1.
        assert m.q[1,0,0] == 2.
        assert m.q[1,1,0] == 4.
        assert m.q[1,1,1] == 8.


        
        # Distribution test - does the scattered charge add back up?
        p.nq = 20
        p.q = 1
        
        pLoader_params = cp.copy(self.pLoader_params)
        pLoader_params['load_type'] = 'randDis'
        pLoader_params['pos'] =  np.array([[0,0,0]])
        pLoader_params['dx'] = 2

        pLoader = particleLoader(**pLoader_params)
        pLoader.run([p],self.sim)
        self.mLoader.run(m,self.sim)
        
        self.kpa.trilinear_qScatter([p],m,self.sim)
        charge_sum = np.sum(m.q)
        
        assert 19.99 <= charge_sum <= 20.01
        
        # 2D test - does the function work just in the plane?
        p.nq = 4
        p.q = 1

        pLoader_params = cp.copy(self.pLoader_params)
        pLoader_params['load_type'] = 'direct'
        pLoader_params['pos'] =  np.array([[0.5,0.5,0],[1.5,0.5,0],
                                        [0.5,1.5,0],[1.5,1.5,0]])

        pLoader = particleLoader(**pLoader_params)
        pLoader.run([p],self.sim)
        self.mLoader.run(m,self.sim)
        self.kpa.trilinear_qScatter([p],m,self.sim)

        assert m.q[:,:,2].all() == np.zeros((3,3)).all() 
        assert m.q[:,:,0].all() == np.zeros((3,3)).all() 
        assert np.sum(m.q[:,:,1]) == p.nq*p.q
        
        
    def test_trilinearGather(self):
        p = cp.copy(self.p)
        m = cp.copy(self.m)
        self.pLoader.run([p],self.sim)
        self.mLoader.run(m,self.sim)
        
        m.E[0,:,:,:] = 1
        m.E[1,:,:,:] = 2
        m.E[2,:,:,:] = 3
        self.kpa.trilinear_gather(p,m)
        assert p.E[0,0] == 1
        assert p.E[0,1] == 2
        assert p.E[0,2] == 3
        
        m.E[:,:,:,0] = 1
        m.E[:,:,:,1] = 2
        m.E[:,:,:,2] = 3

        p.nq = 4
        p.E = np.zeros((p.nq,3))
        p.pos = np.array([[1,1,-1.5],[1,1,-0.5],
                          [1,1,0.5],[1,1,1.5]],dtype=np.float)
        self.kpa.trilinear_gather(p,m)

        linearIncrease = np.array([1.25,1.75,2.25,2.75])
        assert p.E[:,0].all() == linearIncrease.all()
        assert p.E[:,1].all() == linearIncrease.all()
        assert p.E[:,2].all() == linearIncrease.all()
        
        
    def test_polyGather(self):

        pos = 0.2
        
        species_params = {}
        pLoader_params = {}
        
        species_params['nq'] = 1
        species_params['mq'] = 1
        species_params['q'] = 1
        p = species(**species_params)
        
        pLoader_params['load_type'] = 'direct'
        pLoader_params['vel'] = np.zeros((species_params['nq'],3),dtype=np.float)
        pLoader_params['pos'] = np.array([[0,0,pos]])
        pLoader = particleLoader(**pLoader_params)
        
        m = cp.copy(self.m)
        
        mLoader_params = cp.copy(self.mLoader_params)
        mLoader_params['resolution'] = [2,2,10]
        mLoader = meshLoader(**mLoader_params)
        
        pLoader.run([p],self.sim)
        mLoader.run(m,self.sim)
        self.kpa.mesh_boundary_z = 'open'
        
        k = 5
        k_gather = 5
        m.E[2,1,1,:] = m.z[1,1,:]**k
        self.kpa.gather_order = k_gather
        self.kpa.poly_interpol_setup([p],m,self.sim)
        self.kpa.poly_gather_1d_odd(p,m)
        assert np.all(np.isclose(p.E,p.pos**k))

        k = 4
        k_gather = 4
        m.E[2,1,1,:] = m.z[1,1,:]**k
        self.kpa.gather_order = k_gather
        self.kpa.poly_interpol_setup([p],m,self.sim)
        self.kpa.poly_gather_1d_even(p,m)
        assert np.all(np.isclose(p.E,p.pos**k))

        
    def test_polyScatter(self):
        pos = -0.1
        
        species_params = {}
        pLoader_params = {}
        
        species_params['nq'] = 1
        species_params['mq'] = 1
        species_params['q'] = 10
        p = species(**species_params)
        
        pLoader_params['load_type'] = 'direct'
        pLoader_params['vel'] = np.zeros((species_params['nq'],3),dtype=np.float)
        pLoader_params['pos'] = np.array([[0,0,pos]])
        pLoader = particleLoader(**pLoader_params)
        
        m = cp.copy(self.m)
        
        mLoader_params = cp.copy(self.mLoader_params)
        mLoader_params['resolution'] = [2,2,10]
        mLoader = meshLoader(**mLoader_params)
        
        pLoader.run([p],self.sim)
        mLoader.run(m,self.sim)
        self.kpa.mesh_boundary_z = 'open'
        
        k = 2
        self.kpa.scatter_order = k
        self.kpa.poly_interpol_setup([p],m,self.sim)
        self.kpa.quadratic_qScatter_1d([p],m,self.sim)
        print(m.q[1,1,:])
        print(m.q.sum())

        
    def test_poisson_setup_fixed(self):
        p = species()
        m = mesh()
        
        sim_params = {}
        sim_params['xlimits'] = [-1,1]
        sim_params['ylimits'] = [-1,1]
        sim_params['zlimits'] = [-1,1]
        sim = controller(**sim_params)
        
        
        mLoader_params = cp.copy(self.mLoader_params)
        mLoader_params['res'] = [4,6,8]
        mLoader = meshLoader(**mLoader_params)
        mLoader.run(m,sim)

        kpa = kpps_analysis()
        sim.ndim = 1
        Dk = kpa.poisson_cube2nd_setup(p,m,sim)
        Dk = Dk.toarray()
        
        sim.ndim = 2
        Ek= kpa.poisson_cube2nd_setup(p,m,sim)
        Ek = Ek.toarray()
        
        sim.ndim = 3
        Fk = kpa.poisson_cube2nd_setup(p,m,sim)
        Fk = Fk.toarray()

        #Test all FDM matrices are the correct size
        assert np.all(Dk.shape == m.zres-1)
        assert np.all(Ek.shape == (m.yres-1)*(m.zres-1))
        assert np.all(Fk.shape == (m.yres-1)*(m.zres-1)*(m.xres-1))
        
        #Test 1D matrix values
        assert Dk[1,0] == 1/m.dz**2
        assert Dk[1,1] == -2*(1/m.dz**2)
        assert Dk[1,2] == 1/m.dz**2
        
        #Test 2D matrix values
        assert Ek[1,0] == 1/m.dz**2
        assert Ek[1,1] == -2*(1/m.dy**2+1/m.dz**2)
        assert Ek[1,2] == 1/m.dz**2
        assert Ek[1,1+np.int(m.zres-1)] == 1/m.dy**2
        
        #Test 3D matrix values
        assert Fk[1,0] == 1/m.dz**2
        assert Fk[1,1] == -2*(1/m.dx**2+1/m.dy**2+1/m.dz**2)
        assert Fk[1,2] == 1/m.dz**2
        assert Fk[1,1+np.int(m.zres-1)] == 1/m.dy**2
        assert Fk[1,1+np.int((m.zres-1)*(m.yres-1))] == 1/m.dx**2
        
        return Dk,Ek,Fk
        
    def test_indexMethods(self):
        O = np.array([0,0,0])
        dh = 1
        pos = np.array([0.9,1.4,2.5])
        
        kpa = kpps_analysis()
        
        li = kpa.lower_index(pos,O,dh)
        ui = kpa.upper_index(pos,O,dh)
        ci = kpa.close_index(pos,O,dh)
        
        assert (li==[0,1,2]).all()
        assert (ui==[1,2,3]).all()
        assert (ci==[1,1,3]).all()
        
        
    def test_toMethods(self):
        kpa = kpps_analysis()
        
        m = np.zeros((3,3,3),dtype=np.float)
        m[0,0,0] = 1
        m[1,0,0] = 2
        m[1,1,0] = 3
        m[1,1,1] = 4
        m[-1,-1,-2] = 5
        x = kpa.meshtoVector(m)
        
        assert x[0] == 1
        assert x[9] == 2
        assert x[12] == 3
        assert x[13] == 4
        assert x[25] == 5
        
        
    def test_external_E(self):
        spec = species()
        field = mesh()
        kpa = kpps_analysis()
        
        spec.pos = np.array([[10,0,0]])
        spec.a = 1
        spec.nq = spec.pos.shape[0]
        
        omegaB = 25
        omegaE = 4.9
        epsilon = -1

        kpa.E_type = 'transform'
        kpa.E_transform = np.array([[1,0,0],[0,1,0],[0,0,-2]])
        kpa.E_magnitude = -epsilon*omegaE**2/spec.a
        kpa.B_type = 'uniform'
        kpa.B_transform = np.array([0,0,1])
        kpa.B_magnitude = omegaB/spec.a
        
        spec = kpa.eFieldImposed(spec,field)
        
        assert spec.E.shape == (1,3)
        assert math.isclose(spec.E[0,0],240.1)
        
test = Test_units_analysis()
test.setup()
test.test_trilinearGather()