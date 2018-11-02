import numpy as np
from species import species
from mesh import mesh
from caseHandler import caseHandler
from simulationManager import simulationManager

from kpps_analysis import kpps_analysis

class Test_fields:
    def setup(self):
        species_params = {}
        case_params = {}
        
        species_params['nq'] = 8
        species_params['mq'] = 1
        species_params['q'] = 8
        self.spec_params = species_params
        
        case_params['particle_init'] = 'direct'
        case_params['vel'] = np.zeros((8,3),dtype=np.float)
        case_params['pos'] = np.array([[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],
                                      [-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1]])
        
        case_params['mesh_init'] = 'box'
        case_params['xlimits'] = [-2,2]
        case_params['ylimits'] = [-2,2]
        case_params['zlimits'] = [-2,2]
        case_params['resolution'] = [2]
        case_params['store_node_pos'] = True
        
        self.case_params = case_params
        
        self.sim = simulationManager()
        self.kpa = kpps_analysis()
        
    def test_trilinearScatter(self):
        p = species(**self.spec_params)
        m = mesh()
        
        # Simple test - '8 particles, 8 cells, 1 cube'
        case = caseHandler(species=p,mesh=m,**self.case_params)
        
        assert m.q[0,0,0] == 0
        self.kpa.trilinear_qScatter(p,m,self.sim)
        assert m.q[0,0,0] == 1/m.dv
        assert m.q[1,1,1] == 8/m.dv
        assert m.q[1,1,0] == 4/m.dv
        assert m.q[1,0,0] == 2/m.dv
        
        # Distribution test - does the scattered charge add back up?
        p.nq = 20
        p.q = 1
        
        case2_params = self.case_params
        case2_params['particle_init'] = 'random'
        case2_params['pos'] =  np.array([[0,0,0]])
        case2_params['dx'] = 2

        case2 = caseHandler(species=p,mesh=m,**case2_params)
        self.kpa.trilinear_qScatter(p,m,self.sim)
        charge_sum = np.sum(m.q)

        assert 19.99/m.dv <= charge_sum <= 20.01/m.dv
        
        # 2D test - does the function work just in the plane?
        p.nq = 4
        p.q = 1

        case3_params = self.case_params
        case3_params['particle_init'] = 'direct'
        case3_params['pos'] =  np.array([[0.5,0.5,0],[1.5,0.5,0],
                                        [0.5,1.5,0],[1.5,1.5,0]])

        case3 = caseHandler(species=p,mesh=m,**case3_params)

        self.kpa.trilinear_qScatter(p,m,self.sim)

        assert m.q[:,:,2].all() == np.zeros((3,3)).all() 
        assert m.q[:,:,0].all() == np.zeros((3,3)).all() 
        assert m.q[1:,1:,1].all() == np.array([[1,1],[1,1]]/m.dv).all()
        assert np.sum(m.q[:,:,1]) == p.nq*p.q/m.dv
        
    def test_trilinearGather(self):
        p = species(**self.spec_params)
        m = mesh()
        case = caseHandler(species=p,mesh=m,**self.case_params)
        
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
        
tf = Test_fields()
tf.setup()
tf.test_trilinearScatter()
tf.test_trilinearGather() 
        
        
