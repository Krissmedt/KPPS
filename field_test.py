import numpy as np
from species import species
from mesh import mesh
from caseHandler import caseHandler

from kpps_analysis import kpps_analysis

class Test_fields:
    def setup(self):
        species_params = {}
        case_params = {}
        
        species_params['mq'] = 1
        species_params['q'] = 8
        
        self.p = species(**species_params)
        self.m = mesh()
        
        case_params['particle_init'] = 'direct'
        case_params['pos'] = np.array([[0,0,0]])
        case_params['vel'] = np.array([[0,0,0]])
        
        case_params['mesh_init'] = 'box'
        case_params['xlimits'] = [-1,1]
        case_params['ylimits'] = [-1,1]
        case_params['zlimits'] = [-1,1]
        case_params['resolution'] = [1,1,1]
        case_params['store_node_pos'] = True
        
        self.case = caseHandler(species=self.p,mesh=self.m,**case_params)
        self.kpa = kpps_analysis()
        
    def test_trilinearScatter(self):
        assert self.m.q[0,0,0] == 0
        self.kpa.trilinear_qScatter(self.p,self.m)
        assert self.m.q[0,0,0] == 1
        
    def test_trilinearGather(self):
        self.m.E[0,:,:,:] = 1
        self.kpa.trilinear_gather(self.p,self.m)
        print(self.p.E)
        assert self.p.E[0,0] == 8
        
tf = Test_fields()
tf.setup()
tf.test_trilinearScatter()
tf.test_trilinearGather() 
        
        