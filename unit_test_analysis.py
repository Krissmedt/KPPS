#!/usr/bin/env python3

## Dependencies
import numpy as np
import random as rand

## Required modules
from kpps_analysis import kpps_analysis
from species import species
from mesh import mesh

class Test_units:
    def test_poisson(self):
        spec = species()
        field = mesh()
        n = 3
        field.res = (n-1)
        field.dx = 1
        field.dy = 1
        field.dz = 1
        
        kpa = kpps_analysis()
        Fk = kpa.poisson_cube2nd_setup(spec,field)
        main_diag_i = rand.randint(0,n**3-1)
        off_diag_i = 1
        off2_diag_i = 1
        
        assert Fk[main_diag_i,main_diag_i] == -2/(field.dx+field.dy+field.dz)
        assert Fk[off_diag_i,off_diag_i+n] == 1
        assert Fk[off2_diag_i,off2_diag_i+n**2] == 1
        
        
    def test_toMethods(self):
        kpa = kpps_analysis()
        
        mesh = np.zeros((3,3,3),dtype=np.float)
        mesh[0,0,0] = 1
        mesh[1,0,0] = 2
        mesh[1,1,0] = 3
        mesh[1,1,1] = 4
        mesh[-1,-1,-2] = 5
        x = kpa.meshtoVector(mesh)
        
        assert x[0] == 1
        assert x[9] == 2
        assert x[12] == 3
        assert x[13] == 4
        assert x[25] == 5
        
        
tu = Test_units()
tu.test_toMethods()
tu.test_poisson()