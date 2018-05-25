#!/usr/bin/env python3

## Dependencies
import numpy as np
import matplotlib.pyplot as plt

## Required modules
from species import species
from simulationManager import simulationManager
from dataHandler import dataHandler
from caseHandler import caseHandler
from kpps_analysis import kpps_analysis

class Test_units:
    def test_species(self):
        particles = species()
        assert particles.nq == 1
        assert particles.q == 1
        assert particles.mq == 1
        
        particles = species(nq=10,q=5, mq=2.5)
        assert particles.nq == 10
        assert particles.q == 5
        assert particles.mq == 2.5
        
        assert (particles.B == particles.E).all()
        assert (particles.E == particles.F).all()
        assert (particles.F == particles.vel).all()
        assert (particles.vel == particles.pos).all()
        
        assert particles.qtype == 'custom'
        
        
    def test_simManager(self):
        sim = simulationManager()
        assert sim.tStart == 0
        assert sim.tEnd == 1
        assert sim.tSteps == 100
        assert sim.dt == 0.01
        
        assert sim.ts == 0
        sim.updateTime()
        sim.updateTime()
        sim.updateTime()
        assert sim.ts == 3
        assert sim.t == 0.03
        assert sim.tArray == [0,0.01,0.02,0.03]
        
        sim = simulationManager(tEnd=5,tSteps=100)
        assert sim.dt == 0.05
        
        sim = simulationManager(dt=0.1,tSteps=100)
        assert sim.tEnd == 10

        sim = simulationManager(dt=0.1,tEnd=20)
        assert sim.tSteps == 200
        
        
    def test_caseHandler(self):
        particles = species(nq=4)
        case = caseHandler(particles,dimensions=2,distribution={'random':''})
        assert (particles.pos[:,0] != np.array([0,0,0,0])).all()
        assert (particles.pos[:,1] != np.array([0,0,0,0])).all()
        assert (particles.pos[:,2] == np.array([0,0,0,0])).all()
        
        positions = np.array([[1,2,3],[4,5,6]],dtype=np.float)
        velocities = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]]) 
        particles = species(nq=4)
        case = caseHandler(particles,
                           dimensions=3,
                           explicitSetup={'positions':positions,
                                          'velocities':velocities})
    
        assert (particles.pos[:2,:] == positions).all()
        assert (particles.pos[2:4,:] == 0).all()
        assert (particles.vel[:2,:] == velocities).all()
        assert (particles.vel[2:4,:] == 0).all()