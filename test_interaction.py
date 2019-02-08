# -*- coding: utf-8 -*-

## Dependencies
import numpy as np

## Required modules
from kpps import kpps
from kpps_analysis import kpps_analysis
from simulationManager import simulationManager as simMan

class Test_coulomb:
    def setup(self):
        self.nq = 2
        self.mq = 1
        self.q = 100
        
        self.schemes = {'lobatto':'boris_SDC',
                        'legendre':'boris_SDC',
                        'boris':'boris_synced'}
        
        self.iterations = [3]
        
        self.tEnd = 0.005
        self.dt = 0.001
        self.sampleRate = 1
        
        self.simS = {'t0':0,'tEnd':self.tEnd,'dt':self.dt,
                     'id':'int_test','percentBar':False}
    
        self.sS = {'nq':self.nq,'mq':self.mq,'q':self.q}
        
        self.aS = {'interactionModelling':'intra',
                   'particleIntegration':1,
                   'M':3,
                   'K':1,
                   'nodeType':1,
                   'units':' '}
        
        self.dS = {'record':{'sampleInterval':self.sampleRate},
                   'plot':{'tPlot':'xyz'}
                   }

        self.cS = {'dimensions':3, 'explicit':1}
        

    def test_forceVector(self):
        pos1 = np.array([-0.1,0,0])
        pos2 = np.array([0.1,0,0])
        sim = simMan(**self.simS)
        kpa = kpps_analysis(sim,**self.aS)

        
        F = kpa.coulombForce(self.q,pos1,pos2)
        assert -198.95 < F[0] < -198.93
        assert F[1] == 0
        assert F[2] == 0
        
        
    def test_behaviour(self):
        self.tolerance = 0.02
        self.target = 0.26
        maxPos = self.target + self.tolerance/2
        minPos = self.target - self.tolerance/2
        
        for key, value in self.schemes.items():
            for K in self.iterations:
                self.aS['K'] = K
                self.aS['particleIntegration'] = value
                self.aS['nodeType'] = key
                
                # Check along x-axis
                x0 = [[-0.1,0,0],[0.1,0,0]]
                self.cS['explicit'] = {'expType':'direct','positions':x0}
                kppsObject = kpps(caseSettings=self.cS, simSettings=self.simS,
                                  speciesSettings=self.sS, analysisSettings=self.aS,
                                  dataSettings=self.dS)
                data = kppsObject.run()
                data.convertToNumpy()

                assert minPos < data.xArray[5,1] < maxPos
                assert data.xArray[5,1] == -data.xArray[5,0]
                assert data.yArray[5,1] == 0
                assert data.zArray[5,1] == 0
                
                # Check along y-axis
                x0 = [[0,-0.1,0],[0,0.1,0]]
                self.cS['explicit'] = {'expType':'direct','positions':x0}
                kppsObject = kpps(caseSettings=self.cS, simSettings=self.simS,
                                  speciesSettings=self.sS, analysisSettings=self.aS,
                                  dataSettings=self.dS)
                data = kppsObject.run()
                data.convertToNumpy()
                
                assert minPos < data.yArray[5,1] < maxPos
                assert data.yArray[5,1] == -data.yArray[5,0]
                assert data.xArray[5,1] == 0
                assert data.zArray[5,1] == 0
                
                
                # Check along z-axis
                x0 = [[0,0,-0.1],[0,0,0.1]]
                self.cS['explicit'] = {'expType':'direct','positions':x0}
                kppsObject = kpps(caseSettings=self.cS, simSettings=self.simS,
                                  speciesSettings=self.sS, analysisSettings=self.aS,
                                  dataSettings=self.dS)
                data = kppsObject.run()
                data.convertToNumpy()
                
                assert minPos < data.zArray[5,1] < maxPos
                assert data.zArray[5,1] == -data.zArray[5,0]
                assert data.yArray[5,1] == 0
                assert data.xArray[5,1] == 0
                
                
ct = Test_coulomb()
ct.setup()
ct.test_forceVector()