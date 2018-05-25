#!/usr/bin/env python3

from kpps_ced_ms import kpps

class Test_kpps:
    def test_electricField(self):
        efTest = dict(
        simSettings = {'tEnd':100,'tSteps':100000},
        
        speciesSettings = {},
        
        caseSettings = {'dimensions':2,
                        'explicitSetup':{'positions':[1,0,0]}},
        
        analysisSettings = {'electricField':{'sPenning':[1,0,0], 'magnitude':100},
                            'timeIntegration':'boris'},
        
        dataSettings = {'record':{'sampleRate':1},'tPlot':'x'})
        
        efTester = kpps(**efTest)
        efTest_data = efTester.run()
        
        for x in efTest_data.xArray:
            assert x <= 1.001
            assert x >= -1.001