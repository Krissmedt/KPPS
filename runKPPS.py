from kpps_ced_ms import kpps


model = dict(
        simSettings = {'tEnd':100,'tSteps':10000},
    
        speciesSettings = {},
        
        caseSettings = {'dimensions':2,
                        'explicitSetup':{'positions':[1,0,0]}},
        
        analysisSettings = {'electricField':{'sPenning':[1,0,0], 'magnitude':100},
                            'timeIntegration':'boris'},
        
        dataSettings = {'record':{'sampleRate':1},
                        'plot':{'tPlot':'x'}})

kpps = kpps(**model)
data = kpps.run()

check = data.xArray