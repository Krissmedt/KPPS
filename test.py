from kpps_ced_ms import kpps


model = dict(
simSettings = {'ndim':3,'tEnd':10,'tSteps':100},

speciesSettings = {'nq':1,'qtype':'proton'},

caseSettings = {'distribution':{'random':''},
                'explicitSetup':{'velocities':[0.,1.,0.]}},

analysisSettings = {'electricField':{'ftype':'sPenning', 'magnitude':1000},
                    'interactionModelling':'intra',
                    'magneticField':{'uniform':[0,0,1], 'magnitude':1000},
                    'timeIntegration':'boris'},

dataSettings = {'write':{'sampleRate':1,'foldername':'simple'},
                'record':{'sampleRate':1},
                'plot':{'tPlot':'xyz','sPlot':''}})


kpps = kpps(**model)
kpps.run()