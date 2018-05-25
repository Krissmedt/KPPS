from kpps_ced_ms import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np


nq = 10
mq = 1
alpha = 1.
q = alpha*mq

x0 = [10,0,0]
v0 = [100,0,100]

omegaB = 25.0
omegaE = 4.9
epsilon = -1

bMag = omegaB/alpha
eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])

tEnd = 8
dt = 0.01

#x0 = np.array([10,0,0])
#v0 = np.array([100,0,100])


## Numerical solution ##
model = dict(
        simSettings = {'tEnd':tEnd,'dt':dt},
    
        speciesSettings = {'nq':nq,'mq':mq,'q':q},
        
        caseSettings = {'dimensions':3,
                        'distribution':{'random':''}
                        #'explicitSetup':{'positions':x0,'velocities':v0}
                        },
        
        analysisSettings = {'electricField':{'general':eTransform, 'magnitude':eMag},
                            'magneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                            'timeIntegration':'boris'},
        
        dataSettings = {'write':{'sampleRate':1,'foldername':'simple'},
                        'record':{'sampleRate':1},
                        'plot':{'tPlot':'xyz'}
                        })

kppsObject = kpps(**model)
data = kppsObject.run()
    


