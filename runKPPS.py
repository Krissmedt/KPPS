from kpps_ced_ms import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np

nq = 1
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


## Analytical solution ##
x0 = np.array([10,0,0])
v0 = np.array([100,0,100])
omegaTilde = sqrt(-2 * epsilon) * omegaE
omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
Rminus = (omegaPlus*x0[0] + v0[1])/(omegaPlus - omegaMinus)
Rplus = x0[0] - Rminus
Iminus = (omegaPlus*x0[1] + v0[0])/(omegaPlus - omegaMinus)
Iplus = x0[1] - Iminus

xAnalyt = Rplus*cos(omegaPlus*tEnd) + Rminus*cos(omegaMinus*tEnd) + Iplus*sin(omegaPlus*tEnd) + Iminus*sin(omegaMinus*tEnd)
yAnalyt = Iplus*cos(omegaPlus*tEnd) + Iminus*cos(omegaMinus*tEnd) - Rplus*sin(omegaPlus*tEnd) - Rminus*sin(omegaMinus*tEnd)
zAnalyt = x0[2] * cos(omegaTilde * tEnd) + v0[2]/omegaTilde * sin(omegaTilde*tEnd)


model = dict(
        simSettings = {'tEnd':16,'tSteps':1000},
    
    speciesSettings = {'nq':1,'mq':mq,'q':q},
            
        caseSettings = {'dimensions':3,
                        'explicitSetup':{'positions':x0,'velocities':v0}},
        
        analysisSettings = {'electricField':{'general':eTransform, 'magnitude':eMag},
                            'magneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                            'timeIntegration':'boris'},
        
        dataSettings = {'record':{'sampleRate':1},
                        'plot':{'tPlot':'xyz'}})


kpps = kpps(**model)
data = kpps.run()

check = data.xArray