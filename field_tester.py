from species import species
from kpps_analysis import kpps_analysis as kpa
from simulationManager import simulationManager
from mesh import mesh
import numpy as np


omegaB = 25.0
omegaE = 4.9
epsilon = -1

alpha = 1

eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])
bMag = omegaB/alpha

speciesSettings = {}
simSettings = {'t0':0,'tEnd':1,'dt':1,'percentBar':False}

fieldSettings = {'box':{'xlim':[-1,1],'ylim':[-1,1],'zlim':[-1,1]},
                    'resolution':[10]}
                        

analysisSettings = {'imposedElectricField':{'general':eTransform, 'magnitude':eMag},
                    'imposedMagneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                    'fieldIntegration':{'imposeFields':True}}

p = species(**speciesSettings)
sim = simulationManager(**simSettings)
analyser = kpa(sim,**analysisSettings)
fields = mesh(**fieldSettings)

fields = analyser.imposed_field_mesh(p,fields,sim)
print(fields.E[:,:,:,5])
print(fields.B[:,:,:,5])


