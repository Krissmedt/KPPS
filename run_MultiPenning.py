from kpps import kpps
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

nq = 2
#schemes = {'lobatto':'boris_SDC','legendre':'boris_SDC','boris':'boris_synced'}
schemes = {'legendre':'boris_SDC'}

M = 3
iterations = [3]

tEnd = 1
#tEnd = 16.0
#dt = np.array([12.8,6.4,3.2,1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125])
#dt = np.array([0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4])
#dt = dt/omegaB 
                        
dt = np.array([0.01])

sampleRate = 1

log = False
partTraj = np.linspace(1,nq,nq,dtype=np.int)


mq = 1.
alpha = 10
q = alpha*mq

omegaB = 25.0
omegaE = 4.9
epsilon = -1

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])
H = np.kron(H,np.identity(nq))

bMag = omegaB/alpha
eMag = -epsilon*omegaE**2/alpha
eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])


x0 = [[10,0,0],[10.1,0.1,0.1]]
v0 = [[100,0,100],[100,0,100]]
dx = 0.001
dv = 5

tsteps = floor(tEnd/dt[-1]) +1



## Numerical solution ##
figNo = 50
for key, value in schemes.items():
    for K in iterations:
        tNum = []
        xNum = []
        yNum = []
        zNum = []
        dNum = []
        for i in range(0,len(dt)): 
            finalTs = floor(tEnd/dt[i])
            
            label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
            label = label_order + ", dt=" + str(dt[-1])
            
            model = dict(
                    simSettings = {'t0':0,'tEnd':tEnd,'dt':dt[i],'id':label,'percentBar':False},
                
                    speciesSettings = {'nq':nq,'mq':mq,'q':q},
                    
                    fieldSettings = {'box':{'xlim':[-1,1],'ylim':[-1,1],'zlim':[-1,1]},
                                     'resolution':[10]},
                    
                    caseSettings = {'dimensions':3,
                                    'explicit':{'expType':'direct','positions':x0,'velocities':v0},
                                    'dx':dx,'dv':dv},
                    
                    analysisSettings = {'imposedElectricField':{'general':eTransform, 'magnitude':eMag},
                                        'interactionModelling':'intra',
                                        'imposedMagneticField':{'uniform':[0,0,1], 'magnitude':bMag},
                                        'fieldIntegration':{'imposeFields':True},
                                        'particleIntegration':value,
                                        'M':M,
                                        'K':K,
                                        'nodeType':key,
                                        'penningEnergy':H,
                                        'centreMass':True,
                                        'units':' '},
                    
                    dataSettings = {#'write':{'sampleRate':1,'foldername':'simple'},
                                    'record':{'sampleInterval':sampleRate}
                                    ,'plot':{'tPlot':'xyz'}
                                    ,'trajectory_plot':{'particles':partTraj,'limits':[20,20,15]}
                                    })
            
            kppsObject = kpps(**model)
            data = kppsObject.run()
            data.convertToNumpy()
    
            if log == True:
                filename = key + "_" + value + "_"  + str(M) + "_" + str(K) + "_" + str(dt[i]) + "dt.txt"              
                np.savetxt('x_' + filename,data.xArray)
                np.savetxt('y_' + filename,data.yArray)
                np.savetxt('z_' + filename,data.zArray)
                np.savetxt('h_' + filename,data.hArray)
                np.savetxt('cm_' + filename,data.cmArray)
            
            label_order = key + "-" + value + ", M=" + str(M) + ", K=" + str(K)
            label_traj = label_order + ", dt=" + str(dt[-1])
            

            ## CoM Trajectory Plot
            fig = plt.figure(figNo+1)
            ax = fig.gca(projection='3d')
            ax.plot3D(data.cmArray[:,0],data.cmArray[:,1],data.cmArray[:,2],label=label_traj)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([-15,15])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            
            figNo += 2
        
