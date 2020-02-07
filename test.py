import io
import pickle as pk
import numpy as np
import time
import copy
import cmath as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh
from species import species
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor

def periodic_particles2(species,axis,limits,**kwargs):
    for pii in range(0,species.nq):
        if species.pos[pii,axis] < limits[0]:
            overshoot = limits[0]-species.pos[pii,axis]
            species.pos[pii,axis] = limits[1] - overshoot % (limits[1]-limits[0])

        elif species.pos[pii,axis] >= limits[1]:
            overshoot = species.pos[pii,axis] - limits[1]
            species.pos[pii,axis] = limits[0] + overshoot % (limits[1]-limits[0])
            
def periodic_particles(species,axis,limits,**kwargs):
        undershoot = limits[0]-species.pos[:,axis]
        cross = np.argwhere(undershoot>0)
        species.pos[cross,axis] = limits[1] - undershoot[cross] % (limits[1]-limits[0])

        overshoot = species.pos[:,axis] - limits[1]
        cross = np.argwhere(overshoot>=0)
        species.pos[cross,axis] = limits[0] + overshoot[cross] % (limits[1]-limits[0])
            
            

prtls = species()
limits = np.array([0,1])
prtls.pos = np.random.rand(200000,3)*1.5 - 0.2
prtls.nq = prtls.pos.shape[0]

pos_out_pre = copy.deepcopy(prtls.pos)

t1 = time.time()
periodic_particles(prtls,0,limits)
t2 = time.time()

pos_out = prtls.pos
prtls.pos[:,0] = pos_out_pre[:,0]

t3 = time.time()
periodic_particles2(prtls,0,limits)
t4 = time.time()

pos_out2 = prtls.pos



run_time_opt = t2-t1
run_time_unopt = t4-t3

print(run_time_opt/run_time_unopt * 100)