""" This script solves the for the harmonic motion of a mass-spring oscillator
(undamped) using three different numerical schemes for time integration, to
check for their suitability in the simulation of oscillatory motion. """

## Dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import pyTools as pt
import random as rand
from copy import copy, deepcopy
from math import exp, sqrt, floor, fabs, fsum, pi, e, sin, cos

## Initialisation ## 
m = 1
k = pi**2
x0 = 1
w = sqrt(k/m)

dt = 1 *2/w
te = 100
tsteps = floor(te/dt)

t = np.zeros((tsteps,1),dtype=np.float)
x = np.zeros((tsteps,4),dtype=np.float)
v = np.zeros(4,dtype=np.float)
a = np.zeros(4,dtype=np.float)

x[0,0:4] = x0


a[2] = -k/m * x[0,2]
v[2] = v[2] + a[2]*dt

## Main Time Loop ##
for ti in range(1,tsteps):
    t[ti] = dt*(ti)
    
    #Exact solution
    x[ti,0] = x0*cos(w*t[ti])

    #Forward Euler
    a[1] = -k/m * x[ti-1,1]
    v[1] = v[1] + dt*a[1]
    x[ti,1] = x[ti-1,1] + dt*v[1]
    
    #Leapfrog 1
    x[ti,2] = x[ti-1,2] + v[2]*dt
    a[2] = -k/m * x[ti,2]
    v[2] = v[2] + a[2]*dt

    #Leapfrog 2
    x[ti,3] = x[ti-1,3] + v[3]*dt + 1/2 * (-k/m * x[ti-1,3])*dt**2
    v[3] = v[3]+ 1/2 * (-k/m * x[ti-1,3] - k/m * x[ti,3])*dt
    
    
plt.figure
exact, = plt.plot(t,x[:,0], label='Exact')
fe, = plt.plot(t,x[:,1], label='F. Euler')
lf1, = plt.plot(t,x[:,2], label='Leapfrog 1')
lf2, = plt.plot(t,x[:,3], label='Leapfrog 2')
plt.legend(handles=[exact,fe,lf1,lf2])

print(max(x[:,0])) 
print(max(x[:,1])) 
print(max(x[:,2])) 
print(max(x[:,3])) 