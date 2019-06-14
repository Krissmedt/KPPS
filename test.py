import io
import pickle as pk
import numpy as np
import time
import cmath as math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mesh import mesh

def exponential_pos(t,c1,c2):
    x = math.log(1/2 *c1*(math.tanh(1/2 * math.sqrt(c1*(c2 + t)**2))**2 - 1))
    return math.sqrt(x.real**2 - x.imag**2)
"""
def exponential_vel(t,c1,c2):
    v_num = c1*(c2+t)*math.tanh(1/2 * math.sqrt(c1*(c2+t)**2)) * math.sech(1/2 * math.sqrt(c1*(c2+t)**2))
    v_den = math.sqrt(c1*(c2+t)**2) * (math.tanh(1/2 * math.sqrt(c1*(c2+t)**2))-1)
    v1 = v_num/v_den
    v2 = -math.sqrt(c1) * math.tanh(1/2 * math.sqrt(c1)*(c2+t))
    
    return v1,v2
""" 

c1 = -10
c2 = 10
t = 0

tA = []
xA = []

dt = 0.1
for ts in range(0,100):
    t = ts*dt 
    x = exponential_pos(t,c1,c2)
    
    tA.append(t)
    xA.append(x.real)
    
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(t,x)
