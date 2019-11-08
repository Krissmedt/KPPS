from decimal import Decimal
import io 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataHandler2 import dataHandler2
import matplotlib.animation as animation
import cmath as cm

def E(x):
    E = np.power(-x,3)
    
    return E
    
res = 10000
x = np.linspace(-1,1,res+1)
    
fig = plt.figure(1,dpi=150)
ax = fig.add_subplot(1,1,1)
ax.plot(x,E(x),c=(0.1,0.1,0.8,0.8),label="$E(x)$")
ax.set_xlabel('$x$')
ax.set_ylabel('$E_x$')
ax.set_title('Oscillating Particle Experiment (NLO)')
ax.legend()
fig.savefig('nlo_setup.svg', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')
fig.savefig('nlo_setup.png', dpi=150, facecolor='w', edgecolor='w',orientation='portrait')