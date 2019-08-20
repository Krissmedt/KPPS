import io
import pickle as pk
import numpy as np
import time
import cmath as math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh

res = 11
z = np.linspace(-1,1,res)
rho = -3*np.power(z,2)
E = -np.power(z,3)
phi = -0.25 * np.power(z,4)
phi_solve = [-0.25,   -0.106,  -0.0388, -0.0148, -0.01,   -0.01,   -0.01,   -0.0148, -0.0388, -0.106,  -0.25]

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
#ax1.plot(z,rho)
#ax1.plot(z,E)
ax1.plot(z,phi)
ax1.plot(z,phi_solve)
 