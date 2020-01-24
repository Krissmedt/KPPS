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
#from caseFile_landau1D import *
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor




L = 4*pi
x = np.linspace(0,L,1000)

k = 0.5
a = 0.05

cosx = 1+ a*np.cos(k*x)

fig = plt.figure(4)
ax_pos = fig.add_subplot(111)
ax_pos.plot(x,cosx)
ax_pos.legend()
