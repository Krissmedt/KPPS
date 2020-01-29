import io
import pickle as pk
import numpy as np
import time
import cmath as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh
from caseFile_landau1D import *
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor

L = 2*pi
tend = 30

dx_mag = 1e-4
dx_mode = 1

v = 1
dv_mag = 0
dv_mode = 1

a = -1
omega_p = 1

#Nq is particles per species, total nq = 2*nq
#ppc = 20
nq = 2000

prefix = 'TE'+str(tend) + '_a' + str(dx_mag)
simulate = True
plot = True

restart = False
restart_ts = 14

slow_factor = 1
############################ Linear Analysis ##################################
k2 = dx_mode**2
v2 = v**2

roots = [None,None,None,None]
roots[0] = cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[1] = cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[2] = -cm.sqrt(k2 * v2+ omega_p**2 + omega_p * cm.sqrt(4*k2*v2+omega_p**2))
roots[3] = -cm.sqrt(k2 * v2+ omega_p**2 - omega_p * cm.sqrt(4*k2*v2+omega_p**2))

real_slope = roots[1].imag