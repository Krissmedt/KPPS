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
from caseFile_landau1D import *
import scipy.interpolate as scint
from math import sqrt, fsum, pi, exp, cos, sin, floor


def analytical_sol(t,omegaE,omegaB,H,x0,v0,epsilon=-1):
    omegaPlus = 1/2 * (omegaB + sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
    omegaMinus = 1/2 * (omegaB - sqrt(omegaB**2 + 4 * epsilon * omegaE**2))
    Rminus = (omegaPlus*x0[0,0] + v0[0,1])/(omegaPlus - omegaMinus)
    Rplus = x0[0,0] - Rminus
    Iminus = (omegaPlus*x0[0,1] - v0[0,0])/(omegaPlus - omegaMinus)
    Iplus = x0[0,1] - Iminus
    omegaTilde = sqrt(-2 * epsilon) * omegaE
    
    x = Rplus*cos(omegaPlus*t) + Rminus*cos(omegaMinus*t) + Iplus*sin(omegaPlus*t) + Iminus*sin(omegaMinus*t)
    y = Iplus*cos(omegaPlus*t) + Iminus*cos(omegaMinus*t) - Rplus*sin(omegaPlus*t) - Rminus*sin(omegaMinus*t)
    z = x0[0,2] * cos(omegaTilde * t) + v0[0,2]/omegaTilde * sin(omegaTilde*t)
    
    vx = Rplus*-omegaPlus*sin(omegaPlus*t) + Rminus*-omegaMinus*sin(omegaMinus*t) + Iplus*omegaPlus*cos(omegaPlus*t) + Iminus*omegaMinus*cos(omegaMinus*t)
    vy = Iplus*-omegaPlus*sin(omegaPlus*t) + Iminus*-omegaMinus*sin(omegaMinus*t) - Rplus*omegaPlus*cos(omegaPlus*t) - Rminus*omegaMinus*cos(omegaMinus*t)
    vz = x0[0,2] * -omegaTilde * sin(omegaTilde * t) + v0[0,2]/omegaTilde * omegaTilde * cos(omegaTilde*t)

    u = np.array([x,vx,y,vy,z,vz])
    energy = u.transpose() @ H @ u
    
    return u, energy 



t = 0.1
omegaB = 25.0
omegaE = 4.9
epsilon = -1
mq = 1

H1 = epsilon*omegaE**2
H = np.array([[H1,1,H1,1,-2*H1,1]])
H = mq/2 * np.diag(H[0])

x0 = np.array([[10,0,0]])
v0 = np.array([[100,0,100]])

u, energy =analytical_sol(t,omegaE,omegaB,H,x0,v0)