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

def lower_index(pos,O,dh):
    li = np.floor((pos-O)/dh)
    li = np.array(li,dtype=np.int)
    
    return li


pos = np.array([[9.4,8.4,7.3],[0.7,9.7,2.3],[1.6,4.8,3.],[2.7,0.9,3.2],[7.4,6.5,2.3]])
O = np.array([0,0,0])
dh = 1
li = lower_index(pos,O,dh)
rpos = pos - O - li*dh