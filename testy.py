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
from gauss_lobatto import CollGaussLobatto


a = np.array([1,2,3,4,5,6])
b = a[::2]
print(b)