import io
import pickle as pk
import numpy as np
import time
import cmath as math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mesh import mesh

p = 3
k = p
yj = [-2,1,3]
yj = np.array(yj)

c = [1,1,1]
c = np.array(c)

for j in range(0,k+1):
    for m in range(0,j):
        print(m)
    for m in range(j+1,k+1):
        print(m)

y = yj*c
y_x = y.sum()