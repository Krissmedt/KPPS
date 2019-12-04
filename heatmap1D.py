import numpy as np
import scipy.sparse as sps
from math import sqrt, fsum, pi
import matplotlib.pyplot as plt
import scipy.interpolate as scint

def trig(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

def hill(x, y,z):
    return -x**2  - y**2 - z**2


func = hill
n = 1000

grid_x, grid_y, grid_z = np.mgrid[-1:1:100j, 
                          -1:1:100j,-1:1:100j]


points = np.random.rand(n,3)*2
points = points-1

values = func(points[:,0], points[:,1],points[:,2])

grid_z0 = scint.griddata(points, values, (grid_x, grid_y, grid_z), method='nearest')
grid_z1 = scint.griddata(points, values, (grid_x, grid_y,grid_z), method='linear')
#grid_z2 = scint.griddata(points, values, (grid_x, grid_y,grid_z), method='cubic')

plt.subplot(221)
plt.imshow(func(grid_x, grid_y,grid_z)[:,50,:].T, extent=(-1,1,-1,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0[:,50,:].T, extent=(-1,1,-1,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1[:,50,:].T, extent=(-1,1,-1,1), origin='lower')
plt.title('Linear')
#plt.subplot(224)
#plt.imshow(grid_z2.T, extent=(-1,1,-1,1), origin='lower')
#plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()