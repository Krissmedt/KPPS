import numpy as np
import matplotlib.pyplot as plt
import random


def lagrange_poly(x,k,xp,yp):
    yj = np.array(yp)
    xj = np.array(xp)
    
    c = np.ones(k+1)
    
    for j in range(0,k+1):
        for m in range(0,j):
            c[j] *= (x - xj[m])/(xj[j] - xj[m])
        for m in range(j+1,k+1):
            c[j] *= (x - xj[m])/(xj[j] - xj[m])
    
    y = yj*c
    y_x = y.sum()
    
    return y_x


L = 2.
res = 1000
dx = L/res

k = 3
xp = np.linspace(0,L,k+1)

yRange = [1,2,3,4,5,6,7,8,9,10,11,12]
yp = random.sample(yRange,k+1)
#yp = [5,7,3,1]


y_array = []
x_array = []

for xi in range(0,res):
    x = xi*dx
    y = lagrange_poly(x,k,xp,yp)
    
    x_array.append(x)
    y_array.append(y)
    

fig_dt = plt.figure(1)
ax_dt = fig_dt.add_subplot(1, 1, 1)
ax_dt.plot(x_array,y_array)
