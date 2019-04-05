import io
import pickle as pk
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mesh import mesh

def update_hist(num, data, histogram_axis):
    hist_ax.cla()
    hist_ax.hist(data[num,:],100)
    hist_ax.set_xlim([0,10])
    hist_ax.set_xlabel('No.')
    hist_ax.set_ylabel(r'f')
    hist_ax.set_ylim([0, 20])

    return hist_ax


N = 1000
Nt = 100

data = np.zeros((Nt,N),dtype=np.float)

for ti in range(Nt):
    data[ti,:] = np.random.random(N) * 10

fig = plt.figure(1,dpi=150)
hist_ax = fig.add_subplot(1,1,1)
#histogram = hist_ax.hist(data[0,:],bins=100,label=r'data histogram')[0]




fps = 10
# Creating the Animation object
hist_ani = animation.FuncAnimation(fig, update_hist, Nt, 
                                   fargs=(data,hist_ax),
                                   interval=1000/fps)



hist_ani.save('hist.mp4')
plt.show()