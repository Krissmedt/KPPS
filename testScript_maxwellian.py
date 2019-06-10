import io
import pickle as pk
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh


vth = 0.5
nq = 2**14
L = 1
n = 500/L
bins = 100

A = n/(np.sqrt(math.pi)*vth)
U = np.linspace(-1,1,100)
f1 = np.zeros(100)
v = np.zeros(nq)
u = 0

v[:] = u

for i in range(0,100):
    f1[i] = A*math.exp(-U[i]**2/vth**2)
    

for pii in range(0,nq-1,2):
    U1 = random.random()
    U2 = random.random()
    Z0 = np.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)
    Z1 = np.sqrt(-2*math.log(U1))*math.sin(2*math.pi*U2)
    
    V0 = vth/np.sqrt(2) * Z0
    V1 = vth/np.sqrt(2) * Z1
    v[pii] += V0
    v[pii+1] += V1
    
mean = np.mean(v)
lowest = np.min(v)
highest = np.max(v)
variance = np.sum((v-mean)**2)/nq
real_variance = vth**2/2
sub_length = L/20
sub_index_length = np.int(nq/20)
sil_half = np.int(sub_index_length/2)
min_index = np.int(nq/20/2+1)
max_index = np.int(nq-nq/(20*2)-1)
random_mid = min_index + np.int(random.random()*(nq-2*min_index))
v_sub = v[random_mid-sil_half:random_mid+sil_half]
mean_sub = np.mean(v_sub)
variance_sub = np.sum((v_sub-mean_sub)**2)/(nq/20)

print("The sample mean is "+str(mean) + ", vs. real value 0")
print("The sub-sample mean is "+str(mean_sub) + ", vs. real value 0")
print("The sample variance is "+str(variance) + ", vs. real value " + str(real_variance))
print("The sub-sample variance is "+str(variance_sub) + ", vs. real value " + str(real_variance))
print("The sample lowest extremity is "+str(lowest))
print("The sample largest extremity is "+str(highest))


fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.plot(U,f1)
ax.hist(v,bins)
#plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(1,1,1)
ax.hist(v,bins)

fig = plt.figure(3)
ax = fig.add_subplot(1,1,1)
ax.hist(v_sub,bins)
plt.show()
