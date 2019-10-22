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

def w1(x,H,C):
    w = (x**2 + 3*H*x + 2*H**2 + C)/(2*H**2)
    return w
    
def w2(x,H,C):
    w = 1-(x**2+C)/(H**2)
    return w

def w3(x,H,C):
    w = (x**2 - 3*H*x + 2*H**2 + C)/(2*H**2)
    return w

H = 1
C = H**2/4
res = 1000
L = 3*H
dx = L/res
X = []
W1 = []
W2 = []
W3 = []

for xi in range(0,res):
    x = -3*H/2 + xi*dx
    y1 = w1(x,H,C)
    y2 = w2(x,H,C)
    y3 = w3(x,H,C)
    
    X.append(x)
    W1.append(y1)
    W2.append(y2)
    W3.append(y3)
    
    
sample_x = 0
h1 = -1 - sample_x
h2 =  0 - sample_x
h3 = 1 - sample_x

w_1 = w1(h1,H,C)
w_2 = w2(h2,H,C)
w_3 = w3(h3,H,C)

print(w_1)
print(w_2)
print(w_3)
print(w_1+w_2+w_3)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X,W1,label='w1')
ax.plot(X,W2,label='w2')
ax.plot(X,W3,label='w3')

ax.legend()