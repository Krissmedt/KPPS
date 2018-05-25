#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:41:30 2018

@author: mn12kms

Spectral Deferred Corrections implmented using explicit Euler as base method
to construct alternative version of (2.7) in Ruprecht and Speck (2016). 
Uses collocation class also by Ruprecht to calculate quadrature weights (Qmj).

ODE to solve: u' = cos(x)   with    u(0) = u0
Exact solution: u = sin(x) + u0
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from Collocation import CollBase

def f(u,x,konst):
    #f = math.cos(x)
    f = -math.e**(konst*u)
    return f

konst = 0.25

def f2(u,x):
    konst = 0.25
    f = -math.e**(konst*u)
    return f

M = 8
K = 8

xend = 10
res = 10
x = np.linspace(0,xend,res)
xn = len(x)

u = np.zeros(xn,dtype=np.float)
u_exact = np.zeros(xn,dtype=np.float)
u0 = 10
u[0] = u0
u_exact[0] = u0

c1 = math.e**(-u[0]*konst)/konst


U = np.zeros(M+1,dtype=np.float)
Un = np.zeros(M+1,dtype=np.float)

F = np.zeros(M,dtype=np.float)

for xi in range(1,xn):
    coll = CollBase(M,x[xi-1],x[xi])
    [coll.nodes, coll.weights] = coll._GaussLegendre(M,
                                             coll.tleft,
                                             coll.tright)
    c = coll.nodes
    coll.Qmat = coll._gen_Qmatrix
    coll.Smat = coll._gen_Smatrix
    coll.delta_m = coll._gen_deltas
    U[:] = u[xi-1]
    Un = U
    for k in range(0,K):
        for m in range(1,M+1):
            sumInner = np.sum(coll.Smat[m,1:]*f(U[m],c[m-1],konst))
            Un[m] = Un[m-1] + coll.delta_m[m-1]*(f(Un[m-1],c[m-1],konst)-f(U[m-1],c[m-1],konst)) + sumInner
            
        U = Un
    
    for j in range(0,M):
        F[j] = f(U[j],c[j],konst)
        
    u[xi] = u[xi-1] + np.sum(np.dot(coll.weights,F))
    #u_exact[xi] = math.sin(x[xi]) + u_exact[0]
    u_exact[xi] = -1/konst * math.log(konst*(c1+x[xi]))
    


error = math.fabs(max(u_exact - u))
        
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x,u)
ax.plot(x,u_exact)
ax.set_xscale('linear')
ax.set_xlabel('$x$')
ax.set_yscale('linear')
ax.set_ylabel('$u$')  

