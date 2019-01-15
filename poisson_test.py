import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import copy as cp
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Inputs
L = 1
nx = 20



dx = L/(nx-1) #Define spacing
x = np.linspace(0,L,nx) #Define vector of node poistions

midL = int(nx/2-1) #Determine node left of particle
midR = int(nx/2) #Determine node right of particle

# Construct sparse 1D finite difference matrix for discretised Poisson's equation in 1D:
# (phi_i-1 - 2*phi_i + phi_i+1)/Delta_x^2 = 4*pi*rho,
# where phi is the electric potential and rho is the charge density.
diag = [1,-2,1]
Dk = sps.diags(diag,offsets=[-1,0,1],shape=(nx-2,nx-2))/dx**2
DK_show = Dk.toarray()

# Establish charge vector for particle at mid-domain (L/2) of charge q = 1,
# for even number of nodes (the two middle nodes each gets half charge)
q = np.zeros(nx,dtype = np.float)
q[midL] = 0.5
q[midR] = 0.5

rho = q/dx # charge density = charge divided by cell length in 1D


# Initialise the electric potential vector with boundary conditions given by
# the exact solution phi_exact = -1/r, where r is the distance to the particle.
phi = np.zeros(nx,dtype = np.float)
phi[0] = -1/(L/2)
phi[-1] = -1/(L/2)

# Establish boundary condition vector by moving the BC's from the boundary 
# nodes to the relevant interior node positions and divide by the right factor.
BC = np.zeros(nx,dtype = np.float)
BC[1] += phi[0]/dx**2
BC[-2] += phi[-1]/dx**2

# Define right hand side of Ax = b (Dk * phi = b in this case) in accordance
# with FDM version of Poisson's equation.
b = rho[1:-1]*4*math.pi - BC[1:-1]

# Calculate numerical solution via scipy sparse solver and exact solution.
phi[1:-1] = spla.spsolve(Dk,b)
phi_exact = -1/abs(x-L/2)

# Plot and compare
fig = plt.figure(1)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x,phi,label='potential numerical')
ax1.plot(x,phi_exact,label='potential exact')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('E potential (statV/cm)')
ax1.legend()
ax1.set_title('Electric Potential in 1D for unit charge particle at x = ' 
              + str(L/2) + ' and Nx = ' + str(nx),fontsize=16)
plt.show()