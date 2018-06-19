from boris_sdc import boris_sdc
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


tend   = 10
nsteps = 1000
dt     = tend/float(nsteps)
taxis  = np.linspace(0, tend, nsteps+1)

kiter = 3
sdc = boris_sdc(dt = dt, kiter = kiter, nsteps = nsteps, integrate=False)
boris = boris_sdc(dt = dt, kiter = kiter, nsteps = nsteps, integrate=False)
positions, velocities, stats = sdc.run()
kpositions, kvelocities, kstats = boris.boris_synced()

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot3D(positions[0,:], positions[1,:], positions[2,:])
ax.plot3D(kpositions[0,:], kpositions[1,:], kpositions[2,:])
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-15, 15])



fig = plt.figure(2)
omega = np.sqrt(-2.0*sdc.P.epsilon)*sdc.P.omega_e
vert_pos_exact = positions[2,0]*np.cos(omega*taxis) + velocities[2,0]*np.sin(omega*taxis)/omega
plt.plot(taxis, positions[2,:], 'r-')
plt.plot(taxis, kpositions[2,:], 'g-')
plt.plot(taxis, vert_pos_exact, 'b-')
plt.ylabel('z')
err_z = np.abs(vert_pos_exact[nsteps] - positions[2,nsteps])/np.abs(positions[2,nsteps])
err_kz = np.abs(vert_pos_exact[nsteps] - kpositions[2,nsteps])/np.abs(kpositions[2,nsteps])
print ("Final error in z-coordinate: %5.3e for Boris-SDC" % err_z)
print ("Final error in z-coordinate: %5.3e for Kris Boris" % err_kz)

fig = plt.figure(3)
plt.semilogy(taxis[1:], stats['energy_errors'])
plt.semilogy(taxis[1:], kstats['energy_errors'])
plt.ylabel('Energy error')


"""
fig = plt.figure(4)
plt.semilogy(taxis[1:], stats['residuals'][0,:], 'r')
#plt.semilogy(taxis[1:], stats['residuals'][1,:], 'b')
#plt.semilogy(taxis[1:], stats['residuals'][kiter-1,:], 'g')
plt.ylabel('Residuals')
"""
plt.show()

