import numpy as np

class problem(object):

  def __init__(self):
    self.e_z = np.array([0, 0, 1])
    self.omega_b = 25.0
    self.alpha   = 1.0
    self.epsilon = -1.0
    self.omega_e = 4.9
    self.E_mat = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, -2] ])
    self.H_mat = np.eye(6)
    self.H_mat[0,0] = self.epsilon*self.omega_e**2
    self.H_mat[1,1] = self.epsilon*self.omega_e**2
    self.H_mat[2,2] = -2*self.epsilon*self.omega_e**2
    self.x0 = np.array([10, 0, 0])
    self.v0 = np.array([100, 0, 100])
  
  def getB(self,x):
    return (self.omega_b/self.alpha)*self.e_z

  def getE(self,x):
    return -self.epsilon*(self.omega_e**2/self.alpha)*self.E_mat.dot(x)

  def getEnergy(self, x, v):
    u = np.zeros(6)
    u[0:3] = x
    u[3:6] = v
    return np.transpose(u).dot( self.H_mat.dot(u) )

  def x(self,t):
    return np.ones(3) # complete later

  def v(self,t):
    return np.ones(3) # complete later

