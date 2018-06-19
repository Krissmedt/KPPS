from computeWeights import computeWeights
from problem import problem
import numpy as np

from problem import problem

class boris_sdc(object):

  def __init__(self, dt, kiter, nsteps, integrate=True):
  
    self.dt     = dt
    self.kiter  = kiter
    self.nsteps = nsteps
  
    self.S, self.S1, self.S2, self.q = computeWeights()
    self.S  *= dt
    self.S1 *= dt
    self.S2 *= dt
    self.q  *= dt
    self.kiter = kiter
    self.integrate = integrate

    # Define the problem
    self.P = problem()

    self.stats = {}
    self.stats['residuals']     = np.zeros((kiter+1,nsteps))
    self.stats['increments']    = np.zeros((kiter,nsteps))
    self.stats['energy_errors'] = np.zeros(nsteps)

    # Buffers to store approximations at time steps
    self.positions       = np.zeros((3,nsteps+1))
    self.velocities      = np.zeros((3,nsteps+1))
    self.positions[:,0]  = self.P.x0
    self.velocities[:,0] = self.P.v0

    self.stats['exact_energy'] = self.P.getEnergy(self.positions[:,0], self.velocities[:,0])

    self.stats['errors']      = np.zeros((2,nsteps+1))
    self.stats['errors'][0,0] = 0.0
    self.stats['errors'][1,0] = 0.0

  def F(self,x,v):
    return self.P.getE(x) + np.cross(v, self.P.getB(x))
  
  def getEnergyError(self, x, v):
    return np.abs( self.P.getEnergy(x, v) - self.stats['exact_energy'])/np.abs(self.stats['exact_energy'])
  
  def updateIntegrals(self, x0, v0, x, v):
    
    F = np.zeros((3,3))
    for jj in range(3):
      F[:,jj] = self.F(x[:,jj], v[:,jj]) # F[:,jj] = F_j

    # Set integral terms to zero
    self.I_m_mp1    = np.zeros((3,3))
    self.IV_m_mp1   = np.zeros((3,3))
  
    for jj in range(3):
      # self.I_m_mp1[:,jj] equals I_j^j+1
      for kk in range(3):
        self.I_m_mp1[:,jj]    += self.S[jj,kk]*F[:,kk]
        self.IV_m_mp1[:,jj]   += self.S[jj,kk]*v[:,kk]
        
      # Compute residuals
    res_x = np.zeros((3,3))
    res_v = np.zeros((3,3))
    
    res_v[:,0] = v[:,0] - v0 - self.I_m_mp1[:,0]
    res_v[:,1] = v[:,1] - v[:,0] - self.I_m_mp1[:,1]
    res_v[:,2] = v[:,2] - v[:,1] - self.I_m_mp1[:,2]

    res_x[:,0] = x[:,0] - x0 - self.IV_m_mp1[:,0]
    res_x[:,1] = x[:,1] - x[:,0] - self.IV_m_mp1[:,1]
    res_x[:,2] = x[:,2] - x[:,1] - self.IV_m_mp1[:,2]
    
    return max(np.linalg.norm(res_v, np.inf), np.linalg.norm(res_x, np.inf))
    
  def finalUpdateStep(self, x, v, x0, v0):
    if not self.integrate:
      return x[:,2], v[:,2]
    else:
      raise

  def boris_trick(self,x_old, x_new, v_old, c_i, mydt):
    E_np12 = 0.5*( self.P.getE(x_old) + self.P.getE(x_new) )
    t      = 0.5*mydt*self.P.getB(x_new)
    s      = 2.0*t/(1.0 + np.dot(t, t))
    v_min  = v_old + 0.5*mydt*E_np12 + 0.5*c_i
    v_star = v_min + np.cross(v_min, t)
    v_plu  = v_min + np.cross(v_star, s)
    return v_plu + 0.5*mydt*E_np12 + 0.5*c_i
   
  def boris_synced(self):
    k = self.dt * 0.5
    
    for nn in range(0,self.nsteps):
      self.positions[:,nn+1] = self.positions[:,nn] + self.dt * self.velocities[:,nn]
      E_np12 = 0.5*( self.P.getE(self.positions[:,nn]) + self.P.getE(self.positions[:,nn+1]) )
      vMinus = self.velocities[:,nn] + k*E_np12
      t = k*self.P.getB(self.positions[:,nn+1])
      s = 2.0*t/(1.0+np.dot(t,t))
      
      vDash = vMinus + np.cross(vMinus,t)
      vPlus = vMinus + np.cross(vDash,s)
    
      self.velocities[:,nn+1] = vPlus + k*E_np12
      
      self.stats['energy_errors'][nn] = self.getEnergyError(self.positions[:,nn+1], self.velocities[:,nn+1])
      
      pos_ex = self.P.x(float(nn+1)*self.dt)
      vel_ex = self.P.v(float(nn+1)*self.dt)
      self.stats['errors'][0,nn+1] = np.linalg.norm( pos_ex - self.positions[:,nn+1] , np.inf )/np.linalg.norm(pos_ex, np.inf)
      self.stats['errors'][1,nn+1] = np.linalg.norm( vel_ex - self.velocities[:,nn+1] , np.inf )/np.linalg.norm(vel_ex, np.inf)
      
    return self.positions, self.velocities, self.stats
  
  def run(self):
  
    delta_tau = [self.dt*0.5, self.dt*0.5]
    
    # Buffers for k+1 and k solution at integer step
    x_old = np.zeros((3,3))
    v_old = np.zeros((3,3))
    x_new = np.zeros((3,3))
    v_new = np.zeros((3,3))
    
    for nn in range(self.nsteps):
      # Fill in initial values using end value from last step
      x_old[:,0] = self.positions[:,nn]
      v_old[:,0] = self.velocities[:,nn]
      
      '''
      Predictor step: populate self.x_old and self.v_old
      '''
      for j in range(2): # note: we use M=3 nodes here and this is hardcoded
        x_old[:,j+1] = x_old[:,j] + delta_tau[j]*v_old[:,j] + 0.5*delta_tau[j]**2*self.F(x_old[:,j], v_old[:,j])
        c_m = -0.5*delta_tau[j]*np.cross( v_old[:,j], self.P.getB(x_old[:,j]) - self.P.getB(x_old[:,j+1]))
        v_old[:,j+1] = self.boris_trick(x_old[:,j], x_old[:,j+1], v_old[:,j], c_m, delta_tau[j])
      
      '''
      SDC iteration
      '''
      for kk in range(self.kiter):
        # Update integral terms... this also computes the residuals
        self.stats['residuals'][kk,nn] = self.updateIntegrals(self.positions[:,nn], self.velocities[:,nn], x_old, v_old)

        # First value at new iteration is equal to starting value (Lobatto nodes)
        x_new[:,0]      = self.positions[:,nn]
        v_new[:,0]      = self.velocities[:,nn]

        for j in range(2):
          x_new[:,j+1] = x_new[:,j] + delta_tau[j]*( v_new[:,j] - v_old[:,j] ) + 0.5*delta_tau[j]**2*( self.F(x_new[:,j],v_new[:,j]) - self.F(x_old[:,j],v_old[:,j])) + self.IV_m_mp1[:,j+1]
          c_m  = -0.5*delta_tau[j]*np.cross(v_new[:,j],self.P.getB(x_new[:,j])-self.P.getB(x_new[:,j+1]))
          c_m += -0.5*delta_tau[j]*( self.F(x_old[:,j],v_old[:,j]) + self.F(x_old[:,j+1], v_old[:,j+1]))
          c_m += self.I_m_mp1[:,j+1]
          v_new[:,j+1] = self.boris_trick(x_new[:,j], x_new[:,j+1], v_new[:,j], c_m, delta_tau[j])
      
        
        ### Prepare next iteration
        self.stats['increments'][kk,nn] = max( np.linalg.norm(x_new - x_old, np.inf) , np.linalg.norm(v_new - v_old, np.inf) )
        x_old      = np.copy(x_new)
        v_old      = np.copy(v_new)

      '''
      Prepare next time step
      '''
      # Compute residual after final iteration
      self.stats['residuals'][self.kiter,nn] = self.updateIntegrals(self.positions[:,nn], self.velocities[:,nn], x_old, v_old)
      
      self.positions[:,nn+1], self.velocities[:,nn+1] = self.finalUpdateStep(x_old, v_old, self.positions[:,nn], self.velocities[:,nn])
      self.stats['energy_errors'][nn] = self.getEnergyError(self.positions[:,nn+1], self.velocities[:,nn+1])
      
      pos_ex = self.P.x(float(nn+1)*self.dt)
      vel_ex = self.P.v(float(nn+1)*self.dt)
      self.stats['errors'][0,nn+1] = np.linalg.norm( pos_ex - self.positions[:,nn+1] , np.inf )/np.linalg.norm(pos_ex, np.inf)
      self.stats['errors'][1,nn+1] = np.linalg.norm( vel_ex - self.velocities[:,nn+1] , np.inf )/np.linalg.norm(vel_ex, np.inf)

    return self.positions, self.velocities, self.stats

