#!/usr/bin/env python3

"""
For the following analysis class, the most important notation rule is that 
'pos' and 'vel' variables refer to particle data stored as a Nx3 matrix where 
'N' is the number of particles and thus each row represents a particle with
the columns storing the x,y,z components for the variable for each particle.

Conversely, 'x' and 'v' variables refer to particle data stored as a 1xd
vector, with d=3N, so the x,y,z components of the particle variable occur in
interchanging sequence like [1x,1y,1z,2x,2y,2z,...,Nx,Ny,Nz].
"""

## Dependencies
import numpy as np
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
import time

## Class
class kpps_analysis:
    def __init__(self,simulationManager,**kwargs):
        self.params = kwargs
        
        # Set default values
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
        
        self.E_type = 'none'
        self.E_magnitude = 1
        self.E_transform = np.zeros((3,3),dtype=np.float)
        
        self.coulomb = self.coulomb_cgs
        self.lambd = 0 
        
        self.B_type = 'none'
        self.B_magnitude = 1
        self.B_transform = np.zeros((1,3),dtype=np.float)
        
        self.imposeFields = False
        

        
        # Set params according to inputs
        for key, value in self.params.items():
            setattr(self,key,value)
        
        # Initialise pre- and post-analysis lists
        self.preAnalysis = []
        self.postAnalysis = []
        
        # Load required particle integration methods
        self.particleIntegration = []
        if 'particleIntegration' in self.params:
            if self.params['particleIntegration'] == 'boris_staggered':
                self.particleIntegration.append(self.boris_staggered)
                simulationManager.rhs_dt = 1
                
            if self.params['particleIntegration'] == 'boris_synced':
                self.particleIntegration.append(self.boris_synced)
                simulationManager.rhs_dt = 1
                
            if self.params['particleIntegration'] == 'boris_SDC':
                self.preAnalysis.append(self.collSetup)
                self.particleIntegration.append(self.boris_SDC)


        if 'nodeType' in self.params:
            if self.params['nodeType'] == 'lobatto':
                self.ssi = 1    #Set sweep-start-index 'ssi'
                self.collocationClass = CollGaussLobatto
                self.updateStep = self.lobatto_update
                simulationManager.rhs_dt = (self.M - 1)*self.K
                
            elif self.params['nodeType'] == 'legendre':
                self.ssi = 0 
                self.collocationClass = CollGaussLegendre
                self.updateStep = self.legendre_update
                simulationManager.rhs_dt = (self.M + 1)*self.K
                    
            else:
                self.ssi = 1
                self.collocationClass = CollGaussLobatto
                self.updateStep = self.lobatto_update
                simulationManager.rhs_dt = (self.M - 1)*self.K
               
            
        # Load required field analysis/integration methods
        self.fieldIntegration = []
        self.fieldGathering = []
        if self.fieldAnalysis == 'single' or 'coulomb':
            self.fieldGathering.append(self.eFieldImposed)
            self.fieldGathering.append(self.bFieldImposed)
                
        if self.fieldAnalysis == 'coulomb':
              self.fieldGathering.append(self.coulomb)   
              
        if self.fieldAnalysis == 'pic':
            self.fieldGathering.append(self.scatter) 
            self.fieldIntegration.append(self.pic_simple)

            if self.imposeFields  == True:
                self.preAnalysis.append(self.imposed_field_mesh)
        
        
        # Load post-integration hook methods
        self.hooks = []
        if 'penningEnergy' in self.params:
            self.preAnalysis.append(self.energy_calc_penning)
            self.hooks.append(self.energy_calc_penning)
            self.H = self.params['penningEnergy']
            
        if self.params['centreMass_check'] == True:
            self.preAnalysis.append(self.centreMass)
            self.hooks.append(self.centreMass)
            
        if 'residual_check' in self.params and self.params['residual_check'] == True:
            self.hooks.append(self.display_residuals)
        
        ## Physical constants
        if 'units' in kwargs:
            if self.params['units'] == 'si':
                self.makeSI()
                self.coulomb = self.coulomb_si
            elif self.params['units'] == 'cgs':
                pass
            elif self.params['units'] == 'custom':
                pass

                
    ## Analysis modules
    def fieldIntegrator(self,species,fields,**kwargs):     
        for method in self.fieldIntegration:
            method(species,fields)

        return species


    def fieldGather(self,species,fields,**kwargs):
        #Establish field values at particle positions via methods specified at initialisation.
        
        species.E = np.zeros((len(species.E),3),dtype=np.float)
        species.B = np.zeros((len(species.B),3),dtype=np.float)
        
        for method in self.fieldGathering:
            method(species,fields)

        return species
    

    def particleIntegrator(self,species,fields,simulationManager, **kwargs):
        for method in self.particleIntegration:
            method(species,fields,simulationManager)

        return species
    
    def runHooks(self,species,fields,simulationManager,**kwargs):
        for method in self.hooks:
            method(species,fields,simulationManager)
            
        return species
    
    
    def preAnalyser(self,species,fields,simulationManager,**kwargs):
        for method in self.preAnalysis:
            method(species, fields, simulationManager)

        return species
    
    def postAnalyser(self,species,fields,simulationManager,**kwargs):
        for method in self.postAnalysis:
            method(species,fields,simulationManager)
        
        return species
    
    ## Electric field methods
    def eFieldImposed(self,species,fields,**kwargs):
        if self.E_type == "custom":
            for pii in range(0,species.nq):
                direction = np.dot(self.E_transform,species.pos[pii,:])
                species.E[pii,:] += direction * self.E_magnitude
                        
        return species
    
    
    def coulomb_pair(self,species,pii,pjj):
        rpos = species.pos[pii,:] - species.pos[pjj,:]
        denom = np.power(np.linalg.norm(rpos)**2 + self.lambd**2,3/2)
        species.E[pii,:] += species.q*rpos/denom
        
        
    def coulomb_cgs(self, species,fields,**kwargs):
        for pii in range(0,species.nq):
            for pjj in range(0,pii):
                self.coulomb_pair(species,pii,pjj)
              
            for pjj in range(pii+1,species.nq):
                self.coulomb_pair(species,pii,pjj)

        return species
    
    
    def coulomb_si(self, species,fields,**kwargs):
        species.E = self.coulomb_cgs(species,fields) * 1/(4*pi*self.ep0)
        
        return species

    
    
    ## Magnetic field methods
    def bFieldImposed(self,species,fields,**kwargs):
        if self.B_type == 'uniform':
            try:
                species.B[:,0:] = np.multiply(self.B_magnitude,self.B_transform)
            except TypeError:
                print("TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")
        return species
        
    
    ## Field analysis methods
    def imposed_field_mesh(self,species,fields,simulationManager):
        k = self.E_magnitude
               
        if self.E_type == "custom":
            inputMatrix = np.array(self.E_transform)
            for xi in range(0,len(fields.pos[0,:,0,0])):
                for yi in range(0,len(fields.pos[0,0,:,0])):
                    for zi in range(0,len(fields.pos[0,0,0,:])):
                        direction = np.dot(inputMatrix,fields.pos[:,xi,yi,zi])
                        fields.E[:,xi,yi,zi] += direction * k        
        
        bMag = self.B_magnitude
        if self.B_type == "uniform":
            direction = np.array(self.B_transform)
            try:
                for xi in range(0,len(fields.pos[0,:,0,0])):
                    for yi in range(0,len(fields.pos[0,0,:,0])):
                        for zi in range(0,len(fields.pos[0,0,0,:])):
                            fields.B[:,xi,yi,zi] = np.multiply(bMag,direction)
            except TypeError:
                print("TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")

        return fields
    
    
    ## Time-integration methods
    def boris(self, vel, E, B, dt, alpha, ck=0):
        """
        Applies Boris' trick for given velocity, electric and magnetic 
        field for vector data in the shape (N x 3), i.e. particles as rows 
        and x,y,z components for the vector as the columns.
        k = delta_t * alpha / 2
        """ 
        
        k = dt*alpha/2
        
        tau = k*B

        vMinus = vel + dt/2 * (alpha*E + ck)
        
        tauMag = np.linalg.norm(tau,axis=1)
        vDash = vMinus + np.cross(vMinus,tau)
        
        tm = 2/(1+tauMag**2)

        for col in range(0,3):
            vDash[:,col] = tm[:] * vDash[:,col]

        vPlus = vMinus + np.cross(vDash,tau)
        
        vel_new = vPlus + dt/2 * (alpha*E + ck)
        
        return vel_new
    
    
    def boris_staggered(self, species,fields, simulationParameters):
        dt = simulationParameters.dt
        alpha = species.nq/species.mq

        self.fieldGather(species)
        
        species.vel = self.boris(species.vel,species.E,species.B,dt,alpha)
        species.pos = species.pos + simulationParameters.dt * species.vel
        return species
    
    
    def boris_synced(self, species,fields, simulationParameters):
        dt = simulationParameters.dt
        alpha = species.q/species.mq

        species.pos = species.pos + dt * (species.vel + dt/2 * self.lorentz_std(species,fields))
        
        E_old = species.E
        self.fieldGather(species,fields)
        E_new = species.E
        
        E_half = (E_old+E_new)/2
        
        species.vel = self.boris(species.vel,E_half,species.B,dt,alpha)
        return species
        
    
    def collSetup(self,species,fields,simulationManager,**kwargs):
        coll = self.collocationClass(self.M,0,1) #Initialise collocation/quadrature analysis object (class is Daniels old code)
        self.nodes = coll._getNodes
        self.weights = coll._getWeights(coll.tleft,coll.tright) #Get M  nodes and weights 


        self.Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
        self.Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix

        self.delta_m = coll._gen_deltas         #Generate vector of node spacings

        
    def boris_SDC(self, species,fields, simulationManager,**kwargs):        

        M = self.M
        K = self.K
        d = 3*species.nq
        
        dt = simulationManager.dt
        t = simulationManager.t
        
        #Remap collocation weights from [0,1] to [tn,tn+1]
        nodes = (t-dt) + self.nodes * dt
        weights = self.weights * dt 

        Qmat = self.Qmat * dt
        Smat = self.Smat * dt

        dm = self.delta_m * dt

        #Define required calculation matrices
        QE = np.zeros((M+1,M+1),dtype=np.float)
        QI = np.zeros((M+1,M+1),dtype=np.float)
        QT = np.zeros((M+1,M+1),dtype=np.float)
        
        SX = np.zeros((M+1,M+1),dtype=np.float)
        
        for i in range(0,M):
            QE[(i+1):,i] = dm[i]
            QI[(i+1):,i+1] = dm[i] 
        
        QT = 1/2 * (QE + QI)
        QX = QE @ QT + (QE*QE)/2
        SX[:,:] = QX[:,:]
        SX[1:,:] = QX[1:,:] - QX[0:-1,:]      

        SQ = Smat @ Qmat
        
        self.x_con = np.zeros((K,M))
        self.x_res = np.zeros((K,M))
        self.v_con = np.zeros((K,M))
        self.v_res = np.zeros((K,M))
        
        x0 = np.zeros((d,M+1),dtype=np.float)
        v0 = np.zeros((d,M+1),dtype=np.float)
        
        xn = np.zeros((d,M+1),dtype=np.float)
        vn = np.zeros((d,M+1),dtype=np.float)
        
        
        #Populate node solutions with x0, v0
        for m in range(0,M+1):
            x0[:,m] = self.toVector(species.pos)
            v0[:,m] = self.toVector(species.vel)

        x = np.copy(x0)
        v = np.copy(v0)
        
        xn[:,:] = x[:,:]
        vn[:,:] = v[:,:]
        
        #print()
        #print(simulationManager.ts)
        for k in range(1,K+1):
            #print("k = " + str(k))
            
            for m in range(self.ssi,M):
                #print("m = " + str(m))
                #Determine next node (m+1) positions
                sumSX = 0
                for l in range(1,m+1):
                    sumSX += SX[m+1,l]*(self.lorentzf(species,fields,xn[:,l],vn[:,l]) - self.lorentzf(species,fields,x[:,l],v[:,l]))

                sumSQ = 0
                for l in range(1,M+1):
                    sumSQ += SQ[m+1,l]*self.lorentzf(species,fields,x[:,l],v[:,l])
                
                xQuad = xn[:,m] + dm[m]*v[:,0] + sumSQ
                xn[:,m+1] = xQuad + sumSX 
                
                
                #Determine next node (m+1) velocities
                sumS = 0
                for l in range(1,M+1):
                    sumS += Smat[m+1,l] * self.lorentzf(species,fields,x[:,l],v[:,l])
                
                vQuad = vn[:,m] + sumS
                
                ck_dm = -1/2 * (self.lorentzf(species,fields,x[:,m+1],v[:,m+1])
                        +self.lorentzf(species,fields,x[:,m],v[:,m])) + 1/dm[m] * sumS
                
                #Sample the electric field at the half-step positions (yields form Nx3)
                half_E = (self.gatherE(species,fields,xn[:,m])+self.gatherE(species,fields,xn[:,m+1]))/2
                
                
                #Resort all other 3d vectors to shape Nx3 for use in Boris function
                v_oldNode = self.toMatrix(vn[:,m])
                ck_dm = self.toMatrix(ck_dm)
                
                v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,ck_dm)
                vn[:,m+1] = self.toVector(v_new)
                
                
                self.calc_residuals(k,m,x,xn,xQuad,v,vn,vQuad)
                
                
            x[:,:] = xn[:,:]
            v[:,:] = vn[:,:]
                
        species = self.updateStep(species,fields,x,v,x0,v0,weights,Qmat)

        return species
    
    
    def lobatto_update(self,species,fields,x,v,*args):
        pos = x[:,-1]
        vel = v[:,-1]

        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)

        return species
    
    
    def legendre_update(self,species,fields,x,v,x0,v0,weights,Qmat):
        M = self.M
        d = 3*species.nq
        
        Id = np.identity(d)
        q = np.zeros(M+1,dtype=np.float)
        q[1:] = weights
        q = np.kron(q,Id)
        qQ = q @ np.kron(Qmat,Id)
        
        V0 = self.toVector(v0.transpose())
        F = self.FXV(species,fields,x,v)
        
        vel = v0[:,0] + q @ F
        pos = x0[:,0] + q @ V0 + qQ @ F
        
        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)
        return species
    
    
    ## Additional analysis
    def calc_residuals(self,k,m,x,xn,xQuad,v,vn,vQuad):
        self.x_con[k-1,m] = np.average(np.abs(xn[:,m+1] - x[:,m+1]))
        self.x_res[k-1,m] = np.average(np.linalg.norm(xn[:,m+1]-xQuad))
        
        self.v_res[k-1,m] = np.average(np.linalg.norm(vn[:,m+1]-vQuad))
        self.v_con[k-1,m] = np.average(np.abs(vn[:,m+1] - v[:,m+1]))
        
        
    def display_residuals(self,species,fields,simulationManager):
        print("Position convergence:")
        print(self.x_con)
        
        print("Velocity convergence:")  
        print(self.v_con)
        
        print("Position residual:")
        print(self.x_res)
        
        print("Velocity residual:")
        print(self.v_res)
        
        
    def get_u(self,x,v):
        assert len(x) == len(v)
        d = len(x)
        
        Ix = np.array([1,0])
        Iv = np.array([0,1])
        Id = np.identity(d)
        
        u = np.kron(Id,Ix).transpose() @ x + np.kron(Id,Iv).transpose() @ v
        return u
    
    
    def energy_calc_penning(self,species,fields,simulationManager,**kwargs):
        x = self.toVector(species.pos)
        v = self.toVector(species.vel)
        u = self.get_u(x,v)
        
        species.energy = u.transpose() @ self.H @ u
        
        return species
    
    
    def centreMass(self,species,fields,simulationManager,**kwargs):
        nq = np.float(species.nq)
        mq = np.float(species.mq)

        species.cm[0] = np.sum(species.pos[:,0]*mq)/(nq*mq)
        species.cm[1] = np.sum(species.pos[:,1]*mq)/(nq*mq)
        species.cm[2] = np.sum(species.pos[:,2]*mq)/(nq*mq)
        
        
    ## Additional methods
    def lorentzf(self,species,fields,xm,vm):
        species.pos = species.toMatrix(xm)
        species.vel = species.toMatrix(vm)

        self.fieldGather(species,fields)

        F = species.a*(species.E + np.cross(species.vel,species.B))
        F = species.toVector(F)
        return F
    
    def lorentz_std(self,species,fields):
        self.fieldGather(species,fields)
        F = species.a*(species.E + np.cross(species.vel,species.B))
        return F
    
    
    
    def FXV(self,species,fields,x,v):
        dxM = np.shape(x)
        d = dxM[0]
        M = dxM[1]-1
        
        F = np.zeros((d,M+1),dtype=np.float)
        for m in range(0,M+1):
            F[:,m] = self.lorentzf(species,fields,x[:,m],v[:,m])
        
        F = self.toVector(F.transpose())
        return F
    
    
    
    def gatherE(self,species,fields,x):
        species.pos = self.toMatrix(x,3)
        
        self.fieldGather(species,fields)
        
        return species.E
    
    def gatherB(self,species,fields,x):
        species.pos = self.toMatrix(x,3)
        
        self.fieldGather(species,fields)
        
        return species.B
        
    def toVector(self,storageMatrix):
        rows = storageMatrix.shape[0]
        columns = storageMatrix.shape[1]
        vector = np.zeros(rows*columns)
        
        for i in range(0,columns):
            vector[i::columns] = storageMatrix[:,i]
        return vector
    
    
    def toMatrix(self,vector,columns=3):
        rows = int(len(vector)/columns)
        matrix = np.zeros((rows,columns))
        
        for i in range(0,columns):
            matrix[:,i] = vector[i::columns]
        return matrix
    
        
    def nope(self,species):
        return species
    
    def makeSI(self):
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
    