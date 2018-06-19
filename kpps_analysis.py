#!/usr/bin/env python3

## Dependencies
import numpy as np
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto

## Class
class kpps_analysis:
    ## Physical constants
    mu0 = 1
    ep0 = 1
    q0 = 1
    
    def __init__(self,**kwargs):
        # Load required particle integration methods
        self.particleIntegration = []
        if 'particleIntegration' in kwargs:
            if kwargs['particleIntegration'] == 'boris':
                self.particleIntegration.append(self.boris)
                
            if kwargs['particleIntegration'] == 'boris_synced':
                self.particleIntegration.append(self.boris_synced)
                
            if kwargs['particleIntegration'] == 'boris_SDC':
                self.particleIntegration.append(self.boris_SDC)
        
        if 'M' in kwargs:
            self.M = kwargs['M']
        else:
            self.M = 1
            
        if 'K' in kwargs:
            self.K = kwargs['K']
        else: 
            self.K = self.M
            
        # Load required field integration methods
        self.fieldIntegration = []
        if 'fieldIntegration' in kwargs:
            if kwargs['fieldIntegration'] == 'pic':
                self.fieldIntegration.append(self.nope)
                
                
        # Load required field gathering methods
        self.fieldGathering = []
        if 'imposedElectricField' in kwargs:
            self.fieldGathering.append(self.eFieldImposed)
            self.imposedEParams = kwargs['imposedElectricField']
            
        if 'interactionModelling' in kwargs:
            if kwargs['interactionModelling'] == 'full':
                self.fieldGathering.append(self.nope)
            elif kwargs['interactionModelling'] == 'intra':
                self.fieldGathering.append(self.coulombIntra)   
            else:
                self.fieldGathering.append(self.nope) 
                
        if 'imposedMagneticField' in kwargs:
            self.fieldGathering.append(self.bFieldImposed)
            self.imposedBParams = kwargs['imposedMagneticField']
        
        
    ## Analysis modules
    def fieldIntegrator(self,species,**kwargs):
        for method in self.fieldIntegration:
            method(species)
        
        return species


    def particleIntegrator(self,species,simulationManager, **kwargs):
        i = 0
        for method in self.particleIntegration:
            i += 1
            method(species, simulationManager)
        
        return species
    
    def fieldGather(self,species,**kwargs):
        #Establish field values at particle positions via methods specified at initialisation.
        
        species.E = np.zeros((len(species.E),3),dtype=np.float)
        species.B = np.zeros((len(species.B),3),dtype=np.float)
        
        for method in self.fieldGathering:
            method(species)
        
        return species

    ## Electric field methods
    def eFieldImposed(self,species,**kwargs):
        k = 1
        if "magnitude" in self.imposedEParams:
            k = self.imposedEParams["magnitude"]
            
        if "sPenning" in self.imposedEParams:
            direction = np.array(self.imposedEParams['sPenning'])
            species.E += - species.pos * direction * k
        elif "general" in self.imposedEParams:
            inputMatrix = np.array(self.imposedEParams['general'])
            for pii in range(0,species.nq):
                direction = np.dot(inputMatrix,species.pos[pii,:])
                species.E[pii,:] += direction * k


        return species


    def coulombIntra(self, species,**kwargs):
        try:
            pos = species.pos
            E = species.E
        except AttributeError:
            print("Input species object either has no position array named"
                  + " 'pos' or electric field array named 'E'.")
        
        nq = len(pos)
        
        for pii in range(0,nq):
            for pjj in range(0,nq):
                if pii==pjj:
                    continue
                E[pii,:] = E[pii,:] + self.coulombForce(species.q,
                                                   pos[pii,:],
                                                   pos[pjj,:])
        
        species.E += E
        return species
    
    
    def coulombForce(self,q2,pos1,pos2):
        """
        Returns the electric field contribution on particle 1 w.r.t. 
        particle 2, where the charge of particle 2 'q2' is given in units 
        of the elementary charge q0 (i.e. actual charge = q2*q0).
        """
        
        rpos = pos1-pos2
        r = sqrt(fsum(rpos**2))
        rUnit = rpos/r
        
        Ec = 1/(4*pi*self.ep0) * q2/r**2 * rUnit
        return Ec
    
    
    
    ## Magnetic field methods
    def bFieldImposed(self,species,**kwargs):
        species.B = np.array(species.B)
        settings = self.imposedBParams
        
        if "magnitude" in settings:
            bMag = settings["magnitude"]
        else:
            bMag = 1
        
        if "uniform" in settings:
            direction = np.array(settings["uniform"])
            try:
                species.B[:,0:] = np.multiply(bMag,direction)
            except TypeError:
                print("TypeError raised, did you input a length 3 vector "
                      + "to define the uniform magnetic field?")

        return species
        
    
    
    ## Time-integration methods
    def boris(self, species, simulationParameters):
        nq = len(species.pos)
        k = simulationParameters.dt * species.nq/(2*species.mq)
        vPlus = np.zeros((nq,3),dtype=np.float)

        self.fieldGather(species)
        
        tau = k*species.B
        vMinus = species.vel + k*species.E
        tauMag = np.linalg.norm(tau,axis=1)
        vDash = vMinus + np.cross(vMinus,tau)
        vPlus = vMinus + np.cross(2/(1+tauMag**2)*vDash,tau)
        
        species.vel = vPlus + k*species.E
        species.pos = species.pos + simulationParameters.dt * species.vel
        return species
    
    
    def boris_synced(self, species, simulationParameters):
        nq = len(species.pos)
        k = simulationParameters.dt * species.nq/(2*species.mq)
        
        self.fieldGather(species)
        E = species.E

        species.pos = species.pos + simulationParameters.dt * species.vel
        self.fieldGather(species)
        En = species.E
        
        E_half = (E+En)/2
        vMinus = species.vel + k*E_half

        t = k*species.B
        tmag = np.linalg.norm(t)
        s = 2*t/(1+tmag**2)
        vDash = vMinus + np.cross(vMinus,t)
        vPlus = vMinus + np.cross(vDash,s)
        species.vel = vPlus + k*E_half

        return species
    
    
    def boris_SDC(self, species, simulationManager,**kwargs):        
        M = self.M
        K = self.K
        d = 3*species.nq
        
        tleft = simulationManager.t - simulationManager.dt
        tright = simulationManager.t
        
        #coll = CollGaussLegendre(M,tleft,tright)    #Initialise collocation/quadrature analysis object (class is Daniels old code)
        coll = CollGaussLobatto(M,tleft,tright)
        coll.nodes = coll._getNodes
        coll.weights = coll._getWeights(coll.tleft,coll.tright) #Get M Gauss-Legendre nodes and weights 
            
        tau = coll.nodes                          #Nickname the vector of collocation points for ease of use
        coll.Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
        coll.Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix

        coll.delta_m = coll._gen_deltas         #Generate vector of node spacings
        dm = coll.delta_m

        
        #Define permutation operators
        Ix = np.array([[1],[0]])
        Iv = np.array([[0],[1]])
        Ixv = np.array([[0,1],[0,0]])
        Id = np.identity(d)
        
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
        
        
        q = np.zeros(M+1,dtype=np.float)
        q[1:] = coll.weights
        qv = np.kron(q,Iv)
        qxv = np.kron(q,Ixv)
        qQ = q @ coll.Qmat

        q = np.kron(q,Id)
        qv = np.kron(qv,Id)
        qxv = np.kron(qxv,Id)
        qQ = np.kron(qQ,Id) 
        
        Qx = np.kron(coll.Qmat,Ix)
        Qx = np.kron(Qx,Id)
        
        sx = np.kron(coll.Smat,Ix)
        SQ = coll.Smat @ coll.Qmat

        I2d = np.identity(2*d)
        I_P = np.ones(M+1)
        I_R = np.ones(M+1)
        I_R[0:M] = 0
        
        T_P = np.kron(I2d,I_P)
        T_R = np.kron(I_R,I2d).transpose()

        x = np.zeros((d,M+1),dtype=np.float)
        v = np.zeros((d,M+1),dtype=np.float)
        
        xn = np.zeros((d,M+1),dtype=np.float)
        vn = np.zeros((d,M+1),dtype=np.float)
        
        x.transpose()[:,0:] = species.toVector(species.pos)
        v.transpose()[:,0:] = species.toVector(species.vel)
        u0 = np.kron(Id,Ix) @ x[:,0] + np.kron(Id,Iv) @ v[:,0]
        
        xn[:,:] = x[:,:]
        vn[:,:] = v[:,:]
        
        for k in range(0,K):
            for m in range(1,M):
                
                #Determine next node (m+1) positions
                sumSX = 0
                for l in range(1,m+1):
                    sumSX += SX[m+1,l]*(self.lorentzf(species,xn[:,l],vn[:,l]) - self.lorentzf(species,x[:,l],v[:,l]))
                    
                
                sumSQ = 0
                for l in range(1,M+1):
                    sumSQ += SQ[m+1,l]*self.lorentzf(species,x[:,l],v[:,l])
                
                xn[:,m+1] = xn[:,m] + dm[m]*v[:,0] + sumSX + sumSQ
                
                
                #Sample the electric field at the half-step positions
                half_E = (self.gatherE(species,xn[:,m])+self.gatherE(species,xn[:,m+1]))/2
                
                
                #Determine next node (m+1) velocities
                sumS = 0
                for l in range(1,M+1):
                    sumS += coll.Smat[m+1,l] * self.lorentzf(species,x[:,l],v[:,l])
                    
                ck = -1/2 * (self.lorentzf(species,x[:,m+1],v[:,m+1])+self.lorentzf(species,x[:,m],v[:,m])) + 1/dm[m] * sumS
                t_mag = species.a * self.gatherB(species,xn[:,m]) * simulationManager.dt/2
                s_mag = 2*t_mag/(1+np.linalg.norm(t_mag)**2)
                
                vMinus = vn[:,m] + dm[m]/2 * (species.a * half_E + ck)
                
                #Resort 3d to shape d/3 x 3 to define cross-product
                vMinus = species.toMatrix(vMinus)
                t_mag = species.toMatrix(t_mag)
                s_mag = species.toMatrix(s_mag)

                vPlus = vMinus + np.cross((vMinus + np.cross(vMinus,t_mag)),s_mag)
                vPlus = species.toVector(vPlus)
                
                vn[:,m+1] = vPlus + dm[m]/2 * (species.a * half_E + ck)
                
            x[:,:] = xn[:,:]
            v[:,:] = vn[:,:]

        #F = self.FXV(species,xn,vn)
        
        #v0 = v[:,0]
        #vel = v0[:,np.newaxis] + q @ F
        
        #x0 = x[:,0]
        #pos = x0[:,np.newaxis] + q @ self.toVector(v.transpose())[:,np.newaxis] + qQ @ F
        

        pos = xn[:,-1]
        vel = vn[:,-1]
        

        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)
        #Q_coll = Qx @ q + qv
        #C_coll = T_R + qxv
        
        #u = C_coll @ T_P @ u0 + Q_coll @ F
        #species.pos = species.toMatrix(u[0::2])
        #species.vel = species.toMatrix(u[1::2])
        
        return species
    
    
    ## Additional methods
    def lorentzf(self,species,x,v):
        species.pos = species.toMatrix(x)
        species.vel = species.toMatrix(v)
        
        self.fieldGather(species)
        F = species.a*(species.E + np.cross(species.vel,species.B))
        F = species.toVector(F)
        return F
    
    def FXV(self,species,X,V):
        dim = X.shape[0]
        M = X.shape[1] - 1
        
        F = np.zeros(dim*(M+1),dtype=np.float)
        
        for m in range(0,M+1):
            species.pos = species.toMatrix(X[:,m])
            species.vel = species.toMatrix(V[:,m])
            
            self.fieldGather(species)
            
            Fm = species.a*(species.E + np.cross(species.vel,species.B))
            F[dim*m:dim*(m+1)] = self.toVector(Fm)
            
        F = F[:,np.newaxis]
        return F
    
    
    def systemF(self,species,x,v):
        dim = x.shape[0]
        M = x.shape[1] - 1
        
        Ix = np.array([[1],[0]])
        Iv = np.array([[0],[1]])
        Id = np.identity(dim*(M+1))
        
        Fv = np.zeros(dim*(M+1),dtype=np.float)
        Fx = np.zeros(dim*(M+1),dtype=np.float)
        
        for m in range(0,M+1):
            species.pos = species.toMatrix(x[:,m])
            species.vel = species.toMatrix(v[:,m])
            
            self.fieldGather(species)
            
            FvN = species.a*(species.E + np.cross(species.vel,species.B))
            Fv[dim*m:dim*(m+1)] = self.toVector(FvN)
            
            Fx[dim*m:dim*(m+1)] = v[:,m]
            
        Fx = Fx[:,np.newaxis]
        Fv = Fv[:,np.newaxis]
        F = np.kron(Id,Ix) @ Fx + np.kron(Id,Iv) @ Fv
        return F
    
    def gatherE(self,species,x):
        species.pos = species.toMatrix(x)
        
        self.fieldGather(species)
        
        return species.toVector(species.E)
    
    def gatherB(self,species,x):
        species.pos = species.toMatrix(x)
        
        self.fieldGather(species)
        
        return species.toVector(species.B)
        
    def toVector(self,storageMatrix):
        rows = storageMatrix.shape[0]
        columns = storageMatrix.shape[1]
        vector = np.zeros(rows*columns,dtype=np.float)
        
        for i in range(0,rows):
            vector[columns*i:columns*(i+1)] = storageMatrix[i,:]
            
        return vector
        

    def toMatrix(self,vector):
        self.matrix = np.zeros((self.nq,3),dtype=np.float)
        for pii in range(0,self.nq):
            self.matrix[pii,0] = vector[3*pii]
            self.matrix[pii,1] = vector[3*pii+1]
            self.matrix[pii,2] = vector[3*pii+2]
    
        
    def nope(self,species):
        return species
    
    def makeSI(self):
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
    