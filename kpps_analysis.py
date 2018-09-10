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
        
        # Initialise pre- and post-analysis lists
        self.preAnalysis = []
        self.postAnalysis = []
        
        # Load required particle integration methods
        self.particleIntegration = []
        if 'particleIntegration' in kwargs:
            if kwargs['particleIntegration'] == 'boris_staggered':
                self.particleIntegration.append(self.boris_staggered)
                simulationManager.rhs_dt = 1
                
            if kwargs['particleIntegration'] == 'boris_synced':
                self.particleIntegration.append(self.boris_synced)
                simulationManager.rhs_dt = 1
                
            if kwargs['particleIntegration'] == 'boris_SDC':
                self.preAnalysis.append(self.collSetup)
                self.particleIntegration.append(self.boris_SDC)

                if 'M' in kwargs:
                    self.M = kwargs['M']
                else:
                    self.M = 1
                    
                if 'K' in kwargs:
                    self.K = kwargs['K']
                else: 
                    self.K = self.M

                if 'nodeType' in kwargs:
                    if kwargs['nodeType'] == 'lobatto':
                        self.ssi = 1    #Set sweep-start-index 'ssi'
                        self.collocationClass = CollGaussLobatto
                        self.updateStep = self.lobatto_update
                        simulationManager.rhs_dt = (self.M - 1)*self.K
                        
                    elif kwargs['nodeType'] == 'legendre':
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
        if 'fieldIntegration' in kwargs:
            self.fieldParams = kwargs['fieldIntegration']
            
            if 'scattering' in self.fieldParams:
                if self.fieldParams['scattering'] == 'linear':
                    self.fieldIntegration.append(self.linear_scatt)
                else:
                    self.fieldIntegration.append(self.linear_scatt)
            else:
                self.fieldIntegration.append(self.linear_scatt)
            
            
            if 'pic' in self.fieldParams:
                if self.fieldParams['pic'] == 'simple':
                    self.fieldIntegration.append(self.pic_simple)
            else:
                self.fieldIntegration.append(self.pic_simple)

            
            self.preAnalysis.append(self.initialise_field_mesh)
                
            
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
        
        
        # Load post-integration hook methods
        self.hooks = []
        if 'penningEnergy' in kwargs:
            self.preAnalysis.append(self.energy_calc_penning)
            self.hooks.append(self.energy_calc_penning)
            self.H = kwargs['penningEnergy']
            
        if 'centreMass' in kwargs and kwargs['centreMass'] == True:
            self.preAnalysis.append(self.centreMass)
            self.hooks.append(self.centreMass)

        
        ## Physical constants
        self.mu0 = 1
        self.ep0 = 1
        self.q0 = 1
        
        if 'units' in kwargs:
            if kwargs['units'] == 'si':
                self.makeSI()
                
    
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
    

    def particleIntegrator(self,species,simulationManager, **kwargs):
        for method in self.particleIntegration:
            method(species, simulationManager)

        return species
    
    def runHooks(self,species,simulationManager,**kwargs):
        for method in self.hooks:
            method(species,simulationManager)
            
        return species
    
    
    def preAnalyser(self,species,fields,simulationManager,**kwargs):
        for method in self.preAnalysis:
            method(species, fields, simulationManager)

        return species
    
    def postAnalyser(self,species,simulationManager,**kwargs):
        for method in self.postAnalysis:
            method(species, simulationManager)
        
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
        except AttributeError:
            print("Input species object either has no position array named"
                  + " 'pos' or electric field array named 'E'.")
        
        nq = len(pos)
        for pii in range(0,nq):
            for pjj in range(0,nq):
                if pii==pjj:
                    #E[pii,:] = 0
                    continue
                #print(E)

                species.E[pii,:] += species.E[pii,:] + self.coulombForce(species.q,
                                                   pos[pii,:],
                                                   pos[pjj,:])

        return species
    
    
    def coulombForce(self,q2,pos1,pos2):
        """
        Returns the electric field contribution on particle 1 w.r.t. 
        particle 2, where the charge of particle 2 'q2' is given in units 
        of the elementary charge q0 (i.e. actual charge = q2*q0).
        """

        rpos = pos1-pos2
        r = np.sqrt(np.sum(np.power(rpos,2)))
        rUnit = rpos/r
        
        Ec = 1/(4*pi*self.ep0) * q2*self.q0/r**2 * rUnit
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
        
    
    ## Field analysis methods
    def initialise_field_mesh(self,fields):
        k = 1
        if "magnitude" in self.imposedEParams:
            k = self.imposedEParams["magnitude"]
            
        if "sPenning" in self.imposedEParams:
            direction = np.array(self.imposedEParams['sPenning'])
            for xi in range(0,len(fields.pos[0,:,0,0])):
                for yi in range(0,len(fields.pos[0,0,:,0])):
                    for zi in range(0,len(fields.pos[0,0,0,:])):
                        fields.E[:,xi,yi,zi] += - fields.pos[:,xi,yi,zi] * direction * k
                        
        elif "general" in self.imposedEParams:
            inputMatrix = np.array(self.imposedEParams['general'])
            for xi in range(0,len(fields.pos[0,:,0,0])):
                for yi in range(0,len(fields.pos[0,0,:,0])):
                    for zi in range(0,len(fields.pos[0,0,0,:])):
                        direction = np.dot(inputMatrix,fields.pos[:,xi,yi,zi])
                        fields.E[:,xi,yi,zi] += direction * k

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
    
    
    def boris_staggered(self, species, simulationParameters):
        dt = simulationParameters.dt
        alpha = species.nq/species.mq

        self.fieldGather(species)
        
        species.vel = self.boris(species.vel,species.E,species.B,dt,alpha)
        species.pos = species.pos + simulationParameters.dt * species.vel
        return species
    
    
    def boris_synced(self, species, simulationParameters):
        dt = simulationParameters.dt
        alpha = species.q/species.mq

        species.pos = species.pos + dt * (species.vel + dt/2 * self.lorentz_std(species))
        
        E_old = species.E
        self.fieldGather(species)
        E_new = species.E
        
        E_half = (E_old+E_new)/2
        
        species.vel = self.boris(species.vel,E_half,species.B,dt,alpha)
        return species
        
    
    def collSetup(self,species,simulationManager,**kwargs):
        coll = self.collocationClass(self.M,0,1) #Initialise collocation/quadrature analysis object (class is Daniels old code)
        self.nodes = coll._getNodes
        self.weights = coll._getWeights(coll.tleft,coll.tright) #Get M  nodes and weights 


        self.Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
        self.Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix

        self.delta_m = coll._gen_deltas         #Generate vector of node spacings

        
    def boris_SDC(self, species, simulationManager,**kwargs):        

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
                    sumSX += SX[m+1,l]*(self.lorentzf(species,xn[:,l],vn[:,l]) - self.lorentzf(species,x[:,l],v[:,l]))

                sumSQ = 0
                for l in range(1,M+1):
                    sumSQ += SQ[m+1,l]*self.lorentzf(species,x[:,l],v[:,l])
                
                xn[:,m+1] = xn[:,m] + dm[m]*v[:,0] + sumSX + sumSQ
                
                #Determine next node (m+1) velocities
                sumS = 0
                for l in range(1,M+1):
                    sumS += Smat[m+1,l] * self.lorentzf(species,x[:,l],v[:,l])
            
                
                ck_dm = -1/2 * (self.lorentzf(species,x[:,m+1],v[:,m+1])+self.lorentzf(species,x[:,m],v[:,m])) + 1/dm[m] * sumS
                #print(ck_dm)
                
                #Sample the electric field at the half-step positions (yields form Nx3)
                half_E = (self.gatherE(species,xn[:,m])+self.gatherE(species,xn[:,m+1]))/2
                
                
                #Resort all other 3d vectors to shape Nx3 for use in Boris function
                v_oldNode = self.toMatrix(vn[:,m])
                ck_dm = self.toMatrix(ck_dm)
                
                v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,ck_dm)
                vn[:,m+1] = self.toVector(v_new)

            
            """
            check_node = 2
            ch_i = check_node
            
            x_res = np.linalg.norm(xn[:,ch_i]-xn[:,ch_i-1]-dm[m]*v[:,0]-sumSQ)
            v_res = np.linalg.norm(vn[:,ch_i]-vn[:,ch_i-1]-sumS)
            f_diff = (self.lorentzf(species,xn[:,ch_i],vn[:,ch_i]) 
                      - self.lorentzf(species,x[:,ch_i],v[:,ch_i]))
            f_conv = np.linalg.norm(f_diff)
            
            
            print("k = " + str(k) + ", iter f() conv. = " + str(f_conv))
            print("k = " + str(k) + ", x-residual = " + str(x_res))
            print("k = " + str(k) + ", v-residual = " + str(v_res))
            """
            
            x[:,:] = xn[:,:]
            v[:,:] = vn[:,:]
            
        
        species = self.updateStep(species,x,v,x0,v0,weights,Qmat)

        return species
    
    
    def lobatto_update(self,species,x,v,*args):
        pos = x[:,-1]
        vel = v[:,-1]

        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)

        return species
    
    
    def legendre_update(self,species,x,v,x0,v0,weights,Qmat):
        M = self.M
        d = 3*species.nq
        
        Id = np.identity(d)
        q = np.zeros(M+1,dtype=np.float)
        q[1:] = weights
        q = np.kron(q,Id)
        qQ = q @ np.kron(Qmat,Id)
        
        V0 = self.toVector(v0.transpose())
        F = self.FXV(species,x,v)
        
        vel = v0[:,0] + q @ F
        pos = x0[:,0] + q @ V0 + qQ @ F
        
        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)
        return species
    
    
    ## Additional analysis
    def get_u(self,x,v):
        assert len(x) == len(v)
        d = len(x)
        
        Ix = np.array([1,0])
        Iv = np.array([0,1])
        Id = np.identity(d)
        
        u = np.kron(Id,Ix).transpose() @ x + np.kron(Id,Iv).transpose() @ v
        return u
    
    
    def energy_calc_penning(self,species,simulationManager,**kwargs):
        x = self.toVector(species.pos)
        v = self.toVector(species.vel)
        u = self.get_u(x,v)
        
        species.energy = u.transpose() @ self.H @ u
        
        return species
    
    
    def centreMass(self,species,simulationManager,**kwargs):
        nq = np.float(species.nq)
        mq = np.float(species.mq)

        species.cm[0] = np.sum(species.pos[:,0]*mq)/(nq*mq)
        species.cm[1] = np.sum(species.pos[:,1]*mq)/(nq*mq)
        species.cm[2] = np.sum(species.pos[:,2]*mq)/(nq*mq)
        
        
    ## Additional methods
    def lorentzf(self,species,xm,vm):
        species.pos = species.toMatrix(xm)
        species.vel = species.toMatrix(vm)

        self.fieldGather(species)

        F = species.a*(species.E + np.cross(species.vel,species.B))
        F = species.toVector(F)
        return F
    
    def lorentz_std(self,species):
        self.fieldGather(species)
        F = species.a*(species.E + np.cross(species.vel,species.B))
        return F
    
    
    
    def FXV(self,species,x,v):
        dxM = np.shape(x)
        d = dxM[0]
        M = dxM[1]-1
        
        F = np.zeros((d,M+1),dtype=np.float)
        for m in range(0,M+1):
            F[:,m] = self.lorentzf(species,x[:,m],v[:,m])
        
        F = self.toVector(F.transpose())
        return F
    
    
    
    def gatherE(self,species,x):
        species.pos = self.toMatrix(x,3)
        
        self.fieldGather(species)
        
        return species.E
    
    def gatherB(self,species,x):
        species.pos = self.toMatrix(x,3)
        
        self.fieldGather(species)
        
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
    