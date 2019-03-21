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
import scipy.sparse as sps
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
import time

## Class
class kpps_analysis:
    def __init__(self,**kwargs):
        ## Default values
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
        
         # Hook inputs
        self.pre_hook_list = []
        self.hook_list = []
        
         # Quick hook selection flags
        self.centreMass_check = False
        self.coulomb_field_check = False
        self.residual_check = False
        self.rhs_check = False
        
        
        self.particleIntegration = False
        self.particleIntegrator = 'boris_SDC'
        self.nodeType = 'lobatto'
        self.rhs_dt = 1
        self.gather = self.coulomb
        self.bound_cross_methods = []
        self.looped_axes = []
        
        
        self.fieldIntegration = False
        self.field_type = 'custom' #Can be pic, coulomb or custom
        self.FDMat = None
        self.mesh_boundary_z = 'fixed'
        self.mesh_boundary_y = 'fixed'
        self.mesh_boundary_x = 'fixed'
        self.poisson_M_adjust_1d = self.none
        self.poisson_M_adjust_2d = self.none
        self.poisson_M_adjust_3d = self.none
        self.pot_differentiate_z = self.pot_diff_fixed_z
        self.pot_differentiate_y = self.pot_diff_fixed_y
        self.pot_differentiate_x = self.pot_diff_fixed_x
        self.mi_z0 = 1
        self.mi_y0 = 1
        self.mi_x0 = 1
        self.solver_post = self.none
        
        self.external_fields = False
        self.background = self.none
        self.scatter = self.trilinear_qScatter
        self.scatter_BC = self.none
        self.fIntegrator_setup = self.poisson_cube2nd_setup
        self.fIntegrator = self.poisson_cube2nd
        self.imposeFields = False
        
        
        self.units = 'cgs'
        
        # Initialise operation lists
        self.preAnalysis_methods = []
        self.fieldIntegrator_methods = []
        self.particleIntegrator_methods = []
        self.fieldGather_methods = []
        self.hooks = []
        self.postAnalysis_methods = []
        
        ## Dummy values
        self.pot_diff_list = []
        self.unit_scale_poisson = 1
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = kwargs
        for key, value in self.params.items():
            setattr(self,key,value)
            
            
        # check for other intuitive parameter names
        name_dict = {}
        name_dict['looped_axes'] = ['periodic_axes','mirrored_axes']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    setattr(self,key,getattr(self,name))
                except AttributeError:
                    pass
 
        
        # Setup required boundary methods
        for ax in self.looped_axes:
            method_name = 'periodic_particles_' + ax
            self.bound_cross_methods.append(method_name)
        
        
        # Setup required particle-field interpolation methods
        if self.particleIntegration == True and self.fieldIntegration == True:
            if self.field_type == 'pic':
                self.gather = self.trilinear_gather
                self.scatter = self.trilinear_qScatter  
            elif self.field_type == 'coulomb':
                self.gather = self.coulomb 
                self.scatter = self.none
            else:
                pass
        
            self.fieldIntegrator_methods.append(self.scatter)
        
        # Setup required particle analysis methods
        if self.particleIntegration == True:
            self.particleIntegrator_methods.append(self.particleIntegrator)
            if self.particleIntegrator == 'boris_SDC':
                self.preAnalysis_methods.append(self.collSetup)
            
            self.fieldGather_methods.append(self.gather)  
            
            
        # Setup required field analysis methods
        if self.fieldIntegration == True:           
            if self.field_type == 'pic':
                self.preAnalysis_methods.append(self.fIntegrator_setup)
                self.fieldIntegrator_methods.append(self.background) 
                self.fieldIntegrator_methods.append(self.fIntegrator)
     
                if self.imposeFields  == True:
                    self.preAnalysis_methods.append(self.imposed_field_mesh)
                    
        if self.external_fields == True:
            self.fieldGather_methods.append(self.eFieldImposed)
            self.fieldGather_methods.append(self.bFieldImposed)
            

                
        
        # Load hook methods
        if self.rhs_check == True:
            self.preAnalysis_methods.append(self.rhs_tally)
        
        for hook in self.pre_hook_list:
            self.preAnalysis_methods.append(hook)
        
        
        if 'penningEnergy' in self.params:
            self.preAnalysis_methods.append(self.energy_calc_penning)
            self.hooks.append(self.energy_calc_penning)
            self.H = self.params['penningEnergy']
        
        if self.coulomb_field_check == True:
            self.preAnalysis_methods.append(self.coulomb_field)
            self.hooks.append(self.coulomb_field)
            
        if self.centreMass_check == True:
            self.preAnalysis_methods.append(self.centreMass)
            self.hooks.append(self.centreMass)
            
        if self.residual_check == True:
            self.hooks.append(self.display_residuals)
            
        self.poisson_M_adjust_1d = self.stringtoMethod(self.poisson_M_adjust_1d)
        self.poisson_M_adjust_2d = self.stringtoMethod(self.poisson_M_adjust_2d)
        self.poisson_M_adjust_3d = self.stringtoMethod(self.poisson_M_adjust_3d)
        
        self.setup_OpsList(self.preAnalysis_methods)
        self.setup_OpsList(self.fieldIntegrator_methods)
        self.setup_OpsList(self.fieldGather_methods)
        self.setup_OpsList(self.particleIntegrator_methods)
        self.setup_OpsList(self.bound_cross_methods)
        self.setup_OpsList(self.hooks)
        self.setup_OpsList(self.postAnalysis_methods)
        
        ## Physical constants
        if self.units == 'si':
            self.makeSI()
            self.coulomb = self.coulomb_si
            self.unit_scale_poisson = 1/self.ep0
        elif self.units == 'cgs':
            self.unit_scale_poisson = 4*pi
        elif self.units == 'custom':
            pass

                
########################### Main Run Loops ####################################
    def run_fieldIntegrator(self,species_list,fields,simulationManager,**kwargs):     
        fields.q = np.zeros((fields.q.shape),dtype=np.float)
        for method in self.fieldIntegrator_methods:
            method(species_list,fields,simulationManager)
        return species_list


    def fieldGather(self,species,fields,**kwargs):
        #Establish field values at particle positions via methods specified at initialisation.
        
        species.E = np.zeros((len(species.E),3),dtype=np.float)
        species.B = np.zeros((len(species.B),3),dtype=np.float)
        
        for method in self.fieldGather_methods:
                method(species,fields)
            
        return species
    

    def run_particleIntegrator(self,species_list,fields,simulationManager,**kwargs):
        for species in species_list:
            for method in self.particleIntegrator_methods:
                method(species,fields,simulationManager)
    
        return species_list
    
    def runHooks(self,species_list,fields,simulationManager,**kwargs):
        for method in self.hooks:
            method(species_list,fields,simulationManager)
            
        return species_list, fields
    
    
    def run_preAnalyser(self,species_list,fields,simulationManager,**kwargs):
        for method in self.preAnalysis_methods:
            method(species_list, fields, simulationManager)

        return species_list, fields
    
    def run_postAnalyser(self,species_list,fields,simulationManager,**kwargs):
        for method in self.postAnalysis_methods:
            method(species_list,fields,simulationManager)
        
        return species_list, fields
    

##################### Imposed E-Field Methods #################################
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

    
    
##################### Imposed B-Field Methods #################################
    def bFieldImposed(self,species,fields,**kwargs):
        if self.B_type == 'uniform':
            try:
                species.B[:,0:] = np.multiply(self.B_magnitude,self.B_transform)
            except TypeError:
                print("Analyser: TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")

        return species
        
    
########################## Field Analysis Methods #############################
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
                print("Analyser: TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")

        return fields
    
    def coulomb_field(self,species,fields,simulationManager,**kwargs):
        #Needs mesh position storing turned on
        rpos_array = np.zeros((3,fields.xres+1,
                      fields.yres+1,
                      fields.zres+1),dtype=np.float)
        fields.CE = np.zeros((3,fields.xres+1,
                              fields.yres+1,
                              fields.zres+1),dtype=np.float)
        
        for pii in range(0,species.nq):
            rpos_array[0,:,:,:] = fields.pos[0,:,:,:] - species.pos[pii][0]
            rpos_array[1,:,:,:] = fields.pos[1,:,:,:] - species.pos[pii][1]
            rpos_array[2,:,:,:] = fields.pos[2,:,:,:] - species.pos[pii][2]
            
            rmag_array = np.sum(rpos_array**2,axis=0)**(1/2)
            
            fields.CE[0,:,:,:] += rpos_array[0,:,:,:] / rmag_array**3
            fields.CE[1,:,:,:] += rpos_array[1,:,:,:] / rmag_array**3
            fields.CE[2,:,:,:] += rpos_array[2,:,:,:] / rmag_array**3
        return fields


    def poisson_cube2nd_setup(self,species_list,fields,simulationManager,**kwargs):
        self.interior_shape = fields.res-1
        nx = self.interior_shape[0]
        ny = self.interior_shape[1]
        nz = self.interior_shape[2]
        
        FDMatrix_adjust_z = self.none
        FDMatrix_adjust_y = self.none
        FDMatrix_adjust_x = self.none
        
        if self.mesh_boundary_z == 'open':
            self.interior_shape[2] += 1
            nz +=1
            self.mi_z0 = 0
            FDMatrix_adjust_z = self.poisson_M_adjust_1d
            self.scatter_BC = self.scatter_periodicBC_1d
            self.solver_post = self.mirrored_boundary_z
            self.pot_differentiate_z = self.pot_diff_open_z
            
            
        k = np.zeros(3,dtype=np.float)
        k[0] = -2*(1/fields.dz**2)
        k[1] = -2*(1/fields.dy**2 + 1/fields.dz**2)
        k[2] = -2*(1/fields.dx**2 + 1/fields.dy**2 + 1/fields.dz**2)
        
        diag = [1/fields.dz**2,k[simulationManager.ndim-1],1/fields.dz**2]
        Dk = sps.diags(diag,offsets=[-1,0,1],shape=(nz,nz))
        self.FDMat = Dk
        FDMatrix_adjust_z(species_list,fields,simulationManager)
        self.pot_diff_list.append(self.pot_differentiate_z)
        
        if simulationManager.ndim >= 2:
            if self.mesh_boundary_y == 'open':
                self.interior_shape[1] += 1
                ny += 1
                self.mi_y0 = 0
                
                FDMatrix_adjust_y = self.poisson_M_adjust_2d
                self.pot_differentiate_y = self.pot_diff_open_y
                
            I = sps.identity(nz)
            diag = sps.diags([1],shape=(ny,ny))
            off_diag = sps.diags([1,1],offsets=[-1,1],shape=(ny,ny))
            FDMatrix_adjust_y(species,fields,simulationManager)
            
            Ek = sps.kron(diag,Dk) + sps.kron(off_diag,I/fields.dy**2)
            self.FDMat = Ek
            self.pot_diff_list.append(self.pot_differentiate_y)
            
        if simulationManager.ndim == 3:
            if self.mesh_boundary_y == 'open':
                self.interior_shape[0] += 1
                nx += 1
                self.mi_x0 = 0
                
                FDMatrix_adjust_x = self.poisson_M_adjust_3d
                self.pot_differentiate_x = self.pot_diff_open_x
                
            J = sps.identity(nz*ny)
            diag = sps.diags([1],shape=(nx,nx))
            off_diag = sps.diags([1,1],offsets=[-1,1],shape=(nx,nx))
            FDMatrix_adjust_x(species_list,fields,simulationManager)
            
            Fk = sps.kron(diag,Ek) + sps.kron(off_diag,J/fields.dx**2)
            self.FDMat = Fk
            self.pot_diff_list.append(self.pot_differentiate_x)
            

        return self.FDMat
    
        
    
        
    def poisson_cube2nd(self,species_list,fields,simulationManager,**kwargs):
        
        rho = self.meshtoVector(fields.rho[self.mi_x0:-2,
                                           self.mi_y0:-2,
                                           self.mi_z0:-2])

        phi = sps.linalg.spsolve(self.FDMat,rho*self.unit_scale_poisson - fields.BC_vector)
        phi = self.vectortoMesh(phi,self.interior_shape)
        
        fields.phi[self.mi_x0:-2,
                   self.mi_y0:-2,
                   self.mi_z0:-2] = phi

        self.solver_post(species_list,fields,simulationManager)
        
        for nd in range(0,simulationManager.ndim):
            self.pot_diff_list[nd](fields)
            
        return fields
        
    
    def pot_diff_fixed_x(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field x-component differentiation
        fields.E[0,0,:,:] = -2*(fields.phi[0,:,:]-fields.phi[1,:,:])
        fields.E[0,1:n[0]-1,:,:] = -(fields.phi[0:n[0]-2,:,:] - fields.phi[2:n[0],:,:])
        fields.E[0,n[0]-1,:,:] = -2*(fields.phi[n[0]-2,:,:]-fields.phi[n[0]-1,:,:])
        fields.E[0,:,:,:]/(2*fields.dx)

        return fields
    
    def pot_diff_fixed_y(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])
        
        #E-field y-component differentiation
        fields.E[1,:,0,:] = -2*(fields.phi[:,0,:]-fields.phi[:,1,:])
        fields.E[1,:,1:n[1]-1,:] = -(fields.phi[:,0:n[1]-2,:] - fields.phi[:,2:n[1],:])
        fields.E[1,:,n[1]-1,:] = -2*(fields.phi[:,n[1]-2,:]-fields.phi[:,n[1]-1,:])
        fields.E[1,:,:,:]/(2*fields.dy)
        
        return fields
    
    
    def pot_diff_fixed_z(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])
        
        #E-field z-component differentiation
        fields.E[2,:,:,0] = -2*(fields.phi[:,:,0]-fields.phi[:,:,1])
        fields.E[2,:,:,1:n[2]-1] = -(fields.phi[:,:,0:n[2]-2] - fields.phi[:,:,2:n[2]])
        fields.E[2,:,:,n[2]-1] = -2*(fields.phi[:,:,n[2]-2]-fields.phi[:,:,n[2]-1])
        fields.E[2,:,:,:]/(2*fields.dz)
        
        return fields
    
    
    def pot_diff_open_x(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field x-component differentiation
        fields.E[0,0,:,:] = -(fields.phi[-3,:,:]-fields.phi[1,:,:])
        fields.E[0,1:n[0]-1,:,:] = -(fields.phi[0:n[0]-2,:,:] - fields.phi[2:n[0],:,:])
        fields.E[0,-2,:,:] = fields.E[0,0,:,:]
        fields.E[0,:,:,:]/(2*fields.dx)

        return fields
    
    def pot_diff_open_y(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])
        
        #E-field y-component differentiation
        fields.E[1,:,0,:] = -(fields.phi[:,-3,:]-fields.phi[:,1,:])
        fields.E[1,:,1:n[1]-1,:] = -(fields.phi[:,0:n[1]-2,:] - fields.phi[:,2:n[1],:])
        fields.E[1,:,-2,:] = fields.E[1,:,0,:]
        fields.E[1,:,:,:]/(2*fields.dy)
        
        return fields
    
    
    def pot_diff_open_z(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field z-component differentiation
        fields.E[2,:,:,0] = -(fields.phi[:,:,-3]-fields.phi[:,:,1])
        fields.E[2,:,:,1:n[2]-1] = -(fields.phi[:,:,0:n[2]-2] - fields.phi[:,:,2:n[2]])
        fields.E[2,:,:,-2] = fields.E[2,:,:,0]
        fields.E[2,:,:,:]/(2*fields.dz)
        
        return fields
    
    
    def trilinear_gather(self,species,mesh):
        #print(mesh.rho.sum())
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
        for pii in range(0,species.nq):
            li = self.cell_index(species.pos[pii],O,mesh.dh)
            rpos = species.pos[pii] - O - li*mesh.dh
            w = self.trilinear_weights(rpos,mesh.dh)
            i,j,k = li
            species.E[pii] = (w[0]*mesh.E[:,i,j,k] +
                              w[1]*mesh.E[:,i,j,k+1] +
                              w[2]*mesh.E[:,i,j+1,k] + 
                              w[3]*mesh.E[:,i,j+1,k+1] +
                              w[4]*mesh.E[:,i+1,j,k] +
                              w[5]*mesh.E[:,i+1,j,k+1] +
                              w[6]*mesh.E[:,i+1,j+1,k] + 
                              w[7]*mesh.E[:,i+1,j+1,k+1])
            
        return species
            
    
    def trilinear_qScatter(self,species_list,mesh,simulationManager):
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
        
        for species in species_list:
            for pii in range(0,species.nq):
                li = self.cell_index(species.pos[pii],O,mesh.dh)
                rpos = species.pos[pii] - O - li*mesh.dh
                w = self.trilinear_weights(rpos,mesh.dh)
                
                mesh.q[li[0],li[1],li[2]] += species.q * w[0]
                mesh.q[li[0],li[1],li[2]+1] += species.q * w[1]
                mesh.q[li[0],li[1]+1,li[2]] += species.q * w[2]
                mesh.q[li[0],li[1]+1,li[2]+1] += species.q * w[3]
                mesh.q[li[0]+1,li[1],li[2]] += species.q * w[4]
                mesh.q[li[0]+1,li[1],li[2]+1] += species.q * w[5]
                mesh.q[li[0]+1,li[1]+1,li[2]] += species.q * w[6]
                mesh.q[li[0]+1,li[1]+1,li[2]+1] += species.q * w[7]
        
        print(mesh.q[1,1,:])
        self.scatter_BC(species,mesh)
        print(mesh.q[1,1,:])
        mesh.rho = mesh.q/mesh.dv
        return mesh
            
            
    def trilinear_weights(self,rpos,dh):
        h = rpos/dh
        
        w = np.zeros(8,dtype=np.float)
        w[0] = (1-h[0])*(1-h[1])*(1-h[2])
        w[1] = (1-h[0])*(1-h[1])*(h[2])
        w[2] = (1-h[0])*(h[1])*(1-h[2])
        w[3] = (1-h[0])*(h[1])*(h[2])
        w[4] = (h[0])*(1-h[1])*(1-h[2])
        w[5] = (h[0])*(1-h[1])*(h[2])
        w[6] = (h[0])*(h[1])*(1-h[2])
        w[7] = (h[0])*(h[1])*(h[2])
        
        return w
    
    def cell_index(self,pos,O,dh):
        li = np.floor((pos-O)/dh)
        li = np.array(li,dtype=np.int)
        
        return li
    
    
    
######################## Particle Analysis Methods ############################
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
    
    
    def boris_staggered(self,species,mesh,simulationParameters,**kwargs):
        dt = simulationParameters.dt
        alpha = species.nq/species.mq

        self.fieldGather(species)
        
        species.vel = self.boris(species.vel,species.E,species.B,dt,alpha)
        species.pos = species.pos + simulationParameters.dt * species.vel
        self.check_boundCross(species,mesh,**kwargs)
        return species
    
    
    def boris_synced(self,species,mesh,simulationParameters,**kwargs):
        dt = simulationParameters.dt
        alpha = species.a
        species.pos = species.pos + dt * (species.vel + dt/2 * self.lorentz_std(species,mesh))
        self.check_boundCross(species,mesh,**kwargs)
        
        E_old = species.E
        self.fieldGather(species,mesh)
        E_new = species.E
        
        E_half = (E_old+E_new)/2
        
        species.vel = self.boris(species.vel,E_half,species.B,dt,alpha)
        return species
        
    
    def collSetup(self,species,fields,simulationManager,**kwargs):
        if self.nodeType == 'lobatto':
            self.ssi = 1    #Set sweep-start-index 'ssi'
            self.collocationClass = CollGaussLobatto
            self.updateStep = self.lobatto_update
            self.rhs_dt = (self.M - 1)*self.K
            
        elif self.nodeType == 'legendre':
            self.ssi = 0 
            self.collocationClass = CollGaussLegendre
            self.updateStep = self.legendre_update
            self.rhs_dt = (self.M + 1)*self.K
        
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
    
    
    def lobatto_update(self,species,mesh,x,v,*args,**kwargs):
        pos = x[:,-1]
        vel = v[:,-1]

        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)
        self.check_boundCross(species,mesh,**kwargs)

        return species
    
    
    def legendre_update(self,species,mesh,x,v,x0,v0,weights,Qmat,**kwargs):
        M = self.M
        d = 3*species.nq
        
        Id = np.identity(d)
        q = np.zeros(M+1,dtype=np.float)
        q[1:] = weights
        q = np.kron(q,Id)
        qQ = q @ np.kron(Qmat,Id)
        
        V0 = self.toVector(v0.transpose())
        F = self.FXV(species,mesh,x,v)
        
        vel = v0[:,0] + q @ F
        pos = x0[:,0] + q @ V0 + qQ @ F
        
        species.pos = species.toMatrix(pos)
        species.vel = species.toMatrix(vel)
        self.check_boundCross(species,mesh,**kwargs)
        return species
    
    
    def lorentzf(self,species,mesh,xm,vm,**kwargs):
        species.pos = species.toMatrix(xm)
        species.vel = species.toMatrix(vm)
        self.check_boundCross(species,mesh,**kwargs)

        self.fieldGather(species,mesh)

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
    
    
    
    def gatherE(self,species,mesh,x,**kwargs):
        species.pos = self.toMatrix(x,3)
        self.check_boundCross(species,mesh,**kwargs)
        
        self.fieldGather(species,mesh)
        
        return species.E
    
    def gatherB(self,species,mesh,x,**kwargs):
        species.pos = self.toMatrix(x,3)
        self.check_boundCross(species,mesh,**kwargs)
        
        self.fieldGather(species,mesh)
        
        return species.B
    
    
####################### Boundary Analysis Methods #############################
    def check_boundCross(self,species,mesh,**kwargs):
        for method in self.bound_cross_methods:
                method(species,mesh,**kwargs)
        return species
    
    def periodic_particles_x(self,species,mesh):    
        self.periodic_particles(species,0,mesh.xlimits)
        
    def periodic_particles_y(self,species,mesh):    
        self.periodic_particles(species,1,mesh.ylimits)
        
    def periodic_particles_z(self,species,mesh):    
        self.periodic_particles(species,2,mesh.zlimits)
    
    def periodic_particles(self,species,axis,limits):
        for pii in range(0,species.nq):
            if species.pos[pii,axis] < limits[0]:
                overshoot = limits[0]-species.pos[pii,axis]
                species.pos[pii,axis] = limits[1] - overshoot % (limits[1]-limits[0])

            elif species.pos[pii,axis] >= limits[1]:
                overshoot = species.pos[pii,axis] - limits[1]
                species.pos[pii,axis] = limits[0] + overshoot % (limits[1]-limits[0])
        
        
    def periodic_matrix_1d(self,species_list,mesh,controller):
        FDMat = self.FDMat.toarray()
        
        FDMat[0,:-1] = 0.
        FDMat[0,-1] = 1/mesh.dz**2
        FDMat[-1,0] = 1/mesh.dz**2

        BC_vector = np.zeros(mesh.BC_vector.shape[0]+1,dtype=np.float)

        BC_vector[0] = mesh.BC_vector[0]
        mesh.BC_vector = BC_vector
        
        self.FDMat = sps.csr_matrix(FDMat)

        
    def constant_phi_1d(self,species_list,mesh,controller):
        FDMat = self.FDMat.toarray()
        
        FDMat[0,:] = 1/mesh.dz**2
        FDMat[-1,0] = 1/mesh.dz**2

        BC_vector = np.zeros(mesh.BC_vector.shape[0]+1,dtype=np.float)

        BC_vector[0] = mesh.BC_vector[0]
        mesh.BC_vector = BC_vector
        
        self.FDMat = sps.csr_matrix(FDMat)
        
    
    def scatter_periodicBC_1d(self,species_list,mesh):
        mesh.q[1,1,0] += mesh.q[1,1,-2]       
        mesh.q[1,1,-2] = mesh.q[1,1,0] 
        
    def mirrored_boundary_z(self,species_list,mesh,controller):
        mesh.phi[:,:,-2] = mesh.phi[:,:,0]
        
    
################################ Hook methods #################################
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
    
    
    def energy_calc_penning(self,species_list,fields,simulationManager,**kwargs):
        for species in species_list:
            x = self.toVector(species.pos)
            v = self.toVector(species.vel)
            u = self.get_u(x,v)
            
            species.energy = u.transpose() @ self.H @ u
        
        return species_list
    
    
    def centreMass(self,species_list,fields,simulationManager,**kwargs):
        for species in species_list:
            nq = np.float(species.nq)
            mq = np.float(species.mq)
    
            species.cm[0] = np.sum(species.pos[:,0]*mq)/(nq*mq)
            species.cm[1] = np.sum(species.pos[:,1]*mq)/(nq*mq)
            species.cm[2] = np.sum(species.pos[:,2]*mq)/(nq*mq)
            
        return species_list
        

    def rhs_tally(self,species_list,fields,simulationManager):
        rhs_eval = self.rhs_dt * simulationManager.tSteps
        simulationManager.rhs_eval = rhs_eval
        
        return rhs_eval

############################ Misc. functionality ##############################
        
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
    
    def meshtoVector(self,mesh):
        shape = np.shape(mesh)
        x = np.zeros(shape[0]*shape[1]*shape[2],dtype=np.float)
        xi = 0
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    x[xi] = mesh[i,j,k]
                    xi += 1
        return x
    
    
    def vectortoMesh(self,x,shape):
        mesh = np.zeros(shape,dtype=np.float)
        xi = 0
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    mesh[i,j,k] = x[xi]
                    xi += 1
        return mesh
    
    def diagonals(self,N):
        lower = np.tri(N,k=-1) - np.tri(N,k=-2) 
        middle = np.identity(N)
        upper = np.tri(N,k=1) - np.tri(N,k=0) 
    
        return lower, middle, upper
    
    
    def makeSI(self):
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
        
        
    def setup_OpsList(self,opsList):
        for method in opsList:
            i = opsList.index(method)
            try:
                opsList[i] = getattr(self,method)
            except TypeError:
                pass
            
        return opsList
    
    def stringtoMethod(self,front):
        try:
            function = getattr(self,front)
            front = function
        except TypeError:
            pass
        
        return front
        
    def none(self,*args,**kwargs):
        pass
