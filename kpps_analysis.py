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
#import scipy.interpolate as scint
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
import time
import copy as cp
import matplotlib.pyplot as plt
## Class
class kpps_analysis:
    def __init__(self,**kwargs):
        ## Default values
        self.mu0 = 4*pi*10**(-7) #Vacuum permeability (H/m) 
        self.ep0 = 8.854187817*10**(-12) #Vacuum permittivity (F/m)
        self.q0 = 1.602176620898*10**(-19) #Elementary charge (C)
        
        self.E_type = 'none'
        self.E_magnitude = 0
        self.E_transform = np.zeros((3,3),dtype=np.float)
        self.static_E = 0
        self.custom_static_E = self.none
        self.custom_external_E = self.none
        
        self.coulomb = self.coulomb_cgs
        self.lambd = 0 
        
        self.B_type = 'none'
        self.B_magnitude = 0
        self.B_transform = np.zeros((1,3),dtype=np.float)
        self.static_B = 0
        self.custom_static_B = self.none
        self.custom_external_B = self.none
        
         # Hook inputs
        self.pre_hook_list = []
        self.hook_list = []
        
         # Quick hook selection flags
        self.centreMass_check = False
        self.coulomb_field_check = False
        self.residual_check = False
        self.convergence_check = False
        self.rhs_check = False
        
        
        self.particleIntegration = False
        self.particleIntegrator = 'boris_SDC'
        self.nodeType = 'lobatto'
        self.M = 2
        self.K = 1
        self.rhs_dt = 1
        self.gather = self.none
        self.bound_cross_methods = []
        self.looped_axes = []
        self.calc_residuals = self.calc_residuals_max
        self.SDC_residual_type = 'nodal'
        self.display_residuals = self.display_residuals_max
        
        self.fieldIntegration = False
        self.field_type = 'custom' #Can be pic, coulomb or custom
        self.field_solver = self.direct_solve
        self.iter_x0 = None
        self.iter_tol = 1e-05
        self.iter_max = None
        self.niter = 0
        self.FDMat = None
        self.precon = None
        self.scatter_order = 1
        self.gather_order = 1
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
        self.mi_zN = -2
        self.mi_yN = -2
        self.mi_xN = -2
        self.solver_pre = self.none
        self.solver_post = self.none
        
        self.external_fields = False
        self.custom_q_background = self.none
        self.custom_rho_background = self.none
        self.custom_E_background = self.none
        self.custom_B_background = self.none
        
        self.scatter = self.none
        self.scatter_BC = self.none
        self.fIntegrator_setup = self.poisson_cube2nd_setup
        self.fIntegrator = self.poisson_cube2nd
        self.external_fields_mesh = False
        
        
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
        self.params = cp.deepcopy(kwargs)
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
                if self.gather_order == 1:
                    self.gather = self.trilinear_gather
                elif self.gather_order%2 == 0:
                    self.gather = self.poly_gather_1d_even
                    self.preAnalysis_methods.append(self.poly_gather_setup)
                elif self.gather_order%2 != 0:
                    self.gather = self.poly_gather_1d_odd
                    self.preAnalysis_methods.append(self.poly_gather_setup)
                    
                self.scatter = self.trilinear_qScatter
                
                    
            elif self.field_type == 'coulomb':
                self.gather = self.coulomb 
                self.scatter = self.none
                
                
            else:
                pass
            
            self.fieldIntegrator_methods.append(self.scatter)
            
            
        # Setup required field analysis methods
        if self.fieldIntegration == True:           
            if self.field_type == 'pic':
                self.field_solver = self.stringtoMethod(self.field_solver)
                self.preAnalysis_methods.append(self.fIntegrator_setup)
                self.preAnalysis_methods.append(self.calc_background)
                self.preAnalysis_methods.append(self.impose_background)
                self.preAnalysis_methods.append(self.scatter)
                self.preAnalysis_methods.append(self.fIntegrator)
                self.fieldIntegrator_methods.append(self.fIntegrator)
                


        if self.external_fields_mesh  == True:
            self.preAnalysis_methods.append(self.calc_static_E)
            self.preAnalysis_methods.append(self.calc_static_B)
            
            self.fieldIntegrator_methods.append(self.impose_static_E)
            self.fieldIntegrator_methods.append(self.impose_static_B)
                
        if self.external_fields == True:
            self.fieldGather_methods.append(self.eFieldImposed)
            self.fieldGather_methods.append(self.bFieldImposed)
            
            
        # Setup required particle analysis methods
        if self.particleIntegration == True:
            self.particleIntegrator_methods.append(self.particleIntegrator)
            
            if  'boris_SDC' in self.particleIntegrator:
                self.preAnalysis_methods.append(self.collSetup)
                
            self.fieldGather_methods.append(self.gather)
                

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
            
        if self.residual_check == True and self.particleIntegrator == 'boris_SDC':
            self.hooks.append(self.display_residuals)
            
        if self.convergence_check == True and self.particleIntegrator == 'boris_SDC':
            self.hooks.append(self.display_convergence)
        
        self.scatter_BC = self.stringtoMethod(self.scatter_BC)
        
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
    def run_particleIntegrator(self,species_list,fields,simulationManager,**kwargs):
        for method in self.particleIntegrator_methods:
            method(species_list,fields,simulationManager)
            #print(abs(species_list[0].pos[0,0]-13.2063)/abs(-13.2063))
        return species_list
        
    def run_fieldIntegrator(self,species_list,fields,simulationManager,**kwargs):     
        fields = self.impose_background(species_list,fields,simulationManager)
        self.niter = 0
        for method in self.fieldIntegrator_methods:
            method(species_list,fields,simulationManager)
        
        fields.gmres_iters += self.niter
        return species_list


    def fieldGather(self,species,fields,**kwargs):
        #Establish field values at particle positions via methods specified at initialisation.
        species.E = np.zeros(species.E.shape,dtype=np.float)
        species.B = np.zeros(species.B.shape,dtype=np.float)

        for method in self.fieldGather_methods:
            method(species,fields)

        return species
    
    def runHooks(self,species_list,fields,**kwargs):
        for method in self.hooks:
            #print(method)
            method(species_list,fields,**kwargs)

        return species_list, fields
    
    
    def run_preAnalyser(self,species_list,mesh,controller,**kwargs):
        print("Running pre-processing:")        
        print("Checking for boundary crossings...")
        for species in species_list:
            self.check_boundCross(species,mesh,**kwargs)
        
        print("Performing pre-run analysis...")
        for method in self.preAnalysis_methods:
            #print(method)
            method(species_list, mesh,controller,**kwargs)

        for species in species_list:
            print("Evaluating initial field for " + species.name + " species.")
            
            t_bc = time.time()
            self.check_boundCross(species,mesh,**kwargs)
            
            t_fg = time.time()
            self.fieldGather(species,mesh,**kwargs)
            
            t_lntz = time.time()
            species.E_half = species.E
            species.lntz = species.a*(species.E + np.cross(species.vel,species.B))
            
            controller.runTimeDict['bound_cross_check'] += t_fg - t_bc 
            controller.runTimeDict['gather'] += t_lntz - t_fg

        return species_list, mesh
    
    def run_postAnalyser(self,species_list,fields,simulationManager,**kwargs):
        print("Running post-processing...")
        for species in species_list:
            for method in self.postAnalysis_methods:
                #print(method)
                method(species_list,fields,simulationManager)
        
        return species_list, fields
    

##################### Imposed E-Field Methods #################################
    def eFieldImposed(self,species,fields,**kwargs):

        if self.E_type == "transform":
            for pii in range(0,species.nq):
                direction = np.dot(self.E_transform,species.pos[pii,:])
                species.E[pii,:] += direction * self.E_magnitude
                
        if self.E_type == "exponential":
                direction = species.pos[pii,:]/np.linalg.norm(species.pos[pii,:])
                species.E[pii,:] += direction * np.exp(species.pos[pii,:])
                
        if self.E_type == "custom":
            fields = self.custom_external_E(species,fields,controller=None)

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
        self.coulomb_cgs(species,fields) * 1/(4*pi*self.ep0)
        
        return species

    
    
##################### Imposed B-Field Methods #################################
    def bFieldImposed(self,species,fields,**kwargs):
        if self.B_type == 'uniform':
            try:
                species.B[:,0:] += np.multiply(self.B_magnitude,self.B_transform)
            except TypeError:
                print("Analyser: TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")
                
        if self.E_type == "custom":
            fields = self.custom_external_B(species,fields,controller=None)

        return species
        
    
########################## Field Analysis Methods #############################
    def calc_static_E(self,species_list,fields,controller):
        if self.E_type == "transform":
            inputMatrix = np.array(self.E_transform)
            for xi in range(0,len(fields.pos[0,:,0,0])):
                for yi in range(0,len(fields.pos[0,0,:,0])):
                    for zi in range(0,len(fields.pos[0,0,0,:])):
                        direction = np.dot(inputMatrix,fields.pos[:,xi,yi,zi])
                        fields.E[:,xi,yi,zi] += direction * self.E_magnitude 

        if self.E_type == "custom":
            fields, static_E = self.custom_static_E(species_list,fields,controller)

        self.static_E = np.zeros(np.shape(fields.E))
        self.static_E[:] = fields.E[:]
        
        return fields
    
    
    def calc_static_B(self,species_list,fields,controller):

        if self.B_type == "uniform":
            bMag = self.B_magnitude
            direction = np.array(self.B_transform)
            try:
                for xi in range(0,len(fields.pos[0,:,0,0])):
                    for yi in range(0,len(fields.pos[0,0,:,0])):
                        for zi in range(0,len(fields.pos[0,0,0,:])):
                            fields.B[:,xi,yi,zi] -= np.multiply(bMag,direction)
            except TypeError:
                print("Analyser: TypeError raised, did you input a length 3 vector "
                      + "as transform to define the uniform magnetic field?")
                
        if self.B_type == "custom":
            fields, static_B = self.custom_static_B(species_list,fields,controller=None)
     
        self.static_B = np.zeros(np.shape(fields.B))
        self.static_B[:] = fields.B[:]
        return fields
    
    def calc_background(self,species_list,fields,controller=None):
        self.custom_q_background(species_list,fields,controller=controller,q_bk=fields.q_bk)
        self.custom_rho_background(species_list,fields,controller=controller,rho_bk=fields.rho_bk)
        self.custom_E_background(species_list,fields,controller=controller,E_bk=fields.E_bk)
        self.custom_B_background(species_list,fields,controller=controller,B_bk=fields.B_bk)

        return fields
        
    def impose_static_E(self,species_list,fields,controller=None):
        fields.E += self.static_E

        return fields
    
    def impose_static_B(self,species_list,fields,controller=None):
        fields.B += self.static_B
        
        return fields
    
    
    def impose_background(self,species_list,fields,controller=None):
        fields.q[:,:,:] = fields.q_bk[:,:,:]
        fields.rho[:,:,:] = fields.rho_bk[:,:,:]
        fields.E[:,:,:,:] = fields.E_bk[:,:,:,:]
        fields.B[:,:,:,:] = fields.B_bk[:,:,:,:]

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


    def poisson_cube2nd_setup(self,species_list,fields,controller,**kwargs):
        tStart = time.time()
        
        self.interior_shape = fields.res-1
        nx = self.interior_shape[0]
        ny = self.interior_shape[1]
        nz = self.interior_shape[2]
        
        FDMatrix_adjust_z = self.none
        FDMatrix_adjust_y = self.none
        FDMatrix_adjust_x = self.none
        
        if self.mesh_boundary_z == 'open':
            self.interior_shape[2] += 1
            FDMatrix_adjust_z = self.poisson_M_adjust_1d
            self.scatter_BC = self.scatter_periodicBC_1d
            self.pot_differentiate_z = self.pot_diff_open_z
            
        nz = self.interior_shape[2] 
        k = np.zeros(3,dtype=np.float)
        k[0] = -2*(1/fields.dz**2)
        k[1] = -2*(1/fields.dy**2 + 1/fields.dz**2)
        k[2] = -2*(1/fields.dx**2 + 1/fields.dy**2 + 1/fields.dz**2)
        
        diag = [1/fields.dz**2,k[controller.ndim-1],1/fields.dz**2]
        Dk = sps.diags(diag,offsets=[-1,0,1],shape=(nz,nz))
        self.FDMat = Dk

        FDMatrix_adjust_z(species_list,fields,controller)
        self.pot_diff_list.append(self.pot_differentiate_z)
        
        if controller.ndim >= 2:
            if self.mesh_boundary_y == 'open':
                self.interior_shape[1] += 1
                FDMatrix_adjust_y = self.poisson_M_adjust_2d
                self.pot_differentiate_y = self.pot_diff_open_y
            
            ny = self.interior_shape[1]
            I = sps.identity(nz)
            diag = sps.diags([1],shape=(ny,ny))
            off_diag = sps.diags([1,1],offsets=[-1,1],shape=(ny,ny))
            FDMatrix_adjust_y(species_list,fields,controller)
            
            Ek = sps.kron(diag,Dk) + sps.kron(off_diag,I/fields.dy**2)
            self.FDMat = Ek
            self.pot_diff_list.append(self.pot_differentiate_y)
            
        if controller.ndim == 3:
            if self.mesh_boundary_x == 'open':
                self.interior_shape[0] += 1
                FDMatrix_adjust_x = self.poisson_M_adjust_3d
                self.pot_differentiate_x = self.pot_diff_open_x
            
            nx = self.interior_shape[0]
            J = sps.identity(nz*ny)
            diag = sps.diags([1],shape=(nx,nx))
            off_diag = sps.diags([1,1],offsets=[-1,1],shape=(nx,nx))
            FDMatrix_adjust_x(species_list,fields,controller)
            
            Fk = sps.kron(diag,Ek) + sps.kron(off_diag,J/fields.dx**2)
            self.FDMat = Fk
            self.pot_diff_list.append(self.pot_differentiate_x)
            
        controller.runTimeDict['FD_setup'] = time.time() - tStart
        
        ilu = sps.linalg.spilu(self.FDMat,drop_tol=0.5,fill_factor=2,)
        Mx = lambda x: ilu.solve(x)
        self.precon = sps.linalg.LinearOperator((self.FDMat.shape[0],self.FDMat.shape[1]), Mx)

        return self.FDMat
    
        
    def poisson_cube2nd(self,species_list,fields,controller):
        tst = time.time()
        rho = self.meshtoVector(fields.rho[self.mi_x0:self.mi_xN,
                                           self.mi_y0:self.mi_yN,
                                           self.mi_z0:self.mi_zN])

        self.solver_pre(species_list,fields,controller)
        
        phi = self.field_solver(self.FDMat,rho*self.unit_scale_poisson,fields.BC_vector)
        phi = self.vectortoMesh(phi,self.interior_shape)

        fields.phi[self.mi_x0:self.mi_xN,
                   self.mi_y0:self.mi_yN,
                   self.mi_z0:self.mi_zN] = phi

        self.solver_post(species_list,fields,controller)

        for nd in range(0,controller.ndim):
            self.pot_diff_list[nd](fields)
            
        controller.runTimeDict['field_solve'] += time.time() - tst

        return fields
    
    def direct_solve(self,FDMat,rho,BC_vector):
        phi = sps.linalg.spsolve(FDMat, -rho - BC_vector)
        
        return phi
    
    def gmres_solve(self,FDMat,rho,BC_vector):
        phi, self.solver_code = sps.linalg.gmres(FDMat, -rho - BC_vector,
                                                   x0=self.iter_x0,
                                                   tol=self.iter_tol,
                                                   maxiter=self.iter_max,
                                                   M=self.precon,
                                                   callback = self.iterative_counter)
        
        self.iter_x0 = phi
        
        return phi
    
    def bicgstab_solve(self,FDMat,rho,BC_vector):
        phi, self.solver_code = sps.linalg.bicgstab(FDMat, -rho - BC_vector,
                                                   x0=self.iter_x0,
                                                   tol=self.iter_tol,
                                                   maxiter=self.iter_max,
                                                   M=self.precon,
                                                   callback = self.iterative_counter)
        
        self.iter_x0 = phi
        
        return phi
    
    def iterative_counter(self,ck=None):
        self.niter += 1

    
    def pot_diff_fixed_x(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field x-component differentiation
        fields.E[0,0,:,:] = 2*(fields.phi[0,:,:]-fields.phi[1,:,:])
        fields.E[0,1:n[0]-1,:,:] = (fields.phi[0:n[0]-2,:,:] - fields.phi[2:n[0],:,:])
        fields.E[0,n[0]-1,:,:] = 2*(fields.phi[n[0]-2,:,:]-fields.phi[n[0]-1,:,:])
        fields.E[0,:,:,:] =  fields.E[0,:,:,:]/(2*fields.dx)

        return fields
    
    def pot_diff_fixed_y(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])
        
        #E-field y-component differentiation
        fields.E[1,:,0,:] = 2*(fields.phi[:,0,:]-fields.phi[:,1,:])
        fields.E[1,:,1:n[1]-1,:] = (fields.phi[:,0:n[1]-2,:] - fields.phi[:,2:n[1],:])
        fields.E[1,:,n[1]-1,:] = 2*(fields.phi[:,n[1]-2,:]-fields.phi[:,n[1]-1,:])
        fields.E[1,:,:,:] = fields.E[1,:,:,:]/(2*fields.dy)
        
        return fields
    
    
    def pot_diff_fixed_z(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field z-component differentiation
        fields.E[2,:,:,0] = 2*(fields.phi[:,:,0]-fields.phi[:,:,1])
        fields.E[2,:,:,1:n[2]-1] = (fields.phi[:,:,0:n[2]-2] - fields.phi[:,:,2:n[2]])
        fields.E[2,:,:,n[2]-1] = 2*(fields.phi[:,:,n[2]-2]-fields.phi[:,:,n[2]-1])
        fields.E[2,:,:,:] = fields.E[2,:,:,:]/(2*fields.dz)

        return fields
    
    
    def pot_diff_open_x(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field x-component differentiation
        fields.E[0,0,:,:] = (fields.phi[-3,:,:]-fields.phi[1,:,:])
        fields.E[0,1:n[0]-1,:,:] = (fields.phi[0:n[0]-2,:,:] - fields.phi[2:n[0],:,:])
        fields.E[0,-2,:,:] = fields.E[0,0,:,:]
        fields.E[0,:,:,:] = fields.E[0,:,:,:]/(2*fields.dx)

        return fields
    
    def pot_diff_open_y(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])
        
        #E-field y-component differentiation
        fields.E[1,:,0,:] = (fields.phi[:,-3,:]-fields.phi[:,1,:])
        fields.E[1,:,1:n[1]-1,:] = (fields.phi[:,0:n[1]-2,:] - fields.phi[:,2:n[1],:])
        fields.E[1,:,-2,:] = fields.E[1,:,0,:]
        fields.E[1,:,:,:] = fields.E[1,:,:,:]/(2*fields.dy)
        
        return fields
    
    
    def pot_diff_open_z(self,fields):
        ## Differentiate over electric potential for electric field
        n = np.shape(fields.phi[0:-1,0:-1,0:-1])

        #E-field z-component differentiation
        fields.E[2,:,:,0] = (fields.phi[:,:,-3]-fields.phi[:,:,1])
        fields.E[2,:,:,1:n[2]-1] = (fields.phi[:,:,0:n[2]-2] - fields.phi[:,:,2:n[2]])
        fields.E[2,:,:,-2] = fields.E[2,:,:,0]
        fields.E[2,:,:,:] = fields.E[2,:,:,:]/(2*fields.dz)
        
        return fields
    
    
    def trilinear_gather(self,species,mesh):
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
    
        li = self.lower_index(species.pos,O,mesh.dh)
        rpos = species.pos - O - li*mesh.dh
        w = self.trilinear_weights(rpos,mesh.dh)
        
        i = li[:,0]
        j = li[:,1]
        k = li[:,2]

        for comp in range(0,3):
            species.E[:,comp] += w[:,0]*mesh.E[comp,i,j,k]
            species.E[:,comp] += w[:,1]*mesh.E[comp,i,j,k+1]
            species.E[:,comp] += w[:,2]*mesh.E[comp,i,j+1,k]
            species.E[:,comp] += w[:,3]*mesh.E[comp,i,j+1,k+1]
            species.E[:,comp] += w[:,4]*mesh.E[comp,i+1,j,k]
            species.E[:,comp] += w[:,5]*mesh.E[comp,i+1,j,k+1]
            species.E[:,comp] += w[:,6]*mesh.E[comp,i+1,j+1,k]
            species.E[:,comp] += w[:,7]*mesh.E[comp,i+1,j+1,k+1]
    
        return species
    
    
    def poly_interpol_setup(self,species_list,mesh,controller):
        kg = self.gather_order
        ks = self.scatter_order
        kList = [kg,ks]

        nodesList = []
        for k in kList:
            interpol_nodes = np.zeros((mesh.res[2]+1,k+2),dtype=np.int)
            if self.mesh_boundary_z == 'open':
                for i in range(0,mesh.res[2]+1):
                    interpol_nodes[i,0] = i
                    min_j = i - np.ceil((k+1)/2)+1
                    max_j = (i + np.ceil((k)/2))
                    interpol_nodes[i,1:] = np.linspace(min_j,max_j,k+1)%(mesh.res[2]+1)
    
    #        else:
    #            for i in range(0,mesh.res[2]):
    #                min_j = i - np.floor(k/2)
    #                max_j = i + np.floor((k+1)/2)
    #                mesh.interpol_nodes[i,:] = np.linspace(min_j,max_j,k+1)
                
            interpol_nodes = interpol_nodes.astype(int)
            nodesList.append(interpol_nodes)

        mesh.gather_nodes = nodesList[0]
        mesh.scatter_nodes = nodesList[1]
        
        return mesh
        
    def poly_gather_1d_odd(self,species,mesh):
        index_finder = self.lower_index
        species = self.poly_gather_1d(species,mesh,index_finder)
        
        return species
    
    
    def poly_gather_1d_even(self,species,mesh):
        index_finder = self.close_index
        species = self.poly_gather_1d(species,mesh,index_finder)
        
        return species
    
    
    def poly_gather_1d(self,species,mesh,index_method):
        k = self.gather_order
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
        for pii in range(0,species.nq):
            Ej = []
            index = index_method(species.pos[pii],O,mesh.dh)
            
            xj_i = mesh.gather_nodes[index[2],1:]
            
            xj = mesh.z[1,1,xj_i]
            c = np.ones(k+1)
            
            for j in range(0,k+1):
                for m in range(0,j):
                    c[j] *= (species.pos[pii,2] - xj[m])/(xj[j] - xj[m])
                for m in range(j+1,k+1):
                    c[j] *= (species.pos[pii,2] - xj[m])/(xj[j] - xj[m])
                    
            for i in xj_i:    
                Ej.append(mesh.E[2,1,1,i])
            Ej = np.array(Ej)

            E = Ej*c
            species.E[pii,2] = E.sum()
                
        return species
            
    
    def trilinear_qScatter(self,species_list,mesh,controller):
        tst = time.time()
        
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
        for species in species_list:
            li = self.lower_index(species.pos,O,mesh.dh)
            rpos = species.pos - O - li*mesh.dh
            w = self.trilinear_weights(rpos,mesh.dh)

            i = li[:,0]
            j = li[:,1]
            k = li[:,2]
            
            np.add.at(mesh.q,tuple([i,j,k]),species.q*w[:,0])
            np.add.at(mesh.q,tuple([i,j,k+1]),species.q*w[:,1])
            np.add.at(mesh.q,tuple([i,j+1,k]),species.q*w[:,2])
            np.add.at(mesh.q,tuple([i,j+1,k+1]),species.q*w[:,3])
            np.add.at(mesh.q,tuple([i+1,j,k]),species.q*w[:,4])
            np.add.at(mesh.q,tuple([i+1,j,k+1]),species.q*w[:,5])
            np.add.at(mesh.q,tuple([i+1,j+1,k]),species.q*w[:,6])
            np.add.at(mesh.q,tuple([i+1,j+1,k+1]),species.q*w[:,7])

        self.scatter_BC(species_list,mesh,controller)

        mesh.rho += mesh.q/mesh.dv
        
        controller.runTimeDict['scatter'] += time.time() - tst
        return mesh
    
    
    def quadratic_qScatter_1d(self,species_list,mesh,controller):
        O = np.array([mesh.xlimits[0],mesh.ylimits[0],mesh.zlimits[0]])
        for species in species_list:
            for pii in range(0,species.nq):
                ci = self.close_index(species.pos[pii],O,mesh.dh)
                rpos = species.pos[pii] - O - ci*mesh.dh
                w = self.quadratic_weights_1d(rpos,mesh.dh)

                mesh.q[ci[0],ci[1],mesh.scatter_nodes[ci[2],1]] += species.q*w[0]
                mesh.q[ci[0],ci[1],mesh.scatter_nodes[ci[2],2]] += species.q*w[1]
                mesh.q[ci[0],ci[1],mesh.scatter_nodes[ci[2],3]] += species.q*w[2]
                
            self.scatter_BC(species,mesh,controller)
            
        mesh.rho += mesh.q/mesh.dv
        return mesh
    
    
#    def griddata_qScatter(self,species_list,mesh,controller):
#        ## Not working, establishes convex hull around particles and only
#        ## interpolates to mesh nodes within hull.
#        ## Doesn't appear cumulative either or to spread charge over a cell.
#        for species in species_list:
#            mesh.q += scint.griddata(species.pos,species.vals_at_p(species.q),
#                                     (mesh.x,mesh.y,mesh.z),
#                                     method='linear',fill_value=0)
#
#        self.scatter_BC(species,mesh,controller)
#        mesh.rho += mesh.q/mesh.dv
#        return mesh
            
    
    def trilinear_weights(self,rpos,dh):
        h = rpos/dh
        
        w = np.zeros((rpos.shape[0],8),dtype=np.float)
        w[:,0] = (1-h[:,0])*(1-h[:,1])*(1-h[:,2])
        w[:,1] = (1-h[:,0])*(1-h[:,1])*(h[:,2])
        w[:,2] = (1-h[:,0])*(h[:,1])*(1-h[:,2])
        w[:,3] = (1-h[:,0])*(h[:,1])*(h[:,2])
        w[:,4] = (h[:,0])*(1-h[:,1])*(1-h[:,2])
        w[:,5] = (h[:,0])*(1-h[:,1])*(h[:,2])
        w[:,6] = (h[:,0])*(h[:,1])*(1-h[:,2])
        w[:,7] = (h[:,0])*(h[:,1])*(h[:,2])
        
        return w
    
    
    def quadratic_weights_1d(self,rpos,dh):
        h = rpos/dh
        w = np.zeros(3,dtype=np.float)
        
        w[0] = 1/2*h[2]**2 - 1/2*h[2]
        w[1] = 1-h[2]**2
        w[2] = 1/2*h[2]**2 + 1/2*h[2]
        
        return w
    
    def lower_index(self,pos,O,dh):
        li = np.floor((pos-O)/dh)
        li = np.array(li,dtype=np.int)
        
        return li
    
    def upper_index(self,pos,O,dh):
        ui = np.ceil((pos-O)/dh)
        ui = np.array(ui,dtype=np.int)
        
        return ui
    
    def close_index(self,pos,O,dh):
        i = (pos-O)/dh

        li = np.floor((pos-O)/dh)
        ui = np.ceil((pos-O)/dh)
        
        ci = np.where(i-li <= ui-i,li,ui)
        ci = np.array(ci,dtype=np.int)

        return ci
    
    
    
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
    
    
    def boris_staggered(self,species_list,mesh,controller,**kwargs):        
        dt = controller.dt
        self.run_fieldIntegrator(species_list,mesh,controller)
        tst = time.time()
        for species in species_list:
            alpha = species.a
            
            t_gather = time.time()
            self.fieldGather(species,mesh)
            
            t_boris = time.time()
            species.vel = self.boris(species.vel,species.E,species.B,dt,alpha)
            
            t_pos = time.time()
            species.pos = species.pos + controller.dt * species.vel
            
            t_bc = time.time()
            self.check_boundCross(species,mesh,**kwargs)
            
            controller.runTimeDict['bound_cross_check'] += time.time() - t_bc
            controller.runTimeDict['gather'] += t_boris - t_gather
            controller.runTimeDict['boris'] += t_pos - t_boris
            controller.runTimeDict['pos_push'] += t_bc - t_pos
            
        controller.runTimeDict['particle_push'] += time.time() - tst
            
        return species_list

    
    def boris_synced(self,species_list,mesh,controller,**kwargs):
        tst = time.time()
        
        dt = controller.dt
        for species in species_list:
            alpha = species.a
            
            t_pos = time.time()
            species.pos = species.pos + dt * (species.vel + dt/2 * self.lorentz_std(species,mesh))
            
            t_bc = time.time()
            self.check_boundCross(species,mesh,**kwargs)
            
            controller.runTimeDict['bound_cross_check'] += time.time() - t_bc
            controller.runTimeDict['pos_push'] += t_bc - t_pos
        
        controller.runTimeDict['particle_push'] += time.time() - tst
        self.run_fieldIntegrator(species_list,mesh,controller)
        
        tmid = time.time()
        for species in species_list:
            t_gather = time.time()
            E_old = species.E
            self.fieldGather(species,mesh)
            E_new = species.E
    
            species.E_half = (E_old+E_new)/2
            
            t_boris = time.time()
            species.vel = self.boris(species.vel,species.E_half,species.B,dt,alpha)
            
            controller.runTimeDict['gather'] += t_boris - t_gather
            controller.runTimeDict['boris'] += time.time() - t_boris
            
        controller.runTimeDict['particle_push'] += time.time() - tmid
        return species_list
        
    
    def collSetup(self,species_list,fields,controller=None,**kwargs):
        M = self.M
        K = self.K
        dt = controller.dt

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

        for species in species_list:
            self.fieldGather(species,fields)
            species.F = species.a*(species.E + np.cross(species.vel,species.B))

        self.coll_params = {}
        
        self.coll_params['dt'] = controller.dt
        
        #Remap collocation weights from [0,1] to [tn,tn+1]
        #nodes = (t-dt) + self.nodes * dt
        self.coll_params['weights'] = self.weights * dt 
        
        Qmat = self.Qmat * dt
        Smat = self.Smat * dt
        delta_m = self.delta_m * dt

        self.coll_params['Qmat'] = Qmat
        self.coll_params['Smat'] = Smat
        self.coll_params['dm'] = delta_m

        #Define required calculation matrices
        QE = np.zeros((M+1,M+1),dtype=np.float)
        QI = np.zeros((M+1,M+1),dtype=np.float)
        QT = np.zeros((M+1,M+1),dtype=np.float)
        
        SX = np.zeros((M+1,M+1),dtype=np.float)
        
        for i in range(0,M):
            QE[(i+1):,i] = delta_m[i]
            QI[(i+1):,i+1] = delta_m[i] 
        
        QT = 1/2 * (QE + QI)
        QX = QE @ QT + (QE*QE)/2
        SX[:,:] = QX[:,:]
        SX[1:,:] = QX[1:,:] - QX[0:-1,:]      
        
        
        self.coll_params['SX'] = SX
        self.coll_params['SQ'] = Smat @ Qmat

        for species in species_list:
            d = 3*species.nq
            species.x0 = np.zeros((d,M+1),dtype=np.float)
            species.v0 = np.zeros((d,M+1),dtype=np.float)
            
            species.xn = np.zeros((d,M+1),dtype=np.float)
            species.vn = np.zeros((d,M+1),dtype=np.float)
            
            species.F = np.zeros((d,M+1),dtype=np.float)
            species.Fn = np.zeros((d,M+1),dtype=np.float)
            
            species.x_con = np.zeros((K,M))
            species.x_res = np.zeros((K,M))
            species.v_con = np.zeros((K,M))
            species.v_res = np.zeros((K,M))
            
            # Required residual matrices
            if self.SDC_residual_type == 'matrix':
                species.U0 = np.zeros((2*d*(M+1),1),dtype=np.float)
                species.Uk = np.zeros((2*d*(M+1),1),dtype=np.float) 
                species.R = np.zeros((K,1),dtype=np.float) 
                species.FXV = np.zeros((d*(M+1),1),dtype=np.float)   
                
                Ix = np.array([[1],[0]])
                Iv = np.array([[0],[1]],np.newaxis)
                Ixv = np.array([[0,1],[0,0]])
                Id = np.identity(d)
                
                size = (M+1)*2*d
                species.Imd = np.identity(size)
                
                QQ = self.Qmat @ self.Qmat
                QQX = np.kron(QQ,Ix)
                QQX = np.kron(QQX,Id)
                
                QV = np.kron(self.Qmat,Iv)
                QV = np.kron(QV,Id)
                
                QXV = np.kron(self.Qmat,Ixv)
                QXV = np.kron(QXV,Id)
                
                species.Cc = species.Imd + QXV
                species.Qc = QQX + QV
                
                self.calc_R = self.calc_residual
                
            elif self.SDC_residual_type == 'nodal':
                species.Rx = np.zeros((K,M),dtype=np.float) 
                species.Rv = np.zeros((K,M),dtype=np.float) 
                self.calc_R = self.calc_residual2
                

    def boris_SDC(self, species_list,fields, controller,**kwargs):
        tst = time.time()
        
        M = self.M
        K = self.K

        #Remap collocation weights from [0,1] to [tn,tn+1]
        weights =  self.coll_params['weights']

        Qmat =  self.coll_params['Qmat']
        Smat =  self.coll_params['Smat']

        dm =  self.coll_params['dm']

        SX =  self.coll_params['SX'] 

        SQ =  self.coll_params['SQ']

        for species in species_list:
            ## Populate node solutions with x0, v0, F0 ##
            species.x0[:,0] = self.toVector(species.pos)
            species.v0[:,0] = self.toVector(species.vel)
            species.F[:,0] = self.toVector(species.lntz)
            species.En_m0 = species.E

            for m in range(1,M+1):
                species.x0[:,m] = species.x0[:,0]
                species.v0[:,m] = species.v0[:,0]
                species.F[:,m] = species.F[:,0]
            #############################################
            
            species.x = np.copy(species.x0)
            species.v = np.copy(species.v0)
            
            species.xn[:,:] = species.x[:,:]
            species.vn[:,:] = species.v[:,:]
            species.Fn[:,:] = species.F[:,:]
        
        controller.runTimeDict['particle_push'] += time.time() - tst
        
        #print()
        #print(simulationManager.ts)
        for k in range(1,K+1):
            #print("k = " + str(k))
            for species in species_list:
                species.En_m = species.En_m0 #reset electric field values for new sweep
                
            for m in range(self.ssi,M):
                for species in species_list:
                    t_pos = time.time()
                    #print("m = " + str(m))
                    #Determine next node (m+1) positions
                    sumSQ = 0
                    for l in range(1,M+1):
                        sumSQ += SQ[m+1,l]*species.F[:,l]
                    
                    sumSX = 0
                    for l in range(1,m+1):
                        sumSX += SX[m+1,l]*(species.Fn[:,l] - species.F[:,l])
                        
                    species.xQuad = species.xn[:,m] + dm[m]*species.v[:,0] + sumSQ
                              
                    ### POSITION UPDATE FOR NODE m/SWEEP k ###
                    species.xn[:,m+1] = species.xQuad + sumSX 
                    
                    ##########################################
                    
                    sumS = 0
                    for l in range(1,M+1):
                        sumS += Smat[m+1,l] * species.F[:,l]
                    
                    species.vQuad = species.vn[:,m] + sumS
                    
                    species.ck_dm = -1/2 * (species.F[:,m+1]+species.F[:,m]) + 1/dm[m] * sumS
                    print(species.ck_dm)
                    ### FIELD GATHER FOR m/k NODE m/SWEEP k ###
                    species.pos = self.toMatrix(species.xn[:,m+1],3)
#                    print(species.pos)
                    
                    t_bc = time.time()
                    self.check_boundCross(species,fields,**kwargs)
                    
                    controller.runTimeDict['bound_cross_check'] += time.time() - t_bc
                    controller.runTimeDict['pos_push'] += t_bc - t_pos
                    
                controller.runTimeDict['particle_push'] += time.time() - t_pos
                self.run_fieldIntegrator(species_list,fields,controller)
                
                tmid = time.time()
                for species in species_list:
                    t_gather = time.time()
                    self.fieldGather(species,fields)
                    ###########################################
                    
                    #Sample the electric field at the half-step positions (yields form Nx3)
                    half_E = (species.En_m+species.E)/2
                    species.En_m = species.E              #Save m+1 value as next node's m value
                    
                    #Resort all other 3d vectors to shape Nx3 for use in Boris function
                    t_boris = time.time()
                    v_oldNode = self.toMatrix(species.vn[:,m])
                    species.ck_dm = self.toMatrix(species.ck_dm)
                    
                    ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
                    v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
                    species.vn[:,m+1] = self.toVector(v_new)
                    
                    ##########################################
                    
                    controller.runTimeDict['boris'] += time.time() - t_boris
                    controller.runTimeDict['gather'] += t_boris - t_gather
                    
                    self.calc_residuals(species,m,k)
                    
                    ### LORENTZ UPDATE FOR NODE m/SWEEP k ###
                    species.vel = v_new

                    species.lntz = species.a*(species.E + np.cross(species.vel,species.B))
                    species.Fn[:,m+1] = species.toVector(species.lntz)
                    
                    #########################################
                
                tFin = time.time()
                controller.runTimeDict['particle_push'] += tFin - tmid
                    
            for species in species_list:
                species.F[:,:] = species.Fn[:,:]
                species.x[:,:] = species.xn[:,:]
                species.v[:,:] = species.vn[:,:]
                
                self.calc_R(species,M,k)
                
                
                

        species_list = self.updateStep(species_list,fields,weights,Qmat)
        controller.runTimeDict['particle_push'] += time.time() - tFin
        
#        print(species_list[0].Rx)
#        print(species_list[0].Rv)
        return species_list
    
    
    def boris_SDC_2018(self, species_list,fields, controller,**kwargs):
        tst = time.time()
        
        M = self.M
        K = self.K

        #Remap collocation weights from [0,1] to [tn,tn+1]
        weights =  self.coll_params['weights']

        q =  self.coll_params['Qmat']
        Smat =  self.coll_params['Smat']

        dm =  self.coll_params['dm']

        SX =  self.coll_params['SX'] 

        SQ =  self.coll_params['SQ']

        for species in species_list:
            ## Populate node solutions with x0, v0, F0 ##
            species.x0[:,0] = self.toVector(species.pos)
            species.v0[:,0] = self.toVector(species.vel)
            species.F[:,0] = self.toVector(species.lntz)
            species.En_m0 = species.E
            species.Bn_m0 = species.B

            for m in range(1,M+1):
                species.x0[:,m] = species.x0[:,0]
                species.v0[:,m] = species.v0[:,0]
                species.F[:,m] = species.F[:,0]
            #############################################
            
            species.x = np.copy(species.x0)
            species.v = np.copy(species.v0)
            
            species.xn[:,:] = species.x[:,:]
            species.vn[:,:] = species.v[:,:]
            species.Fn[:,:] = species.F[:,:]
        
        controller.runTimeDict['particle_push'] += time.time() - tst
        
        #print()
        #print(simulationManager.ts)
        for k in range(1,K+1):
            #print("k = " + str(k))
            for species in species_list:
                species.En_m = species.En_m0 #reset electric field values for new sweep
                species.Bn_m = species.Bn_m0 #reset magnetic field values for new sweep
                
            for m in range(self.ssi,M):
                for species in species_list:
                    t_pos = time.time()
                    
                    #print("m = " + str(m))
                    #Determine next node (m+1) positions
                    
                    # Calculate collocation terms required for pos update
                    IV = 0
                    for j in range(1,M+1):
                        IV += (q[m+1,j]-q[m,j])*species.v[:,j]

                    ### POSITION UPDATE FOR NODE m/SWEEP k ###
                    species.xn[:,m+1] = species.xn[:,m]
                    species.xn[:,m+1] += dm[m]* (species.vn[:,m]-species.v[:,m])
                    species.xn[:,m+1] += dm[m]/2 * (species.Fn[:,m]-species.F[:,m])
                    species.xn[:,m+1] += IV
                    
                    ##########################################
                    
                    ### FIELD GATHER FOR m/k NODE m/SWEEP k ###
                    species.pos = np.reshape(species.xn[:,m+1],(species.nq,3))
#                    print(species.pos)
                    t_bc = time.time()
                    self.check_boundCross(species,fields,**kwargs)
                    
                    controller.runTimeDict['bound_cross_check'] += time.time() - t_bc
                    controller.runTimeDict['pos_push'] += t_bc - t_pos
                    
                controller.runTimeDict['particle_push'] += time.time() - t_pos
                self.run_fieldIntegrator(species_list,fields,controller)
                
                tmid = time.time()
                for species in species_list:
                    t_gather = time.time()
                    self.fieldGather(species,fields)
                    ###########################################
                    
                    #Sample the electric field at the half-step positions (yields form Nx3)
                    half_E = (species.En_m+species.E)/2
                    species.En_m = species.E              #Save m+1 value as next node's m value
                    species.Bn_m = species.B
                    
                    
                    t_boris = time.time()
                    # Calculate collocation terms required for pos update
                    IF = 0
                    for j in range(1,M+1):
                        IF += (q[m+1,j]-q[m,j])*species.F[:,j]
                        
                    c = -dm[m]/2 * np.cross(species.vn[:,m].reshape((species.nq,3)),
                                                                    species.Bn_m)
            
                    c += -dm[m]/2 * np.reshape(species.F[:,m]+species.F[:,m+1],
                                              (species.nq,3)) + IF.reshape((species.nq,3))
                            
                    c += dm[m]/2 * np.cross(species.vn[:,m].reshape((species.nq,3)),
                                                           species.B)
                    
                    #Resort all other 3d vectors to shape Nx3 for use in Boris function
                    v_oldNode = np.reshape(species.vn[:,m],(species.nq,3))
                    species.ck_dm = c
                    print(species.ck_dm)
                    
                    ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
                    v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
                    species.vn[:,m+1] = np.ravel(v_new)
                    
                    ##########################################
                    
                    controller.runTimeDict['boris'] += time.time() - t_boris
                    controller.runTimeDict['gather'] += t_boris - t_gather
                    
                    ### LORENTZ UPDATE FOR NODE m/SWEEP k ###
                    species.vel = v_new

                    species.lntz = species.a*(species.E + np.cross(species.vel,species.B))
                    species.Fn[:,m+1] = species.toVector(species.lntz)
                    
                    #########################################
                
                tFin = time.time()
                controller.runTimeDict['particle_push'] += tFin - tmid
                    
            for species in species_list:
                species.F[:,:] = species.Fn[:,:]
                species.x[:,:] = species.xn[:,:]
                species.v[:,:] = species.vn[:,:]
                
                self.calc_R(species,M,k)
                
        species_list = self.updateStep(species_list,fields,weights,q)
        controller.runTimeDict['particle_push'] += time.time() - tFin
        
#        print(species_list[0].Rx)
#        print(species_list[0].Rv)
        return species_list
    
    def fieldInterpolator(self,species_list,mesh,controller,m=1):
        mesh.E[2,:,:,:] = (1-self.nodes[m-1])*mesh.En0[2,:,:,:] + (self.nodes[m-1])*mesh.En1[2,:,:,:]
    
    
    def lobatto_update(self,species_list,mesh,*args,**kwargs):
        for species in species_list:
            pos = species.x[:,-1]
            vel = species.v[:,-1]
            
            species.pos = species.toMatrix(pos)
            species.vel = species.toMatrix(vel)
            self.check_boundCross(species,mesh,**kwargs)

        return species_list
    
    
    def legendre_update(self,species_list,mesh,weights,Qmat,**kwargs):
        for species in species_list:
            M = self.M
            d = 3*species.nq
            
            Id = np.identity(d)
            q = np.zeros(M+1,dtype=np.float)
            q[1:] = weights
            q = np.kron(q,Id)
            qQ = q @ np.kron(Qmat,Id)
            
            V0 = self.toVector(species.v0.transpose())
            F = self.FXV(species,mesh)
            
            vel = species.v0[:,0] + q @ F
            pos = species.x0[:,0] + q @ V0 + qQ @ F
            
            species.pos = species.toMatrix(pos)
            species.vel = species.toMatrix(vel)
            self.check_boundCross(species,mesh,**kwargs)
        return species_list
    
    
    def lorentzf(self,species,mesh,m,**kwargs):
        species.pos = species.toMatrix(species.x[:,m])
        species.vel = species.toMatrix(species.v[:,m])
        self.check_boundCross(species,mesh,**kwargs)

        self.fieldGather(species,mesh)

        F = species.a*(species.E + np.cross(species.vel,species.B))
        F = species.toVector(F)
        return F
    
    def lorentz_std(self,species,fields):
        F = species.a*(species.E + np.cross(species.vel,species.B))
        return F
    
    
    
    def FXV(self,species,fields):
        dxM = np.shape(species.x)
        d = dxM[0]
        M = dxM[1]-1
        
        F = np.zeros((d,M+1),dtype=np.float)
        for m in range(0,M+1):
            F[:,m] = self.lorentzf(species,fields,m)
        
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
    
    def periodic_particles_x(self,species,mesh,**kwargs):    
        self.periodic_particles(species,0,mesh.xlimits)
        
    def periodic_particles_y(self,species,mesh,**kwargs):    
        self.periodic_particles(species,1,mesh.ylimits)
        
    def periodic_particles_z(self,species,mesh,**kwargs):  
        self.periodic_particles(species,2,mesh.zlimits)
    
    def periodic_particles(self,species,axis,limits,**kwargs):
            undershoot = limits[0]-species.pos[:,axis]
            cross = np.argwhere(undershoot>0)
            species.pos[cross,axis] = limits[1] - undershoot[cross] % (limits[1]-limits[0])
    
            overshoot = species.pos[:,axis] - limits[1]
            cross = np.argwhere(overshoot>=0)
            species.pos[cross,axis] = limits[0] + overshoot[cross] % (limits[1]-limits[0])
        
        
    def simple_1d(self,species,mesh,controller):
        self.mi_z0 = 0
        FDMat = self.FDMat.toarray()
        
        FDMat[0,1] = 1/mesh.dz**2
        FDMat[-1,0] = 1/mesh.dz**2

        BC_vector = np.zeros(mesh.BC_vector.shape[0]+1,dtype=np.float)
        BC_vector[1:] = mesh.BC_vector
        mesh.BC_vector = BC_vector

        self.FDMat = sps.csc_matrix(FDMat)
        self.solver_post = self.mirrored_boundary_z
        

    def fixed_phi_1d(self,species,mesh,controller):
        self.mi_z0 = 0
        FDMat = self.FDMat.toarray()
        
        FDMat[0,0] = 1
        FDMat[0,1:] = 0
        FDMat[-1,0] = 1/mesh.dz**2

        BC_vector = np.zeros(mesh.BC_vector.shape[0]+1,dtype=np.float)
        BC_vector[1:] = mesh.BC_vector
        mesh.BC_vector = BC_vector
        
        
        self.FDMat = sps.csr_matrix(FDMat)
        
        self.rho_mod_i = [0]
        self.rho_mod_vals = [0]

        self.solver_pre = self.rho_mod_1d
        self.solver_post = self.mirrored_boundary_z
        

        
    def constant_phi_1d(self,species,mesh,controller):
        self.mi_z0 = 0
        FDMat = self.FDMat.toarray()
        
        FDMat[0,:] = 1
        FDMat[-1,0] = 1/mesh.dz**2
        
        BC_vector = np.zeros(mesh.BC_vector.shape[0]+1,dtype=np.float)
        BC_vector[1:] = mesh.BC_vector
        mesh.BC_vector = BC_vector
        
        self.FDMat = sps.csr_matrix(FDMat)
        
        self.rho_mod_i = [0]
        self.rho_mod_vals = [0]

        self.solver_pre = self.rho_mod_1d
        self.solver_post = self.mirrored_boundary_z
        
        
    def integral_phi_1d(self,species,mesh,controller):
        self.mi_z0 = 0
        self.mi_zN = -1
        self.interior_shape[2] += 1
        
        FDMat = self.FDMat.toarray()
        
        FDMat[0,-1] = 1/mesh.dz**2
        FDMat[-1,0] = 1/mesh.dz**2
        
        N = FDMat.shape[0]+1
        FDMat_exp = np.zeros((N,N),dtype=np.float)
        FDMat_exp[:-1,:-1] = FDMat
        FDMat_exp[-1,:-1] = mesh.dz 
        FDMat_exp[:-1,-1] = 1.
        
        BC_vector = np.zeros(mesh.BC_vector.shape[0]+2,dtype=np.float)
        BC_vector[1:-1] = mesh.BC_vector
        mesh.BC_vector = BC_vector
        
        self.rho_mod_i = [-2]
        self.rho_mod_vals = [0]
        self.solver_pre = self.rho_mod_1d
        
        self.FDMat = sps.csr_matrix(FDMat_exp)
        self.solver_post = self.mirrored_boundary_z

    def scatter_periodicBC_1d(self,species,mesh,controller):
        mesh.q[1,1,0] += (mesh.q[1,1,-2]-mesh.q_bk[1,1,-2])
        mesh.q[1,1,-2] = mesh.q[1,1,0] 
        
        
    def rho_mod_1d(self,species,mesh,controller):
        j = 0
        for index in self.rho_mod_i:
            mesh.rho[1,1,index] = self.rho_mod_vals[j]
            j += 1
            
        return mesh.rho
        
    def mirrored_boundary_z(self,species,mesh,controller):
        mesh.phi[:,:,-2] = mesh.phi[:,:,0]
        mesh.rho[:,:,-2] = mesh.rho[:,:,0]
        mesh.q[:,:,-2] = mesh.q[:,:,0]
        mesh.E[:,:,:,-2] = mesh.E[:,:,:,0]
        mesh.B[:,:,:,-2] = mesh.B[:,:,:,0]
        
    def half_volume_BC_z(self,species,mesh,controller):
        mesh.q[:,:,0] = mesh.q[:,:,0]*2
        mesh.q[:,:,-2] = mesh.q[:,:,-2]*2
        mesh.rho[:,:,0] = mesh.rho[:,:,0]*2
        mesh.rho[:,:,-2] = mesh.rho[:,:,-2]*2
        
        
################################ Hook methods #################################
    def ES_vel_rewind(self,species_list,mesh,controller=None):
        dt = controller.dt
        for species in species_list:
            self.fieldGather(species,mesh)
            species.vel = species.vel - species.E * species.a * dt/2
 
        
    def calc_residuals_avg(self,species,m,k):
        s = species
        s.x_con[k-1,m] = np.average(np.abs(s.xn[:,m+1] - s.x[:,m+1]))
        s.x_res[k-1,m] = np.average(np.linalg.norm(s.xn[:,m+1]-s.xQuad))
        
        s.v_res[k-1,m] = np.average(np.linalg.norm(s.vn[:,m+1]-s.vQuad))
        s.v_con[k-1,m] = np.average(np.abs(s.vn[:,m+1] - s.v[:,m+1]))
        
    def calc_residuals_max(self,species,m,k):
        s = species
        s.x_con[k-1,m] = np.max(np.abs(s.xn[:,m+1] - s.x[:,m+1]))
        s.x_res[k-1,m] = np.max(np.linalg.norm(s.xn[:,m+1]-s.xQuad))
        
        s.v_res[k-1,m] = np.max(np.linalg.norm(s.vn[:,m+1]-s.vQuad))
        s.v_con[k-1,m] = np.max(np.abs(s.vn[:,m+1] - s.v[:,m+1]))
        
    def calc_residual(self,species,M,k):
        s = species
        d = s.nq*3
        
        u0 = self.get_u(s.x[:,0],s.v[:,0])
        
        for m in range(0,M+1):
            u = self.get_u(s.x[:,m],s.v[:,m])
            s.U0[2*d*m:2*d*(m+1)] = u0
            s.Uk[2*d*m:2*d*(m+1)] = u
            s.FXV[d*m:d*(m+1)] = s.F[:,m,np.newaxis]
            
        print((s.Qc @ s.FXV).shape)
        
    def calc_residual2(self,species,M,k):
        s = species
        q =  self.coll_params['Qmat']

        for m in range(1,M+1):
            for j in range(1,m+1):
                qvsum = q[m,j] * s.v[:,j]
                qfsum = q[m,j] * s.F[:,j] 
                
            s.Rx[k-1,m-1] = np.max(np.linalg.norm(s.x[:,0] + qvsum - s.x[:,m]))
            s.Rx[k-1,m-1] = np.max(np.linalg.norm(s.v[:,0] + qfsum - s.v[:,m]))
        
        
    
    def display_convergence(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position convergence, " + str(species.name) + ":")
            print(species.x_con)
            
            print("Velocity convergence, " + str(species.name) + ":")  
            print(species.v_con)
        
        
    def display_residuals_full(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position residual, " + str(species.name) + ":")
            print(species.x_res)
            
            print("Velocity residual, " + str(species.name) + ":")
            print(species.v_res)
            
    def display_residuals_max(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position residual, " + str(species.name) + ":")
            print(np.max(species.x_res,1))
            
            print("Velocity residual, " + str(species.name) + ":")
            print(np.max(species.v_res,1))
        
        
    def get_u(self,x,v):
        assert len(x) == len(v)
        d = len(x)
        
        Ix = np.array([1,0])
        Iv = np.array([0,1])
        Id = np.identity(d)
        
        u = np.kron(Id,Ix).transpose() @ x + np.kron(Id,Iv).transpose() @ v
        return u[:,np.newaxis]
    
    
    def energy_calc_penning(self,species_list,fields,**kwargs):
        for species in species_list:
            x = self.toVector(species.pos)
            v = self.toVector(species.vel)
            u = self.get_u(x,v)
            
            species.energy = u.transpose() @ self.H @ u
        
        return species_list
    
    def kinetic_energy(self,species_list,fields,**kwargs):
        for species in species_list:
            species.KE = 0.5 * species.mq * np.linalg.norm(species.vel,ord=2,axis=1)
            species.KE_sum = np.sum(species.KE)
            
        return species
            
    def field_energy(self,species_list,fields,**kwargs):
        fields.PE = fields.q*fields.phi
        fields.PE_sum = np.sum(fields.PE[:-2,:-2,:-2])
        
        return fields
        
    def centreMass(self,species_list,fields,**kwargs):
        for species in species_list:
            nq = np.float(species.nq)
            mq = np.float(species.mq)
    
            species.cm[0] = np.sum(species.pos[:,0]*mq)/(nq*mq)
            species.cm[1] = np.sum(species.pos[:,1]*mq)/(nq*mq)
            species.cm[2] = np.sum(species.pos[:,2]*mq)/(nq*mq)
            
        return species_list
        

    def rhs_tally(self,species_list,fields,controller,**kwargs):
        try:
            rhs_eval = self.rhs_dt * controller.tSteps
            controller.rhs_eval = rhs_eval
        except:
            print('Could not retrieve controller, rhs eval set to zero.')
            rhs_eval = 0
            
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
