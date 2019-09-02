from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def type_setup(exptype,species_params, loader_params,analysis_params):
    if exptype == 1:
        analysis_params['fieldIntegration'] = False
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'none'
        analysis_params['external_fields'] = True
        analysis_params['external_fields_mesh'] = False
        
    elif exptype == 2:
        analysis_params['fieldIntegration'] = True
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'trilinear_gather'
        analysis_params['external_fields'] = False
        analysis_params['external_fields_mesh'] = True
        
    elif exptype == 3:
        analysis_params['custom_rho_background'] = nonLinear_ion_bck
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'trilinear_gather'
        analysis_params['external_fields'] = False
        analysis_params['external_fields_mesh'] = False
        analysis_params['preAnalysis_methods'] = ['calc_background','fIntegrator_setup','impose_background','fIntegrator',update_background]
        
    elif expType == 4:
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'trilinear_gather'
        analysis_params['external_fields'] = False
        analysis_params['external_fields_mesh'] = False
        analysis_params['preAnalysis_methods'] = ['trilinear_qScatter','fIntegrator_setup','impose_background','fIntegrator']
        
        
    return analysis_params
        

def type_setup_spec(exptype,res,spec1_params,loader1_params,spec2_params,loader2_params):
    if exptype == 1:
        species_params = [spec1_params]
        loader_params = [loader1_params]
        
    elif exptype == 2:
        species_params = [spec1_params]
        loader_params = [loader1_params]
        
    elif exptype == 3:
        species_params = [spec1_params]
        loader_params = [loader1_params]
        
    elif exptype == 4:  
    spec2_params['name'] = 'spec2'
    spec2_params['nq'] = res*20
    spec2_params['q'] = -1
    spec2_params['mq'] = 1
    
    
    loader1_params['load_type'] = 'direct'
    loader1_params['speciestoLoad'] = [1]
    loader1_params['pos'] = np.array([[0,0,0.5],[0,0,0.75]])
    loader1_params['vel'] = np.array([[0,0,0],[0,0,0]])
        
        species_params = [spec1_params,spec2_params]
        loader_params = [loader1_params,loader2_params]
    
    return species_params, loader_params
        
def nonLinear_ext_E(species,mesh,controller=None):
    nq = species.pos.shape[0]
    for pii in range(0,nq):
        species.E[pii,2] += -np.power(species.pos[pii,2],3)
    
    return species


def nonLinear_mesh_E(species_list,mesh,controller=None):
    for zi in range(0,mesh.E[2,1,1,:].shape[0]-1):
        z = mesh.zlimits[0] + zi * mesh.dz
        mesh.E[2,1,1,zi] += -np.power(z,3)

    static_E = np.zeros(mesh.E.shape)
    static_E[:] = mesh.E[:]

    return mesh, static_E

def nonLinear_ion_bck(species_list,mesh,controller=None,rho_bk=None):
    for zi in range(0,mesh.rho.shape[2]-1):
        z = mesh.zlimits[0] + mesh.dz * zi
        rho_bk[1,1,zi] = - 3*np.power(z,2)
    
    mesh.rho_bk = rho_bk
    return rho_bk

def quartic_potential(pos):
    phi = 1/4 * np.power(pos[2],4)

    return phi

def update_background(species_list,mesh,controller=None):
    mesh.E_bk = mesh.E    


def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species_list,mesh,controller):
    threshold = 1e-10
    mesh.rho[1,1,:-1] += mesh.node_charge/mesh.dz
    mesh.rho[np.abs(mesh.rho) < threshold] = 0

