from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def type_setup(exptype,analysis_params):
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
        
    elif exptype == 4:
        analysis_params['field_type'] = 'custom'
        analysis_params['gather'] = 'trilinear_gather'
        analysis_params['external_fields'] = False
        analysis_params['external_fields_mesh'] = False
        analysis_params['scatter_BC'] = 'half_volume_BC_z'
        analysis_params['preAnalysis_methods'] = [type4_pre,'trilinear_qScatter','fIntegrator_setup','fIntegrator',update_background,type4_post]
        
    return analysis_params
        

def type_setup_spec(exptype,res,ppc,spec1_params,loader1_params,spec2_params,loader2_params):
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
        
        nq = res*ppc
        
        y_range = np.linspace(0,-1,num=np.int(nq/2))
        x_range2 = np.power(-y_range,1/3)
        x_range1 = np.flip(-x_range2,0)
        x_range = np.append(x_range1[:-1],x_range2)
        
        nq = x_range.shape[0]
        spec2_params['nq'] = nq
        spec2_params['q'] = -2/nq
        loader2_params['pos'] = np.zeros((nq,3),dtype=np.float)
        loader2_params['vel'] = np.zeros((nq,3),dtype=np.float)

        loader2_params['load_type'] = 'direct'
        loader2_params['speciestoLoad'] = [1]
        loader2_params['pos'][:,2] = x_range
        
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
    mesh.q_bk = mesh.q
    mesh.rho_bk = mesh.rho
    mesh.E_bk = mesh.E    
    

def type4_pre(species_list,mesh,controller=None):
    species_list[0].q = 0
    
    
def type4_post(species_list,mesh,controller=None):
    species_list[0].q = 1
    del species_list[1]

def bc_pot(pos):
    phi = 0
    #phi = pos[2]**2
    return phi

def ion_bck(species_list,mesh,controller=None,rho_bk=None):
    threshold = 1e-10
    rho_bk[1,1,:-1] += mesh.node_charge/mesh.dz
    rho_bk[np.abs(rho_bk) < threshold] = 0
    
    mesh.rho_bk = rho_bk
    return rho_bk

