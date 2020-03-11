#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:44:36 2020

@author: mn12kms
"""


    def boris_SDC_dirty(self, species_list,fields, controller,**kwargs):
        tst = time.time()

        M = self.M
        K = self.K
        
        weights =  self.coll_params['weights']

        Qmat =  self.coll_params['Qmat']
        Smat =  self.coll_params['Smat']

        dm =  self.coll_params['dm']

        SX =  self.coll_params['SX'] 

        SQ =  self.coll_params['SQ']
        
        fields.En0 = np.copy(fields.E)
        
        for species in species_list:
            ## Populate node solutions with x0, v0, F0 ##
            species.x0[:,0] = self.toVector(species.pos)
            species.v0[:,0] = self.toVector(species.vel)
            species.F[:,0] = self.toVector(species.lntz)
            species.En_m0 = species.E

        self.boris_synced(species_list,fields,controller,**kwargs)
        
        fields.En1 = np.copy(fields.E)
        
        for species in species_list:
            ## Populate node solutions with x0, v0, F0 ##
            species.x0[:,-1] = self.toVector(species.pos)
            species.v0[:,-1] = self.toVector(species.vel)
            species.F[:,-1] = self.toVector(species.lntz)

            for m in range(1,M):
                species.x0[:,m] = (1-self.nodes[m-1])*species.x0[:,0] + (self.nodes[m-1])*species.x0[:,-1]
                species.v0[:,m] = (1-self.nodes[m-1])*species.v0[:,0] + (self.nodes[m-1])*species.v0[:,-1]
                species.F[:,m] = (1-self.nodes[m-1])*species.F[:,0] + (self.nodes[m-1])*species.F[:,-1]
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
                    
                    ### FIELD GATHER FOR m/k NODE m/SWEEP k ###
                    species.pos = self.toMatrix(species.xn[:,m+1],3)
                    
                    t_bc = time.time()
                    self.check_boundCross(species,fields,**kwargs)
                    
                    controller.runTimeDict['bound_cross_check'] += time.time() - t_bc
                    controller.runTimeDict['pos_push'] += t_bc - t_pos
                    
                controller.runTimeDict['particle_push'] += time.time() - t_pos
                
                self.fieldInterpolator(species_list,fields,controller,m=m)
                
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

        species_list = self.updateStep(species_list,fields,weights,Qmat)
        controller.runTimeDict['particle_push'] += time.time() - tFin
        
        return species_list