from fields import fields
import numpy as np

class fieldTester:
    def __init__(self):
        omegaB = 25.0
        omegaE = 4.9
        epsilon = -1
        
        alpha = 1
        
        eMag = -epsilon*omegaE**2/alpha
        eTransform = np.array([[1,0,0],[0,1,0],[0,0,-2]])
        
        self.imposedEParams = {'general':eTransform, 
                               #'sPenning':[1,0,0],
                               'magnitude':eMag}
    
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


settings = {'box':{'xlim':[-1,1],'ylim':[-1,1],'zlim':[-1,1]},
         'resolution':[10]}

f = fields(**settings)
ft = fieldTester()




f = ft.initialise_field_mesh(f)
print(f.E[:,:,:,5])

