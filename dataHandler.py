## Dependencies ##
import vtk_writer as vtk_writer
import os

## Class ##
class dataHandler:
    sampleEvery = 1
    dataFoldername = "./"
    writer = vtk_writer.VTK_XML_Serial_Unstructured()
    
    def __init__(self, species, simulationManager, **kwargs):
        if 'foldername' in kwargs:
            if kwargs['foldername'] == "simple":
                delimiter = "_"
                
                entries = ['kpps',
                           str(simulationManager.ndim) + 'D',
                           str(species.nq) + 'p',
                           str(simulationManager.tSteps) + 'k']
                
                self.dataFoldername += delimiter.join(entries)
        
        self.mkDataDir()
        
        if 'sampleRate' in kwargs:
            self.sampleEvery = kwargs['sampleRate']
        
            
    def mkDataDir(self):
        k = 1
        error = True
        append = ""
        while error == True:
            try:
                os.mkdir(self.dataFoldername + append)
                error = False
            except FileExistsError:
                append = "(" + str(k) + ")"
                error = True
                k = k+1
                
        self.dataFoldername = self.dataFoldername + append
    
    
    def writeData(self,species,simulationManager):
        ts = simulationManager.ts
        pos = species.pos
        
        if ts % self.sampleEvery:
            filename = self.dataFoldername + "/" + str(ts) + ".vtu"
            self.writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
        
        if ts == simulationManager.tSteps:
            filename = self.dataFoldername + "/" + str(ts) + ".vtu"
            self.writer.snapshot(filename,pos[:,0],pos[:,1],pos[:,2])
            
            filename = self.dataFoldername + "/" + self.dataFoldername[2:]
            self.writer.writePVD(filename + ".pvd")