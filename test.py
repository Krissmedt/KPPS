import io
import pickle as pk
import numpy as np
import time
import cmath as math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from mesh import mesh

k = 3
res = 15
interpol_nodes = np.zeros((res,k+1),dtype=np.int)

for i in range(0,res):
    min_j = i - np.floor(k/2)
    max_j = i + np.floor((k+1)/2)
    
    if max_j > res:
        max_j -= res
        
    interpol_nodes[i,:] = np.linspace(min_j,max_j,k+1)
    
