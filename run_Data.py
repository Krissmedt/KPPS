from kpps import kpps
from dataHandler2 import dataHandler2 as dh
from math import sqrt, fsum, pi, exp, cos, sin, floor
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sim_folder = 'simple_penning'


data_params['time_plot_vars'] = ['pos']
data_params['trajectories'] = [1]
data_params['domain_limits'] = [20,20,15]


plot_params = {}
plot_params['legend.fontsize'] = 12
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 10
data_params['plot_params'] = plot_params

dho = dh()
dho.load_sim(sim_name=sim_folder,overwrite=True)

print(dho.controller_obj.simID)
