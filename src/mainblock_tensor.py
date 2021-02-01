#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:47:48 2021

@author: larslaheij
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from discretize.utils import mkvc
import SimPEG.electromagnetics.frequency_domain as fdem
import numpy as np
import Meshing as M
import Utils as utils
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from discretize import TensorMesh
from LapentaEstimator import interpolate_linear
import time
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


domain = ((-500, 4500), (-1000, 1000), (-1200, 200))
cell_width = 300

xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-1000, 1000, 101))
zz = np.zeros(np.shape(xx))
surface = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

box_coordinates = ((200, 600), (-500, 500), (-1000, -800))

# Defining transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
print("Number of transmitters", len(source_locations))
#wider area around transmitter
#xa, ya, za = np.meshgrid(np.linspace(-200,200,101), np.linspace(-200,200,101), np.linspace(-200,200,101))
#wider_area = np.c_[mkvc(xa), mkvc(ya), mkvc(za)]

# Define receiver locations
N = 21
xrx, yrx, zrx = np.meshgrid(np.linspace(2000, 4000, N), [0], [0])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
frequencies = [1.0]  # Frequency (Hz)
omegas = [2.0 * np.pi]  # Radial frequency (Hz)
print("Number of receivers", len(receiver_locations))

survey = utils.define_survey(frequencies, receiver_locations, source_locations, ntx)

# domain dimensions
x_length = np.abs(domain[0][0] - domain[0][1])
y_length = np.abs(domain[1][0] - domain[1][1])
z_length = np.abs(domain[2][0] - domain[2][1])

# number of cells needed in each dimension
nbx = 2 ** int(np.ceil(np.log(x_length / cell_width) / np.log(2.0)))
nby = 2 ** int(np.ceil(np.log(y_length / cell_width) / np.log(2.0)))
nbz = 2 ** int(np.ceil(np.log(z_length / cell_width) / np.log(2.0)))

# define base mesh 
hx = cell_width * np.ones(nbx)
hy = cell_width * np.ones(nby)
hz = cell_width * np.ones(nbz)
mesh = TensorMesh([hx, hy, hz], origin=[domain[0][0], domain[1][0], domain[2][0]])



# for i in range(0, 60):
#     M.plot_mesh_slice(mesh, 'z', i, save=True)
# exit()
#
print(mesh)

print("Total number of cells", mesh.nC)
print("Total number of cell faces", mesh.n_faces)
print("Total number of cell edges", mesh.n_edges)

# Resistivity in Ohm m
res_background = 1.0
res_block = 100.0

# Conductivity in S/m
conductivity_background = 1 / 100.0
conductivity_block = 1 / 1.0

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, surface)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())

ind_block = utils.get_ind_block(mesh, ind_active, box_coordinates)
model[ind_block] = res_block

x_edges = mesh.edges_x
y_edges = mesh.edges_y
z_edges = mesh.edges_z


#Solution 
simulationelectricfield = fdem.simulation.Simulation3DElectricField(
            mesh, survey=survey, rhoMap=model_map, Solver=Solver
        )
# Electric field solution
timeStart = time.perf_counter() 
fieldselectric = simulationelectricfield.fields(model)
EF = fieldselectric[:, 'eSolution']
timeend = time.perf_counter()
print('time is',timeend-timeStart) 
EF = np.reshape(EF, len(EF))

EF_x = EF[0:mesh.n_edges_x]
EF_y = EF[mesh.n_edges_x:mesh.n_edges_x + mesh.n_edges_y]
EF_z = EF[
       mesh.n_edges_x + mesh.n_edges_y:mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z]
'''
EF_x_inter, EF_y_inter, EF_z_inter = interpolate_linear(x_edges, y_edges, z_edges, EF_x, EF_y, EF_z)

xoctree = np.load('xlst.npy')
yoctree = np.load('ylst.npy')
zoctree = np.load('zlst.npy')

EF_x_ten = EF_x_inter(xoctree,yoctree,zoctree)
EF_y_ten = EF_y_inter(xoctree,yoctree,zoctree)
EF_z_ten = EF_z_inter(xoctree,yoctree,zoctree)

np.save('EF_x_ten.npy',EF_x_ten)
np.save('EF_y_ten.npy',EF_y_ten)
np.save('EF_z_ten.npy',EF_z_ten)
'''

