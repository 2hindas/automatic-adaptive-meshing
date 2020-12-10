#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:19:30 2020

@author: larslaheij
"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R
from SimPEG.utils import plot2Ddata, surface2ind_topo
import SimPEG
from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
    
xx, yy = np.meshgrid(np.linspace(0, 5, 11), np.linspace(0, 5, 11))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)] #Surface
    
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
print("Number of transmitters",len(source_locations))

# Define receiver locations
xrx, yrx, zrx = np.meshgrid([2], [0], [0])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
frequencies = [1.0]                   # Frequency (Hz)
omegas = [2.0*np.pi]                  # Radial frequency (Hz)
source_list = []  # Create empty list to store sources
print("Number of receivers",len(receiver_locations))

# Each unique location and frequency defines a new transmitter
for ii in range(len(frequencies)):
    for jj in range(ntx):

        # Define receivers of different type at each location
        bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[jj, :], "z", "real"
        )
        bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[jj, :], "z", "imag"
        )
        receivers_list = [bzr_receiver, bzi_receiver]

        # Must define the transmitter properties and associated receivers
        source_list.append(
            fdem.sources.MagDipole(
                receivers_list,
                frequencies[ii],
                source_locations[jj],
                orientation="z",
                moment=1,
            )
        )
survey = fdem.Survey(source_list)
    
domain= ((0, 5), (0, 5), (0, 5))
dx = 2
dy = 2
dz = 1

# domain dimensions
x_length = np.abs(domain[0][0] - domain[0][1]) 
y_length = np.abs(domain[1][0] - domain[1][1]) 
z_length = np.abs(domain[2][0] - domain[2][1]) 

# number of cells necessary in each dimension
nbx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
nby = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
nbz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))

# Define base mesh (domain and finest discretization)
hx = [(dx,nbx)]
hy = [(dy,nby)]
hz = [(dz,nbz)]
mesh = TensorMesh([hx, hy, hz], origin=[0,0,0])

mesh.plot_grid(edges=True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Resistivity in Ohm m
res_background = 1.0
# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())

simulationelectricfield = fdem.simulation.Simulation3DElectricField(
    mesh, survey=survey, rhoMap=model_map, Solver=Solver
)
#Electric field solution
fieldselectric = simulationelectricfield.fields(model)
Electricfield = fieldselectric[:,'eSolution']
