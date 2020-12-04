#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:42 2020

@author: larslaheij
"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

domain= ((-500, 4500), (-1000, 1000), (-1200, 200))
xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-1000, 1000, 101))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)] #Surface

src = [0, 0, 0, 0, 0]        # x-dir. source at the origin, 50 m above seafloor
off = np.arange(20, 41)*100  # Offsets
rec = [off, off*0, 0]        # In-line receivers on the seafloor

# Defining transmitter locations
N = 21
xtx, ytx, ztx = np.meshgrid(np.linspace(2000, 4000, N), [0], [10])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
#print(source_locations)

# Define receiver locations
# Define receiver locations
xrx, yrx, zrx = np.meshgrid(np.linspace(2000, 4000, N), [0], [0])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
frequencies = [1.0]                   # Frequency (Hz)
source_list = []  # Create empty list to store sources
print(len(receiver_locations))

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

# minimum cell width in each dimension
dx = 50
dy = 50
dz = 50

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
mesh = TreeMesh([hx, hy, hz], origin=[-500,-1000,-1200])

# Define rotation matrix
# 10 degrees rotation around the x-axis
rotation = R.from_euler('x', 10, degrees=True).as_matrix()

# Define inner points for rectangular box
x = np.linspace(0, 4000, 30)
y = np.linspace(-500, 500, 30)
z = np.linspace(-1200, -1000, 30)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
)
# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
)
# Mesh refinement at block location
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)

mesh.finalize()
# The total number of cells
nC = mesh.nC
print(nC)

# Resistivity in Ohm m)
res_background = 1.0
res_block = 100.0
conductivity_background = 1/100.0
conductivity_block = 1/1.0

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())
ind_block = (
    (mesh.gridCC[ind_active, 0] <= 4000.0)
    & (mesh.gridCC[ind_active, 0] >= 0.0)
    & (mesh.gridCC[ind_active, 1] <= 500.0)
    & (mesh.gridCC[ind_active, 1] >= -500.0)
    & (mesh.gridCC[ind_active, 2] <= -1000.0)
    & (mesh.gridCC[ind_active, 2] >= -1200.0)
)
model[ind_block] = res_block

simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, rhoMap=model_map, Solver=Solver
)
# Compute predicted data for a your model.)
