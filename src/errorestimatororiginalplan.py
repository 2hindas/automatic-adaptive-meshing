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
import SimPEG
from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import numdifftools as nd

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

#Surface Definition
domain= ((-500, 4500), (-1000, 1000), (-1200, 200))
xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-1000, 1000, 101))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)] #Surface

src = [0, 0, 0, 0, 0]        # x-dir. source at the origin, 50 m above seafloor
off = np.arange(20, 41)*100  # Offsets
rec = [off, off*0, 0]        # In-line receivers on the seafloor

# Defining transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
print("Number of transmitters",len(source_locations))

# Define receiver locations
N = 21
xrx, yrx, zrx = np.meshgrid(np.linspace(2000, 4000, N), [0], [0])
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

# minimum cell width in each dimension
dx = 50 #As calculated by 10% of the minimum skin depth
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
#rotation = R.from_euler('x', 10, degrees=True).as_matrix()


# Define inner points for rectangular box
x = np.linspace(200, 600, 30)
y = np.linspace(-500, 500, 30)
z = np.linspace(-1000, -800, 30)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
)
# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, source_locations, octree_levels=[2,4], method="radial", finalize=False
)

mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# Mesh refinement at block location
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False
)


mesh.finalize()
# The total number of cells
nC = mesh.nC
print("Total number of cells", nC)

Cellfaces = mesh.n_faces
print("Total number of cell faces", Cellfaces)

Celledges = mesh.n_edges
print("Total number of cell edges", Celledges)


# Resistivity in Ohm m
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
    (mesh.gridCC[ind_active, 0] <= 600.0)
    & (mesh.gridCC[ind_active, 0] >= 200.0)
    & (mesh.gridCC[ind_active, 1] <= 500.0)
    & (mesh.gridCC[ind_active, 1] >= -500.0)
    & (mesh.gridCC[ind_active, 2] <= -800.0)
    & (mesh.gridCC[ind_active, 2] >= -1000.0)
)
model[ind_block] = res_block
'''
# Plot cell volumes
v = mesh.cell_volumes
mesh.plot_slice(np.log10(v),ind = 8,grid=True)
plt.show()
'''
#-----------------------------------------------------------------------------
#This part of the code is relevant as error estimator indicator
xyzcells = mesh.cell_centers #Cell-centers

xedges = mesh.edges_x # x-edges
yedges = mesh.edges_y # y-edges
zedges = mesh.edges_z # z-edges

xfaces = mesh.faces_x # x-faces
yfaces = mesh.faces_y # y-faces
zfaces = mesh.faces_z # z-faces
#Solution
simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, rhoMap=model_map, Solver=Solver
)

simulationelectricfield = fdem.simulation.Simulation3DElectricField(
    mesh, survey=survey, rhoMap=model_map, Solver=Solver
)
#Compute magnetic flux density
fields = simulation.fields(model)
MagFluxDensity = fields[:, 'bSolution'] #Field of the magnetic flux density


#Source field
sources = simulation.getSourceTerm(frequencies[0])
Sm = sources[0] 


#Curl of Electric field computed on the cell faces
Ce = Sm - 1j*omegas[0]*MagFluxDensity #Curl electric field
Ce = np.reshape(Ce,len(Ce))
CeX = Ce[0:(mesh.n_faces_x)]
CeY = Ce[(mesh.n_faces_x):(mesh.n_faces_x)+(mesh.n_faces_y)]
CeZ = Ce[(mesh.n_faces_x)+(mesh.n_faces_y):(mesh.n_faces_x)+(mesh.n_faces_y)+(mesh.n_faces_z)]


#Interpolate curl using radial basis interpolation 'mutiquadric'
InterpolatedCeX = sp.interpolate.Rbf(xfaces[:,0],xfaces[:,1],xfaces[:,2],CeX)
InterpolatedCeY = sp.interpolate.Rbf(yfaces[:,0],yfaces[:,1],yfaces[:,2],CeY)
InterpolatedCeZ = sp.interpolate.Rbf(zfaces[:,0],zfaces[:,1],zfaces[:,2],CeZ)


#Electric field solution
fieldselectric = simulationelectricfield.fields(model)
Electricfield = fieldselectric[:,'eSolution']
Electricfield = np.reshape(Electricfield,len(Electricfield))
Ex = Electricfield[0:(mesh.n_edges_x)]
Ey = Electricfield[(mesh.n_edges_x):(mesh.n_edges_x)+(mesh.n_edges_y)]
Ez = Electricfield[(mesh.n_edges_x)+(mesh.n_edges_y):(mesh.n_edges_x)+(mesh.n_edges_y)+(mesh.n_edges_z)]


#Interpolate Electric field using radial basis interpolation 'mutiquadric'
InterpolatedEx = sp.interpolate.Rbf(xedges[:,0],xedges[:,1],xedges[:,2],Ex)
InterpolatedEy = sp.interpolate.Rbf(yedges[:,0],yedges[:,1],yedges[:,2],Ey)
InterpolatedEz = sp.interpolate.Rbf(zedges[:,0],zedges[:,1],zedges[:,2],Ez)



#Operators and functions used for integrating
# x is in these functions a vector x = (x,y,z)
#Curl operator
def Curl(f,x):
    jac = nd.Jacobian(f)(x)
    return np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])
#Interpolated Electric field
def Efield(x):
    return np.array([InterpolatedEx(x[0],x[1],x[2]),InterpolatedEy(x[0],x[1],x[2]),InterpolatedEz(x[0],x[1],x[2])])
#Interpolated curl of the Electric field
def Curlfield(x):
    return np.array([InterpolatedCeX(x[0],x[1],x[2]),InterpolatedCeY(x[0],x[1],x[2]),InterpolatedCeZ(x[0],x[1],x[2])])
#Computes error between interpolated curl and the curl of the interpolated electric field
def Errorcomputer(x):
    return np.linalg.norm(Curlfield(x)-Curl(Efield,x))
#Define Search area
searcharea = xyzcells[(xyzcells[:,0]>(0-dx)) & (xyzcells[:,0]<(4000+dx)) 
                      & (xyzcells[:,1]>(-500-dy)) & (xyzcells[:,1]<(500+dy)) 
                      & (xyzcells[:,2]>(-1000-dz)) & (xyzcells[:,2]<(-800+dz)) ]
#------------------------------------------------------------------------------
#Code to be optimized
errorcellcenters = []
for i in searcharea:
    errorcellcenters.append(Errorcomputer(i))
#------------------------------------------------------------------------------
np.save('error.npy',np.array(errorcellcenters))
#hoetegebruiken = np.load('curl1.npy')
#Compute error in every cell center
percentage = 0.05 #Percentage of the grid you want to refine
Ncellstorefine = int(np.ceil(percentage*len(searcharea)))
#Cells required to refine (5% of the domain with the largest errors)
cellstorefine = searcharea[np.argpartition(np.load('error.npy'),-Ncellstorefine)[-Ncellstorefine:]]
print('Error estimator is finished')

#-----------------------------------------------------------------------------
#plot cells to refine
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cellstorefine[:,0], cellstorefine[:,1], cellstorefine[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
#-----------------------------------------------------------------------------
















