#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:37:50 2020

@author: larslaheij
"""
#-----------------------------------------------------------------------------
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
import matplotlib as mpl
import scipy as sp
import numdifftools as nd

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

#-----------------------------------------------------------------------------
#This part of the code is relevant as error estimator indicator
xyzcells = mesh.cell_centers #Cell-centers

xedges = mesh.edges_x # x-edges
yedges = mesh.edges_y # y-edges
zedges = mesh.edges_z # z-edges

xfaces = mesh.faces_x # x-faces
yfaces = mesh.faces_y # y-faces
zfaces = mesh.faces_z # z-faces

#Solution by forward modelling for magnetic flux density and electric field
#This uses a rhoMap (resistivity)
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
#Curl: Ce = Sm - i*omega*b
Ce = np.reshape(Ce,len(Ce))
CeX = Ce[0:(mesh.n_faces_x)]
CeY = Ce[(mesh.n_faces_x):(mesh.n_faces_x)+(mesh.n_faces_y)]
CeZ = Ce[(mesh.n_faces_x)+(mesh.n_faces_y):(mesh.n_faces_x)+(mesh.n_faces_y)+(mesh.n_faces_z)]


#Interpolate curl using radial basis interpolation 'mutiquadric'
InterpolatedCeX = sp.interpolate.Rbf(xfaces[:,0],xfaces[:,1],xfaces[:,2],CeX)
InterpolatedCeY = sp.interpolate.Rbf(yfaces[:,0],yfaces[:,1],yfaces[:,2],CeY)
InterpolatedCeZ = sp.interpolate.Rbf(zfaces[:,0],zfaces[:,1],zfaces[:,2],CeZ)

'''
#Interpolate curl using nearest-neigbour interpolation
InterpolatedCeX = sp.interpolate.NearestNDInterpolator(xfaces,CeX)
InterpolatedCeY = sp.interpolate.NearestNDInterpolator(yfaces,CeY)
InterpolatedCeZ = sp.interpolate.NearestNDInterpolator(zfaces,CeZ)

'''
'''
#Interpolate curl using Linear interpolation
InterpolatedCeX = sp.interpolate.LinearNDInterpolator(xfaces,CeX)
InterpolatedCeY = sp.interpolate.LinearNDInterpolator(yfaces,CeY)
InterpolatedCeZ = sp.interpolate.LinearNDInterpolator(zfaces,CeZ)

'''


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

'''
#Interpolate electric field using nearest-neigbour interpolation
InterpolatedEx = sp.interpolate.NearestNDInterpolator(xedges,Ex)
InterpolatedEy = sp.interpolate.NearestNDInterpolator(yedges,Ey)
InterpolatedEz = sp.interpolate.NearestNDInterpolator(zedges,Ez)
'''
'''
#Interpolate electric field using Linear interpolation
InterpolatedEx = sp.interpolate.LinearNDInterpolator(xedges,Ex)
InterpolatedEy = sp.interpolate.LinearNDInterpolator(yedges,Ey)
InterpolatedEz = sp.interpolate.LinearNDInterpolator(zedges,Ez)
'''

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

#Define Search area in which cells need to be refined.
#-----------------------------------------------------------------------------
#Code to be optimized
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
'''
#plot cells to refine
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cellstorefine[:,0], cellstorefine[:,1], cellstorefine[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
'''
#-----------------------------------------------------------------------------
