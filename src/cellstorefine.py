#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:40:49 2020

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
from multiprocessing import Pool
import errorestimatorexample as f

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

    
#Define Search area
searcharea = f.xyzcells[(f.xyzcells[:,0]>(0-f.dx)) & (f.xyzcells[:,0]<(4000+f.dx)) 
                      & (f.xyzcells[:,1]>(-500-f.dy)) & (f.xyzcells[:,1]<(500+f.dy)) 
                      & (f.xyzcells[:,2]>(-1000-f.dz)) & (f.xyzcells[:,2]<(-800+f.dz)) ]

if __name__ == '__main__':
    with Pool(32) as p:
            errorcellcenters = p.map(f.Errorcomputer,searcharea.tolist())

        
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
