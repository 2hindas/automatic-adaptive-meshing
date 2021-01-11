#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:25:20 2021

@author: larslaheij
"""
import emg3d
import numpy as np
from main import mesh, Extest, Eytest, Eztest

src = [0, 0, 0, 0, 0]        # x-dir. source at the origin, 50 m above seafloor
off = np.arange(20, 41)*100  # Offsets
rec = [off, off*0, 0]        # In-line receivers on the seafloor
freq = 1.0                   # Frequency (Hz)

res_background = 1 #Resistivity of the background
res_block = 100 #Resistivity of the block

grid = emg3d.construct_mesh(
        frequency=freq,
        min_width_limits=100,
        properties=res_background,
        center=(0, 0, 0),
        domain=([-500, 4500], [-1000, 1000], [-1200, 200]),
        verb=0,
)

# Background model
model_background = emg3d.Model(grid, property_x=res_background*np.ones(grid.vnC))

# Block model
res = res_background*np.ones(grid.vnC).ravel('F')
xx = (grid.gridCC[:, 0] >= 200) & (grid.gridCC[:, 0] <= 600)
yy = (grid.gridCC[:, 1] >= -500) & (grid.gridCC[:, 1] <= 500)
zz = (grid.gridCC[:, 2] > -1000)*(grid.gridCC[:, 2] < -800)

res[xx*yy*zz] = res_block
model_block = emg3d.Model(grid, property_x=res)

#Define solver
solver_opts = {
        'verb': 2, 'sslsolver': True,
        'semicoarsening': True, 'linerelaxation': True
}
#Solve for source field
sfield = emg3d.get_source_field(grid, src, freq, strength=0)
#Solve E-field
efield_block = emg3d.solve(grid, model_block, sfield, **solver_opts) #The solution
xedges = grid.edges_x
yedges = grid.edges_y
zedges = grid.edges_z

Ex_inter = efield_block[0:len(xedges)] #x-component of the electric field on the x-edges
Ey_inter = efield_block[len(xedges):len(xedges) + len(yedges)] #y-component of the electric field on the y-edges
Ez_inter = efield_block[
           len(xedges) + len(yedges):len(xedges) + len(yedges) + len(zedges)] #z-component of the electric field on the z-edges

xsearch = np.where((xedges[:,0]>-500) & (xedges[:,0]<4500) & (xedges[:,1]>-1000) & (xedges[:,1]<1000) & (xedges[:,2]>-1200) & (xedges[:,2]<200))
ysearch = np.where((yedges[:,0]>-500) & (yedges[:,0]<4500) & (yedges[:,1]>-1000) & (yedges[:,1]<1000) & (yedges[:,2]>-1200) & (yedges[:,2]<200))
zsearch = np.where((zedges[:,0]>-500) & (zedges[:,0]<4500) & (zedges[:,1]>-1000) & (zedges[:,1]<1000) & (zedges[:,2]>-1200) & (zedges[:,2]<200))

Ex = np.take(Ex_inter,xsearch[0])
Ey = np.take(Ey_inter,ysearch[0])
Ez = np.take(Ez_inter,zsearch[0])

xedges = xedges[(xedges[:,0]>-500) & (xedges[:,0]<4500) & (xedges[:,1]>-1000) & (xedges[:,1]<1000) & (xedges[:,2]>-1200) & (xedges[:,2]<200)]
yedges = yedges[(yedges[:,0]>-500) & (yedges[:,0]<4500) & (yedges[:,1]>-1000) & (yedges[:,1]<1000) & (yedges[:,2]>-1200) & (yedges[:,2]<200)]
zedges = zedges[(zedges[:,0]>-500) & (zedges[:,0]<4500) & (zedges[:,1]>-1000) & (zedges[:,1]<1000) & (zedges[:,2]>-1200) & (zedges[:,2]<200)]


#Compute relative difference of octree mesh and tensor mesh (x-component on x-edges)
relative_difference_Efieldx = []
for i in range(len(xedges)):
    # This equation is sensitive to catastrophic failure
    relative_diff = np.abs((Ex[i]-Extest(*xedges[i]))/Ex[i])
    relative_difference_Efieldx.append(relative_diff)

#Compute relative difference of octree mesh and tensor mesh (x-component on x-edges)
relative_difference_Efieldy = []
for i in range(len(yedges)):
    # This equation is sensitive to catastrophic failure
    relative_diff = np.abs((Ey[i]-Eytest(*yedges[i]))/Ey[i])
    relative_difference_Efieldy.append(relative_diff)
    
#Compute relative difference of octree mesh and tensor mesh (x-component on x-edges)
relative_difference_Efieldz = []
for i in range(len(zedges)):
    # This equation is sensitive to catastrophic failure
    relative_diff = np.abs((Ez[i]-Eztest(*zedges[i]))/Ez[i])
    relative_difference_Efieldz.append(relative_diff)
    

    



