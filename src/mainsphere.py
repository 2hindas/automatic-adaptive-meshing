#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:40:02 2021

@author: larslaheij
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from discretize.utils import mkvc
import numpy as np
import Meshing as M
import Utils as utils
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from LapentaEstimator import iterator


domain = ((-500, 4500), (-1000, 1000), (-1200, 200))
cell_width = 50

xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-1000, 1000, 101))
zz = np.zeros(np.shape(xx))
surface = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

sphere_origin = ((1000), (500), (-500))
radius = 350
sphere_surface = M.create_sphere_surface(sphere_origin,radius,101)
mesh = M.create_octree_mesh(domain, cell_width, sphere_surface)

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

# Refine at certain locations
M.refine_at_locations(mesh, source_locations)
#M.refine_at_locations(mesh,wider_area)
M.refine_at_locations(mesh, receiver_locations)

mesh.finalize()

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

ind_sphere = utils.get_ind_sphere(mesh, ind_active, sphere_origin,radius)
model[ind_sphere] = res_block

mesh, ex, ey, ez = iterator(mesh, domain, surface, cell_width, sphere_surface, sphere_origin
                , receiver_locations, source_locations, survey
                , res_background, res_block, model_map
                , model, ind_sphere, lim_iterations=5,radius=radius,type_object='sphere')

print(mesh)



'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cells_to_refine[:, 0], cells_to_refine[:, 1], cells_to_refine[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# refine cells
'''