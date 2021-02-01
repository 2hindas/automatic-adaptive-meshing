#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from discretize.utils import mkvc
import numpy as np
import src.Meshing as M
import src.Utils as utils
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from src.ErrorEstimator import iterator

# Domain
domain = ((-500, 4500), (-2000, 2000), (-2500, 200))
cell_width = 50

# Define the top geophysical surface
xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-2000, 2000, 101))
zz = np.zeros(np.shape(xx))
surface = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Define the model
sphere_origin = ((1000), (300), (-1200))
radius = 600
sphere_surface = M.create_sphere_surface(sphere_origin, radius, 101)
mesh = M.create_octree_mesh(domain, cell_width, sphere_surface)

# Defining transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
N = 21
xrx, yrx, zrx = np.meshgrid(np.linspace(2000, 4000, N), [0], [0])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

# Define survey
frequencies = [1.0]  # Frequency (Hz)
omegas = [2.0 * np.pi]  # Radial frequency (Hz)
survey = utils.define_survey(frequencies, receiver_locations, source_locations, ntx)

# Refine at certain locations
M.refine_at_locations(mesh, source_locations)
M.refine_at_locations(mesh, receiver_locations)
mesh.finalize()

# Resistivity in Ohm m
res_background = 1.0
res_block = 100.0

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, surface)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())
ind_sphere = utils.get_ind_sphere(mesh, ind_active, sphere_origin, radius)
model[ind_sphere] = res_block

# Run the adaptive meshing algorithm
mesh, ex, ey, ez, diff_list = iterator(mesh, domain, surface, cell_width, sphere_surface,
                                       sphere_origin
                                       , receiver_locations, source_locations, survey
                                       , res_background, res_block, ind_sphere, lim_iterations=20,
                                       radius=radius, interpolation='rbf', type_object='sphere')

# Print summary of the final mesh
print(mesh)

# Plot a few z-slices of the mesh
for i in range(0, 64):
    M.plot_mesh_slice(mesh, 'z', i, save=False)