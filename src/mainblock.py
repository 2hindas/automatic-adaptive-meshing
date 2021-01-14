#!/usr/bin/env python3


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from discretize.utils import mkvc
import numpy as np
import src.Meshing as M
import src.Utils as utils
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from src.LapentaEstimator import iterator
import matplotlib.pyplot as plt

domain = ((-500, 4500), (-2000, 2000), (-2500, 200))
cell_width = 50

xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-2000, 2000, 101))
zz = np.zeros(np.shape(xx))
surface = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

box_coordinates = ((200, 600), (-500, 500), (-1000, -800))
box_surface = M.create_box_surface(box_coordinates, cell_width, 'z', 15)
mesh = M.create_octree_mesh(domain, cell_width, box_surface)

# Defining transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
print("Number of transmitters", len(source_locations))
# wider area around transmitter
# xa, ya, za = np.meshgrid(np.linspace(-200,200,101), np.linspace(-200,200,101), np.linspace(-200,200,101))
# wider_area = np.c_[mkvc(xa), mkvc(ya), mkvc(za)]

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
# M.refine_at_locations(mesh,wider_area)
M.refine_at_locations(mesh, receiver_locations)

mesh.finalize()

# for i in range(0, 60):
#     M.plot_mesh_slice(mesh, 'z', i, save=True)
# exit()

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

mesh, ex, ey, ez, diff_list = iterator(mesh, domain, surface, cell_width, box_surface,
                                       box_coordinates
                                       , receiver_locations, source_locations, survey
                                       , res_background, res_block, model_map
                                       , model, ind_block, lim_iterations=15, interpolation='rbf')

print(mesh)
plt.figure(1)
plt.plot(diff_list[:, 0], diff_list[:, 1])
plt.xlabel('Number of iterations')
plt.ylabel('Average relative difference curr. and prev. iteration')
plt.title('Convergence of a sphere in an adaptive grid')
plt.show()

print(diff_list[:, 0])
print(diff_list[:, 1])

np.savetxt('rotated_block_relative_diff.csv', diff_list, delimiter=',')

for i in range(0, 4000):
    M.plot_mesh_slice(mesh, 'z', i, save=True)

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
