#!/usr/bin/env python3

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import emg3d
import requests
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline
from discretize.utils import mkvc
import numpy as np
import src.Meshing as M
import src.Utils as utils
from SimPEG import maps
from src.ErrorEstimator import iteratornonobject
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

plt.style.use('ggplot')
plt.ion()

# Load model
fname = 'GemPy-II.h5'
if not os.path.isfile(fname):
    url = ("https://github.com/emsig/emg3d-gallery/blob/master/examples/"
           f"data/models/{fname}?raw=true")
    with open(fname, 'wb') as f:
        t = requests.get(url)
        f.write(t.content)

data = emg3d.load(fname)
model, mesh = data['model'], data['mesh']

# Check model
print(model)

# Check mesh
print(mesh)

# Convert the tensor mesh into an adaptive octree mesh
domain = ((9000, 19000), (2500, 12500), (-5500, -500))
cell_width = 500

# Acquisition source frequencies (Hz)
frequencies = [1.0]  # only use one frequency #frequencies = [0.5, 1.0]
omegas = [2.0 * np.pi]  # Radial frequency (Hz)

# Define seafloor as every first cell with resistivity lower than 0.33
seafloor = np.ones((mesh.shape_cells[0], mesh.shape_cells[1]))
for i in range(mesh.shape_cells[0]):
    for ii in range(mesh.shape_cells[1]):
        seafloor[i, ii] = mesh.nodes_z[:-1][
            model.property_x[i, ii, :] < 0.33][0]

# Create a 2D interpolation function from the seafloor
bathymetry = RectBivariateSpline(
    mesh.cell_centers_x, mesh.cell_centers_y, seafloor)

# Seafloor x-, y- and z-coordinates, required for making the octree mesh
seafloor = np.reshape(seafloor, (len(mesh.cell_centers_x) * len(mesh.cell_centers_y), 1))
xseafloor, yseafloor = np.meshgrid(mesh.cell_centers_x, mesh.cell_centers_y)
seafloorxyz = np.c_[mkvc(xseafloor), mkvc(yseafloor), mkvc(seafloor)]
seafloorxyz = seafloorxyz[(seafloorxyz[:, 0] > 9000) & ((seafloorxyz[:, 0] < 19000))
                          & (seafloorxyz[:, 1] > 2500) & (seafloorxyz[:, 1] < 12500)]

# Defining transmitter location
src_x = 2 * 5000
src_y = 7500

# Source depths: 50 m above seafloor
src_z = bathymetry(src_x, src_y).ravel() + 50
xtx, ytx, ztx = np.meshgrid([src_x], [src_y], [src_z[0]])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
rec_x = np.arange(11, 18) * 1e3
rec_y = np.arange(3) * 1e3 + 6500
RZ = bathymetry(rec_x, rec_y)
RX, RY = np.meshgrid(rec_x, rec_y, indexing='ij')
xrx, yrx, zrx = RX, RY, RZ
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

# Create survey
survey = utils.define_survey(frequencies, receiver_locations, source_locations, ntx)

# Generate octree mesh
octreemesh = M.create_octree_mesh(domain, cell_width, seafloorxyz)
M.refine_at_locations(octreemesh, source_locations)
M.refine_at_locations(octreemesh, receiver_locations)
octreemesh.finalize()

# Interpolate resistivity values to map them from the tensor mesh onto the octree mesh
resshape = np.reshape(model.property_x.transpose(), mesh.n_cells)
resfunction = NearestNDInterpolator(mesh.cell_centers, resshape)

# Plot resistivity model
mesh.plot_3d_slicer(model.property_x, xslice=12000, yslice=7000,
                    pcolor_opts={'norm': LogNorm(vmin=0.3, vmax=200)})

# Plot survey in figure above
fig = plt.gcf()
fig.suptitle('Resistivity model (Ohm.m) and survey layout')
axs = fig.get_children()
axs[1].plot(receiver_locations[:, 0], receiver_locations[:, 1], 'bv')
axs[2].plot(receiver_locations[:, 0], receiver_locations[:, 2], 'bv')
axs[3].plot(receiver_locations[:, 2], receiver_locations[:, 1], 'bv')
axs[1].plot(source_locations[:, 0], source_locations[:, 1], 'r*')
axs[2].plot(source_locations[:, 0], source_locations[:, 2], 'r*')
axs[3].plot(source_locations[:, 2], source_locations[:, 1], 'r*')
plt.ioff()
plt.show()

# Find cells that are active in the forward modeling (cells below user-defined surface)
ind_active = np.array([True] * octreemesh.n_cells)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(octreemesh, ind_active, 1e8)

# Define model. Models in SimPEG are vector arrays
model = resfunction(octreemesh.cell_centers)

# In this function the octree mesh will be re-iterated.
octreemesh, Extest, Eytest, Eztest, diff_list = iteratornonobject(octreemesh, domain
                                                                  , cell_width, seafloorxyz
                                                                  , receiver_locations,
                                                                  source_locations, survey
                                                                  , resfunction, model_map
                                                                  , model, lim_iterations=20)

# Print converged octree mesh
print(octreemesh)

# Store relevant data: mesh, interpolated functions of the electric field and convergence list.
np.save('convergence.npy', diff_list)
np.save('interpolatorEx.npy', Extest)
np.save('interpolatorEy.npy', Eytest)
np.save('interpolatorEz.npy', Eztest)
octreemesh.save('octree.json')
